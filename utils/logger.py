"""
Logger Module
Configures logging for the FX-Ai application
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Any
from queue import Queue
import atexit

class MT5TimeFormatter(logging.Formatter):
    """
    Custom logging formatter that uses MT5 server time instead of local time
    UPDATED: Now uses ClockSynchronizer for reliable server time without excessive MT5 API calls
    """

    def __init__(self, fmt=None, datefmt=None, mt5_connector=None, clock_sync=None):
        super().__init__(fmt, datefmt)
        self.mt5_connector = mt5_connector
        self.clock_sync = clock_sync  # ClockSynchronizer instance
        # Cache MT5 time to avoid deadlocks from excessive MT5 API calls
        self._last_mt5_time = None
        self._last_local_time = 0
        self._cache_duration = 1.0  # Cache for 1 second to reduce MT5 calls

    def formatTime(self, record, datefmt=None):
        """
        Use MT5 server time via ClockSynchronizer for consistent timestamps.
        Falls back to UTC time for consistency when MT5 is unavailable.
        
        UPDATED: Now uses ClockSynchronizer which already handles caching and time synchronization.
        """
        current_time = record.created
        
        # Try ClockSynchronizer first (best option - already cached and synchronized)
        if self.clock_sync:
            try:
                # Get synchronized MT5 server time (already cached by ClockSynchronizer)
                server_time = self.clock_sync.get_synced_time()
                if server_time:
                    time_to_format = server_time
                else:
                    time_to_format = datetime.utcfromtimestamp(current_time)
            except Exception:
                time_to_format = datetime.utcfromtimestamp(current_time)
        
        # Fallback to direct MT5 connector with caching
        elif self.mt5_connector:
            # Calculate time offset from last cache
            time_offset = current_time - self._last_local_time if self._last_mt5_time is not None else float('inf')
            
            # If cache is still valid (< 1 second old), use cached time + offset
            if self._last_mt5_time is not None and time_offset < self._cache_duration:
                adjusted_time = self._last_mt5_time + time_offset
                time_to_format = datetime.fromtimestamp(adjusted_time)
            else:
                # Cache expired or not initialized, get fresh MT5 server time
                try:
                    server_time = self.mt5_connector.get_server_time()
                    if server_time:
                        self._last_mt5_time = server_time.timestamp()
                        self._last_local_time = current_time
                        time_to_format = server_time
                    else:
                        # Fallback to UTC time if server time unavailable
                        time_to_format = datetime.utcfromtimestamp(current_time)
                except Exception:
                    # Fallback to UTC time on error
                    time_to_format = datetime.utcfromtimestamp(current_time)
        else:
            # No MT5 connector or ClockSynchronizer - use local time
            time_to_format = datetime.fromtimestamp(current_time)
        
        # Format the time
        if datefmt:
            return time_to_format.strftime(datefmt)
        else:
            return time_to_format.strftime('%Y-%m-%d %H:%M:%S')

class MT5TimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """
    Custom TimedRotatingFileHandler that uses MT5 server time for rotation timing and file naming
    """

    def __init__(self, filename, when='midnight', interval=1, backupCount=0,
                 encoding=None, delay=False, utc=False, atTime=None,
                 mt5_connector=None, clock_sync=None):
        """
        Initialize MT5TimedRotatingFileHandler

        Args:
            filename: Log file base name
            when: Time unit for rotation ('S', 'M', 'H', 'D', 'midnight')
            interval: Number of time units between rotations
            backupCount: Number of backup files to keep
            encoding: File encoding
            delay: Delay file opening until first record
            utc: Use UTC time (ignored, we use MT5 time)
            atTime: Specific time for rotation (ignored, we use MT5 time)
            mt5_connector: MT5 connector for server time
            clock_sync: ClockSynchronizer for accurate server time
        """
        # Store MT5 connector and clock sync references
        self.mt5_connector = mt5_connector
        self.clock_sync = clock_sync

        # Initialize with basic settings (we'll override timing methods)
        super().__init__(filename, when=when, interval=interval, backupCount=backupCount,
                        encoding=encoding, delay=delay, utc=False, atTime=atTime)

        # Set custom suffix for MT5 server time based naming
        self.suffix = "_%Y_%m_%d.log"

    def _getMT5Time(self) -> datetime:
        """
        Get current MT5 server time, with fallbacks

        Returns:
            datetime: Current MT5 server time or fallback
        """
        try:
            # Try ClockSynchronizer first (preferred method)
            if self.clock_sync:
                server_time = self.clock_sync.get_server_time()
                if server_time:
                    return server_time

            # Fallback to direct MT5 connector
            elif self.mt5_connector:
                server_time = self.mt5_connector.get_server_time()
                if server_time:
                    return server_time

        except Exception:
            pass

        # Final fallback to local time
        return datetime.now()

    def shouldRollover(self, record):
        """
        Determine if log file should be rolled over based on MT5 server time

        Args:
            record: LogRecord to check

        Returns:
            bool: True if rollover should occur
        """
        # Get current MT5 server time
        current_time = self._getMT5Time()

        # Check if we've crossed the rotation boundary
        if self.when == 'midnight':
            # Rotate at MT5 server midnight
            rollover_datetime = datetime.fromtimestamp(self.rolloverAt)
            should_rotate = (current_time.date() != rollover_datetime.date())
        else:
            # For other intervals, use standard logic but with MT5 time
            should_rotate = super().shouldRollover(record)

        return should_rotate

    def doRollover(self):
        """
        Perform log file rollover using MT5 server time for naming
        """
        try:
            # Get current MT5 server time for filename
            current_time = self._getMT5Time()

            # Generate filename with MT5 server date
            if self.when == 'midnight':
                # Use yesterday's date for the rotated file (since rotation happens at midnight)
                rollover_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                if current_time.hour < 12:  # If it's early morning, use previous day
                    from datetime import timedelta
                    rollover_time = rollover_time - timedelta(days=1)
            else:
                rollover_time = current_time

            # Format filename suffix with MT5 server date
            suffix = rollover_time.strftime(self.suffix)

            # Create the rotated filename
            if self.backupCount > 0:
                # Find the next available backup number
                for i in range(self.backupCount - 1, 0, -1):
                    sfn = self.rotation_filename(f"{self.baseFilename}.{i}{suffix}")
                    dfn = self.rotation_filename(f"{self.baseFilename}.{i+1}{suffix}")
                    if os.path.exists(sfn):
                        try:
                            if os.path.exists(dfn):
                                os.remove(dfn)
                            os.rename(sfn, dfn)
                        except (OSError, PermissionError) as e:
                            # File is locked, skip rotation for this backup
                            print(f"Warning: Could not rotate log file {sfn} -> {dfn}: {e}")

                # Rotate current file
                dfn = self.rotation_filename(f"{self.baseFilename}.1{suffix}")
                try:
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    self.rotate(self.baseFilename, dfn)
                except (OSError, PermissionError) as e:
                    # File is locked, skip rotation
                    print(f"Warning: Could not rotate current log file to {dfn}: {e}")

            else:
                # No backups, just rotate to dated file
                dfn = self.rotation_filename(f"{self.baseFilename}{suffix}")
                try:
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    self.rotate(self.baseFilename, dfn)
                except (OSError, PermissionError) as e:
                    # File is locked, skip rotation
                    print(f"Warning: Could not rotate log file to {dfn}: {e}")

            # Update rollover time for next rotation
            self._updateRolloverTime()
            
        except Exception as e:
            # If rollover fails completely, log to console and continue
            print(f"Warning: Log file rollover failed: {e}")
            # Try to update rollover time anyway to prevent repeated failures
            try:
                self._updateRolloverTime()
            except Exception:
                pass

    def _updateRolloverTime(self):
        """
        Update the rollover time based on MT5 server time
        """
        current_time = self._getMT5Time()

        if self.when == 'midnight':
            # Next rollover at MT5 server midnight
            next_midnight = current_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            self.rolloverAt = int(next_midnight.timestamp())
        else:
            # For other intervals, calculate next rollover time
            # This is a simplified implementation - could be enhanced for other intervals
            next_rollover = current_time + timedelta(seconds=self.interval * self._getIntervalSeconds())
            self.rolloverAt = int(next_rollover.timestamp())

    def _getIntervalSeconds(self):
        """
        Get interval in seconds based on 'when' parameter

        Returns:
            int: Interval in seconds
        """
        intervals = {
            'S': 1,      # Seconds
            'M': 60,     # Minutes
            'H': 3600,   # Hours
            'D': 86400,  # Days
            'midnight': 86400  # Days (handled specially)
        }
        return intervals.get(self.when, 86400)

def setup_logger(name: str = 'FX-Ai', level: str = 'INFO',
                log_file: Optional[str] = None, max_bytes: int = 10485760,
                backup_count: int = 5, rotation_type: str = 'size',
                mt5_connector: Optional[Any] = None, clock_sync: Optional[Any] = None) -> logging.Logger:
    """
    Set up logger with file and console handlers

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file path (optional)
        max_bytes: Maximum log file size in bytes (for size-based rotation)
        backup_count: Number of backup log files to keep
        rotation_type: 'size' for RotatingFileHandler or 'time' for TimedRotatingFileHandler
        mt5_connector: MT5 connector instance for server time logging
        clock_sync: ClockSynchronizer instance for accurate server time (preferred)

    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters with ClockSynchronizer support
    file_formatter = MT5TimeFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        mt5_connector=mt5_connector,
        clock_sync=clock_sync
    )
    console_formatter = MT5TimeFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        mt5_connector=mt5_connector,
        clock_sync=clock_sync
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(console_formatter)
    
    # Queue-based file logging to prevent deadlocks
    # Main thread writes to queue (non-blocking), separate thread writes to file
    if log_file:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            # Create a queue for log records
            log_queue = Queue(-1)  # Unlimited size
            
            # Create appropriate file handler based on rotation type
            if rotation_type == 'time':
                # MT5 server time-based rotating file handler
                file_handler = MT5TimedRotatingFileHandler(
                    log_file,
                    when='midnight',
                    interval=1,
                    backupCount=backup_count,
                    mt5_connector=mt5_connector,
                    clock_sync=clock_sync
                )
            else:
                # Basic file handler (size-based rotation handled separately if needed)
                file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            
            # Create queue listener (separate thread handles file writes)
            queue_listener = logging.handlers.QueueListener(
                log_queue, 
                file_handler,
                respect_handler_level=True
            )
            queue_listener.start()
            
            # Add queue handler to logger (non-blocking writes to queue)
            queue_handler = logging.handlers.QueueHandler(log_queue)
            logger.addHandler(queue_handler)
            
            # Add console handler after queue setup
            logger.addHandler(console_handler)
            
            # Ensure queue listener stops on exit
            atexit.register(queue_listener.stop)
            
            # Store listener reference for cleanup
            logger._queue_listener = queue_listener  # type: ignore
            
        except Exception as e:
            # Fallback to console-only if queue setup fails
            logger.addHandler(console_handler)
            logger.error(f"Failed to set up queue-based file logging: {e}")
    else:
        # No file logging requested, console only
        logger.addHandler(console_handler)

    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get logger by name

    Args:
        name: Logger name

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)

def set_log_level(logger: logging.Logger, level: str):
    """
    Set logging level for logger and its handlers

    Args:
        logger: Logger instance
        level: New logging level
    """
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    for handler in logger.handlers:
        handler.setLevel(getattr(logging, level.upper(), logging.INFO))

def add_file_handler(logger: logging.Logger, log_file: str,
                    max_bytes: int = 10485760, backup_count: int = 5, rotation_type: str = 'size',
                    mt5_connector: Optional[Any] = None, clock_sync: Optional[Any] = None):
    """
    Add file handler to existing logger

    Args:
        logger: Logger instance
        log_file: Log file path
        max_bytes: Maximum log file size (for size-based rotation)
        backup_count: Number of backup files
        rotation_type: 'size' for RotatingFileHandler or 'time' for TimedRotatingFileHandler
        mt5_connector: MT5 connector instance for server time logging
        clock_sync: ClockSynchronizer instance for accurate server time (preferred)
    """
    try:
        # Check if file handler already exists
        for handler in logger.handlers:
            if isinstance(handler, (logging.handlers.RotatingFileHandler, logging.handlers.TimedRotatingFileHandler, MT5TimedRotatingFileHandler)):
                if handler.baseFilename == os.path.abspath(log_file):
                    return  # Already exists

        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Create file formatter with MT5 time support
        file_formatter = MT5TimeFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            mt5_connector=mt5_connector,
            clock_sync=clock_sync
        )

        if rotation_type == 'time':
            # MT5 server time-based rotating file handler (daily rotation with MT5 date in filename)
            file_handler = MT5TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=backup_count,
                mt5_connector=mt5_connector,
                clock_sync=clock_sync
            )
        else:
            # Rotating file handler (size-based rotation)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
        
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)

    except Exception as e:
        logger.error(f"Failed to add file handler: {e}")

def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls

    Args:
        logger: Logger instance

    Returns:
        decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} returned: {result}")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator

def update_all_loggers_with_mt5(mt5_connector: Any):
    """
    Update all existing loggers to use MT5 server time
    
    Args:
        mt5_connector: MT5 connector instance for server time
    """
    try:
        # Update all existing loggers' formatters
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler.formatter, MT5TimeFormatter):
                handler.formatter.mt5_connector = mt5_connector
        
        # Update all named loggers
        for logger_name in logging.Logger.manager.loggerDict:
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers:
                if isinstance(handler.formatter, MT5TimeFormatter):
                    handler.formatter.mt5_connector = mt5_connector
                    
    except Exception as e:
        print(f"Error updating loggers with MT5: {e}")

def shutdown_logger(logger: logging.Logger):
    """
    Properly shutdown logger and stop queue listener if present
    
    Args:
        logger: Logger instance to shutdown
    """
    try:
        # Stop queue listener if it exists
        if hasattr(logger, '_queue_listener'):
            logger._queue_listener.stop()  # type: ignore
            delattr(logger, '_queue_listener')
        
        # Close and remove all handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
            
    except Exception as e:
        print(f"Error shutting down logger: {e}")