"""
Logger Module
Configures logging for the FX-Ai application
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional, Any

class MT5TimeFormatter(logging.Formatter):
    """
    Custom logging formatter that uses MT5 server time instead of local time
    """

    def __init__(self, fmt=None, datefmt=None, mt5_connector=None):
        super().__init__(fmt, datefmt)
        self.mt5_connector = mt5_connector

    def formatTime(self, record, datefmt=None):
        """
        Override formatTime to use MT5 server time
        """
        if self.mt5_connector:
            try:
                # Get MT5 server time
                mt5_time = self.mt5_connector.get_server_time()
                if mt5_time:
                    # Format according to datefmt or default
                    if datefmt:
                        return mt5_time.strftime(datefmt)
                    else:
                        return mt5_time.strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                # If MT5 time fails, log the error and fall back to local time
                print(f"Warning: Failed to get MT5 time for logging: {e}")

        # Fallback to local time if MT5 connector not available or fails
        ct = datetime.fromtimestamp(record.created)
        if datefmt:
            return ct.strftime(datefmt)
        else:
            return ct.strftime('%Y-%m-%d %H:%M:%S')

def setup_logger(name: str = 'FX-Ai', level: str = 'INFO',
                log_file: Optional[str] = None, max_bytes: int = 10485760,
                backup_count: int = 5, rotation_type: str = 'size',
                mt5_connector: Optional[Any] = None) -> logging.Logger:
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

    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    file_formatter = MT5TimeFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        mt5_connector=mt5_connector
    )
    console_formatter = MT5TimeFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        mt5_connector=mt5_connector
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file specified)
    if log_file:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            if rotation_type == 'time':
                # Timed rotating file handler (daily rotation with date in filename)
                file_handler = logging.handlers.TimedRotatingFileHandler(
                    log_file,
                    when='midnight',
                    interval=1,
                    backupCount=backup_count
                )
                # Set custom suffix to include underscore and .log extension
                file_handler.suffix = "_%Y_%m_%d.log"
            else:
                # Rotating file handler (size-based rotation)
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count
                )
            
            file_handler.setLevel(logging.DEBUG)  # Log everything to file
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        except Exception as e:
            logger.error(f"Failed to set up file logging: {e}")

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
                    max_bytes: int = 10485760, backup_count: int = 5, rotation_type: str = 'size'):
    """
    Add file handler to existing logger

    Args:
        logger: Logger instance
        log_file: Log file path
        max_bytes: Maximum log file size (for size-based rotation)
        backup_count: Number of backup files
        rotation_type: 'size' for RotatingFileHandler or 'time' for TimedRotatingFileHandler
    """
    try:
        # Check if file handler already exists
        for handler in logger.handlers:
            if isinstance(handler, (logging.handlers.RotatingFileHandler, logging.handlers.TimedRotatingFileHandler)):
                if handler.baseFilename == os.path.abspath(log_file):
                    return  # Already exists

        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Create file handler
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )

        if rotation_type == 'time':
            # Timed rotating file handler (daily rotation with date in filename)
            file_handler = logging.handlers.TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=backup_count
            )
            # Set custom suffix to include underscore and .log extension
            file_handler.suffix = "_%Y_%m_%d.log"
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