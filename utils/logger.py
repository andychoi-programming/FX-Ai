"""
Logger Module
Configures logging for the FX-Ai application
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional

def setup_logger(name: str = 'FX-Ai', level: str = 'INFO',
                log_file: Optional[str] = None, max_bytes: int = 10485760,
                backup_count: int = 5, rotation_type: str = 'size') -> logging.Logger:
    """
    Set up logger with file and console handlers

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file path (optional)
        max_bytes: Maximum log file size in bytes (for size-based rotation)
        backup_count: Number of backup log files to keep
        rotation_type: 'size' for RotatingFileHandler or 'time' for TimedRotatingFileHandler

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
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
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
                file_handler.suffix = "_%Y-%m-%d.log"
                file_handler.extMatch = r"^\d{4}-\d{2}-\d{2}\.log$"
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
            file_handler.suffix = "_%Y-%m-%d.log"
            file_handler.extMatch = r"^\d{4}-\d{2}-\d{2}\.log$"
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

class TradingLogger:
    """Specialized logger for trading activities"""

    def __init__(self, name: str = 'Trading', log_file: Optional[str] = None):
        """
        Initialize trading logger

        Args:
            name: Logger name
            log_file: Log file path
        """
        self.logger = setup_logger(name, 'INFO', log_file)

    def log_signal(self, symbol: str, signal: dict):
        """Log trading signal"""
        self.logger.info(f"SIGNAL - {symbol}: {signal}")

    def log_trade(self, symbol: str, action: str, volume: float, price: float):
        """Log trade execution"""
        self.logger.info(f"TRADE - {symbol}: {action.upper()} {volume} lots @ {price}")

    def log_pnl(self, symbol: str, pnl: float, balance: float):
        """Log profit/loss"""
        pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
        self.logger.info(f"PNL - {symbol}: {pnl_str}, Balance: {balance:.2f}")

    def log_error(self, symbol: str, error: str):
        """Log trading error"""
        self.logger.error(f"ERROR - {symbol}: {error}")

    def log_performance(self, metrics: dict):
        """Log performance metrics"""
        self.logger.info(f"PERFORMANCE: {metrics}")

# Global trading logger instance
trading_logger = TradingLogger('FX-Trading', 'logs/trading.log')