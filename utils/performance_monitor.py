"""
Performance Monitoring Module
Track function execution times and performance metrics
"""

import logging
import time
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)


def monitor_performance(func: Callable, warning_threshold: float = 5.0) -> Callable:
    """
    Track function execution time

    Args:
        func: Function to monitor
        warning_threshold: Time in seconds above which to log a warning (default: 5.0)

    Returns:
        Wrapped function with performance tracking
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start

            if duration > warning_threshold:  # Warn if any operation takes > threshold seconds
                logger.warning(f"{func.__name__} took {duration:.2f}s - consider optimization")

            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
            raise

    return wrapper


def monitor_performance_async(func: Callable, warning_threshold: float = 5.0) -> Callable:
    """
    Track async function execution time

    Args:
        func: Async function to monitor
        warning_threshold: Time in seconds above which to log a warning (default: 5.0)

    Returns:
        Wrapped async function with performance tracking
    """
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start

            if duration > warning_threshold:  # Warn if any operation takes > threshold seconds
                logger.warning(f"{func.__name__} took {duration:.2f}s - consider optimization")

            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
            raise

    return wrapper


class PerformanceTracker:
    """Track performance metrics across the application"""

    def __init__(self):
        self.metrics = {}
        self.logger = logging.getLogger(__name__)

    def track_execution(self, operation: str, duration: float):
        """Track execution time for an operation"""
        if operation not in self.metrics:
            self.metrics[operation] = []

        self.metrics[operation].append(duration)

        # Keep only last 100 measurements
        if len(self.metrics[operation]) > 100:
            self.metrics[operation] = self.metrics[operation][-100:]

        # Warn if average is too high
        avg_duration = sum(self.metrics[operation]) / len(self.metrics[operation])
        if avg_duration > 2.0:  # Average > 2 seconds
            self.logger.warning(f"{operation} average execution time: {avg_duration:.2f}s")

    def get_metrics_report(self) -> dict:
        """Get performance metrics report"""
        report = {}
        for operation, times in self.metrics.items():
            if times:
                report[operation] = {
                    'count': len(times),
                    'avg': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'last': times[-1]
                }
        return report