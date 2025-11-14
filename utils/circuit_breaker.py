"""
Circuit Breaker Module
Prevents cascade failures from bad data/models
"""

import logging
import time
from datetime import datetime
from typing import Callable, Any

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Prevent cascade failures from bad data/models"""

    def __init__(self, failure_threshold: int = 3, timeout: int = 300):
        """
        Initialize circuit breaker

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before trying again
        """
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.is_open = False
        self.logger = logging.getLogger(__name__)

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result or raises exception
        """
        if self.is_open:
            if self._should_attempt_reset():
                self.is_open = False
                self.failure_count = 0
                self.logger.info("Circuit breaker reset - attempting operation")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")

        try:
            result = func(*args, **kwargs)
            self.failure_count = 0  # Reset on success
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.failure_count >= self.failure_threshold:
                self.is_open = True
                self.logger.error(f"Circuit breaker OPENED due to repeated failures ({self.failure_count}/{self.failure_threshold})")

            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if not self.last_failure_time:
            return True

        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed > self.timeout

    def get_status(self) -> dict:
        """Get circuit breaker status"""
        return {
            'is_open': self.is_open,
            'failure_count': self.failure_count,
            'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'next_attempt': (self.last_failure_time.isoformat() if self.last_failure_time and self.is_open else None)
        }