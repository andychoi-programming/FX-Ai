"""
FX-Ai Time Management System
Centralized time handling for consistent trading hours and market sessions
"""

import logging
from datetime import datetime, time, timedelta
from typing import Optional, Tuple
import pytz

logger = logging.getLogger(__name__)


class TimeManager:
    """
    Centralized time management for FX-Ai trading system.

    Handles all time-related logic including:
    - Trading hours (22:30 MT5 server time close)
    - Forex market sessions and weekends
    - Time zone conversions
    - Consistent time source usage
    """

    # Trading Hours Configuration
    MT5_CLOSE_TIME = time(22, 30)  # 22:30 MT5 server time (when we stop trading)
    MT5_CLOSE_BUFFER_END = time(23, 59)  # End of no-trade buffer (until midnight)

    # Forex Market Hours (EST - Eastern Standard Time)
    # Forex is 24/5 but we follow our own trading discipline
    FOREX_WEEKEND_START = "Friday 17:00 EST"  # When forex week "ends"
    FOREX_WEEKEND_END = "Sunday 17:00 EST"    # When forex week "starts"

    # Our Trading Discipline (within forex market hours)
    TRADING_DAYS = [0, 1, 2, 3, 4]  # Monday-Friday (0=Monday)
    TRADING_START_EST = time(0, 0)   # Midnight EST (our start)
    TRADING_END_EST = time(17, 0)    # 5PM EST Friday (our end)

    # Time Zones
    EST = pytz.timezone('US/Eastern')
    UTC = pytz.timezone('UTC')

    def __init__(self, mt5_connector=None):
        """
        Initialize TimeManager

        Args:
            mt5_connector: MT5 connector instance for server time
        """
        self.mt5 = mt5_connector
        self._last_closure_date = None

    def get_mt5_server_time(self) -> Optional[datetime]:
        """
        Get current MT5 server time (preferred method)

        Returns:
            datetime: MT5 server time, or None if unavailable
        """
        try:
            if self.mt5 and hasattr(self.mt5, 'get_server_time'):
                server_time = self.mt5.get_server_time()
                if server_time:
                    return server_time
        except Exception as e:
            logger.warning(f"Failed to get MT5 server time: {e}")

        logger.warning("MT5 server time unavailable, using local time")
        return None

    def get_current_time(self) -> datetime:
        """
        Get current time for trading decisions.
        Uses MT5 server time when available, falls back to local time.

        Returns:
            datetime: Current time for trading logic
        """
        mt5_time = self.get_mt5_server_time()
        if mt5_time:
            return mt5_time

        # Fallback to local time
        return datetime.now()

    def is_trading_allowed(self) -> Tuple[bool, str]:
        """
        Check if trading is currently allowed.

        Returns:
            Tuple[bool, str]: (is_allowed, reason)
        """
        current_time = self.get_current_time()
        current_time_only = current_time.time()
        current_date = current_time.date()

        # Check if it's a weekend (no forex trading)
        if self._is_weekend(current_date):
            return False, "Weekend - forex markets closed"

        # Check if we're in the no-trade buffer after daily close
        if self._is_in_no_trade_buffer(current_time_only):
            close_time_str = f"{self.MT5_CLOSE_TIME.hour:02d}:{self.MT5_CLOSE_TIME.minute:02d}"
            return False, f"After daily close time ({close_time_str} MT5 time)"

        # Check forex market hours (EST)
        if not self._is_forex_market_open():
            return False, "Forex market closed (outside EST trading hours)"

        return True, "Trading allowed"

    def should_close_positions(self) -> Tuple[bool, str]:
        """
        Check if all positions should be closed (daily closure).

        Returns:
            Tuple[bool, str]: (should_close, reason)
        """
        current_time = self.get_current_time()
        current_time_only = current_time.time()
        current_date = current_time.date()

        # Only close once per day
        if self._last_closure_date == current_date:
            return False, "Already closed positions today"

        # Check if it's time to close (past 22:30 MT5 time)
        if current_time_only >= self.MT5_CLOSE_TIME:
            self._last_closure_date = current_date
            close_time_str = f"{self.MT5_CLOSE_TIME.hour:02d}:{self.MT5_CLOSE_TIME.minute:02d}"
            return True, f"Past daily close time ({close_time_str} MT5 time)"

        return False, "Not yet time to close positions"

    def _is_weekend(self, date) -> bool:
        """
        Check if given date is a weekend (Saturday/Sunday).

        Args:
            date: Date to check

        Returns:
            bool: True if weekend
        """
        return date.weekday() >= 5  # 5=Saturday, 6=Sunday

    def _is_in_no_trade_buffer(self, current_time: time) -> bool:
        """
        Check if current time is in the no-trade buffer after daily close.

        Args:
            current_time: Current time

        Returns:
            bool: True if in no-trade buffer
        """
        return self.MT5_CLOSE_TIME <= current_time <= self.MT5_CLOSE_BUFFER_END

    def _is_forex_market_open(self) -> bool:
        """
        Check if we're within our defined trading discipline hours.
        Forex is 24/5, but we follow disciplined trading hours.

        Returns:
            bool: True if within our trading discipline
        """
        try:
            # Get current time in EST
            now_est = datetime.now(self.EST)
            now_time = now_est.time()
            now_weekday = now_est.weekday()  # 0=Monday, 6=Sunday

            # Only trade Monday-Friday
            if now_weekday not in self.TRADING_DAYS:
                return False

            # Within our trading hours (we start at midnight EST)
            # Note: Forex is 24/5, but we stop at 22:30 MT5 time daily
            return True

        except Exception as e:
            logger.warning(f"Error checking forex market hours: {e}")
            return True  # Default to allowing trading if check fails

    def is_forex_weekend(self) -> bool:
        """
        Check if it's the forex weekend (Saturday/Sunday).

        Returns:
            bool: True if forex weekend
        """
        try:
            now_est = datetime.now(self.EST)
            return now_est.weekday() >= 5  # Saturday/Sunday
        except Exception as e:
            logger.warning(f"Error checking forex weekend: {e}")
            return False

    def get_forex_session_status(self) -> dict:
        """
        Get detailed forex session status.

        Returns:
            dict: Session information
        """
        try:
            now_est = datetime.now(self.EST)
            now_utc = datetime.now(self.UTC)

            return {
                'current_est': now_est.isoformat(),
                'current_utc': now_utc.isoformat(),
                'weekday': now_est.weekday(),
                'weekday_name': now_est.strftime('%A'),
                'is_weekend': self.is_forex_weekend(),
                'is_trading_day': now_est.weekday() in self.TRADING_DAYS,
                'next_session_start': self._get_next_session_start(),
                'session_status': self._get_session_status()
            }
        except Exception as e:
            logger.warning(f"Error getting session status: {e}")
            return {'error': str(e)}

    def _get_next_session_start(self) -> str:
        """Get next trading session start time."""
        try:
            now_est = datetime.now(self.EST)
            current_weekday = now_est.weekday()

            if current_weekday < 4:  # Monday-Thursday
                # Next session starts tomorrow at midnight EST
                next_session = (now_est + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            elif current_weekday == 4:  # Friday
                # Next session starts Monday at midnight EST
                days_until_monday = 3  # Saturday, Sunday, Monday
                next_session = (now_est + timedelta(days=days_until_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
            else:  # Weekend
                # Next session starts Monday at midnight EST
                days_until_monday = 7 - current_weekday  # Days until next Monday
                next_session = (now_est + timedelta(days=days_until_monday)).replace(hour=0, minute=0, second=0, microsecond=0)

            return next_session.isoformat()
        except Exception as e:
            logger.warning(f"Error calculating next session: {e}")
            return "unknown"

    def _get_session_status(self) -> str:
        """Get current session status description."""
        try:
            now_est = datetime.now(self.EST)
            weekday = now_est.weekday()

            if weekday >= 5:  # Weekend
                return "Forex Weekend - No Trading"
            elif weekday == 4:  # Friday
                return "Friday Trading Session"
            else:  # Monday-Thursday
                return f"{now_est.strftime('%A')} Trading Session"
        except Exception as e:
            return "Unknown"

    def get_next_trading_session_start(self) -> datetime:
        """
        Get the start time of the next trading session.

        Returns:
            datetime: Next trading session start time
        """
        current_time = self.get_current_time()

        # If we're in no-trade buffer, next session starts at midnight
        if self._is_in_no_trade_buffer(current_time.time()):
            next_session = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            next_session += timedelta(days=1)
            return next_session

        # If it's weekend, next session starts Monday at midnight
        if self._is_weekend(current_time.date()):
            days_until_monday = (7 - current_time.weekday()) % 7
            if days_until_monday == 0:  # It's Sunday, wait for Monday
                days_until_monday = 1
            next_session = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            next_session += timedelta(days=days_until_monday)
            return next_session

        # Otherwise, next session starts tomorrow at midnight
        next_session = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        next_session += timedelta(days=1)
        return next_session

    def get_time_until_close(self) -> Optional[timedelta]:
        """
        Get time remaining until daily close.

        Returns:
            timedelta: Time until close, or None if already closed
        """
        current_time = self.get_current_time()
        current_time_only = current_time.time()

        if current_time_only >= self.MT5_CLOSE_TIME:
            return None  # Already closed

        # Calculate time until close
        close_today = current_time.replace(
            hour=self.MT5_CLOSE_TIME.hour,
            minute=self.MT5_CLOSE_TIME.minute,
            second=0,
            microsecond=0
        )

        return close_today - current_time

    def get_trading_status_summary(self) -> dict:
        """
        Get comprehensive trading status summary.

        Returns:
            dict: Status information
        """
        current_time = self.get_current_time()
        is_allowed, reason = self.is_trading_allowed()
        should_close, close_reason = self.should_close_positions()

        return {
            'current_time': current_time.isoformat(),
            'trading_allowed': is_allowed,
            'trading_reason': reason,
            'should_close_positions': should_close,
            'close_reason': close_reason,
            'time_until_close': str(self.get_time_until_close()) if self.get_time_until_close() else None,
            'next_session_start': self.get_next_trading_session_start().isoformat(),
            'is_weekend': self._is_weekend(current_time.date()),
            'forex_market_open': self._is_forex_market_open(),
            'mt5_time_available': self.get_mt5_server_time() is not None
        }

    def reset_daily_closure_flag(self):
        """Reset the daily closure flag (for testing or manual override)"""
        self._last_closure_date = None
        logger.info("Daily closure flag reset")


# Global instance for easy access
_time_manager_instance = None


def get_time_manager(mt5_connector=None) -> TimeManager:
    """
    Get global TimeManager instance (singleton pattern).

    Args:
        mt5_connector: MT5 connector instance

    Returns:
        TimeManager: Global time manager instance
    """
    global _time_manager_instance
    if _time_manager_instance is None:
        _time_manager_instance = TimeManager(mt5_connector)
    elif mt5_connector and _time_manager_instance.mt5 != mt5_connector:
        _time_manager_instance.mt5 = mt5_connector

    return _time_manager_instance
