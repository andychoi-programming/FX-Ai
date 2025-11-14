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

    # Time Zones (constants that don't change)
    EST = pytz.timezone('US/Eastern')
    UTC = pytz.timezone('UTC')

    def __init__(self, mt5_connector=None, config=None):
        """
        Initialize TimeManager

        Args:
            mt5_connector: MT5 connector instance for server time
            config: Configuration dictionary
        """
        self.mt5 = mt5_connector
        self._last_closure_date = None
        self.config = config

        # Load time configuration from config
        self._load_time_config()

    def _load_time_config(self):
        """Load time configuration from config.json"""
        if not self.config:
            # Fallback to hardcoded values if no config provided
            self.MT5_CLOSE_TIME = time(22, 30)
            self.MT5_CLOSE_BUFFER_END = time(23, 59)
            self.MT5_OPEN_TIME = time(0, 0)
            self.MT5_TRADING_START = time(1, 0)
            self.MT5_FORCE_CLOSE_TIME = time(20, 30)
            self.MT5_IMMEDIATE_CLOSE_TIME = time(22, 0)
            self.FOREX_WEEKEND_START = "Friday 17:00 EST"
            self.FOREX_WEEKEND_END = "Sunday 17:00 EST"
            self.TRADING_DAYS = [0, 1, 2, 3, 4]
            self.TRADING_START_EST = time(0, 0)
            self.TRADING_END_EST = time(17, 0)
            return

        time_config = self.config.get('time_restrictions', {})
        mt5_times = time_config.get('mt5_trading_times', {})
        forex_hours = time_config.get('forex_market_hours', {})
        discipline = time_config.get('trading_discipline', {})

        # Parse time strings to time objects
        def parse_time(time_str):
            if isinstance(time_str, str) and ':' in time_str:
                hours, minutes = map(int, time_str.split(':'))
                return time(hours, minutes)
            return time(22, 30)  # fallback

        self.MT5_CLOSE_TIME = parse_time(mt5_times.get('mt5_close_time', '22:30'))
        self.MT5_CLOSE_BUFFER_END = parse_time(mt5_times.get('mt5_close_buffer_end', '23:59'))
        self.MT5_OPEN_TIME = parse_time(mt5_times.get('mt5_open_time', '00:00'))
        self.MT5_TRADING_START = parse_time(mt5_times.get('mt5_trading_start', '01:00'))
        self.MT5_FORCE_CLOSE_TIME = parse_time(mt5_times.get('mt5_force_close_time', '20:30'))
        self.MT5_IMMEDIATE_CLOSE_TIME = parse_time(mt5_times.get('mt5_immediate_close_time', '22:00'))

        self.FOREX_WEEKEND_START = forex_hours.get('weekend_start', 'Friday 17:00 EST')
        self.FOREX_WEEKEND_END = forex_hours.get('weekend_end', 'Sunday 17:00 EST')

        self.TRADING_DAYS = discipline.get('trading_days', [0, 1, 2, 3, 4])
        self.TRADING_START_EST = parse_time(discipline.get('trading_start_est', '00:00'))
        self.TRADING_END_EST = parse_time(discipline.get('trading_end_est', '17:00'))

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
        Trading allowed 1 hour after market open (01:00 MT5 time).

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

        # Check if we're before trading start time (1 hour after market open)
        if current_time_only < self.MT5_TRADING_START:
            start_time_str = f"{self.MT5_TRADING_START.hour:02d}:{self.MT5_TRADING_START.minute:02d}"
            return False, f"Before trading start time ({start_time_str} MT5 time)"

        # Check forex market hours (EST)
        if not self._is_forex_market_open():
            return False, "Forex market closed (outside EST trading hours)"

        return True, "Trading allowed"

    def should_close_positions(self) -> Tuple[bool, str]:
        """
        Check if all positions should be closed.
        - Close all positions 2 hours before market close (20:30 MT5 time)
        - Close any remaining positions immediately after 22:00 MT5 time

        Returns:
            Tuple[bool, str]: (should_close, reason)
        """
        current_time = self.get_current_time()
        current_date = current_time.date()

        # Only close once per day (for weekend closure)
        if self._last_closure_date == current_date:
            return False, "Already processed daily closure"

        # Check if it's Friday after 22:00 (close before weekend)
        if self._is_friday_evening_close():
            self._last_closure_date = current_date
            return True, "Friday evening - closing before weekend"

        # Check if it's after immediate close time (22:00 MT5 time)
        current_time_only = current_time.time()
        if current_time_only >= self.MT5_IMMEDIATE_CLOSE_TIME:
            self._last_closure_date = current_date
            return True, f"After immediate close time ({self.MT5_IMMEDIATE_CLOSE_TIME.strftime('%H:%M')} MT5 time)"

        # Check if it's time to close all positions (2 hours before market close)
        if current_time_only >= self.MT5_FORCE_CLOSE_TIME:
            self._last_closure_date = current_date
            close_time_str = f"{self.MT5_FORCE_CLOSE_TIME.hour:02d}:{self.MT5_FORCE_CLOSE_TIME.minute:02d}"
            return True, f"2 hours before market close ({close_time_str} MT5 time)"

        # Check if system is shutting down (this would be set externally)
        if self._is_weekend(current_time.date()):
            return True, "Weekend - forex markets closed"

        return False, "AI-driven timing: no forced daily closure"

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

    def _is_friday_evening_close(self) -> bool:
        """
        Check if it's Friday evening and time to close before weekend.

        Returns:
            bool: True if should close for weekend
        """
        try:
            now_est = datetime.now(self.EST)
            return now_est.weekday() == 4 and now_est.hour >= 22  # Friday and past 22:00 EST
        except Exception as e:
            logger.warning(f"Error checking Friday evening close: {e}")
            return False

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

    def get_current_session(self) -> str:
        """
        Get the current forex trading session.

        Sessions:
        - Sydney: 22:00-07:00 GMT (overlaps with Tokyo start, London end)
        - Tokyo: 00:00-09:00 GMT (overlaps with Sydney end, London start)
        - London: 08:00-16:00 GMT (overlaps with Tokyo end, NY start)
        - New York: 13:30-22:00 GMT (overlaps with London end, Sydney start)

        Returns:
            str: Current session ('sydney', 'tokyo', 'london', 'newyork', 'overlap', 'none')
        """
        try:
            # Use MT5 server time for consistent session detection
            current_time = self.get_current_time()
            hour = current_time.hour
            minute = current_time.minute

            # Check for overlap periods first (most specific conditions)
            # London/NY overlap: 13:00-16:30 GMT
            if (hour == 13 and minute >= 0) or (hour == 14 or hour == 15) or (hour == 16 and minute < 30):
                return 'overlap'

            # Tokyo/London overlap: 08:00-09:00 GMT
            elif hour == 8:
                return 'overlap'

            # Sydney/Tokyo overlap: 00:00-01:00 GMT
            elif hour == 0:
                return 'overlap'

            # Sydney/London overlap: 16:00-17:00 GMT (London close, Sydney still active briefly)
            elif hour == 16 and minute >= 30:
                return 'overlap'

            # Sydney session: 22:00-07:00 GMT (most active Asian session)
            elif hour >= 22 or hour < 7:
                return 'sydney'

            # Tokyo session: 01:00-08:00 GMT
            elif 1 <= hour < 8:
                return 'tokyo'

            # London session: 08:00-16:00 GMT (most active European trading)
            elif 8 <= hour < 16:
                return 'london'

            # New York session: 13:30-22:00 GMT
            elif (hour == 13 and minute >= 30) or (14 <= hour < 22):
                return 'newyork'

            else:
                return 'none'

        except Exception as e:
            logger.warning(f"Error determining current session: {e}")
            return 'none'

    def is_preferred_session(self, config: dict) -> bool:
        """
        Check if current session is in preferred sessions list.

        Args:
            config: Configuration dictionary

        Returns:
            bool: True if current session is preferred
        """
        try:
            current_session = self.get_current_session()
            session_config = config.get('trading_rules', {}).get('session_filter', {})
            preferred_sessions = session_config.get('preferred_sessions')

            return current_session in preferred_sessions

        except Exception as e:
            logger.warning(f"Error checking preferred session: {e}")
            return True  # Default to allowing if check fails

    def get_session_signal_threshold(self, config: dict) -> float:
        """
        Get session-aware signal strength threshold.

        Args:
            config: Configuration dictionary

        Returns:
            float: Adjusted signal threshold for current session
        """
        try:
            current_session = self.get_current_session()
            base_threshold = config.get('trading', {}).get('min_signal_strength')

            # Try to get session-specific thresholds from config first
            session_config = config.get('trading_rules', {}).get('session_filter', {})
            config_thresholds = session_config.get('signal_thresholds')

            # Default session-specific thresholds (can be overridden by config)
            default_thresholds = {
                'london': 0.550,    # Most active session - lower threshold
                'newyork': 0.575,  # Second most active
                'overlap': 0.550,  # Session overlaps are good
                'tokyo': 0.600,    # Standard threshold
                'none': 0.650      # Higher threshold when no active session
            }

            # Use config thresholds if available, otherwise use defaults
            session_thresholds = {**default_thresholds, **config_thresholds}
            session_threshold = session_thresholds.get(current_session, base_threshold)

            # Ensure threshold doesn't go below minimum (adjusted for current signal quality)
            min_threshold = config.get('performance_thresholds', {}).get('min_session_threshold', 0.200)
            return max(min_threshold, min(session_threshold, base_threshold))

        except Exception as e:
            logger.warning(f"Error getting session threshold: {e}")
            return config.get('trading', {}).get('min_signal_strength')

    def is_optimal_trading_hour(self, config: dict) -> bool:
        """
        Check if current hour is optimal for trading based on learned performance.

        Args:
            config: Configuration dictionary with learned hourly weights

        Returns:
            bool: True if current hour has above-average performance weight
        """
        try:
            # Use MT5 server time for consistent hour checking
            current_time = self.get_current_time()
            current_hour = current_time.hour

            # Get hourly weights from adaptive learning config
            adaptive_config = config.get('adaptive_learning', {}).get('session_time_optimization', {})
            hourly_weights = adaptive_config.get('hourly_weights', {})

            if not hourly_weights:
                return True  # No optimization data, allow trading

            current_weight = hourly_weights.get(str(current_hour), 1.0)
            avg_weight = sum(hourly_weights.values()) / len(hourly_weights)

            # Consider hour optimal if weight is above average
            return current_weight >= avg_weight * 0.9  # 90% of average threshold

        except Exception as e:
            logger.warning(f"Error checking optimal trading hour: {e}")
            return True  # Default to allowing if check fails

    def get_hourly_performance_weight(self, config: dict) -> float:
        """
        Get the learned performance weight for the current hour.

        Args:
            config: Configuration dictionary

        Returns:
            float: Performance weight (higher = better performance)
        """
        try:
            # Use MT5 server time for consistent weight lookup
            current_time = self.get_current_time()
            current_hour = current_time.hour
            adaptive_config = config.get('adaptive_learning', {}).get('session_time_optimization', {})
            hourly_weights = adaptive_config.get('hourly_weights', {})

            return hourly_weights.get(str(current_hour), 1.0)

        except Exception as e:
            logger.warning(f"Error getting hourly performance weight: {e}")
            return 1.0

    def get_session_info(self) -> dict:
        """
        Get comprehensive session information.

        Returns:
            dict: Session details
        """
        try:
            current_session = self.get_current_session()
            now_utc = datetime.now(self.UTC)

            session_hours = {
                'london': {'start': 8, 'end': 16, 'name': 'London'},
                'newyork': {'start': 13, 'end': 22, 'name': 'New York'},
                'tokyo': {'start': 0, 'end': 9, 'name': 'Tokyo'},
                'overlap': {'start': None, 'end': None, 'name': 'Session Overlap'},
                'none': {'start': None, 'end': None, 'name': 'No Active Session'}
            }

            session_data = session_hours.get(current_session, session_hours['none'])

            return {
                'current_session': current_session,
                'session_name': session_data['name'],
                'session_start_hour': session_data['start'],
                'session_end_hour': session_data['end'],
                'current_hour_utc': now_utc.hour,
                'is_active_session': current_session != 'none',
                'is_london_session': current_session == 'london',
                'is_overlap': current_session == 'overlap'
            }

        except Exception as e:
            logger.warning(f"Error getting session info: {e}")
            return {'error': str(e)}


# Global instance for easy access
_time_manager_instance = None


def get_time_manager(mt5_connector=None, config=None) -> TimeManager:
    """
    Get global TimeManager instance (singleton pattern).

    Args:
        mt5_connector: MT5 connector instance
        config: Configuration dictionary

    Returns:
        TimeManager: Global time manager instance
    """
    global _time_manager_instance
    if _time_manager_instance is None:
        _time_manager_instance = TimeManager(mt5_connector, config)
    elif mt5_connector and _time_manager_instance.mt5 != mt5_connector:
        _time_manager_instance.mt5 = mt5_connector
    if config and _time_manager_instance.config != config:
        _time_manager_instance.config = config
        _time_manager_instance._load_time_config()

    return _time_manager_instance
