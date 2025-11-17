"""
FX-Ai Schedule Manager
Manages optimal trading hours for each symbol
All times in MT5 SERVER time (GMT+2) - NO timezone conversions!
"""

import json
import logging
from datetime import datetime
from pathlib import Path


class ScheduleManager:
    """
    Manages trading schedules for all 30 symbols
    All times in SERVER time only - what MT5 shows!
    """

    def __init__(self, config_path="config/symbol_schedules.json"):
        """
        Initialize schedule manager

        Args:
            config_path: Path to JSON config file with schedules
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.schedules = {}
        self.global_settings = {}

        # Load schedules from config file
        self._load_schedules()

        self.logger.info(f"ScheduleManager initialized with {len(self.schedules)} symbol schedules")

    def _load_schedules(self):
        """Load schedules from JSON config file"""
        try:
            config_file = Path(self.config_path)

            if not config_file.exists():
                self.logger.error(f"Config file not found: {self.config_path}")
                self._use_default_schedules()
                return

            with open(config_file, 'r') as f:
                config = json.load(f)

            self.global_settings = config.get('global_settings', {})
            self.schedules = config.get('symbol_schedules', {})

            if not self.schedules:
                self.logger.warning("No symbol schedules found, using defaults")
                self._use_default_schedules()

        except Exception as e:
            self.logger.error(f"Error loading schedules: {e}")
            self._use_default_schedules()

            self.logger.info(f"Loaded schedules for {len(self.schedules)} symbols")

        except Exception as e:
            self.logger.error(f"Error loading schedules: {e}")
            self._use_default_schedules()

    def _use_default_schedules(self):
        """Fallback to default schedules if config not found"""
        self.logger.warning("Using default schedules (08:00-23:00 for all symbols)")

        self.global_settings = {
            'enable_24hour_trading': True,
            'force_close_hour': 23,
            'force_close_minute': 55
        }

        # Default: 08:00-23:00 for all symbols
        default_schedule = {'start_hour': 8, 'end_hour': 23}

        symbols = self.config.get('schedule_symbols', {}).get('trading_symbols', [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD',
            'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD',
            'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNZD',
            'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD',
            'NZDCAD', 'NZDCHF', 'NZDJPY',
            'CADCHF', 'CADJPY', 'CHFJPY',
            'XAUUSD', 'XAGUSD'
        ])

        for symbol in symbols:
            self.schedules[symbol] = default_schedule

    def can_trade_symbol(self, symbol, current_time=None):
        """
        Check if we can trade this symbol RIGHT NOW

        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'GBPJPY')
            current_time: Optional datetime object to use instead of now()

        Returns:
            bool: True if symbol can be traded now, False otherwise
        """
        # Get current time (server time from system)
        if current_time is None:
            now = datetime.now()
        else:
            now = current_time
        current_hour = now.hour
        current_minute = now.minute

        # Check if symbol has a schedule
        if symbol not in self.schedules:
            self.logger.warning(f"No schedule found for {symbol}, using default 08:00-23:00")
            return 8 <= current_hour < 23

        schedule = self.schedules[symbol]
        start_hour = schedule['start_hour']
        end_hour = schedule['end_hour']

        # Handle schedules that cross midnight
        # Example: start=22, end=6 means trade from 22:00 to 06:00 next day
        if start_hour > end_hour:
            # Crosses midnight
            can_trade = (current_hour >= start_hour) or (current_hour < end_hour)
        else:
            # Normal schedule within same day
            can_trade = start_hour <= current_hour < end_hour

        # TEMPORARY DEBUG LOGGING for Sydney session testing
        self.logger.info(f"Schedule check: {symbol} @ {current_hour}:{current_minute:02d} = {can_trade} "
                        f"(Schedule: {start_hour:02d}:00-{end_hour:02d}:00)")

        return can_trade

    def should_force_close_all(self):
        """
        Check if it's time to force close all positions

        Returns:
            bool: True if should close all positions
        """
        if not self.global_settings.get('enable_24hour_trading', True):
            # 24-hour trading disabled, use simple close
            return datetime.now().hour == 23

        # Force close at specific time (default: 23:45 server)
        force_hour = self.global_settings.get('force_close_hour', 23)
        force_minute = self.global_settings.get('force_close_minute', 45)

        now = datetime.now()

        # Close during the force close window (e.g., 23:45-23:59)
        if now.hour == force_hour and now.minute >= force_minute:
            return True

        return False

    def get_schedule_info(self, symbol):
        """
        Get human-readable schedule info for a symbol

        Args:
            symbol: Trading symbol

        Returns:
            str: Schedule description
        """
        if symbol not in self.schedules:
            return "No schedule (default: 08:00-23:00 server)"

        schedule = self.schedules[symbol]
        start = f"{schedule['start_hour']:02d}:00"
        end = f"{schedule['end_hour']:02d}:00"
        comment = schedule.get('comment', '')

        return f"{start}-{end} server | {comment}"

    def get_active_symbols_now(self, all_symbols):
        """
        Get list of symbols that can be traded right now

        Args:
            all_symbols: List of all available symbols

        Returns:
            list: Symbols that are currently in their trading hours
        """
        active = []

        for symbol in all_symbols:
            if self.can_trade_symbol(symbol):
                active.append(symbol)

        return active

    def get_schedule_summary(self):
        """
        Get summary of current schedule status

        Returns:
            dict: Summary information
        """
        now = datetime.now()
        current_hour = now.hour

        total_symbols = len(self.schedules)
        active_symbols = sum(1 for symbol in self.schedules if self.can_trade_symbol(symbol))

        return {
            'current_server_time': f"{current_hour:02d}:{now.minute:02d}",
            'total_symbols': total_symbols,
            'active_symbols': active_symbols,
            'inactive_symbols': total_symbols - active_symbols,
            'force_close_enabled': self.global_settings.get('enable_24hour_trading', True),
            'force_close_time': f"{self.global_settings.get('force_close_hour', 23):02d}:{self.global_settings.get('force_close_minute', 55):02d}"
        }

    def log_schedule_status(self):
        """Log current schedule status"""
        summary = self.get_schedule_summary()

        self.logger.info("=" * 60)
        self.logger.info(f"SCHEDULE STATUS (Server Time)")
        self.logger.info("=" * 60)
        self.logger.info(f"Current Time: {summary['current_server_time']} server")
        self.logger.info(f"Active Symbols: {summary['active_symbols']}/{summary['total_symbols']}")
        self.logger.info(f"Force Close: {summary['force_close_time']} server")
        self.logger.info("=" * 60)

    def get_next_trading_time(self, symbol):
        """
        Get when this symbol will next be tradeable

        Args:
            symbol: Trading symbol

        Returns:
            str: Description of next trading time
        """
        if self.can_trade_symbol(symbol):
            return "TRADING NOW"

        if symbol not in self.schedules:
            return "Unknown (no schedule)"

        schedule = self.schedules[symbol]
        start_hour = schedule['start_hour']
        current_hour = datetime.now().hour

        if current_hour < start_hour:
            hours_until = start_hour - current_hour
            return f"Opens in {hours_until}h at {start_hour:02d}:00 server"
        else:
            hours_until = (24 - current_hour) + start_hour
            return f"Opens in {hours_until}h at {start_hour:02d}:00 server (tomorrow)"

    def get_current_session(self, current_time=None):
        """
        Determine the current trading session based on MT5 server time (GMT+2).

        Market sessions in UTC:
        - Sydney: 22:00-07:00 UTC -> 00:00-09:00 GMT+2
        - Tokyo: 00:00-09:00 UTC -> 02:00-11:00 GMT+2
        - London: 08:00-16:30 UTC -> 10:00-18:30 GMT+2
        - New York: 13:00-22:00 UTC -> 15:00-00:00 GMT+2

        Args:
            current_time: datetime object (uses MT5 server time if None)

        Returns:
            str: Session name ('tokyo_sydney', 'london', 'new_york', 'overlap', or 'closed')
        """
        if current_time is None:
            current_time = datetime.now()

        hour = current_time.hour

        # All times in GMT+2 (MT5 server time)

        # New York session (15:00-00:00 server time / 13:00-22:00 UTC)
        if 15 <= hour < 24:
            # Check for overlap with London (15:00-18:30)
            if 15 <= hour < 19:  # 18:30 = 18 for simplicity
                return "overlap"  # London-NY overlap
            return "new_york"

        # Sydney/Tokyo session (00:00-11:00 server time)
        if 0 <= hour < 11:
            # Check for overlap with London (10:00-11:00)
            if 10 <= hour < 11:
                return "overlap"  # Tokyo-London overlap
            return "tokyo_sydney"

        # London session (10:00-18:30 server time / 08:00-16:30 UTC)
        if 10 <= hour < 19:  # Using 19:00 to include 18:30
            # Already checked overlap with Tokyo above
            # Already checked overlap with NY above
            return "london"

        # Closed / Low liquidity (rare in 24-hour forex)
        return "closed"


# ===== Convenience Functions =====

def create_schedule_manager(config_path="config/symbol_schedules.json"):
    """
    Factory function to create schedule manager

    Args:
        config_path: Path to config file

    Returns:
        ScheduleManager: Initialized schedule manager
    """
    return ScheduleManager(config_path)


def can_trade_now(symbol, schedule_manager):
    """
    Quick check if symbol can be traded now

    Args:
        symbol: Trading symbol
        schedule_manager: ScheduleManager instance

    Returns:
        bool: True if can trade
    """
    return schedule_manager.can_trade_symbol(symbol)