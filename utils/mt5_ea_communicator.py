# FX-Ai MT5 EA Communicator
# Handles communication between Python FX-Ai system and MT5 Expert Advisor

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

class MT5EACommunicator:
    """Communicates with FX-Ai MT5 Expert Advisor via file-based signals"""

    def __init__(self, signal_file: str = "fxai_signals.txt", logger=None):
        self.signal_file = signal_file
        self.logger = logger or logging.getLogger(__name__)

        # Get MT5 data path (Common Files folder)
        self.mt5_data_path = self._get_mt5_data_path()
        self.signal_file_path = self.mt5_data_path / "MQL5" / "Files" / signal_file

        self.logger.info(f"MT5 EA Communicator initialized. Signal file: {self.signal_file_path}")

    def _get_mt5_data_path(self) -> Path:
        """Get MT5 data path for file communication"""
        # Try to find MT5 data path
        possible_paths = [
            Path(os.environ.get('APPDATA', '')) / "MetaQuotes" / "Terminal" / "Common",
            Path("C:/Users/Public/Documents/MetaQuotes/Terminal/Common"),
            Path("C:/ProgramData/MetaQuotes/Terminal/Common"),
        ]

        for path in possible_paths:
            if path.exists():
                return path

        # Fallback to current directory
        self.logger.warning("MT5 data path not found, using current directory")
        return Path.cwd()

    def send_signal(self, symbol: str, direction: str, entry_price: float = 0.0,
                   stop_loss: int = 0, take_profit: int = 0, lot_size: float = 0.01) -> bool:
        """Send trading signal to MT5 EA

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            direction: 'BUY' or 'SELL'
            entry_price: Entry price (0.0 for market order)
            stop_loss: Stop loss in points
            take_profit: Take profit in points
            lot_size: Position size in lots

        Returns:
            bool: True if signal sent successfully
        """
        try:
            # Validate inputs
            if direction not in ['BUY', 'SELL']:
                self.logger.error(f"Invalid direction: {direction}")
                return False

            # Create signal line
            timestamp = datetime.now().strftime("%Y.%m.%d %H:%M:%S")
            signal_line = ",".join([
                symbol,
                direction,
                f"{entry_price:.5f}",
                str(stop_loss),
                str(take_profit),
                f"{lot_size:.5f}",
                timestamp
            ])

            # Ensure signal file directory exists
            self.signal_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write signal to file (append mode)
            with open(self.signal_file_path, 'a', encoding='utf-8') as f:
                f.write(signal_line + '\n')

            self.logger.info(f"Signal sent to MT5 EA: {symbol} {direction}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send signal to MT5 EA: {e}")
            return False

    def clear_signals(self) -> bool:
        """Clear all signals from the file"""
        try:
            # Keep header but clear signals
            header = "# FX-Ai Signal File\n# Format: SYMBOL,DIRECTION,PRICE,STOPLOSS,TAKEPROFIT,LOTSIZE,TIMESTAMP\n"

            with open(self.signal_file_path, 'w', encoding='utf-8') as f:
                f.write(header)

            self.logger.info("Signals cleared")
            return True

        except Exception as e:
            self.logger.error(f"Failed to clear signals: {e}")
            return False

    def get_signal_count(self) -> int:
        """Get number of pending signals"""
        try:
            if not self.signal_file_path.exists():
                return 0

            count = 0
            with open(self.signal_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        count += 1

            return count

        except Exception as e:
            self.logger.error(f"Failed to count signals: {e}")
            return 0

    def is_ea_connected(self) -> bool:
        """Check if MT5 EA is reading signals (basic check)"""
        try:
            # Check if signal file exists and is accessible
            return self.signal_file_path.exists() and self.signal_file_path.stat().st_size > 0
        except:
            return False

# Example usage and integration with FX-Ai system
if __name__ == "__main__":
    # Example integration
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize communicator
    ea_comm = MT5EACommunicator(logger=logger)

    # Example: Send a buy signal
    success = ea_comm.send_signal(
        symbol="EURUSD",
        direction="BUY",
        entry_price=0.0,  # Market order
        stop_loss=500,    # 50 pips
        take_profit=1000  # 100 pips
    )

    if success:
        print("✅ Signal sent to MT5 EA")
    else:
        print("❌ Failed to send signal")

    # Check signal count
    count = ea_comm.get_signal_count()
    print(f"📊 Pending signals: {count}")