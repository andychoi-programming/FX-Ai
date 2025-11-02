import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from live_trading.ml_trading_system import MLTradingSystem
import schedule

class TradingOrchestrator:
    """Orchestrates the complete ML trading system"""

    def __init__(self, config_file: str = "config/trading_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.trading_system = None
        self.is_running = False

        # Setup logging
        self._setup_logging()

        # Performance tracking
        self.performance_log = Path("logs/performance.json")
        self.daily_stats = {}

        self.logger = logging.getLogger(__name__)

    def _load_config(self) -> Dict:
        """Load trading configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "trading": {
                    "symbols": [
                        "EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "EURJPY",
                        "GBPJPY", "AUDJPY", "USDCAD", "EURCAD", "GBPCAD",
                        "AUDCAD", "USDCHF", "EURCHF", "GBPCHF", "AUDCHF",
                        "NZDUSD", "EURNZD", "GBPNZD", "AUDNZD", "NZDJPY",
                        "CADJPY", "CHFJPY", "NZDCAD", "CADCHF", "NZDCHF"
                    ],
                    "timeframes": ["H1", "D1"],
                    "risk_per_trade": 50,
                    "max_positions": 3,
                    "trading_hours": {
                        "start": "08:00",
                        "end": "20:00"
                    }
                },
                "system": {
                    "cycle_interval_minutes": 15,
                    "max_daily_trades": 10,
                    "daily_loss_limit": 200,
                    "log_level": "INFO"
                }
            }

    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "trading_system.log"),
                logging.StreamHandler()
            ]
        )

    def initialize_system(self):
        """Initialize the trading system"""
        try:
            self.logger.info("Initializing ML Trading System...")
            self.trading_system = MLTradingSystem(self.config)
            self.logger.info("System initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            return False

    def is_trading_hours(self) -> bool:
        """Check if current time is within trading hours"""
        now = datetime.now().time()
        start_time = datetime.strptime(self.config['trading']['trading_hours']['start'], "%H:%M").time()
        end_time = datetime.strptime(self.config['trading']['trading_hours']['end'], "%H:%M").time()

        return start_time <= now <= end_time

    def check_daily_limits(self) -> bool:
        """Check if daily trading limits have been reached"""
        today = datetime.now().date().isoformat()

        if today not in self.daily_stats:
            self.daily_stats[today] = {
                'trades': 0,
                'pnl': 0.0,
                'winning_trades': 0,
                'losing_trades': 0
            }

        # Check trade limit
        if self.daily_stats[today]['trades'] >= self.config['system']['max_daily_trades']:
            self.logger.info("Daily trade limit reached")
            return False

        # Check loss limit
        if self.daily_stats[today]['pnl'] <= -self.config['system']['daily_loss_limit']:
            self.logger.warning("Daily loss limit reached")
            return False

        return True

    def run_trading_cycle(self):
        """Execute one complete trading cycle"""
        try:
            if not self.is_trading_hours():
                self.logger.debug("Outside trading hours")
                return

            if not self.check_daily_limits():
                self.logger.info("Daily limits reached, skipping cycle")
                return

            self.logger.info("Executing trading cycle...")
            self.trading_system.run_trading_cycle()

            # Update performance stats
            self._update_performance_stats()

        except Exception as e:
            self.logger.error(f"Trading cycle failed: {e}")

    def _update_performance_stats(self):
        """Update daily performance statistics"""
        try:
            # Get current positions and calculate P&L
            status = self.trading_system.get_system_status()

            today = datetime.now().date().isoformat()
            if today not in self.daily_stats:
                self.daily_stats[today] = {
                    'trades': 0,
                    'pnl': 0.0,
                    'winning_trades': 0,
                    'losing_trades': 0
                }

            # This is a simplified version - in production you'd track closed trades
            # For now, just log active positions
            self.logger.info(f"Active positions: {status['active_positions']}")

        except Exception as e:
            self.logger.error(f"Performance update failed: {e}")

    def start_trading(self):
        """Start the automated trading system"""
        if not self.initialize_system():
            self.logger.error("Failed to initialize system")
            return

        self.is_running = True
        self.logger.info("Starting automated trading...")

        # Schedule trading cycles
        interval = self.config['system']['cycle_interval_minutes']
        schedule.every(interval).minutes.do(self.run_trading_cycle)

        # Run initial cycle
        self.run_trading_cycle()

        # Main trading loop
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            self.logger.info("Trading stopped by user")
        except Exception as e:
            self.logger.error(f"Trading loop error: {e}")
        finally:
            self.stop_trading()

    def stop_trading(self):
        """Stop the trading system gracefully"""
        self.logger.info("Stopping trading system...")
        self.is_running = False

        if self.trading_system:
            # Close all positions
            for symbol in list(self.trading_system.active_positions.keys()):
                self.trading_system.close_position(symbol, "System shutdown")

        # Save final performance stats
        self._save_performance_stats()

        self.logger.info("Trading system stopped")

    def _save_performance_stats(self):
        """Save performance statistics to file"""
        try:
            with open(self.performance_log, 'w') as f:
                json.dump(self.daily_stats, f, indent=2, default=str)
            self.logger.info("Performance stats saved")
        except Exception as e:
            self.logger.error(f"Failed to save performance stats: {e}")

    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        status = {
            'is_running': self.is_running,
            'trading_hours': self.is_trading_hours(),
            'daily_limits_ok': self.check_daily_limits(),
            'config': self.config
        }

        if self.trading_system:
            status.update(self.trading_system.get_system_status())

        # Add performance summary
        today = datetime.now().date().isoformat()
        if today in self.daily_stats:
            status['daily_stats'] = self.daily_stats[today]

        return status

    def print_status_report(self):
        """Print comprehensive status report"""
        status = self.get_system_status()

        print("\n" + "="*60)
        print("ML TRADING SYSTEM STATUS REPORT")
        print("="*60)
        print(f"System Running: {status['is_running']}")
        print(f"Trading Hours: {status['trading_hours']}")
        print(f"Daily Limits OK: {status['daily_limits_ok']}")

        if 'active_positions' in status:
            print(f"Active Positions: {status['active_positions']}")
            print(f"Total Symbols: {status['total_symbols']}")

        if 'daily_stats' in status:
            stats = status['daily_stats']
            print(f"\nDaily Stats:")
            print(f"  Trades: {stats['trades']}")
            print(f"  P&L: ${stats['pnl']:.2f}")
            print(f"  Winning Trades: {stats['winning_trades']}")
            print(f"  Losing Trades: {stats['losing_trades']}")

        if 'positions' in status and status['positions']:
            print(f"\nActive Positions:")
            for pos in status['positions']:
                print(f"  {pos['symbol']}: {pos['direction']} @ {pos['entry_price']:.5f} "
                      f"({pos['lot_size']} lots)")

        print("="*60)

def main():
    """Main entry point"""
    orchestrator = TradingOrchestrator()

    # Print initial status
    orchestrator.print_status_report()

    # Start trading
    try:
        orchestrator.start_trading()
    except KeyboardInterrupt:
        print("\nStopping trading system...")
    finally:
        orchestrator.print_status_report()

if __name__ == "__main__":
    main()