"""
FX-Ai Application Core
Main application class with initialization and core functionality
"""

import asyncio
import signal
import logging
from datetime import datetime, timezone
from typing import Dict
from utils.logger import setup_logger
from utils.config_loader import ConfigLoader
from app.component_initializer import ComponentInitializer
from app.trading_orchestrator import TradingOrchestrator


class FXAiApplication:
    """Main FX-Ai Trading Application with Adaptive Learning"""

    def __init__(self):
        """Initialize the FX-Ai application"""
        config_loader = ConfigLoader()
        config_loader.load_config()
        self.config = config_loader.config
        self.logger = setup_logger(
            'FX-Ai',
            self.config.get(
                'logging',
                {}).get(
                    'level',
                    'INFO'),
            self.config.get(
                'logging',
                {}).get(
                    'file'),
            rotation_type=self.config.get(
                    'logging',
                    {}).get(
                        'rotation_type',
                'size'))
        self.logger.info(
            "FX-Ai Application initialized with Adaptive Learning")
        self.logger.info("=" * 50)

        # Initialize components
        self.mt5 = None
        self.clock_sync = None
        self.risk_manager = None
        self.market_data = None
        self.fundamental_collector = None
        self.technical_analyzer = None
        self.sentiment_analyzer = None
        self.ml_predictor = None
        self.backtest_engine = None
        self.trading_engine = None
        self.adaptive_learning = None
        self.market_regime_detector = None
        self.reinforcement_agent = None
        self.advanced_risk_metrics = None
        self.correlation_manager = None
        self.param_manager = None

        # Initialize component managers
        self.component_initializer = None
        self.trading_orchestrator = None

        # Control flags
        self.running = False
        self.learning_enabled = self.config.get(
            'ml', {}).get('adaptive_learning', True)

        # Trading parameters
        self.magic_number = self.config.get('trading', {}).get('magic_number', 123456)

        # Background tasks
        self.fundamental_monitor_task = None

        # Performance tracking
        self.session_stats = {
            'start_time': self.get_current_mt5_time(),
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'models_retrained': 0,
            'parameters_optimized': 0,
            'rl_models_saved': 0
        }

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.shutdown_handler)
        signal.signal(signal.SIGTERM, self.shutdown_handler)

    def get_current_mt5_time(self) -> datetime:
        """
        Get current MT5 server time for consistent timestamping across the system.
        This ensures all trading decisions and logging use the same time source.

        Returns:
            datetime: MT5 server time or local time as fallback (always timezone-aware)
        """
        try:
            if self.mt5:
                server_time = self.mt5.get_server_time()
                if server_time:
                    return server_time
        except Exception as e:
            self.logger.warning(f"Failed to get MT5 server time: {e}")

        # Fallback to local time if MT5 time unavailable
        # Make it timezone-aware to match MT5 server time format
        return datetime.now(timezone.utc)

    def shutdown_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown()

    def shutdown(self):
        """Shutdown the application gracefully"""
        self.logger.info("Shutting down FX-Ai application...")

        # Set running flag to False to stop loops
        self.running = False

        # Cancel background tasks
        if self.fundamental_monitor_task and not self.fundamental_monitor_task.done():
            self.fundamental_monitor_task.cancel()
            self.logger.info("Cancelled fundamental monitoring task")

        # Close MT5 connection
        if self.mt5:
            try:
                self.mt5.disconnect()
                self.logger.info("MT5 connection closed")
            except Exception as e:
                self.logger.error(f"Error closing MT5 connection: {e}")

        # Print final session summary
        self.print_session_summary()

        self.logger.info("FX-Ai application shutdown complete")

    def print_session_summary(self):
        """Print comprehensive session statistics"""
        self.logger.info("=" * 60)
        self.logger.info("FX-AI SESSION SUMMARY")
        self.logger.info("=" * 60)

        duration = datetime.now(timezone.utc) - self.session_stats['start_time']
        hours = duration.total_seconds() / 3600

        self.logger.info(f"Session Duration: {hours:.2f} hours")
        self.logger.info(f"Total Trades: {self.session_stats['total_trades']}")
        self.logger.info(f"Winning Trades: {self.session_stats['winning_trades']}")
        self.logger.info(f"Losing Trades: {self.session_stats['losing_trades']}")
        self.logger.info(f"Total Profit: ${self.session_stats['total_profit']:.2f}")

        if self.session_stats['total_trades'] > 0:
            win_rate = (self.session_stats['winning_trades'] / self.session_stats['total_trades']) * 100
            avg_profit = self.session_stats['total_profit'] / self.session_stats['total_trades']
            self.logger.info(f"Win Rate: {win_rate:.1f}%")
            self.logger.info(f"Average Profit per Trade: ${avg_profit:.2f}")

        self.logger.info(f"Models Retrained: {self.session_stats['models_retrained']}")
        self.logger.info(f"Parameters Optimized: {self.session_stats['parameters_optimized']}")
        self.logger.info(f"RL Models Saved: {self.session_stats['rl_models_saved']}")

        self.logger.info("=" * 60)

    async def initialize_components(self) -> bool:
        """
        Initialize all trading system components

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing FX-Ai components...")

            # Initialize component initializer
            self.component_initializer = ComponentInitializer(self)
            success = await self.component_initializer.initialize_components()

            if success:
                # Initialize trading orchestrator
                self.trading_orchestrator = TradingOrchestrator(self)
                self.logger.info("All components initialized successfully")
                return True
            else:
                self.logger.error("Failed to initialize components")
                return False

        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            return False

    async def run(self):
        """Run the FX-Ai application"""
        self.logger.info("=" * 50)
        self.logger.info("FX-Ai Trading System Starting...")
        self.logger.info("=" * 50)

        # Initialize components
        if not await self.initialize_components():
            self.logger.error("Failed to initialize components")
            return

        # Set running flag to True before starting trading loop
        self.running = True

        # Start trading loop
        try:
            await self.trading_orchestrator.trading_loop()
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
        finally:
            self.shutdown()