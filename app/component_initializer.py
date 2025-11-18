"""
FX-Ai Component Initializer
Handles initialization of all trading system components
"""

import os
import logging
import sys
from pathlib import Path

# Add modules directory to path for ScheduleManager
sys.path.append(str(Path(__file__).parent.parent / 'modules'))

from utils.time_manager import get_time_manager
from utils.config_loader import ConfigLoader
from ai.adaptive_learning_manager import AdaptiveLearningManager
from ai.market_regime_detector import MarketRegimeDetector
from ai.reinforcement_learning_agent import RLAgent
from ai.advanced_risk_metrics import AdvancedRiskMetrics
from ai.correlation_manager import CorrelationManager
from ai.ml_predictor import MLPredictor
from core.dynamic_parameter_manager import DynamicParameterManager
from analysis.sentiment_analyzer import SentimentAnalyzer
from analysis.technical_analyzer import TechnicalAnalyzer
from analysis.fundamental_analyzer import (
    FundamentalAnalyzer as FundamentalDataCollector)
from data.market_data_manager import MarketDataManager
from core.clock_sync import ClockSynchronizer
from core.risk_manager import RiskManager
from schedule_manager import ScheduleManager  # type: ignore
from core.trading_engine import TradingEngine
from core.mt5_connector import MT5Connector
from core.position_sync_manager import PositionSyncManager
from ai.learning_database import LearningDatabase
from utils.logger import setup_logger


class ComponentInitializer:
    """Handles initialization of all FX-Ai trading components"""

    def __init__(self, app):
        """Initialize with reference to main application"""
        self.app = app

    async def validate_configuration(self):
        """Validate critical configuration before trading starts"""
        from utils.exceptions import ConfigurationError, MissingParametersError

        self.app.logger.info("Validating system configuration...")

        # 1. Check optimal_parameters.json exists
        params_path = "models/parameter_optimization/optimal_parameters.json"
        if not os.path.exists(params_path):
            raise MissingParametersError(
                f"Optimized parameters file not found: {params_path}\n"
                "Please run optimization first: python backtest/optimize_fast_3year.py"
            )

        # 2. Verify all trading symbols have optimized parameters
        try:
            with open(params_path, 'r') as f:
                optimal_params = __import__('json').load(f)

            trading_symbols = self.app.config.get('trading', {}).get('symbols', [])
            missing_symbols = []

            for symbol in trading_symbols:
                if symbol not in optimal_params:
                    missing_symbols.append(symbol)

            if missing_symbols:
                raise MissingParametersError(
                    f"Missing optimized parameters for symbols: {missing_symbols}\n"
                    f"Please run optimization for these symbols."
                )

            self.app.logger.info(f"[OK] All {len(trading_symbols)} symbols have optimized parameters")

        except FileNotFoundError:
            raise MissingParametersError(f"Could not read parameters file: {params_path}")
        except Exception as e:
            raise ConfigurationError(f"Error validating parameters: {e}")

        # 3. Validate MT5 configuration
        mt5_config = self.app.config.get('mt5', {})
        required_mt5_keys = ['login', 'password', 'server']
        missing_mt5 = [key for key in required_mt5_keys if not mt5_config.get(key)]

        if missing_mt5:
            raise ConfigurationError(f"Missing MT5 configuration: {missing_mt5}")

        self.app.logger.info("[OK] Configuration validation passed")

    async def initialize_components(self):
        """Initialize all trading components"""
        try:
            self.app.logger.info("Initializing components...")

            # Validate configuration first
            await self.validate_configuration()

            # 1. MT5 Connection
            self.app.logger.info("Initializing MT5 connection...")
            # Load MT5 credentials from environment variables
            import os
            from dotenv import load_dotenv
            load_dotenv()

            self.app.mt5 = MT5Connector(
                os.getenv('MT5_LOGIN'),
                os.getenv('MT5_PASSWORD'),
                os.getenv('MT5_SERVER'),
                os.getenv('MT5_PATH')
            )
            if not self.app.mt5.connect():
                raise Exception("Failed to connect to MT5")

            # Initialize Time Manager
            self.app.logger.info("Initializing Time Manager...")
            self.app.time_manager = get_time_manager(self.app.mt5, self.app.config)

            # Initialize Schedule Manager
            self.app.logger.info("Initializing Schedule Manager...")
            self.app.schedule_manager = ScheduleManager(config_path="config/symbol_schedules.json")
            self.app.schedule_manager.log_schedule_status()

            # Initialize Clock Synchronizer first (needed for logger timestamps)
            self.app.logger.info("Initializing Clock Synchronizer...")
            self.app.clock_sync = ClockSynchronizer(self.app.mt5)
            self.app.clock_sync.start_sync_thread()

            # Force immediate sync to get MT5 time for log filename
            self.app.logger.info("Performing initial clock sync for log filename...")
            sync_result = self.app.clock_sync.force_sync()
            if sync_result.get('mt5_time'):
                self.app.logger.info(f"MT5 time synchronized: {sync_result['mt5_time']}")
            else:
                self.app.logger.warning("MT5 time sync failed, using local time for log filename")

            # Reconfigure logger to use ClockSynchronizer for server time
            self.app.logger.info("Switching to MT5 server time for trading logs...")

            # Create a separate trading logger that uses MT5 time
            # Keep the startup logger for system initialization (local time initially, then MT5 time)
            trading_logger = setup_logger(
                'FX-Ai-Trading',  # Separate logger for trading operations
                self.app.config.get(
                    'logging',
                    {}).get(
                        'level',
                        'INFO'),
                'D:/FX-Ai-Data/logs/FX-Trading',  # Different base filename for trading logs
                rotation_type=self.app.config.get(
                    'logging',
                    {}).get(
                        'rotation_type',
                        'size'),
                mt5_connector=self.app.mt5,
                clock_sync=self.app.clock_sync)

            # Store the trading logger separately
            # Startup logger (self.app.logger) keeps local time logs
            # Trading logger (self.app.trading_logger) uses MT5 time
            self.app.trading_logger = trading_logger

            # Log the switch with both loggers
            self.app.logger.info("Startup logger (local time) will continue for system messages")
            self.app.trading_logger.info("Trading logger (MT5 server time) now active for trading operations")

            # Update all existing loggers to use ClockSynchronizer for server time
            from utils.logger import MT5TimeFormatter
            root_logger = logging.getLogger()
            mt5_formatter = MT5TimeFormatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                mt5_connector=self.app.mt5,
                clock_sync=self.app.clock_sync
            )

            # Update root logger handlers
            for handler in root_logger.handlers:
                handler.setFormatter(mt5_formatter)

            # Update ALL existing child loggers to ensure they use the new formatter
            for logger_name in list(logging.Logger.manager.loggerDict.keys()):
                child_logger = logging.getLogger(logger_name)
                if hasattr(child_logger, 'handlers'):
                    for handler in child_logger.handlers:
                        handler.setFormatter(mt5_formatter)

            # Risk Manager
            self.app.logger.info("Initializing risk manager...")
            db_path = os.path.join('data', 'performance_history.db')
            self.app.risk_manager = RiskManager(self.app.config, db_path=db_path, mt5_connector=self.app.mt5)

            # Reset daily trade tracking at startup to fix false "already traded" status
            self.app.logger.info("Resetting daily trade tracking at startup...")
            self._reset_daily_trade_tracking()

            # Market Data Manager
            self.app.logger.info("Initializing market data manager...")
            self.app.market_data_manager = MarketDataManager(self.app.mt5, self.app.config)

            # Fundamental Data Collector
            self.app.logger.info("Initializing fundamental data collector...")
            self.app.fundamental_collector = FundamentalDataCollector(self.app.config)

            # Technical Analyzer
            self.app.logger.info("Initializing technical analyzer...")
            self.app.technical_analyzer = TechnicalAnalyzer(self.app.config)

            # Sentiment Analyzer
            self.app.logger.info("Initializing sentiment analyzer...")
            self.app.sentiment_analyzer = SentimentAnalyzer(self.app.config)

            # ML Predictor
            self.app.logger.info("Initializing ML Predictor...")
            self.app.ml_predictor = MLPredictor(self.app.config)

            # Parameter Manager
            self.app.logger.info("Initializing Parameter Manager...")
            self.app.param_manager = DynamicParameterManager(self.app.config)

            # Adaptive Learning Manager
            self.app.logger.info("Initializing Adaptive Learning Manager...")
            self.app.adaptive_learning = AdaptiveLearningManager(
                self.app.config,
                ml_predictor=self.app.ml_predictor,
                risk_manager=self.app.risk_manager,
                mt5_connector=self.app.mt5
            )

            # Initialize learning database
            self.app.logger.info("Initializing learning database...")
            self.app.adaptive_learning.init_database()

            # Market Regime Detector
            self.app.logger.info("Initializing Market Regime Detector...")
            self.app.market_regime_detector = MarketRegimeDetector(self.app.config)

            # Reinforcement Learning Agent
            self.app.logger.info("Initializing Reinforcement Learning Agent...")
            self.app.reinforcement_agent = RLAgent(self.app.config)

            # Advanced Risk Metrics
            self.app.logger.info("Initializing Advanced Risk Metrics...")
            self.app.advanced_risk_metrics = AdvancedRiskMetrics(self.app.config)

            # Correlation Manager
            self.app.logger.info("Initializing Correlation Manager...")
            self.app.correlation_manager = CorrelationManager(self.app.config)

            # Trading Engine
            self.app.logger.info("Initializing trading engine...")
            self.app.trading_engine = TradingEngine(
                self.app.mt5,
                self.app.risk_manager,
                self.app.technical_analyzer,
                self.app.sentiment_analyzer,
                self.app.fundamental_collector,
                self.app.ml_predictor,
                self.app.adaptive_learning  # Pass adaptive learning as last positional arg
            )

            # Initialize database
            self.app.logger.info("Initializing learning database...")
            db = LearningDatabase()
            db.init_database()

            # Position Sync Manager
            self.app.logger.info("Initializing position synchronization manager...")
            self.app.position_sync_manager = PositionSyncManager(
                db_path=db.db_path,  # Use the same path as the learning database
                logger=self.app.logger
            )

            # Sync with existing MT5 positions on startup
            self.app.logger.info("Syncing with existing MT5 positions...")
            self.app.trading_engine.sync_with_mt5_positions()

            self.app.logger.info(" All components initialized successfully")
            return True

        except Exception as e:
            self.app.logger.error(f"Failed to initialize components: {e}")
            return False

    def _reset_daily_trade_tracking(self):
        """Reset daily trade tracking to fix false 'already traded today' status"""
        try:
            # Clear in-memory tracking
            if hasattr(self.app.risk_manager, 'daily_trades_per_symbol'):
                self.app.risk_manager.daily_trades_per_symbol.clear()
                self.app.logger.info("Cleared in-memory daily trade tracking")

            # Get today's date using the same method as risk manager
            today, _, success = self.app.risk_manager._get_mt5_server_date_reliable()
            if not success:
                # Fallback to local time if MT5 not available during init
                from datetime import datetime
                today = datetime.now().date()
                self.app.logger.info("Using local time for daily trade reset (MT5 not available)")

            import sqlite3
            db_path = os.path.join('data', 'performance_history.db')

            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Check if ANY actual trades exist today
                cursor.execute("SELECT COUNT(*) FROM trades WHERE DATE(timestamp) = ?", (today,))
                actual_trades_today = cursor.fetchone()[0]

                if actual_trades_today == 0:
                    # No real trades today, so clear daily_trades table
                    cursor.execute("DELETE FROM daily_trade_counts WHERE trade_date = ?", (today,))
                    cleared = cursor.rowcount
                    conn.commit()
                    self.app.logger.info(f"Cleared {cleared} false 'already traded' flags from database for date {today}")
                else:
                    self.app.logger.info(f"{actual_trades_today} real trades exist today ({today}) - keeping tracking")

                conn.close()
            else:
                self.app.logger.info("Database not found - using in-memory tracking only")

            self.app.logger.info("Daily trade tracking reset complete")

        except Exception as e:
            self.app.logger.error(f"Error resetting daily trade tracking: {e}")