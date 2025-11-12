"""
FX-Ai: Advanced Forex Trading System for MT5
Main application with Adaptive Learning Integration
Version 3.0
"""

# Add project root to path
from utils.logger import setup_logger
from utils.time_manager import get_time_manager
from utils.config_loader import ConfigLoader
from ai.adaptive_learning_manager import AdaptiveLearningManager
from ai.market_regime_detector import MarketRegimeDetector
from ai.reinforcement_learning_agent import RLAgent
from ai.advanced_risk_metrics import AdvancedRiskMetrics
from ai.correlation_manager import CorrelationManager
from ai.ml_predictor import MLPredictor
from live_trading.dynamic_parameter_manager import DynamicParameterManager
from analysis.sentiment_analyzer import SentimentAnalyzer
from analysis.technical_analyzer import TechnicalAnalyzer
from analysis.fundamental_analyzer import (
    FundamentalAnalyzer as FundamentalDataCollector)
from data.market_data_manager import MarketDataManager
from core.clock_sync import ClockSynchronizer
from core.risk_manager import RiskManager
from core.trading_engine import TradingEngine
from core.mt5_connector import MT5Connector
import MetaTrader5 as mt5
import time as time_module
import threading
import json
import signal
from datetime import datetime, time, timezone
from typing import Dict
import asyncio
import pandas as pd
import sys
import os
import logging
import traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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

    async def validate_configuration(self):
        """Validate critical configuration before trading starts"""
        from utils.exceptions import ConfigurationError, MissingParametersError

        self.logger.info("Validating system configuration...")

        # 1. Check optimal_parameters.json exists
        params_path = "models/parameter_optimization/optimal_parameters.json"
        if not os.path.exists(params_path):
            raise MissingParametersError(
                f"Optimized parameters file not found: {params_path}\n"
                "Please run optimization first: python backtest/optimize_fast_3year.py"
            )

        # 2. Verify all trading symbols have optimized parameters
        with open(params_path, 'r') as f:
            optimized_params = json.load(f)

        trading_symbols = self.config.get('trading', {}).get('symbols', [])
        missing_symbols = [s for s in trading_symbols if s not in optimized_params]

        if missing_symbols:
            self.logger.warning(
                f"No optimized parameters for: {', '.join(missing_symbols)}\n"
                f"These symbols will use fallback parameters from config"
            )
        else:
            self.logger.info(f"[OK] All {len(trading_symbols)} symbols have optimized parameters")

        # 3. Validate MT5 connection settings
        mt5_config = self.config.get('mt5', {})
        if not mt5_config.get('login') or not mt5_config.get('password'):
            raise ConfigurationError(
                "MT5 credentials missing. Please check:\n"
                "1. .env file exists with MT5_LOGIN and MT5_PASSWORD\n"
                "2. config/config.json has MT5 settings"
            )

        # 4. Validate risk settings
        risk_per_trade = self.config.get('trading', {}).get('risk_per_trade', 0)
        if risk_per_trade <= 0:
            raise ConfigurationError(f"Invalid risk_per_trade: {risk_per_trade}. Must be > 0")

        max_positions = self.config.get('trading', {}).get('max_positions', 0)
        if max_positions <= 0 or max_positions > 100:
            raise ConfigurationError(f"Invalid max_positions: {max_positions}. Must be 1-100")

        # 5. Validate model directory exists
        model_dir = "models"
        if not os.path.exists(model_dir):
            self.logger.warning(f"Model directory not found: {model_dir}")
            os.makedirs(model_dir, exist_ok=True)
            self.logger.info(f"Created model directory: {model_dir}")

        self.logger.info("[OK] Configuration validation passed")

    async def initialize_components(self):
        """Initialize all trading components"""
        try:
            self.logger.info("Initializing components...")

            # Validate configuration first
            await self.validate_configuration()

            # 1. MT5 Connection
            self.logger.info("Initializing MT5 connection...")
            self.mt5 = MT5Connector(self.config)
            if not self.mt5.connect():
                raise Exception("Failed to connect to MT5")

            # Initialize Time Manager
            self.logger.info("Initializing Time Manager...")
            self.time_manager = get_time_manager(self.mt5)

            # Initialize Clock Synchronizer first (needed for logger timestamps)
            self.logger.info("Initializing Clock Synchronizer...")
            self.clock_sync = ClockSynchronizer(self.mt5)
            self.clock_sync.start_sync_thread()

            # Reconfigure logger to use ClockSynchronizer for server time
            self.logger.info("Reconfiguring logger to use MT5 server time...")
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
                        'size'),
                mt5_connector=self.mt5,
                clock_sync=self.clock_sync)

            # Update all existing loggers to use ClockSynchronizer for server time
            from utils.logger import MT5TimeFormatter
            root_logger = logging.getLogger()
            mt5_formatter = MT5TimeFormatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                mt5_connector=self.mt5,
                clock_sync=self.clock_sync
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
            self.logger.info("Initializing risk manager...")
            db_path = os.path.join('data', 'performance_history.db')
            self.risk_manager = RiskManager(self.config, db_path=db_path, mt5_connector=self.mt5)

            # Market Data Manager
            self.logger.info("Initializing market data manager...")
            self.market_data = MarketDataManager(self.mt5, self.config)

            # Fundamental Data Collector
            self.logger.info("Initializing fundamental data collector...")
            self.fundamental_collector = FundamentalDataCollector(self.config)

            # Technical Analyzer
            self.logger.info("Initializing technical analyzer...")
            self.technical_analyzer = TechnicalAnalyzer(self.config)

            # Sentiment Analyzer
            self.logger.info("Initializing sentiment analyzer...")
            self.sentiment_analyzer = SentimentAnalyzer(self.config)

            # ML Predictor
            self.logger.info("Initializing ML Predictor...")
            self.ml_predictor = MLPredictor(self.config)

            # Parameter Manager
            self.logger.info("Initializing Parameter Manager...")
            self.param_manager = DynamicParameterManager(self.config)

            # Adaptive Learning Manager
            self.logger.info("Initializing Adaptive Learning Manager...")
            self.adaptive_learning = AdaptiveLearningManager(
                self.config,
                ml_predictor=self.ml_predictor,
                risk_manager=self.risk_manager,
                mt5_connector=self.mt5
            )

            # Market Regime Detector
            self.logger.info("Initializing Market Regime Detector...")
            self.market_regime_detector = MarketRegimeDetector(self.config)

            # Reinforcement Learning Agent
            self.logger.info("Initializing Reinforcement Learning Agent...")
            self.reinforcement_agent = RLAgent(self.config)

            # Advanced Risk Metrics
            self.logger.info("Initializing Advanced Risk Metrics...")
            self.advanced_risk_metrics = AdvancedRiskMetrics(self.config)

            # Correlation Manager
            self.logger.info("Initializing Correlation Manager...")
            self.correlation_manager = CorrelationManager(self.config)

            # Trading Engine
            self.logger.info("Initializing trading engine...")
            self.trading_engine = TradingEngine(
                self.mt5,
                self.risk_manager,
                self.technical_analyzer,
                self.sentiment_analyzer,
                self.fundamental_collector,
                self.ml_predictor,
                self.adaptive_learning  # Pass adaptive learning as last positional arg
            )

            # Sync with existing MT5 positions on startup
            self.logger.info("Syncing with existing MT5 positions...")
            self.trading_engine.sync_with_mt5_positions()

            self.logger.info(" All components initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            return False

    async def trading_loop(self):
        """Main trading loop with adaptive learning integration"""
        self.logger.info("Starting adaptive trading loop...")
        self.running = True

        # Start fundamental monitor background task
        self.fundamental_monitor_task = asyncio.create_task(self.fundamental_monitor_loop())

        loop_count = 0

        while self.running:
            try:
                loop_count += 1

                # 1. Get market data
                symbols = self.config.get('trading', {}).get('symbols', [])
                market_data_dict = {}
                bars_dict = {}

                for symbol in symbols:
                    start_time = time_module.time()
                    try:
                        # Minimal delay to prevent MT5 API overload
                        time_module.sleep(0.05)  # Reduced from 0.5 to 0.05 seconds
                        data = self.market_data.get_latest_data(symbol)  # type: ignore
                        response_time = time_module.time() - start_time

                        if data is not None:
                            market_data_dict[symbol] = data
                            # Get bars for technical analysis - INCREASED to 100 bars for ML model training
                            # Only fetch H1 and H4 (reduced from 5 timeframes to 2)
                            bars_m1 = None  # Disabled to reduce API load
                            bars_m5 = None  # Disabled to reduce API load
                            bars_h1 = self.market_data.get_bars(  # type: ignore
                                symbol, mt5.TIMEFRAME_H1, 100)  # Increased to 100 for ML training (needs 80+)
                            bars_h4 = self.market_data.get_bars(  # type: ignore
                                symbol, mt5.TIMEFRAME_H4, 50)  # Keep at 50, only used for trend confirmation
                            bars_d1 = None  # Disabled to reduce API load

                            if bars_h1 is not None:  # Use H1 as primary
                                bars_dict[symbol] = {
                                    'M1': bars_m1,
                                    'M5': bars_m5,
                                    'H1': bars_h1,
                                    'H4': bars_h4,
                                    'D1': bars_d1
                                }
                    except Exception as e:
                        response_time = time_module.time() - start_time
                        self.logger.warning(f"Failed to get market data for {symbol}: {e}")

                if not market_data_dict:
                    await asyncio.sleep(10)
                    continue

                # 2. Collect fundamental data (with caching)
                start_time = time_module.time()
                try:
                    fundamental_data = self.fundamental_collector.collect()  # type: ignore
                    response_time = time_module.time() - start_time
                except Exception as e:
                    response_time = time_module.time() - start_time
                    self.logger.warning(f"Failed to collect fundamental data: {e}")
                    fundamental_data = {}

                # 3. Generate trading signals with adaptive weights
                signals = []

                # Log status - show what we're doing
                active_symbols = len(market_data_dict)

                self.logger.info(f"Loop {loop_count}: Checking {active_symbols} active symbols (0 halted)")
                self.logger.info(f"DEBUG: market_data_dict has {len(market_data_dict)} symbols, bars_dict has {len(bars_dict)} symbols")

                # Show every 5 loops for more visibility
                if loop_count % 5 == 0:
                    self.logger.info(f"System status: Monitoring {len(symbols)} pairs, {active_symbols} tradeable")

                symbol_counter = 0
                for symbol, market_data in market_data_dict.items():
                    symbol_counter += 1
                    symbol_start_time = time_module.time()
                    self.logger.info(f">>> Starting iteration for symbol {symbol_counter}/{len(market_data_dict)}: {symbol}")
                    self.logger.info(f"{symbol}: About to access current_data from market_data_dict...")
                    self.logger.debug(f"Processing symbol {symbol_counter}/{len(market_data_dict)}: {symbol}")
                    # Get adaptive parameters if learning is enabled
                    adaptive_start = time_module.time()
                    if self.adaptive_learning:
                        signal_weights = self.adaptive_learning.get_current_weights()
                        adaptive_params = self.adaptive_learning.get_adaptive_parameters()
                        # Force fixed risk
                        adaptive_params['risk_multiplier'] = 1.0

                        # Apply regime adaptation if available
                        if self.market_regime_detector and hasattr(self.adaptive_learning, 'get_regime_adapted_parameters'):
                            self.logger.debug(f"{symbol}: Applying regime adaptation...")

                            # Get current market regime
                            market_regime = 'ranging'  # default
                            if bars_dict.get(symbol, {}).get('H1') is not None:
                                historical_data = bars_dict[symbol]['H1']
                                if len(historical_data) >= 30:
                                    regime_analysis = self.market_regime_detector.analyze_regime(symbol, historical_data)
                                    market_regime = regime_analysis.primary_regime.value

                            # Get portfolio risk metrics for risk-aware tuning
                            portfolio_metrics = {}
                            if self.advanced_risk_metrics:
                                try:
                                    portfolio_metrics = self.advanced_risk_metrics.assess_portfolio_risk()
                                except Exception as e:
                                    self.logger.debug(f"Could not get portfolio metrics: {e}")

                            # Apply regime and risk-aware adaptation
                            adaptive_params = self.adaptive_learning.get_regime_adapted_parameters(
                                symbol, market_regime, portfolio_metrics)
                            self.logger.debug(f"{symbol}: Regime adaptation applied")
                    else:
                        # Adjusted weights for testing (more weight to
                        # technical)
                        signal_weights = {
                            'technical_score': 0.40,
                            'ml_prediction': 0.20,
                            'sentiment_score': 0.20,
                            'fundamental_score': 0.10,
                            'support_resistance': 0.10
                        }
                        adaptive_params = self.get_default_parameters()

                    # Technical analysis with adaptive parameters
                    bars = bars_dict.get(symbol)
                    if bars is None or not isinstance(
                            bars, dict) or 'H1' not in bars:
                        continue

                    self.logger.debug(f"{symbol}: Starting technical analysis...")
                    tech_start = time_module.time()
                    technical_signals = await self.technical_analyzer.analyze(  # type: ignore
                        symbol,
                        bars
                    )
                    self.logger.info(f"{symbol}: Technical analysis took {time_module.time() - tech_start:.2f}s")
                    self.logger.debug(f"{symbol}: Technical analysis complete")
                    technical_score = technical_signals.get(
                        'overall_score', 0.5)

                    # ML prediction
                    ml_start = time_module.time()
                    if self.ml_predictor is not None:
                        # ML predictor available
                        start_time = time_module.time()
                        try:
                            self.logger.info(f"{symbol}: About to call ml_predictor.predict()...")
                            ml_prediction = await self.ml_predictor.predict(symbol, bars, technical_signals)
                            response_time = time_module.time() - start_time
                            self.logger.info(f"{symbol}: ML prediction returned successfully in {response_time:.2f}s")
                        except Exception as e:
                            response_time = time_module.time() - start_time

                            # GRACEFUL DEGRADATION: Fall back to technical analysis only
                            self.logger.warning(
                                f"{symbol}: ML prediction failed, falling back to technical analysis only: {e}")

                            # Calculate fallback from technical score
                            technical_signal_strength = (technical_score - 0.5) * 2  # Convert 0-1 to -1 to 1
                            fallback_confidence = min(abs(technical_signal_strength), 0.75)  # Cap at 0.75 to indicate fallback

                            ml_prediction = {
                                'direction': 'BUY' if technical_signal_strength > 0 else 'SELL',
                                'probability': technical_score,  # Use original 0-1 scale
                                'confidence': fallback_confidence,
                                'source': 'technical_fallback',  # Mark as fallback
                                'signal_strength': technical_signal_strength
                            }

                            self.logger.info(
                                f"{symbol}: Fallback prediction - {ml_prediction['direction']} "
                                f"(confidence: {fallback_confidence:.2f}, signal: {technical_signal_strength:.2f})")
                    else:
                        # ML predictor disabled - use technical analysis only
                        # Use technical_score as signal strength (ranges -1 to 1, centered at 0.5)
                        technical_signal_strength = (technical_score - 0.5) * 2  # Convert 0-1 to -1 to 1
                        fallback_confidence = min(abs(technical_signal_strength), 0.75)
                        ml_prediction = {
                            'direction': 'BUY' if technical_signal_strength > 0 else 'SELL',
                            'probability': technical_score,  # Use original 0-1 scale
                            'confidence': fallback_confidence,
                            'source': 'technical_only',
                            'signal_strength': technical_signal_strength
                        }

                    # Sentiment analysis
                    start_time = time_module.time()
                    try:
                        sentiment_result = await self.sentiment_analyzer.analyze_sentiment(  # type: ignore
                            symbol)
                        response_time = time_module.time() - start_time
                        sentiment_score = sentiment_result.get(
                            'overall_score', 0.5)
                    except Exception as e:
                        response_time = time_module.time() - start_time
                        sentiment_score = 0.5  # Default neutral score
                        self.logger.warning(f"Sentiment analysis failed for {symbol}: {e}")

                    # Calculate weighted signal with adaptive weights
                    signal_strength = (
                        signal_weights.get(
                            'technical_score',
                            0.25)
                        * technical_score
                        + signal_weights.get(
                            'ml_prediction',
                            0.30)
                        * ml_prediction.get(
                            'probability',
                            0)
                        + signal_weights.get(
                            'sentiment_score',
                            0.20)
                        * sentiment_score
                        + signal_weights.get(
                            'fundamental_score',
                            0.15)
                        * fundamental_data.get(  # type: ignore
                            symbol,
                            {}).get(  # type: ignore
                                'score',
                                0.5)
                        + signal_weights.get(
                            'sr_score',
                            0.10)
                        * market_data.get(
                            'sr_score',
                            0.5))

                    # Log all signal strengths for debugging
                    self.logger.info(
                        f"{symbol} signal: strength={signal_strength:.3f} "
                        f"(Tech:{technical_score:.3f}, "
                        f"ML:{ml_prediction.get('probability', 0):.3f}, "
                        f"Sent:{sentiment_score:.3f})"
                    )

                    # Apply symbol-specific minimum thresholds to increase trade frequency
                    # Lower thresholds for forex pairs to allow more trading opportunities
                    metal_symbols = ['XAU', 'XAG', 'GOLD']
                    if any(metal in symbol for metal in metal_symbols):
                        # Keep higher threshold for metals due to higher volatility/risk
                        min_threshold = 0.5
                    else:
                        # Lower threshold for forex pairs to increase trade frequency
                        min_threshold = 0.3  # Reduced from 0.4 to allow more forex trading

                    self.logger.info(
                        f"{symbol}: threshold={min_threshold:.3f}, "
                        f"strength={signal_strength:.3f}")

                    if signal_strength > min_threshold:
                        # Get current market price for entry
                        current_data = market_data_dict.get(symbol, {})
                        entry_price = current_data.get('ask') if ml_prediction.get(
                            'direction') == 1 else current_data.get('bid', 0)

                        # Skip if no valid price data
                        if entry_price <= 0:
                            self.logger.info(
                                f"{symbol}: Skipping - no valid entry price")
                            continue

                        self.logger.info(
                            f"{symbol}: Processing signal - entry_price={entry_price}, "
                            f"direction={ml_prediction.get('direction')}")

                        self.logger.info(f"{symbol}: Starting validation checks...")

                        # Check market regime and adapt strategy
                        if self.market_regime_detector:
                            # Reuse H1 data already fetched above (no additional API call)
                            historical_data = bars_dict.get(symbol, {}).get('H1') if symbol in bars_dict else None

                            if historical_data is not None and len(historical_data) >= 30:
                                regime_analysis = self.market_regime_detector.analyze_regime(
                                    symbol, historical_data)

                                self.logger.debug(
                                    f"{symbol}: Market regime - {regime_analysis.primary_regime.value} "
                                    f"(confidence: {regime_analysis.confidence:.2f}, "
                                    f"ADX: {regime_analysis.adx_value:.1f})")

                                # Store regime info for adaptive learning
                                if self.adaptive_learning:
                                    self.adaptive_learning.update_regime_data(
                                        symbol, regime_analysis)
                            else:
                                self.logger.debug(
                                    f"{symbol}: Insufficient data for regime analysis")

                        self.logger.info(f"{symbol}: Validation checks passed, calculating SL/TP...")

                        # Calculate stop loss using ATR (more sophisticated
                        # than fixed percentage)
                        atr_value = technical_signals.get(
                            'atr', {}).get('value', 0)

                        # For metals, use optimized sl_pips directly from parameter manager
                        if 'XAU' in symbol or 'XAG' in symbol or 'GOLD' in symbol:
                            optimal_params = self.param_manager.get_optimal_parameters(symbol, 'H1')  # type: ignore

                            # Different defaults for Gold vs Silver
                            if 'XAG' in symbol:
                                sl_pips = optimal_params.get('sl_pips', 300)  # Default 300 pips for XAGUSD
                                pip_size = 0.001  # Silver: 1 pip = 0.001 (1 point)
                            else:  # XAU/GOLD
                                sl_pips = optimal_params.get('sl_pips', 200)  # Default 200 pips for XAUUSD
                                pip_size = 0.10  # Gold: 1 pip = 0.10 (10 points)

                            stop_loss_distance = sl_pips * pip_size
                            sl_atr_multiplier = stop_loss_distance / atr_value if atr_value > 0 else 0  # For logging only
                            self.logger.info(
                                f"{symbol} using optimized metal stop loss: {sl_pips} pips (pip_size={pip_size}) = {stop_loss_distance:.5f}")
                        elif atr_value > 0:
                            self.logger.debug(
                                f"{symbol}: ATR available ({atr_value:.5f}) - "
                                f"proceeding with signal")
                            # Use adaptive ATR multiplier for stop loss distance (forex only)
                            base_multiplier = adaptive_params.get(
                                'stop_loss_atr_multiplier', 3.0)
                            sl_atr_multiplier = base_multiplier
                            stop_loss_distance = atr_value * sl_atr_multiplier
                            self.logger.debug(
                                f"{symbol} ATR-based stop loss: "
                                f"ATR={atr_value:.5f}, "
                                f"multiplier={sl_atr_multiplier:.1f}, "
                                f"distance={stop_loss_distance:.5f}")
                        else:
                            # Fallback to 2% if ATR not available
                            stop_loss_distance = entry_price * 0.02
                            self.logger.debug(
                                f"{symbol} fallback stop loss: 2% of entry = "
                                f"{stop_loss_distance:.5f}")

                        # ===== SENTIMENT & FUNDAMENTAL ADJUSTMENTS FOR SL =====
                        # Adjust stop loss based on sentiment and fundamental analysis
                        sl_adjustment_factor = 1.0  # Default: no adjustment

                        # Sentiment-based SL adjustment
                        if sentiment_score > 0.7:
                            # Strong positive sentiment: Tighten SL (ride winners, protect gains)
                            sl_adjustment_factor *= 0.90  # Reduce SL distance by 10%
                            self.logger.info(f"{symbol}: Strong positive sentiment ({sentiment_score:.2f}) - tightening SL by 10%")
                        elif sentiment_score < 0.3:
                            # Strong negative sentiment: Widen SL (give room, avoid premature stops)
                            sl_adjustment_factor *= 1.15  # Increase SL distance by 15%
                            self.logger.info(f"{symbol}: Strong negative sentiment ({sentiment_score:.2f}) - widening SL by 15%")

                        # Fundamental-based SL adjustment
                        fundamental_score = fundamental_data.get(symbol, {}).get('score', 0.5)  # type: ignore
                        high_impact_news = fundamental_data.get(symbol, {}).get('high_impact_news_soon', False)  # type: ignore

                        if high_impact_news:
                            # High-impact news coming: Widen SL to avoid being stopped out by volatility
                            sl_adjustment_factor *= 1.20  # Increase SL distance by 20%
                            self.logger.info(f"{symbol}: High-impact news detected - widening SL by 20%")
                        elif fundamental_score < 0.3:
                            # Weak fundamentals: Widen SL slightly (less confident trade)
                            sl_adjustment_factor *= 1.10  # Increase SL distance by 10%
                            self.logger.info(f"{symbol}: Weak fundamentals ({fundamental_score:.2f}) - widening SL by 10%")

                        # Apply adjustment
                        stop_loss_distance = stop_loss_distance * sl_adjustment_factor

                        if sl_adjustment_factor != 1.0:
                            self.logger.info(f"{symbol}: Total SL adjustment factor: {sl_adjustment_factor:.2f}x (distance: {stop_loss_distance:.5f})")

                        # No minimum SL restriction - use optimized parameters directly
                        # The stop_loss_distance is already calculated using optimized values from
                        # DynamicParameterManager or adaptive learning

                        if ml_prediction.get('direction') == 1:  # BUY
                            stop_loss = entry_price - stop_loss_distance
                        else:  # SELL
                            stop_loss = entry_price + stop_loss_distance

                        # Calculate take profit using adaptive ATR multiplier or optimized parameters
                        take_profit = None
                        tp_atr_multiplier = 0  # Initialize to avoid reference errors
                        if atr_value > 0:
                            # Use optimized TP for metals, ATR-based for forex
                            if 'XAU' in symbol or 'XAG' in symbol or 'GOLD' in symbol:
                                # Get optimized take profit for metals from parameter manager
                                optimal_params = self.param_manager.get_optimal_parameters(symbol, 'H1')  # type: ignore

                                # Different defaults and pip sizes for Gold vs Silver
                                if 'XAG' in symbol:
                                    tp_pips = optimal_params.get('tp_pips', 1200)  # Default 1200 pips for XAGUSD
                                    pip_size = 0.001  # Silver: 1 pip = 0.001 (1 point)
                                else:  # XAU/GOLD
                                    tp_pips = optimal_params.get('tp_pips', 3300)  # Default 3300 pips for XAUUSD
                                    pip_size = 0.10  # Gold: 1 pip = 0.10 (10 points)

                                # Convert TP pips to distance
                                take_profit_distance = tp_pips * pip_size
                                tp_atr_multiplier = take_profit_distance / atr_value if atr_value > 0 else 0  # Calculate for logging
                                self.logger.info(f"{symbol} using optimized metal take profit: {tp_pips} pips (pip_size={pip_size}) = {take_profit_distance:.5f}")
                            else:
                                # Use ATR-based TP for forex
                                base_tp_multiplier = adaptive_params.get(
                                    'take_profit_atr_multiplier', 9.0)
                                tp_atr_multiplier = base_tp_multiplier
                                take_profit_distance = atr_value * tp_atr_multiplier

                            # ===== DYNAMIC SL/TP ADJUSTMENTS BASED ON ANALYZER OUTPUT =====
                        # Get base SL/TP values before analyzer adjustments
                        base_sl_pips = 0
                        base_tp_pips = 0

                        if 'XAU' in symbol or 'XAG' in symbol or 'GOLD' in symbol:
                            # For metals, get base values from optimized parameters
                            optimal_params = self.param_manager.get_optimal_parameters(symbol, 'H1')  # type: ignore
                            if 'XAG' in symbol:
                                base_sl_pips = optimal_params.get('sl_pips', 300)
                                base_tp_pips = optimal_params.get('tp_pips', 1200)
                            else:  # XAU/GOLD
                                base_sl_pips = optimal_params.get('sl_pips', 200)
                                base_tp_pips = optimal_params.get('tp_pips', 3300)
                        else:
                            # For forex, calculate ATR-based base values
                            if atr_value > 0:
                                sl_atr_multiplier = adaptive_params.get('stop_loss_atr_multiplier', 3.0)
                                tp_atr_multiplier = adaptive_params.get('take_profit_atr_multiplier', 9.0)
                                base_sl_distance = atr_value * sl_atr_multiplier
                                base_tp_distance = atr_value * tp_atr_multiplier

                                # Convert distances back to pips for analyzer input
                                pip_size = 0.01 if symbol.endswith('JPY') else 0.0001
                                base_sl_pips = base_sl_distance / pip_size
                                base_tp_pips = base_tp_distance / pip_size

                        # Get analyzer-based SL/TP adjustments
                        analyzer_adjustments = {}

                        # Technical analyzer adjustments (if available)
                        if hasattr(self.technical_analyzer, 'get_sl_tp_adjustments'):
                            try:
                                tech_adjustments = self.technical_analyzer.get_sl_tp_adjustments(  # type: ignore
                                    symbol, base_sl_pips, base_tp_pips, technical_signals)
                                analyzer_adjustments['technical'] = tech_adjustments
                            except Exception as e:
                                self.logger.debug(f"Technical analyzer SL/TP adjustment failed: {e}")

                        # Sentiment analyzer adjustments (if available)
                        if hasattr(self.sentiment_analyzer, 'get_sl_tp_adjustments'):
                            try:
                                sentiment_adjustments = self.sentiment_analyzer.get_sl_tp_adjustments(  # type: ignore
                                    symbol, base_sl_pips, base_tp_pips, sentiment_result)
                                analyzer_adjustments['sentiment'] = sentiment_adjustments
                            except Exception as e:
                                self.logger.debug(f"Sentiment analyzer SL/TP adjustment failed: {e}")

                        # Fundamental analyzer adjustments (NEW!)
                        if hasattr(self.fundamental_collector, 'get_sl_tp_adjustments'):
                            try:
                                fundamental_adjustments = self.fundamental_collector.get_sl_tp_adjustments(  # type: ignore
                                    symbol, base_sl_pips, base_tp_pips)
                                analyzer_adjustments['fundamental'] = fundamental_adjustments
                            except Exception as e:
                                self.logger.debug(f"Fundamental analyzer SL/TP adjustment failed: {e}")

                        # Apply analyzer adjustments (prioritize fundamental > sentiment > technical)
                        final_sl_pips = base_sl_pips
                        final_tp_pips = base_tp_pips
                        adjustment_reasons = []

                        for analyzer_name in ['fundamental', 'sentiment', 'technical']:
                            if analyzer_name in analyzer_adjustments:
                                adj = analyzer_adjustments[analyzer_name]
                                if adj.get('sl_pips', base_sl_pips) != base_sl_pips:
                                    final_sl_pips = adj['sl_pips']
                                    adjustment_reasons.append(f"{analyzer_name}: SL {base_sl_pips:.0f} -> {final_sl_pips:.0f} ({adj.get('reason', 'unknown')})")
                                if adj.get('tp_pips', base_tp_pips) != base_tp_pips:
                                    final_tp_pips = adj['tp_pips']
                                    adjustment_reasons.append(f"{analyzer_name}: TP {base_tp_pips:.0f} -> {final_tp_pips:.0f} ({adj.get('reason', 'unknown')})")

                        # Log adjustments
                        if adjustment_reasons:
                            self.logger.info(f"{symbol}: Dynamic SL/TP adjustments applied:")
                            for reason in adjustment_reasons:
                                self.logger.info(f"  {reason}")

                        # Convert final pips back to distance
                        if 'XAU' in symbol or 'XAG' in symbol or 'GOLD' in symbol:
                            # For metals, use the adjusted pips directly
                            if 'XAG' in symbol:
                                pip_size = 0.001
                            else:  # XAU/GOLD
                                pip_size = 0.10

                            stop_loss_distance = final_sl_pips * pip_size
                            take_profit_distance = final_tp_pips * pip_size
                        else:
                            # For forex, convert pips back to distance
                            pip_size = 0.01 if symbol.endswith('JPY') else 0.0001
                            stop_loss_distance = final_sl_pips * pip_size
                            take_profit_distance = final_tp_pips * pip_size

                            if ml_prediction.get('direction') == 1:  # BUY
                                take_profit = entry_price + take_profit_distance
                        # Calculate take profit based on final adjusted distances
                        if ml_prediction.get('direction') == 1:  # BUY
                            take_profit = entry_price + take_profit_distance
                        else:  # SELL
                            take_profit = entry_price - take_profit_distance

                        self.logger.debug(
                            f"{symbol} final take profit: distance={take_profit_distance:.5f}, level={take_profit:.5f}")

                        # ===== LEGACY FALLBACK (should not be reached with new logic) =====
                        # Keep this for safety but it should not execute with the new analyzer-based logic above
                        if take_profit is None:
                            self.logger.warning(f"{symbol}: Take profit not set by analyzer logic, using fallback")
                            # Fallback take profit: 6% of entry price (3x the 2% stop loss fallback)
                            take_profit_distance = entry_price * 0.06

                            if ml_prediction.get('direction') == 1:  # BUY
                                take_profit = entry_price + take_profit_distance
                            else:  # SELL
                                take_profit = entry_price - take_profit_distance
                            self.logger.debug(
                                f"{symbol} fallback take profit: 6% of entry = {take_profit_distance:.5f}, level={take_profit:.5f}")

                        # Format take profit for logging
                        tp_display = f"{
                            take_profit:.5f}" if take_profit is not None else "None"

                        # Determine initial action based on ML prediction
                        action = 'BUY' if ml_prediction.get('direction') == 1 else 'SELL'

                        # Validate risk-reward ratio before proceeding
                        if stop_loss is not None and take_profit is not None:
                            risk_distance = abs(stop_loss - entry_price)
                            reward_distance = abs(take_profit - entry_price)
                            actual_ratio = reward_distance / risk_distance if risk_distance > 0 else 0

                            min_ratio = self.config.get('trading', {}).get('min_risk_reward_ratio', 2.0)
                            if actual_ratio < min_ratio:
                                self.logger.info(
                                    f"{symbol} {action} rejected: insufficient " f"reward ratio {
                                        actual_ratio:.2f}:1 " f"(required: {min_ratio}:1)")
                                continue  # Skip this trade

                            self.logger.info(
                                f"{symbol} {action} validated: "
                                f"{actual_ratio:.1f}:1 risk-reward ratio")

                        self.logger.info(f"{symbol}: Risk-reward check passed, checking RL agent...")

                        # ===== REINFORCEMENT LEARNING DECISION =====
                        rl_decision = 'hold'  # Default action
                        if self.reinforcement_agent and self.reinforcement_agent.enabled:
                            try:
                                # Get regime data (may be None if regime detection wasn't performed)
                                regime_adx = 25  # Default
                                regime_type = 'ranging'  # Default
                                if self.market_regime_detector:
                                    historical_data = bars_dict.get(symbol, {}).get('H1')
                                    if historical_data is not None and len(historical_data) >= 30:
                                        regime_analysis = self.market_regime_detector.analyze_regime(symbol, historical_data)
                                        regime_adx = regime_analysis.adx_value
                                        regime_type = regime_analysis.primary_regime.value

                                # Prepare state for RL agent
                                current_state = {
                                    'rsi': technical_signals.get('rsi', {}).get('value', 50),
                                    'adx': regime_adx,
                                    'volatility_ratio': technical_signals.get('atr', {}).get('value', 0) / entry_price if entry_price > 0 else 0,
                                    'trend_strength': technical_signals.get('adx', {}).get('value', 25),
                                    'market_regime': regime_type,
                                    'signal_strength': signal_strength,
                                    'position_status': 0  # No position currently
                                }

                                # Get RL action recommendation
                                rl_decision = self.reinforcement_agent.choose_action_from_dict(current_state)
                                self.logger.info(f"{symbol}: RL decision - {rl_decision}")
                                # TEMPORARILY DISABLED: Allow all signals during European session for testing
                                # Only proceed if RL recommends buy/sell (not hold or close)
                                # if rl_decision not in ['buy', 'sell']:
                                #     self.logger.info(f"{symbol}: Skipping - RL recommends {rl_decision}")
                                #     continue

                                # Override direction if RL disagrees with ML prediction
                                if rl_decision == 'buy' and ml_prediction.get('direction') == -1:
                                    self.logger.debug(f"{symbol}: RL overriding ML sell to buy")
                                    action = 'BUY'
                                    entry_price = current_data.get('ask', entry_price)
                                    stop_loss = entry_price - stop_loss_distance
                                    take_profit = entry_price + take_profit_distance
                                elif rl_decision == 'sell' and ml_prediction.get('direction') == 1:
                                    self.logger.debug(f"{symbol}: RL overriding ML buy to sell")
                                    action = 'SELL'
                                    entry_price = current_data.get('bid', entry_price)
                                    stop_loss = entry_price + stop_loss_distance
                                    take_profit = entry_price - take_profit_distance

                            except Exception as e:
                                self.logger.warning(f"{symbol}: RL decision failed, proceeding with ML: {e}")
                                rl_decision = 'buy' if ml_prediction.get('direction') == 1 else 'sell'

                        signal = {
                            'symbol': symbol,
                            'action': action,
                            'strength': signal_strength,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'technical_signals': technical_signals,
                        }
                        signal['ml_score'] = ml_prediction.get('probability', 0)
                        signal['technical_score'] = technical_score
                        signal['sentiment_score'] = sentiment_score
                        signal['rl_decision'] = rl_decision
                        signal['timestamp'] = self.get_current_mt5_time()
                        signal['adaptive_params'] = adaptive_params

                        # Add signal to list (freeze fixed by disabling AdaptiveLearningManager database)
                        signals.append(signal)

                        # SKIP this log too - it accesses signal dict which might freeze
                        # self.logger.info(
                        #     f"Signal generated for {symbol}: {signal['action']} "
                        #     f"(strength: {signal_strength:.3f}, threshold: {min_threshold:.3f}) "
                        #     f"Entry: {entry_price:.5f}, SL: {stop_loss:.5f} ({sl_atr_multiplier:.1f}x ATR), "
                        #     f"TP: {tp_display} ({tp_atr_multiplier:.1f}x ATR)"
                        # )

                    # Log symbol processing completion for debugging (end of symbol loop iteration)
                    # This runs for ALL symbols, whether they generate signals or not
                    symbol_elapsed = time_module.time() - symbol_start_time
                    self.logger.info(f"Completed processing {symbol} ({symbol_counter}/{len(market_data_dict)}) in {symbol_elapsed:.2f}s")

                # 4. Execute trades with risk management
                if signals:
                    self.logger.info(f"Generated {len(signals)} trading signal(s), evaluating for execution...")
                else:
                    self.logger.info("No trading signals generated this cycle")

                # DEBUG: Check signals before day trading filter
                self.logger.info(f"DEBUG: Signals before day trading check: {len(signals)}")

                # Check if trading is allowed (before 22:30 MT5 server time)
                # TEMPORARILY DISABLED FOR TESTING
                # if self.config.get('trading', {}).get('day_trading_only', True):
                #     is_allowed, reason = self.time_manager.is_trading_allowed()
                #     self.logger.info(f"DEBUG: Day trading check - allowed: {is_allowed}, reason: {reason}")
                #     if not is_allowed:
                #         self.logger.info(f"Trading halted: {reason}")
                #         signals = []  # Clear all signals to prevent trading

                # DEBUG: Check signals after day trading filter
                self.logger.info(f"DEBUG: Signals after day trading check: {len(signals)}")

                for signal in signals:
                    # DEBUG CHECKPOINT 2: Starting signal processing
                    self.logger.info(f"DEBUG CHECKPOINT 2: Processing signal for {signal['symbol']}")

                    # Validate signal has required trading parameters
                    self.logger.debug(
                        f"Processing signal for {signal['symbol']}: "
                        f"action={signal['action']}, "
                        f"strength={signal.get('strength', 0):.3f}, "
                        f"entry={signal.get('entry_price', 'None')}, "
                        f"stop={signal.get('stop_loss', 'None')}, "
                        f"tp={signal.get('take_profit', 'None')}"
                    )
                    if not signal.get(
                            'entry_price') or not signal.get('stop_loss'):
                        self.logger.warning(
                            f"Skipping signal for {signal['symbol']} - "
                            f"missing entry/stop loss data "
                            f"(entry: {signal.get('entry_price', 'None')}, "
                            f"stop: {signal.get('stop_loss', 'None')})"
                        )
                        # DEBUG CHECKPOINT 2A: Signal rejected - missing data
                        self.logger.info(f"DEBUG CHECKPOINT 2A: Signal for {signal['symbol']} rejected - missing entry/stop data")
                        continue

                    # DEBUG CHECKPOINT 3: Signal passed basic validation
                    self.logger.info(f"DEBUG CHECKPOINT 3: Signal for {signal['symbol']} passed basic validation")

                    # Update risk metrics before checking limits
                    # Note: New RiskManager doesn't have update_metrics method

                    # ===== DYNAMIC CORRELATION ANALYSIS =====
                    if self.correlation_manager:
                        try:
                            # Get currently open positions for correlation check
                            open_positions = []
                            if self.mt5:
                                positions = self.mt5.get_positions()
                                open_positions = [pos['symbol'] for pos in positions] if positions else []

                            # Update correlation manager with current positions
                            self.correlation_manager.update_open_positions(open_positions)

                            # Get market conditions for sophisticated correlation analysis
                            market_conditions = {}
                            if hasattr(self, 'market_regime_detector') and self.market_regime_detector:
                                try:
                                    # Get current market conditions
                                    volatility = self.market_regime_detector.get_current_volatility()
                                    trend_strength = self.market_regime_detector.get_trend_strength()
                                    market_conditions = {
                                        'volatility': volatility,
                                        'trend_strength': trend_strength
                                    }
                                except Exception:
                                    pass

                            # Check if correlated trading is allowed
                            correlation_allowed = True
                            correlation_reason = "Correlation analysis passed"

                            if open_positions:
                                # For each open position, check if we should allow correlated trading
                                for open_symbol in open_positions:
                                    allowed, reason = self.correlation_manager.should_allow_correlated_trading(
                                        signal['symbol'], open_symbol, market_conditions
                                    )
                                    if not allowed:
                                        correlation_allowed = False
                                        correlation_reason = reason
                                        break

                            if not correlation_allowed:
                                self.logger.warning(
                                    f"Correlation analysis blocked {signal['symbol']}: {correlation_reason}")
                                continue

                            # Get correlation-adjusted position size
                            base_position_size = signal.get('position_size', 0.01)
                            adjusted_size = self.correlation_manager.get_correlation_adjusted_size(
                                signal['symbol'], base_position_size, open_positions)

                            if adjusted_size != base_position_size:
                                signal['position_size'] = adjusted_size
                                self.logger.info(
                                    f"Adjusted position size for {signal['symbol']} from {base_position_size} to {adjusted_size} due to correlation")

                        except Exception as e:
                            self.logger.error(f"Correlation analysis failed for {signal['symbol']}: {e}")

                    # Check risk limits with adaptive multiplier
                    risk_check = self.risk_manager.can_trade(signal['symbol'])  # type: ignore
                    self.logger.info(
                        f"Risk check for {
                            signal['symbol']} {
                            signal['action']}: {risk_check}")
                    if risk_check:
                        # DEBUG CHECKPOINT 4: Risk check passed
                        self.logger.info(f"DEBUG CHECKPOINT 4: Risk check PASSED for {signal['symbol']}")
                    else:
                        # DEBUG CHECKPOINT 4A: Risk check failed
                        self.logger.info(f"DEBUG CHECKPOINT 4A: Risk check FAILED for {signal['symbol']}")
                        continue

                    # ===== ADVANCED RISK ASSESSMENT =====
                    # Using cached H1 data already fetched in bars_dict (no additional API calls)
                    if self.advanced_risk_metrics and loop_count % 10 == 0:  # Run every 10th loop to reduce overhead
                        try:
                            # Use already-fetched H1 data from bars_dict (100 bars available)
                            symbol_hist_data = bars_dict.get(signal['symbol'], {}).get('H1')

                            if symbol_hist_data is not None and len(symbol_hist_data) >= 30:
                                # Calculate quick risk metrics using available data
                                symbol_df = pd.DataFrame(symbol_hist_data)
                                symbol_df['returns'] = symbol_df['close'].pct_change()

                                # Calculate VaR and CVaR for this symbol
                                var_95 = self.advanced_risk_metrics.calculate_var(symbol_df['returns'].dropna(), confidence=0.95)
                                cvar_95 = self.advanced_risk_metrics.calculate_cvar(symbol_df['returns'].dropna(), confidence=0.95)

                                # Calculate Sharpe ratio
                                sharpe = self.advanced_risk_metrics.calculate_sharpe_ratio(symbol_df['returns'].dropna())

                                # Get account balance
                                account_info = mt5.account_info()  # type: ignore
                                account_balance = account_info.balance if account_info else 10000

                                # Calculate position risk as percentage of account
                                trade_risk = abs(signal['entry_price'] - signal['stop_loss']) * signal.get('position_size', 0.01)
                                risk_pct = (trade_risk / account_balance) * 100

                                # Risk warning if position risk > 2% of account
                                if risk_pct > 2.0:
                                    self.logger.warning(
                                        f"{signal['symbol']}: High risk trade - {risk_pct:.2f}% of account "
                                        f"(VaR95: {var_95 * 100:.2f}%, CVaR95: {cvar_95 * 100:.2f}%, Sharpe: {sharpe:.2f})"
                                    )

                                # Log advanced metrics every 50th loop
                                if loop_count % 50 == 0:
                                    self.logger.info(
                                        f"{signal['symbol']} Risk Metrics: "
                                        f"VaR(95%): {var_95 * 100:.2f}%, "
                                        f"CVaR(95%): {cvar_95 * 100:.2f}%, "
                                        f"Sharpe: {sharpe:.2f}, "
                                        f"Position Risk: {risk_pct:.2f}%"
                                    )

                        except Exception as e:
                            self.logger.error(f"Advanced risk assessment failed for {signal['symbol']}: {e}")

                    # Check free margin percentage
                    account_info = mt5.account_info()  # type: ignore
                    if account_info and account_info.margin_free < 0.20 * account_info.equity:
                        self.logger.warning(
                            f"Free margin ${
                                account_info.margin_free:.2f} is " f"less than 20% of equity ${
                                account_info.equity:.2f}, " f"skipping trade for {
                                signal['symbol']}")
                        continue

                    # Calculate stop loss pips from signal
                    entry_price = signal['entry_price']
                    stop_loss = signal['stop_loss']
                    if signal['action'] == 'BUY':
                        stop_loss_distance = entry_price - stop_loss
                    else:  # SELL
                        stop_loss_distance = stop_loss - entry_price

                    # Convert to pips using proper pip size for each symbol
                    symbol_info = mt5.symbol_info(signal['symbol'])  # type: ignore
                    if symbol_info:
                        point = symbol_info.point
                        digits = symbol_info.digits
                        # Determine pip size correctly for each symbol type
                        metal_symbols = ['XAU', 'XAG', 'GOLD']
                        if any(metal in signal['symbol'] for metal in metal_symbols):
                            # Metals: 1 pip = 10 points (0.1 for 2-digit
                            # symbols)
                            pip_size = point * 10
                        elif digits == 3 or digits == 5:
                            pip_size = point * 10  # 3/5 digit brokers: 1 pip = 10 points
                        else:
                            pip_size = point  # 2/4 digit brokers: 1 pip = 1 point
                        stop_loss_pips = stop_loss_distance / pip_size
                    else:
                        # Fallback to old logic
                        if 'JPY' in signal['symbol']:
                            stop_loss_pips = stop_loss_distance / 0.01
                        else:
                            stop_loss_pips = stop_loss_distance / 0.0001

                    # Use optimized stop loss directly without minimum enforcement
                    # stop_loss_pips already calculated from optimized parameters

                    # Check if minimum lot size would exceed risk limit for
                    # volatile symbols
                    min_lot_size = self.config.get(
                        'trading', {}).get('min_lot_size', 0.01)
                    metal_symbols = ['XAU', 'XAG', 'GOLD']
                    if any(metal in signal['symbol'] for metal in metal_symbols):
                        # Calculate risk with minimum lot size
                        min_lot_risk = self.risk_manager.calculate_risk_for_lot_size(  # type: ignore
                            signal['symbol'], min_lot_size, stop_loss_pips)

                        max_risk_limit = self.config.get(
                            'trading', {}).get('risk_per_trade', 50.0)
                        if min_lot_risk > max_risk_limit * 1.05:  # Allow 5% tolerance
                            self.logger.warning(
                                f"Skipping {signal['symbol']} - even minimum lot size "
                                f"{min_lot_size} would risk ${min_lot_risk:.2f} "
                                f"(limit: ${max_risk_limit:.2f})"
                            )
                            continue

                    # Calculate position size with risk amount adjusted for
                    # metals
                    base_risk_amount = 50  # Fixed $50 risk per trade for forex
                    metal_symbols = ['XAU', 'XAG', 'GOLD']
                    if any(metal in signal['symbol'] for metal in metal_symbols):
                        risk_amount = base_risk_amount  # Keep $50 risk for metals
                        self.logger.debug(
                            f"Metal detected: {
                                signal['symbol']}, using $50 risk")
                    else:
                        risk_amount = base_risk_amount
                        self.logger.debug(
                            f"Forex detected: {
                                signal['symbol']}, using $50 risk")

                    position_size = self.risk_manager.calculate_position_size(  # type: ignore
                        signal['symbol'], stop_loss_pips, risk_amount)

                    # Debug logging
                    self.logger.info(
                        f"Position sizing for {signal['symbol']}: "
                        f"risk_amount=${risk_amount}, "
                        f"stop_pips={stop_loss_pips}, "
                        f"lot_size={position_size:.4f}")

                    # Execute trade with stop loss
                    start_time = time_module.time()
                    # DEBUG CHECKPOINT 5: About to execute trade
                    self.logger.info(f"DEBUG CHECKPOINT 5: About to execute trade for {signal['symbol']} {signal['action']} lot_size={position_size:.4f}")
                    try:
                        self.logger.info(f"DEBUG: About to call place_order for {signal['symbol']}")
                        trade_result = await self.trading_engine.place_order(  # type: ignore
                                signal['symbol'],
                                signal['action'].lower(),
                                position_size,
                                stop_loss=signal.get('stop_loss'),
                                # Now using calculated take profit
                                take_profit=signal.get('take_profit'),
                                comment=f"FX-Ai {
                                    signal['action']} {
                                    signal['strength']:.3f}"
                            )
                        response_time = time_module.time() - start_time
                        success = trade_result.get('success', False)
                        # DEBUG CHECKPOINT 6: Trade execution result
                        self.logger.info(f"DEBUG CHECKPOINT 6: Trade execution result for {signal['symbol']}: success={success}, error={trade_result.get('error', 'None')}")
                    except Exception as e:
                        response_time = time_module.time() - start_time
                        trade_result = {'success': False, 'error': str(e)}
                        self.logger.error(f"Trade execution failed for {signal['symbol']}: {e}")
                        # DEBUG CHECKPOINT 6A: Trade execution exception
                        self.logger.info(f"DEBUG CHECKPOINT 6A: Trade execution EXCEPTION for {signal['symbol']}: {e}")

                    if trade_result.get('success', False):
                        self.session_stats['total_trades'] += 1

                        # Record trade for daily limit tracking (ONE TRADE PER SYMBOL PER DAY)
                        self.risk_manager.record_trade(signal['symbol'])  # type: ignore
                        self.logger.info(f"{signal['symbol']}: Trade executed and recorded - no more trades allowed today")

                        # Log active positions after successful trade
                        await self._log_active_positions()

                        # ===== REINFORCEMENT LEARNING EXPERIENCE =====
                        if self.reinforcement_agent and self.reinforcement_agent.enabled:
                            try:
                                # Get regime data (use defaults if not available)
                                regime_adx = 25
                                regime_type = 'ranging'
                                if self.market_regime_detector:
                                    historical_data = bars_dict.get(signal['symbol'], {}).get('H1')
                                    if historical_data is not None and len(historical_data) >= 30:
                                        regime_analysis = self.market_regime_detector.analyze_regime(signal['symbol'], historical_data)
                                        regime_adx = regime_analysis.adx_value
                                        regime_type = regime_analysis.primary_regime.value

                                # Record initial state and action for RL learning
                                initial_state = {
                                    'rsi': signal['technical_signals'].get('rsi', {}).get('value', 50),
                                    'adx': regime_adx,
                                    'volatility_ratio': signal['technical_signals'].get('atr', {}).get('value', 0) / signal['entry_price'] if signal['entry_price'] > 0 else 0,
                                    'trend_strength': signal['technical_signals'].get('adx', {}).get('value', 25),
                                    'market_regime': regime_type,
                                    'signal_strength': signal['strength'],
                                    'position_status': 1 if signal['action'] == 'BUY' else -1  # Position opened
                                }
                                action_taken = signal.get('rl_decision', 'buy' if signal['action'] == 'BUY' else 'sell')

                                # Store experience for later learning when trade closes
                                rl_experience = {
                                    'ticket': trade_result.get('order', 0),
                                    'initial_state': initial_state,
                                    'action': action_taken,
                                    'entry_price': signal['entry_price'],
                                    'symbol': signal['symbol'],
                                    'timestamp': self.get_current_mt5_time(),
                                    'closure_reason': 'natural_exit'  # Default, will be updated if forced closure
                                }

                                # Store in RL agent for learning when trade closes
                                if hasattr(self.reinforcement_agent, 'pending_experiences'):
                                    self.reinforcement_agent.pending_experiences[trade_result.get('order', 0)] = rl_experience

                                self.logger.debug(f"RL experience recorded for ticket {trade_result.get('order', 0)}")

                            except Exception as e:
                                self.logger.warning(f"Failed to record RL experience: {e}")

                        # Record trade for learning
                        if self.adaptive_learning:
                            trade_data = {
                                'timestamp': self.get_current_mt5_time(),
                                'symbol': signal['symbol'],
                                'direction': signal['action'],
                                'entry_price': trade_result.get(
                                    'price',
                                    0),
                                'signal_strength': signal['strength'],
                                'ml_score': signal['ml_score'],
                                'technical_score': signal['technical_score'],
                                'sentiment_score': signal['sentiment_score'],
                                'model_version': self.ml_predictor.get_model_version(signal['symbol']) if self.ml_predictor else 'disabled'}

                            # Start monitoring thread for this trade
                            threading.Thread(
                                target=self.monitor_trade,
                                args=(trade_result.get(
                                    'order', 0), trade_data),
                                daemon=True
                            ).start()

                # 5. Monitor and update positions
                self.logger.debug(f"About to start position monitoring for {len(symbols)} symbols")
                self.logger.info(f"Starting position monitoring for {len(symbols)} symbols")
                for symbol in symbols:
                    self.logger.debug(f"Monitoring positions for {symbol}")
                    self.logger.info(f"Monitoring positions for {symbol}")
                    await self.trading_engine.manage_positions(symbol, self.time_manager, self.adaptive_learning)  # type: ignore

                    # NEW: Monitor correlation changes for open positions
                    if self.correlation_manager and hasattr(self.correlation_manager, 'monitor_correlation_changes'):
                        try:
                            correlation_action = self.correlation_manager.monitor_correlation_changes(symbol)
                            if correlation_action.get('action') != 'none':
                                self.logger.info(f"Correlation action for {symbol}: {correlation_action}")

                                # Execute correlation-based actions
                                await self._handle_correlation_action(symbol, correlation_action)
                        except Exception as e:
                            self.logger.error(f"Error monitoring correlations for {symbol}: {e}")

                # 6. Check for model retraining trigger (every hour)
                # Every hour (assuming 10s loop)
                if self.adaptive_learning and loop_count % 360 == 0:
                    self.logger.info("Triggering periodic model evaluation...")

                    # Get performance summary
                    performance = self.adaptive_learning.get_performance_summary()

                    self.logger.info(
                        f"Performance Update - Win Rate: {
                            performance['performance_metrics'].get(
                                'overall_win_rate',
                                0):.2%}, " f"Total Trades: {
                            performance['total_trades']}")

                    # Log adapted weights
                    self.logger.info(
                        f"Current Signal Weights: {
                            json.dumps(
                                performance['signal_weights'],
                                indent=2)}")
                    self.logger.info(
                        f"Adaptive Parameters: " f"{
                            json.dumps(
                                performance['adaptive_params'],
                                indent=2)}")

                    # Log comprehensive performance summary
                    self._log_performance_summary()

                    self.session_stats['models_retrained'] += 1

                    # Save RL model periodically
                    if self.reinforcement_agent:
                        try:
                            self.reinforcement_agent.save_model()
                            self.logger.info("RL model saved during periodic evaluation")
                        except Exception as e:
                            self.logger.error(f"Error saving RL model during evaluation: {e}")

                    # Auto-learn from previous day logs (every 6 hours)
                    if self.adaptive_learning and loop_count % (360 * 6) == 0:
                        try:
                            self.adaptive_learning.auto_learn_from_previous_day_logs()
                        except Exception as e:
                            self.logger.error(f"Error in auto log learning: {e}")

                    # Check learning thread health every hour (360 * 6 * 10 seconds = 1 hour)
                    if self.adaptive_learning and loop_count % (360 * 6) == 0:
                        try:
                            thread_status = self.adaptive_learning.get_thread_status()
                            self.logger.info(f"Learning thread status: {thread_status}")

                            if not thread_status.get('thread_alive', False):
                                self.logger.warning("Learning thread is not alive - attempting restart")
                                self.adaptive_learning.restart_learning_thread()
                                # Check again after restart
                                time_module.sleep(0.1)
                                new_status = self.adaptive_learning.get_thread_status()
                                if new_status.get('thread_alive', False):
                                    self.logger.info("Learning thread successfully restarted")
                                else:
                                    self.logger.error("Failed to restart learning thread")
                        except Exception as e:
                            self.logger.error(f"Error checking learning thread health: {e}")

                # 7. Check for time-based closure (always check after 22:00)
                try:
                    await self.check_time_based_closure()
                except Exception as e:
                    self.logger.error(f"Error in time-based closure: {e}")

                # Sleep before next iteration
                await asyncio.sleep(10)

            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(30)

    def monitor_trade(self, ticket: int, trade_data: dict):
        """Monitor trade outcome for learning with time-based exits"""
        try:
            # Get symbol-specific optimal holding times
            symbol = trade_data['symbol']
            if self.adaptive_learning:
                symbol_params = self.adaptive_learning.get_symbol_optimal_holding_time(
                    symbol)
                optimal_holding_hours = symbol_params['optimal_holding_hours']
                max_holding_minutes = symbol_params['max_holding_minutes']
                confidence_score = symbol_params['confidence_score']
            else:
                # Fallback to global parameters
                adaptive_params = self.adaptive_learning.get_adaptive_parameters(
                ) if self.adaptive_learning else {}
                optimal_holding_hours = adaptive_params.get(
                    'optimal_holding_hours', 4.0)
                max_holding_minutes = adaptive_params.get(
                    'max_holding_minutes', 480)
                confidence_score = 0.0

            optimal_holding_minutes = optimal_holding_hours * 60

            # Log symbol-specific parameters
            self.logger.info(
                f"Monitoring {symbol} with optimal holding: "
                f"{optimal_holding_hours:.1f}h "
                f"(max: {max_holding_minutes}min, confidence: "
                f"{confidence_score:.2f})")

            # Wait for trade to complete
            while True:
                time_module.sleep(30)  # Check every 30 seconds

                position = self.trading_engine.get_position_by_ticket(ticket)  # type: ignore

                if position is None:  # Trade closed
                    # Get trade history
                    history = self.trading_engine.get_trade_history(ticket)  # type: ignore

                    if history:
                        # Calculate profit
                        exit_price = history['exit_price']
                        entry_price = trade_data['entry_price']

                        if trade_data['direction'] == 'BUY':
                            profit_pips = (exit_price - entry_price) * 10000
                        else:
                            profit_pips = (entry_price - exit_price) * 10000

                        # Simplified calculation
                        profit_pct = (profit_pips / 100)

                        # Update trade data
                        trade_data['exit_price'] = exit_price
                        trade_data['profit'] = history.get('profit', 0)
                        trade_data['profit_pct'] = profit_pct
                        trade_data['duration_minutes'] = (
                            self.get_current_mt5_time() - trade_data['timestamp']).seconds // 60
                        trade_data['volume'] = history.get('volume', 0)

                        # Record trade result for cooldown management
                        actual_profit = history.get('profit', 0)
                        self.risk_manager.record_trade_result(  # type: ignore
                            trade_data['symbol'], actual_profit)

                        # Record for learning
                        if self.adaptive_learning:
                            self.adaptive_learning.record_trade(trade_data)

                        # ===== ANALYZER ACCURACY EVALUATION =====
                        if self.reinforcement_agent and hasattr(self.reinforcement_agent, 'record_trade_with_analyzer_evaluation'):
                            try:
                                # Prepare trade outcome data for analyzer evaluation
                                trade_outcome = {
                                    'symbol': trade_data['symbol'],
                                    'profit_pct': profit_pct,
                                    'entry_signals': {
                                        'technical_score': trade_data.get('technical_score', 0.5),
                                        'fundamental_score': 0.5,  # Placeholder - could be enhanced
                                        'sentiment_score': trade_data.get('sentiment_score', 0.5),
                                        'signal_strength': trade_data.get('signal_strength', 0.5)
                                    }
                                }

                                # Record trade with analyzer evaluation
                                self.reinforcement_agent.record_trade_with_analyzer_evaluation(
                                    trade_outcome, self.adaptive_learning)

                                self.logger.debug(f"Analyzer evaluation completed for {trade_data['symbol']}")

                            except Exception as e:
                                self.logger.warning(f"Failed to evaluate analyzer accuracy: {e}")

                        # ===== REINFORCEMENT LEARNING UPDATE =====
                        if self.reinforcement_agent and self.reinforcement_agent.enabled:
                            try:
                                # Get the stored experience for this ticket
                                ticket = trade_data.get('ticket', 0)
                                experience = self.reinforcement_agent.pending_experiences.get(ticket)
                                if experience is None:
                                    self.logger.debug(f"No RL experience found for ticket {ticket}")
                                    continue

                                # Determine closure reason from trade data
                                closure_reason = trade_data.get('closure_reason', 'natural_exit')

                                reward = self.reinforcement_agent.calculate_reward(
                                    experience['entry_price'],
                                    exit_price,
                                    trade_data['direction'],
                                    trade_data['duration_minutes'],
                                    closure_reason=closure_reason
                                )

                                # Create next state (position closed)
                                next_state = experience['initial_state'].copy()
                                next_state['position_status'] = 0  # Position closed
                                next_state['closure_reason'] = closure_reason  # Add closure reason to state

                                # Learn from this experience
                                self.reinforcement_agent.update_q_table(
                                    experience['initial_state'],
                                    experience['action'],
                                    reward,
                                    next_state
                                )

                                # Save RL model periodically (every 10 trades)
                                self.session_stats['rl_models_saved'] += 1
                                if self.session_stats['rl_models_saved'] % 10 == 0:
                                    try:
                                        self.reinforcement_agent.save_model()
                                        self.logger.debug(f"RL model saved after {self.session_stats['rl_models_saved']} trades")
                                    except Exception as e:
                                        self.logger.error(f"Error saving RL model after trade: {e}")

                                # Remove from pending experiences
                                del self.reinforcement_agent.pending_experiences[ticket]

                                self.logger.debug(f"RL updated: action={experience['action']}, reward={reward:.4f}, reason={closure_reason}")

                            except Exception as e:
                                self.logger.warning(f"Failed to update RL agent: {e}")

                        # Update session stats
                        if profit_pct > 0:
                            self.session_stats['winning_trades'] += 1
                        else:
                            self.session_stats['losing_trades'] += 1

                        self.session_stats['total_profit'] += profit_pct

                        # Log comprehensive trade outcome
                        self._log_trade_outcome(trade_data, history, closure_reason)

                    break
                else:
                    # Position still open - check time-based exit conditions
                    holding_minutes = (
                        self.get_current_mt5_time() - trade_data['timestamp']).seconds // 60

                    # Check for immediate closure after 22:00 MT5 time
                    current_time = self.time_manager.get_current_time()
                    current_time_only = current_time.time()
                    immediate_close_time = self.time_manager.MT5_IMMEDIATE_CLOSE_TIME
                    
                    if current_time_only >= immediate_close_time:
                        self.logger.info(
                            f"Immediately closing {trade_data['symbol']} position - after {immediate_close_time.strftime('%H:%M')} MT5 time")
                        try:
                            # Close position at market
                            close_result = asyncio.run(self.trading_engine.close_position(position))  # type: ignore
                            if close_result:
                                self.logger.info(f"Successfully closed position for immediate time-based exit")

                                # Store closure reason in pending experience for RL learning
                                if hasattr(self.reinforcement_agent, 'pending_experiences') and ticket in self.reinforcement_agent.pending_experiences:
                                    self.reinforcement_agent.pending_experiences[ticket]['closure_reason'] = f"immediate_close_after_{immediate_close_time.strftime('%H:%M')}"

                            else:
                                self.logger.warning(f"Failed to close position for immediate time-based exit")
                        except Exception as e:
                            self.logger.error(f"Error closing position for immediate time-based exit: {e}")
                        break

                    # Check for time-based closure (market close buffer)
                    should_close_time, close_reason = self.time_manager.should_close_positions()
                    if should_close_time:
                        self.logger.info(
                            f"Closing {trade_data['symbol']} position due to time-based exit: {close_reason}")
                        try:
                            # Close position at market
                            close_result = asyncio.run(self.trading_engine.close_position(position))  # type: ignore
                            if close_result:
                                self.logger.info(f"Successfully closed position for time-based exit: {close_reason}")

                                # Store closure reason in pending experience for RL learning
                                if hasattr(self.reinforcement_agent, 'pending_experiences') and ticket in self.reinforcement_agent.pending_experiences:
                                    self.reinforcement_agent.pending_experiences[ticket]['closure_reason'] = close_reason

                            else:
                                self.logger.warning(f"Failed to close position for time-based exit: {close_reason}")
                        except Exception as e:
                            self.logger.error(f"Error closing position for time-based exit: {e}")
                        break

                    # Check maximum holding time
                    if holding_minutes >= max_holding_minutes:
                        self.logger.info(
                            f"Closing {trade_data['symbol']} position due to "
                            f"max holding time ({max_holding_minutes} minutes)")
                        try:
                            # Get position object first
                            position = self.trading_engine.get_position_by_ticket(ticket)  # type: ignore
                            if position:
                                # Close position at market
                                close_result = asyncio.run(self.trading_engine.close_position(position))  # type: ignore
                                if close_result:
                                    self.logger.info(
                                        "Successfully closed position for max time exit")
                                else:
                                    self.logger.warning(
                                        "Failed to close position for max time exit")
                            else:
                                self.logger.warning(f"Position {ticket} not found for closing")
                        except Exception as e:
                            self.logger.error(
                                f"Error closing position for max time: {e}")
                        break

                    # Check optimal holding time with profit
                    elif holding_minutes >= optimal_holding_minutes:
                        # Check if position is in profit
                        current_price = position.get('price_current', 0)
                        entry_price = trade_data['entry_price']

                        if trade_data['direction'] == 'BUY' and current_price > entry_price:
                            profit_pct = (
                                (current_price - entry_price) / entry_price) * 100
                            if profit_pct > 0.1:  # At least 0.1% profit
                                self.logger.info(
                                    f"Closing {trade_data['symbol']} position at "
                                    f"optimal time ({optimal_holding_minutes} min) "
                                    f"with {profit_pct:.2f}% profit")
                                try:
                                    position_obj = self.trading_engine.get_position_by_ticket(ticket)  # type: ignore
                                    if position_obj:
                                        close_result = asyncio.run(self.trading_engine.close_position(position_obj))  # type: ignore
                                        if close_result:
                                            self.logger.info(
                                                "Successfully closed position for optimal time exit")
                                        else:
                                            self.logger.warning(
                                                "Failed to close position for optimal time exit")
                                    else:
                                        self.logger.warning(f"Position {ticket} not found for closing")
                                except Exception as e:
                                    self.logger.error(
                                        f"Error closing position for optimal time: {e}")
                                break
                        elif trade_data['direction'] == 'SELL' and current_price < entry_price:
                            profit_pct = (
                                (entry_price - current_price) / entry_price) *  100
                            if profit_pct > 0.1:  # At least 0.1% profit
                                self.logger.info(
                                    f"Closing {trade_data['symbol']} position at "
                                    f"optimal time ({optimal_holding_minutes} min) "
                                    f"with {profit_pct:.2f}% profit")
                                try:
                                    position_obj = self.trading_engine.get_position_by_ticket(ticket)  # type: ignore
                                    if position_obj:
                                        close_result = asyncio.run(self.trading_engine.close_position(position_obj))  # type: ignore
                                        if close_result:
                                            self.logger.info(
                                                "Successfully closed position for optimal time exit")

                                        else:
                                            self.logger.warning(
                                                "Failed to close position for optimal time exit")
                                    else:
                                        self.logger.warning(f"Position {ticket} not found for closing")
                                except Exception as e:
                                    self.logger.error(
                                        f"Error closing position for optimal time: {e}")
                                break

        except Exception as e:
            self.logger.error(f"Error monitoring trade {ticket}: {e}")

    async def _handle_correlation_action(self, symbol: str, correlation_action: Dict):
        """
        Handle correlation-based trading actions

        Args:
            symbol: Base symbol
            correlation_action: Action recommended by correlation manager
       
        """
        try:
            action_type = correlation_action.get('action')

            if action_type == 'exit_recommended':
                # Consider closing the position due to high correlation
                confidence = correlation_action.get('confidence', 0.5)
                correlation = correlation_action.get('correlation', 0)
                correlated_symbol = correlation_action.get('correlated_symbol', '')

                if confidence > 0.7: # High confidence
                    self.logger.info(f"High-confidence correlation exit: Closing {symbol} due to {correlation:.2f} correlation with {correlated_symbol}")
                    # Get position and close it
                    positions = self.mt5.get_positions() if self.mt5 else []
                    for position in positions:
                        if position.get('symbol') == symbol:
                            close_result = await self.trading_engine.close_position(position)  # type: ignore
                            if close_result:
                                self.logger.info(f"Successfully closed {symbol} based on correlation analysis")
                            break
                else:
                    self.logger.info(f"Correlation exit suggested for {symbol} (confidence: {confidence:.2f}) - monitoring continues")

            elif action_type == 'entry_recommended':
                # Consider opening a correlated position
                confidence = correlation_action.get('confidence', 0.5)
                new_symbol = correlation_action.get('symbol', '')
                base_symbol = correlation_action.get('base_symbol', '')

                if confidence > 0.7 and new_symbol:  # High confidence
                    self.logger.info(f"High-confidence correlation entry: Opening {new_symbol} based on low correlation with {base_symbol}")

                    # Check if we can trade this symbol
                    if self.risk_manager and self.risk_manager.can_trade(new_symbol)[0]:
                        # Generate a small position for the correlated pair
                        signal = {
                            'symbol': new_symbol,
                            'action': 'BUY',  # Default to buy, could be enhanced
                            'entry_price': None,  # Will be set by trading engine
                            'position_size': 0.01,  # Small position
                            'stop_loss': None,
                            'take_profit': None
                        }

                        # Execute the trade
                        trade_result = await self.trading_engine.execute_trade(signal)  # type: ignore
                        if trade_result:
                            self.logger.info(f"Successfully opened {new_symbol} based on correlation analysis")
                        else:
                            self.logger.warning(f"Failed to open {new_symbol} position")
                    else:
                        self.logger.info(f"Cannot open {new_symbol} - risk limits exceeded")
                else:
                    self.logger.info(f"Correlation entry suggested for {new_symbol} (confidence: {confidence:.2f}) - monitoring continues")

            elif action_type == 'exit_consideration':
                # Log correlation increase for monitoring
                correlation = correlation_action.get('correlation', 0)
                correlated_symbol = correlation_action.get('correlated_symbol', '')
                self.logger.info(f"Correlation monitoring: {symbol} correlation with {correlated_symbol} is now {correlation:.2f}")

        except Exception as e:
            self.logger.error(f"Error handling correlation action for {symbol}: {e}")

    async def check_time_based_closure(self):
        """Check if positions should be closed based on time - always check after 22:00"""
        try:
            # Use TimeManager for consistent time handling
            should_close, reason = self.time_manager.should_close_positions()

            if should_close:
                self.logger.info(f"Time-based closure triggered - closing all positions: {reason}")
                self._last_closure_date = self.time_manager._last_closure_date  # Sync with TimeManager

                # Safe async handling for close_all_positions
                if self.trading_engine:
                    if hasattr(self.trading_engine, 'close_all_positions'):
                        try:
                            close_method = self.trading_engine.close_all_positions
                            if asyncio.iscoroutinefunction(close_method):
                                await close_method()
                            else:
                                loop = asyncio.get_event_loop()
                                await loop.run_in_executor(None, close_method)
                        except Exception as e:
                            self.logger.error(f"Error closing positions: {e}")
                    else:
                        self.logger.warning(
                            "close_all_positions method not found on trading_engine")
                else:
                    self.logger.warning("Trading engine not initialized yet")

        except Exception as e:
            self.logger.error(f"Error in time-based closure: {e}")

    def learn_from_logs(self, log_date: str = None):
        """Manually trigger learning from trading logs"""
        try:
            if self.adaptive_learning:
                self.adaptive_learning.learn_from_logs(log_date)
                self.logger.info(f"Completed learning from logs for date: {log_date or 'latest'}")
            else:
                self.logger.warning("Adaptive learning not initialized")
        except Exception as e:
            self.logger.error(f"Error learning from logs: {e}")
        """Get default trading parameters"""
        return {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'min_signal_strength': 0.6,
            'max_correlation': 0.8,
            'risk_multiplier': 1.0,
            'trailing_stop_distance': 20
        }

    def shutdown_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("Shutdown signal received")
        self.shutdown()

    def shutdown(self):
        """Gracefully shutdown the application"""
        self.logger.info("Shutting down FX-Ai...")
        self.running = False

        # Cancel fundamental monitor task
        if self.fundamental_monitor_task and not self.fundamental_monitor_task.done():
            self.logger.info("Cancelling fundamental monitor task...")
            self.fundamental_monitor_task.cancel()
            try:
                # Wait for task to cancel (with timeout)
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we can't wait here, just cancel
                    pass
                else:
                    # If loop is not running, we can wait briefly
                    loop.run_until_complete(asyncio.wait_for(self.fundamental_monitor_task, timeout=5.0))
            except Exception as e:
                self.logger.debug(f"Error waiting for fundamental monitor cancellation: {e}")

        # Print session summary
        self.print_session_summary()

        # Close all positions if configured
        if self.config.get('trading', {}).get('close_on_shutdown', True):
            try:
                if hasattr(self.trading_engine, 'close_all_positions'):  # type: ignore
                    close_method = self.trading_engine.close_all_positions  # type: ignore
                    loop = asyncio.get_event_loop()
                    if asyncio.iscoroutinefunction(close_method):
                        loop.create_task(close_method())
                    else:
                        # Call synchronous close directly in shutdown context
                        try:
                            close_method()  # type: ignore
                        except Exception as e:
                            self.logger.error(f"Error in synchronous close_all_positions: {e}")
                else:
                    self.logger.warning(
                        'No close_all_positions method on trading_engine')
            except Exception as e:
                self.logger.error(
                    f"Error scheduling close_all_positions on shutdown: {e}")

        # Save learning state
        if self.reinforcement_agent:
            try:
                self.reinforcement_agent.save_model()
                self.logger.info("RL model saved successfully")
            except Exception as e:
                self.logger.error(f"Error saving RL model: {e}")

        if self.adaptive_learning:
            try:
                self.adaptive_learning.save_signal_weights()
                self.logger.info("Adaptive learning state saved successfully")
            except Exception as e:
                self.logger.error(f"Error saving adaptive learning state: {e}")

        # Stop clock synchronization - DISABLED
        # if self.clock_sync:
        #     self.clock_sync.stop_sync_thread()

        # Stop system health monitoring
        # Disconnect from MT5
        if self.mt5:
            self.mt5.disconnect()

        self.logger.info("FX-Ai shutdown complete")

    def print_session_summary(self):
        """Print comprehensive trading session summary"""
        duration = self.get_current_mt5_time() - self.session_stats['start_time']

        self.logger.info("=" * 80)
        self.logger.info("FINAL SESSION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Session Duration: {duration}")

        # Log comprehensive performance summary
        self._log_performance_summary()

        # Additional session information
        if self.learning_enabled:
            self.logger.info("LEARNING SYSTEM SUMMARY:")
            self.logger.info(f"Parameters Optimized: {self.session_stats['parameters_optimized']}")

            if self.adaptive_learning:
                performance = self.adaptive_learning.get_performance_summary()
                self.logger.info("Final Adaptive Learning State:")
                self.logger.info(f"Signal Weights: {json.dumps(performance['signal_weights'], indent=2)}")
                self.logger.info(f"Adaptive Parameters: {json.dumps(performance['adaptive_params'], indent=2)}")

        self.logger.info("=" * 80)
        self.logger.info("FX-Ai Session Complete")
        self.logger.info("=" * 80)

        self.logger.info("=" * 60)

    async def run(self):
        """Run the FX-Ai application"""
        self.logger.info("=" * 50)
        self.logger.info("FX-Ai Trading System Starting...")
        self.logger.info("=" * 50)

        # Initialize components
        if not await self.initialize_components():
            self.logger.error("Failed to initialize components")
            return

        # Start trading loop
        try:
            await self.trading_loop()
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
        finally:
            self.shutdown()

    async def _log_active_positions(self):
        """Log all currently active positions with detailed metrics"""
        try:
            # Get all positions from MT5
            positions = mt5.positions_get()
            if positions is None:
                positions = []

            active_fxai_positions = []
            for position in positions:
                if hasattr(position, 'magic') and position.magic == self.magic_number:
                    active_fxai_positions.append(position)

            if active_fxai_positions:
                total_unrealized_pnl = 0.0
                total_risk = 0.0

                self.logger.info(f"ACTIVE POSITIONS ({len(active_fxai_positions)}):")
                for pos in active_fxai_positions:
                    direction = "LONG" if pos.type == mt5.ORDER_TYPE_BUY else "SHORT"
                    commission = getattr(pos, 'commission', 0.0)
                    swap = getattr(pos, 'swap', 0.0)
                    pnl = pos.profit + swap + commission
                    total_unrealized_pnl += pnl

                    # Calculate duration
                    duration_hours = 0
                    if hasattr(pos, 'time') and pos.time > 0:
                        duration_hours = (self.get_current_mt5_time().timestamp() - pos.time) / 3600

                    # Calculate P&L percentage
                    pnl_pct = 0.0
                    if pos.price_open > 0:
                        if direction == "LONG":
                            pnl_pct = ((pos.price_current - pos.price_open) / pos.price_open) * 100
                        else:
                            pnl_pct = ((pos.price_open - pos.price_current) / pos.price_open) * 100

                    # Calculate risk metrics
                    risk_amount = 0.0
                    if pos.sl > 0:
                        pip_size = self._get_pip_size(pos.symbol)
                        if direction == "LONG":
                            risk_pips = (pos.price_open - pos.sl) / pip_size
                        else:
                            risk_pips = (pos.sl - pos.price_open) / pip_size
                        risk_amount = risk_pips * pos.volume * 10  # Approximate dollar risk
                        total_risk += risk_amount

                    sl_display = f"{pos.sl:.5f}" if pos.sl > 0 else "None"
                    tp_display = f"{pos.tp:.5f}" if pos.tp > 0 else "None"

                    self.logger.info(
                        f"  {pos.symbol} {direction} | "
                        f"Size: {pos.volume:.2f} lots | "
                        f"Entry: {pos.price_open:.5f} | "
                        f"Current: {pos.price_current:.5f} | "
                        f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) | "
                        f"Duration: {duration_hours:.1f}h | "
                        f"Risk: ${risk_amount:.2f} | "
                        f"SL: {sl_display} | "
                        f"TP: {tp_display}"
                    )

                # Summary statistics
                avg_pnl_pct = (total_unrealized_pnl / len(active_fxai_positions)) if active_fxai_positions else 0
                self.logger.info(f"PORTFOLIO SUMMARY: Total P&L: ${total_unrealized_pnl:.2f} | "
                               f"Avg P&L: ${avg_pnl_pct:.2f} | Total Risk: ${total_risk:.2f}")

            else:
                self.logger.info("ACTIVE POSITIONS: None")

        except Exception as e:
            self.logger.error(f"Error logging active positions: {e}")

    def _get_pip_size(self, symbol: str) -> float:
        """Get pip size for a symbol"""
        if 'JPY' in symbol:
            return 0.01
        elif 'XAU' in symbol or 'XAG' in symbol:
            return 0.01 if 'XAG' in symbol else 0.1
        else:
            return 0.0001

    def _log_trade_outcome(self, trade_data: dict, history: dict, closure_reason: str = "natural_exit"):
        """Log comprehensive trade outcome with detailed metrics and analysis"""
        try:
            symbol = trade_data['symbol']
            direction = trade_data['direction']
            entry_price = trade_data['entry_price']
            exit_price = trade_data.get('exit_price', history.get('exit_price', 0))
            volume = trade_data.get('volume', history.get('volume', 0))
            duration_minutes = trade_data.get('duration_minutes', 0)
            profit = trade_data.get('profit', history.get('profit', 0))
            profit_pct = trade_data.get('profit_pct', 0)

            # Calculate pip movement
            pip_size = self._get_pip_size(symbol)
            if direction == 'BUY':
                pip_movement = (exit_price - entry_price) / pip_size
            else:
                pip_movement = (entry_price - exit_price) / pip_size

            # Calculate risk metrics
            risk_amount = 0.0
            risk_pct = 0.0
            rr_ratio = 0.0
            if 'sl' in trade_data and trade_data['sl'] > 0:
                if direction == 'BUY':
                    risk_pips = (entry_price - trade_data['sl']) / pip_size
                else:
                    risk_pips = (trade_data['sl'] - entry_price) / pip_size
                risk_amount = abs(risk_pips) * volume * 10  # Approximate dollar risk
                if risk_amount > 0:
                    rr_ratio = abs(profit) / risk_amount

            # Determine trade outcome
            is_win = profit > 0
            outcome = "WIN" if is_win else "LOSS"

            # Calculate performance metrics
            duration_hours = duration_minutes / 60
            profit_per_hour = profit / duration_hours if duration_hours > 0 else 0

            # Log comprehensive trade outcome
            self.logger.info("=" * 80)
            self.logger.info(f"TRADE COMPLETED - {outcome}")
            self.logger.info("=" * 80)

            # Basic trade information
            self.logger.info(f"Symbol: {symbol} | Direction: {direction} | Volume: {volume:.2f} lots")
            self.logger.info(f"Entry Price: {entry_price:.5f} | Exit Price: {exit_price:.5f}")
            self.logger.info(f"Duration: {duration_minutes} minutes ({duration_hours:.1f} hours)")

            # Financial results
            self.logger.info(f"P&L: ${profit:.2f} ({profit_pct:+.2f}%) | Pips: {pip_movement:+.1f}")
            self.logger.info(f"Profit/Hour: ${profit_per_hour:.2f}")

            # Risk metrics
            if risk_amount > 0:
                self.logger.info(f"Risk Amount: ${risk_amount:.2f} | Risk:Reward Ratio: {rr_ratio:.2f}")
            else:
                self.logger.info("Risk Amount: Not set | Risk:Reward Ratio: N/A")

            # Exit analysis
            self.logger.info(f"Exit Reason: {closure_reason}")

            # Signal analysis (if available)
            if 'technical_score' in trade_data:
                self.logger.info(f"Entry Signals - Technical: {trade_data['technical_score']:.2f} | "
                               f"Sentiment: {trade_data.get('sentiment_score', 0.5):.2f} | "
                               f"Signal Strength: {trade_data.get('signal_strength', 0.5):.2f}")

            # Performance analysis
            if is_win:
                self.logger.info("PERFORMANCE: Excellent trade execution!"                if rr_ratio >= 2.0 else
                               "PERFORMANCE: Good trade, consider optimizing risk management"                if rr_ratio >= 1.0 else
                               "PERFORMANCE: Profitable but high risk, review entry criteria")
            else:
                self.logger.info("PERFORMANCE: Loss - analyze entry signals and market conditions"                if abs(rr_ratio) <= 1.0 else
                               "PERFORMANCE: Large loss - review risk management and stop loss placement")

            # Market regime context (if available)
            if hasattr(self, 'market_regime_detector') and self.market_regime_detector:
                try:
                    regime = self.market_regime_detector.get_current_regime(symbol)
                    if regime:
                        self.logger.info(f"Market Regime: {regime['regime']} (Volatility: {regime['volatility']:.2f})")
                except Exception as e:
                    self.logger.debug(f"Could not get market regime: {e}")

            # Adaptive learning insights (if available)
            if self.adaptive_learning:
                try:
                    insights = self.adaptive_learning.get_trade_insights(symbol, is_win)
                    if insights:
                        self.logger.info(f"Learning Insights: {insights}")
                except Exception as e:
                    self.logger.debug(f"Could not get learning insights: {e}")

            self.logger.info("=" * 80)

            # Update session performance tracking
            self._update_performance_metrics(trade_data, is_win, profit, duration_minutes)

        except Exception as e:
            self.logger.error(f"Error logging trade outcome: {e}")

    def _update_performance_metrics(self, trade_data: dict, is_win: bool, profit: float, duration_minutes: int):
        """Update session performance metrics for analysis"""
        try:
            symbol = trade_data['symbol']

            # Initialize symbol metrics if not exists
            if symbol not in self.session_stats['symbol_performance']:
                self.session_stats['symbol_performance'][symbol] = {
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_profit': 0.0,
                    'avg_duration': 0.0,
                    'best_trade': float('-inf'),
                    'worst_trade': float('inf')
                }

            symbol_stats = self.session_stats['symbol_performance'][symbol]
            symbol_stats['total_trades'] += 1
            symbol_stats['wins' if is_win else 'losses'] += 1
            symbol_stats['total_profit'] += profit
            symbol_stats['best_trade'] = max(symbol_stats['best_trade'], profit)
            symbol_stats['worst_trade'] = min(symbol_stats['worst_trade'], profit)

            # Update average duration
            total_duration = symbol_stats['avg_duration'] * (symbol_stats['total_trades'] - 1) + duration_minutes
            symbol_stats['avg_duration'] = total_duration / symbol_stats['total_trades']

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    def _log_performance_summary(self):
        """Log comprehensive performance summary with key metrics"""
        try:
            total_trades = self.session_stats['winning_trades'] + self.session_stats['losing_trades']

            if total_trades == 0:
                self.logger.info("PERFORMANCE SUMMARY: No trades completed yet")
                return

            win_rate = self.session_stats['winning_trades'] / total_trades
            avg_profit = self.session_stats['total_profit'] / total_trades
            total_profit = self.session_stats['total_profit']

            # Calculate additional metrics
            profit_factor = 0.0
            if self.session_stats['losing_trades'] > 0:
                avg_win = self.session_stats.get('avg_win', 0)
                avg_loss = self.session_stats.get('avg_loss', 0)
                if avg_loss != 0:
                    profit_factor = abs(avg_win / avg_loss) if avg_win > 0 else 0

            # Calculate drawdown metrics (simplified)
            max_drawdown = self.session_stats.get('max_drawdown', 0)

            self.logger.info("=" * 80)
            self.logger.info("PERFORMANCE SUMMARY")
            self.logger.info("=" * 80)

            # Overall statistics
            self.logger.info(f"Total Trades: {total_trades}")
            self.logger.info(f"Win Rate: {win_rate:.1%} ({self.session_stats['winning_trades']}W / {self.session_stats['losing_trades']}L)")
            self.logger.info(f"Total P&L: ${total_profit:.2f}")
            self.logger.info(f"Average P&L per Trade: ${avg_profit:.2f}")

            # Advanced metrics
            if profit_factor > 0:
                self.logger.info(f"Profit Factor: {profit_factor:.2f}")
            self.logger.info(f"Max Drawdown: ${max_drawdown:.2f}")

            # Symbol performance
            if self.session_stats['symbol_performance']:
                self.logger.info("SYMBOL PERFORMANCE:")
                for symbol, stats in self.session_stats['symbol_performance'].items():
                    symbol_win_rate = stats['wins'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
                    self.logger.info(f"  {symbol}: {stats['total_trades']} trades | "
                                   f"Win Rate: {symbol_win_rate:.1%} | "
                                   f"P&L: ${stats['total_profit']:.2f} | "
                                   f"Avg Duration: {stats['avg_duration']:.0f}min")

            # System health metrics
            self.logger.info("SYSTEM HEALTH:")
            self.logger.info(f"Models Retrained: {self.session_stats.get('models_retrained', 0)}")
            self.logger.info(f"RL Models Saved: {self.session_stats.get('rl_models_saved', 0)}")
            self.logger.info(f"Emergency Stops: {self.session_stats.get('emergency_stops', 0)}")

            # Performance analysis
            if win_rate >= 0.6:
                performance_rating = "EXCELLENT"
            elif win_rate >= 0.5:
                performance_rating = "GOOD"
            elif win_rate >= 0.4:
                performance_rating = "FAIR"
            else:
                performance_rating = "NEEDS IMPROVEMENT"

            self.logger.info(f"PERFORMANCE RATING: {performance_rating}")

            if profit_factor >= 2.0:
                self.logger.info("ANALYSIS: Strong profit factor indicates good risk management")
            elif profit_factor >= 1.5:
                self.logger.info("ANALYSIS: Moderate profit factor - consider optimizing win/loss ratio")
            else:
                self.logger.info("ANALYSIS: Low profit factor - focus on cutting losses and letting profits run")

            self.logger.info("=" * 80)

        except Exception as e:
            self.logger.error(f"Error logging performance summary: {e}")

    async def fundamental_monitor_loop(self):
        """Background fundamental monitor that checks for breaking news every 5 minutes"""
        self.logger.info("Fundamental Monitor started - checking for breaking news every 5 minutes")

        while self.running:
            try:
                # Wait 5 minutes
                await asyncio.sleep(300)

                if not self.running:
                    break

                self.logger.info("Fundamental Monitor: Checking for breaking news...")

                # Get all active positions
                active_positions = []
                symbols = self.config.get('trading', {}).get('symbols', [])

                for symbol in symbols:
                    try:
                        positions = mt5.positions_get(symbol=symbol)
                        if positions:
                            for position in positions:
                                if hasattr(position, 'magic') and position.magic == self.magic_number:
                                    active_positions.append(position)
                    except Exception as e:
                        self.logger.debug(f"Error checking positions for {symbol}: {e}")

                if not active_positions:
                    self.logger.debug("Fundamental Monitor: No active positions to monitor")
                    continue

                self.logger.info(f"Fundamental Monitor: Monitoring {len(active_positions)} active positions")

                # Check each position for fundamental updates
                for position in active_positions:
                    try:
                        await self.trading_engine.check_fundamental_updates_during_trade(position)  # type: ignore
                    except Exception as e:
                        self.logger.error(f"Error in fundamental check for {position.symbol}: {e}")

                self.logger.info("Fundamental Monitor: Check completed")

            except Exception as e:
                self.logger.error(f"Error in fundamental monitor loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

        self.logger.info("Fundamental Monitor stopped")


def main():
    """Main entry point with crash protection"""
    try:
        # Create and run application
        app = FXAiApplication()

        # Run async main
        asyncio.run(app.run())

    except KeyboardInterrupt:
        logging.info("System stopped by user (Ctrl+C)")
        sys.exit(0)

    except Exception as e:
        # Log the full crash details
        logging.critical("=" * 70)
        logging.critical("FATAL CRASH DETECTED")
        logging.critical("=" * 70)
        logging.critical(f"Error: {e}")
        logging.critical("Full traceback:")
        logging.critical(traceback.format_exc())
        logging.critical("=" * 70)

        # Also write to separate crash log
        try:
            os.makedirs("logs", exist_ok=True)
            with open("logs/crash_log.txt", "a", encoding="utf-8") as f:
                from datetime import datetime
                f.write(f"\n{'=' * 70}\n")
                f.write(f"CRASH at {datetime.now()}\n")
                f.write(f"{'=' * 70}\n")
                f.write(f"Error: {e}\n\n")
                f.write(traceback.format_exc())
                f.write(f"{'=' * 70}\n\n")
        except BaseException:
            pass

        # Exit with error code so watchdog knows it crashed
        sys.exit(1)

if __name__ == "__main__":
    main()
