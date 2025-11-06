"""
FX-Ai: Advanced Forex Trading System for MT5
Main application with Adaptive Learning Integration
Version 3.0
"""

# Add project root to path
from utils.time_manager import get_time_manager
from utils.config_loader import ConfigLoader
from ai.adaptive_learning_manager import AdaptiveLearningManager
from ai.market_regime_detector import MarketRegimeDetector
from ai.reinforcement_learning_agent import RLAgent
from ai.advanced_risk_metrics import AdvancedRiskMetrics
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
from utils.time_manager import get_time_manager
import MetaTrader5 as mt5
import time as time_module
import threading
import json
import signal
from datetime import datetime, time
import asyncio
import pandas as pd
import sys
import os
import logging
import traceback
from typing import Dict, Any, Optional
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
        self.param_manager = None

        # Control flags
        self.running = False
        self.learning_enabled = self.config.get(
            'ml', {}).get('adaptive_learning', True)

        # Performance tracking
        self.session_stats = {
            'start_time': datetime.now(),
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'models_retrained': 0,
            'parameters_optimized': 0
        }

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.shutdown_handler)
        signal.signal(signal.SIGTERM, self.shutdown_handler)

    async def validate_configuration(self):
        """Validate critical configuration before trading starts"""
        from utils.exceptions import ConfigurationError, MissingParametersError
        import os
        import json
        
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
            self.risk_manager = RiskManager(self.config)

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

            self.logger.info(" All components initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            return False

    async def trading_loop(self):
        """Main trading loop with adaptive learning integration"""
        self.logger.info("Starting adaptive trading loop...")
        self.running = True

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
                            adaptive_params = self.adaptive_learning.get_regime_adapted_parameters(
                                symbol, adaptive_params)
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
                            0.25) *
                        technical_score +
                        signal_weights.get(
                            'ml_prediction',
                            0.30) *
                        ml_prediction.get(
                            'probability',
                            0) +
                        signal_weights.get(
                            'sentiment_score',
                            0.20) *
                        sentiment_score +
                        signal_weights.get(
                            'fundamental_score',
                            0.15) *
                        fundamental_data.get(  # type: ignore
                            symbol,
                            {}).get(  # type: ignore
                                'score',
                                0.5) +
                        signal_weights.get(
                            'sr_score',
                            0.10) *
                        market_data.get(
                            'sr_score',
                            0.5))

                    # Log all signal strengths for debugging
                    self.logger.info(
                        f"{symbol} signal: strength={signal_strength:.3f} "
                        f"(Tech:{technical_score:.3f}, "
                        f"ML:{ml_prediction.get('probability', 0):.3f}, "
                        f"Sent:{sentiment_score:.3f})"
                    )

                    # Apply adaptive minimum threshold
                    min_threshold = 0.4  # Lowered from 0.5 to allow trading during European session
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
                                    'take_profit_atr_multiplier', 6.0)
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
                                tp_atr_multiplier = adaptive_params.get('take_profit_atr_multiplier', 6.0)
                                base_sl_distance = atr_value * sl_atr_multiplier
                                base_tp_distance = atr_value * tp_atr_multiplier

                                # Convert distances back to pips for analyzer input
                                pip_size = 0.0001 if symbol.endswith(('JPY', 'XAG')) else 0.0001
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
                            pip_size = 0.0001 if symbol.endswith(('JPY', 'XAG')) else 0.0001
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

                            if actual_ratio < 2.0:  # Lowered from 2.9 to 2.0 for European session
                                self.logger.info(
                                    f"{symbol} {action} rejected: insufficient " f"reward ratio {
                                        actual_ratio:.2f}:1 " f"(required: 2.0:1)")
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
                        signal['timestamp'] = datetime.now()
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

                # Check if trading is allowed (before 22:30 MT5 server time)
                if self.config.get('trading', {}).get('day_trading_only', True):
                    is_allowed, reason = self.time_manager.is_trading_allowed()

                    if not is_allowed:
                        self.logger.info(f"Trading halted: {reason}")
                        signals = []  # Clear all signals to prevent trading

                # 4. Execute trades with risk management
                if signals:
                    self.logger.info(f"Generated {len(signals)} trading signal(s), evaluating for execution...")
                else:
                    self.logger.info(f"No trading signals generated this cycle")
                
                for signal in signals:
                    # Validate signal has required trading parameters
                    print(f"Processing signal for {signal['symbol']}: action={signal['action']}", flush=True)
                    # self.logger.info(
                    #     f"Processing signal for {signal['symbol']}: "
                    #     f"action={signal['action']}, "
                    #     f"strength={signal.get('strength', 0):.3f}, "
                    #     f"entry={signal.get('entry_price', 'None')}, "
                    #     f"stop={signal.get('stop_loss', 'None')}, "
                    #     f"tp={signal.get('take_profit', 'None')}")
                    if not signal.get(
                            'entry_price') or not signal.get('stop_loss'):
                        print(f"Skipping signal for {signal['symbol']} - missing entry/stop loss data", flush=True)
                        # self.logger.warning(
                        #     f"Skipping signal for {signal['symbol']} - "
                        #     f"missing entry/stop loss data "
                        #     f"(entry: {signal.get('entry_price', 'None')}, "
                        #     f"stop: {signal.get('stop_loss', 'None')})"
                        # )
                        continue

                    # Update risk metrics before checking limits
                    # Note: New RiskManager doesn't have update_metrics method

                    # Check risk limits with adaptive multiplier
                    risk_check = self.risk_manager.can_trade(signal['symbol'])  # type: ignore
                    self.logger.info(
                        f"Risk check for {
                            signal['symbol']} {
                            signal['action']}: {risk_check}")
                    if risk_check:
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
                                            f"(VaR95: {var_95*100:.2f}%, CVaR95: {cvar_95*100:.2f}%, Sharpe: {sharpe:.2f})"
                                        )
                                    
                                    # Log advanced metrics every 50th loop
                                    if loop_count % 50 == 0:
                                        self.logger.info(
                                            f"{signal['symbol']} Risk Metrics: "
                                            f"VaR(95%): {var_95*100:.2f}%, "
                                            f"CVaR(95%): {cvar_95*100:.2f}%, "
                                            f"Sharpe: {sharpe:.2f}, "
                                            f"Position Risk: {risk_pct:.2f}%"
                                        )

                            except Exception as e:
                                print(f"Advanced risk assessment failed for {signal['symbol']}: {e}", flush=True)

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
                        try:
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
                        except Exception as e:
                            response_time = time_module.time() - start_time
                            trade_result = {'success': False, 'error': str(e)}
                            print(f"Trade execution failed for {signal['symbol']}: {e}", flush=True)

                        if trade_result.get('success', False):
                            self.session_stats['total_trades'] += 1
                            
                            # Record trade for daily limit tracking (ONE TRADE PER SYMBOL PER DAY)
                            self.risk_manager.record_trade(signal['symbol'])  # type: ignore
                            self.logger.info(f"{signal['symbol']}: Trade executed and recorded - no more trades allowed today")

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
                                        'timestamp': datetime.now()
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
                                    'timestamp': datetime.now(),
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
                for symbol in symbols:
                    await self.trading_engine.manage_positions(symbol)  # type: ignore

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

                    self.session_stats['models_retrained'] += 1

                # 7. Check for day trading closure
                if self.config.get('trading', {}).get(
                        'day_trading_only', True):
                    try:
                        await self.check_day_trading_closure()
                    except Exception as e:
                        self.logger.error(f"Error in day trading closure: {e}")

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
                            datetime.now() - trade_data['timestamp']).seconds // 60
                        trade_data['volume'] = history.get('volume', 0)

                        # Record trade result for cooldown management
                        actual_profit = history.get('profit', 0)
                        self.risk_manager.record_trade_result(  # type: ignore
                            trade_data['symbol'], actual_profit)

                        # Record for learning
                        if self.adaptive_learning:
                            self.adaptive_learning.record_trade(trade_data)

                        # ===== REINFORCEMENT LEARNING UPDATE =====
                        if self.reinforcement_agent and self.reinforcement_agent.enabled:
                            try:
                                # Get the stored experience for this ticket
                                if hasattr(self.reinforcement_agent, 'pending_experiences') and ticket in self.reinforcement_agent.pending_experiences:
                                    experience = self.reinforcement_agent.pending_experiences[ticket]
                                    
                                    # Calculate reward based on trade outcome
                                    reward = self.reinforcement_agent.calculate_reward(
                                        experience['entry_price'],
                                        exit_price,
                                        trade_data['direction'],
                                        trade_data['duration_minutes']
                                    )
                                    
                                    # Create next state (position closed)
                                    next_state = experience['initial_state'].copy()
                                    next_state['position_status'] = 0  # Position closed
                                    
                                    # Learn from this experience
                                    self.reinforcement_agent.update_q_table(
                                        experience['initial_state'], 
                                        experience['action'], 
                                        reward, 
                                        next_state
                                    )
                                    
                                    # Remove from pending experiences
                                    del self.reinforcement_agent.pending_experiences[ticket]
                                    
                                    self.logger.debug(f"RL updated: action={experience['action']}, reward={reward:.4f}")
                                    
                            except Exception as e:
                                self.logger.warning(f"Failed to update RL agent: {e}")

                        # Update session stats
                        if profit_pct > 0:
                            self.session_stats['winning_trades'] += 1
                        else:
                            self.session_stats['losing_trades'] += 1

                        self.session_stats['total_profit'] += profit_pct

                        self.logger.info(
                            f"Trade completed - {trade_data['symbol']}: "
                            f"{'WIN' if profit_pct > 0 else 'LOSS'} {profit_pct:.2f}% "
                            f"Duration: {trade_data['duration_minutes']} minutes"
                        )

                    break
                else:
                    # Position still open - check time-based exit conditions
                    holding_minutes = (
                        datetime.now() - trade_data['timestamp']).seconds // 60

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
                                (entry_price - current_price) / entry_price) * 100
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

    async def check_day_trading_closure(self):
        """Check if positions should be closed for day trading - uses MT5 server time"""
        try:
            if not self.config.get('trading', {}).get(
                    'day_trading_only', True):
                return

            # Use TimeManager for consistent time handling
            should_close, reason = self.time_manager.should_close_positions()

            if should_close:
                self.logger.info(f"Day trading hours ending - closing all positions: {reason}")
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
            self.logger.error(f"Error in day trading closure: {e}")

    def get_default_parameters(self) -> dict:
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

        # Stop clock synchronization - DISABLED
        # if self.clock_sync:
        #     self.clock_sync.stop_sync_thread()

        # Stop system health monitoring
        # Disconnect from MT5
        if self.mt5:
            self.mt5.disconnect()

        self.logger.info("FX-Ai shutdown complete")

    def print_session_summary(self):
        """Print trading session summary"""
        duration = datetime.now() - self.session_stats['start_time']

        self.logger.info("=" * 60)
        self.logger.info("SESSION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Duration: {duration}")
        self.logger.info(f"Total Trades: {self.session_stats['total_trades']}")
        self.logger.info(
            f"Winning Trades: {self.session_stats['winning_trades']}")
        self.logger.info(
            f"Losing Trades: {self.session_stats['losing_trades']}")

        if self.session_stats['total_trades'] > 0:
            win_rate = self.session_stats['winning_trades'] / \
                self.session_stats['total_trades'] * 100
            self.logger.info(f"Win Rate: {win_rate:.1f}%")

        self.logger.info(
            f"Total Profit: {self.session_stats['total_profit']:.2f}%")

        if self.learning_enabled:
            self.logger.info(
                f"Models Retrained: {self.session_stats['models_retrained']}")
            self.logger.info(
                f"Parameters Optimized: {
                    self.session_stats['parameters_optimized']}")

            if self.adaptive_learning:
                performance = self.adaptive_learning.get_performance_summary()
                self.logger.info(
                    f"Final Signal Weights: {
                        json.dumps(
                            performance['signal_weights'],
                            indent=2)}")
                self.logger.info(
                    f"Final Adaptive Parameters: {
                        json.dumps(
                            performance['adaptive_params'],
                            indent=2)}")

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
            import os
            os.makedirs("logs", exist_ok=True)
            with open("logs/crash_log.txt", "a", encoding="utf-8") as f:
                from datetime import datetime
                f.write(f"\n{'=' * 70}\n")
                f.write(f"CRASH at {datetime.now()}\n")
                f.write(f"{'=' * 70}\n")
                f.write(f"Error: {e}\n\n")
                f.write(traceback.format_exc())
                f.write(f"{'=' * 70}\n\n")
        except:
            pass
        
        # Exit with error code so watchdog knows it crashed
        sys.exit(1)


if __name__ == "__main__":
    main()
