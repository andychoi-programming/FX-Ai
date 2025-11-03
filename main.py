"""
FX-Ai: Advanced Forex Trading System for MT5
Main application with Adaptive Learning Integration
Version 3.0
"""

# Add project root to path
from utils.logger import setup_logger
from utils.config_loader import ConfigLoader
from ai.adaptive_learning_manager import AdaptiveLearningManager
from ai.market_regime_detector import MarketRegimeDetector
from ai.ensemble_predictor import EnsemblePredictor
from ai.reinforcement_learning_agent import RLAgent
from ai.advanced_risk_metrics import AdvancedRiskMetrics
from ai.anomaly_detector import AnomalyDetector
from ai.system_health_monitor import SystemHealthMonitor
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
from datetime import datetime, time
import asyncio
import pandas as pd
import sys
import os
from typing import Dict, Any
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
        self.ensemble_predictor = None
        self.backtest_engine = None
        self.trading_engine = None
        self.adaptive_learning = None
        self.market_regime_detector = None
        self.reinforcement_agent = None
        self.advanced_risk_metrics = None
        self.anomaly_detector = None
        self.system_health_monitor = None
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

            # Reconfigure logger to use MT5 server time
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
                mt5_connector=self.mt5)

            # 2. Clock Synchronization
            self.logger.info("Initializing clock synchronizer...")
            self.clock_sync = ClockSynchronizer(self.mt5)
            self.clock_sync.start_sync_thread()

            # 3. Risk Manager
            self.logger.info("Initializing risk manager...")
            self.risk_manager = RiskManager(self.config)

            # 4. Market Data Manager
            self.logger.info("Initializing market data manager...")
            self.market_data = MarketDataManager(self.mt5, self.config)

            # 5. Fundamental Data Collector
            self.logger.info("Initializing fundamental data collector...")
            self.fundamental_collector = FundamentalDataCollector(self.config)

            # 6. Technical Analyzer
            self.logger.info("Initializing technical analyzer...")
            self.technical_analyzer = TechnicalAnalyzer(self.config)

            # 7. Sentiment Analyzer
            self.logger.info("Initializing sentiment analyzer...")
            self.sentiment_analyzer = SentimentAnalyzer(self.config)

            # 8. ML Predictor
            self.logger.info("Initializing ML predictor...")
            self.ml_predictor = MLPredictor(self.config)

            # 8.3. Parameter Manager
            self.logger.info("Initializing Parameter Manager...")
            self.param_manager = DynamicParameterManager(self.config)

            # 8.5. Ensemble Predictor
            self.logger.info("Initializing Ensemble Predictor...")
            self.ensemble_predictor = EnsemblePredictor(self.config)
            self.ensemble_predictor.initialize_models()

            # 9. Adaptive Learning Manager
            self.logger.info("Initializing Adaptive Learning Manager...")
            self.adaptive_learning = AdaptiveLearningManager(
                self.config,
                ml_predictor=self.ml_predictor,
                risk_manager=self.risk_manager,
                mt5_connector=self.mt5
            )

            # Start continuous learning thread
            if self.learning_enabled:
                self.logger.info("Starting continuous learning thread...")
                # Note: Thread is already started in AdaptiveLearningManager.__init__()

            # 9.5. Market Regime Detector
            self.logger.info("Initializing Market Regime Detector...")
            self.market_regime_detector = MarketRegimeDetector(self.config)

            # 9.6. Reinforcement Learning Agent
            self.logger.info("Initializing Reinforcement Learning Agent...")
            self.reinforcement_agent = RLAgent(self.config)

            # 9.7. Advanced Risk Metrics
            self.logger.info("Initializing Advanced Risk Metrics...")
            self.advanced_risk_metrics = AdvancedRiskMetrics(self.config)

            # 9.8. Anomaly Detector
            self.logger.info("Initializing Anomaly Detector...")
            self.anomaly_detector = AnomalyDetector(self.config)

            # 9.9. System Health Monitor
            self.logger.info("Initializing System Health Monitor...")
            self.system_health_monitor = SystemHealthMonitor(
                self.config.get('system_health_monitoring', {}))

            # 10. Trading Engine
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
                        data = self.market_data.get_latest_data(symbol)
                        response_time = time_module.time() - start_time

                        if self.system_health_monitor:
                            self.system_health_monitor.record_api_call(
                                'market_data.get_latest_data', response_time, data is not None)

                        if data is not None:
                            market_data_dict[symbol] = data
                            # Get bars for technical analysis from multiple
                            # timeframes
                            bars_m1 = self.market_data.get_bars(
                                symbol, mt5.TIMEFRAME_M1, 100)
                            bars_m5 = self.market_data.get_bars(
                                symbol, mt5.TIMEFRAME_M5, 100)
                            bars_h1 = self.market_data.get_bars(
                                symbol, mt5.TIMEFRAME_H1, 100)
                            bars_h4 = self.market_data.get_bars(
                                symbol, mt5.TIMEFRAME_H4, 100)
                            bars_d1 = self.market_data.get_bars(
                                symbol, mt5.TIMEFRAME_D1, 100)

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
                        if self.system_health_monitor:
                            self.system_health_monitor.record_api_call(
                                'market_data.get_latest_data', response_time, False, str(e))
                        self.logger.warning(f"Failed to get market data for {symbol}: {e}")

                if not market_data_dict:
                    await asyncio.sleep(10)
                    continue

                # 2. Collect fundamental data (with caching)
                start_time = time_module.time()
                try:
                    fundamental_data = self.fundamental_collector.collect()
                    response_time = time_module.time() - start_time
                    if self.system_health_monitor:
                        self.system_health_monitor.record_api_call(
                            'fundamental_collector.collect', response_time, True)
                except Exception as e:
                    response_time = time_module.time() - start_time
                    if self.system_health_monitor:
                        self.system_health_monitor.record_api_call(
                            'fundamental_collector.collect', response_time, False, str(e))
                    self.logger.warning(f"Failed to collect fundamental data: {e}")
                    fundamental_data = {}

                # 3. Generate trading signals with adaptive weights
                signals = []

                for symbol, market_data in market_data_dict.items():
                    # ===== ANOMALY DETECTION =====
                    anomaly_report = None
                    if self.anomaly_detector and self.anomaly_detector.enabled:
                        try:
                            # Get historical data for anomaly detection
                            historical_data = self.market_data.get_bars(symbol, mt5.TIMEFRAME_H1, 300)  # Last 300 hours
                            if historical_data is not None and len(historical_data) >= 200:
                                hist_df = pd.DataFrame(historical_data)
                                anomaly_report = self.anomaly_detector.get_comprehensive_anomaly_report(symbol, hist_df)

                                if anomaly_report['overall_anomalies_detected']:
                                    self.logger.warning(
                                        f"Anomalies detected for {symbol}: "
                                        f"severity={anomaly_report['overall_severity_score']:.2f}, "
                                        f"recommendation={anomaly_report['recommendation']}"
                                    )

                                    # Skip trading based on anomaly severity
                                    if anomaly_report['recommendation'] == 'halt_trading':
                                        self.logger.warning(f"Halting trading for {symbol} due to severe anomalies")
                                        continue
                                    elif anomaly_report['recommendation'] == 'reduce_position_size':
                                        self.logger.info(f"Reducing position size for {symbol} due to anomalies")
                                        # This will be handled in risk management
                                    elif anomaly_report['recommendation'] == 'increase_stops':
                                        self.logger.info(f"Increasing stops for {symbol} due to anomalies")
                                        # This will be handled in risk management

                        except Exception as e:
                            self.logger.warning(f"Anomaly detection failed for {symbol}: {e}")

                    # Get adaptive parameters if learning is enabled
                    if self.adaptive_learning:
                        signal_weights = self.adaptive_learning.get_current_weights()
                        adaptive_params = self.adaptive_learning.get_adaptive_parameters()
                        # Force fixed risk
                        adaptive_params['risk_multiplier'] = 1.0

                        # Apply regime adaptation if available
                        if self.market_regime_detector and hasattr(self.adaptive_learning, 'get_regime_adapted_parameters'):
                            adaptive_params = self.adaptive_learning.get_regime_adapted_parameters(
                                symbol, adaptive_params)
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

                    technical_signals = await self.technical_analyzer.analyze(
                        symbol,
                        bars
                    )
                    technical_score = technical_signals.get(
                        'overall_score', 0.5)

                    # ML prediction (use ensemble if available)
                    if self.ensemble_predictor and self.ensemble_predictor.enabled:
                        try:
                            # Prepare features for ensemble prediction
                            features_df = self.ml_predictor.prepare_features(symbol, bars, technical_signals)
                            if features_df is not None and not features_df.empty:
                                ensemble_pred, ensemble_info = self.ensemble_predictor.predict(features_df)
                                ml_prediction = {
                                    'direction': 1 if ensemble_pred[0] == 1 else -1,
                                    'probability': ensemble_info.get('probabilities', [0.5])[0],
                                    'confidence': ensemble_info.get('confidence', [0.5])[0],
                                    'model': 'ensemble',
                                    'ensemble_info': ensemble_info
                                }
                                self.logger.debug(
                                    f"{symbol}: Ensemble prediction - direction: {ml_prediction['direction']}, "
                                    f"prob: {ml_prediction['probability']:.3f}, conf: {ml_prediction['confidence']:.3f}")
                            else:
                                # Fallback to regular ML predictor
                                ml_prediction = await self.ml_predictor.predict(symbol, bars, technical_signals)
                        except Exception as e:
                            self.logger.warning(f"{symbol}: Ensemble prediction failed, using regular ML: {e}")
                            ml_prediction = await self.ml_predictor.predict(symbol, bars, technical_signals)
                    else:
                        # Use regular ML predictor
                        start_time = time_module.time()
                        try:
                            ml_prediction = await self.ml_predictor.predict(symbol, bars, technical_signals)
                            response_time = time_module.time() - start_time
                            if self.system_health_monitor:
                                self.system_health_monitor.record_api_call(
                                    'ml_predictor.predict', response_time, True)
                        except Exception as e:
                            response_time = time_module.time() - start_time
                            if self.system_health_monitor:
                                self.system_health_monitor.record_api_call(
                                    'ml_predictor.predict', response_time, False, str(e))
                            
                            # GRACEFUL DEGRADATION: Fall back to technical analysis only
                            self.logger.warning(
                                f"{symbol}: ML prediction failed, falling back to technical analysis only: {e}")
                            
                            # Use signal_strength as fallback confidence
                            # Strong buy (>0.7) or strong sell (<-0.7) = higher confidence
                            fallback_confidence = min(abs(signal_strength), 0.75)  # Cap at 0.75 to indicate fallback
                            
                            ml_prediction = {
                                'direction': 'BUY' if signal_strength > 0 else 'SELL',
                                'probability': (signal_strength + 1.0) / 2.0,  # Normalize -1..1 to 0..1
                                'confidence': fallback_confidence,
                                'source': 'technical_fallback',  # Mark as fallback
                                'signal_strength': signal_strength
                            }
                            
                            self.logger.info(
                                f"{symbol}: Fallback prediction - {ml_prediction['direction']} "
                                f"(confidence: {fallback_confidence:.2f}, signal: {signal_strength:.2f})")

                    # Sentiment analysis
                    start_time = time_module.time()
                    try:
                        sentiment_result = await self.sentiment_analyzer.analyze_sentiment(
                            symbol)
                        response_time = time_module.time() - start_time
                        sentiment_score = sentiment_result.get(
                            'overall_score', 0.5)
                        if self.system_health_monitor:
                            self.system_health_monitor.record_api_call(
                                'sentiment_analyzer.analyze_sentiment', response_time, True)
                    except Exception as e:
                        response_time = time_module.time() - start_time
                        sentiment_score = 0.5  # Default neutral score
                        if self.system_health_monitor:
                            self.system_health_monitor.record_api_call(
                                'sentiment_analyzer.analyze_sentiment', response_time, False, str(e))
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
                        fundamental_data.get(
                            symbol,
                            {}).get(
                                'score',
                                0.5) +
                        signal_weights.get(
                            'sr_score',
                            0.10) *
                        market_data.get(
                            'sr_score',
                            0.5))

                    # Debug: Log all signal strengths
                    self.logger.debug(
                        f"{symbol} signal: strength={signal_strength:.3f} "
                        f"(Tech:{technical_score:.3f}, "
                        f"ML:{ml_prediction.get('probability', 0):.3f}, "
                        f"Sent:{sentiment_score:.3f})"
                    )

                    # Apply adaptive minimum threshold
                    min_threshold = 0.5  # Override adaptive threshold for testing
                    self.logger.debug(
                        f"{symbol}: threshold={min_threshold:.3f}, "
                        f"strength={signal_strength:.3f}")

                    if signal_strength > min_threshold:
                        # Get current market price for entry
                        current_data = market_data_dict.get(symbol, {})
                        entry_price = current_data.get('ask') if ml_prediction.get(
                            'direction') == 1 else current_data.get('bid', 0)

                        # Skip if no valid price data
                        if entry_price <= 0:
                            self.logger.debug(
                                f"{symbol}: Skipping - no valid entry price")
                            continue

                        self.logger.debug(
                            f"{symbol}: Processing signal - entry_price={entry_price}, "
                            f"direction={ml_prediction.get('direction')}")

                        # ===== NEW LEARNING FEATURES =====

                        # 1. Check entry timing recommendation
                        if self.adaptive_learning:
                            current_hour = datetime.now().hour
                            timing_recommendation = (
                                self.adaptive_learning.get_entry_timing_recommendation(
                                    symbol, current_hour))
                            if not timing_recommendation['recommended']:
                                self.logger.debug(
                                    f"{symbol}: Skipping - poor timing "
                                    f"(win_rate: {timing_recommendation['win_rate']:.2f})")
                                continue

                        # 2. Check entry filters (when NOT to enter)
                        if self.adaptive_learning:
                            current_conditions = {
                                'hour': datetime.now().hour,
                                'volatility': technical_signals.get(
                                    'atr', {}).get(
                                        'value', 0),
                                'spread': current_data.get(
                                    'spread', 0)
                            }
                            should_enter = (
                                self.adaptive_learning.should_enter_based_on_filters(
                                    symbol, current_conditions))
                            if not should_enter:
                                self.logger.debug(
                                    f"{symbol}: Skipping - entry filter triggered")
                                continue

                        # 3. Get optimized SL/TP parameters for this symbol
                        if self.adaptive_learning:
                            symbol_sl_tp = self.adaptive_learning.get_symbol_sl_tp_params(
                                symbol)
                            # Override global parameters with symbol-specific optimized ones
                            # Only use if confidence is high enough
                            if symbol_sl_tp['confidence'] > 0.5:
                                adaptive_params['stop_loss_atr_multiplier'] = symbol_sl_tp['sl_atr_multiplier']
                                adaptive_params['take_profit_atr_multiplier'] = symbol_sl_tp['tp_atr_multiplier']
                                self.logger.debug(
                                    f"{symbol}: Using optimized SL/TP - "
                                    f"SL:{symbol_sl_tp['sl_atr_multiplier']:.2f}, "
                                    f"TP:{symbol_sl_tp['tp_atr_multiplier']:.2f}")

                        # 4. Check economic calendar impact (avoid trading
                        # during high-impact events)
                        if self.adaptive_learning:
                            events_to_avoid = self.adaptive_learning.should_avoid_economic_events(
                                hours_ahead=24)
                            if events_to_avoid:
                                self.logger.debug(
                                    f"{symbol}: Skipping - upcoming high-impact events: "
                                    f"{events_to_avoid}")
                                continue

                        # 5. Apply optimized technical indicator parameters
                        if self.adaptive_learning:
                            optimized_tech_params = self.adaptive_learning.get_optimized_technical_params(
                                symbol)
                            if optimized_tech_params:
                                # Update technical analyzer with optimized
                                # parameters
                                for param_key, param_value in optimized_tech_params.items():
                                    # This would require extending technical_analyzer to accept
                                    # dynamic params
                                    # For now, just log the optimized
                                    # parameters
                                    self.logger.debug(
                                        f"{symbol}: Using optimized {param_key} = {param_value}")

                        # 6. Apply optimized fundamental weights
                        if self.adaptive_learning:
                            optimized_fundamental_weights = (
                                self.adaptive_learning.get_optimized_fundamental_weights())
                            if optimized_fundamental_weights:
                                # Update fundamental collector with optimized weights
                                # This would require extending
                                # fundamental_collector to accept dynamic
                                # weights
                                self.logger.debug(
                                    f"{symbol}: Using optimized fundamental weights: "
                                    f"{optimized_fundamental_weights}")

                        # 7. Apply optimized sentiment parameters
                        # TODO: Implement get_optimized_sentiment_params method in AdaptiveLearningManager
                        # if self.adaptive_learning:
                        #     optimized_sentiment_params = self.adaptive_learning.get_optimized_sentiment_params()
                        #     if optimized_sentiment_params:
                        #         # Update sentiment analyzer with optimized parameters
                        #         # This would require extending sentiment_analyzer to accept dynamic params
                        #         self.logger.debug(
                        #             f"{symbol}: Using optimized sentiment params: {optimized_sentiment_params}")

                        # 8. Check interest rate impact expectations
                        if self.adaptive_learning:
                            # Extract currency from symbol (e.g., EURUSD ->
                            # EUR, USD)
                            base_currency = symbol[:3]
                            quote_currency = symbol[3:6]

                            base_expectations = self.adaptive_learning.get_interest_rate_expectations(
                                base_currency)
                            quote_expectations = self.adaptive_learning.get_interest_rate_expectations(
                                quote_currency)

                            # Simple logic: avoid trading if both currencies
                            # have strong rate correlations
                            if base_expectations and quote_expectations:
                                base_corr = max([v.get('correlation', 0)
                                                for v in base_expectations.values()])
                                quote_corr = max(
                                    [v.get('correlation', 0) for v in quote_expectations.values()])

                                if base_corr > 0.7 and quote_corr > 0.7:
                                    self.logger.debug(
                                        f"{symbol}: Skipping - high interest rate "
                                        f"correlation ({base_corr:.2f}, {quote_corr:.2f})")
                                    continue

                        # 9. Check temporal analysis recommendations (optimal trading times)
                        if self.adaptive_learning:
                            temporal_recommendation = (
                                self.adaptive_learning.get_comprehensive_temporal_recommendation(
                                    symbol))
                            if not temporal_recommendation['recommended']:
                                self.logger.debug(
                                    f"{symbol}: Skipping - poor temporal timing "
                                    f"(confidence: {temporal_recommendation['confidence']:.2f}, "
                                    f"reason: {temporal_recommendation['reason']})")
                                continue

                        # 10. Check market regime and adapt strategy
                        if self.market_regime_detector:
                            # Get historical data for regime analysis
                            historical_data = self.market_data.get_bars(
                                symbol, timeframe=mt5.TIMEFRAME_H1, count=300)

                            if historical_data is not None and len(historical_data) >= 200:
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

                        # ===== END NEW LEARNING FEATURES =====

                        # Calculate stop loss using ATR (more sophisticated
                        # than fixed percentage)
                        atr_value = technical_signals.get(
                            'atr', {}).get('value', 0)
                        if atr_value > 0:
                            self.logger.debug(
                                f"{symbol}: ATR available ({atr_value:.5f}) - "
                                f"proceeding with signal")
                            # Use adaptive ATR multiplier for stop loss distance
                            # Use higher multipliers for precious metals to
                            # meet broker requirements
                            base_multiplier = adaptive_params.get(
                                'stop_loss_atr_multiplier', 3.0)  # Increased from 2.0
                            if 'XAU' in symbol or 'XAG' in symbol:
                                # Precious metals need higher multipliers due
                                # to lower ATR on M1 timeframe
                                if 'XAUUSD' in symbol:
                                    # XAUUSD ATR is very wide, reduce
                                    # multiplier to allow proper position
                                    # sizing
                                    sl_atr_multiplier = 0.3  # Much lower multiplier for XAUUSD
                                else:
                                    # Keep 4.0 for XAGUSD
                                    sl_atr_multiplier = max(
                                        base_multiplier, 4.0)
                            else:
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
                                optimal_params = self.param_manager.get_optimal_parameters(symbol, 'H1')
                                tp_pips = optimal_params.get('tp_pips', 600)  # Default 600 pips for metals
                                # Convert TP pips to distance
                                take_profit_distance = tp_pips * 0.01  # Metals: 1 pip = 0.01
                                tp_atr_multiplier = take_profit_distance / atr_value if atr_value > 0 else 0  # Calculate for logging
                                self.logger.info(f"{symbol} using optimized metal take profit: {tp_pips} pips = {take_profit_distance:.5f}")
                            else:
                                # Use ATR-based TP for forex
                                base_tp_multiplier = adaptive_params.get(
                                    'take_profit_atr_multiplier', 6.0)
                                tp_atr_multiplier = base_tp_multiplier
                                take_profit_distance = atr_value * tp_atr_multiplier

                            if ml_prediction.get('direction') == 1:  # BUY
                                take_profit = entry_price + take_profit_distance
                            else:  # SELL
                                take_profit = entry_price - take_profit_distance
                            self.logger.debug(
                                f"{symbol} take profit: ATR={
                                    atr_value:.5f}, multiplier={
                                    tp_atr_multiplier:.1f}, " f"distance={
                                    take_profit_distance:.5f}, level={
                                    take_profit:.5f}")
                        else:
                            # Fallback take profit: 6% of entry price (3x the
                            # 2% stop loss fallback)
                            take_profit_distance = entry_price * 0.06

                            if ml_prediction.get('direction') == 1:  # BUY
                                take_profit = entry_price + take_profit_distance
                            else:  # SELL
                                take_profit = entry_price - take_profit_distance
                            self.logger.debug(
                                f"{symbol} fallback take profit: 6% of entry = " f"{
                                    take_profit_distance:.5f}, " f"level={
                                    take_profit:.5f}")

                        # Format take profit for logging
                        tp_display = f"{
                            take_profit:.5f}" if take_profit is not None else "None"

                        # Validate risk-reward ratio before proceeding
                        if stop_loss is not None and take_profit is not None:
                            risk_distance = abs(stop_loss - entry_price)
                            reward_distance = abs(take_profit - entry_price)
                            actual_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
                            action = 'BUY' if ml_prediction.get(
                                'direction') == 1 else 'SELL'

                            if actual_ratio < 2.9:
                                self.logger.debug(
                                    f"{symbol} {action} rejected: insufficient " f"reward ratio {
                                        actual_ratio:.2f}:1 " f"(required: 2.9:1)")
                                continue  # Skip this trade

                            self.logger.debug(
                                f"{symbol} {action} validated: "
                                f"{actual_ratio:.1f}:1 risk-reward ratio")

                        # ===== REINFORCEMENT LEARNING DECISION =====
                        rl_decision = 'hold'  # Default action
                        if self.reinforcement_agent and self.reinforcement_agent.enabled:
                            try:
                                # Prepare state for RL agent
                                current_state = {
                                    'rsi': technical_signals.get('rsi', {}).get('value', 50),
                                    'adx': regime_analysis.adx_value if 'regime_analysis' in locals() else 25,
                                    'volatility_ratio': technical_signals.get('atr', {}).get('value', 0) / entry_price if entry_price > 0 else 0,
                                    'trend_strength': technical_signals.get('adx', {}).get('value', 25),
                                    'market_regime': regime_analysis.primary_regime.value if 'regime_analysis' in locals() else 'ranging',
                                    'signal_strength': signal_strength,
                                    'position_status': 0  # No position currently
                                }

                                # Get RL action recommendation
                                rl_decision = self.reinforcement_agent.choose_action_from_dict(current_state)
                                self.logger.debug(f"{symbol}: RL decision - {rl_decision}")

                                # Only proceed if RL recommends buy/sell (not hold or close)
                                if rl_decision not in ['buy', 'sell']:
                                    self.logger.debug(f"{symbol}: Skipping - RL recommends {rl_decision}")
                                    continue

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
                            'ml_score': ml_prediction.get(
                                'probability',
                                0),
                            'technical_score': technical_score,
                            'sentiment_score': sentiment_score,
                            'rl_decision': rl_decision,
                            'timestamp': datetime.now(),
                            'adaptive_params': adaptive_params}
                        signals.append(signal)

                        self.logger.info(
                            f"Signal generated for {symbol}: {
                                signal['action']} " f"(strength: {
                                signal_strength:.3f}, " f"threshold: {
                                min_threshold:.3f}) " f"Entry: {
                                entry_price:.5f}, " f"SL: {
                                stop_loss:.5f} ({
                                sl_atr_multiplier:.1f}x ATR), " f"TP: {tp_display} ({
                                tp_atr_multiplier:.1f}x ATR)")

                # Check if trading is allowed (before 22:30 MT5 server time)
                if self.config.get('trading', {}).get('day_trading_only', True):
                    # Get MT5 server time instead of local computer time
                    server_time = self.mt5_connector.get_server_time()
                    if server_time:
                        current_time = server_time.time()
                    else:
                        current_time = datetime.now().time()  # Fallback to local time if server time unavailable
                    
                    close_hour = self.config.get('trading', {}).get('close_hour', 22)
                    close_minute = self.config.get('trading', {}).get('close_minute', 30)
                    close_time = time(close_hour, close_minute)
                    
                    if current_time >= close_time:
                        self.logger.info(f"Trading halted: MT5 server time {current_time} is after trading close time {close_hour:02d}:{close_minute:02d}")
                        signals = []  # Clear all signals to prevent trading

                # 4. Execute trades with risk management
                for signal in signals:
                    # Validate signal has required trading parameters
                    self.logger.info(
                        f"Processing signal for {signal['symbol']}: "
                        f"action={signal['action']}, "
                        f"strength={signal.get('strength', 0):.3f}, "
                        f"entry={signal.get('entry_price', 'None')}, "
                        f"stop={signal.get('stop_loss', 'None')}, "
                        f"tp={signal.get('take_profit', 'None')}")
                    if not signal.get(
                            'entry_price') or not signal.get('stop_loss'):
                        self.logger.warning(
                            f"Skipping signal for {signal['symbol']} - "
                            f"missing entry/stop loss data "
                            f"(entry: {signal.get('entry_price', 'None')}, "
                            f"stop: {signal.get('stop_loss', 'None')})"
                        )
                        continue

                    # Update risk metrics before checking limits
                    # Note: New RiskManager doesn't have update_metrics method

                    # Check risk limits with adaptive multiplier
                    risk_check = self.risk_manager.can_trade(signal['symbol'])
                    self.logger.info(
                        f"Risk check for {
                            signal['symbol']} {
                            signal['action']}: {risk_check}")
                    if risk_check:
                        # ===== ADVANCED RISK ASSESSMENT =====
                        if self.advanced_risk_metrics:
                            try:
                                # Get current positions for portfolio risk assessment
                                current_positions = {}
                                positions = self.trading_engine.get_all_positions()
                                for pos in positions:
                                    current_positions[pos['symbol']] = pos['volume']

                                # Get historical data for risk calculation
                                market_data_risk = {}
                                for sym in symbols:
                                    hist_data = self.market_data.get_bars(sym, mt5.TIMEFRAME_H1, 252)  # 1 year
                                    if hist_data is not None:
                                        market_data_risk[sym] = pd.DataFrame(hist_data)

                                # Assess portfolio risk
                                portfolio_risk = self.advanced_risk_metrics.assess_portfolio_risk(
                                    current_positions, market_data_risk)

                                # Get risk limits
                                account_info = mt5.account_info()
                                account_balance = account_info.balance if account_info else 10000
                                risk_limits = self.advanced_risk_metrics.get_risk_limits(signal['symbol'], account_balance)

                                # Check advanced risk limits
                                risk_warnings = portfolio_risk.get('warnings', [])
                                if risk_warnings:
                                    self.logger.warning(f"Portfolio risk warnings for {signal['symbol']}: {risk_warnings}")
                                    # Allow trade but log warnings
                                    for warning in risk_warnings:
                                        self.logger.warning(f"Risk Warning: {warning}")

                                # Check individual trade risk
                                trade_risk = abs(signal['entry_price'] - signal['stop_loss']) * position_size
                                if trade_risk > risk_limits['max_risk_per_trade']:
                                    self.logger.warning(
                                        f"Trade risk (${trade_risk:.2f}) exceeds limit (${risk_limits['max_risk_per_trade']:.2f}) for {signal['symbol']}")
                                    continue

                                # Log advanced risk metrics
                                if portfolio_risk:
                                    self.logger.debug(
                                        f"Advanced risk metrics for {signal['symbol']}: "
                                        f"Portfolio VaR: {portfolio_risk.get('portfolio_var_95', 0):.4f}, "
                                        f"Max DD: {portfolio_risk.get('portfolio_max_dd', 0):.4f}")

                            except Exception as e:
                                self.logger.warning(f"Advanced risk assessment failed for {signal['symbol']}: {e}")

                        # Check free margin percentage
                        account_info = mt5.account_info()
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
                        symbol_info = mt5.symbol_info(signal['symbol'])
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
                            min_lot_risk = self.risk_manager.calculate_risk_for_lot_size(
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

                        position_size = self.risk_manager.calculate_position_size(
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
                            trade_result = await self.trading_engine.place_order(
                                signal['symbol'],
                                signal['action'].lower(),
                                position_size,
                                stop_loss=signal.get('stop_loss'),
                                # Now using calculated take profit
                                take_profit=signal.get('take_profit'),
                                comment=f"FX-Ai {
                                    signal['action']} {
                                    signal_strength:.3f}"
                            )
                            response_time = time_module.time() - start_time
                            success = trade_result.get('success', False)
                            if self.system_health_monitor:
                                self.system_health_monitor.record_api_call(
                                    'trading_engine.place_order', response_time, success,
                                    trade_result.get('error', 'Unknown error') if not success else None)
                        except Exception as e:
                            response_time = time_module.time() - start_time
                            trade_result = {'success': False, 'error': str(e)}
                            if self.system_health_monitor:
                                self.system_health_monitor.record_api_call(
                                    'trading_engine.place_order', response_time, False, str(e))
                            self.logger.error(f"Trade execution failed for {signal['symbol']}: {e}")

                        if trade_result.get('success', False):
                            self.session_stats['total_trades'] += 1

                            # ===== REINFORCEMENT LEARNING EXPERIENCE =====
                            if self.reinforcement_agent and self.reinforcement_agent.enabled:
                                try:
                                    # Record initial state and action for RL learning
                                    initial_state = {
                                        'rsi': technical_signals.get('rsi', {}).get('value', 50),
                                        'adx': regime_analysis.adx_value if 'regime_analysis' in locals() else 25,
                                        'volatility_ratio': technical_signals.get('atr', {}).get('value', 0) / signal['entry_price'] if signal['entry_price'] > 0 else 0,
                                        'trend_strength': technical_signals.get('adx', {}).get('value', 25),
                                        'market_regime': regime_analysis.primary_regime.value if 'regime_analysis' in locals() else 'ranging',
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
                                    'model_version': self.ml_predictor.get_model_version(
                                        signal['symbol'])}

                                # Start monitoring thread for this trade
                                threading.Thread(
                                    target=self.monitor_trade,
                                    args=(trade_result.get(
                                        'order', 0), trade_data),
                                    daemon=True
                                ).start()

                # 5. Monitor and update positions
                for symbol in symbols:
                    await self.trading_engine.manage_positions(symbol)

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

                position = self.trading_engine.get_position_by_ticket(ticket)

                if position is None:  # Trade closed
                    # Get trade history
                    history = self.trading_engine.get_trade_history(ticket)

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
                        self.risk_manager.record_trade_result(
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
                            # Close position at market
                            close_result = self.trading_engine.close_position(
                                ticket, trade_data['symbol'])
                            if close_result and close_result.get('success'):
                                self.logger.info(
                                    "Successfully closed position for max time exit")
                            else:
                                self.logger.warning(
                                    "Failed to close position for max time exit")
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
                                    close_result = self.trading_engine.close_position(
                                        ticket, trade_data['symbol'])
                                    if close_result and close_result.get(
                                            'success'):
                                        self.logger.info(
                                            "Successfully closed position for optimal time exit")
                                    else:
                                        self.logger.warning(
                                            "Failed to close position for optimal time exit")
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
                                    close_result = self.trading_engine.close_position(
                                        ticket, trade_data['symbol'])
                                    if close_result and close_result.get(
                                            'success'):
                                        self.logger.info(
                                            "Successfully closed position for optimal time exit")
                                    else:
                                        self.logger.warning(
                                            "Failed to close position for optimal time exit")
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

            # Get MT5 server time instead of local computer time
            server_time = self.mt5_connector.get_server_time()
            if server_time:
                current_time = server_time.time()
                today = server_time.date()
            else:
                current_time = datetime.now().time()  # Fallback to local time
                today = datetime.now().date()
            
            # Get close time from config, default to 22:30 (10:30 PM)
            close_hour = self.config.get('trading', {}).get('close_hour', 22)
            close_minute = self.config.get('trading', {}).get('close_minute', 30)
            close_time = time(close_hour, close_minute)

            # Only close once per day - check if we haven't already closed
            # today
            if current_time >= close_time:
                # Check if we already closed positions today
                if hasattr(
                        self,
                        '_last_closure_date') and self._last_closure_date == today:
                    return  # Already closed today

                self.logger.info(
                    f"Day trading hours ending at {close_hour:02d}:{close_minute:02d} - closing all positions")
                self._last_closure_date = today  # Mark as closed today

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

    def get_default_weights(self) -> dict:
        """Get default signal weights"""
        return {
            'technical_score': 0.40,  # Increased for testing
            'ml_prediction': 0.20,    # Decreased for testing
            'sentiment_score': 0.20,
            'fundamental_score': 0.10,
            'support_resistance': 0.10
        }

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
                if hasattr(self.trading_engine, 'close_all_positions'):
                    close_method = self.trading_engine.close_all_positions
                    loop = asyncio.get_event_loop()
                    if asyncio.iscoroutinefunction(close_method):
                        loop.create_task(close_method())
                    else:
                        # Schedule synchronous close to run in executor
                        loop.create_task(
                            loop.run_in_executor(None, close_method))
                else:
                    self.logger.warning(
                        'No close_all_positions method on trading_engine')
            except Exception as e:
                self.logger.error(
                    f"Error scheduling close_all_positions on shutdown: {e}")

        # Stop clock synchronization
        if self.clock_sync:
            self.clock_sync.stop_sync_thread()

        # Stop system health monitoring
        if self.system_health_monitor:
            self.system_health_monitor.stop_monitoring()
            # Export final health report
            try:
                health_report = self.system_health_monitor.get_health_report()
                self.logger.info(f"Final health status: {health_report.get('status', 'unknown')}")
                if health_report.get('critical_issues'):
                    self.logger.warning(f"Critical issues at shutdown: {health_report['critical_issues']}")
            except Exception as e:
                self.logger.error(f"Error getting final health report: {e}")

        # Disconnect from MT5
        if self.mt5:
            self.mt5.disconnect()

        self.logger.info("FX-Ai shutdown complete")

    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        if not self.system_health_monitor:
            return {'status': 'not_initialized', 'message': 'System health monitor not initialized'}

        try:
            return self.system_health_monitor.get_health_report()
        except Exception as e:
            self.logger.error(f"Error getting health report: {e}")
            return {'status': 'error', 'message': str(e)}

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        if not self.system_health_monitor:
            return {'status': 'not_initialized', 'message': 'System health monitor not initialized'}

        try:
            return self.system_health_monitor.get_performance_stats()
        except Exception as e:
            self.logger.error(f"Error getting performance stats: {e}")
            return {'status': 'error', 'message': str(e)}

    def export_health_data(self, filepath: str = None) -> str:
        """Export health monitoring data to file"""
        if not self.system_health_monitor:
            return "System health monitor not initialized"

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"health_report_{timestamp}.json"

        try:
            self.system_health_monitor.export_health_data(filepath)
            return f"Health data exported to {filepath}"
        except Exception as e:
            error_msg = f"Error exporting health data: {e}"
            self.logger.error(error_msg)
            return error_msg

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

        # Start system health monitoring
        if self.system_health_monitor:
            self.system_health_monitor.start_monitoring()
            self.logger.info("System health monitoring started")

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
    """Main entry point"""
    # Create and run application
    app = FXAiApplication()

    # Run async main
    asyncio.run(app.run())


if __name__ == "__main__":
    main()
