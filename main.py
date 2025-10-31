"""
FX-Ai: Advanced Forex Trading System for MT5
Main application with Adaptive Learning Integration
Version 3.0
"""

import sys
import os
import asyncio
import logging
from datetime import datetime, time
import signal
import json
import threading
import time as time_module
import MetaTrader5 as mt5

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.mt5_connector import MT5Connector
from core.trading_engine import TradingEngine
from core.risk_manager import RiskManager
from data.market_data_manager import MarketDataManager
from analysis.fundamental_analyzer import FundamentalAnalyzer as FundamentalDataCollector
from analysis.technical_analyzer import TechnicalAnalyzer
from analysis.sentiment_analyzer import SentimentAnalyzer
from ai.ml_predictor import MLPredictor
from ai.adaptive_learning_manager import AdaptiveLearningManager
from utils.config_loader import ConfigLoader
from utils.logger import setup_logger

class FXAiApplication:
    """Main FX-Ai Trading Application with Adaptive Learning"""
    
    def __init__(self):
        """Initialize the FX-Ai application"""
        config_loader = ConfigLoader()
        config_loader.load_config()
        self.config = config_loader.config
        self.logger = setup_logger('FX-Ai', self.config.get('logging', {}).get('level', 'INFO'),
                                   self.config.get('logging', {}).get('file'),
                                   rotation_type=self.config.get('logging', {}).get('rotation_type', 'size'))
        self.logger.info("FX-Ai Application initialized with Adaptive Learning")
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
        
        # Control flags
        self.running = False
        self.learning_enabled = self.config.get('ml', {}).get('adaptive_learning', True)
        
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
    
    async def initialize_components(self):
        """Initialize all trading components"""
        try:
            self.logger.info("Initializing components...")
            
            # 1. MT5 Connection
            self.logger.info("Initializing MT5 connection...")
            self.mt5 = MT5Connector(self.config)
            if not self.mt5.connect():
                raise Exception("Failed to connect to MT5")
            
            # 2. Risk Manager
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
            
            # Models will be loaded on-demand during prediction
            
            # 8. Adaptive Learning Manager (NEW)
            if self.learning_enabled:
                self.logger.info("Initializing Adaptive Learning Manager...")
                self.adaptive_learning = AdaptiveLearningManager(self.config, mt5_connector=self.mt5)
                self.logger.info("  Adaptive Learning enabled - System will improve over time")
            else:
                self.adaptive_learning = None
            
            # 11. Trading Engine with adaptive components
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
        last_learning_update = datetime.now()
        
        while self.running:
            try:
                loop_count += 1
                
                # 1. Get market data
                symbols = self.config.get('trading', {}).get('symbols', [])
                market_data_dict = {}
                bars_dict = {}
                
                for symbol in symbols:
                    data = self.market_data.get_latest_data(symbol)
                    if data is not None:
                        market_data_dict[symbol] = data
                        # Get bars for technical analysis from multiple timeframes
                        bars_m1 = self.market_data.get_bars(symbol, mt5.TIMEFRAME_M1, 100)
                        bars_m5 = self.market_data.get_bars(symbol, mt5.TIMEFRAME_M5, 100)
                        bars_h1 = self.market_data.get_bars(symbol, mt5.TIMEFRAME_H1, 100)
                        bars_h4 = self.market_data.get_bars(symbol, mt5.TIMEFRAME_H4, 100)
                        bars_d1 = self.market_data.get_bars(symbol, mt5.TIMEFRAME_D1, 100)
                        
                        if bars_h1 is not None:  # Use H1 as primary
                            bars_dict[symbol] = {
                                'M1': bars_m1,
                                'M5': bars_m5, 
                                'H1': bars_h1,
                                'H4': bars_h4,
                                'D1': bars_d1
                            }
                
                if not market_data_dict:
                    await asyncio.sleep(10)
                    continue
                
                # 2. Collect fundamental data (with caching)
                fundamental_data = self.fundamental_collector.collect()
                
                # 3. Generate trading signals with adaptive weights
                signals = []
                
                for symbol, market_data in market_data_dict.items():
                    # Get adaptive parameters if learning is enabled
                    if self.adaptive_learning:
                        signal_weights = self.adaptive_learning.get_current_weights()
                        adaptive_params = self.adaptive_learning.get_adaptive_parameters()
                        adaptive_params['risk_multiplier'] = 1.0  # Force fixed risk
                    else:
                        # Adjusted weights for testing (more weight to technical)
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
                    if bars is None or not isinstance(bars, dict) or 'H1' not in bars:
                        continue
                        
                    technical_signals = await self.technical_analyzer.analyze(
                        symbol, 
                        bars
                    )
                    technical_score = technical_signals.get('overall_score', 0.5)
                    
                    # ML prediction
                    ml_prediction = await self.ml_predictor.predict(symbol, bars, technical_signals)
                    
                    # Sentiment analysis
                    sentiment_result = await self.sentiment_analyzer.analyze_sentiment(symbol)
                    sentiment_score = sentiment_result.get('overall_score', 0.5)
                    
                    # Calculate weighted signal with adaptive weights
                    signal_strength = (
                        signal_weights.get('technical_score', 0.25) * technical_score +
                        signal_weights.get('ml_prediction', 0.30) * ml_prediction.get('probability', 0) +
                        signal_weights.get('sentiment_score', 0.20) * sentiment_score +
                        signal_weights.get('fundamental_score', 0.15) * fundamental_data.get(symbol, {}).get('score', 0.5) +
                        signal_weights.get('support_resistance', 0.10) * market_data.get('sr_score', 0.5)
                    )
                    
                    # Debug: Log all signal strengths
                    self.logger.debug(
                        f"{symbol} signal: strength={signal_strength:.3f} (Tech:{technical_score:.3f}, ML:{ml_prediction.get('probability', 0):.3f}, Sent:{sentiment_score:.3f})"
                    )
                    
                    # Apply adaptive minimum threshold
                    min_threshold = 0.5  # Temporarily override adaptive threshold for testing
                    self.logger.debug(f"{symbol}: threshold={min_threshold:.3f}, strength={signal_strength:.3f}")
                    
                    if signal_strength > min_threshold:
                        # Get current market price for entry
                        current_data = market_data_dict.get(symbol, {})
                        entry_price = current_data.get('ask') if ml_prediction.get('direction') == 1 else current_data.get('bid', 0)
                        
                        # Skip if no valid price data
                        if entry_price <= 0:
                            self.logger.debug(f"{symbol}: Skipping - no valid entry price")
                            continue
                        
                        self.logger.debug(f"{symbol}: Processing signal - entry_price={entry_price}, direction={ml_prediction.get('direction')}")
                        
                        # Calculate stop loss using ATR (more sophisticated than fixed percentage)
                        atr_value = technical_signals.get('atr', {}).get('value', 0)
                        if atr_value > 0:
                            self.logger.debug(f"{symbol}: ATR available ({atr_value:.5f}) - proceeding with signal")
                            # Use adaptive ATR multiplier for stop loss distance
                            # Use higher multipliers for precious metals to meet broker requirements
                            base_multiplier = adaptive_params.get('stop_loss_atr_multiplier', 3.0)  # Increased from 2.0
                            if 'XAU' in symbol or 'XAG' in symbol:
                                # Precious metals need higher multipliers due to lower ATR on M1 timeframe
                                if 'XAUUSD' in symbol:
                                    # XAUUSD ATR is very wide, reduce multiplier to allow proper position sizing
                                    sl_atr_multiplier = 0.3  # Much lower multiplier for XAUUSD
                                else:
                                    sl_atr_multiplier = max(base_multiplier, 4.0)  # Keep 4.0 for XAGUSD
                            else:
                                sl_atr_multiplier = base_multiplier
                            
                            stop_loss_distance = atr_value * sl_atr_multiplier
                            self.logger.debug(f"{symbol} ATR-based stop loss: ATR={atr_value:.5f}, multiplier={sl_atr_multiplier:.1f}, distance={stop_loss_distance:.5f}")
                        else:
                            # Fallback to 2% if ATR not available
                            stop_loss_distance = entry_price * 0.02
                            self.logger.debug(f"{symbol} fallback stop loss: 2% of entry = {stop_loss_distance:.5f}")
                        
                        # Ensure minimum stop distance matches position sizing requirements
                        min_sl_pips = self.config.get('risk_management', {}).get('minimum_sl_pips', 25)
                        # Use higher minimum SL for metals due to higher volatility
                        if 'XAU' in symbol or 'GOLD' in symbol:
                            min_sl_pips = max(min_sl_pips, 200)  # Minimum 200 pips for gold
                        elif 'XAG' in symbol:
                            min_sl_pips = max(min_sl_pips, 100)  # Minimum 100 pips for silver (reduced for tradability)
                        
                        # Convert minimum pips to distance
                        if 'JPY' in symbol:
                            min_stop_distance = min_sl_pips * 0.01   # JPY pairs: 1 pip = 0.01
                        elif 'XAG' in symbol or 'XAU' in symbol:
                            min_stop_distance = min_sl_pips * 0.01   # Metals: 1 pip = 0.01 (same as JPY)
                        else:
                            min_stop_distance = min_sl_pips * 0.0001 # Forex: 1 pip = 0.0001
                        
                        original_distance = stop_loss_distance
                        stop_loss_distance = max(stop_loss_distance, min_stop_distance)
                        if original_distance != stop_loss_distance:
                            self.logger.info(f"{symbol} stop loss adjusted from {original_distance:.5f} to {stop_loss_distance:.5f} (min {min_sl_pips} pips = {min_stop_distance:.5f})")
                        
                        if ml_prediction.get('direction') == 1:  # BUY
                            stop_loss = entry_price - stop_loss_distance
                        else:  # SELL
                            stop_loss = entry_price + stop_loss_distance
                        
                        # Calculate take profit using adaptive ATR multiplier
                        take_profit = None
                        if atr_value > 0:
                            # Use appropriate TP multiplier for precious metals to ensure proper risk-reward
                            base_tp_multiplier = adaptive_params.get('take_profit_atr_multiplier', 6.0)
                            if 'XAU' in symbol or 'XAG' in symbol or 'GOLD' in symbol:
                                tp_atr_multiplier = max(base_tp_multiplier, 4.0)  # Minimum 4.0x ATR for metals to ensure proper RR
                            else:
                                tp_atr_multiplier = base_tp_multiplier
                            
                            take_profit_distance = atr_value * tp_atr_multiplier
                            
                            # Ensure minimum take profit distance for 1:3 reward ratio
                            min_tp_distance = min_stop_distance * 3  # 3x the minimum stop loss distance
                            original_tp_distance = take_profit_distance
                            take_profit_distance = max(take_profit_distance, min_tp_distance)
                            
                            if original_tp_distance != take_profit_distance:
                                self.logger.info(f"{symbol} take profit adjusted from {original_tp_distance:.5f} to {take_profit_distance:.5f} (min {min_tp_distance:.5f} for 1:3 ratio)")
                            
                            if ml_prediction.get('direction') == 1:  # BUY
                                take_profit = entry_price + take_profit_distance
                            else:  # SELL
                                take_profit = entry_price - take_profit_distance
                            self.logger.debug(f"{symbol} take profit: ATR={atr_value:.5f}, multiplier={tp_atr_multiplier:.1f}, distance={take_profit_distance:.5f}, level={take_profit:.5f}")
                        else:
                            # Fallback take profit: 6% of entry price (3x the 2% stop loss fallback)
                            take_profit_distance = entry_price * 0.06
                            # Ensure minimum take profit distance
                            min_tp_distance = min_stop_distance * 3
                            take_profit_distance = max(take_profit_distance, min_tp_distance)
                            
                            if ml_prediction.get('direction') == 1:  # BUY
                                take_profit = entry_price + take_profit_distance
                            else:  # SELL
                                take_profit = entry_price - take_profit_distance
                            self.logger.debug(f"{symbol} fallback take profit: 6% of entry = {take_profit_distance:.5f}, level={take_profit:.5f}")
                        
                        # Format take profit for logging
                        tp_display = f"{take_profit:.5f}" if take_profit is not None else "None"
                        
                        # Validate risk-reward ratio before proceeding
                        if stop_loss is not None and take_profit is not None:
                            risk_distance = abs(stop_loss - entry_price)
                            reward_distance = abs(take_profit - entry_price)
                            actual_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
                            action = 'BUY' if ml_prediction.get('direction') == 1 else 'SELL'
                            
                            if actual_ratio < 2.9:
                                self.logger.debug(f"{symbol} {action} rejected: insufficient reward ratio {actual_ratio:.2f}:1 (required: 2.9:1)")
                                continue  # Skip this trade
                            
                            self.logger.debug(f"{symbol} {action} validated: {actual_ratio:.1f}:1 risk-reward ratio")
                        
                        signal = {
                            'symbol': symbol,
                            'action': 'BUY' if ml_prediction.get('direction') == 1 else 'SELL',
                            'strength': signal_strength,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'ml_score': ml_prediction.get('probability', 0),
                            'technical_score': technical_score,
                            'sentiment_score': sentiment_score,
                            'timestamp': datetime.now(),
                            'adaptive_params': adaptive_params
                        }
                        signals.append(signal)
                        
                        self.logger.info(
                            f"Signal generated for {symbol}: {signal['action']} "
                            f"(strength: {signal_strength:.3f}, threshold: {min_threshold:.3f}) "
                            f"Entry: {entry_price:.5f}, SL: {stop_loss:.5f} ({sl_atr_multiplier:.1f}x ATR), TP: {tp_display} ({tp_atr_multiplier:.1f}x ATR)"
                        )
                
                # 4. Execute trades with risk management
                for signal in signals:
                    # Validate signal has required trading parameters
                    self.logger.info(f"Processing signal for {signal['symbol']}: action={signal['action']}, strength={signal.get('strength', 0):.3f}, entry={signal.get('entry_price', 'None')}, stop={signal.get('stop_loss', 'None')}, tp={signal.get('take_profit', 'None')}")
                    if not signal.get('entry_price') or not signal.get('stop_loss'):
                        self.logger.warning(
                            f"Skipping signal for {signal['symbol']} - missing entry/stop loss data "
                            f"(entry: {signal.get('entry_price', 'None')}, stop: {signal.get('stop_loss', 'None')})"
                        )
                        continue

                    # Update risk metrics before checking limits
                    # Note: New RiskManager doesn't have update_metrics method
                    
                    # Check risk limits with adaptive multiplier
                    risk_check = self.risk_manager.can_trade(signal['symbol'])
                    self.logger.info(f"Risk check for {signal['symbol']} {signal['action']}: {risk_check}")
                    if risk_check:
                        # Check free margin percentage
                        account_info = mt5.account_info()
                        if account_info and account_info.margin_free < 0.20 * account_info.equity:
                            self.logger.warning(f"Free margin ${account_info.margin_free:.2f} is less than 20% of equity ${account_info.equity:.2f}, skipping trade for {signal['symbol']}")
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
                            if 'XAU' in signal['symbol'] or 'XAG' in signal['symbol'] or 'GOLD' in signal['symbol']:
                                pip_size = point * 10  # Metals: 1 pip = 10 points (0.1 for 2-digit symbols)
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
                        
                        # Enforce minimum stop loss distance for consistent risk calculation
                        min_sl_pips = self.config.get('risk_management', {}).get('minimum_sl_pips', 25)
                        
                        # Use appropriate minimum SL for metals with $50 risk
                        if 'XAU' in signal['symbol'] or 'XAG' in signal['symbol'] or 'GOLD' in signal['symbol']:
                            # With $50 risk and 0.01 min lot, we can use reasonable SL
                            # 0.01 lots * pip_value * pips = $50
                            # For XAUUSD: pip_value ≈ $10, so pips = 50/10 = 5, but we need minimum for stability
                            min_sl_pips = max(min_sl_pips, 50)  # At least 50 pips for metals with $50 risk
                        
                        stop_loss_pips = max(stop_loss_pips, min_sl_pips)
                        
                        # Check if minimum lot size would exceed risk limit for volatile symbols
                        min_lot_size = self.config.get('trading', {}).get('min_lot_size', 0.01)
                        if 'XAU' in signal['symbol'] or 'XAG' in signal['symbol'] or 'GOLD' in signal['symbol']:
                            # Calculate risk with minimum lot size
                            min_lot_risk = self.risk_manager.calculate_risk_for_lot_size(
                                signal['symbol'], min_lot_size, stop_loss_pips
                            )
                            
                            max_risk_limit = self.config.get('trading', {}).get('risk_per_trade', 50.0)
                            if min_lot_risk > max_risk_limit * 1.05:  # Allow 5% tolerance
                                self.logger.warning(
                                    f"Skipping {signal['symbol']} - even minimum lot size {min_lot_size} "
                                    f"would risk ${min_lot_risk:.2f} (limit: ${max_risk_limit:.2f})"
                                )
                                continue
                        
                        # Calculate position size with risk amount adjusted for metals
                        base_risk_amount = 50  # Fixed $50 risk per trade for forex
                        if 'XAU' in signal['symbol'] or 'XAG' in signal['symbol'] or 'GOLD' in signal['symbol']:
                            risk_amount = base_risk_amount  # Keep $50 risk for metals
                            self.logger.debug(f"Metal detected: {signal['symbol']}, using $50 risk")
                        else:
                            risk_amount = base_risk_amount
                            self.logger.debug(f"Forex detected: {signal['symbol']}, using $50 risk")
                        
                        position_size = self.risk_manager.calculate_position_size(
                            signal['symbol'],
                            stop_loss_pips,
                            risk_amount
                        )
                        
                        # Debug logging
                        self.logger.info(f"Position sizing for {signal['symbol']}: risk_amount=${risk_amount}, stop_pips={stop_loss_pips}, lot_size={position_size:.4f}")
                        
                        # Execute trade with stop loss
                        trade_result = await self.trading_engine.place_order(
                            signal['symbol'],
                            signal['action'].lower(),
                            position_size,
                            stop_loss=signal.get('stop_loss'),
                            take_profit=signal.get('take_profit'),  # Now using calculated take profit
                            comment=f"FX-Ai {signal['action']} {signal_strength:.3f}"
                        )
                        
                        if trade_result.get('success', False):
                            self.session_stats['total_trades'] += 1
                            
                            # Record trade for learning
                            if self.adaptive_learning:
                                trade_data = {
                                    'timestamp': datetime.now(),
                                    'symbol': signal['symbol'],
                                    'direction': signal['action'],
                                    'entry_price': trade_result.get('price', 0),
                                    'signal_strength': signal['strength'],
                                    'ml_score': signal['ml_score'],
                                    'technical_score': signal['technical_score'],
                                    'sentiment_score': signal['sentiment_score'],
                                    'model_version': self.ml_predictor.get_model_version(signal['symbol'])
                                }
                                
                                # Start monitoring thread for this trade
                                threading.Thread(
                                    target=self.monitor_trade,
                                    args=(trade_result.get('order', 0), trade_data),
                                    daemon=True
                                ).start()
                
                # 5. Monitor and update positions
                for symbol in symbols:
                    await self.trading_engine.manage_positions(symbol)
                
                # 6. Check for model retraining trigger (every hour)
                if self.adaptive_learning and loop_count % 360 == 0:  # Every hour (assuming 10s loop)
                    self.logger.info("Triggering periodic model evaluation...")
                    
                    # Get performance summary
                    performance = self.adaptive_learning.get_performance_summary()
                    
                    self.logger.info(
                        f"Performance Update - Win Rate: {performance['performance_metrics'].get('overall_win_rate', 0):.2%}, "
                        f"Total Trades: {performance['total_trades']}"
                    )
                    
                    # Log adapted weights
                    self.logger.info(f"Current Signal Weights: {json.dumps(performance['signal_weights'], indent=2)}")
                    self.logger.info(f"Adaptive Parameters: {json.dumps(performance['adaptive_params'], indent=2)}")
                    
                    self.session_stats['models_retrained'] += 1
                
                # 7. Check for day trading closure
                if self.config.get('trading', {}).get('day_trading_only', True):
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
            # Get adaptive parameters for time-based exits
            adaptive_params = self.adaptive_learning.get_adaptive_parameters() if self.adaptive_learning else {}
            max_holding_minutes = adaptive_params.get('max_holding_minutes', 480)  # Default 8 hours
            optimal_holding_hours = adaptive_params.get('optimal_holding_hours', 4.0)
            optimal_holding_minutes = optimal_holding_hours * 60

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

                        profit_pct = (profit_pips / 100)  # Simplified calculation

                        # Update trade data
                        trade_data['exit_price'] = exit_price
                        trade_data['profit'] = history.get('profit', 0)
                        trade_data['profit_pct'] = profit_pct
                        trade_data['duration_minutes'] = (datetime.now() - trade_data['timestamp']).seconds // 60
                        trade_data['volume'] = history.get('volume', 0)

                        # Record trade result for cooldown management
                        actual_profit = history.get('profit', 0)
                        self.risk_manager.record_trade_result(trade_data['symbol'], actual_profit)

                        # Record for learning
                        if self.adaptive_learning:
                            self.adaptive_learning.record_trade(trade_data)

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
                    holding_minutes = (datetime.now() - trade_data['timestamp']).seconds // 60

                    # Check maximum holding time
                    if holding_minutes >= max_holding_minutes:
                        self.logger.info(f"Closing {trade_data['symbol']} position due to max holding time ({max_holding_minutes} minutes)")
                        try:
                            # Close position at market
                            close_result = self.trading_engine.close_position(ticket, trade_data['symbol'])
                            if close_result and close_result.get('success'):
                                self.logger.info(f"Successfully closed position for max time exit")
                            else:
                                self.logger.warning(f"Failed to close position for max time exit")
                        except Exception as e:
                            self.logger.error(f"Error closing position for max time: {e}")
                        break

                    # Check optimal holding time with profit
                    elif holding_minutes >= optimal_holding_minutes:
                        # Check if position is in profit
                        current_price = position.get('price_current', 0)
                        entry_price = trade_data['entry_price']

                        if trade_data['direction'] == 'BUY' and current_price > entry_price:
                            profit_pct = ((current_price - entry_price) / entry_price) * 100
                            if profit_pct > 0.1:  # At least 0.1% profit
                                self.logger.info(f"Closing {trade_data['symbol']} position at optimal time ({optimal_holding_minutes} min) with {profit_pct:.2f}% profit")
                                try:
                                    close_result = self.trading_engine.close_position(ticket, trade_data['symbol'])
                                    if close_result and close_result.get('success'):
                                        self.logger.info(f"Successfully closed position for optimal time exit")
                                    else:
                                        self.logger.warning(f"Failed to close position for optimal time exit")
                                except Exception as e:
                                    self.logger.error(f"Error closing position for optimal time: {e}")
                                break
                        elif trade_data['direction'] == 'SELL' and current_price < entry_price:
                            profit_pct = ((entry_price - current_price) / entry_price) * 100
                            if profit_pct > 0.1:  # At least 0.1% profit
                                self.logger.info(f"Closing {trade_data['symbol']} position at optimal time ({optimal_holding_minutes} min) with {profit_pct:.2f}% profit")
                                try:
                                    close_result = self.trading_engine.close_position(ticket, trade_data['symbol'])
                                    if close_result and close_result.get('success'):
                                        self.logger.info(f"Successfully closed position for optimal time exit")
                                    else:
                                        self.logger.warning(f"Failed to close position for optimal time exit")
                                except Exception as e:
                                    self.logger.error(f"Error closing position for optimal time: {e}")
                                break

        except Exception as e:
            self.logger.error(f"Error monitoring trade {ticket}: {e}")
    
    async def check_day_trading_closure(self):
        """Check if positions should be closed for day trading - FIXED"""
        try:
            if not self.config.get('trading', {}).get('day_trading_only', True):
                return

            current_time = datetime.now().time()
            close_time = time(21, 0)  # 9 PM

            # Only close once per day - check if we haven't already closed today
            if current_time >= close_time:
                # Check if we already closed positions today
                today = datetime.now().date()
                if hasattr(self, '_last_closure_date') and self._last_closure_date == today:
                    return  # Already closed today

                self.logger.info("Day trading hours ending - closing all positions")
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
                        self.logger.warning("close_all_positions method not found on trading_engine")
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
                        loop.create_task(loop.run_in_executor(None, close_method))
                else:
                    self.logger.warning('No close_all_positions method on trading_engine')
            except Exception as e:
                self.logger.error(f"Error scheduling close_all_positions on shutdown: {e}")
        
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
        self.logger.info(f"Winning Trades: {self.session_stats['winning_trades']}")
        self.logger.info(f"Losing Trades: {self.session_stats['losing_trades']}")
        
        if self.session_stats['total_trades'] > 0:
            win_rate = self.session_stats['winning_trades'] / self.session_stats['total_trades'] * 100
            self.logger.info(f"Win Rate: {win_rate:.1f}%")
        
        self.logger.info(f"Total Profit: {self.session_stats['total_profit']:.2f}%")
        
        if self.learning_enabled:
            self.logger.info(f"Models Retrained: {self.session_stats['models_retrained']}")
            self.logger.info(f"Parameters Optimized: {self.session_stats['parameters_optimized']}")
            
            if self.adaptive_learning:
                performance = self.adaptive_learning.get_performance_summary()
                self.logger.info(f"Final Signal Weights: {json.dumps(performance['signal_weights'], indent=2)}")
                self.logger.info(f"Final Adaptive Parameters: {json.dumps(performance['adaptive_params'], indent=2)}")
        
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
    """Main entry point"""
    # Create and run application
    app = FXAiApplication()
    
    # Run async main
    asyncio.run(app.run())

if __name__ == "__main__":
    main()
