"""
FX-Ai Adaptive Learning Manager
Implements continuous learning and strategy improvement using modular components
"""

import os
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
from collections import deque

# Import new modular components
from .learning_database import LearningDatabase
from .learning_scheduler import LearningScheduler
from .trade_analyzer import TradeAnalyzer
from .learning_algorithms import LearningAlgorithms

logger = logging.getLogger(__name__)


class AdaptiveLearningManager:
    """
    Manages continuous learning and adaptation for FX-Ai using modular components
    - Database operations via LearningDatabase
    - Scheduling via LearningScheduler
    - Trade analysis via TradeAnalyzer
    - Learning algorithms via LearningAlgorithms
    """

    def __init__(
            self,
            config: dict,
            ml_predictor=None,
            backtest_engine=None,
            risk_manager=None,
            mt5_connector=None):
        """Initialize the adaptive learning system with modular components"""
        self.config = config
        self.ml_predictor = ml_predictor
        self.backtest_engine = backtest_engine
        self.risk_manager = risk_manager
        self.mt5 = mt5_connector
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Initialize modular components
        self.db = LearningDatabase(config=config)
        self.scheduler = LearningScheduler()
        self.analyzer = TradeAnalyzer()
        self.algorithms = LearningAlgorithms(config)

        # Learning configuration
        self.retrain_interval = config.get(
            'adaptive_learning', {}).get(
            'retrain_interval_hours', 24)
        self.min_trades_for_update = config.get(
            'adaptive_learning', {}).get(
            'min_trades_for_update', 50)

        # Performance tracking (legacy compatibility)
        self.trade_history = deque(maxlen=1000)
        self.performance_metrics = {}
        self.model_versions = {}
        self.parameter_history = []

        # Signal weights (will be adapted based on performance)
        self.signal_weights = config.get('adaptive_learning', {}).get(
            'signal_weights', {
                'ml_prediction': 0.30,
                'technical_score': 0.25,
                'sentiment_score': 0.20,
                'fundamental_score': 0.15,
                'support_resistance': 0.10
            }
        )

        # Trading parameters (will be optimized)
        self.adaptive_params = config.get('adaptive_learning', {}).get(
            'adaptive_params', {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'min_signal_strength': 0.6,
                'max_correlation': 0.8,
                'risk_multiplier': 1.0,
                'trailing_stop_distance': 20,
                'stop_loss_atr_multiplier': 2.0,  # ATR multiplier for SL
                'take_profit_atr_multiplier': 9.0,  # ATR multiplier for take profit (1:3 ratio)
                'max_holding_minutes': 480,  # Maximum holding time in minutes (8 hours)
                'min_holding_minutes': 15,   # Minimum holding time in minutes
                'optimal_holding_hours': 4.0  # Target optimal holding period
            }
        )

        # Force reset risk multiplier to prevent inflation
        self.adaptive_params['risk_multiplier'] = 1.0

        # Register scheduler callbacks to use this manager's methods
        self._register_scheduler_callbacks()

        # Schedule periodic tasks
        self.scheduler.schedule_tasks()

        # Start the continuous learning thread
        self.scheduler.restart_learning_thread()

        # Initialize database
        self.db.init_database()

        logger.info("Adaptive Learning Manager initialized with modular components")

        # Initialize smart defaults for known forex patterns
        self._load_smart_defaults()

    def _load_smart_defaults(self):
        """Load known forex patterns to bootstrap learning system"""
        self.smart_defaults = {
            "day_adjustments": {
                "monday": {
                    "hours": "08:00-11:00",
                    "threshold_add": 0.080,
                    "confidence": 0.7,
                    "reason": "Monday morning gap risk"
                },
                "friday": {
                    "hours": "18:00-23:00",
                    "threshold_add": 0.100,
                    "confidence": 0.8,
                    "reason": "Friday afternoon position closing"
                },
                "wednesday": {
                    "hours": "12:00-15:00",
                    "threshold_add": -0.030,
                    "confidence": 0.6,
                    "reason": "Mid-week liquidity sweet spot"
                }
            },

            "session_preferences": self.config.get('adaptive_learning_preferences', {}).get('session_preferences', {
                "tokyo_sydney": {
                    "preferred_pairs": ["AUDUSD", "NZDUSD", "AUDJPY", "NZDJPY", "AUDNZD", "USDJPY"],
                    "threshold_multiplier": 1.0,
                    "reason": "Asian pairs during Asian sessions"
                },
                "london": {
                    "preferred_pairs": ["EURUSD", "GBPUSD", "EURGBP", "EURCHF", "EURJPY", "GBPJPY"],
                    "threshold_multiplier": 0.85,
                    "reason": "European pairs during London session"
                },
                "new_york": {
                    "preferred_pairs": ["EURUSD", "GBPUSD", "USDCAD", "USDJPY"],
                    "threshold_multiplier": 0.90,
                    "reason": "Major pairs during NY session"
                }
            }),

            "time_based_patterns": {
                "market_open_gaps": {
                    "sessions": ["tokyo_sydney", "london"],
                    "hours_after_open": 2,
                    "threshold_add": 0.050,
                    "reason": "Avoid gap-related volatility"
                },
                "session_overlap": {
                    "threshold_add": -0.040,
                    "reason": "Increased liquidity during overlaps"
                },
                "end_of_day": {
                    "hours_before_close": 2,
                    "threshold_add": 0.060,
                    "reason": "Position closing pressure"
                }
            }
        }

        logger.info("Smart defaults loaded for known forex patterns")

    def get_smart_threshold_adjustment(self, symbol: str, session: str, current_time: datetime, base_threshold: float) -> tuple:
        """Get threshold adjustment based on known forex patterns

        Returns:
            tuple: (adjusted_threshold, reason, confidence)
        """
        adjustment = 0.0
        reasons = []
        confidence = 0.5  # Base confidence

        # Day of week adjustments
        day_name = current_time.strftime('%A').lower()
        if day_name in self.smart_defaults["day_adjustments"]:
            day_pattern = self.smart_defaults["day_adjustments"][day_name]
            # Check if current time is within the specified hours
            if self._time_in_range(current_time, day_pattern["hours"]):
                adjustment += day_pattern["threshold_add"]
                reasons.append(day_pattern["reason"])
                confidence = max(confidence, day_pattern["confidence"])

        # Session preference adjustments
        if session in self.smart_defaults["session_preferences"]:
            session_pref = self.smart_defaults["session_preferences"][session]
            if symbol in session_pref["preferred_pairs"]:
                multiplier = session_pref["threshold_multiplier"]
                adjustment *= multiplier
                reasons.append(session_pref["reason"])
                confidence = max(confidence, 0.7)

        # Time-based patterns
        for pattern_name, pattern in self.smart_defaults["time_based_patterns"].items():
            if pattern_name == "market_open_gaps":
                if session in pattern["sessions"]:
                    # Check if we're within X hours of session start
                    session_start = self._get_session_start_time(session, current_time.date())
                    if session_start and (current_time - session_start).total_seconds() < (pattern["hours_after_open"] * 3600):
                        adjustment += pattern["threshold_add"]
                        reasons.append(pattern["reason"])
                        confidence = max(confidence, 0.6)

            elif pattern_name == "end_of_day":
                # Check if we're within X hours of session end
                session_end = self._get_session_end_time(session, current_time.date())
                if session_end and (session_end - current_time).total_seconds() < (pattern["hours_before_close"] * 3600):
                    adjustment += pattern["threshold_add"]
                    reasons.append(pattern["reason"])
                    confidence = max(confidence, 0.6)

        adjusted_threshold = base_threshold + adjustment
        reason_str = "; ".join(reasons) if reasons else "No smart adjustments"

        return adjusted_threshold, reason_str, confidence

    def _time_in_range(self, check_time: datetime, time_range: str) -> bool:
        """Check if time is within a range like '08:00-11:00'"""
        try:
            start_str, end_str = time_range.split('-')
            start_time = datetime.strptime(start_str, '%H:%M').time()
            end_time = datetime.strptime(end_str, '%H:%M').time()

            check_time_only = check_time.time()
            return start_time <= check_time_only <= end_time
        except:
            return False

    def _get_session_start_time(self, session: str, date) -> Optional[datetime]:
        """Get session start time (simplified)"""
        # This is a simplified version - in production you'd use proper session times
        session_starts = {
            "tokyo_sydney": 22,  # 10:00 PM UTC (6:00 AM Tokyo)
            "london": 8,         # 8:00 AM UTC
            "new_york": 13       # 1:00 PM UTC
        }
        if session in session_starts:
            return datetime.combine(date, datetime.min.time().replace(hour=session_starts[session]))
        return None

    def _get_session_end_time(self, session: str, date) -> Optional[datetime]:
        """Get session end time (simplified)"""
        session_ends = {
            "tokyo_sydney": 7,   # 7:00 AM UTC
            "london": 16,        # 4:00 PM UTC
            "new_york": 21       # 9:00 PM UTC
        }
        if session in session_ends:
            return datetime.combine(date, datetime.min.time().replace(hour=session_ends[session]))
        return None

    def _register_scheduler_callbacks(self):
        """Register callbacks for scheduled tasks"""
        self.scheduler.register_task_callback('retrain_models', self.retrain_models)
        self.scheduler.register_task_callback('evaluate_performance', self.evaluate_performance)
        self.scheduler.register_task_callback('optimize_parameters', self.optimize_parameters)
        self.scheduler.register_task_callback('adjust_signal_weights', self.adjust_signal_weights)
        self.scheduler.register_task_callback('update_all_symbol_holding_times', self.analyzer.update_all_symbol_holding_times)
        self.scheduler.register_task_callback('analyze_entry_timing', self.analyzer.analyze_entry_timing)
        self.scheduler.register_task_callback('optimize_symbol_sl_tp', self.optimize_symbol_sl_tp)
        self.scheduler.register_task_callback('update_entry_filters', self.update_entry_filters)
        self.scheduler.register_task_callback('optimize_technical_indicators', self.optimize_technical_indicators)
        self.scheduler.register_task_callback('optimize_fundamental_weights', self.optimize_fundamental_weights)
        self.scheduler.register_task_callback('analyze_economic_calendar_impact', self.analyze_economic_calendar_impact)
        self.scheduler.register_task_callback('analyze_interest_rate_impact', self.analyze_interest_rate_impact)
        self.scheduler.register_task_callback('optimize_sentiment_parameters', self.optimize_sentiment_parameters)
        self.scheduler.register_task_callback('analyze_adjustment_performance', self.analyze_adjustment_performance)
        self.scheduler.register_task_callback('clean_old_data', self.db.clean_old_data)

    # Database operations delegated to LearningDatabase
    def init_database(self):
        """Initialize SQLite database for trade and performance tracking"""
        return self.db.init_database()

    def load_daily_trade_counts(self) -> dict:
        """Load daily trade counts from database"""
        return self.db.load_daily_trade_counts()

    def save_daily_trade_count(self, symbol: str, trade_date: str, count: int):
        """Save daily trade count to database"""
        return self.db.save_daily_trade_count(symbol, trade_date, count)

    def record_trade(self, trade_data: dict):
        """Record a trade in the database"""
        return self.db.record_trade(trade_data)

    def record_open_trade(self, trade_data: dict):
        """Record an open trade in the database for monitoring"""
        # For open trades, set exit_price to None and other fields accordingly
        open_trade_data = trade_data.copy()
        open_trade_data['exit_price'] = None
        open_trade_data['closure_reason'] = None
        open_trade_data['forced_closure'] = 0
        return self.db.record_trade(open_trade_data)

    def record_trade_closure(self, ticket: int, reason: str, entry_price: float, exit_price: float, symbol: str = None):
        """Record trade closure in database"""
        return self.db.record_trade_closure(ticket, reason, entry_price, exit_price, symbol)

    def record_position_adjustment(self, ticket: int, symbol: str, old_sl: float, old_tp: float, new_sl: float, new_tp: float, reason: str):
        """Record position SL/TP adjustment in database"""
        return self.db.record_position_adjustment(ticket, symbol, old_sl, old_tp, new_sl, new_tp, reason)

    def get_optimal_entry_timing(self, symbol: str) -> dict:
        """Get optimal entry timing for a symbol"""
        return self.db.get_optimal_entry_timing(symbol)

    # Scheduling operations delegated to LearningScheduler
    def schedule_tasks(self):
        """Schedule periodic learning tasks"""
        return self.scheduler.schedule_tasks()

    def run_continuous_learning(self):
        """Run continuous learning loop"""
        return self.scheduler.run_continuous_learning()

    def _check_scheduler_health(self):
        """Check scheduler health"""
        return self.scheduler._check_scheduler_health()

    def restart_learning_thread(self):
        """Restart learning thread"""
        return self.scheduler.restart_learning_thread()

    def get_thread_status(self):
        """Get thread status"""
        return self.scheduler.get_thread_status()

    # Trade analysis operations delegated to TradeAnalyzer
    def analyze_trade_timing_patterns(self, trade_data: dict):
        """Analyze trade timing patterns"""
        return self.analyzer.analyze_trade_timing_patterns(trade_data)

    # Learning algorithms operations delegated to LearningAlgorithms
    def retrain_models(self):
        """Retrain ML models"""
        return self.algorithms.retrain_models()

    def evaluate_performance(self):
        """Evaluate performance"""
        return self.algorithms.evaluate_performance()

    def optimize_parameters(self):
        """Optimize parameters"""
        return self.algorithms.optimize_parameters()

    def adjust_signal_weights(self):
        """Adjust signal weights"""
        return self.algorithms.adjust_signal_weights()

    def calculate_signal_strength(self, technical_score: float, fundamental_score: float, sentiment_score: float, ml_score: float = 0.0) -> float:
        """
        Calculate overall signal strength using adaptive weights

        Args:
            technical_score: Technical analysis score (0-1)
            fundamental_score: Fundamental analysis score (0-1)
            sentiment_score: Sentiment analysis score (0-1)
            ml_score: ML prediction score (0-1), defaults to 0.0

        Returns:
            float: Combined signal strength (0-1)
        """
        try:
            # Use adaptive signal weights if available, otherwise use defaults
            weights = getattr(self, 'signal_weights', {
                'technical_score': 0.4,
                'fundamental_score': 0.3,
                'sentiment_score': 0.3,
                'ml_prediction': 0.0  # Default to 0 if not set
            })

            # Calculate weighted signal strength
            signal_strength = (
                technical_score * weights.get('technical_score', 0.4) +
                fundamental_score * weights.get('fundamental_score', 0.3) +
                sentiment_score * weights.get('sentiment_score', 0.3) +
                ml_score * weights.get('ml_prediction', 0.0)
            )

            # Ensure result is in 0-1 range
            return max(0.0, min(1.0, signal_strength))

        except Exception as e:
            # Fallback to simple average if anything goes wrong
            self.logger.warning(f"Error calculating adaptive signal strength: {e}, using simple average")
            return (technical_score + fundamental_score + sentiment_score + ml_score) / 4.0

    def optimize_symbol_sl_tp(self, symbol: str):
        """Optimize SL/TP for symbol"""
        return self.algorithms.optimize_symbol_sl_tp(symbol)

    def update_entry_filters(self):
        """Update entry filters"""
        return self.algorithms.update_entry_filters()

    def optimize_technical_indicators(self):
        """Optimize technical indicators"""
        return self.algorithms.optimize_technical_indicators()

    def optimize_fundamental_weights(self):
        """Optimize fundamental weights"""
        return self.algorithms.optimize_fundamental_weights()

    def analyze_economic_calendar_impact(self):
        """Analyze economic calendar impact"""
        return self.algorithms.analyze_economic_calendar_impact()

    def analyze_interest_rate_impact(self):
        """Analyze interest rate impact"""
        return self.algorithms.analyze_interest_rate_impact()

    def optimize_sentiment_parameters(self):
        """Optimize sentiment parameters"""
        return self.algorithms.optimize_sentiment_parameters()

    def analyze_adjustment_performance(self):
        """Analyze adjustment performance"""
        return self.algorithms.analyze_adjustment_performance()

    # Legacy compatibility methods - delegate to appropriate modules
    def get_adaptive_params(self) -> dict:
        """Get current adaptive parameters"""
        return self.adaptive_params

    def get_signal_weights(self) -> dict:
        """Get current signal weights"""
        return self.signal_weights

    def get_current_weights(self) -> dict:
        """Get current signal weights for trading decisions"""
        return self.algorithms.get_current_weights()

    def get_adaptive_parameters(self) -> dict:
        """Get current adaptive parameters"""
        return self.algorithms.get_adaptive_parameters()

    def get_symbol_optimal_holding_time(self, symbol: str) -> dict:
        """Get optimal holding time for a symbol"""
        return self.analyzer.get_symbol_optimal_holding_time(symbol)

    def save_signal_weights(self):
        """Save signal weights to database"""
        return self.db.save_signal_weights()

    # Methods that need to be implemented or stubbed
    def get_regime_adapted_parameters(self, symbol: str, regime, portfolio_metrics: dict = None):
        """Get regime-adapted parameters (stub implementation)"""
        logger.warning("get_regime_adapted_parameters not implemented - using default parameters")
        return self.get_adaptive_parameters()

    def update_regime_data(self, symbol: str, regime_data: dict):
        """Update regime data (stub implementation)"""
        logger.warning("update_regime_data not implemented")
        # Could be implemented to store regime data in database
        pass

    def get_performance_summary(self) -> dict:
        """Get performance summary (stub implementation)"""
        logger.warning("get_performance_summary not implemented - using basic metrics")
        return self.algorithms.evaluate_performance()

    def auto_learn_from_previous_day_logs(self):
        """Auto learn from previous day logs (stub implementation)"""
        logger.warning("auto_learn_from_previous_day_logs not implemented")
        # Could be implemented to analyze log files and learn from them
        pass

    def learn_from_logs(self, log_date: str):
        """Learn from logs for specific date (stub implementation)"""
        logger.warning(f"learn_from_logs not implemented for date: {log_date}")
        # Could be implemented to analyze specific log files
        pass

    def get_trade_insights(self, symbol: str, is_win: bool) -> dict:
        """Get trade insights for a symbol (stub implementation)"""
        logger.warning(f"get_trade_insights not implemented for symbol: {symbol}")
        return {
            'symbol': symbol,
            'insights': [],
            'recommendations': []
        }

    def update_adaptive_params(self, updates: dict):
        """Update adaptive parameters"""
        self.adaptive_params.update(updates)
        logger.info(f"Updated adaptive parameters: {updates}")

    def update_signal_weights(self, updates: dict):
        """Update signal weights"""
        self.signal_weights.update(updates)
        # Normalize weights to sum to 1.0
        total = sum(self.signal_weights.values())
        if total > 0:
            self.signal_weights = {k: v/total for k, v in self.signal_weights.items()}
        logger.info(f"Updated signal weights: {self.signal_weights}")

    def get_performance_metrics(self) -> dict:
        """Get current performance metrics"""
        return self.performance_metrics

    def get_trade_history(self) -> list:
        """Get recent trade history"""
        return list(self.trade_history)

    def get_model_versions(self) -> dict:
        """Get current model versions"""
        return self.model_versions

    def get_parameter_history(self) -> list:
        """Get parameter optimization history"""
        return self.parameter_history

    def should_adjust_existing_trade(self, symbol: str, current_sl: float, current_tp: float, trade_timestamp) -> dict:
        """
        Determine if an existing trade's SL/TP should be adjusted based on adaptive learning.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            current_sl: Current stop loss price
            current_tp: Current take profit price
            trade_timestamp: When the trade was opened

        Returns:
            dict: {
                'should_adjust': bool,
                'new_sl_atr_multiplier': float (optional),
                'new_tp_atr_multiplier': float (optional)
            }
        """
        try:
            # Get current market data for the symbol
            if not self.mt5:
                logger.warning("MT5 connector not available for trade adjustment analysis")
                return {'should_adjust': False}

            # Get current price
            tick = self.mt5.get_current_price(symbol)
            if not tick:
                logger.warning(f"Could not get tick data for {symbol}")
                return {'should_adjust': False}

            # Use bid/ask prices appropriately
            current_price = (tick['bid'] + tick['ask']) / 2  # Use mid price

            # Simple heuristic: if SL is below current price and TP is above, it's a buy
            # If SL is above current price and TP is below, it's a sell
            is_buy_position = current_sl < current_price < current_tp
            is_sell_position = current_tp < current_price < current_sl

            if not (is_buy_position or is_sell_position):
                logger.warning(f"Could not determine position type for {symbol}: SL={current_sl}, Price={current_price}, TP={current_tp}")
                return {'should_adjust': False}

            # Calculate current risk-reward ratio
            if is_buy_position:
                risk = current_price - current_sl
                reward = current_tp - current_price
            else:  # sell position
                risk = current_sl - current_price
                reward = current_price - current_tp

            if risk <= 0:
                logger.warning(f"Invalid risk calculation for {symbol}: risk={risk}")
                return {'should_adjust': False}

            current_rr_ratio = reward / risk

            # Get optimal parameters for this symbol
            optimal_params = self.get_symbol_optimal_holding_time(symbol)
            target_rr_ratio = optimal_params.get('target_rr_ratio', 3.0)  # Default 3:1

            # If current RR ratio is significantly worse than target, suggest adjustment
            if current_rr_ratio < target_rr_ratio * 0.8:  # More than 20% below target
                logger.info(f"Trade adjustment needed for {symbol}: current RR {current_rr_ratio:.2f}:1 < target {target_rr_ratio:.2f}:1")

                # Suggest improved ATR multipliers for better RR ratio
                # Use more conservative SL and more aggressive TP
                new_sl_multiplier = self.adaptive_params.get('stop_loss_atr_multiplier', 2.0)
                new_tp_multiplier = self.adaptive_params.get('take_profit_atr_multiplier', 9.0)

                # Adjust TP multiplier to achieve target RR ratio
                if new_sl_multiplier > 0:
                    target_tp_multiplier = new_sl_multiplier * target_rr_ratio
                    new_tp_multiplier = min(target_tp_multiplier, new_tp_multiplier * 1.2)  # Cap increase

                return {
                    'should_adjust': True,
                    'new_sl_atr_multiplier': new_sl_multiplier,
                    'new_tp_atr_multiplier': new_tp_multiplier
                }

            # No adjustment needed
            return {'should_adjust': False}

        except Exception as e:
            logger.error(f"Error in should_adjust_existing_trade for {symbol}: {e}")
            return {'should_adjust': False}