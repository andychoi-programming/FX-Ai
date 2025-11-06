"""
FX-Ai Adaptive Learning Manager
Implements continuous learning and strategy improvement
"""

import os
import json
import logging
import threading
import time
from datetime import datetime, timedelta

# Import schedule library
try:
    import schedule  # type: ignore
except ImportError:
    # Fallback if schedule is not available
    class ScheduleFallback:
        @staticmethod
        def every(interval):
            return Every(interval)
        @staticmethod
        def run_pending():
            now = time.time()
            with schedule_lock:
                for job in list(schedule_jobs):
                    if now >= job['next_run']:
                        try:
                            threading.Thread(
                                target=job['func'], daemon=True).start()
                        except Exception:
                            pass
                        if job['type'] == 'interval':
                            job['next_run'] = now + job['interval']
                        elif job['type'] == 'daily':
                            job['next_run'] = job['next_run'] + 24 * 3600
    schedule = ScheduleFallback()

    class Every:
        def __init__(self, interval=None):
            self.interval = 1 if interval is None else interval
            self.unit = None
            self.at_time = None

        @property
        def day(self):
            self.unit = 'day'
            return self

        @property
        def hours(self):
            self.unit = 'hours'
            return self

        def at(self, time_str):
            self.at_time = time_str
            return self

        def do(self, job):
            with schedule_lock:
                if self.unit == 'hours':
                    interval_seconds = float(self.interval) * 3600.0
                    next_run = time.time() + interval_seconds
                    schedule_jobs.append({
                        'func': job, 'interval': interval_seconds,
                        'next_run': next_run, 'type': 'interval'
                    })
                elif self.unit == 'day':
                    at = self.at_time or "00:00"
                    try:
                        h, m = [int(x) for x in at.split(":")]
                    except Exception:
                        h, m = 0, 0
                    now_dt = datetime.now()
                    next_run_dt = now_dt.replace(
                        hour=h, minute=m, second=0, microsecond=0
                    )
                    if next_run_dt <= now_dt:
                        next_run_dt = next_run_dt + timedelta(days=1)
                    schedule_jobs.append({
                        'func': job, 'interval': 24 * 3600,
                        'next_run': next_run_dt.timestamp(), 'type': 'daily'
                    })
                else:
                    interval_seconds = (
                        float(self.interval)
                        if self.interval is not None else 1.0
                    )
                    next_run = time.time() + interval_seconds
                    schedule_jobs.append({
                        'func': job, 'interval': interval_seconds,
                        'next_run': next_run, 'type': 'interval'
                    })
            return True

    def every(interval=None):
        return Every(interval)

    def run_pending():
        now = time.time()
        with schedule_lock:
            for job in list(schedule_jobs):
                if now >= job['next_run']:
                    try:
                        threading.Thread(
                            target=job['func'], daemon=True).start()
                    except Exception:
                        pass
                    if job['type'] == 'interval':
                        job['next_run'] = now + job['interval']
                    elif job['type'] == 'daily':
                        job['next_run'] = job['next_run'] + 24 * 3600

from typing import Optional
import pandas as pd
import numpy as np
from collections import deque
import sqlite3
import MetaTrader5 as mt5  # type: ignore

logger = logging.getLogger(__name__)

# Global variables for schedule fallback
schedule_lock = threading.Lock()
schedule_jobs = []


class AdaptiveLearningManager:
    """
    Manages continuous learning and adaptation for FX-Ai
    - Periodic model retraining
    - Performance-based weight adjustment
    - Parameter optimization
    - Trade feedback integration
    """

    def __init__(
            self,
            config: dict,
            ml_predictor=None,
            backtest_engine=None,
            risk_manager=None,
            mt5_connector=None):
        """Initialize the adaptive learning system"""
        self.config = config
        self.ml_predictor = ml_predictor
        self.backtest_engine = backtest_engine
        self.risk_manager = risk_manager
        self.mt5 = mt5_connector

        # Learning configuration
        self.retrain_interval = config.get(
            'adaptive_learning', {}).get(
            'retrain_interval_hours', 24)
        self.min_trades_for_update = config.get(
            'adaptive_learning', {}).get(
            'min_trades_for_update', 50)
        self.performance_window = config.get(
            'adaptive_learning', {}).get(
            'performance_window_days', 1825)
        self.adaptation_rate = config.get(
            'adaptive_learning', {}).get(
            'adaptation_rate', 0.1)

        # Performance tracking
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
                # ATR multiplier for take profit (1:3 ratio)
                'take_profit_atr_multiplier': 6.0,
                # Maximum holding time in minutes (8 hours)
                'max_holding_minutes': 480,
                'min_holding_minutes': 15,   # Minimum holding time in minutes
                'optimal_holding_hours': 4.0  # Target optimal holding period
            }
        )

        # Force reset risk multiplier to prevent inflation
        self.adaptive_params['risk_multiplier'] = 1.0

        # Temporal analysis configuration
        temporal_config = config.get('adaptive_learning', {}).get('temporal_analysis', {})
        self.temporal_enabled = temporal_config.get('enabled', True)
        self.temporal_confidence_threshold = temporal_config.get('confidence_threshold', 0.6)
        self.temporal_min_trades = temporal_config.get('min_trades_required', 20)
        self.temporal_timeframes = temporal_config.get('timeframes', {
            'daily': True,
            'weekly': True,
            'monthly': True,
            'yearly': True
        })
        self.temporal_weights = temporal_config.get('performance_weights', {
            'win_rate_weight': 0.4,
            'sharpe_ratio_weight': 0.4,
            'max_drawdown_weight': 0.2
        })

        # Market regime configuration
        regime_config = config.get('adaptive_learning', {}).get('market_regime', {})
        self.regime_enabled = regime_config.get('enabled', True)
        self.regime_adaptation_strength = regime_config.get('adaptation_strength', 0.3)
        self.regime_history_length = regime_config.get('history_length', 100)

        # Regime data storage
        self.regime_history = {}  # symbol -> list of regime analyses
        self.regime_performance = {}  # (symbol, regime) -> performance metrics

        # Schedule periodic tasks
        self.schedule_tasks()

        logger.info("Adaptive Learning Manager initialized")

    def init_database(self):
        """Initialize SQLite database for trade and performance tracking"""
        self.db_path = os.path.join('data', 'performance_history.db')
        os.makedirs('data', exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                volume REAL,
                profit REAL,
                profit_pct REAL,
                signal_strength REAL,
                ml_score REAL,
                technical_score REAL,
                sentiment_score REAL,
                duration_minutes INTEGER,
                model_version TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                model_type TEXT,
                accuracy REAL,
                precision REAL,
                recall REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                avg_profit REAL,
                total_trades INTEGER
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parameter_optimization (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                parameter_name TEXT,
                old_value REAL,
                new_value REAL,
                improvement_pct REAL,
                validation_score REAL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS symbol_holding_times (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE,
                optimal_holding_hours REAL,
                max_holding_minutes INTEGER,
                min_holding_minutes INTEGER,
                avg_profit_by_duration TEXT,  -- JSON duration buckets
                total_trades INTEGER,
                last_updated DATETIME,
                confidence_score REAL
            )
        ''')

        # Entry timing analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entry_timing_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                hour_of_day INTEGER,  -- 0-23
                day_of_week INTEGER,  -- 0-6 (Monday-Sunday)
                market_volatility REAL,  -- ATR normalized volatility
                spread_pips REAL,      -- Spread in pips at entry
                total_trades INTEGER,
                profitable_trades INTEGER,
                avg_profit REAL,
                win_rate REAL,
                last_updated DATETIME
            )
        ''')

        # Daily temporal analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_temporal_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                day_of_month INTEGER,  -- 1-31
                month_of_year INTEGER, -- 1-12
                year INTEGER,          -- Year
                total_trades INTEGER,
                profitable_trades INTEGER,
                avg_profit REAL,
                win_rate REAL,
                avg_volatility REAL,
                avg_spread_pips REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                last_updated DATETIME,
                UNIQUE(symbol, day_of_month, month_of_year, year)
            )
        ''')

        # Weekly temporal analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weekly_temporal_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                week_of_year INTEGER, -- 1-52
                year INTEGER,          -- Year
                total_trades INTEGER,
                profitable_trades INTEGER,
                avg_profit REAL,
                win_rate REAL,
                avg_volatility REAL,
                avg_spread_pips REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                last_updated DATETIME,
                UNIQUE(symbol, week_of_year, year)
            )
        ''')

        # Monthly temporal analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS monthly_temporal_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                month_of_year INTEGER, -- 1-12
                year INTEGER,          -- Year
                total_trades INTEGER,
                profitable_trades INTEGER,
                avg_profit REAL,
                win_rate REAL,
                avg_volatility REAL,
                avg_spread_pips REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                last_updated DATETIME,
                UNIQUE(symbol, month_of_year, year)
            )
        ''')

        # Yearly temporal analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS yearly_temporal_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                year INTEGER,          -- Year
                total_trades INTEGER,
                profitable_trades INTEGER,
                avg_profit REAL,
                win_rate REAL,
                avg_volatility REAL,
                avg_spread_pips REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                last_updated DATETIME,
                UNIQUE(symbol, year)
            )
        ''')

        # Per-symbol SL/TP optimization table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS symbol_sl_tp_optimization (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE,
                optimal_sl_atr_multiplier REAL,
                optimal_tp_atr_multiplier REAL,
                optimal_rr_ratio REAL,  -- Risk-reward ratio
                avg_win_rate REAL,
                avg_profit_factor REAL,
                total_trades INTEGER,
                last_updated DATETIME,
                confidence_score REAL
            )
        ''')

        # Entry filter learning table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entry_filters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                filter_type TEXT,  -- 'time_filter', 'volatility_filter', etc.
                condition_value REAL,
                should_enter BOOLEAN,  -- Whether to enter when met
                total_trades INTEGER,
                profitable_trades INTEGER,
                win_rate REAL,
                last_updated DATETIME
            )
        ''')

        # Technical indicator optimization table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS technical_indicator_optimization (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                indicator_name TEXT,  -- 'vwap', 'ema', 'rsi', 'atr', etc.
                parameter_name TEXT,  -- 'period', 'fast_period', etc.
                optimal_value REAL,
                performance_score REAL,  -- win_rate, profit_factor, etc.
                total_trades INTEGER,
                last_updated DATETIME,
                confidence_score REAL
            )
        ''')

        # Fundamental weight optimization table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fundamental_weight_optimization (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT,  -- 'myfxbook', 'fxstreet', 'fxblue', etc.
                optimal_weight REAL,
                prediction_accuracy REAL,
                total_predictions INTEGER,
                last_updated DATETIME,
                market_condition TEXT  -- 'trending', 'ranging', etc.
            )
        ''')

        # Economic calendar impact table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS economic_calendar_impact (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_name TEXT,
                event_impact TEXT,  -- 'high', 'medium', 'low'
                hours_before_event INTEGER,
                hours_after_event INTEGER,
                avg_trade_performance REAL,
                total_trades INTEGER,
                should_avoid_trading BOOLEAN,
                last_updated DATETIME,
                currency_pair TEXT
            )
        ''')

        # Interest rate impact table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interest_rate_impact (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                currency TEXT,
                rate_change REAL,  -- percentage change
                time_horizon TEXT,  -- '1h', '4h', '1d', '1w'
                avg_price_movement REAL,
                total_observations INTEGER,
                correlation_strength REAL,
                last_updated DATETIME
            )
        ''')

        # Sentiment parameter optimization table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_parameter_optimization (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parameter_name TEXT,  -- 'sentiment_threshold', etc.
                optimal_value REAL,
                performance_impact REAL,
                total_trades INTEGER,
                last_updated DATETIME,
                market_condition TEXT
            )
        ''')

        # Position adjustment tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS position_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                old_sl REAL,
                old_tp REAL,
                new_sl REAL,
                new_tp REAL,
                adjustment_reason TEXT,
                adjustment_timestamp TEXT NOT NULL
            )
        ''')

        # Adjustment performance analysis
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS adjustment_performance_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_date TEXT NOT NULL,
                success_rate REAL NOT NULL,
                total_adjustments INTEGER NOT NULL,
                successful_adjustments INTEGER NOT NULL,
                avg_profit_impact REAL
            )
        ''')

        conn.commit()
        conn.close()

    def schedule_tasks(self):
        """Schedule periodic learning tasks"""
        # Model retraining
        schedule.every(self.retrain_interval).hours.do(self.retrain_models)

        # Performance evaluation
        schedule.every(6).hours.do(self.evaluate_performance)

        # Parameter optimization
        schedule.every(12).hours.do(self.optimize_parameters)

        # Signal weight adjustment
        schedule.every(4).hours.do(self.adjust_signal_weights)

        # Symbol-specific holding time optimization
        schedule.every(24).hours.do(self.update_all_symbol_holding_times)

        # Entry timing analysis
        schedule.every(12).hours.do(self.analyze_entry_timing)

        # Per-symbol SL/TP optimization
        schedule.every(24).hours.do(self.optimize_symbol_sl_tp)

        # Entry filter learning
        schedule.every(8).hours.do(self.update_entry_filters)

        # Technical indicator optimization
        schedule.every(24).hours.do(self.optimize_technical_indicators)

        # Fundamental weight optimization
        schedule.every(12).hours.do(self.optimize_fundamental_weights)

        # Economic calendar learning
        schedule.every(6).hours.do(self.analyze_economic_calendar_impact)

        # Interest rate impact analysis
        schedule.every(24).hours.do(self.analyze_interest_rate_impact)

        # Sentiment parameter optimization
        schedule.every(12).hours.do(self.optimize_sentiment_parameters)

        # Position adjustment performance analysis
        schedule.every(24).hours.do(self.analyze_adjustment_performance)

        # Clean old data
        schedule.every(1).day.at("00:00").do(self.clean_old_data)

    def run_continuous_learning(self):
        """Background thread for continuous learning"""
        logger.info("Starting continuous learning thread")

        while True:
            try:
                schedule.run_pending()
                threading.Event().wait(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in continuous learning: {e}")

    def record_trade(self, trade_data: dict):
        """Record completed trade for learning"""
        try:
            # Add to memory
            self.trade_history.append(trade_data)

            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO trades (
                    timestamp, symbol, direction, entry_price, exit_price,
                    volume, profit, profit_pct, signal_strength,
                    ml_score, technical_score, sentiment_score,
                    duration_minutes, model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data['timestamp'],
                trade_data['symbol'],
                trade_data['direction'],
                trade_data['entry_price'],
                trade_data['exit_price'],
                trade_data['volume'],
                trade_data['profit'],
                trade_data['profit_pct'],
                trade_data.get('signal_strength', 0),
                trade_data.get('ml_score', 0),
                trade_data.get('technical_score', 0),
                trade_data.get('sentiment_score', 0),
                trade_data.get('duration_minutes', 0),
                trade_data.get('model_version', 'v1')
            ))

            conn.commit()
            conn.close()

            # Trigger immediate learning if significant event
            if abs(trade_data['profit_pct']) > 5:  # Large win/loss
                self.trigger_immediate_learning(trade_data)

        except Exception as e:
            logger.error(f"Error recording trade: {e}")

    def retrain_models(self):
        """Periodically retrain ML models with recent data"""
        logger.info("Starting model retraining...")

        try:
            # Get recent market data
            symbols = self.config.get('trading', {}).get('symbols', [])

            for symbol in symbols:
                # Fetch recent data (last 5 years)
                market_data = self.fetch_recent_market_data(symbol, days=1825)

                if market_data is not None and len(
                        market_data) > self.min_trades_for_update:
                    # Get recent trades for this symbol
                    recent_trades = self.get_recent_trades(symbol, days=1825)

                    # Prepare training data with trade outcomes
                    training_data = self.prepare_training_data(
                        market_data, recent_trades)

                    # Retrain the model
                    old_version = self.model_versions.get(symbol, 'v1')
                    new_version = f"v{datetime.now().strftime('%Y%m%d_%H%M')}"

                    # Backup old model
                    self.backup_model(symbol, old_version)

                    # Train new model
                    if self.ml_predictor is None:
                        logger.error("ML predictor not available for retraining")
                        return
                    
                    metrics = self.ml_predictor.update_models(
                        [symbol], {symbol: training_data})

                    # Validate new model
                    if self.validate_new_model(symbol, metrics):
                        self.model_versions[symbol] = new_version
                        logger.info(
                            f"Model updated for {symbol}: "
                            f"{old_version} -> {new_version}")

                        # Record performance
                        self.record_model_performance(symbol, metrics)
                    else:
                        # Rollback to old model
                        self.rollback_model(symbol, old_version)
                        logger.warning(
                            f"Model validation failed for {symbol}, "
                            f"keeping {old_version}")

        except Exception as e:
            logger.error(f"Error in model retraining: {e}")

    def evaluate_performance(self):
        """Evaluate trading performance and identify areas for improvement"""
        logger.info("Evaluating trading performance...")

        try:
            conn = sqlite3.connect(self.db_path)

            # Get recent trades (last 5 years with recency weighting)
            query = '''
                SELECT symbol, direction, profit_pct, signal_strength,
                       ml_score, technical_score, sentiment_score, timestamp
                FROM trades
                WHERE timestamp > datetime('now', '-1825 days')
            '''

            df = pd.read_sql_query(query, conn)
            conn.close()

            if len(df) > 0:
                # Convert timestamp and calculate recency weights
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['days_old'] = (pd.Timestamp.now() - df['timestamp']).dt.days  # type: ignore

                # Exponential decay weighting: more weight on recent trades
                # (half-life ~30 days)
                df['weight'] = np.exp(-df['days_old'] / 30.0)

                # Normalize weights for analysis
                total_weight = df['weight'].sum()

                # Calculate weighted performance metrics
                self.performance_metrics = {
                    'overall_win_rate': (
                        (df['profit_pct'] > 0).astype(int) *
                        df['weight']).sum() /
                    total_weight,
                    'avg_profit': (
                        df['profit_pct'] *
                        df['weight']).sum() /
                    total_weight,
                    'sharpe_ratio': self.calculate_weighted_sharpe(df),
                    'max_drawdown': self.calculate_max_drawdown(
                        df['profit_pct'].cumsum()),
                }

                # Analyze performance by signal strength with weighting
                for component in [
                    'ml_score',
                    'technical_score',
                        'sentiment_score']:
                    if component in df.columns:
                        # Group by signal strength quartiles
                        df[f'{component}_quartile'] = pd.qcut(
                            df[component], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
                        performance_by_quartile = df.groupby(
                            f'{component}_quartile'
                        ).apply(
                            lambda x: pd.Series({
                                'mean': (
                                    (x['profit_pct'] * x['weight']).sum() /
                                    x['weight'].sum()
                                ),
                                'count': len(x),
                                'weighted_count': x['weight'].sum()
                            })
                        )

                        # Store insights
                        self.performance_metrics[
                            f'{component}_performance'
                        ] = performance_by_quartile.to_dict(
                        )

                # Identify best and worst performing setups
                self.identify_patterns()

                logger.info(
                    f"Performance evaluation complete: Win rate={
                        self.performance_metrics['overall_win_rate']:.2%}")

        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")

    def optimize_parameters(self):
        """Optimize trading parameters based on recent performance"""
        logger.info("Starting parameter optimization...")

        try:
            # Get recent market data for backtesting
            test_data = self.prepare_backtest_data()

            if test_data is None:
                return

            # Parameters to optimize
            param_ranges = {
                # Existing parameters
                'rsi_oversold': (20, 40),
                'rsi_overbought': (60, 80),
                'min_signal_strength': (0.5, 0.8),
                'trailing_stop_distance': (15, 30),
                'stop_loss_atr_multiplier': (1.5, 3.0),
                'take_profit_atr_multiplier': (2.0, 6.0),

                # Technical indicator parameters
                'vwap_period': (10, 50),
                'ema_fast_period': (5, 20),
                'ema_slow_period': (15, 50),
                'rsi_period': (7, 21),
                'atr_period': (7, 21),

                # Fundamental source weights
                'myfxbook_weight': (0.1, 0.4),
                'fxstreet_weight': (0.1, 0.4),
                'fxblue_weight': (0.1, 0.3),
                'investing_weight': (0.1, 0.3),
                'forexclientsentiment_weight': (0.05, 0.2),

                # Sentiment parameters
                'sentiment_threshold': (0.1, 0.5),
                'sentiment_time_decay': (0.5, 2.0),
                'keyword_weight_multiplier': (0.8, 1.5)
            }

            best_params = self.adaptive_params.copy()
            best_score = self.calculate_optimization_score(
                best_params, test_data)

            # Grid search with walk-forward validation
            for param_name, (min_val, max_val) in param_ranges.items():
                test_values = np.linspace(min_val, max_val, 5)

                for test_value in test_values:
                    test_params = best_params.copy()
                    test_params[param_name] = test_value

                    # Run backtest with new parameters
                    score = self.calculate_optimization_score(
                        test_params, test_data)

                    if score > best_score:
                        old_value = best_params[param_name]
                        best_params[param_name] = test_value
                        best_score = score

                        # Record optimization
                        self.record_parameter_change(
                            param_name, old_value, test_value, score)

                        logger.info(
                            f"Optimized {param_name}: {
                                old_value:.2f} -> {
                                test_value:.2f} (score: {
                                score:.4f})")

            # Apply optimized parameters with gradual adaptation
            self.apply_optimized_parameters(best_params)

        except Exception as e:
            logger.error(f"Error in parameter optimization: {e}")

    def adjust_signal_weights(self):
        """Adjust signal component weights based on their predictive power"""
        logger.info("Adjusting signal weights...")

        try:
            if len(self.trade_history) < self.min_trades_for_update:
                return

            # Analyze correlation between signal components and outcomes
            trades_df = pd.DataFrame(
                list(self.trade_history)[-200:])  # Last 200 trades

            if 'profit_pct' not in trades_df.columns:
                return

            correlations = {}
            for component in [
                'ml_score',
                'technical_score',
                    'sentiment_score']:
                if component in trades_df.columns:
                    correlation = trades_df[component].corr(
                        trades_df['profit_pct'])
                    correlations[component.replace(
                        '_score', '_prediction')] = abs(correlation)

            # Normalize correlations to sum to 1
            total_correlation = sum(correlations.values())
            if total_correlation > 0:
                # Gradual adjustment using adaptation rate
                for component, correlation in correlations.items():
                    if component in self.signal_weights:
                        new_weight = correlation / total_correlation
                        old_weight = self.signal_weights[component]

                        # Smooth adjustment
                        adjusted_weight = old_weight * \
                            (1 - self.adaptation_rate) + \
                            new_weight * self.adaptation_rate
                        self.signal_weights[component] = adjusted_weight

                        logger.info(
                            f"Adjusted {component} weight: "
                            f"{old_weight:.3f} -> {adjusted_weight:.3f}")

            # Save updated weights
            self.save_signal_weights()

        except Exception as e:
            logger.error(f"Error adjusting signal weights: {e}")

    def trigger_immediate_learning(self, trade_data: dict):
        """Trigger immediate learning for significant events"""
        try:
            symbol = trade_data['symbol']
            profit_pct = trade_data['profit_pct']

            logger.info(
                f"Significant trade event for {symbol}: {
                    profit_pct:.2f}% - triggering immediate learning")

            # Adjust risk parameters if large loss
            if profit_pct < -3:
                self.adaptive_params['risk_multiplier'] *= 0.95
                logger.info(
                    f"Reduced risk multiplier to {
                        self.adaptive_params['risk_multiplier']:.2f}")

            # Increase confidence if large win
            elif profit_pct > 5:
                if self.adaptive_params['risk_multiplier'] < 1.0:
                    self.adaptive_params['risk_multiplier'] *= 1.02
                    logger.info(
                        f"Increased risk multiplier to {
                            self.adaptive_params['risk_multiplier']:.2f}")

            # Update model if pattern detected
            self.check_for_new_patterns(trade_data)

        except Exception as e:
            logger.error(f"Error in immediate learning: {e}")

    def identify_patterns(self):
        """Identify successful and unsuccessful trading patterns"""
        try:
            if len(self.trade_history) < 50:
                return

            trades_df = pd.DataFrame(list(self.trade_history))

            # Identify winning conditions
            winning_trades = trades_df[trades_df['profit_pct'] > 0]
            losing_trades = trades_df[trades_df['profit_pct'] <= 0]

            if len(winning_trades) > 10 and len(losing_trades) > 10:
                # Find common characteristics of winners
                winning_patterns = {
                    'avg_signal_strength': (
                        winning_trades['signal_strength'].mean()
                    ),
                    'avg_ml_score': winning_trades.get(
                        'ml_score',
                        pd.Series()).mean(),
                    'common_hour': (
                        winning_trades['timestamp'].dt.hour.mode().values[0]
                        if 'timestamp' in winning_trades else None
                    )}

                # Find common characteristics of losers
                losing_patterns = {
                    'avg_signal_strength': (
                        losing_trades['signal_strength'].mean()
                    ),
                    'avg_ml_score': losing_trades.get(
                        'ml_score',
                        pd.Series()).mean(),
                }

                # Adjust minimum signal strength based on patterns
                if (winning_patterns['avg_signal_strength'] >
                        losing_patterns['avg_signal_strength']):
                    new_threshold = (
                        (winning_patterns['avg_signal_strength'] +
                         self.adaptive_params['min_signal_strength']) / 2
                    )
                    self.adaptive_params['min_signal_strength'] = new_threshold
                    logger.info(
                        f"Adjusted min_signal_strength to {
                            new_threshold:.3f}")

        except Exception as e:
            logger.error(f"Error identifying patterns: {e}")

    def calculate_optimization_score(
            self,
            params: dict,
            test_data: pd.DataFrame) -> float:
        """Calculate optimization score for parameters using backtest"""
        try:
            # Create a strategy function that uses the test parameters
            def test_strategy(data_dict, symbol):
                signals = []
                data = data_dict[symbol]

                for i in range(len(data)):
                    row = data.iloc[i]

                    # Simple RSI-based signal generation using test parameters
                    rsi = row.get('rsi', 50)
                    if rsi < params.get(
                        'rsi_oversold',
                            30) and row['close'] > row['open']:
                        # Buy signal
                        entry_price = row['close']
                        atr = row.get(
                            'atr', entry_price * 0.01)  # Fallback ATR

                        # Use adaptive ATR multipliers for SL/TP
                        sl_multiplier = params.get(
                            'stop_loss_atr_multiplier', 2.0)
                        tp_multiplier = params.get(
                            'take_profit_atr_multiplier', 4.0)

                        stop_loss = entry_price - (atr * sl_multiplier)
                        take_profit = entry_price + (atr * tp_multiplier)

                        # Track holding time
                        max_holding = params.get('max_holding_minutes', 480)
                        min_holding = params.get('min_holding_minutes', 15)

                        signals.append({
                            'action': 'BUY',
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'strength': (
                                abs(row['close'] - row['open']) / row['open']
                            ),
                            'max_holding_minutes': max_holding,
                            'min_holding_minutes': min_holding
                        })

                    elif (rsi > params.get('rsi_overbought', 70) and
                            row['close'] < row['open']):
                        # Sell signal
                        entry_price = row['close']
                        atr = row.get(
                            'atr', entry_price * 0.01)  # Fallback ATR

                        # Use adaptive ATR multipliers for SL/TP
                        sl_multiplier = params.get(
                            'stop_loss_atr_multiplier', 2.0)
                        tp_multiplier = params.get(
                            'take_profit_atr_multiplier', 4.0)

                        stop_loss = entry_price + (atr * sl_multiplier)
                        take_profit = entry_price - (atr * tp_multiplier)

                        # Track holding time
                        max_holding = params.get('max_holding_minutes', 480)
                        min_holding = params.get('min_holding_minutes', 15)

                        signals.append({
                            'action': 'SELL',
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'strength': (
                                abs(row['close'] - row['open']) / row['open']
                            ),
                            'max_holding_minutes': max_holding,
                            'min_holding_minutes': min_holding
                        })

                return signals

            # Run backtest with the test strategy
            if hasattr(self, 'backtest_engine') and self.backtest_engine:
                # Use actual backtest engine if available
                start_date = test_data.index.min()
                end_date = test_data.index.max()

                # Prepare data in the format expected by backtest engine
                # data_dict = {symbol: test_data for symbol in [
                #     'EURUSD']}  # Simplified for single symbol

                results = self.backtest_engine.run_backtest(
                    start_date, end_date, ['EURUSD'], test_strategy
                )

                if results and 'metrics' in results:
                    metrics = results['metrics']
                    # Calculate score based on Sharpe ratio and win rate
                    sharpe = metrics.get('sharpe_ratio', 0)
                    win_rate = metrics.get('win_rate', 0)
                    profit_factor = metrics.get('profit_factor', 1)

                    # Weighted score combining multiple metrics
                    score = (sharpe * 0.4 + win_rate * 0.3 +
                             min(profit_factor, 3) * 0.3)
                    return max(score, 0)
            else:
                # Fallback to simple simulation if backtest engine not
                # available
                simulated_returns = []

                for i in range(len(test_data) - 1):
                    row = test_data.iloc[i]
                    next_row = test_data.iloc[i + 1]

                    # Generate signal based on parameters
                    rsi_signal = 1 if row['close'] > row['open'] and row.get(
                        'rsi', 50) < params.get('rsi_oversold', 30) else -1
                    signal_strength = abs(
                        row['close'] - row['open']) / row['open']

                    if signal_strength > params.get(
                            'min_signal_strength', 0.001):
                        # Calculate return with ATR-based SL/TP simulation
                        atr = row.get('atr', row['close'] * 0.01)
                        sl_multiplier = params.get(
                            'stop_loss_atr_multiplier', 2.0)
                        tp_multiplier = params.get(
                            'take_profit_atr_multiplier', 4.0)

                        # Simulate price movement
                        price_change = (
                            next_row['close'] - row['close']) / row['close']

                        # Check if hit SL or TP
                        if rsi_signal > 0:  # Long position
                            sl_level = -atr * sl_multiplier / row['close']
                            tp_level = atr * tp_multiplier / row['close']
                        else:  # Short position
                            sl_level = atr * sl_multiplier / row['close']
                            tp_level = -atr * tp_multiplier / row['close']

                        # Time-based exit logic
                        max_holding_minutes = params.get(
                            'max_holding_minutes', 480)
                        optimal_holding_hours = params.get(
                            'optimal_holding_hours', 4.0)
                        optimal_holding_minutes = optimal_holding_hours * 60

                        # Simulate holding time (assume 5-minute bars, so each
                        # iteration = 5 minutes)
                        holding_minutes = 0  # Placeholder

                        # Exit conditions: SL, TP, or max holding time
                        if price_change <= sl_level:
                            # Hit stop loss
                            realized_return = sl_level * rsi_signal
                            exit_reason = 'stop_loss'
                        elif price_change >= tp_level:
                            # Hit take profit
                            realized_return = tp_level * rsi_signal
                            exit_reason = 'take_profit'
                        elif holding_minutes >= max_holding_minutes:
                            # Max holding time reached - exit at current price
                            realized_return = price_change * rsi_signal
                            exit_reason = 'max_time'
                        elif (holding_minutes >= optimal_holding_minutes and
                                price_change > 0):
                            # Optimal holding time reached with profit - take
                            # profit
                            realized_return = price_change * rsi_signal * \
                                0.8  # Slightly reduce profit for optimal exit
                            exit_reason = 'optimal_time'
                        else:
                            # Position still open, use actual price change
                            realized_return = price_change * rsi_signal
                            exit_reason = 'open'

                        # Only count completed trades (not open positions)
                        if exit_reason != 'open':
                            simulated_returns.append(
                                realized_return *
                                params.get(
                                    'risk_multiplier',
                                    1.0))
                    else:
                        simulated_returns.append(0)  # No trade

                if simulated_returns:
                    avg_return = np.mean(simulated_returns)
                    volatility = np.std(simulated_returns)
                    sharpe = avg_return / \
                        (volatility + 1e-6) if volatility > 0 else 0
                    score = sharpe * params.get('risk_multiplier', 1.0)
                    return max(score, 0)

            return 0.0

        except Exception as e:
            logger.error(f"Error calculating optimization score: {e}")
            return 0.0

    def apply_optimized_parameters(self, new_params: dict):
        """Gradually apply optimized parameters"""
        for param_name, new_value in new_params.items():
            if (param_name in self.adaptive_params and
                    param_name != 'risk_multiplier'):
                old_value = self.adaptive_params[param_name]
                # Gradual transition
                adjusted_value = old_value * \
                    (1 - self.adaptation_rate) + \
                    new_value * self.adaptation_rate
                self.adaptive_params[param_name] = adjusted_value

    def get_current_weights(self) -> dict:
        """Get current signal weights for trading decisions"""
        return self.signal_weights.copy()

    def get_adaptive_parameters(self) -> dict:
        """Get current adaptive parameters"""
        params = self.adaptive_params.copy()
        params['risk_multiplier'] = 1.0  # Always fixed at 1.0
        return params

    def prepare_backtest_data(self) -> Optional[pd.DataFrame]:
        """Prepare data for backtesting"""
        try:
            # Get recent data for backtesting (last 5 years)
            symbols = self.config.get('trading', {}).get('pairs', [])[
                :3]  # Test with first 3 symbols

            if not symbols:
                return None

            # Get data for primary symbol
            primary_symbol = symbols[0]
            data = self.fetch_recent_market_data(primary_symbol, days=1825)

            if data is not None and len(data) > 100:
                # Add technical indicators for backtesting
                data['rsi'] = self.calculate_rsi(data['close'])
                data['sma_20'] = data['close'].rolling(20).mean()
                data['sma_50'] = data['close'].rolling(50).mean()
                return data
            else:
                logger.warning("Insufficient data for backtesting")
                return None

        except Exception as e:
            logger.error(f"Error preparing backtest data: {e}")
            return None

    def fetch_recent_market_data(
            self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Fetch recent market data for retraining"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Fetch historical data from MT5
            if self.mt5 is None:
                logger.error("MT5 connector not available")
                return None
            
            rates = self.mt5.get_historical_data(
                symbol=symbol,
                timeframe=mt5.TIMEFRAME_M1,  # 1-minute data
                start_date=start_date,
                end_date=end_date
            )

            if rates is not None and len(rates) > 0:
                # Convert to DataFrame
                df = pd.DataFrame(rates)
                df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('timestamp', inplace=True)
                return df
            else:
                logger.warning(f"No historical data available for {symbol}")
                return None

        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None

    def get_recent_trades(self, symbol: str, days: int) -> pd.DataFrame:
        """Get recent trades from database"""
        conn = sqlite3.connect(self.db_path)
        query = f'''
            SELECT * FROM trades
            WHERE symbol = ? AND timestamp > datetime('now', '-{days} days')
        '''
        df = pd.read_sql_query(query, conn, params=(symbol,))
        conn.close()
        return df

    def prepare_training_data(
            self,
            market_data: pd.DataFrame,
            trades: pd.DataFrame) -> pd.DataFrame:
        """Prepare training data with trade outcomes"""
        # Merge market data with trade outcomes
        # Add labels based on successful trades
        return market_data

    def validate_new_model(self, symbol: str, metrics: dict) -> bool:
        """Validate new model meets minimum performance criteria"""
        min_accuracy = self.config.get(
            'adaptive_learning', {}).get(
            'min_accuracy', 0.6)
        return metrics.get('accuracy', 0) >= min_accuracy

    def backup_model(self, symbol: str, version: str):
        """Backup existing model before update"""
        # Save model to backup location
        pass

    def rollback_model(self, symbol: str, version: str):
        """Rollback to previous model version"""
        # Restore from backup
        pass

    def record_model_performance(self, symbol: str, metrics: dict):
        """Record model performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO model_performance (
                timestamp, symbol, model_type, accuracy, precision,
                recall, sharpe_ratio, max_drawdown, win_rate, \
                avg_profit, total_trades
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            symbol,
            'ensemble',
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('sharpe_ratio', 0),
            metrics.get('max_drawdown', 0),
            metrics.get('win_rate', 0),
            metrics.get('avg_profit', 0),
            metrics.get('total_trades', 0)
        ))

        conn.commit()
        conn.close()

    def record_parameter_change(
            self,
            param_name: str,
            old_value: float,
            new_value: float,
            score: float):
        """Record parameter optimization"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        improvement = ((new_value - old_value) / old_value *
                       100) if old_value != 0 else 0

        cursor.execute('''
            INSERT INTO parameter_optimization (
                timestamp, parameter_name, old_value, new_value,
                improvement_pct, validation_score
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            param_name,
            old_value,
            new_value,
            improvement,
            score
        ))

        conn.commit()
        conn.close()

    def check_for_new_patterns(self, trade_data: dict):
        """Check for new patterns in recent trades"""
        # Analyze for new patterns that should trigger model update
        pass

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def save_signal_weights(self):
        """Save updated signal weights to file"""
        weights_file = os.path.join('config', 'adaptive_weights.json')
        os.makedirs('config', exist_ok=True)

        with open(weights_file, 'w') as f:
            json.dump({
                'signal_weights': self.signal_weights,
                'adaptive_params': self.adaptive_params,
                'last_update': datetime.now().isoformat()
            }, f, indent=2)

    def clean_old_data(self):
        """Clean old data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Keep only last 5 years of trades (1825 days)
            cursor.execute('''
                DELETE FROM trades
                WHERE timestamp < datetime('now', '-1825 days')
            ''')

            # Keep only last 5 years of model performance
            cursor.execute('''
                DELETE FROM model_performance
                WHERE timestamp < datetime('now', '-1825 days')
            ''')

            conn.commit()
            conn.close()

            logger.info("Cleaned old data from database (keeping 5 years)")

        except Exception as e:
            logger.error(f"Error cleaning old data: {e}")

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()  # type: ignore
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()  # type: ignore
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series(index=prices.index)

    def calculate_weighted_sharpe(self, df: pd.DataFrame) -> float:
        """Calculate weighted Sharpe ratio"""
        try:
            if len(df) < 2:
                return 0.0

            # Calculate weighted returns
            weighted_returns = df['profit_pct'] * df['weight']
            total_weight = df['weight'].sum()
            avg_return = weighted_returns.sum() / total_weight

            # Calculate weighted volatility
            variance = ((df['profit_pct'] - avg_return) **
                        2 * df['weight']).sum() / total_weight
            volatility = np.sqrt(variance) + 1e-6

            return avg_return / volatility
        except Exception:
            return 0.0

    def analyze_symbol_holding_performance(
            self, symbol: str, min_trades: int = 20) -> Optional[dict]:
        """Analyze trade performance by holding duration"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get trades for this symbol
            cursor.execute('''
                SELECT duration_minutes, profit_pct
                FROM trades
                WHERE symbol = ? AND profit_pct IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 500
            ''', (symbol,))

            trades = cursor.fetchall()
            conn.close()

            if len(trades) < min_trades:
                return None

            # Group trades by duration buckets (in hours)
            duration_buckets = {
                '0-1h': [], '1-2h': [], '2-3h': [], '3-4h': [], '4-6h': [],
                '6-8h': [], '8-12h': [], '12-24h': [], '24h+': []
            }

            for duration_min, profit_pct in trades:
                duration_hours = duration_min / 60.0

                if duration_hours < 1:
                    duration_buckets['0-1h'].append(profit_pct)
                elif duration_hours < 2:
                    duration_buckets['1-2h'].append(profit_pct)
                elif duration_hours < 3:
                    duration_buckets['2-3h'].append(profit_pct)
                elif duration_hours < 4:
                    duration_buckets['3-4h'].append(profit_pct)
                elif duration_hours < 6:
                    duration_buckets['4-6h'].append(profit_pct)
                elif duration_hours < 8:
                    duration_buckets['6-8h'].append(profit_pct)
                elif duration_hours < 12:
                    duration_buckets['8-12h'].append(profit_pct)
                elif duration_hours < 24:
                    duration_buckets['12-24h'].append(profit_pct)
                else:
                    duration_buckets['24h+'].append(profit_pct)

            # Calculate average profit for each bucket (only if bucket has
            # enough trades)
            avg_profit_by_duration = {}
            for bucket, profits in duration_buckets.items():
                if len(profits) >= 5:  # Minimum 5 trades per bucket
                    avg_profit_by_duration[bucket] = sum(
                        profits) / len(profits)

            return {
                'symbol': symbol,
                'total_trades': len(trades),
                'avg_profit_by_duration': avg_profit_by_duration,
                'best_bucket': max(
                    avg_profit_by_duration.items(),
                    key=lambda x: x[1]) if avg_profit_by_duration else None}

        except Exception as e:
            logger.error(
                f"Error analyzing symbol holding performance "
                f"for {symbol}: {e}")
            return None

    def calculate_optimal_holding_times(self, symbol: str) -> Optional[dict]:
        """Calculate optimal holding times for a specific symbol"""
        try:
            analysis = self.analyze_symbol_holding_performance(symbol)
            if not analysis or not analysis['best_bucket']:
                return None

            best_bucket = analysis['best_bucket'][0]
            best_avg_profit = analysis['best_bucket'][1]

            # Convert bucket to optimal holding hours
            bucket_ranges = {
                '0-1h': (0.5, 60, 15),    # optimal: 0.5h, max: 1h, min: 15min
                '1-2h': (1.5, 120, 30),   # optimal: 1.5h, max: 2h, min: 30min
                '2-3h': (2.5, 180, 60),   # optimal: 2.5h, max: 3h, min: 1h
                '3-4h': (3.5, 240, 90),   # optimal: 3.5h, max: 4h, min: 1.5h
                '4-6h': (5.0, 360, 120),  # optimal: 5h, max: 6h, min: 2h
                '6-8h': (7.0, 480, 180),  # optimal: 7h, max: 8h, min: 3h
                '8-12h': (10.0, 720, 240),  # optimal: 10h, max: 12h, min: 4h
                '12-24h': (18.0, 1440, 360),  # optimal: 18h, max: 24h, min: 6h
                '24h+': (36.0, 2880, 720)  # optimal: 36h, max: 48h, min: 12h
            }

            if best_bucket in bucket_ranges:
                optimal_hours, max_minutes, min_minutes = \
                    bucket_ranges[best_bucket]

                # Calculate confidence score based on profit difference and
                # sample size
                all_profits = list(analysis['avg_profit_by_duration'].values())
                if len(all_profits) > 1:
                    profit_std = np.std(all_profits)
                    confidence = min(1.0,
                                     best_avg_profit / (profit_std + 0.01))
                else:
                    confidence = 0.5

                return {
                    'symbol': symbol,
                    'optimal_holding_hours': optimal_hours,
                    'max_holding_minutes': max_minutes,
                    'min_holding_minutes': min_minutes,
                    'best_bucket': best_bucket,
                    'best_avg_profit': best_avg_profit,
                    'total_trades': analysis['total_trades'],
                    'confidence_score': confidence,
                    'avg_profit_by_duration': json.dumps(
                        analysis['avg_profit_by_duration'])}

            return None

        except Exception as e:
            logger.error(
                f"Error calculating optimal holding times for {symbol}: {e}")
            return None

    def update_symbol_holding_times(self, symbol: str):
        """Update optimal holding times for a symbol in the database"""
        try:
            optimal_times = self.calculate_optimal_holding_times(symbol)
            if not optimal_times:
                return

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Insert or replace symbol holding times
            cursor.execute('''
                INSERT OR REPLACE INTO symbol_holding_times
                (symbol, optimal_holding_hours, max_holding_minutes,
                 min_holding_minutes,
                 avg_profit_by_duration, total_trades, last_updated,
                 confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                optimal_times['symbol'],
                optimal_times['optimal_holding_hours'],
                optimal_times['max_holding_minutes'],
                optimal_times['min_holding_minutes'],
                optimal_times['avg_profit_by_duration'],
                optimal_times['total_trades'],
                datetime.now(),
                optimal_times['confidence_score']
            ))

            conn.commit()
            conn.close()

            logger.info(
                f"Updated optimal holding times for {symbol}: {
                    optimal_times['optimal_holding_hours']}h " f"(confidence: {
                    optimal_times['confidence_score']:.2f})")

        except Exception as e:
            logger.error(
                f"Error updating symbol holding times for {symbol}: {e}")

    def get_symbol_optimal_holding_time(self, symbol: str) -> dict:
        """Get optimal holding times for a specific symbol"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT optimal_holding_hours, max_holding_minutes,
                       min_holding_minutes,
                       confidence_score, total_trades
                FROM symbol_holding_times
                WHERE symbol = ?
            ''', (symbol,))

            result = cursor.fetchone()
            conn.close()

            if result:
                optimal_hours, max_minutes, min_minutes, confidence, \
                    total_trades = result
                return {
                    'optimal_holding_hours': optimal_hours,
                    'max_holding_minutes': max_minutes,
                    'min_holding_minutes': min_minutes,
                    'confidence_score': confidence,
                    'total_trades': total_trades,
                    'found': True
                }
            else:
                # Return default global parameters if no symbol-specific data
                return {
                    'optimal_holding_hours': self.adaptive_params.get(
                        'optimal_holding_hours',
                        4.0),
                    'max_holding_minutes': self.adaptive_params.get(
                        'max_holding_minutes',
                        480),
                    'min_holding_minutes': self.adaptive_params.get(
                        'min_holding_minutes',
                        15),
                    'confidence_score': 0.0,
                    'total_trades': 0,
                    'found': False}

        except Exception as e:
            logger.error(
                f"Error getting symbol optimal holding time for {symbol}: {e}")
            return {
                'optimal_holding_hours': self.adaptive_params.get(
                    'optimal_holding_hours',
                    4.0),
                'max_holding_minutes': self.adaptive_params.get(
                    'max_holding_minutes',
                    480),
                'min_holding_minutes': self.adaptive_params.get(
                    'min_holding_minutes',
                    15),
                'confidence_score': 0.0,
                'total_trades': 0,
                'found': False}

    def update_all_symbol_holding_times(self):
        """Update optimal holding times for all symbols"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all symbols with at least 20 trades
            cursor.execute('''
                SELECT symbol, COUNT(*) as trade_count
                FROM trades
                GROUP BY symbol
                HAVING trade_count >= 20
            ''')

            symbols = cursor.fetchall()
            conn.close()

            for symbol, trade_count in symbols:
                logger.info(
                    f"Updating holding times for {symbol} "
                    f"({trade_count} trades)")
                self.update_symbol_holding_times(symbol)

        except Exception as e:
            logger.error(f"Error updating all symbol holding times: {e}")

    def get_all_symbol_holding_times(self) -> dict:
        """Get optimal holding times for all symbols"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT symbol, optimal_holding_hours, max_holding_minutes,
                       min_holding_minutes,
                       confidence_score, total_trades, last_updated
                FROM symbol_holding_times
                ORDER BY confidence_score DESC, total_trades DESC
            ''')

            results = cursor.fetchall()
            conn.close()

            symbol_times = {}
            for row in results:
                symbol, opt_hours, max_min, min_min, confidence, \
                    trades, last_updated = row
                symbol_times[symbol] = {
                    'optimal_holding_hours': opt_hours,
                    'max_holding_minutes': max_min,
                    'min_holding_minutes': min_min,
                    'confidence_score': confidence,
                    'total_trades': trades,
                    'last_updated': last_updated
                }

            return symbol_times

        except Exception as e:
            logger.error(f"Error getting all symbol holding times: {e}")
            return {}

    def get_performance_summary(self) -> dict:
        """Get performance summary with current weights and parameters"""
        return {
            'signal_weights': self.get_current_weights(),
            'adaptive_params': self.get_adaptive_parameters()
        }

    # ===== NEW LEARNING METHODS =====

    def analyze_entry_timing(self):
        """Analyze profitable entry timing patterns across multiple timeframes"""
        logger.info("Analyzing comprehensive temporal patterns...")

        try:
            symbols = self.config.get('trading', {}).get('symbols', [])

            for symbol in symbols:
                # Get recent trades for this symbol (2 years for comprehensive analysis)
                trades_df = self.get_recent_trades_df(symbol, days=730)

                if len(trades_df) < 50:  # Need more trades for temporal analysis
                    logger.info(f"Skipping {symbol}: insufficient trades ({len(trades_df)})")
                    continue

                # Prepare temporal features
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                trades_df['hour'] = trades_df['timestamp'].dt.hour
                trades_df['day_of_week'] = trades_df['timestamp'].dt.dayofweek
                trades_df['day_of_month'] = trades_df['timestamp'].dt.day
                trades_df['week_of_year'] = trades_df['timestamp'].dt.isocalendar().week
                trades_df['month_of_year'] = trades_df['timestamp'].dt.month
                trades_df['year'] = trades_df['timestamp'].dt.year

                # Calculate additional metrics
                trades_df['is_profitable'] = trades_df['profit'] > 0

                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # 1. Analyze HOURLY patterns (existing)
                self._analyze_hourly_patterns(cursor, symbol, trades_df)

                # 2. Analyze DAILY patterns
                self._analyze_daily_patterns(cursor, symbol, trades_df)

                # 3. Analyze WEEKLY patterns
                self._analyze_weekly_patterns(cursor, symbol, trades_df)

                # 4. Analyze MONTHLY patterns
                self._analyze_monthly_patterns(cursor, symbol, trades_df)

                # 5. Analyze YEARLY patterns
                self._analyze_yearly_patterns(cursor, symbol, trades_df)

                conn.commit()
                conn.close()

                logger.info(f"Completed comprehensive temporal analysis for {symbol}")

        except Exception as e:
            logger.error(f"Error in comprehensive temporal analysis: {e}")

    def _analyze_hourly_patterns(self, cursor, symbol: str, trades_df: pd.DataFrame):
        """Analyze hourly trading patterns"""
        try:
            hourly_performance = trades_df.groupby('hour').agg({
                'profit': ['count', 'mean', lambda x: (x > 0).mean()],
                'profit_pct': 'mean'
            }).round(4)

            hourly_performance.columns = ['total_trades', 'avg_profit', 'win_rate', 'avg_profit_pct']
            hourly_performance = hourly_performance.reset_index()

            for _, row in hourly_performance.iterrows():
                if row['total_trades'] >= 5:  # Minimum trades per hour
                    cursor.execute('''
                        INSERT OR REPLACE INTO entry_timing_analysis
                        (symbol, hour_of_day, day_of_week, market_volatility, spread_pips,
                         total_trades, profitable_trades, avg_profit, win_rate, last_updated)
                        VALUES (?, ?, -1, 0, 0, ?, ?, ?, ?, ?)
                    ''', (
                        symbol, int(row['hour']), int(row['total_trades']),
                        int(row['total_trades'] * row['win_rate']),
                        row['avg_profit'], row['win_rate'], datetime.now()
                    ))

        except Exception as e:
            logger.error(f"Error analyzing hourly patterns for {symbol}: {e}")

    def _analyze_daily_patterns(self, cursor, symbol: str, trades_df: pd.DataFrame):
        """Analyze daily trading patterns"""
        try:
            # Group by day of month, month, and year
            daily_performance = trades_df.groupby(['day_of_month', 'month_of_year', 'year']).agg({
                'profit': ['count', 'mean', 'sum', lambda x: (x > 0).mean()],
                'profit_pct': 'mean'
            }).round(4)

            daily_performance.columns = ['total_trades', 'avg_profit', 'total_profit', 'win_rate', 'avg_profit_pct']
            daily_performance = daily_performance.reset_index()

            # Calculate additional metrics
            daily_performance['sharpe_ratio'] = daily_performance.apply(
                lambda row: self._calculate_sharpe_ratio(trades_df, row.name), axis=1
            )
            daily_performance['max_drawdown'] = daily_performance.apply(
                lambda row: self._calculate_max_drawdown(trades_df, row.name), axis=1
            )

            for _, row in daily_performance.iterrows():
                if row['total_trades'] >= 3:  # Minimum trades per day
                    cursor.execute('''
                        INSERT OR REPLACE INTO daily_temporal_analysis
                        (symbol, day_of_month, month_of_year, year, total_trades,
                         profitable_trades, avg_profit, win_rate, avg_volatility,
                         avg_spread_pips, sharpe_ratio, max_drawdown, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?)
                    ''', (
                        symbol, int(row['day_of_month']), int(row['month_of_year']),
                        int(row['year']), int(row['total_trades']),
                        int(row['total_trades'] * row['win_rate']),
                        row['avg_profit'], row['win_rate'],
                        row['sharpe_ratio'], row['max_drawdown'], datetime.now()
                    ))

        except Exception as e:
            logger.error(f"Error analyzing daily patterns for {symbol}: {e}")

    def _analyze_weekly_patterns(self, cursor, symbol: str, trades_df: pd.DataFrame):
        """Analyze weekly trading patterns"""
        try:
            weekly_performance = trades_df.groupby(['week_of_year', 'year']).agg({
                'profit': ['count', 'mean', 'sum', lambda x: (x > 0).mean()],
                'profit_pct': 'mean'
            }).round(4)

            weekly_performance.columns = ['total_trades', 'avg_profit', 'total_profit', 'win_rate', 'avg_profit_pct']
            weekly_performance = weekly_performance.reset_index()

            weekly_performance['sharpe_ratio'] = weekly_performance.apply(
                lambda row: self._calculate_sharpe_ratio(trades_df, row.name), axis=1
            )
            weekly_performance['max_drawdown'] = weekly_performance.apply(
                lambda row: self._calculate_max_drawdown(trades_df, row.name), axis=1
            )

            for _, row in weekly_performance.iterrows():
                if row['total_trades'] >= 5:  # Minimum trades per week
                    cursor.execute('''
                        INSERT OR REPLACE INTO weekly_temporal_analysis
                        (symbol, week_of_year, year, total_trades, profitable_trades,
                         avg_profit, win_rate, avg_volatility, avg_spread_pips,
                         sharpe_ratio, max_drawdown, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?)
                    ''', (
                        symbol, int(row['week_of_year']), int(row['year']),
                        int(row['total_trades']),
                        int(row['total_trades'] * row['win_rate']),
                        row['avg_profit'], row['win_rate'],
                        row['sharpe_ratio'], row['max_drawdown'], datetime.now()
                    ))

        except Exception as e:
            logger.error(f"Error analyzing weekly patterns for {symbol}: {e}")

    def _analyze_monthly_patterns(self, cursor, symbol: str, trades_df: pd.DataFrame):
        """Analyze monthly trading patterns"""
        try:
            monthly_performance = trades_df.groupby(['month_of_year', 'year']).agg({
                'profit': ['count', 'mean', 'sum', lambda x: (x > 0).mean()],
                'profit_pct': 'mean'
            }).round(4)

            monthly_performance.columns = ['total_trades', 'avg_profit', 'total_profit', 'win_rate', 'avg_profit_pct']
            monthly_performance = monthly_performance.reset_index()

            monthly_performance['sharpe_ratio'] = monthly_performance.apply(
                lambda row: self._calculate_sharpe_ratio(trades_df, row.name), axis=1
            )
            monthly_performance['max_drawdown'] = monthly_performance.apply(
                lambda row: self._calculate_max_drawdown(trades_df, row.name), axis=1
            )

            for _, row in monthly_performance.iterrows():
                if row['total_trades'] >= 10:  # Minimum trades per month
                    cursor.execute('''
                        INSERT OR REPLACE INTO monthly_temporal_analysis
                        (symbol, month_of_year, year, total_trades, profitable_trades,
                         avg_profit, win_rate, avg_volatility, avg_spread_pips,
                         sharpe_ratio, max_drawdown, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?)
                    ''', (
                        symbol, int(row['month_of_year']), int(row['year']),
                        int(row['total_trades']),
                        int(row['total_trades'] * row['win_rate']),
                        row['avg_profit'], row['win_rate'],
                        row['sharpe_ratio'], row['max_drawdown'], datetime.now()
                    ))

        except Exception as e:
            logger.error(f"Error analyzing monthly patterns for {symbol}: {e}")

    def _analyze_yearly_patterns(self, cursor, symbol: str, trades_df: pd.DataFrame):
        """Analyze yearly trading patterns"""
        try:
            yearly_performance = trades_df.groupby('year').agg({
                'profit': ['count', 'mean', 'sum', lambda x: (x > 0).mean()],
                'profit_pct': 'mean'
            }).round(4)

            yearly_performance.columns = ['total_trades', 'avg_profit', 'total_profit', 'win_rate', 'avg_profit_pct']
            yearly_performance = yearly_performance.reset_index()

            yearly_performance['sharpe_ratio'] = yearly_performance.apply(
                lambda row: self._calculate_sharpe_ratio(trades_df, row.name), axis=1
            )
            yearly_performance['max_drawdown'] = yearly_performance.apply(
                lambda row: self._calculate_max_drawdown(trades_df, row.name), axis=1
            )

            for _, row in yearly_performance.iterrows():
                if row['total_trades'] >= 20:  # Minimum trades per year
                    cursor.execute('''
                        INSERT OR REPLACE INTO yearly_temporal_analysis
                        (symbol, year, total_trades, profitable_trades, avg_profit,
                         win_rate, avg_volatility, avg_spread_pips, sharpe_ratio,
                         max_drawdown, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?)
                    ''', (
                        symbol, int(row['year']), int(row['total_trades']),
                        int(row['total_trades'] * row['win_rate']),
                        row['avg_profit'], row['win_rate'],
                        row['sharpe_ratio'], row['max_drawdown'], datetime.now()
                    ))

        except Exception as e:
            logger.error(f"Error analyzing yearly patterns for {symbol}: {e}")

    def _calculate_sharpe_ratio(self, trades_df: pd.DataFrame, group_key) -> float:
        """Calculate Sharpe ratio for a group of trades"""
        try:
            # This is a simplified Sharpe ratio calculation
            # In practice, you'd want daily returns and risk-free rate
            if len(trades_df) < 5:
                return 0.0

            returns = trades_df['profit_pct']
            if returns.std() == 0:
                return 0.0

            return (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized

        except Exception:
            return 0.0

    def _calculate_max_drawdown(self, trades_df: pd.DataFrame, group_key) -> float:
        """Calculate maximum drawdown for a group of trades"""
        try:
            if len(trades_df) < 3:
                return 0.0

            cumulative = (1 + trades_df['profit_pct']).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return abs(drawdown.min()) if len(drawdown) > 0 else 0.0

        except Exception:
            return 0.0

    def optimize_symbol_sl_tp(self):
        """Optimize SL/TP parameters per symbol based on historical
        performance"""
        logger.info("Optimizing per-symbol SL/TP parameters...")

        try:
            symbols = self.config.get('trading', {}).get('symbols', [])

            for symbol in symbols:
                # Get recent trades for this symbol
                trades_df = self.get_recent_trades_df(symbol, days=365)

                if len(trades_df) < 30:
                    continue

                # Test different SL/TP combinations
                best_params = self.find_optimal_sl_tp_for_symbol(
                    symbol, trades_df)

                if best_params:
                    # Store optimal parameters
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()

                    cursor.execute('''
                        INSERT OR REPLACE INTO symbol_sl_tp_optimization
                        (symbol, optimal_sl_atr_multiplier,
                         optimal_tp_atr_multiplier,
                         optimal_rr_ratio, avg_win_rate, avg_profit_factor,
                         total_trades,
                         last_updated, confidence_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        best_params['sl_multiplier'],
                        best_params['tp_multiplier'],
                        best_params['rr_ratio'],
                        best_params['win_rate'],
                        best_params['profit_factor'],
                        best_params['total_trades'],
                        datetime.now(),
                        best_params['confidence']
                    ))

                    conn.commit()
                    conn.close()

                    logger.info(
                        f"Optimized SL/TP for {symbol}: SL={
                            best_params['sl_multiplier']:.2f}, " f"TP={
                            best_params['tp_multiplier']:.2f}, RR={
                            best_params['rr_ratio']:.1f}")

        except Exception as e:
            logger.error(f"Error optimizing symbol SL/TP: {e}")

    def find_optimal_sl_tp_for_symbol(
            self,
            symbol: str,
            trades_df: pd.DataFrame) -> Optional[dict]:
        """Find optimal SL/TP multipliers for a specific symbol"""
        try:
            # Test different combinations of SL/TP multipliers
            sl_multipliers = [1.5, 2.0, 2.5, 3.0]
            tp_multipliers = [3.0, 4.0, 5.0, 6.0, 7.0]

            best_score = -float('inf')
            best_params = None

            for sl_mult in sl_multipliers:
                for tp_mult in tp_multipliers:
                    # Simulate trades with these parameters
                    score = self.simulate_sl_tp_performance(
                        trades_df, sl_mult, tp_mult)

                    if score['score'] > best_score:
                        best_score = score
                        rr_ratio = tp_mult / sl_mult
                        best_params = {
                            'sl_multiplier': sl_mult,
                            'tp_multiplier': tp_mult,
                            'rr_ratio': rr_ratio,
                            'win_rate': score['win_rate'],
                            'profit_factor': score['profit_factor'],
                            'total_trades': score['total_trades'],
                            # Confidence based on sample size
                            'confidence': min(1.0, score['total_trades'] / 100)
                        }

            return best_params

        except Exception as e:
            logger.error(f"Error finding optimal SL/TP for {symbol}: {e}")
            return None

    def simulate_sl_tp_performance(
            self,
            trades_df: pd.DataFrame,
            sl_mult: float,
            tp_mult: float) -> dict:
        """Simulate performance with given SL/TP multipliers"""
        try:
            total_trades = len(trades_df)
            winning_trades = 0
            total_profit = 0
            gross_profit = 0
            gross_loss = 0

            for _, trade in trades_df.iterrows():
                # Simulate SL/TP based on ATR multipliers
                # This is a simplified simulation - in reality would need ATR
                # data
                entry_price = trade['entry_price']
                direction = 1 if trade['direction'] == 'BUY' else -1

                # Simplified SL/TP calculation (would use actual ATR in real
                # implementation)
                atr_estimate = abs(
                    trade.get(
                        'exit_price',
                        entry_price) - entry_price) * 0.1  # Rough ATR estimate

                sl_distance = atr_estimate * sl_mult
                tp_distance = atr_estimate * tp_mult

                sl_price = entry_price - (sl_distance * direction)
                tp_price = entry_price + (tp_distance * direction)

                exit_price = trade.get('exit_price', entry_price)
                profit = trade.get('profit', 0)

                # Determine if trade hit SL or TP
                if direction > 0:  # BUY
                    if exit_price >= tp_price:
                        winning_trades += 1
                        gross_profit += profit
                    elif exit_price <= sl_price:
                        gross_loss += abs(profit)
                else:  # SELL
                    if exit_price <= tp_price:
                        winning_trades += 1
                        gross_profit += profit
                    elif exit_price >= sl_price:
                        gross_loss += abs(profit)

                total_profit += profit

            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            profit_factor = gross_profit / \
                gross_loss if gross_loss > 0 else float('inf')

            return {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': total_trades,
                'score': win_rate * min(profit_factor, 3.0)  # Combined score
            }

        except Exception as e:
            logger.error(f"Error simulating SL/TP performance: {e}")
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'total_trades': 0,
                'score': 0}

    def update_entry_filters(self):
        """Learn when NOT to enter trades based on historical outcomes"""
        logger.info("Updating entry filters...")

        try:
            symbols = self.config.get('trading', {}).get('symbols', [])

            for symbol in symbols:
                # Get recent trades
                trades_df = self.get_recent_trades_df(symbol, days=180)

                if len(trades_df) < 50:
                    continue

                # Analyze different filter conditions
                filters = self.analyze_filter_conditions(trades_df)

                # Store filter rules
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                for filter_info in filters:
                    cursor.execute('''
                        INSERT OR REPLACE INTO entry_filters
                        (symbol, filter_type, condition_value, should_enter,
                         total_trades, profitable_trades, win_rate,
                         last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        filter_info['type'],
                        filter_info['condition_value'],
                        filter_info['should_enter'],
                        filter_info['total_trades'],
                        filter_info['profitable_trades'],
                        filter_info['win_rate'],
                        datetime.now()
                    ))

                conn.commit()
                conn.close()

                logger.info(f"Updated entry filters for {symbol}")

        except Exception as e:
            logger.error(f"Error updating entry filters: {e}")

    def analyze_filter_conditions(self, trades_df: pd.DataFrame) -> list:
        """Analyze different conditions to determine when not to enter
        trades"""
        filters = []

        try:
            # High volatility filter - don't enter when ATR is too high
            trades_df['volatility'] = trades_df['profit_pct'].rolling(10).std()
            high_vol_trades = trades_df[
                trades_df['volatility'] > trades_df['volatility'].quantile(
                    0.8)]
            low_vol_trades = trades_df[
                trades_df['volatility'] <= trades_df['volatility'].quantile(
                    0.8)]

            if len(high_vol_trades) > 10:
                high_vol_win_rate = (high_vol_trades['profit_pct'] > 0).mean()
                low_vol_win_rate = (low_vol_trades['profit_pct'] > 0).mean()

                # If high volatility trades perform worse, create filter
                if high_vol_win_rate < low_vol_win_rate * 0.8:
                    filters.append({
                        'type': 'high_volatility_filter',
                        'condition_value': trades_df['volatility'].quantile(
                            0.8),
                        'should_enter': False,
                        'total_trades': len(high_vol_trades),
                        'profitable_trades': int(len(high_vol_trades) *
                                                 high_vol_win_rate),
                        'win_rate': high_vol_win_rate
                    })

            # Time of day filter - avoid certain hours
            trades_df['hour'] = pd.to_datetime(trades_df['timestamp']).dt.hour
            hourly_win_rates = trades_df.groupby(
                'hour')['profit_pct'].apply(lambda x: (x > 0).mean())

            bad_hours = hourly_win_rates[hourly_win_rates < 0.3].index.tolist()

            for hour in bad_hours:
                hour_trades = trades_df[trades_df['hour'] == hour]
                if len(hour_trades) > 5:
                    win_rate = (hour_trades['profit_pct'] > 0).mean()
                    filters.append({
                        'type': 'bad_hour_filter',
                        'condition_value': hour,
                        'should_enter': False,
                        'total_trades': len(hour_trades),
                        'profitable_trades': int(len(hour_trades) * win_rate),
                        'win_rate': win_rate
                    })

        except Exception as e:
            logger.error(f"Error analyzing filter conditions: {e}")

        return filters

    def get_recent_trades_df(
            self,
            symbol: str,
            days: int = 30) -> pd.DataFrame:
        """Get recent trades as DataFrame for analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = '''
                SELECT * FROM trades
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            '''

            cutoff_date = datetime.now() - timedelta(days=days)
            df = pd.read_sql_query(query, conn, params=[symbol, cutoff_date])
            conn.close()

            return df

        except Exception as e:
            logger.error(f"Error getting recent trades for {symbol}: {e}")
            return pd.DataFrame()

    def update_regime_data(self, symbol: str, regime_analysis):
        """Update regime data for adaptive learning"""
        if not self.regime_enabled:
            return

        try:
            # Store regime history
            if symbol not in self.regime_history:
                self.regime_history[symbol] = []

            self.regime_history[symbol].append({
                'timestamp': datetime.now(),
                'regime': regime_analysis.primary_regime.value,
                'confidence': regime_analysis.confidence,
                'adx': regime_analysis.adx_value,
                'volatility_ratio': regime_analysis.volatility_ratio,
                'trend_strength': regime_analysis.trend_strength,
                'regime_score': regime_analysis.regime_score
            })

            # Keep only recent history
            if len(self.regime_history[symbol]) > self.regime_history_length:
                self.regime_history[symbol] = self.regime_history[symbol][-self.regime_history_length:]

            # Update regime performance tracking
            regime_key = (symbol, regime_analysis.primary_regime.value)
            if regime_key not in self.regime_performance:
                self.regime_performance[regime_key] = {
                    'total_trades': 0,
                    'profitable_trades': 0,
                    'total_profit': 0.0,
                    'win_rate': 0.0,
                    'avg_profit': 0.0
                }

            logger.debug(f"Updated regime data for {symbol}: {regime_analysis.primary_regime.value}")

        except Exception as e:
            logger.error(f"Error updating regime data for {symbol}: {e}")

    def get_regime_adapted_parameters(self, symbol: str, base_params: dict) -> dict:
        """Get parameters adapted to current market regime"""
        if not self.regime_enabled or symbol not in self.regime_history:
            return base_params

        try:
            current_regime = self.regime_history[symbol][-1] if self.regime_history[symbol] else None
            if not current_regime:
                return base_params

            regime_key = (symbol, current_regime['regime'])
            performance = self.regime_performance.get(regime_key, {})

            adapted_params = base_params.copy()

            # Adapt based on regime performance
            if performance.get('total_trades', 0) >= 10:
                win_rate = performance.get('win_rate', 0.5)

                # Adjust risk based on regime performance
                if win_rate > 0.6:  # Good regime
                    adapted_params['risk_multiplier'] = min(
                        base_params.get('risk_multiplier', 1.0) * 1.2, 2.0)
                elif win_rate < 0.4:  # Bad regime
                    adapted_params['risk_multiplier'] = max(
                        base_params.get('risk_multiplier', 1.0) * 0.8, 0.5)

                # Adjust holding time based on regime
                if current_regime['regime'] in ['trending_up', 'trending_down']:
                    adapted_params['max_holding_hours'] = min(
                        base_params.get('max_holding_hours', 4) * 1.5, 12)
                elif current_regime['regime'] == 'high_volatility':
                    adapted_params['max_holding_hours'] = max(
                        base_params.get('max_holding_hours', 4) * 0.7, 1)

            # Apply confidence-based adjustments
            confidence = current_regime.get('confidence', 0.5)
            if confidence > 0.8:
                adapted_params['min_signal_strength'] = max(
                    base_params.get('min_signal_strength', 0.6) * 0.9, 0.4)
            elif confidence < 0.3:
                adapted_params['min_signal_strength'] = min(
                    base_params.get('min_signal_strength', 0.6) * 1.2, 0.9)

            logger.debug(f"Adapted parameters for {symbol} in {current_regime['regime']} regime: {adapted_params}")
            return adapted_params

        except Exception as e:
            logger.error(f"Error getting regime-adapted parameters for {symbol}: {e}")
            return base_params

    # ===== GETTER METHODS FOR NEW FEATURES =====

    def get_entry_timing_recommendation(
            self, symbol: str, current_hour: int) -> dict:
        """Get entry timing recommendation for current conditions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM entry_timing_analysis
                WHERE symbol = ? AND hour_of_day = ?
                ORDER BY last_updated DESC LIMIT 1
            ''', (symbol, current_hour))

            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    'recommended': row[8] > 0.5,  # win_rate > 50%
                    'win_rate': row[8],
                    'avg_profit': row[7],
                    'total_trades': row[5]
                }

            return {
                'recommended': True,
                'win_rate': 0.5,
                'avg_profit': 0,
                'total_trades': 0}

        except Exception as e:
            logger.error(f"Error getting entry timing for {symbol}: {e}")
            return {
                'recommended': True,
                'win_rate': 0.5,
                'avg_profit': 0,
                'total_trades': 0}

    def get_daily_temporal_recommendation(self, symbol: str, day_of_month: int, month_of_year: int, year: int) -> dict:
        """Get daily temporal recommendation for current date"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM daily_temporal_analysis
                WHERE symbol = ? AND day_of_month = ? AND month_of_year = ? AND year = ?
                ORDER BY last_updated DESC LIMIT 1
            ''', (symbol, day_of_month, month_of_year, year))

            row = cursor.fetchone()
            conn.close()

            if row and row[4] >= 3:  # Minimum 3 trades
                win_rate = row[7]
                sharpe_ratio = row[10] if row[10] else 0
                max_drawdown = row[11] if row[11] else 0

                # Recommend if win rate > 55% and Sharpe > 0.5 and drawdown < 10%
                recommended = (win_rate > 0.55 and sharpe_ratio > 0.5 and max_drawdown < 0.1)

                return {
                    'recommended': recommended,
                    'win_rate': win_rate,
                    'avg_profit': row[6],
                    'total_trades': row[4],
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'confidence': min(1.0, row[4] / 20)  # Confidence based on sample size
                }

            return {
                'recommended': True,  # Default to allowing trades if no data
                'win_rate': 0.5,
                'avg_profit': 0,
                'total_trades': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'confidence': 0
            }

        except Exception as e:
            logger.error(f"Error getting daily temporal recommendation for {symbol}: {e}")
            return {
                'recommended': True,
                'win_rate': 0.5,
                'avg_profit': 0,
                'total_trades': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'confidence': 0
            }

    def get_weekly_temporal_recommendation(self, symbol: str, week_of_year: int, year: int) -> dict:
        """Get weekly temporal recommendation for current week"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM weekly_temporal_analysis
                WHERE symbol = ? AND week_of_year = ? AND year = ?
                ORDER BY last_updated DESC LIMIT 1
            ''', (symbol, week_of_year, year))

            row = cursor.fetchone()
            conn.close()

            if row and row[3] >= 5:  # Minimum 5 trades
                win_rate = row[6]
                sharpe_ratio = row[9] if row[9] else 0
                max_drawdown = row[10] if row[10] else 0

                # Recommend if win rate > 52% and Sharpe > 0.3 and drawdown < 15%
                recommended = (win_rate > 0.52 and sharpe_ratio > 0.3 and max_drawdown < 0.15)

                return {
                    'recommended': recommended,
                    'win_rate': win_rate,
                    'avg_profit': row[5],
                    'total_trades': row[3],
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'confidence': min(1.0, row[3] / 50)  # Confidence based on sample size
                }

            return {
                'recommended': True,
                'win_rate': 0.5,
                'avg_profit': 0,
                'total_trades': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'confidence': 0
            }

        except Exception as e:
            logger.error(f"Error getting weekly temporal recommendation for {symbol}: {e}")
            return {
                'recommended': True,
                'win_rate': 0.5,
                'avg_profit': 0,
                'total_trades': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'confidence': 0
            }

    def get_monthly_temporal_recommendation(self, symbol: str, month_of_year: int, year: int) -> dict:
        """Get monthly temporal recommendation for current month"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM monthly_temporal_analysis
                WHERE symbol = ? AND month_of_year = ? AND year = ?
                ORDER BY last_updated DESC LIMIT 1
            ''', (symbol, month_of_year, year))

            row = cursor.fetchone()
            conn.close()

            if row and row[3] >= 10:  # Minimum 10 trades
                win_rate = row[6]
                sharpe_ratio = row[9] if row[9] else 0
                max_drawdown = row[10] if row[10] else 0

                # Recommend if win rate > 50% and Sharpe > 0.2 and drawdown < 20%
                recommended = (win_rate > 0.5 and sharpe_ratio > 0.2 and max_drawdown < 0.2)

                return {
                    'recommended': recommended,
                    'win_rate': win_rate,
                    'avg_profit': row[5],
                    'total_trades': row[3],
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'confidence': min(1.0, row[3] / 100)  # Confidence based on sample size
                }

            return {
                'recommended': True,
                'win_rate': 0.5,
                'avg_profit': 0,
                'total_trades': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'confidence': 0
            }

        except Exception as e:
            logger.error(f"Error getting monthly temporal recommendation for {symbol}: {e}")
            return {
                'recommended': True,
                'win_rate': 0.5,
                'avg_profit': 0,
                'total_trades': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'confidence': 0
            }

    def get_yearly_temporal_recommendation(self, symbol: str, year: int) -> dict:
        """Get yearly temporal recommendation for current year"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM yearly_temporal_analysis
                WHERE symbol = ? AND year = ?
                ORDER BY last_updated DESC LIMIT 1
            ''', (symbol, year))

            row = cursor.fetchone()
            conn.close()

            if row and row[2] >= 20:  # Minimum 20 trades
                win_rate = row[5]
                sharpe_ratio = row[8] if row[8] else 0
                max_drawdown = row[9] if row[9] else 0

                # Recommend if win rate > 48% and Sharpe > 0.1 and drawdown < 25%
                recommended = (win_rate > 0.48 and sharpe_ratio > 0.1 and max_drawdown < 0.25)

                return {
                    'recommended': recommended,
                    'win_rate': win_rate,
                    'avg_profit': row[4],
                    'total_trades': row[2],
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'confidence': min(1.0, row[2] / 200)  # Confidence based on sample size
                }

            return {
                'recommended': True,
                'win_rate': 0.5,
                'avg_profit': 0,
                'total_trades': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'confidence': 0
            }

        except Exception as e:
            logger.error(f"Error getting yearly temporal recommendation for {symbol}: {e}")
            return {
                'recommended': True,
                'win_rate': 0.5,
                'avg_profit': 0,
                'total_trades': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'confidence': 0
            }

    def get_comprehensive_temporal_recommendation(self, symbol: str) -> dict:
        """Get comprehensive temporal recommendation combining all timeframes"""
        try:
            # Check if temporal analysis is enabled
            if not self.temporal_enabled:
                return {
                    'recommended': True,
                    'confidence': 1.0,
                    'reason': 'Temporal analysis disabled'
                }

            now = datetime.now()
            current_hour = now.hour
            current_day = now.day
            current_month = now.month
            current_year = now.year
            current_week = now.isocalendar().week

            # Get recommendations from enabled timeframes
            recommendations = []
            weights = []

            # Hourly (always included if temporal enabled)
            hourly_rec = self.get_entry_timing_recommendation(symbol, current_hour)
            # Add confidence key for consistency (use win_rate as confidence proxy)
            hourly_rec['confidence'] = hourly_rec.get('win_rate', 0.5)
            recommendations.append(hourly_rec)
            weights.append(0.3)

            # Daily
            if self.temporal_timeframes.get('daily', True):
                daily_rec = self.get_daily_temporal_recommendation(symbol, current_day, current_month, current_year)
                recommendations.append(daily_rec)
                weights.append(0.25)

            # Weekly
            if self.temporal_timeframes.get('weekly', True):
                weekly_rec = self.get_weekly_temporal_recommendation(symbol, current_week, current_year)
                recommendations.append(weekly_rec)
                weights.append(0.2)

            # Monthly
            if self.temporal_timeframes.get('monthly', True):
                monthly_rec = self.get_monthly_temporal_recommendation(symbol, current_month, current_year)
                recommendations.append(monthly_rec)
                weights.append(0.15)

            # Yearly
            if self.temporal_timeframes.get('yearly', True):
                yearly_rec = self.get_yearly_temporal_recommendation(symbol, current_year)
                recommendations.append(yearly_rec)
                weights.append(0.1)

            # Calculate weighted recommendation using configured weights
            weighted_score = 0
            total_weight = 0
            confidence_sum = 0

            for rec, weight in zip(recommendations, weights):
                if rec['confidence'] > 0:  # Only include if we have data
                    weighted_score += (1 if rec['recommended'] else 0) * weight * rec['confidence']
                    total_weight += weight * rec['confidence']
                    confidence_sum += rec['confidence']

            if total_weight > 0:
                final_score = weighted_score / total_weight
                overall_recommended = final_score > 0.5
                avg_confidence = confidence_sum / len([r for r in recommendations if r['confidence'] > 0])
            else:
                overall_recommended = True  # Default to allowing trades if no data
                avg_confidence = 0
                final_score = 0.5

            # Apply confidence threshold
            if avg_confidence < self.temporal_confidence_threshold:
                overall_recommended = True  # Allow trades if confidence is too low

            return {
                'recommended': overall_recommended,
                'confidence': avg_confidence,
                'confidence_score': final_score,
                'reason': self._get_recommendation_reason(
                    recommendations, overall_recommended)
            }

        except Exception as e:
            logger.error(f"Error getting comprehensive temporal recommendation for {symbol}: {e}")
            return {
                'recommended': True,
                'confidence': 0,
                'reason': f'Error: {str(e)}'
            }

    def _get_recommendation_reason(self, recommendations: list, overall_recommended: bool) -> str:
        """Generate human-readable reason for the recommendation"""
        try:
            reasons = []

            timeframe_names = ['hourly', 'daily', 'weekly', 'monthly', 'yearly']
            for rec, name in zip(recommendations, timeframe_names):
                if rec['confidence'] > 0:
                    status = "favorable" if rec['recommended'] else "unfavorable"
                    win_rate = rec['win_rate'] * 100
                    reasons.append(f"{name}: {status} ({win_rate:.1f}% win rate)")

            if overall_recommended:
                return f"Overall favorable for trading. {', '.join(reasons)}"
            else:
                return f"Overall unfavorable for trading. {', '.join(reasons)}"

        except Exception:
            return "Recommendation based on temporal analysis"

    def get_symbol_sl_tp_params(self, symbol: str) -> dict:
        """Get optimized SL/TP parameters for a symbol"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM symbol_sl_tp_optimization
                WHERE symbol = ?
                ORDER BY last_updated DESC LIMIT 1
            ''', (symbol,))

            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    'sl_atr_multiplier': row[2],
                    'tp_atr_multiplier': row[3],
                    'rr_ratio': row[4],
                    'win_rate': row[5],
                    'profit_factor': row[6],
                    'confidence': row[9]
                }

            # Return global defaults if no symbol-specific data
            return {
                'sl_atr_multiplier': self.adaptive_params.get(
                    'stop_loss_atr_multiplier',
                    2.0),
                'tp_atr_multiplier': self.adaptive_params.get(
                    'take_profit_atr_multiplier',
                    6.0),
                'rr_ratio': 3.0,
                'win_rate': 0.5,
                'profit_factor': 1.0,
                'confidence': 0.0}

        except Exception as e:
            logger.error(f"Error getting SL/TP params for {symbol}: {e}")
            return {
                'sl_atr_multiplier': 2.0,
                'tp_atr_multiplier': 6.0,
                'rr_ratio': 3.0,
                'win_rate': 0.5,
                'profit_factor': 1.0,
                'confidence': 0.0
            }

    def should_enter_based_on_filters(
            self,
            symbol: str,
            current_conditions: dict) -> bool:
        """Check if entry filters allow trading"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM entry_filters
                WHERE symbol = ? AND should_enter = 0
            ''', (symbol,))

            filters = cursor.fetchall()
            conn.close()

            for filter_row in filters:
                filter_type = filter_row[2]
                condition_value = filter_row[3]

                # Check filter conditions
                if filter_type == 'high_volatility_filter':
                    current_vol = current_conditions.get('volatility', 0)
                    if current_vol > condition_value:
                        return False

                elif filter_type == 'bad_hour_filter':
                    current_hour = current_conditions.get('hour', 0)
                    if current_hour == condition_value:
                        return False

            return True  # No filters triggered

        except Exception as e:
            logger.error(f"Error checking entry filters for {symbol}: {e}")
            return True  # Allow entry on error

    # ===== NEW ADVANCED LEARNING METHODS =====

    def optimize_technical_indicators(self):
        """Optimize technical indicator parameters based on historical
        performance"""
        logger.info("Optimizing technical indicator parameters...")

        try:
            symbols = self.config.get('trading', {}).get('symbols', [])

            for symbol in symbols:
                # Test different parameter combinations for each indicator
                self.optimize_indicator_for_symbol(
                    symbol, 'vwap', 'period', [10, 20, 30, 40, 50])
                self.optimize_indicator_for_symbol(
                    symbol, 'ema_fast', 'period', [5, 9, 12, 15, 20])
                self.optimize_indicator_for_symbol(
                    symbol, 'ema_slow', 'period', [15, 21, 25, 30, 50])
                self.optimize_indicator_for_symbol(
                    symbol, 'rsi', 'period', [7, 14, 21])
                self.optimize_indicator_for_symbol(
                    symbol, 'atr', 'period', [7, 14, 21])

        except Exception as e:
            logger.error(f"Error optimizing technical indicators: {e}")

    def optimize_indicator_for_symbol(
            self,
            symbol: str,
            indicator_name: str,
            param_name: str,
            test_values: list):
        """Test different parameter values for a specific indicator and
        symbol"""
        try:
            # Get recent trades for this symbol
            trades_df = self.get_recent_trades_df(symbol, days=365)

            if len(trades_df) < 50:
                return

            best_score = -float('inf')
            best_value = test_values[0]

            for test_value in test_values:
                # Simulate performance with this parameter value
                score = self.simulate_indicator_performance(
                    trades_df, indicator_name, param_name, test_value)

                if score > best_score:
                    best_score = score
                    best_value = test_value

            # Store optimal parameter
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO technical_indicator_optimization
                (symbol, indicator_name, parameter_name, optimal_value,
                 performance_score, total_trades, last_updated,
                 confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                indicator_name,
                param_name,
                best_value,
                best_score,
                len(trades_df),
                datetime.now(),
                # Confidence based on sample size
                min(1.0, len(trades_df) / 200)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(
                f"Error optimizing {indicator_name} for {symbol}: {e}")

    def simulate_indicator_performance(
            self,
            trades_df: pd.DataFrame,
            indicator_name: str,
            param_name: str,
            param_value: float) -> float:
        """Simulate trading performance with specific indicator parameter"""
        try:
            # This is a simplified simulation - in practice would
            # recalculate indicators
            # For now, use a proxy based on existing trade data
            win_rate = (trades_df['profit_pct'] > 0).mean()

            # Add some variation based on parameter value
            # (simulating optimization)
            # Better parameters should perform better
            optimal_ranges = {
                'vwap_period': (15, 25),
                'ema_fast_period': (8, 12),
                'ema_slow_period': (20, 30),
                'rsi_period': (12, 16),
                'atr_period': (12, 16)
            }

            if indicator_name in optimal_ranges:
                optimal_min, optimal_max = optimal_ranges[indicator_name]
                if optimal_min <= param_value <= optimal_max:
                    # Parameter is in optimal range - boost performance
                    win_rate *= 1.1
                else:
                    # Parameter is outside optimal range - reduce performance
                    win_rate *= 0.9

            return win_rate

        except Exception as e:
            logger.error(f"Error simulating indicator performance: {e}")
            return 0.5

    def optimize_fundamental_weights(self):
        """Optimize fundamental source weights based on prediction accuracy"""
        logger.info("Optimizing fundamental source weights...")

        try:
            # Test different weight combinations for fundamental sources
            sources = [
                'myfxbook',
                'fxstreet',
                'fxblue',
                'investing',
                'forexclientsentiment']

            # Simple optimization: test a few combinations
            test_combinations = [
                [0.25, 0.25, 0.2, 0.2, 0.1],  # Current weights
                [0.3, 0.2, 0.2, 0.2, 0.1],   # More weight to myfxbook
                [0.2, 0.3, 0.2, 0.2, 0.1],   # More weight to fxstreet
                [0.2, 0.2, 0.3, 0.2, 0.1],   # More weight to fxblue
                [0.2, 0.2, 0.2, 0.3, 0.1],   # More weight to investing
            ]

            best_weights = test_combinations[0]
            best_score = self.evaluate_fundamental_weights(
                test_combinations[0])

            for weights in test_combinations[1:]:
                score = self.evaluate_fundamental_weights(weights)
                if score > best_score:
                    best_score = score
                    best_weights = weights

            # Store optimal weights for each source
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            sources = [
                'myfxbook',
                'fxstreet',
                'fxblue',
                'investing',
                'forexclientsentiment']
            for i, source in enumerate(sources):
                cursor.execute('''
                    INSERT OR REPLACE INTO fundamental_weight_optimization
                    (source_name, optimal_weight, prediction_accuracy,
                     total_predictions, last_updated, market_condition)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    source,
                    best_weights[i],
                    best_score,
                    100,  # Placeholder
                    datetime.now(),
                    'general'  # Could be extended for different
                    # market conditions
                ))

            conn.commit()
            conn.close()

            logger.info(
                f"Optimized fundamental weights: "
                f"{dict(zip(sources, best_weights))}")

        except Exception as e:
            logger.error(f"Error optimizing fundamental weights: {e}")

    def evaluate_fundamental_weights(self, weights: list) -> float:
        """Evaluate performance of fundamental weight combination"""
        try:
            # Simplified evaluation - in practice would test against
            # historical data
            # For now, prefer balanced weights
            total_weight = sum(weights)
            if abs(total_weight - 1.0) > 0.01:  # Must sum to 1.0
                return 0.0

            # Score based on balance and reasonable ranges
            # First 4 sources should be balanced
            balance_score = 1.0 - abs(0.25 - sum(weights[:4]) / 4)
            return balance_score * 0.8  # Scale to reasonable score

        except Exception as e:
            logger.error(f"Error evaluating fundamental weights: {e}")
            return 0.5

    def analyze_economic_calendar_impact(self):
        """Analyze impact of economic events on trading performance"""
        logger.info("Analyzing economic calendar impact...")

        try:
            # This is a placeholder for economic calendar integration
            # In a real implementation, you would:
            # 1. Fetch economic calendar data from APIs
            # 2. Correlate events with trade performance
            # 3. Learn to avoid trading during high-impact events

            # For now, create some sample learning based on hypothetical events
            sample_events = [
                ('Non-Farm Payrolls', 'high', 1, 4),
                ('FOMC Meeting', 'high', 2, 24),
                ('CPI', 'medium', 1, 2),
                ('Retail Sales', 'medium', 1, 2),
            ]

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for event_name, impact, hours_before, hours_after in sample_events:
                # Simulate analysis - in practice would analyze real trade data
                should_avoid = impact == 'high'
                # Lower performance during high impact
                avg_performance = 0.45 if should_avoid else 0.55

                cursor.execute('''
                    INSERT OR REPLACE INTO economic_calendar_impact
                    (event_name, event_impact, hours_before_event,
                     hours_after_event, avg_trade_performance, total_trades,
                     should_avoid_trading, last_updated, currency_pair)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event_name,
                    impact,
                    hours_before,
                    hours_after,
                    avg_performance,
                    50,  # Sample trade count
                    should_avoid,
                    datetime.now(),
                    'USD'  # Could be extended per currency
                ))

            conn.commit()
            conn.close()

            logger.info("Updated economic calendar impact analysis")

        except Exception as e:
            logger.error(f"Error analyzing economic calendar impact: {e}")

    def analyze_interest_rate_impact(self):
        """Analyze impact of interest rate changes on currency performance"""
        logger.info("Analyzing interest rate impact...")

        try:
            # This is a placeholder for interest rate analysis
            # In a real implementation, you would:
            # 1. Track central bank rate decisions
            # 2. Monitor currency reactions to rate changes
            # 3. Learn correlations between rate differentials and currency
            # movements

            currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD']
            time_horizons = ['1h', '4h', '1d', '1w']

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for currency in currencies:
                for horizon in time_horizons:
                    # Simulate interest rate impact analysis
                    # In practice, this would analyze real rate change data
                    rate_change = np.random.normal(
                        0, 0.25)  # Sample rate change
                    price_movement = rate_change * \
                        np.random.uniform(0.5, 2.0)  # Correlated movement
                    correlation = np.random.uniform(
                        0.3, 0.8)  # Correlation strength

                    cursor.execute('''
                        INSERT OR REPLACE INTO interest_rate_impact
                        (currency, rate_change, time_horizon,
                         avg_price_movement, total_observations,
                         correlation_strength, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        currency,
                        rate_change,
                        horizon,
                        price_movement,
                        100,  # Sample observation count
                        correlation,
                        datetime.now()
                    ))

            conn.commit()
            conn.close()

            logger.info("Updated interest rate impact analysis")

        except Exception as e:
            logger.error(f"Error analyzing interest rate impact: {e}")

    def optimize_sentiment_parameters(self):
        """Optimize sentiment analysis parameters"""
        logger.info("Optimizing sentiment parameters...")

        try:
            # Test different sentiment parameter combinations
            param_ranges = {
                'sentiment_threshold': [0.1, 0.2, 0.3, 0.4, 0.5],
                'sentiment_time_decay': [0.5, 1.0, 1.5, 2.0],
                'keyword_weight_multiplier': [0.8, 1.0, 1.2, 1.5]
            }

            # Simple grid search for optimal parameters
            best_params = {}
            best_score = -float('inf')

            for threshold in param_ranges['sentiment_threshold']:
                for decay in param_ranges['sentiment_time_decay']:
                    for multiplier in param_ranges[
                            'keyword_weight_multiplier']:
                        score = self.evaluate_sentiment_params(
                            threshold, decay, multiplier)

                        if score > best_score:
                            best_score = score
                            best_params = {
                                'threshold': threshold,
                                'decay': decay,
                                'multiplier': multiplier
                            }

            # Store optimal parameters
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for param_name, value in best_params.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO sentiment_parameter_optimization
                    (parameter_name, optimal_value, performance_impact,
                     total_trades, last_updated, market_condition)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    param_name,
                    value,
                    best_score,
                    200,  # Sample trade count
                    datetime.now(),
                    'general'
                ))

            conn.commit()
            conn.close()

            logger.info(f"Optimized sentiment parameters: {best_params}")

        except Exception as e:
            logger.error(f"Error optimizing sentiment parameters: {e}")

    def evaluate_sentiment_params(
            self,
            threshold: float,
            decay: float,
            multiplier: float) -> float:
        """Evaluate sentiment parameter combination"""
        try:
            # Simplified evaluation - in practice would test against
            # historical sentiment data
            # Prefer moderate threshold, moderate decay, and balanced
            # multiplier
            threshold_score = 1.0 - \
                abs(threshold - 0.3) / 0.3  # Optimal around 0.3
            decay_score = 1.0 - abs(decay - 1.0) / \
                1.0          # Optimal around 1.0
            multiplier_score = 1.0 - \
                abs(multiplier - 1.0) / 0.5  # Optimal around 1.0

            return (threshold_score + decay_score + multiplier_score) / 3.0

        except Exception as e:
            logger.error(f"Error evaluating sentiment params: {e}")
            return 0.5

    # ===== GETTER METHODS FOR NEW OPTIMIZATIONS =====

    def get_optimized_technical_params(self, symbol: str) -> dict:
        """Get optimized technical indicator parameters for a symbol"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT indicator_name, parameter_name, optimal_value
                FROM technical_indicator_optimization
                WHERE symbol = ? AND confidence_score > 0.5
            ''', (symbol,))

            params = {}
            for row in cursor.fetchall():
                indicator, param, value = row[0], row[1], row[2]
                key = f"{indicator}_{param}"
                params[key] = value

            conn.close()
            return params

        except Exception as e:
            logger.error(f"Error getting technical params for {symbol}: {e}")
            return {}

    def get_optimized_fundamental_weights(self) -> dict:
        """Get optimized fundamental source weights"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT source_name, optimal_weight
                FROM fundamental_weight_optimization
                ORDER BY prediction_accuracy DESC
            ''')

            weights = {row[0]: row[1] for row in cursor.fetchall()}
            conn.close()
            return weights

        except Exception as e:
            logger.error(f"Error getting fundamental weights: {e}")
            return {}

    def should_avoid_economic_events(self, hours_ahead: int = 24) -> list:
        """Check if trading should be avoided due to upcoming economic
        events"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT event_name, should_avoid_trading
                FROM economic_calendar_impact
                WHERE should_avoid_trading = 1 AND hours_before_event <= ?
            ''', (hours_ahead,))

            events_to_avoid = [row[0] for row in cursor.fetchall()]
            conn.close()
            return events_to_avoid

        except Exception as e:
            logger.error(f"Error checking economic events: {e}")
            return []

    def get_interest_rate_expectations(self, currency: str) -> dict:
        """Get interest rate impact expectations for a currency"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT time_horizon, correlation_strength, avg_price_movement
                FROM interest_rate_impact
                WHERE currency = ?
                ORDER BY correlation_strength DESC
            ''', (currency,))

            expectations = {}
            for row in cursor.fetchall():
                horizon, correlation, movement = row
                expectations[horizon] = {
                    'correlation': correlation,
                    'expected_movement': movement
                }

            conn.close()
            return expectations

        except Exception as e:
            logger.error(
                f"Error getting interest rate expectations for "
                f"{currency}: {e}")
            return {}

    def get_updated_sl_tp_params(
            self,
            symbol: str,
            trade_timestamp: datetime) -> Optional[dict]:
        """Get updated SL/TP parameters if they've changed since trade
        was opened"""
        try:
            # Check if SL/TP parameters have been updated since the trade was
            # opened
            symbol_params = self.get_symbol_sl_tp_params(symbol)

            if symbol_params['confidence'] < 0.5:
                return None  # Not confident enough in new parameters

            # Check when the parameters were last updated
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT last_updated FROM symbol_sl_tp_optimization
                WHERE symbol = ? AND confidence_score > 0.5
            ''', (symbol,))

            result = cursor.fetchone()
            conn.close()

            if result:
                last_update = datetime.fromisoformat(result[0])
                if last_update > trade_timestamp:
                    # Parameters have been updated since trade was opened
                    return symbol_params

            return None  # No updates since trade was opened

        except Exception as e:
            logger.error(
                f"Error checking for updated SL/TP params for {symbol}: {e}")
            return None

    def should_adjust_existing_trade(
            self,
            symbol: str,
            current_sl: float,
            current_tp: float,
            trade_timestamp: datetime) -> dict:
        """Determine if an existing trade should have its SL/TP adjusted
        based on new learning"""
        try:
            updated_params = self.get_updated_sl_tp_params(
                symbol, trade_timestamp)

            if not updated_params:
                return {'should_adjust': False}

            # Calculate what the new SL/TP should be based on current
            # market conditions
            # This would need current price data - for now, return the updated
            # parameters
            return {
                'should_adjust': True,
                'new_sl_atr_multiplier': updated_params['sl_atr_multiplier'],
                'new_tp_atr_multiplier': updated_params['tp_atr_multiplier'],
                'confidence': updated_params['confidence'],
                'reason': 'Updated optimization parameters available'
            }

        except Exception as e:
            logger.error(
                f"Error determining trade adjustment for {symbol}: {e}")
            return {'should_adjust': False}

    def record_position_adjustment(
            self,
            ticket: int,
            symbol: str,
            old_sl: float,
            old_tp: float,
            new_sl: float,
            new_tp: float,
            adjustment_reason: str):
        """Record when a position's SL/TP was adjusted for learning purposes"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO position_adjustments
                (ticket, symbol, old_sl, old_tp, new_sl, new_tp,
                 adjustment_reason, adjustment_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticket,
                symbol,
                old_sl,
                old_tp,
                new_sl,
                new_tp,
                adjustment_reason,
                datetime.now()
            ))

            conn.commit()
            conn.close()

            logger.info(
                f"Recorded position adjustment for {symbol} ticket {ticket}: "
                f"{adjustment_reason}")

        except Exception as e:
            logger.error(f"Error recording position adjustment: {e}")

    def analyze_adjustment_performance(self):
        """Analyze how position adjustments performed to improve future
        adjustments"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get adjustment history with outcomes
            cursor.execute('''
                SELECT pa.ticket, pa.symbol, pa.adjustment_reason,
                       pa.adjustment_timestamp, t.profit_pct, t.exit_price,
                       t.duration_minutes
                FROM position_adjustments pa
                LEFT JOIN trades t ON pa.ticket = t.ticket
                WHERE t.timestamp > pa.adjustment_timestamp
                AND t.timestamp <= datetime(pa.adjustment_timestamp,
                                             '+24 hours')
            ''')

            adjustments = cursor.fetchall()
            conn.close()

            if not adjustments:
                return

            # Analyze adjustment performance
            successful_adjustments = 0
            total_adjustments = len(adjustments)

            for adj in adjustments:
                ticket, symbol, reason, adj_time, profit_pct, exit_price, \
                    duration = adj

                if profit_pct and profit_pct > 0:
                    successful_adjustments += 1

                # Log adjustment outcome
                outcome = "SUCCESS" if (
                    profit_pct and profit_pct > 0) else "FAILURE"
                logger.info(f"Adjustment analysis - {symbol} ticket "
                            f"{ticket}: {outcome} "
                            f"(reason: {reason}, profit: {profit_pct:.2f}%)")

            success_rate = successful_adjustments / \
                total_adjustments if total_adjustments > 0 else 0

            logger.info(f"Position adjustment analysis: "
                        f"{successful_adjustments}/{total_adjustments} "
                        f"successful ({success_rate:.1%})")

            # Store analysis results for future learning
            self.store_adjustment_learning(success_rate, adjustments)

        except Exception as e:
            logger.error(f"Error analyzing adjustment performance: {e}")

    def store_adjustment_learning(
            self,
            success_rate: float,
            adjustments: list):
        """Store learning from adjustment performance"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO adjustment_performance_analysis
                (analysis_date, success_rate, total_adjustments,
                 successful_adjustments, avg_profit_impact)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                success_rate,
                len(adjustments),
                int(success_rate * len(adjustments)),
                0.0  # Could calculate average profit impact
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error storing adjustment learning: {e}")
