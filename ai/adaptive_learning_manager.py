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

# Lightweight schedule shim: use real 'schedule' package if installed, otherwise provide minimal compatibility.
try:
    import schedule as _real_schedule  # type: ignore
    schedule = _real_schedule
except Exception:
    schedule_jobs = []
    schedule_lock = threading.Lock()

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
                    schedule_jobs.append({'func': job, 'interval': interval_seconds, 'next_run': next_run, 'type': 'interval'})
                elif self.unit == 'day':
                    at = self.at_time or "00:00"
                    try:
                        h, m = [int(x) for x in at.split(":")]
                    except Exception:
                        h, m = 0, 0
                    now_dt = datetime.now()
                    next_run_dt = now_dt.replace(hour=h, minute=m, second=0, microsecond=0)
                    if next_run_dt <= now_dt:
                        next_run_dt = next_run_dt + timedelta(days=1)
                    schedule_jobs.append({'func': job, 'interval': 24*3600, 'next_run': next_run_dt.timestamp(), 'type': 'daily'})
                else:
                    interval_seconds = float(self.interval) if self.interval is not None else 1.0
                    next_run = time.time() + interval_seconds
                    schedule_jobs.append({'func': job, 'interval': interval_seconds, 'next_run': next_run, 'type': 'interval'})
            return True

    def every(interval=None):
        return Every(interval)

    def run_pending():
        now = time.time()
        with schedule_lock:
            for job in list(schedule_jobs):
                if now >= job['next_run']:
                    try:
                        threading.Thread(target=job['func'], daemon=True).start()
                    except Exception:
                        pass
                    if job['type'] == 'interval':
                        job['next_run'] = now + job['interval']
                    elif job['type'] == 'daily':
                        job['next_run'] = job['next_run'] + 24*3600

    # expose a minimal module-like object compatible with usage in this file
    schedule = type('ScheduleModule', (), {'every': every, 'run_pending': run_pending})
from typing import Optional
import pandas as pd
import numpy as np
from collections import deque
import sqlite3
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class AdaptiveLearningManager:
    """
    Manages continuous learning and adaptation for FX-Ai
    - Periodic model retraining
    - Performance-based weight adjustment
    - Parameter optimization
    - Trade feedback integration
    """

    def __init__(self, config: dict, ml_predictor=None, backtest_engine=None, risk_manager=None, mt5_connector=None):
        """Initialize the adaptive learning system"""
        self.config = config
        self.ml_predictor = ml_predictor
        self.backtest_engine = backtest_engine
        self.risk_manager = risk_manager
        self.mt5 = mt5_connector

        # Learning configuration
        self.retrain_interval = config.get('adaptive_learning', {}).get('retrain_interval_hours', 24)
        self.min_trades_for_update = config.get('adaptive_learning', {}).get('min_trades_for_update', 50)
        self.performance_window = config.get('adaptive_learning', {}).get('performance_window_days', 1825)
        self.adaptation_rate = config.get('adaptive_learning', {}).get('adaptation_rate', 0.1)

        # Performance tracking
        self.trade_history = deque(maxlen=1000)
        self.performance_metrics = {}
        self.model_versions = {}
        self.parameter_history = []

        # Signal weights (will be adapted based on performance)
        self.signal_weights = config.get('adaptive_learning', {}).get('signal_weights', {
            'ml_prediction': 0.30,
            'technical_score': 0.25,
            'sentiment_score': 0.20,
            'fundamental_score': 0.15,
            'support_resistance': 0.10
        })

        # Trading parameters (will be optimized)
        self.adaptive_params = config.get('adaptive_learning', {}).get('adaptive_params', {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'min_signal_strength': 0.6,
            'max_correlation': 0.8,
            'risk_multiplier': 1.0,
            'trailing_stop_distance': 20,
            'stop_loss_atr_multiplier': 2.0,  # ATR multiplier for stop loss
            'take_profit_atr_multiplier': 6.0,  # ATR multiplier for take profit (1:3 ratio)
            'max_holding_minutes': 480,  # Maximum holding time in minutes (8 hours)
            'min_holding_minutes': 15,   # Minimum holding time in minutes
            'optimal_holding_hours': 4.0  # Target optimal holding period
        })

        # Force reset risk multiplier to prevent inflation
        self.adaptive_params['risk_multiplier'] = 1.0

        # Initialize database for performance tracking
        self.init_database()

        # Schedule periodic tasks
        self.schedule_tasks()

        # Start background thread for continuous learning
        self.learning_thread = threading.Thread(target=self.run_continuous_learning, daemon=True)
        self.learning_thread.start()

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

        # Clean old data
        schedule.every().day.at("00:00").do(self.clean_old_data)

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

                if market_data is not None and len(market_data) > self.min_trades_for_update:
                    # Get recent trades for this symbol
                    recent_trades = self.get_recent_trades(symbol, days=1825)

                    # Prepare training data with trade outcomes
                    training_data = self.prepare_training_data(market_data, recent_trades)

                    # Retrain the model
                    old_version = self.model_versions.get(symbol, 'v1')
                    new_version = f"v{datetime.now().strftime('%Y%m%d_%H%M')}"

                    # Backup old model
                    self.backup_model(symbol, old_version)

                    # Train new model
                    metrics = self.ml_predictor.update_models([symbol], {symbol: training_data})

                    # Validate new model
                    if self.validate_new_model(symbol, metrics):
                        self.model_versions[symbol] = new_version
                        logger.info(f"Model updated for {symbol}: {old_version} -> {new_version}")

                        # Record performance
                        self.record_model_performance(symbol, metrics)
                    else:
                        # Rollback to old model
                        self.rollback_model(symbol, old_version)
                        logger.warning(f"Model validation failed for {symbol}, keeping {old_version}")

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
                df['days_old'] = (pd.Timestamp.now() - df['timestamp']).dt.days
                
                # Exponential decay weighting: more weight on recent trades (half-life ~30 days)
                df['weight'] = np.exp(-df['days_old'] / 30.0)
                
                # Normalize weights for analysis
                total_weight = df['weight'].sum()
                
                # Calculate weighted performance metrics
                self.performance_metrics = {
                    'overall_win_rate': ((df['profit_pct'] > 0).astype(int) * df['weight']).sum() / total_weight,
                    'avg_profit': (df['profit_pct'] * df['weight']).sum() / total_weight,
                    'sharpe_ratio': self.calculate_weighted_sharpe(df),
                    'max_drawdown': self.calculate_max_drawdown(df['profit_pct'].cumsum()),
                }

                # Analyze performance by signal strength with weighting
                for component in ['ml_score', 'technical_score', 'sentiment_score']:
                    if component in df.columns:
                        # Group by signal strength quartiles
                        df[f'{component}_quartile'] = pd.qcut(df[component], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
                        performance_by_quartile = df.groupby(f'{component}_quartile').apply(
                            lambda x: pd.Series({
                                'mean': (x['profit_pct'] * x['weight']).sum() / x['weight'].sum(),
                                'count': len(x),
                                'weighted_count': x['weight'].sum()
                            })
                        )

                        # Store insights
                        self.performance_metrics[f'{component}_performance'] = performance_by_quartile.to_dict()

                # Identify best and worst performing setups
                self.identify_patterns()

                logger.info(f"Performance evaluation complete: Win rate={self.performance_metrics['overall_win_rate']:.2%}")

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
                'rsi_oversold': (20, 40),
                'rsi_overbought': (60, 80),
                'min_signal_strength': (0.5, 0.8),
                'trailing_stop_distance': (15, 30),
                'stop_loss_atr_multiplier': (1.5, 3.0),  # Optimize SL ATR multiplier
                'take_profit_atr_multiplier': (2.0, 6.0)  # Optimize TP ATR multiplier
            }

            best_params = self.adaptive_params.copy()
            best_score = self.calculate_optimization_score(best_params, test_data)

            # Grid search with walk-forward validation
            for param_name, (min_val, max_val) in param_ranges.items():
                test_values = np.linspace(min_val, max_val, 5)

                for test_value in test_values:
                    test_params = best_params.copy()
                    test_params[param_name] = test_value

                    # Run backtest with new parameters
                    score = self.calculate_optimization_score(test_params, test_data)

                    if score > best_score:
                        old_value = best_params[param_name]
                        best_params[param_name] = test_value
                        best_score = score

                        # Record optimization
                        self.record_parameter_change(param_name, old_value, test_value, score)

                        logger.info(f"Optimized {param_name}: {old_value:.2f} -> {test_value:.2f} (score: {score:.4f})")

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
            trades_df = pd.DataFrame(list(self.trade_history)[-200:])  # Last 200 trades

            if 'profit_pct' not in trades_df.columns:
                return

            correlations = {}
            for component in ['ml_score', 'technical_score', 'sentiment_score']:
                if component in trades_df.columns:
                    correlation = trades_df[component].corr(trades_df['profit_pct'])
                    correlations[component.replace('_score', '_prediction')] = abs(correlation)

            # Normalize correlations to sum to 1
            total_correlation = sum(correlations.values())
            if total_correlation > 0:
                # Gradual adjustment using adaptation rate
                for component, correlation in correlations.items():
                    if component in self.signal_weights:
                        new_weight = correlation / total_correlation
                        old_weight = self.signal_weights[component]

                        # Smooth adjustment
                        adjusted_weight = old_weight * (1 - self.adaptation_rate) + new_weight * self.adaptation_rate
                        self.signal_weights[component] = adjusted_weight

                        logger.info(f"Adjusted {component} weight: {old_weight:.3f} -> {adjusted_weight:.3f}")

            # Save updated weights
            self.save_signal_weights()

        except Exception as e:
            logger.error(f"Error adjusting signal weights: {e}")

    def trigger_immediate_learning(self, trade_data: dict):
        """Trigger immediate learning for significant events"""
        try:
            symbol = trade_data['symbol']
            profit_pct = trade_data['profit_pct']

            logger.info(f"Significant trade event for {symbol}: {profit_pct:.2f}% - triggering immediate learning")

            # Adjust risk parameters if large loss
            if profit_pct < -3:
                self.adaptive_params['risk_multiplier'] *= 0.95
                logger.info(f"Reduced risk multiplier to {self.adaptive_params['risk_multiplier']:.2f}")

            # Increase confidence if large win
            elif profit_pct > 5:
                if self.adaptive_params['risk_multiplier'] < 1.0:
                    self.adaptive_params['risk_multiplier'] *= 1.02
                    logger.info(f"Increased risk multiplier to {self.adaptive_params['risk_multiplier']:.2f}")

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
                    'avg_signal_strength': winning_trades['signal_strength'].mean(),
                    'avg_ml_score': winning_trades.get('ml_score', pd.Series()).mean(),
                    'common_hour': winning_trades['timestamp'].dt.hour.mode().values[0] if 'timestamp' in winning_trades else None
                }

                # Find common characteristics of losers
                losing_patterns = {
                    'avg_signal_strength': losing_trades['signal_strength'].mean(),
                    'avg_ml_score': losing_trades.get('ml_score', pd.Series()).mean(),
                }

                # Adjust minimum signal strength based on patterns
                if winning_patterns['avg_signal_strength'] > losing_patterns['avg_signal_strength']:
                    new_threshold = (winning_patterns['avg_signal_strength'] + self.adaptive_params['min_signal_strength']) / 2
                    self.adaptive_params['min_signal_strength'] = new_threshold
                    logger.info(f"Adjusted min_signal_strength to {new_threshold:.3f}")

        except Exception as e:
            logger.error(f"Error identifying patterns: {e}")

    def calculate_optimization_score(self, params: dict, test_data: pd.DataFrame) -> float:
        """Calculate optimization score for given parameters using backtest simulation"""
        try:
            # Create a strategy function that uses the test parameters
            def test_strategy(data_dict, symbol):
                signals = []
                data = data_dict[symbol]

                for i in range(len(data)):
                    row = data.iloc[i]

                    # Simple RSI-based signal generation using test parameters
                    rsi = row.get('rsi', 50)
                    if rsi < params.get('rsi_oversold', 30) and row['close'] > row['open']:
                        # Buy signal
                        entry_price = row['close']
                        atr = row.get('atr', entry_price * 0.01)  # Fallback ATR

                        # Use adaptive ATR multipliers for SL/TP
                        sl_multiplier = params.get('stop_loss_atr_multiplier', 2.0)
                        tp_multiplier = params.get('take_profit_atr_multiplier', 4.0)

                        stop_loss = entry_price - (atr * sl_multiplier)
                        take_profit = entry_price + (atr * tp_multiplier)

                        # Track holding time
                        holding_minutes = 0
                        max_holding = params.get('max_holding_minutes', 480)
                        min_holding = params.get('min_holding_minutes', 15)

                        signals.append({
                            'action': 'BUY',
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'strength': abs(row['close'] - row['open']) / row['open'],
                            'max_holding_minutes': max_holding,
                            'min_holding_minutes': min_holding
                        })

                    elif rsi > params.get('rsi_overbought', 70) and row['close'] < row['open']:
                        # Sell signal
                        entry_price = row['close']
                        atr = row.get('atr', entry_price * 0.01)  # Fallback ATR

                        # Use adaptive ATR multipliers for SL/TP
                        sl_multiplier = params.get('stop_loss_atr_multiplier', 2.0)
                        tp_multiplier = params.get('take_profit_atr_multiplier', 4.0)

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
                            'strength': abs(row['close'] - row['open']) / row['open'],
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
                data_dict = {symbol: test_data for symbol in ['EURUSD']}  # Simplified for single symbol

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
                    score = (sharpe * 0.4 + win_rate * 0.3 + min(profit_factor, 3) * 0.3)
                    return max(score, 0)
            else:
                # Fallback to simple simulation if backtest engine not available
                simulated_returns = []

                for i in range(len(test_data) - 1):
                    row = test_data.iloc[i]
                    next_row = test_data.iloc[i + 1]

                    # Generate signal based on parameters
                    rsi_signal = 1 if row['close'] > row['open'] and row.get('rsi', 50) < params.get('rsi_oversold', 30) else -1
                    signal_strength = abs(row['close'] - row['open']) / row['open']

                    if signal_strength > params.get('min_signal_strength', 0.001):
                        # Calculate return with ATR-based SL/TP simulation
                        atr = row.get('atr', row['close'] * 0.01)
                        sl_multiplier = params.get('stop_loss_atr_multiplier', 2.0)
                        tp_multiplier = params.get('take_profit_atr_multiplier', 4.0)

                        # Simulate price movement
                        price_change = (next_row['close'] - row['close']) / row['close']

                        # Check if hit SL or TP
                        if rsi_signal > 0:  # Long position
                            sl_level = -atr * sl_multiplier / row['close']
                            tp_level = atr * tp_multiplier / row['close']
                        else:  # Short position
                            sl_level = atr * sl_multiplier / row['close']
                            tp_level = -atr * tp_multiplier / row['close']

                        # Time-based exit logic
                        max_holding_minutes = params.get('max_holding_minutes', 480)
                        optimal_holding_hours = params.get('optimal_holding_hours', 4.0)
                        optimal_holding_minutes = optimal_holding_hours * 60

                        # Simulate holding time (assume 5-minute bars, so each iteration = 5 minutes)
                        holding_minutes = 0  # Placeholder - will be fixed with proper position tracking

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
                        elif holding_minutes >= optimal_holding_minutes and price_change > 0:
                            # Optimal holding time reached with profit - take profit
                            realized_return = price_change * rsi_signal * 0.8  # Slightly reduce profit for optimal exit
                            exit_reason = 'optimal_time'
                        else:
                            # Position still open, use actual price change
                            realized_return = price_change * rsi_signal
                            exit_reason = 'open'

                        # Only count completed trades (not open positions)
                        if exit_reason != 'open':
                            simulated_returns.append(realized_return * params.get('risk_multiplier', 1.0))
                    else:
                        simulated_returns.append(0)  # No trade

                if simulated_returns:
                    avg_return = np.mean(simulated_returns)
                    volatility = np.std(simulated_returns)
                    sharpe = avg_return / (volatility + 1e-6) if volatility > 0 else 0
                    score = sharpe * params.get('risk_multiplier', 1.0)
                    return max(score, 0)

            return 0.0

        except Exception as e:
            logger.error(f"Error calculating optimization score: {e}")
            return 0.0

    def apply_optimized_parameters(self, new_params: dict):
        """Gradually apply optimized parameters"""
        for param_name, new_value in new_params.items():
            if param_name in self.adaptive_params and param_name != 'risk_multiplier':
                old_value = self.adaptive_params[param_name]
                # Gradual transition
                adjusted_value = old_value * (1 - self.adaptation_rate) + new_value * self.adaptation_rate
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
            symbols = self.config.get('trading', {}).get('pairs', [])[:3]  # Test with first 3 symbols

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

    def fetch_recent_market_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Fetch recent market data for retraining"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Fetch historical data from MT5
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

    def prepare_training_data(self, market_data: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
        """Prepare training data with trade outcomes"""
        # Merge market data with trade outcomes
        # Add labels based on successful trades
        return market_data

    def validate_new_model(self, symbol: str, metrics: dict) -> bool:
        """Validate new model meets minimum performance criteria"""
        min_accuracy = self.config.get('adaptive_learning', {}).get('min_accuracy', 0.6)
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
                recall, sharpe_ratio, max_drawdown, win_rate, avg_profit, total_trades
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

    def record_parameter_change(self, param_name: str, old_value: float, new_value: float, score: float):
        """Record parameter optimization"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        improvement = ((new_value - old_value) / old_value * 100) if old_value != 0 else 0

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
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
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
            variance = ((df['profit_pct'] - avg_return) ** 2 * df['weight']).sum() / total_weight
            volatility = np.sqrt(variance) + 1e-6
            
            return avg_return / volatility
        except Exception:
            return 0.0

    def get_performance_summary(self) -> dict:
        """Get performance summary with current weights and parameters"""
        return {
            'signal_weights': self.get_current_weights(),
            'adaptive_params': self.get_adaptive_parameters()
        }
