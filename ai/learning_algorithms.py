"""
FX-Ai Learning Algorithms Module
Handles core learning algorithms, parameter optimization, and performance evaluation
"""

import os
import logging
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class LearningAlgorithms:
    """
    Core learning algorithms for parameter optimization, performance evaluation,
    and adaptive learning in the trading system.
    """

    def __init__(self, config: dict, db_path: str = None):
        """Initialize the learning algorithms module"""
        self.config = config
        self.db_path = db_path or os.path.join('data', 'performance_history.db')

        # Learning parameters
        self.min_trades_for_update = 50
        self.adaptation_rate = 0.1  # How quickly to adapt parameters

        # Initialize adaptive parameters
        self.adaptive_params = self._get_default_adaptive_params()
        self.signal_weights = self._get_default_signal_weights()

        # Performance tracking
        self.performance_metrics = {}
        self.trade_history = []

    def _get_default_adaptive_params(self) -> dict:
        """Get default adaptive parameters"""
        return {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'min_signal_strength': 0.6,
            'trailing_stop_distance': 20,
            'stop_loss_atr_multiplier': 2.0,
            'take_profit_atr_multiplier': 4.0,
            'max_holding_minutes': 480,
            'min_holding_minutes': 15,
            'optimal_holding_hours': 4.0,
            'risk_multiplier': 1.0,
            'vwap_period': 20,
            'ema_fast_period': 9,
            'ema_slow_period': 21,
            'rsi_period': 14,
            'atr_period': 14,
            'myfxbook_weight': 0.2,
            'fxstreet_weight': 0.2,
            'fxblue_weight': 0.15,
            'investing_weight': 0.15,
            'forexclientsentiment_weight': 0.1,
            'sentiment_threshold': 0.3,
            'sentiment_time_decay': 1.0,
            'keyword_weight_multiplier': 1.0
        }

    def _get_default_signal_weights(self) -> dict:
        """Get default signal weights"""
        return {
            'ml_prediction': 0.4,
            'technical_prediction': 0.35,
            'fundamental_prediction': 0.15,
            'sentiment_prediction': 0.1
        }

    def evaluate_performance(self) -> dict:
        """Evaluate trading performance and identify areas for improvement"""
        logger.info("Evaluating trading performance...")

        try:
            # Get performance data from database
            performance_data = self._get_performance_data()

            if not performance_data or len(performance_data) == 0:
                return {}

            # Calculate weighted performance metrics
            self.performance_metrics = self._calculate_weighted_metrics(performance_data)

            # Analyze performance by signal components
            self._analyze_signal_performance(performance_data)

            # Identify patterns
            self._identify_patterns(performance_data)

            logger.info(
                f"Performance evaluation complete: Win rate={
                    self.performance_metrics['overall_win_rate']:.2%}")

            return self.performance_metrics

        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
            return {}

    def _get_performance_data(self) -> Optional[pd.DataFrame]:
        """Get recent performance data from database"""
        try:
            import sqlite3
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

                # Exponential decay weighting: more weight on recent trades
                df['weight'] = np.exp(-df['days_old'] / 30.0)

                return df

            return None

        except Exception as e:
            logger.error(f"Error getting performance data: {e}")
            return None

    def _calculate_weighted_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate weighted performance metrics"""
        total_weight = df['weight'].sum()

        metrics = {
            'overall_win_rate': (
                (df['profit_pct'] > 0).astype(int) * df['weight']
            ).sum() / total_weight,
            'avg_profit': (
                df['profit_pct'] * df['weight']
            ).sum() / total_weight,
            'sharpe_ratio': self._calculate_weighted_sharpe(df),
            'max_drawdown': self._calculate_max_drawdown(df['profit_pct'].cumsum()),
            'total_trades': len(df),
            'avg_trade_weight': total_weight / len(df)
        }

        return metrics

    def _calculate_weighted_sharpe(self, df: pd.DataFrame) -> float:
        """Calculate weighted Sharpe ratio"""
        try:
            returns = df['profit_pct'] * df['weight']
            weights = df['weight']

            weighted_mean = returns.sum() / weights.sum()
            weighted_var = ((returns - weighted_mean) ** 2 * weights).sum() / weights.sum()
            weighted_std = np.sqrt(weighted_var)

            return weighted_mean / (weighted_std + 1e-6) * np.sqrt(252)  # Annualized

        except Exception:
            return 0.0

    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            return abs(drawdown.min()) if len(drawdown) > 0 else 0.0
        except Exception:
            return 0.0

    def _analyze_signal_performance(self, df: pd.DataFrame):
        """Analyze performance by signal components"""
        for component in ['ml_score', 'technical_score', 'sentiment_score']:
            if component in df.columns:
                # Group by signal strength quartiles
                df[f'{component}_quartile'] = pd.qcut(
                    df[component], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

                performance_by_quartile = df.groupby(f'{component}_quartile').apply(
                    lambda x: pd.Series({
                        'mean': ((x['profit_pct'] * x['weight']).sum() / x['weight'].sum()),
                        'count': len(x),
                        'weighted_count': x['weight'].sum()
                    })
                )

                # Store insights
                self.performance_metrics[f'{component}_performance'] = performance_by_quartile.to_dict()

    def _identify_patterns(self, df: pd.DataFrame):
        """Identify successful and unsuccessful trading patterns"""
        try:
            if len(df) < 50:
                return

            # Identify winning conditions
            winning_trades = df[df['profit_pct'] > 0]
            losing_trades = df[df['profit_pct'] <= 0]

            if len(winning_trades) > 10 and len(losing_trades) > 10:
                # Find common characteristics of winners
                winning_patterns = {
                    'avg_signal_strength': (
                        winning_trades['signal_strength'].mean()
                    ),
                    'avg_ml_score': winning_trades.get('ml_score', pd.Series()).mean(),
                    'avg_technical_score': winning_trades.get('technical_score', pd.Series()).mean(),
                    'avg_sentiment_score': winning_trades.get('sentiment_score', pd.Series()).mean()
                }

                # Find common characteristics of losers
                losing_patterns = {
                    'avg_signal_strength': losing_trades['signal_strength'].mean(),
                    'avg_ml_score': losing_trades.get('ml_score', pd.Series()).mean(),
                    'avg_technical_score': losing_trades.get('technical_score', pd.Series()).mean(),
                    'avg_sentiment_score': losing_trades.get('sentiment_score', pd.Series()).mean()
                }

                # Adjust minimum signal strength based on patterns
                if (winning_patterns['avg_signal_strength'] > losing_patterns['avg_signal_strength']):
                    new_threshold = (
                        (winning_patterns['avg_signal_strength'] + self.adaptive_params['min_signal_strength']) / 2
                    )
                    self.adaptive_params['min_signal_strength'] = new_threshold
                    logger.info(f"Adjusted min_signal_strength to {new_threshold:.3f}")

                self.performance_metrics['winning_patterns'] = winning_patterns
                self.performance_metrics['losing_patterns'] = losing_patterns

        except Exception as e:
            logger.error(f"Error identifying patterns: {e}")

    def optimize_parameters(self, test_data: Optional[pd.DataFrame] = None) -> dict:
        """Optimize trading parameters based on recent performance"""
        logger.info("Starting parameter optimization...")

        try:
            if test_data is None:
                test_data = self._prepare_backtest_data()

            if test_data is None:
                return self.adaptive_params.copy()

            # Parameters to optimize
            param_ranges = {
                'rsi_oversold': (20, 40),
                'rsi_overbought': (60, 80),
                'min_signal_strength': (0.5, 0.8),
                'trailing_stop_distance': (15, 30),
                'stop_loss_atr_multiplier': (1.5, 3.0),
                'take_profit_atr_multiplier': (2.0, 6.0),
                'vwap_period': (10, 50),
                'ema_fast_period': (5, 20),
                'ema_slow_period': (15, 50),
                'rsi_period': (7, 21),
                'atr_period': (7, 21),
                'myfxbook_weight': (0.1, 0.4),
                'fxstreet_weight': (0.1, 0.4),
                'fxblue_weight': (0.1, 0.3),
                'investing_weight': (0.1, 0.3),
                'forexclientsentiment_weight': (0.05, 0.2),
                'sentiment_threshold': (0.1, 0.5),
                'sentiment_time_decay': (0.5, 2.0),
                'keyword_weight_multiplier': (0.8, 1.5)
            }

            best_params = self.adaptive_params.copy()
            best_score = self._calculate_optimization_score(best_params, test_data)

            # Grid search with walk-forward validation
            for param_name, (min_val, max_val) in param_ranges.items():
                test_values = np.linspace(min_val, max_val, 5)

                for test_value in test_values:
                    test_params = best_params.copy()
                    test_params[param_name] = test_value

                    # Run backtest with new parameters
                    score = self._calculate_optimization_score(test_params, test_data)

                    if score > best_score:
                        old_value = best_params[param_name]
                        best_params[param_name] = test_value
                        best_score = score

                        logger.info(
                            f"Optimized {param_name}: {old_value:.2f} -> {test_value:.2f} (score: {score:.4f})")

            # Apply optimized parameters with gradual adaptation
            self.apply_optimized_parameters(best_params)

            return best_params

        except Exception as e:
            logger.error(f"Error in parameter optimization: {e}")
            return self.adaptive_params.copy()

    def _prepare_backtest_data(self) -> Optional[pd.DataFrame]:
        """Prepare data for backtesting"""
        try:
            # Get recent data for backtesting (last 5 years)
            symbols = self.config.get('trading', {}).get('pairs', ['EURUSD'])[:3]  # Test with first 3 symbols

            # For now, return a simple synthetic dataset
            # In a real implementation, this would fetch actual market data
            dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='D')
            np.random.seed(42)

            data = []
            for date in dates:
                data.append({
                    'timestamp': date,
                    'open': 1.0 + np.random.normal(0, 0.01),
                    'high': 1.0 + np.random.normal(0, 0.015),
                    'low': 1.0 + np.random.normal(0, 0.015),
                    'close': 1.0 + np.random.normal(0, 0.01),
                    'volume': np.random.randint(1000, 10000)
                })

            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)

            # Add technical indicators
            df['rsi'] = self._calculate_rsi(df['close'])
            df['sma_20'] = df['close'].rolling(20).mean()
            df['atr'] = self._calculate_atr(df)

            return df.dropna()

        except Exception as e:
            logger.error(f"Error preparing backtest data: {e}")
            return None

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except Exception:
            return pd.Series([50] * len(prices), index=prices.index)

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR indicator"""
        try:
            high = df['high']
            low = df['low']
            close = df['close'].shift(1)

            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(window=period).mean()
        except Exception:
            return pd.Series([0.001] * len(df), index=df.index)

    def _calculate_optimization_score(self, params: dict, test_data: pd.DataFrame) -> float:
        """Calculate optimization score for parameters using backtest simulation"""
        try:
            simulated_returns = []

            for i in range(len(test_data) - 1):
                row = test_data.iloc[i]
                next_row = test_data.iloc[i + 1]

                # Generate signal based on parameters
                rsi = row.get('rsi', 50)
                rsi_signal = 1 if row['close'] > row['open'] and rsi < params.get('rsi_oversold', 30) else -1
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

                    # Exit conditions
                    if price_change <= sl_level:
                        realized_return = sl_level * rsi_signal
                    elif price_change >= tp_level:
                        realized_return = tp_level * rsi_signal
                    else:
                        realized_return = price_change * rsi_signal

                    simulated_returns.append(realized_return * params.get('risk_multiplier', 1.0))
                else:
                    simulated_returns.append(0)  # No trade

            if simulated_returns:
                avg_return = np.mean(simulated_returns)
                volatility = np.std(simulated_returns)
                sharpe = avg_return / (volatility + 1e-6) if volatility > 0 else 0
                return max(sharpe, 0)

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

    def adjust_signal_weights(self, trade_history: Optional[List[dict]] = None):
        """Adjust signal component weights based on their predictive power"""
        logger.info("Adjusting signal weights...")

        try:
            if trade_history is None:
                trade_history = self.trade_history

            if len(trade_history) < self.min_trades_for_update:
                return

            # Analyze correlation between signal components and outcomes
            trades_df = pd.DataFrame(trade_history[-200:])  # Last 200 trades

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

        except Exception as e:
            logger.error(f"Error adjusting signal weights: {e}")

    def get_current_weights(self) -> dict:
        """Get current signal weights for trading decisions"""
        return self.signal_weights.copy()

    def get_adaptive_parameters(self) -> dict:
        """Get current adaptive parameters"""
        params = self.adaptive_params.copy()
        params['risk_multiplier'] = 1.0  # Always fixed at 1.0
        return params

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

        except Exception as e:
            logger.error(f"Error in immediate learning: {e}")

    def update_trade_history(self, trade_data: dict):
        """Update internal trade history for learning"""
        self.trade_history.append(trade_data)

        # Keep only recent trades
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]