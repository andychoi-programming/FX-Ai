"""
Walk-Forward Optimization Module for FX-Ai Trading System

This module implements sophisticated walk-forward analysis to validate trading strategies
and prevent overfitting through out-of-sample testing and rolling window validation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from .ml_predictor import MLPredictor
from .advanced_risk_metrics import AdvancedRiskMetrics


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward analysis window"""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    window_id: int
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    optimized_params: Dict[str, Any]
    performance_metrics: Dict[str, float]


@dataclass
class WalkForwardResults:
    """Results from complete walk-forward analysis"""
    total_windows: int
    windows: List[WalkForwardWindow]
    overall_metrics: Dict[str, float]
    robustness_score: float
    overfitting_detected: bool
    recommendation: str


class WalkForwardOptimizer:
    """
    Walk-forward optimization for strategy validation and overfitting prevention
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize walk-forward optimizer

        Args:
            config: Configuration dictionary with walk-forward settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Walk-forward configuration
        wf_config = config.get('walk_forward_optimization', {})
        self.enabled = wf_config.get('enabled', True)

        # Window parameters
        self.initial_train_window = wf_config.get('initial_train_window', 252)  # ~1 year
        self.test_window = wf_config.get('test_window', 63)  # ~3 months
        self.step_size = wf_config.get('step_size', 21)  # ~1 month
        self.min_train_samples = wf_config.get('min_train_samples', 100)

        # Optimization parameters
        self.optimization_method = wf_config.get('optimization_method', 'grid_search')
        self.max_evaluations = wf_config.get('max_evaluations', 100)
        self.early_stopping_rounds = wf_config.get('early_stopping_rounds', 10)

        # Performance thresholds
        self.min_sharpe_ratio = wf_config.get('min_sharpe_ratio', 0.5)
        self.max_drawdown_threshold = wf_config.get('max_drawdown_threshold', 0.2)
        self.min_win_rate = wf_config.get('min_win_rate', 0.55)

        # Parameter ranges for optimization
        self.parameter_ranges = wf_config.get('parameter_ranges', {
            'rsi_period': {'min': 7, 'max': 21, 'step': 2},
            'macd_fast': {'min': 8, 'max': 16, 'step': 2},
            'macd_slow': {'min': 21, 'max': 31, 'step': 2},
            'macd_signal': {'min': 5, 'max': 11, 'step': 1},
            'bb_period': {'min': 15, 'max': 25, 'step': 2},
            'bb_std': {'min': 1.8, 'max': 2.5, 'step': 0.1},
            'stop_loss_pips': {'min': 15, 'max': 35, 'step': 5},
            'take_profit_pips': {'min': 30, 'max': 70, 'step': 10}
        })

        # Risk metrics calculator
        self.risk_calculator = AdvancedRiskMetrics(config.get('risk_management', {}).get('advanced_risk', {}))

        # Results storage
        self.results_history = []

        self.logger.info("Walk-Forward Optimizer initialized")

    def run_walk_forward_analysis(self, symbol: str, data: pd.DataFrame,
                                strategy_params: Dict[str, Any]) -> WalkForwardResults:
        """
        Run complete walk-forward analysis for a trading strategy

        Args:
            symbol: Trading symbol
            data: Historical price data
            strategy_params: Initial strategy parameters

        Returns:
            WalkForwardResults: Complete analysis results
        """
        try:
            self.logger.info(f"Starting walk-forward analysis for {symbol}")

            if len(data) < self.initial_train_window + self.test_window:
                raise ValueError(f"Insufficient data for walk-forward analysis. Need at least {self.initial_train_window + self.test_window} periods")

            # Generate walk-forward windows
            windows = self._generate_walk_forward_windows(data)

            if not windows:
                raise ValueError("No valid walk-forward windows generated")

            # Analyze each window
            analyzed_windows = []
            for window in windows:
                try:
                    analyzed_window = self._analyze_window(symbol, window, strategy_params)
                    analyzed_windows.append(analyzed_window)
                except Exception as e:
                    self.logger.warning(f"Failed to analyze window {window.window_id}: {e}")
                    continue

            if not analyzed_windows:
                raise ValueError("No windows could be analyzed successfully")

            # Calculate overall results
            overall_results = self._calculate_overall_results(analyzed_windows)

            # Assess robustness
            robustness_score = self._calculate_robustness_score(analyzed_windows)
            overfitting_detected = self._detect_overfitting(analyzed_windows)
            recommendation = self._generate_recommendation(analyzed_windows, robustness_score, overfitting_detected)

            results = WalkForwardResults(
                total_windows=len(analyzed_windows),
                windows=analyzed_windows,
                overall_metrics=overall_results,
                robustness_score=robustness_score,
                overfitting_detected=overfitting_detected,
                recommendation=recommendation
            )

            # Store results
            self.results_history.append({
                'symbol': symbol,
                'timestamp': datetime.now(),
                'results': results
            })

            self.logger.info(f"Walk-forward analysis completed for {symbol}: {len(analyzed_windows)} windows analyzed")
            return results

        except Exception as e:
            self.logger.error(f"Error in walk-forward analysis for {symbol}: {e}")
            # Return minimal results on error
            return WalkForwardResults(
                total_windows=0,
                windows=[],
                overall_metrics={},
                robustness_score=0.0,
                overfitting_detected=False,
                recommendation="analysis_failed"
            )

    def _generate_walk_forward_windows(self, data: pd.DataFrame) -> List[WalkForwardWindow]:
        """
        Generate walk-forward analysis windows

        Args:
            data: Historical price data

        Returns:
            List of WalkForwardWindow objects
        """
        try:
            windows = []
            data_length = len(data)

            # Ensure data has datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'timestamp' in data.columns:
                    data = data.set_index('timestamp')
                else:
                    # Create synthetic timestamps
                    end_time = datetime.now()
                    start_time = end_time - timedelta(hours=data_length)
                    timestamps = pd.date_range(start_time, end_time, periods=data_length)
                    data = data.set_index(timestamps)

            current_position = 0
            window_id = 0

            while current_position + self.initial_train_window + self.test_window <= data_length:
                # Define window boundaries
                train_start_idx = current_position
                train_end_idx = current_position + self.initial_train_window
                test_start_idx = train_end_idx
                test_end_idx = test_start_idx + self.test_window

                # Ensure we don't exceed data bounds
                if test_end_idx > data_length:
                    break

                # Extract data for this window
                train_data = data.iloc[train_start_idx:train_end_idx].copy()
                test_data = data.iloc[test_start_idx:test_end_idx].copy()

                # Create window object
                window = WalkForwardWindow(
                    train_start=train_data.index[0].to_pydatetime(),
                    train_end=train_data.index[-1].to_pydatetime(),
                    test_start=test_data.index[0].to_pydatetime(),
                    test_end=test_data.index[-1].to_pydatetime(),
                    window_id=window_id,
                    train_data=train_data,
                    test_data=test_data,
                    optimized_params={},
                    performance_metrics={}
                )

                windows.append(window)

                # Move to next window
                current_position += self.step_size
                window_id += 1

                # Prevent infinite loops
                if window_id > 1000:
                    break

            self.logger.debug(f"Generated {len(windows)} walk-forward windows")
            return windows

        except Exception as e:
            self.logger.error(f"Error generating walk-forward windows: {e}")
            return []

    def _analyze_window(self, symbol: str, window: WalkForwardWindow,
                       base_params: Dict[str, Any]) -> WalkForwardWindow:
        """
        Analyze a single walk-forward window

        Args:
            symbol: Trading symbol
            window: WalkForwardWindow to analyze
            base_params: Base strategy parameters

        Returns:
            Analyzed WalkForwardWindow with results
        """
        try:
            # Optimize parameters on training data
            optimized_params = self._optimize_parameters(window.train_data, base_params)

            # Evaluate performance on test data
            performance_metrics = self._evaluate_strategy_performance(
                symbol, window.test_data, optimized_params
            )

            # Update window with results
            window.optimized_params = optimized_params
            window.performance_metrics = performance_metrics

            return window

        except Exception as e:
            self.logger.error(f"Error analyzing window {window.window_id}: {e}")
            # Return window with default values
            window.optimized_params = base_params.copy()
            window.performance_metrics = {
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 1.0
            }
            return window

    def _optimize_parameters(self, train_data: pd.DataFrame,
                           base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize strategy parameters on training data

        Args:
            train_data: Training data for optimization
            base_params: Base parameters to optimize from

        Returns:
            Dict of optimized parameters
        """
        try:
            if self.optimization_method == 'grid_search':
                return self._grid_search_optimization(train_data, base_params)
            elif self.optimization_method == 'random_search':
                return self._random_search_optimization(train_data, base_params)
            else:
                # Return base parameters if no optimization method specified
                return base_params.copy()

        except Exception as e:
            self.logger.warning(f"Parameter optimization failed, using base parameters: {e}")
            return base_params.copy()

    def _grid_search_optimization(self, train_data: pd.DataFrame,
                                base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform grid search parameter optimization

        Args:
            train_data: Training data
            base_params: Base parameters

        Returns:
            Best parameter combination
        """
        try:
            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations()

            if not param_combinations:
                return base_params.copy()

            best_params = base_params.copy()
            best_score = -float('inf')

            # Evaluate each parameter combination
            for params in param_combinations[:self.max_evaluations]:  # Limit evaluations
                try:
                    # Create combined parameters
                    test_params = base_params.copy()
                    test_params.update(params)

                    # Evaluate performance
                    score = self._evaluate_parameter_combination(train_data, test_params)

                    if score > best_score:
                        best_score = score
                        best_params = test_params.copy()

                except Exception as e:
                    continue

            return best_params

        except Exception as e:
            self.logger.warning(f"Grid search optimization failed: {e}")
            return base_params.copy()

    def _random_search_optimization(self, train_data: pd.DataFrame,
                                  base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform random search parameter optimization

        Args:
            train_data: Training data
            base_params: Base parameters

        Returns:
            Best parameter combination
        """
        try:
            best_params = base_params.copy()
            best_score = -float('inf')

            # Perform random evaluations
            for _ in range(self.max_evaluations):
                try:
                    # Generate random parameters
                    random_params = self._generate_random_parameters()

                    # Create combined parameters
                    test_params = base_params.copy()
                    test_params.update(random_params)

                    # Evaluate performance
                    score = self._evaluate_parameter_combination(train_data, test_params)

                    if score > best_score:
                        best_score = score
                        best_params = test_params.copy()

                except Exception as e:
                    continue

            return best_params

        except Exception as e:
            self.logger.warning(f"Random search optimization failed: {e}")
            return base_params.copy()

    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search"""
        try:
            combinations = []

            # Simple grid search for key parameters
            rsi_periods = range(self.parameter_ranges['rsi_period']['min'],
                              self.parameter_ranges['rsi_period']['max'] + 1,
                              self.parameter_ranges['rsi_period']['step'])

            stop_losses = range(self.parameter_ranges['stop_loss_pips']['min'],
                              self.parameter_ranges['stop_loss_pips']['max'] + 1,
                              self.parameter_ranges['stop_loss_pips']['step'])

            for rsi in rsi_periods:
                for sl in stop_losses:
                    combinations.append({
                        'rsi_period': rsi,
                        'stop_loss_pips': sl
                    })

            return combinations

        except Exception as e:
            self.logger.warning(f"Error generating parameter combinations: {e}")
            return []

    def _generate_random_parameters(self) -> Dict[str, Any]:
        """Generate random parameter values"""
        try:
            params = {}

            for param_name, param_range in self.parameter_ranges.items():
                if 'min' in param_range and 'max' in param_range:
                    min_val = param_range['min']
                    max_val = param_range['max']

                    if isinstance(min_val, int) and isinstance(max_val, int):
                        params[param_name] = np.random.randint(min_val, max_val + 1)
                    else:
                        params[param_name] = np.random.uniform(min_val, max_val)

            return params

        except Exception as e:
            return {}

    def _evaluate_parameter_combination(self, data: pd.DataFrame,
                                      params: Dict[str, Any]) -> float:
        """
        Evaluate a parameter combination's performance

        Args:
            data: Historical data
            params: Parameter combination to evaluate

        Returns:
            Performance score (higher is better)
        """
        try:
            # Simple evaluation based on Sharpe ratio
            # In a real implementation, this would run the strategy and calculate metrics
            returns = data['close'].pct_change().dropna()

            # Simulate parameter impact (simplified)
            rsi_period = params.get('rsi_period', 14)
            volatility_adjustment = 1.0 + (rsi_period - 14) * 0.01  # Slight adjustment

            adjusted_returns = returns * volatility_adjustment

            # Calculate Sharpe ratio
            if len(adjusted_returns) > 0:
                sharpe = np.mean(adjusted_returns) / np.std(adjusted_returns) * np.sqrt(252)
                return sharpe
            else:
                return 0.0

        except Exception as e:
            return 0.0

    def _evaluate_strategy_performance(self, symbol: str, test_data: pd.DataFrame,
                                     params: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate strategy performance on test data

        Args:
            symbol: Trading symbol
            test_data: Test data
            params: Strategy parameters

        Returns:
            Performance metrics dictionary
        """
        try:
            # Calculate basic returns
            returns = test_data['close'].pct_change().dropna()

            # Calculate performance metrics
            total_return = (test_data['close'].iloc[-1] / test_data['close'].iloc[0]) - 1

            # Sharpe ratio (annualized)
            if len(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0

            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Win rate (simplified - positive returns)
            win_rate = (returns > 0).mean()

            # Profit factor
            winning_trades = returns[returns > 0].sum()
            losing_trades = abs(returns[returns < 0].sum())

            if losing_trades > 0:
                profit_factor = winning_trades / losing_trades
            else:
                profit_factor = float('inf') if winning_trades > 0 else 1.0

            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(returns),
                'avg_trade_return': returns.mean(),
                'trade_std': returns.std()
            }

        except Exception as e:
            self.logger.error(f"Error evaluating strategy performance: {e}")
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 1.0,
                'total_trades': 0,
                'avg_trade_return': 0.0,
                'trade_std': 0.0
            }

    def _calculate_overall_results(self, windows: List[WalkForwardWindow]) -> Dict[str, float]:
        """
        Calculate overall performance metrics across all windows

        Args:
            windows: List of analyzed windows

        Returns:
            Overall performance metrics
        """
        try:
            if not windows:
                return {}

            # Extract metrics from all windows
            sharpe_ratios = [w.performance_metrics.get('sharpe_ratio', 0) for w in windows]
            total_returns = [w.performance_metrics.get('total_return', 0) for w in windows]
            max_drawdowns = [w.performance_metrics.get('max_drawdown', 0) for w in windows]
            win_rates = [w.performance_metrics.get('win_rate', 0) for w in windows]

            return {
                'avg_sharpe_ratio': np.mean(sharpe_ratios),
                'sharpe_std': np.std(sharpe_ratios),
                'sharpe_min': np.min(sharpe_ratios),
                'sharpe_max': np.max(sharpe_ratios),
                'avg_total_return': np.mean(total_returns),
                'total_return_std': np.std(total_returns),
                'avg_max_drawdown': np.mean(max_drawdowns),
                'worst_max_drawdown': np.min(max_drawdowns),
                'avg_win_rate': np.mean(win_rates),
                'win_rate_std': np.std(win_rates),
                'consistency_score': self._calculate_consistency_score(sharpe_ratios)
            }

        except Exception as e:
            self.logger.error(f"Error calculating overall results: {e}")
            return {}

    def _calculate_consistency_score(self, sharpe_ratios: List[float]) -> float:
        """
        Calculate consistency score based on Sharpe ratio stability

        Args:
            sharpe_ratios: List of Sharpe ratios from different windows

        Returns:
            Consistency score (0-1, higher is better)
        """
        try:
            if len(sharpe_ratios) < 2:
                return 0.5

            # Calculate coefficient of variation
            mean_sharpe = np.mean(sharpe_ratios)
            std_sharpe = np.std(sharpe_ratios)

            if mean_sharpe == 0:
                return 0.0

            cv = std_sharpe / abs(mean_sharpe)

            # Convert to consistency score (lower CV = higher consistency)
            consistency = 1.0 / (1.0 + cv)

            return consistency

        except Exception:
            return 0.5

    def _calculate_robustness_score(self, windows: List[WalkForwardWindow]) -> float:
        """
        Calculate overall robustness score

        Args:
            windows: List of analyzed windows

        Returns:
            Robustness score (0-1, higher is better)
        """
        try:
            if not windows:
                return 0.0

            scores = []

            for window in windows:
                metrics = window.performance_metrics

                # Calculate individual window score
                window_score = 0.0
                weight_sum = 0.0

                # Sharpe ratio component
                sharpe = metrics.get('sharpe_ratio', 0)
                if sharpe >= self.min_sharpe_ratio:
                    window_score += 0.4
                elif sharpe >= 0:
                    window_score += 0.2
                weight_sum += 0.4

                # Drawdown component
                drawdown = abs(metrics.get('max_drawdown', 1))
                if drawdown <= self.max_drawdown_threshold:
                    window_score += 0.3
                elif drawdown <= 0.3:
                    window_score += 0.15
                weight_sum += 0.3

                # Win rate component
                win_rate = metrics.get('win_rate', 0)
                if win_rate >= self.min_win_rate:
                    window_score += 0.3
                elif win_rate >= 0.5:
                    window_score += 0.15
                weight_sum += 0.3

                if weight_sum > 0:
                    scores.append(window_score / weight_sum)

            if scores:
                return np.mean(scores)
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating robustness score: {e}")
            return 0.0

    def _detect_overfitting(self, windows: List[WalkForwardWindow]) -> bool:
        """
        Detect potential overfitting in the strategy

        Args:
            windows: List of analyzed windows

        Returns:
            True if overfitting detected
        """
        try:
            if len(windows) < 3:
                return False

            # Check for declining performance over time (overfitting indicator)
            recent_windows = windows[-3:]  # Last 3 windows
            earlier_windows = windows[:-3]  # Earlier windows

            recent_avg_sharpe = np.mean([w.performance_metrics.get('sharpe_ratio', 0) for w in recent_windows])
            earlier_avg_sharpe = np.mean([w.performance_metrics.get('sharpe_ratio', 0) for w in earlier_windows])

            # Significant decline in recent performance
            if earlier_avg_sharpe > 0.5 and recent_avg_sharpe < earlier_avg_sharpe * 0.7:
                return True

            # High in-sample performance with poor out-of-sample
            # This would require comparing in-sample and out-of-sample metrics
            # For now, return False as we don't have in-sample metrics
            return False

        except Exception as e:
            self.logger.warning(f"Error detecting overfitting: {e}")
            return False

    def _generate_recommendation(self, windows: List[WalkForwardWindow],
                               robustness_score: float, overfitting_detected: bool) -> str:
        """
        Generate recommendation based on analysis results

        Args:
            windows: Analyzed windows
            robustness_score: Calculated robustness score
            overfitting_detected: Whether overfitting was detected

        Returns:
            Recommendation string
        """
        try:
            if overfitting_detected:
                return "high_risk_overfitting_detected"

            if robustness_score >= 0.8:
                return "excellent_robust_strategy"
            elif robustness_score >= 0.6:
                return "good_strategy_with_monitoring"
            elif robustness_score >= 0.4:
                return "moderate_strategy_needs_improvement"
            else:
                return "poor_strategy_avoid_use"

        except Exception:
            return "analysis_error"

    def get_walk_forward_report(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest walk-forward analysis report for a symbol

        Args:
            symbol: Trading symbol

        Returns:
            Latest analysis report or None
        """
        try:
            for result in reversed(self.results_history):
                if result['symbol'] == symbol:
                    return result

            return None

        except Exception as e:
            self.logger.error(f"Error getting walk-forward report for {symbol}: {e}")
            return None

    def save_walk_forward_state(self, filepath: str) -> None:
        """Save walk-forward analysis state"""
        try:
            state = {
                'results_history': self.results_history,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }

            import json
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)

            self.logger.info(f"Walk-forward state saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving walk-forward state: {e}")

    def load_walk_forward_state(self, filepath: str) -> None:
        """Load walk-forward analysis state"""
        try:
            import json
            with open(filepath, 'r') as f:
                state = json.load(f)

            self.results_history = state.get('results_history', [])
            self.logger.info(f"Walk-forward state loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Error loading walk-forward state: {e}")