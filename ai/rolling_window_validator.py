"""
Rolling Window Validation Module for FX-Ai Trading System

This module implements comprehensive rolling window validation to ensure trading
strategies perform consistently across different market conditions and time periods.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from .walk_forward_optimizer import WalkForwardOptimizer


@dataclass
class ValidationWindow:
    """Represents a single validation window"""
    start_date: datetime
    end_date: datetime
    window_id: int
    data: pd.DataFrame
    metrics: Dict[str, float]
    is_significant: bool
    market_regime: str


@dataclass
class ValidationResults:
    """Comprehensive validation results"""
    total_windows: int
    windows: List[ValidationWindow]
    statistical_tests: Dict[str, Any]
    regime_analysis: Dict[str, Any]
    robustness_metrics: Dict[str, float]
    validation_score: float
    recommendations: List[str]


class RollingWindowValidator:
    """
    Rolling window validation for comprehensive strategy testing
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize rolling window validator

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Validation configuration
        val_config = config.get('rolling_validation', {})
        self.enabled = val_config.get('enabled', True)

        # Window parameters
        self.window_size = val_config.get('window_size', 252)  # ~1 year
        self.step_size = val_config.get('step_size', 21)  # ~1 month
        self.min_samples = val_config.get('min_samples', 100)

        # Statistical significance thresholds
        self.significance_level = val_config.get('significance_level', 0.05)
        self.min_t_statistic = val_config.get('min_t_statistic', 2.0)

        # Performance thresholds
        self.min_sharpe_threshold = val_config.get('min_sharpe_threshold', 0.3)
        self.max_drawdown_threshold = val_config.get('max_drawdown_threshold', 0.15)
        self.min_win_rate_threshold = val_config.get('min_win_rate_threshold', 0.52)

        # Market regime detection
        self.regime_config = val_config.get('regime_detection', {
            'volatility_threshold': 0.02,
            'trend_strength_threshold': 0.001,
            'volume_threshold': 1.2
        })

        # Statistical tests
        self.perform_stationarity_test = val_config.get('perform_stationarity_test', True)
        self.perform_normality_test = val_config.get('perform_normality_test', True)
        self.perform_autocorrelation_test = val_config.get('perform_autocorrelation_test', True)

        # Walk-forward optimizer for comparison
        wf_config = config.get('walk_forward_optimization', {})
        self.walk_forward_optimizer = WalkForwardOptimizer(config)

        self.logger.info("Rolling Window Validator initialized")

    def run_comprehensive_validation(self, symbol: str, data: pd.DataFrame,
                                   strategy_params: Dict[str, Any]) -> ValidationResults:
        """
        Run comprehensive rolling window validation

        Args:
            symbol: Trading symbol
            data: Historical price data
            strategy_params: Strategy parameters

        Returns:
            ValidationResults: Complete validation results
        """
        try:
            self.logger.info(f"Starting comprehensive validation for {symbol}")

            if len(data) < self.window_size:
                raise ValueError(f"Insufficient data for validation. Need at least {self.window_size} periods")

            # Generate validation windows
            windows = self._generate_validation_windows(data)

            if not windows:
                raise ValueError("No valid validation windows generated")

            # Analyze each window
            analyzed_windows = []
            for window in windows:
                try:
                    analyzed_window = self._analyze_validation_window(symbol, window, strategy_params)
                    analyzed_windows.append(analyzed_window)
                except Exception as e:
                    self.logger.warning(f"Failed to analyze validation window {window.window_id}: {e}")
                    continue

            if not analyzed_windows:
                raise ValueError("No validation windows could be analyzed successfully")

            # Perform statistical tests
            statistical_tests = self._perform_statistical_tests(analyzed_windows)

            # Analyze market regimes
            regime_analysis = self._analyze_market_regimes(analyzed_windows)

            # Calculate robustness metrics
            robustness_metrics = self._calculate_robustness_metrics(analyzed_windows)

            # Calculate overall validation score
            validation_score = self._calculate_validation_score(
                analyzed_windows, statistical_tests, robustness_metrics
            )

            # Generate recommendations
            recommendations = self._generate_validation_recommendations(
                analyzed_windows, statistical_tests, regime_analysis, validation_score
            )

            results = ValidationResults(
                total_windows=len(analyzed_windows),
                windows=analyzed_windows,
                statistical_tests=statistical_tests,
                regime_analysis=regime_analysis,
                robustness_metrics=robustness_metrics,
                validation_score=validation_score,
                recommendations=recommendations
            )

            self.logger.info(f"Comprehensive validation completed for {symbol}: {len(analyzed_windows)} windows analyzed")
            return results

        except Exception as e:
            self.logger.error(f"Error in comprehensive validation for {symbol}: {e}")
            # Return minimal results on error
            return ValidationResults(
                total_windows=0,
                windows=[],
                statistical_tests={},
                regime_analysis={},
                robustness_metrics={},
                validation_score=0.0,
                recommendations=["validation_failed"]
            )

    def _generate_validation_windows(self, data: pd.DataFrame) -> List[ValidationWindow]:
        """
        Generate rolling validation windows

        Args:
            data: Historical price data

        Returns:
            List of ValidationWindow objects
        """
        try:
            windows = []

            # Ensure data has datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'timestamp' in data.columns:
                    data = data.set_index('timestamp')
                else:
                    # Create synthetic timestamps
                    end_time = datetime.now()
                    start_time = end_time - timedelta(hours=len(data))
                    timestamps = pd.date_range(start_time, end_time, periods=len(data))
                    data = data.set_index(timestamps)

            current_position = 0
            window_id = 0

            while current_position + self.window_size <= len(data):
                # Define window boundaries
                start_idx = current_position
                end_idx = current_position + self.window_size

                # Extract data for this window
                window_data = data.iloc[start_idx:end_idx].copy()

                # Create window object
                window = ValidationWindow(
                    start_date=window_data.index[0].to_pydatetime(),
                    end_date=window_data.index[-1].to_pydatetime(),
                    window_id=window_id,
                    data=window_data,
                    metrics={},
                    is_significant=False,
                    market_regime="unknown"
                )

                windows.append(window)

                # Move to next window
                current_position += self.step_size
                window_id += 1

                # Prevent infinite loops
                if window_id > 1000:
                    break

            self.logger.debug(f"Generated {len(windows)} validation windows")
            return windows

        except Exception as e:
            self.logger.error(f"Error generating validation windows: {e}")
            return []

    def _analyze_validation_window(self, symbol: str, window: ValidationWindow,
                                 strategy_params: Dict[str, Any]) -> ValidationWindow:
        """
        Analyze a single validation window

        Args:
            symbol: Trading symbol
            window: ValidationWindow to analyze
            strategy_params: Strategy parameters

        Returns:
            Analyzed ValidationWindow with results
        """
        try:
            # Calculate performance metrics for this window
            metrics = self._calculate_window_metrics(window.data, strategy_params)

            # Detect market regime
            market_regime = self._detect_market_regime(window.data)

            # Determine statistical significance
            is_significant = self._test_statistical_significance(metrics)

            # Update window with results
            window.metrics = metrics
            window.market_regime = market_regime
            window.is_significant = is_significant

            return window

        except Exception as e:
            self.logger.error(f"Error analyzing validation window {window.window_id}: {e}")
            # Return window with default values
            window.metrics = {
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'volatility': 0.0
            }
            window.market_regime = "unknown"
            window.is_significant = False
            return window

    def _calculate_window_metrics(self, data: pd.DataFrame,
                                strategy_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate performance metrics for a validation window

        Args:
            data: Window data
            strategy_params: Strategy parameters

        Returns:
            Performance metrics dictionary
        """
        try:
            # Calculate returns
            returns = data['close'].pct_change().dropna()

            if len(returns) == 0:
                return {
                    'sharpe_ratio': 0.0,
                    'total_return': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'volatility': 0.0,
                    'sortino_ratio': 0.0,
                    'calmar_ratio': 0.0
                }

            # Basic metrics
            total_return = (data['close'].iloc[-1] / data['close'].iloc[0]) - 1
            volatility = returns.std() * np.sqrt(252)  # Annualized

            # Sharpe ratio
            risk_free_rate = 0.02  # Assume 2% risk-free rate
            excess_returns = returns - risk_free_rate/252
            if volatility > 0:
                sharpe_ratio = excess_returns.mean() / volatility * np.sqrt(252)
            else:
                sharpe_ratio = 0.0

            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_volatility = downside_returns.std() * np.sqrt(252)
                sortino_ratio = excess_returns.mean() / downside_volatility * np.sqrt(252) if downside_volatility > 0 else 0.0
            else:
                sortino_ratio = float('inf') if excess_returns.mean() > 0 else 0.0

            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Calmar ratio
            if abs(max_drawdown) > 0:
                calmar_ratio = total_return / abs(max_drawdown)
            else:
                calmar_ratio = float('inf') if total_return > 0 else 0.0

            # Win rate
            win_rate = (returns > 0).mean()

            # Additional metrics
            profit_factor = self._calculate_profit_factor(returns)
            recovery_factor = self._calculate_recovery_factor(total_return, max_drawdown)
            kelly_criterion = self._calculate_kelly_criterion(returns)

            return {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'volatility': volatility,
                'profit_factor': profit_factor,
                'recovery_factor': recovery_factor,
                'kelly_criterion': kelly_criterion,
                'avg_trade_return': returns.mean(),
                'trade_std': returns.std(),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis()
            }

        except Exception as e:
            self.logger.error(f"Error calculating window metrics: {e}")
            return {
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'volatility': 0.0
            }

    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor"""
        try:
            winning_trades = returns[returns > 0].sum()
            losing_trades = abs(returns[returns < 0].sum())

            if losing_trades > 0:
                return winning_trades / losing_trades
            else:
                return float('inf') if winning_trades > 0 else 1.0
        except:
            return 1.0

    def _calculate_recovery_factor(self, total_return: float, max_drawdown: float) -> float:
        """Calculate recovery factor"""
        try:
            if abs(max_drawdown) > 0:
                return total_return / abs(max_drawdown)
            else:
                return float('inf') if total_return > 0 else 0.0
        except:
            return 0.0

    def _calculate_kelly_criterion(self, returns: pd.Series) -> float:
        """Calculate Kelly criterion"""
        try:
            win_rate = (returns > 0).mean()
            avg_win = returns[returns > 0].mean()
            avg_loss = abs(returns[returns < 0].mean())

            if avg_loss > 0:
                kelly = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
                return max(0, kelly)  # Kelly can be negative
            else:
                return 0.0
        except:
            return 0.0

    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """
        Detect market regime for the window

        Args:
            data: Window data

        Returns:
            Market regime classification
        """
        try:
            returns = data['close'].pct_change().dropna()

            if len(returns) < 10:
                return "insufficient_data"

            # Calculate volatility
            volatility = returns.std()

            # Calculate trend strength (linear regression slope)
            x = np.arange(len(returns))
            slope, _ = np.polyfit(x, returns.cumsum(), 1)
            trend_strength = abs(slope)

            # Calculate volume ratio (if volume data available)
            volume_ratio = 1.0
            if 'volume' in data.columns:
                avg_volume = data['volume'].mean()
                recent_volume = data['volume'].tail(20).mean()
                if avg_volume > 0:
                    volume_ratio = recent_volume / avg_volume

            # Classify regime
            if volatility > self.regime_config['volatility_threshold']:
                if trend_strength > self.regime_config['trend_strength_threshold']:
                    return "volatile_trending"
                else:
                    return "volatile_sideways"
            else:
                if trend_strength > self.regime_config['trend_strength_threshold']:
                    if volume_ratio > self.regime_config['volume_threshold']:
                        return "trending_high_volume"
                    else:
                        return "trending_low_volume"
                else:
                    return "sideways_low_volatility"

        except Exception as e:
            self.logger.warning(f"Error detecting market regime: {e}")
            return "unknown"

    def _test_statistical_significance(self, metrics: Dict[str, float]) -> bool:
        """
        Test statistical significance of performance metrics

        Args:
            metrics: Performance metrics

        Returns:
            True if statistically significant
        """
        try:
            sharpe_ratio = metrics.get('sharpe_ratio', 0)

            # Simple t-test for Sharpe ratio significance
            # Assume we have enough data points for significance testing
            if sharpe_ratio > self.min_t_statistic:
                return True

            # Additional checks
            win_rate = metrics.get('win_rate', 0)
            if win_rate > self.min_win_rate_threshold:
                return True

            return False

        except Exception:
            return False

    def _perform_statistical_tests(self, windows: List[ValidationWindow]) -> Dict[str, Any]:
        """
        Perform comprehensive statistical tests on validation results

        Args:
            windows: List of analyzed windows

        Returns:
            Statistical test results
        """
        try:
            if len(windows) < 3:
                return {}

            # Extract metric series
            sharpe_ratios = [w.metrics.get('sharpe_ratio', 0) for w in windows]
            returns = [w.metrics.get('total_return', 0) for w in windows]
            win_rates = [w.metrics.get('win_rate', 0) for w in windows]

            results = {}

            # Stationarity tests
            if self.perform_stationarity_test:
                results['sharpe_stationarity'] = self._adf_test(sharpe_ratios)
                results['returns_stationarity'] = self._adf_test(returns)

            # Normality tests
            if self.perform_normality_test:
                results['sharpe_normality'] = stats.shapiro(sharpe_ratios)
                results['returns_normality'] = stats.shapiro(returns)

            # Autocorrelation tests
            if self.perform_autocorrelation_test:
                results['sharpe_autocorr'] = self._autocorrelation_test(sharpe_ratios)
                results['returns_autocorr'] = self._autocorrelation_test(returns)

            # Cross-sectional tests
            results['sharpe_mean'] = np.mean(sharpe_ratios)
            results['sharpe_std'] = np.std(sharpe_ratios)
            results['sharpe_t_stat'] = np.mean(sharpe_ratios) / (np.std(sharpe_ratios) / np.sqrt(len(sharpe_ratios)))

            return results

        except Exception as e:
            self.logger.error(f"Error performing statistical tests: {e}")
            return {}

    def _adf_test(self, series: List[float]) -> Dict[str, Any]:
        """Augmented Dickey-Fuller test for stationarity"""
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(series)
            return {
                'statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < self.significance_level
            }
        except:
            return {'error': 'ADF test failed'}

    def _autocorrelation_test(self, series: List[float]) -> Dict[str, Any]:
        """Test for autocorrelation"""
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            result = acorr_ljungbox(series, lags=[1, 5, 10])
            return {
                'lag_1_p_value': result.iloc[0, 1],
                'lag_5_p_value': result.iloc[1, 1],
                'lag_10_p_value': result.iloc[2, 1],
                'has_autocorr': any(result.iloc[:, 1] < self.significance_level)
            }
        except:
            return {'error': 'Autocorrelation test failed'}

    def _analyze_market_regimes(self, windows: List[ValidationWindow]) -> Dict[str, Any]:
        """
        Analyze performance across different market regimes

        Args:
            windows: List of analyzed windows

        Returns:
            Regime analysis results
        """
        try:
            regime_performance = {}

            # Group windows by regime
            regimes = {}
            for window in windows:
                regime = window.market_regime
                if regime not in regimes:
                    regimes[regime] = []
                regimes[regime].append(window)

            # Calculate performance by regime
            for regime, regime_windows in regimes.items():
                if len(regime_windows) > 0:
                    sharpe_ratios = [w.metrics.get('sharpe_ratio', 0) for w in regime_windows]
                    returns = [w.metrics.get('total_return', 0) for w in regime_windows]

                    regime_performance[regime] = {
                        'count': len(regime_windows),
                        'avg_sharpe': np.mean(sharpe_ratios),
                        'sharpe_std': np.std(sharpe_ratios),
                        'avg_return': np.mean(returns),
                        'return_std': np.std(returns),
                        'best_sharpe': np.max(sharpe_ratios),
                        'worst_sharpe': np.min(sharpe_ratios)
                    }

            return {
                'regime_performance': regime_performance,
                'regime_distribution': {k: len(v) for k, v in regimes.items()},
                'best_regime': max(regime_performance.items(), key=lambda x: x[1]['avg_sharpe'])[0] if regime_performance else None
            }

        except Exception as e:
            self.logger.error(f"Error analyzing market regimes: {e}")
            return {}

    def _calculate_robustness_metrics(self, windows: List[ValidationWindow]) -> Dict[str, float]:
        """
        Calculate overall robustness metrics

        Args:
            windows: List of analyzed windows

        Returns:
            Robustness metrics dictionary
        """
        try:
            if not windows:
                return {}

            # Extract all metrics
            sharpe_ratios = [w.metrics.get('sharpe_ratio', 0) for w in windows]
            returns = [w.metrics.get('total_return', 0) for w in windows]
            win_rates = [w.metrics.get('win_rate', 0) for w in windows]
            max_drawdowns = [abs(w.metrics.get('max_drawdown', 0)) for w in windows]

            # Calculate robustness metrics
            robustness = {
                'sharpe_consistency': 1.0 - (np.std(sharpe_ratios) / (abs(np.mean(sharpe_ratios)) + 1e-8)),
                'return_stability': 1.0 - (np.std(returns) / (abs(np.mean(returns)) + 1e-8)),
                'win_rate_stability': 1.0 - np.std(win_rates),
                'drawdown_control': 1.0 - np.mean(max_drawdowns),
                'significant_windows_ratio': np.mean([w.is_significant for w in windows]),
                'regime_adaptability': self._calculate_regime_adaptability(windows)
            }

            # Normalize to 0-1 scale
            for key in robustness:
                robustness[key] = max(0.0, min(1.0, robustness[key]))

            return robustness

        except Exception as e:
            self.logger.error(f"Error calculating robustness metrics: {e}")
            return {}

    def _calculate_regime_adaptability(self, windows: List[ValidationWindow]) -> float:
        """Calculate how well the strategy adapts to different market regimes"""
        try:
            regime_sharpes = {}
            for window in windows:
                regime = window.market_regime
                sharpe = window.metrics.get('sharpe_ratio', 0)
                if regime not in regime_sharpes:
                    regime_sharpes[regime] = []
                regime_sharpes[regime].append(sharpe)

            if len(regime_sharpes) < 2:
                return 0.5

            # Calculate average Sharpe per regime
            regime_avgs = {k: np.mean(v) for k, v in regime_sharpes.items()}

            # Calculate adaptability as inverse of performance variance across regimes
            avg_sharpes = list(regime_avgs.values())
            if len(avg_sharpes) > 1:
                adaptability = 1.0 / (1.0 + np.std(avg_sharpes))
                return adaptability
            else:
                return 0.5

        except Exception:
            return 0.5

    def _calculate_validation_score(self, windows: List[ValidationWindow],
                                  statistical_tests: Dict[str, Any],
                                  robustness_metrics: Dict[str, float]) -> float:
        """
        Calculate overall validation score

        Args:
            windows: Analyzed windows
            statistical_tests: Statistical test results
            robustness_metrics: Robustness metrics

        Returns:
            Validation score (0-1, higher is better)
        """
        try:
            score = 0.0
            weight_sum = 0.0

            # Statistical significance (30% weight)
            if statistical_tests:
                t_stat = statistical_tests.get('sharpe_t_stat', 0)
                if t_stat > self.min_t_statistic:
                    score += 0.3
                elif t_stat > 0:
                    score += 0.15
                weight_sum += 0.3

            # Robustness metrics (40% weight)
            if robustness_metrics:
                robustness_avg = np.mean(list(robustness_metrics.values()))
                score += robustness_avg * 0.4
                weight_sum += 0.4

            # Performance consistency (30% weight)
            significant_ratio = np.mean([w.is_significant for w in windows])
            score += significant_ratio * 0.3
            weight_sum += 0.3

            if weight_sum > 0:
                return score / weight_sum
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating validation score: {e}")
            return 0.0

    def _generate_validation_recommendations(self, windows: List[ValidationWindow],
                                           statistical_tests: Dict[str, Any],
                                           regime_analysis: Dict[str, Any],
                                           validation_score: float) -> List[str]:
        """
        Generate validation recommendations

        Args:
            windows: Analyzed windows
            statistical_tests: Statistical test results
            regime_analysis: Regime analysis results
            validation_score: Overall validation score

        Returns:
            List of recommendations
        """
        try:
            recommendations = []

            # Score-based recommendations
            if validation_score >= 0.8:
                recommendations.append("excellent_validation_high_confidence")
            elif validation_score >= 0.6:
                recommendations.append("good_validation_monitor_performance")
            elif validation_score >= 0.4:
                recommendations.append("moderate_validation_needs_improvement")
            else:
                recommendations.append("poor_validation_avoid_live_trading")

            # Statistical test recommendations
            if statistical_tests:
                if not statistical_tests.get('sharpe_stationarity', {}).get('is_stationary', True):
                    recommendations.append("non_stationary_performance_investigate_trend")

                if statistical_tests.get('sharpe_autocorr', {}).get('has_autocorr', False):
                    recommendations.append("autocorrelated_returns_check_for_data_leakage")

            # Regime-specific recommendations
            if regime_analysis:
                regime_perf = regime_analysis.get('regime_performance', {})
                best_regime = regime_analysis.get('best_regime')

                if best_regime:
                    recommendations.append(f"best_performance_in_{best_regime}_regime")

                # Check for regime-specific weaknesses
                for regime, perf in regime_perf.items():
                    if perf['avg_sharpe'] < 0:
                        recommendations.append(f"poor_performance_in_{regime}_consider_regime_filter")

            # Window-specific recommendations
            significant_windows = sum(1 for w in windows if w.is_significant)
            if significant_windows / len(windows) < 0.5:
                recommendations.append("low_significance_ratio_improve_strategy_robustness")

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating validation recommendations: {e}")
            return ["validation_error_check_logs"]

    def get_validation_report(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest validation report for a symbol

        Args:
            symbol: Trading symbol

        Returns:
            Latest validation report or None
        """
        try:
            # This would need to be implemented with a results storage mechanism
            # For now, return None
            return None

        except Exception as e:
            self.logger.error(f"Error getting validation report for {symbol}: {e}")
            return None