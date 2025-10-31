"""
Performance Metrics Calculator for FX-Ai Trading System

This module provides comprehensive performance metrics calculation including
Sharpe, Sortino, Calmar ratios and other advanced metrics for strategy evaluation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float

    # Return metrics
    total_return: float
    annualized_return: float
    compound_annual_growth_rate: float

    # Risk metrics
    volatility: float
    max_drawdown: float
    value_at_risk: float
    expected_shortfall: float
    downside_deviation: float

    # Trade metrics
    win_rate: float
    profit_factor: float
    recovery_factor: float
    kelly_criterion: float

    # Statistical metrics
    skewness: float
    kurtosis: float
    autocorrelation: float

    # Benchmark comparison
    benchmark_alpha: float
    benchmark_beta: float
    information_ratio: float

    # Additional metrics
    ulcer_index: float
    sterling_ratio: float
    burke_ratio: float


@dataclass
class RiskMetrics:
    """Detailed risk analysis metrics"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_consecutive_losses: int
    avg_loss_streak: float
    worst_loss_streak: int
    stress_test_results: Dict[str, float]


class PerformanceMetricsCalculator:
    """
    Comprehensive performance metrics calculator for trading strategies
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize performance metrics calculator

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Metrics configuration
        metrics_config = config.get('performance_metrics', {})
        self.enabled = metrics_config.get('enabled', True)

        # Risk-free rate for calculations
        self.risk_free_rate = metrics_config.get('risk_free_rate', 0.02)  # 2% annual

        # VaR/CVaR confidence levels
        self.var_confidence_levels = metrics_config.get('var_confidence_levels', [0.95, 0.99])

        # Benchmark configuration
        self.benchmark_symbol = metrics_config.get('benchmark_symbol', 'SPY')
        self.benchmark_returns = None

        # Trading days per year (for Forex)
        self.trading_days_per_year = metrics_config.get('trading_days_per_year', 252)

        # Minimum data points required
        self.min_data_points = metrics_config.get('min_data_points', 30)

        self.logger.info("Performance Metrics Calculator initialized")

    def calculate_comprehensive_metrics(self, returns: Union[pd.Series, np.ndarray],
                                      benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None,
                                      trading_days: Optional[int] = None) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics

        Args:
            returns: Strategy returns (daily)
            benchmark_returns: Benchmark returns (optional)
            trading_days: Number of trading days in period

        Returns:
            PerformanceMetrics: Complete metrics suite
        """
        try:
            # Convert to numpy array if needed
            if isinstance(returns, pd.Series):
                returns = returns.values
            returns = np.array(returns)

            if len(returns) < self.min_data_points:
                self.logger.warning(f"Insufficient data points: {len(returns)} < {self.min_data_points}")
                return self._get_default_metrics()

            # Calculate trading days
            if trading_days is None:
                trading_days = len(returns)

            # Basic return metrics
            total_return = self._calculate_total_return(returns)
            annualized_return = self._calculate_annualized_return(returns, trading_days)
            cagr = self._calculate_cagr(returns, trading_days)

            # Risk metrics
            volatility = self._calculate_volatility(returns, trading_days)
            max_drawdown = self._calculate_max_drawdown(returns)
            downside_deviation = self._calculate_downside_deviation(returns)

            # Risk-adjusted ratios
            sharpe_ratio = self._calculate_sharpe_ratio(returns, volatility)
            sortino_ratio = self._calculate_sortino_ratio(returns, downside_deviation)
            calmar_ratio = self._calculate_calmar_ratio(annualized_return, max_drawdown)
            omega_ratio = self._calculate_omega_ratio(returns)

            # Tail risk metrics
            var_95, var_99 = self._calculate_value_at_risk(returns)
            cvar_95, cvar_99 = self._calculate_expected_shortfall(returns)

            # Trade metrics
            win_rate = self._calculate_win_rate(returns)
            profit_factor = self._calculate_profit_factor(returns)
            recovery_factor = self._calculate_recovery_factor(total_return, max_drawdown)
            kelly_criterion = self._calculate_kelly_criterion(returns)

            # Statistical metrics
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            autocorrelation = self._calculate_autocorrelation(returns)

            # Benchmark comparison
            if benchmark_returns is not None:
                benchmark_alpha, benchmark_beta, information_ratio = self._calculate_benchmark_metrics(
                    returns, benchmark_returns, trading_days
                )
            else:
                benchmark_alpha, benchmark_beta, information_ratio = 0.0, 1.0, 0.0

            # Additional risk metrics
            ulcer_index = self._calculate_ulcer_index(returns)
            sterling_ratio = self._calculate_sterling_ratio(annualized_return, max_drawdown)
            burke_ratio = self._calculate_burke_ratio(returns, max_drawdown)

            return PerformanceMetrics(
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                omega_ratio=omega_ratio,
                total_return=total_return,
                annualized_return=annualized_return,
                compound_annual_growth_rate=cagr,
                volatility=volatility,
                max_drawdown=max_drawdown,
                value_at_risk=var_95,  # Using 95% VaR as primary
                expected_shortfall=cvar_95,  # Using 95% CVaR as primary
                downside_deviation=downside_deviation,
                win_rate=win_rate,
                profit_factor=profit_factor,
                recovery_factor=recovery_factor,
                kelly_criterion=kelly_criterion,
                skewness=skewness,
                kurtosis=kurtosis,
                autocorrelation=autocorrelation,
                benchmark_alpha=benchmark_alpha,
                benchmark_beta=benchmark_beta,
                information_ratio=information_ratio,
                ulcer_index=ulcer_index,
                sterling_ratio=sterling_ratio,
                burke_ratio=burke_ratio
            )

        except Exception as e:
            self.logger.error(f"Error calculating comprehensive metrics: {e}")
            return self._get_default_metrics()

    def calculate_risk_metrics(self, returns: Union[pd.Series, np.ndarray]) -> RiskMetrics:
        """
        Calculate detailed risk metrics

        Args:
            returns: Strategy returns

        Returns:
            RiskMetrics: Detailed risk analysis
        """
        try:
            if isinstance(returns, pd.Series):
                returns = returns.values
            returns = np.array(returns)

            if len(returns) < self.min_data_points:
                return self._get_default_risk_metrics()

            # Value at Risk
            var_95, var_99 = self._calculate_value_at_risk(returns)

            # Conditional Value at Risk
            cvar_95, cvar_99 = self._calculate_expected_shortfall(returns)

            # Loss streak analysis
            max_consecutive_losses, avg_loss_streak, worst_loss_streak = self._analyze_loss_streaks(returns)

            # Stress test results
            stress_test_results = self._perform_stress_tests(returns)

            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                max_consecutive_losses=max_consecutive_losses,
                avg_loss_streak=avg_loss_streak,
                worst_loss_streak=worst_loss_streak,
                stress_test_results=stress_test_results
            )

        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return self._get_default_risk_metrics()

    def _calculate_total_return(self, returns: np.ndarray) -> float:
        """Calculate total return"""
        try:
            return np.prod(1 + returns) - 1
        except:
            return 0.0

    def _calculate_annualized_return(self, returns: np.ndarray, trading_days: int) -> float:
        """Calculate annualized return"""
        try:
            total_return = self._calculate_total_return(returns)
            if total_return >= -1:  # Avoid log of negative
                years = trading_days / self.trading_days_per_year
                return (1 + total_return) ** (1 / years) - 1
            else:
                return -1.0
        except:
            return 0.0

    def _calculate_cagr(self, returns: np.ndarray, trading_days: int) -> float:
        """Calculate Compound Annual Growth Rate"""
        try:
            return self._calculate_annualized_return(returns, trading_days)
        except:
            return 0.0

    def _calculate_volatility(self, returns: np.ndarray, trading_days: int) -> float:
        """Calculate annualized volatility"""
        try:
            daily_vol = np.std(returns)
            return daily_vol * np.sqrt(self.trading_days_per_year)
        except:
            return 0.0

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return np.min(drawdown)
        except:
            return 0.0

    def _calculate_downside_deviation(self, returns: np.ndarray) -> float:
        """Calculate downside deviation"""
        try:
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                return np.std(negative_returns)
            else:
                return 0.0
        except:
            return 0.0

    def _calculate_sharpe_ratio(self, returns: np.ndarray, volatility: float) -> float:
        """Calculate Sharpe ratio"""
        try:
            excess_returns = returns - self.risk_free_rate / self.trading_days_per_year
            if volatility > 0:
                return np.mean(excess_returns) / volatility
            else:
                return 0.0
        except:
            return 0.0

    def _calculate_sortino_ratio(self, returns: np.ndarray, downside_deviation: float) -> float:
        """Calculate Sortino ratio"""
        try:
            excess_returns = returns - self.risk_free_rate / self.trading_days_per_year
            if downside_deviation > 0:
                return np.mean(excess_returns) / downside_deviation
            else:
                return float('inf') if np.mean(excess_returns) > 0 else 0.0
        except:
            return 0.0

    def _calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        try:
            if abs(max_drawdown) > 0:
                return annualized_return / abs(max_drawdown)
            else:
                return float('inf') if annualized_return > 0 else 0.0
        except:
            return 0.0

    def _calculate_omega_ratio(self, returns: np.ndarray) -> float:
        """Calculate Omega ratio"""
        try:
            threshold = self.risk_free_rate / self.trading_days_per_year
            excess_returns = returns - threshold

            gains = excess_returns[excess_returns > 0].sum()
            losses = abs(excess_returns[excess_returns < 0].sum())

            if losses > 0:
                return gains / losses
            else:
                return float('inf') if gains > 0 else 1.0
        except:
            return 1.0

    def _calculate_value_at_risk(self, returns: np.ndarray) -> Tuple[float, float]:
        """Calculate Value at Risk at 95% and 99% confidence"""
        try:
            var_95 = np.percentile(returns, 5)  # 5th percentile for 95% confidence
            var_99 = np.percentile(returns, 1)  # 1st percentile for 99% confidence
            return var_95, var_99
        except:
            return 0.0, 0.0

    def _calculate_expected_shortfall(self, returns: np.ndarray) -> Tuple[float, float]:
        """Calculate Expected Shortfall (Conditional VaR)"""
        try:
            var_95, var_99 = self._calculate_value_at_risk(returns)

            # CVaR 95%
            tail_returns_95 = returns[returns <= var_95]
            cvar_95 = np.mean(tail_returns_95) if len(tail_returns_95) > 0 else var_95

            # CVaR 99%
            tail_returns_99 = returns[returns <= var_99]
            cvar_99 = np.mean(tail_returns_99) if len(tail_returns_99) > 0 else var_99

            return cvar_95, cvar_99
        except:
            return 0.0, 0.0

    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """Calculate win rate"""
        try:
            return np.mean(returns > 0)
        except:
            return 0.0

    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
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

    def _calculate_kelly_criterion(self, returns: np.ndarray) -> float:
        """Calculate Kelly criterion"""
        try:
            win_rate = self._calculate_win_rate(returns)
            avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
            avg_loss = abs(np.mean(returns[returns < 0])) if np.any(returns < 0) else 0

            if avg_loss > 0:
                kelly = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
                return max(0, kelly)  # Kelly can be negative
            else:
                return 0.0
        except:
            return 0.0

    def _calculate_autocorrelation(self, returns: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation"""
        try:
            return np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
        except:
            return 0.0

    def _calculate_benchmark_metrics(self, returns: np.ndarray, benchmark_returns: np.ndarray,
                                   trading_days: int) -> Tuple[float, float, float]:
        """Calculate benchmark comparison metrics"""
        try:
            # Ensure same length
            min_len = min(len(returns), len(benchmark_returns))
            returns = returns[:min_len]
            benchmark_returns = benchmark_returns[:min_len]

            # Calculate beta (market sensitivity)
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0

            # Calculate alpha (excess return)
            strategy_return = self._calculate_annualized_return(returns, trading_days)
            benchmark_return = self._calculate_annualized_return(benchmark_returns, trading_days)
            alpha = strategy_return - beta * benchmark_return

            # Calculate information ratio
            tracking_error = np.std(returns - benchmark_returns)
            if tracking_error > 0:
                information_ratio = alpha / tracking_error
            else:
                information_ratio = 0.0

            return alpha, beta, information_ratio

        except Exception as e:
            self.logger.warning(f"Error calculating benchmark metrics: {e}")
            return 0.0, 1.0, 0.0

    def _calculate_ulcer_index(self, returns: np.ndarray) -> float:
        """Calculate Ulcer Index"""
        try:
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            ulcer = np.sqrt(np.mean(drawdown ** 2))
            return ulcer
        except:
            return 0.0

    def _calculate_sterling_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """Calculate Sterling ratio (annualized return / average drawdown)"""
        try:
            # Simplified version using max drawdown instead of average
            if abs(max_drawdown) > 0:
                return annualized_return / abs(max_drawdown)
            else:
                return float('inf') if annualized_return > 0 else 0.0
        except:
            return 0.0

    def _calculate_burke_ratio(self, returns: np.ndarray, max_drawdown: float) -> float:
        """Calculate Burke ratio (CAGR / drawdown deviation)"""
        try:
            # Simplified version
            cagr = self._calculate_cagr(returns, len(returns))
            if abs(max_drawdown) > 0:
                return cagr / abs(max_drawdown)
            else:
                return float('inf') if cagr > 0 else 0.0
        except:
            return 0.0

    def _analyze_loss_streaks(self, returns: np.ndarray) -> Tuple[int, float, int]:
        """Analyze loss streaks"""
        try:
            loss_streaks = []
            current_streak = 0

            for ret in returns:
                if ret < 0:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        loss_streaks.append(current_streak)
                    current_streak = 0

            # Handle final streak
            if current_streak > 0:
                loss_streaks.append(current_streak)

            if loss_streaks:
                max_consecutive_losses = max(loss_streaks)
                avg_loss_streak = np.mean(loss_streaks)
                worst_loss_streak = max_consecutive_losses
            else:
                max_consecutive_losses = 0
                avg_loss_streak = 0.0
                worst_loss_streak = 0

            return max_consecutive_losses, avg_loss_streak, worst_loss_streak

        except:
            return 0, 0.0, 0

    def _perform_stress_tests(self, returns: np.ndarray) -> Dict[str, float]:
        """Perform stress tests on returns"""
        try:
            stress_results = {}

            # Historical stress periods simulation
            # 2008 Financial Crisis (extreme negative returns)
            crisis_returns = returns * 2.5  # Amplify volatility
            stress_results['crisis_scenario_return'] = self._calculate_total_return(crisis_returns)

            # Flash crash scenario
            flash_crash_returns = returns.copy()
            # Simulate a sudden 10% drop
            crash_idx = len(flash_crash_returns) // 2
            flash_crash_returns[crash_idx] = -0.1
            stress_results['flash_crash_return'] = self._calculate_total_return(flash_crash_returns)

            # High volatility scenario
            high_vol_returns = returns + np.random.normal(0, np.std(returns) * 2, len(returns))
            stress_results['high_volatility_return'] = self._calculate_total_return(high_vol_returns)

            return stress_results

        except:
            return {'stress_test_error': 0.0}

    def _get_default_metrics(self) -> PerformanceMetrics:
        """Get default metrics when calculation fails"""
        return PerformanceMetrics(
            sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0, omega_ratio=1.0,
            total_return=0.0, annualized_return=0.0, compound_annual_growth_rate=0.0,
            volatility=0.0, max_drawdown=0.0, value_at_risk=0.0, expected_shortfall=0.0,
            downside_deviation=0.0, win_rate=0.0, profit_factor=1.0, recovery_factor=0.0,
            kelly_criterion=0.0, skewness=0.0, kurtosis=0.0, autocorrelation=0.0,
            benchmark_alpha=0.0, benchmark_beta=1.0, information_ratio=0.0,
            ulcer_index=0.0, sterling_ratio=0.0, burke_ratio=0.0
        )

    def _get_default_risk_metrics(self) -> RiskMetrics:
        """Get default risk metrics when calculation fails"""
        return RiskMetrics(
            var_95=0.0, var_99=0.0, cvar_95=0.0, cvar_99=0.0,
            max_consecutive_losses=0, avg_loss_streak=0.0, worst_loss_streak=0,
            stress_test_results={}
        )

    def create_metrics_report(self, metrics: PerformanceMetrics,
                            risk_metrics: RiskMetrics) -> Dict[str, Any]:
        """
        Create a comprehensive metrics report

        Args:
            metrics: Performance metrics
            risk_metrics: Risk metrics

        Returns:
            Formatted report dictionary
        """
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'performance_metrics': {
                    'risk_adjusted_returns': {
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'sortino_ratio': metrics.sortino_ratio,
                        'calmar_ratio': metrics.calmar_ratio,
                        'omega_ratio': metrics.omega_ratio
                    },
                    'returns': {
                        'total_return': metrics.total_return,
                        'annualized_return': metrics.annualized_return,
                        'cagr': metrics.compound_annual_growth_rate
                    },
                    'risk': {
                        'volatility': metrics.volatility,
                        'max_drawdown': metrics.max_drawdown,
                        'value_at_risk_95': metrics.value_at_risk,
                        'expected_shortfall_95': metrics.expected_shortfall,
                        'downside_deviation': metrics.downside_deviation
                    },
                    'trading': {
                        'win_rate': metrics.win_rate,
                        'profit_factor': metrics.profit_factor,
                        'recovery_factor': metrics.recovery_factor,
                        'kelly_criterion': metrics.kelly_criterion
                    },
                    'statistics': {
                        'skewness': metrics.skewness,
                        'kurtosis': metrics.kurtosis,
                        'autocorrelation': metrics.autocorrelation
                    },
                    'benchmark': {
                        'alpha': metrics.benchmark_alpha,
                        'beta': metrics.benchmark_beta,
                        'information_ratio': metrics.information_ratio
                    },
                    'additional': {
                        'ulcer_index': metrics.ulcer_index,
                        'sterling_ratio': metrics.sterling_ratio,
                        'burke_ratio': metrics.burke_ratio
                    }
                },
                'risk_metrics': {
                    'value_at_risk': {
                        'var_95': risk_metrics.var_95,
                        'var_99': risk_metrics.var_99
                    },
                    'conditional_var': {
                        'cvar_95': risk_metrics.cvar_95,
                        'cvar_99': risk_metrics.cvar_99
                    },
                    'loss_streaks': {
                        'max_consecutive_losses': risk_metrics.max_consecutive_losses,
                        'avg_loss_streak': risk_metrics.avg_loss_streak,
                        'worst_loss_streak': risk_metrics.worst_loss_streak
                    },
                    'stress_tests': risk_metrics.stress_test_results
                },
                'summary': {
                    'overall_score': self._calculate_overall_score(metrics),
                    'risk_level': self._assess_risk_level(metrics, risk_metrics),
                    'recommendation': self._generate_recommendation(metrics, risk_metrics)
                }
            }

            return report

        except Exception as e:
            self.logger.error(f"Error creating metrics report: {e}")
            return {}

    def _calculate_overall_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall performance score"""
        try:
            score = 0.0
            weights = 0.0

            # Sharpe ratio (25% weight)
            if metrics.sharpe_ratio > 1.0:
                score += 0.25
            elif metrics.sharpe_ratio > 0.5:
                score += 0.125
            weights += 0.25

            # Win rate (20% weight)
            score += metrics.win_rate * 0.2
            weights += 0.2

            # Profit factor (20% weight)
            if metrics.profit_factor > 1.5:
                score += 0.2
            elif metrics.profit_factor > 1.0:
                score += 0.1
            weights += 0.2

            # Max drawdown penalty (20% weight)
            drawdown_penalty = min(1.0, abs(metrics.max_drawdown) * 5)  # 20% DD = 1.0 penalty
            score += (1.0 - drawdown_penalty) * 0.2
            weights += 0.2

            # Calmar ratio (15% weight)
            if metrics.calmar_ratio > 1.0:
                score += 0.15
            elif metrics.calmar_ratio > 0.5:
                score += 0.075
            weights += 0.15

            return score / weights if weights > 0 else 0.0

        except:
            return 0.0

    def _assess_risk_level(self, metrics: PerformanceMetrics, risk_metrics: RiskMetrics) -> str:
        """Assess overall risk level"""
        try:
            risk_score = 0.0

            # Drawdown risk
            if abs(metrics.max_drawdown) > 0.2:
                risk_score += 0.3
            elif abs(metrics.max_drawdown) > 0.1:
                risk_score += 0.2

            # Volatility risk
            if metrics.volatility > 0.3:
                risk_score += 0.3
            elif metrics.volatility > 0.2:
                risk_score += 0.2

            # VaR risk
            if abs(risk_metrics.var_95) > 0.05:
                risk_score += 0.2
            elif abs(risk_metrics.var_95) > 0.03:
                risk_score += 0.1

            # Loss streak risk
            if risk_metrics.max_consecutive_losses > 5:
                risk_score += 0.2

            if risk_score > 0.6:
                return "high_risk"
            elif risk_score > 0.3:
                return "moderate_risk"
            else:
                return "low_risk"

        except:
            return "unknown_risk"

    def _generate_recommendation(self, metrics: PerformanceMetrics, risk_metrics: RiskMetrics) -> str:
        """Generate investment recommendation"""
        try:
            overall_score = self._calculate_overall_score(metrics)
            risk_level = self._assess_risk_level(metrics, risk_metrics)

            if overall_score > 0.7 and risk_level == "low_risk":
                return "strong_buy"
            elif overall_score > 0.6 and risk_level in ["low_risk", "moderate_risk"]:
                return "buy"
            elif overall_score > 0.4:
                return "hold"
            elif overall_score > 0.2:
                return "weak_sell"
            else:
                return "strong_sell"

        except:
            return "no_recommendation"