"""
Advanced Risk Metrics Module for FX-Ai
Implements comprehensive risk analysis beyond basic ATR-based SL/TP
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class AdvancedRiskMetrics:
    """Advanced risk metrics calculator for trading systems"""

    def __init__(self, config: Dict):
        """
        Initialize advanced risk metrics calculator

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Risk parameters
        self.confidence_levels = [0.95, 0.99, 0.999]
        self.lookback_periods = [30, 60, 90, 252]  # Trading days
        self.risk_free_rate = 0.02  # 2% annual risk-free rate

        # Portfolio risk settings
        self.max_portfolio_var = config.get('risk_management', {}).get('max_portfolio_var', 0.05)
        self.max_correlation = config.get('risk_management', {}).get('max_correlation', 0.7)

        # Initialize risk tracking
        self.portfolio_returns = []
        self.asset_returns = {}
        self.risk_metrics_history = []

        self.logger.info("Advanced Risk Metrics initialized")

    def calculate_var(self, returns: pd.Series, confidence: float = 0.95,
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR)

        Args:
            returns: Historical returns series
            confidence: Confidence level (0.95, 0.99, etc.)
            method: Calculation method ('historical', 'parametric', 'monte_carlo')

        Returns:
            float: VaR value (negative number)
        """
        if len(returns) < 30:
            return 0.0

        try:
            if method == 'historical':
                # Historical VaR
                var = np.percentile(returns, (1 - confidence) * 100)
                return var  # type: ignore

            elif method == 'parametric':
                # Parametric VaR (assuming normal distribution)
                mean = returns.mean()
                std = returns.std()
                var = mean + std * stats.norm.ppf(1 - confidence)
                return var  # type: ignore

            elif method == 'monte_carlo':
                # Monte Carlo VaR
                n_simulations = 10000
                simulated_returns = np.random.normal(returns.mean(), returns.std(), n_simulations)
                var = np.percentile(simulated_returns, (1 - confidence) * 100)
                return var  # type: ignore

            else:
                return np.percentile(returns, (1 - confidence) * 100)  # type: ignore

        except Exception as e:
            self.logger.warning(f"Error calculating VaR: {e}")
            return 0.0

    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall)

        Args:
            returns: Historical returns series
            confidence: Confidence level

        Returns:
            float: CVaR value (negative number)
        """
        if len(returns) < 30:
            return 0.0

        try:
            var = self.calculate_var(returns, confidence)
            # CVaR is the average of returns below VaR
            tail_losses = returns[returns <= var]
            if len(tail_losses) > 0:
                cvar = tail_losses.mean()
                return cvar  # type: ignore
            else:
                return var  # type: ignore

        except Exception as e:
            self.logger.warning(f"Error calculating CVaR: {e}")
            return 0.0

    def calculate_max_drawdown(self, returns: pd.Series) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown and duration

        Args:
            returns: Historical returns series

        Returns:
            tuple: (max_drawdown, peak_index, trough_index)
        """
        if len(returns) < 2:
            return 0.0, 0, 0

        try:
            # Calculate cumulative returns
            cumulative = (1 + returns).cumprod()

            # Calculate running maximum
            running_max = cumulative.expanding().max()

            # Calculate drawdown
            drawdown = (cumulative - running_max) / running_max

            # Find maximum drawdown
            max_dd = drawdown.min()
            trough_idx = drawdown.idxmin()

            # Find the peak before the trough
            peak_idx = running_max.loc[:trough_idx].idxmax()

            return abs(max_dd), peak_idx, trough_idx

        except Exception as e:
            self.logger.warning(f"Error calculating max drawdown: {e}")
            return 0.0, 0, 0

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sharpe ratio

        Args:
            returns: Historical returns series
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            float: Sharpe ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        if len(returns) < 30:
            return 0.0

        try:
            # Annualize returns and volatility
            annual_returns = returns.mean() * 252  # Trading days per year
            annual_volatility = returns.std() * np.sqrt(252)

            if annual_volatility == 0:
                return 0.0

            sharpe = (annual_returns - risk_free_rate) / annual_volatility
            return sharpe

        except Exception as e:
            self.logger.warning(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sortino ratio (downside deviation only)

        Args:
            returns: Historical returns series
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            float: Sortino ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        if len(returns) < 30:
            return 0.0

        try:
            # Calculate downside returns
            downside_returns = returns[returns < 0]

            if len(downside_returns) == 0:
                return float('inf')  # No downside risk

            # Annualize metrics
            annual_returns = returns.mean() * 252
            downside_volatility = downside_returns.std() * np.sqrt(252)

            if downside_volatility == 0:
                return float('inf')

            sortino = (annual_returns - risk_free_rate) / downside_volatility
            return sortino

        except Exception as e:
            self.logger.warning(f"Error calculating Sortino ratio: {e}")
            return 0.0

    def calculate_calmar_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Calmar ratio (return vs max drawdown)

        Args:
            returns: Historical returns series
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            float: Calmar ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        if len(returns) < 30:
            return 0.0

        try:
            max_dd, _, _ = self.calculate_max_drawdown(returns)

            if max_dd == 0:
                return float('inf')

            annual_returns = returns.mean() * 252
            calmar = (annual_returns - risk_free_rate) / max_dd
            return calmar

        except Exception as e:
            self.logger.warning(f"Error calculating Calmar ratio: {e}")
            return 0.0

    def calculate_portfolio_var(self, weights: np.ndarray, returns: pd.DataFrame,
                               confidence: float = 0.95) -> float:
        """
        Calculate portfolio VaR using covariance matrix

        Args:
            weights: Portfolio weights array
            returns: Asset returns DataFrame
            confidence: Confidence level

        Returns:
            float: Portfolio VaR
        """
        try:
            # Calculate covariance matrix
            cov_matrix = returns.cov()

            # Calculate portfolio variance
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

            # Calculate portfolio volatility
            portfolio_volatility = np.sqrt(portfolio_variance)

            # Calculate VaR (assuming normal distribution)
            portfolio_var = portfolio_volatility * stats.norm.ppf(1 - confidence)

            return portfolio_var

        except Exception as e:
            self.logger.warning(f"Error calculating portfolio VaR: {e}")
            return 0.0

    def calculate_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate beta (systematic risk)

        Args:
            asset_returns: Asset returns series
            market_returns: Market/benchmark returns series

        Returns:
            float: Beta coefficient
        """
        try:
            # Calculate covariance and market variance
            covariance = asset_returns.cov(market_returns)
            market_variance = market_returns.var()

            if market_variance == 0:
                return 1.0

            beta = covariance / market_variance  # type: ignore
            return beta  # type: ignore

        except Exception as e:
            self.logger.warning(f"Error calculating beta: {e}")
            return 1.0

    def calculate_tracking_error(self, asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate tracking error (active risk)

        Args:
            asset_returns: Asset returns series
            benchmark_returns: Benchmark returns series

        Returns:
            float: Tracking error (annualized)
        """
        try:
            # Calculate active returns
            active_returns = asset_returns - benchmark_returns

            # Annualize tracking error
            tracking_error = active_returns.std() * np.sqrt(252)
            return tracking_error

        except Exception as e:
            self.logger.warning(f"Error calculating tracking error: {e}")
            return 0.0

    def calculate_information_ratio(self, asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate information ratio (active return / active risk)

        Args:
            asset_returns: Asset returns series
            benchmark_returns: Benchmark returns series

        Returns:
            float: Information ratio
        """
        try:
            # Calculate active returns
            active_returns = asset_returns - benchmark_returns

            # Calculate tracking error
            tracking_error = self.calculate_tracking_error(asset_returns, benchmark_returns)

            if tracking_error == 0:
                return 0.0

            # Annualize active return
            active_return_annual = active_returns.mean() * 252

            information_ratio = active_return_annual / tracking_error
            return information_ratio

        except Exception as e:
            self.logger.warning(f"Error calculating information ratio: {e}")
            return 0.0

    def calculate_risk_parity_weights(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Calculate risk parity portfolio weights

        Args:
            returns: Asset returns DataFrame

        Returns:
            np.ndarray: Risk parity weights
        """
        try:
            n_assets = returns.shape[1]

            # Objective function: minimize variance of risk contributions
            def objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
                risk_contributions = weights * np.dot(returns.cov(), weights) / portfolio_vol
                return np.var(risk_contributions)

            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            ]

            # Bounds
            bounds = [(0, 1) for _ in range(n_assets)]

            # Initial guess
            initial_weights = np.ones(n_assets) / n_assets

            # Optimize
            result = minimize(objective, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)

            if result.success:
                return result.x
            else:
                # Fallback to equal weights
                return np.ones(n_assets) / n_assets

        except Exception as e:
            self.logger.warning(f"Error calculating risk parity weights: {e}")
            return np.ones(returns.shape[1]) / returns.shape[1]

    def assess_portfolio_risk(self, positions: Dict[str, float], market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Comprehensive portfolio risk assessment

        Args:
            positions: Dictionary of symbol -> position_size
            market_data: Dictionary of symbol -> historical_data

        Returns:
            dict: Comprehensive risk metrics
        """
        try:
            if not positions or not market_data:
                return {}

            # Calculate individual asset returns
            asset_returns = {}
            for symbol, data in market_data.items():
                if len(data) > 1:
                    returns = data['close'].pct_change().dropna()
                    asset_returns[symbol] = returns

            if not asset_returns:
                return {}

            # Create returns DataFrame
            returns_df = pd.DataFrame(asset_returns).dropna()

            # Calculate portfolio weights
            total_exposure = sum(abs(size) for size in positions.values())
            if total_exposure == 0:
                return {}

            weights = np.array([positions.get(symbol, 0) / total_exposure
                              for symbol in returns_df.columns])

            # Calculate comprehensive risk metrics
            risk_metrics = {}

            # Individual asset metrics
            for symbol in returns_df.columns:
                asset_rets = returns_df[symbol]
                if len(asset_rets) >= 30:
                    risk_metrics[f'{symbol}_var_95'] = self.calculate_var(asset_rets, 0.95)
                    risk_metrics[f'{symbol}_cvar_95'] = self.calculate_cvar(asset_rets, 0.95)
                    risk_metrics[f'{symbol}_sharpe'] = self.calculate_sharpe_ratio(asset_rets)
                    risk_metrics[f'{symbol}_max_dd'] = self.calculate_max_drawdown(asset_rets)[0]

            # Portfolio metrics
            if len(weights) > 1:
                portfolio_returns = returns_df.dot(weights)
                risk_metrics['portfolio_var_95'] = self.calculate_portfolio_var(weights, returns_df, 0.95)
                risk_metrics['portfolio_sharpe'] = self.calculate_sharpe_ratio(portfolio_returns)  # type: ignore
                risk_metrics['portfolio_max_dd'] = self.calculate_max_drawdown(portfolio_returns)[0]  # type: ignore

                # Correlation matrix
                corr_matrix = returns_df.corr()
                risk_metrics['correlation_matrix'] = corr_matrix.to_dict()

                # Maximum correlation
                max_corr = 0
                for i in range(len(corr_matrix)):
                    for j in range(i+1, len(corr_matrix)):
                        max_corr = max(max_corr, abs(corr_matrix.iloc[i, j]))  # type: ignore
                risk_metrics['max_correlation'] = max_corr

            # Risk warnings
            risk_metrics['warnings'] = []

            if risk_metrics.get('portfolio_var_95', 0) > self.max_portfolio_var:
                risk_metrics['warnings'].append(f"Portfolio VaR ({risk_metrics['portfolio_var_95']:.3f}) exceeds limit ({self.max_portfolio_var:.3f})")

            if risk_metrics.get('max_correlation', 0) > self.max_correlation:
                risk_metrics['warnings'].append(f"Maximum correlation ({risk_metrics['max_correlation']:.2f}) exceeds limit ({self.max_correlation:.2f})")

            return risk_metrics

        except Exception as e:
            self.logger.error(f"Error assessing portfolio risk: {e}")
            return {}

    def get_risk_limits(self, symbol: str, account_balance: float) -> Dict[str, float]:
        """
        Get risk limits for a specific symbol and account

        Args:
            symbol: Trading symbol
            account_balance: Account balance

        Returns:
            dict: Risk limits
        """
        try:
            # Base risk per trade (2% of account)
            base_risk = account_balance * 0.02

            # Adjust based on symbol volatility (simplified)
            volatility_multiplier = 1.0
            if 'XAU' in symbol or 'XAG' in symbol:
                volatility_multiplier = 2.0  # Precious metals are more volatile
            elif 'JPY' in symbol:
                volatility_multiplier = 0.8  # JPY pairs less volatile

            # Calculate limits
            limits = {
                'max_risk_per_trade': base_risk * volatility_multiplier,
                'max_daily_risk': account_balance * 0.05,  # 5% daily risk
                'max_portfolio_risk': account_balance * 0.10,  # 10% total exposure
                'max_drawdown_limit': account_balance * 0.20,  # 20% max drawdown
                'min_sharpe_ratio': 0.5,  # Minimum acceptable Sharpe ratio
            }

            return limits

        except Exception as e:
            self.logger.warning(f"Error calculating risk limits: {e}")
            return {
                'max_risk_per_trade': account_balance * 0.02,
                'max_daily_risk': account_balance * 0.05,
                'max_portfolio_risk': account_balance * 0.10,
                'max_drawdown_limit': account_balance * 0.20,
                'min_sharpe_ratio': 0.5,
            }