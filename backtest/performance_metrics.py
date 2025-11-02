"""
Performance Metrics Module
Calculates various performance metrics for backtesting results
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

class PerformanceMetrics:
    """Calculate and analyze trading performance metrics"""

    def __init__(self, trades_df: pd.DataFrame, initial_capital: float = 10000.0):
        """
        Initialize performance metrics calculator

        Args:
            trades_df: DataFrame with trade data
            initial_capital: Initial trading capital
        """
        self.trades_df = trades_df.copy()
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)

        # Calculate equity curve
        self.equity_curve = self._calculate_equity_curve()

    def _calculate_equity_curve(self) -> pd.Series:
        """Calculate equity curve from trades"""
        if self.trades_df.empty:
            return pd.Series([self.initial_capital])

        # Sort trades by close time
        sorted_trades = self.trades_df.sort_values('close_time')

        # Calculate cumulative P&L
        equity = [self.initial_capital]
        cumulative_pnl = 0

        for _, trade in sorted_trades.iterrows():
            cumulative_pnl += trade['pnl']
            equity.append(self.initial_capital + cumulative_pnl)

        # Create datetime index
        dates = [sorted_trades['close_time'].min() - pd.Timedelta(days=1)] + sorted_trades['close_time'].tolist()

        return pd.Series(equity, index=dates)

    def calculate_basic_metrics(self) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        if self.trades_df.empty:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'gross_profit': 0.0,
                'gross_loss': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }

        winning_trades = self.trades_df[self.trades_df['pnl'] > 0]
        losing_trades = self.trades_df[self.trades_df['pnl'] < 0]

        total_pnl = self.trades_df['pnl'].sum()
        gross_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0

        return {
            'total_trades': len(self.trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades_df) if len(self.trades_df) > 0 else 0,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            'avg_win': winning_trades['pnl'].mean() if not winning_trades.empty else 0,
            'avg_loss': losing_trades['pnl'].mean() if not losing_trades.empty else 0,
            'largest_win': winning_trades['pnl'].max() if not winning_trades.empty else 0,
            'largest_loss': losing_trades['pnl'].min() if not losing_trades.empty else 0
        }

    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk-related metrics"""
        if self.equity_curve.empty or len(self.equity_curve) < 2:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'volatility': 0.0
            }

        # Calculate returns
        returns = self.equity_curve.pct_change().dropna()

        # Maximum drawdown
        peak = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        max_drawdown_pct = abs(max_drawdown)

        # Sharpe ratio (assuming 252 trading days per year for daily returns)
        if len(returns) > 1:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() if not negative_returns.empty else 0
        sortino_ratio = returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0

        # Calmar ratio
        annual_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) ** (252 / len(returns)) - 1
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Volatility
        volatility = returns.std() * np.sqrt(252)

        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'volatility': volatility
        }

    def calculate_trade_metrics(self) -> Dict[str, float]:
        """Calculate trade-specific metrics"""
        if self.trades_df.empty:
            return {
                'avg_trade_duration': 0.0,
                'longest_trade': 0.0,
                'shortest_trade': 0.0,
                'avg_pnl_per_trade': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0,
                'consecutive_wins': 0,
                'consecutive_losses': 0,
                'avg_bars_in_trade': 0.0
            }

        # Calculate trade durations
        durations = (self.trades_df['close_time'] - self.trades_df['open_time']).dt.total_seconds() / 3600  # hours

        # Calculate consecutive wins/losses
        pnl_series = self.trades_df['pnl'] > 0
        consecutive_wins = self._max_consecutive_true(pnl_series)
        consecutive_losses = self._max_consecutive_true(~pnl_series)

        return {
            'avg_trade_duration': durations.mean(),
            'longest_trade': durations.max(),
            'shortest_trade': durations.min(),
            'avg_pnl_per_trade': self.trades_df['pnl'].mean(),
            'best_trade': self.trades_df['pnl'].max(),
            'worst_trade': self.trades_df['pnl'].min(),
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses,
            'avg_bars_in_trade': self.trades_df['bars_held'].mean() if 'bars_held' in self.trades_df.columns else 0
        }

    def _max_consecutive_true(self, series: pd.Series) -> int:
        """Calculate maximum consecutive True values in boolean series"""
        if series.empty:
            return 0

        # Find runs of consecutive True values
        runs = []
        current_run = 0

        for val in series:
            if val:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0

        if current_run > 0:
            runs.append(current_run)

        return max(runs) if runs else 0

    def generate_report(self) -> str:
        """Generate comprehensive performance report"""
        basic_metrics = self.calculate_basic_metrics()
        risk_metrics = self.calculate_risk_metrics()
        trade_metrics = self.calculate_trade_metrics()

        report = f"""
TRADING PERFORMANCE REPORT
{'='*50}

PERIOD: {self.equity_curve.index.min()} to {self.equity_curve.index.max()}
INITIAL CAPITAL: ${self.initial_capital:,.2f}
FINAL CAPITAL: ${self.equity_curve.iloc[-1]:,.2f}
TOTAL RETURN: ${basic_metrics['total_pnl']:,.2f} ({basic_metrics['total_pnl']/self.initial_capital*100:.2f}%)

BASIC METRICS:
{'-'*20}
Total Trades: {basic_metrics['total_trades']}
Winning Trades: {basic_metrics['winning_trades']}
Losing Trades: {basic_metrics['losing_trades']}
Win Rate: {basic_metrics['win_rate']:.2%}
Profit Factor: {basic_metrics['profit_factor']:.2f}
Average Win: ${basic_metrics['avg_win']:.2f}
Average Loss: ${basic_metrics['avg_loss']:.2f}
Largest Win: ${basic_metrics['largest_win']:.2f}
Largest Loss: ${basic_metrics['largest_loss']:.2f}

RISK METRICS:
{'-'*20}
Maximum Drawdown: {risk_metrics['max_drawdown_pct']:.2%}
Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}
Sortino Ratio: {risk_metrics['sortino_ratio']:.2f}
Calmar Ratio: {risk_metrics['calmar_ratio']:.2f}
Volatility: {risk_metrics['volatility']:.2%}

TRADE METRICS:
{'-'*20}
Average Trade Duration: {trade_metrics['avg_trade_duration']:.1f} hours
Longest Trade: {trade_metrics['longest_trade']:.1f} hours
Shortest Trade: {trade_metrics['shortest_trade']:.1f} hours
Average P&L per Trade: ${trade_metrics['avg_pnl_per_trade']:.2f}
Best Trade: ${trade_metrics['best_trade']:.2f}
Worst Trade: ${trade_metrics['worst_trade']:.2f}
Max Consecutive Wins: {trade_metrics['consecutive_wins']}
Max Consecutive Losses: {trade_metrics['consecutive_losses']}
Average Bars in Trade: {trade_metrics['avg_bars_in_trade']:.1f}
"""

        return report

    def get_monthly_returns(self) -> pd.Series:
        """Calculate monthly returns"""
        if self.equity_curve.empty:
            return pd.Series()

        monthly_equity = self.equity_curve.resample('M').last()
        monthly_returns = monthly_equity.pct_change().dropna()
        return monthly_returns

    def get_annual_returns(self) -> pd.Series:
        """Calculate annual returns"""
        if self.equity_curve.empty:
            return pd.Series()

        annual_equity = self.equity_curve.resample('Y').last()
        annual_returns = annual_equity.pct_change().dropna()
        return annual_returns