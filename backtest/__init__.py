"""
Backtest Module
Provides backtesting capabilities for the FX-Ai trading system
"""

from .backtest_engine import BacktestEngine
from .performance_metrics import PerformanceMetrics
from .backtest_config import BacktestConfig

__all__ = ['BacktestEngine', 'PerformanceMetrics', 'BacktestConfig']