"""
Backtest Configuration
Configuration settings for backtesting the FX-Ai trading system
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Any

class BacktestConfig:
    """Configuration for backtesting"""

    def __init__(self):
        # Time period for backtest - Last three years
        self.start_date = datetime(2022, 10, 31)  # Three years ago from today
        self.end_date = datetime(2025, 10, 31)    # Today

        # Trading symbols
        self.symbols = [
            'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD',
            'CADCHF', 'CADJPY', 'CHFJPY', 'EURAUD', 'EURCAD',
            'EURCHF', 'EURJPY', 'EURGBP', 'EURNZD', 'EURUSD',
            'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNZD',
            'GBPUSD', 'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD',
            'USDCAD', 'USDCHF', 'USDJPY', 'XAGUSD', 'XAUUSD'
        ]

        # Timeframe for backtest
        self.timeframe = 'H1'  # MT5 timeframe constant

        # Initial capital
        self.initial_capital = 10000.0

        # Risk management
        self.max_risk_per_trade = 0.02  # 2% of capital
        self.max_open_positions = 30
        self.stop_loss_pips = 50
        self.take_profit_pips = 100

        # Commission and spread (in pips)
        self.commission_per_lot = 0.0  # No commission for demo
        self.spread_pips = 2.0

        # ML model confidence threshold
        self.min_confidence = 0.6

        # Performance tracking
        self.enable_detailed_logging = True
        self.save_trades_to_csv = True

        # File paths
        self.results_dir = 'backtest_results'
        self.trades_csv_path = os.path.join(self.results_dir, 'trades.csv')
        self.performance_report_path = os.path.join(self.results_dir, 'performance_report.txt')

        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)

    def get_mt5_timeframe(self) -> int:
        """Convert timeframe string to MT5 constant"""
        import MetaTrader5 as mt5

        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }

        return timeframe_map.get(self.timeframe, mt5.TIMEFRAME_H1)

    def get_timeframe_string(self) -> str:
        """Get timeframe as string"""
        return self.timeframe

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'symbols': self.symbols,
            'timeframe': self.timeframe,
            'initial_capital': self.initial_capital,
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_open_positions': self.max_open_positions,
            'stop_loss_pips': self.stop_loss_pips,
            'take_profit_pips': self.take_profit_pips,
            'commission_per_lot': self.commission_per_lot,
            'spread_pips': self.spread_pips,
            'min_confidence': self.min_confidence,
            'enable_detailed_logging': self.enable_detailed_logging,
            'save_trades_to_csv': self.save_trades_to_csv
        }