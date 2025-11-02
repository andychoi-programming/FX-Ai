import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging

class DynamicParameterManager:
    """Manages dynamic trading parameters optimized for each symbol and timeframe"""

    def __init__(self, config: Dict):
        self.config = config
        self.symbols = config['trading']['symbols']
        self.parameters_file = Path("models/parameter_optimization/optimal_parameters.json")
        self.parameters = {}
        self.logger = logging.getLogger(__name__)

        self._load_parameters()

    def _load_parameters(self):
        """Load optimized parameters from file"""
        if self.parameters_file.exists():
            try:
                with open(self.parameters_file, 'r') as f:
                    self.parameters = json.load(f)
                self.logger.info(f"Loaded optimized parameters for {len(self.parameters)} symbols")
            except Exception as e:
                self.logger.error(f"Failed to load parameters: {e}")
                self.parameters = {}
        else:
            self.logger.warning("Optimized parameters file not found, using defaults")
            self.parameters = {}

    def get_optimal_parameters(self, symbol: str, timeframe: str = 'H1') -> Dict:
        """
        Get optimal parameters for a symbol and timeframe

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe ('H1', 'D1', 'W1', 'MN1')

        Returns:
            Dict with optimal parameters
        """
        # Try to get symbol-specific parameters
        if symbol in self.parameters and timeframe in self.parameters[symbol]:
            params = self.parameters[symbol][timeframe]['optimal_params']
            self.logger.debug(f"Using optimized params for {symbol} {timeframe}: SL={params.get('sl_pips', 20)}, TP={params.get('tp_pips', 40)}")
            return params

        # Fallback to symbol-specific H1 parameters
        if symbol in self.parameters and 'H1' in self.parameters[symbol]:
            params = self.parameters[symbol]['H1']['optimal_params']
            self.logger.debug(f"Using H1 fallback params for {symbol} {timeframe}")
            return params

        # Final fallback to default parameters
        default_params = {
            'sl_pips': 20,
            'tp_pips': 40,
            'breakeven_trigger': 15,
            'trailing_activation': 20,
            'trailing_distance': 15,
            'entry_hour_start': 8,
            'entry_hour_end': 16,
            'exit_hour': 20,
            'max_holding_hours': 24,
            'best_entry_days': ['All'],
            'avoid_exit_days': ['None']
        }
        self.logger.debug(f"Using default params for {symbol} {timeframe}")
        return default_params

    def get_entry_time_windows(self, symbol: str, timeframe: str = 'H1') -> Tuple[int, int]:
        """Get optimal entry time window for symbol"""
        params = self.get_optimal_parameters(symbol, timeframe)
        return params.get('entry_hour_start', 8), params.get('entry_hour_end', 16)

    def get_exit_time(self, symbol: str, timeframe: str = 'H1') -> int:
        """Get optimal exit hour for symbol"""
        params = self.get_optimal_parameters(symbol, timeframe)
        return params.get('exit_hour', 20)

    def get_max_holding_hours(self, symbol: str, timeframe: str = 'H1') -> int:
        """Get maximum holding hours for symbol"""
        params = self.get_optimal_parameters(symbol, timeframe)
        return params.get('max_holding_hours', 24)

    def get_best_entry_days(self, symbol: str, timeframe: str = 'H1') -> List[str]:
        """Get best entry days for symbol"""
        params = self.get_optimal_parameters(symbol, timeframe)
        return params.get('best_entry_days', ['All'])

    def get_avoid_exit_days(self, symbol: str, timeframe: str = 'H1') -> List[str]:
        """Get days to avoid exiting positions"""
        params = self.get_optimal_parameters(symbol, timeframe)
        return params.get('avoid_exit_days', ['None'])

    def get_risk_parameters(self, symbol: str, timeframe: str = 'H1') -> Tuple[int, int]:
        """Get optimal SL/TP pips for symbol"""
        params = self.get_optimal_parameters(symbol, timeframe)
        return params.get('sl_pips', 20), params.get('tp_pips', 40)

    def get_breakeven_settings(self, symbol: str, timeframe: str = 'H1') -> int:
        """Get optimal breakeven trigger pips"""
        params = self.get_optimal_parameters(symbol, timeframe)
        return params.get('breakeven_trigger', 15)

    def get_trailing_settings(self, symbol: str, timeframe: str = 'H1') -> Tuple[int, int]:
        """Get optimal trailing stop settings"""
        params = self.get_optimal_parameters(symbol, timeframe)
        return params.get('trailing_activation', 20), params.get('trailing_distance', 15)

    def get_monday_entry_delay(self, symbol: str, timeframe: str = 'H1') -> int:
        """Get optimal Monday entry delay in hours"""
        params = self.get_optimal_parameters(symbol, timeframe)
        return params.get('monday_entry_delay', 10)

    def get_friday_early_exit(self, symbol: str, timeframe: str = 'H1') -> int:
        """Get optimal Friday early exit hour"""
        params = self.get_optimal_parameters(symbol, timeframe)
        return params.get('friday_early_exit', 17)

    def should_trade_symbol(self, symbol: str, timeframe: str = 'H1') -> bool:
        """Determine if symbol should be traded based on optimization results"""
        # For comprehensive backtesting, trade all symbols with available models
        # regardless of optimization performance
        return True

    def get_performance_summary(self) -> Dict:
        """Get summary of optimization performance across all symbols"""
        summary = {
            'total_symbols': len(self.symbols),
            'optimized_symbols': len(self.parameters),
            'profitable_symbols': 0,
            'best_performers': [],
            'worst_performers': [],
            'timeframe_coverage': {}
        }

        for symbol, timeframes in self.parameters.items():
            for timeframe, data in timeframes.items():
                pnl = data['best_pnl']
                win_rate = data['performance_metrics']['win_rate']
                trades = data['performance_metrics']['total_trades']

                if pnl > 0 and win_rate > 0.3 and trades > 20:
                    summary['profitable_symbols'] += 1

                # Track best/worst performers
                if trades > 10:
                    performer = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'pnl': pnl,
                        'win_rate': win_rate,
                        'trades': trades
                    }

                    summary['best_performers'].append(performer)
                    summary['worst_performers'].append(performer)

                # Track timeframe coverage
                if timeframe not in summary['timeframe_coverage']:
                    summary['timeframe_coverage'][timeframe] = 0
                summary['timeframe_coverage'][timeframe] += 1

        # Sort performers
        summary['best_performers'] = sorted(
            summary['best_performers'],
            key=lambda x: x['pnl'],
            reverse=True
        )[:5]

        summary['worst_performers'] = sorted(
            summary['worst_performers'],
            key=lambda x: x['pnl']
        )[:5]

        return summary

    def print_summary(self):
        """Print optimization summary"""
        summary = self.get_performance_summary()

        print("\n" + "="*60)
        print("DYNAMIC PARAMETER OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Total symbols configured: {summary['total_symbols']}")
        print(f"Symbols with optimization: {summary['optimized_symbols']}")
        print(f"Profitable symbols (H1): {summary['profitable_symbols']}")

        print(f"\nTimeframe coverage:")
        for tf, count in summary['timeframe_coverage'].items():
            print(f"  {tf}: {count} symbols")

        print(f"\nTop 5 Best Performers (H1):")
        for i, perf in enumerate(summary['best_performers'][:5], 1):
            print(f"{i}. {perf['symbol']}: ${perf['pnl']:.2f} P&L, "
                  f"{perf['win_rate']:.1%} win rate, {perf['trades']} trades")

        print(f"\nTop 5 Worst Performers (H1):")
        for i, perf in enumerate(summary['worst_performers'][:5], 1):
            print(f"{i}. {perf['symbol']}: ${perf['pnl']:.2f} P&L, "
                  f"{perf['win_rate']:.1%} win rate, {perf['trades']} trades")

# Example usage and integration points
if __name__ == "__main__":
    # Example of how to integrate with existing system
    config = {
        'trading': {
            'symbols': ['EURUSD', 'GBPUSD', 'AUDUSD']  # Example
        }
    }

    param_manager = DynamicParameterManager(config)

    # Example: Get optimal parameters for EURUSD
    params = param_manager.get_optimal_parameters('EURUSD', 'H1')
    print(f"EURUSD H1 optimal parameters: {params}")

    # Example: Check if should trade a symbol
    should_trade = param_manager.should_trade_symbol('EURUSD', 'H1')
    print(f"Should trade EURUSD: {should_trade}")

    param_manager.print_summary()