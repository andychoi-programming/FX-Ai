"""
Performance Tracker for FX-Ai Trading System
Real-time tracking of trading performance metrics and statistics
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque
import json
import os


class PerformanceTracker:
    """Track and analyze real-time trading performance"""
    
    def __init__(self, output_file: str = "logs/performance_stats.json"):
        """
        Initialize performance tracker
        
        Args:
            output_file: Path to save performance statistics
        """
        self.output_file = output_file
        self.trades: List[Dict] = []
        self.daily_trades: deque = deque(maxlen=1000)  # Last 1000 trades
        
        # Real-time metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        
        # Symbol-specific tracking
        self.symbol_stats: Dict[str, Dict] = {}
        
        # Session tracking
        self.session_start = datetime.now()
        self.last_trade_time = None
        
        # Peak metrics
        self.peak_profit = 0.0
        self.peak_drawdown = 0.0
        self.current_drawdown = 0.0
        
        # Load previous stats if available
        self._load_stats()
    
    def log_trade(self, symbol: str, entry_price: float, exit_price: float,
                  volume: float, direction: str, profit: float, 
                  duration_minutes: float = 0, strategy: str = ""):
        """
        Log a completed trade
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            exit_price: Exit price
            volume: Trade volume (lots)
            direction: 'BUY' or 'SELL'
            profit: Trade profit/loss in account currency
            duration_minutes: Trade duration in minutes
            strategy: Strategy used (e.g., 'ML', 'RL', 'Technical')
        """
        trade_time = datetime.now()
        is_win = profit > 0
        
        trade_data = {
            'timestamp': trade_time.isoformat(),
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'volume': volume,
            'profit': profit,
            'win': is_win,
            'duration_minutes': duration_minutes,
            'strategy': strategy
        }
        
        self.trades.append(trade_data)
        self.daily_trades.append(trade_data)
        
        # Update counters
        self.total_trades += 1
        if is_win:
            self.winning_trades += 1
            self.total_profit += profit
        else:
            self.losing_trades += 1
            self.total_loss += abs(profit)
        
        self.daily_pnl += profit
        self.last_trade_time = trade_time
        
        # Update peak metrics
        if self.daily_pnl > self.peak_profit:
            self.peak_profit = self.daily_pnl
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = self.peak_profit - self.daily_pnl
            if self.current_drawdown > self.peak_drawdown:
                self.peak_drawdown = self.current_drawdown
        
        # Update symbol-specific stats
        if symbol not in self.symbol_stats:
            self.symbol_stats[symbol] = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'profit': 0.0,
                'avg_duration': 0.0
            }
        
        stats = self.symbol_stats[symbol]
        stats['trades'] += 1
        stats['wins'] += 1 if is_win else 0
        stats['losses'] += 0 if is_win else 1
        stats['profit'] += profit
        
        # Update average duration
        prev_avg = stats['avg_duration']
        stats['avg_duration'] = (prev_avg * (stats['trades'] - 1) + duration_minutes) / stats['trades']
        
        # Save stats periodically
        if self.total_trades % 10 == 0:
            self._save_stats()

    # Backwards-compatible alias for older callers/tests
    def add_trade(self, profit: float, symbol: str = "", direction: str = "", **kwargs):
        """
        Alias to log_trade for backwards compatibility.

        Older code/tests call add_trade(profit, symbol, direction). This method
        offers a simple wrapper that fills required parameters and delegates
        to log_trade.
        """
        # Provide sensible defaults for missing parameters
        entry_price = kwargs.get('entry_price', 0.0)
        exit_price = kwargs.get('exit_price', 0.0)
        volume = kwargs.get('volume', 0.0)
        duration_minutes = kwargs.get('duration_minutes', 0)
        strategy = kwargs.get('strategy', '')
        # Delegate to new log_trade
        self.log_trade(symbol, entry_price, exit_price, volume, direction, profit, duration_minutes, strategy)
    
    def get_statistics(self) -> Dict:
        """
        Get current performance statistics
        
        Returns:
            Dictionary containing all performance metrics
        """
        if self.total_trades == 0:
            return {
                'total_trades': 0,
                'message': 'No trades executed yet'
            }
        
        win_rate = (self.winning_trades / self.total_trades) * 100
        avg_win = self.total_profit / self.winning_trades if self.winning_trades > 0 else 0.0
        avg_loss = self.total_loss / self.losing_trades if self.losing_trades > 0 else 0.0
        profit_factor = self.total_profit / self.total_loss if self.total_loss > 0 else float('inf')
        
        # Calculate expectancy
        expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
        
        # Session duration
        session_duration = datetime.now() - self.session_start
        hours_running = session_duration.total_seconds() / 3600
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'total_profit': round(self.total_profit, 2),
            'total_loss': round(self.total_loss, 2),
            'net_profit': round(self.daily_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'expectancy': round(expectancy, 2),
            'peak_profit': round(self.peak_profit, 2),
            'peak_drawdown': round(self.peak_drawdown, 2),
            'current_drawdown': round(self.current_drawdown, 2),
            'session_hours': round(hours_running, 2),
            'trades_per_hour': round(self.total_trades / hours_running, 2) if hours_running > 0 else 0,
            'last_trade': self.last_trade_time.isoformat() if self.last_trade_time else None,
            'symbol_stats': self.symbol_stats
        }
    
    def get_summary_string(self) -> str:
        """
        Get formatted summary string for console display
        
        Returns:
            Formatted performance summary
        """
        stats = self.get_statistics()
        
        if stats['total_trades'] == 0:
            return "No trades executed yet"
        
        summary = f"""
╔═══════════════════════════════════════════════════════════╗
║           FX-AI PERFORMANCE SUMMARY                       ║
╠═══════════════════════════════════════════════════════════╣
║ Total Trades:     {stats['total_trades']:>6}    Win Rate:      {stats['win_rate']:>6.2f}% ║
║ Wins / Losses:    {stats['winning_trades']:>3} / {stats['losing_trades']:<3}                          ║
╠═══════════════════════════════════════════════════════════╣
║ Net Profit:       ${stats['net_profit']:>8.2f}                          ║
║ Total Profit:     ${stats['total_profit']:>8.2f}                          ║
║ Total Loss:       ${stats['total_loss']:>8.2f}                          ║
║ Profit Factor:    {stats['profit_factor']:>8.2f}                          ║
╠═══════════════════════════════════════════════════════════╣
║ Avg Win:          ${stats['avg_win']:>8.2f}                          ║
║ Avg Loss:         ${stats['avg_loss']:>8.2f}                          ║
║ Expectancy:       ${stats['expectancy']:>8.2f}                          ║
╠═══════════════════════════════════════════════════════════╣
║ Peak Profit:      ${stats['peak_profit']:>8.2f}                          ║
║ Peak Drawdown:    ${stats['peak_drawdown']:>8.2f}                          ║
║ Current Drawdown: ${stats['current_drawdown']:>8.2f}                          ║
╠═══════════════════════════════════════════════════════════╣
║ Session Running:  {stats['session_hours']:>6.2f} hours                      ║
║ Trades/Hour:      {stats['trades_per_hour']:>6.2f}                            ║
╚═══════════════════════════════════════════════════════════╝
"""
        return summary
    
    def get_top_symbols(self, top_n: int = 5) -> List[tuple]:
        """
        Get top performing symbols
        
        Args:
            top_n: Number of top symbols to return
            
        Returns:
            List of (symbol, profit) tuples sorted by profit
        """
        symbol_profits = [(symbol, stats['profit']) 
                         for symbol, stats in self.symbol_stats.items()]
        return sorted(symbol_profits, key=lambda x: x[1], reverse=True)[:top_n]
    
    def reset_daily_stats(self):
        """Reset daily statistics (call at start of new trading day)"""
        self.daily_pnl = 0.0
        self.peak_profit = 0.0
        self.peak_drawdown = 0.0
        self.current_drawdown = 0.0
    
    def _save_stats(self):
        """Save statistics to file"""
        try:
            stats = self.get_statistics()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            
            with open(self.output_file, 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            print(f"Error saving performance stats: {e}")
    
    def _load_stats(self):
        """Load previous statistics from file"""
        try:
            if os.path.exists(self.output_file):
                with open(self.output_file, 'r') as f:
                    stats = json.load(f)
                
                # Restore metrics
                self.total_trades = stats.get('total_trades', 0)
                self.winning_trades = stats.get('winning_trades', 0)
                self.losing_trades = stats.get('losing_trades', 0)
                self.total_profit = stats.get('total_profit', 0.0)
                self.total_loss = stats.get('total_loss', 0.0)
                self.symbol_stats = stats.get('symbol_stats', {})
        except Exception as e:
            print(f"Error loading performance stats: {e}")


# Example usage
if __name__ == "__main__":
    # Create tracker
    tracker = PerformanceTracker()
    
    # Simulate some trades
    tracker.log_trade("EURUSD", 1.1000, 1.1020, 0.1, "BUY", 20.0, 45, "ML")
    tracker.log_trade("GBPUSD", 1.2500, 1.2480, 0.1, "BUY", -20.0, 30, "RL")
    tracker.log_trade("USDJPY", 150.00, 150.30, 0.1, "BUY", 30.0, 60, "Ensemble")
    tracker.log_trade("EURUSD", 1.1050, 1.1070, 0.1, "BUY", 20.0, 40, "ML")
    
    # Display summary
    print(tracker.get_summary_string())
    
    # Get top symbols
    print("\nTop Performing Symbols:")
    for symbol, profit in tracker.get_top_symbols(3):
        print(f"  {symbol}: ${profit:.2f}")
