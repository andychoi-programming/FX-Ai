"""
Symbol Selector Display - View and select trading symbols
Shows all 30 symbols with their optimized parameters and performance metrics
"""

import json
import os
from typing import Dict, List
from colorama import init, Fore, Style

# Initialize colorama for Windows
init()


class SymbolSelector:
    def __init__(self):
        self.optimal_params_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'models', 'parameter_optimization', 'optimal_parameters.json'
        )
        self.config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config', 'config.json'
        )
        self.load_data()

    def load_data(self):
        """Load optimal parameters and current config"""
        try:
            with open(self.optimal_params_path, 'r') as f:
                self.optimal_params = json.load(f)
        except FileNotFoundError:
            print(f"[ERROR] Error: Optimal parameters file not found at {self.optimal_params_path}")
            self.optimal_params = {}
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"[ERROR] Error: Config file not found at {self.config_path}")
            self.config = {'trading': {'symbols': []}}

    def get_all_symbols(self) -> List[str]:
        """Get all 30 symbols in order"""
        return [
            'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD',
            'CADCHF', 'CADJPY', 'CHFJPY',
            'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD', 'EURUSD',
            'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNZD', 'GBPUSD',
            'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD',
            'USDCAD', 'USDCHF', 'USDJPY',
            'XAGUSD', 'XAUUSD'
        ]

    def display_symbol_summary(self):
        """Display summary of all symbols"""
        print("\n" + "=" * 120)
        print(f"{Fore.CYAN}{'SYMBOL PERFORMANCE SUMMARY - ALL 30 PAIRS':^120}{Style.RESET_ALL}")
        print("=" * 120)
        
        all_symbols = self.get_all_symbols()
        active_symbols = self.config.get('trading', {}).get('symbols', [])
        
        # Header
        print(f"\n{Fore.WHITE}{'Symbol':<10} {'Status':<8} {'Validated':<12} {'PnL':<12} {'Trades':<8} "
              f"{'Win%':<8} {'R:R':<8} {'SL':<8} {'TP':<8} {'Close':<8}{Style.RESET_ALL}")
        print("-" * 120)
        
        # Group by category
        forex_pairs = [s for s in all_symbols if 'XAU' not in s and 'XAG' not in s]
        metal_pairs = [s for s in all_symbols if 'XAU' in s or 'XAG' in s]
        
        print(f"\n{Fore.YELLOW}FOREX PAIRS (28):{Style.RESET_ALL}")
        for symbol in forex_pairs:
            self._print_symbol_row(symbol, symbol in active_symbols)
        
        print(f"\n{Fore.YELLOW}METALS (2):{Style.RESET_ALL}")
        for symbol in metal_pairs:
            self._print_symbol_row(symbol, symbol in active_symbols)
        
        print("\n" + "=" * 120)
        
        # Summary statistics
        self._print_statistics(all_symbols, active_symbols)

    def _print_symbol_row(self, symbol: str, is_active: bool):
        """Print a single symbol row"""
        if symbol not in self.optimal_params:
            status = f"{Fore.RED}NO DATA{Style.RESET_ALL}"
            print(f"{symbol:<10} {status:<15} {'N/A':<12} {'N/A':<12} {'N/A':<8} "
                  f"{'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8}")
            return
        
        params = self.optimal_params[symbol]['H1']
        optimal = params.get('optimal_params', {})
        metrics = params.get('performance_metrics', {})
        validated = params.get('all_periods_profitable', False)
        
        # Status color
        if is_active:
            status = f"{Fore.GREEN}ACTIVE{Style.RESET_ALL}"
        else:
            status = f"{Fore.YELLOW}INACTIVE{Style.RESET_ALL}"
        
        # Validation color
        if validated:
            val_text = f"{Fore.GREEN}PASSED{Style.RESET_ALL}"
        else:
            val_text = f"{Fore.RED}FAILED{Style.RESET_ALL}"
        
        # PnL color
        pnl = metrics.get('pnl', 0)
        if pnl > 0:
            pnl_text = f"{Fore.GREEN}${pnl:,.2f}{Style.RESET_ALL}"
        else:
            pnl_text = f"{Fore.RED}${pnl:,.2f}{Style.RESET_ALL}"
        
        trades = metrics.get('total_trades', 0)
        win_rate = metrics.get('win_rate', 0) * 100
        
        # Calculate R:R from SL and TP
        sl_pips = optimal.get('sl_pips', 0)
        tp_pips = optimal.get('tp_pips', 0)
        rr_ratio = tp_pips / sl_pips if sl_pips > 0 else 0
        
        close_hour = optimal.get('hard_close_hour', 22)
        
        print(f"{symbol:<10} {status:<15} {val_text:<19} {pnl_text:<19} {trades:<8} "
              f"{win_rate:6.1f}% {rr_ratio:6.1f}:1 {sl_pips:5.0f} {tp_pips:6.0f} {close_hour:02d}:30")

    def _print_statistics(self, all_symbols: List[str], active_symbols: List[str]):
        """Print summary statistics"""
        total_symbols = len(all_symbols)
        active_count = len(active_symbols)
        inactive_count = total_symbols - active_count
        
        # Count validated symbols
        passed = sum(1 for s in all_symbols if s in self.optimal_params and 
                    self.optimal_params[s]['H1'].get('all_periods_profitable', False))
        failed = sum(1 for s in all_symbols if s in self.optimal_params and 
                    not self.optimal_params[s]['H1'].get('all_periods_profitable', False))
        
        # Calculate total PnL
        total_pnl = sum(self.optimal_params[s]['H1']['performance_metrics'].get('pnl', 0) 
                       for s in all_symbols if s in self.optimal_params)
        
        # Average metrics for active symbols
        if active_symbols:
            active_data = [self.optimal_params[s]['H1'] for s in active_symbols if s in self.optimal_params]
            if active_data:
                avg_win_rate = sum(d['performance_metrics'].get('win_rate', 0) for d in active_data) / len(active_data) * 100
                avg_trades = sum(d['performance_metrics'].get('total_trades', 0) for d in active_data) / len(active_data)
                active_pnl = sum(d['performance_metrics'].get('pnl', 0) for d in active_data)
            else:
                avg_win_rate = avg_trades = active_pnl = 0
        else:
            avg_win_rate = avg_trades = active_pnl = 0
        
        print(f"\n{Fore.CYAN}SUMMARY STATISTICS:{Style.RESET_ALL}")
        print(f"  Total Symbols:        {total_symbols}")
        print(f"  Active (Trading):     {Fore.GREEN}{active_count}{Style.RESET_ALL}")
        print(f"  Inactive (Available): {Fore.YELLOW}{inactive_count}{Style.RESET_ALL}")
        print(f"  Validation Passed:    {Fore.GREEN}{passed}{Style.RESET_ALL}")
        print(f"  Validation Failed:    {Fore.RED}{failed}{Style.RESET_ALL}")
        print(f"\n  Total Backtest PnL:   ${total_pnl:,.2f} (all 30 symbols)")
        print(f"  Active Symbols PnL:   ${active_pnl:,.2f} ({active_count} symbols)")
        if active_symbols:
            print(f"  Active Avg Win Rate:  {avg_win_rate:.1f}%")
            print(f"  Active Avg Trades:    {avg_trades:.0f}")

    def display_detailed_view(self, symbol: str):
        """Display detailed view of a specific symbol"""
        if symbol not in self.optimal_params:
            print(f"\n[ERROR] No data found for {symbol}")
            return
        
        params = self.optimal_params[symbol]['H1']
        optimal = params.get('optimal_params', {})
        metrics = params.get('performance_metrics', {})
        validated = params.get('all_periods_profitable', False)
        
        print("\n" + "=" * 80)
        print(f"{Fore.CYAN}{symbol} - DETAILED PARAMETERS{Style.RESET_ALL}".center(80))
        print("=" * 80)
        
        # Validation Status
        if validated:
            print(f"\n[OK] Validation: {Fore.GREEN}PASSED{Style.RESET_ALL} - Profitable across all 3-year periods")
        else:
            print(f"\n[WARNING] Validation: {Fore.RED}FAILED{Style.RESET_ALL} - Not profitable in all periods")
        
        # Performance Metrics
        print(f"\n{Fore.YELLOW}BACKTEST PERFORMANCE:{Style.RESET_ALL}")
        print(f"  Total PnL:        ${metrics.get('pnl', 0):,.2f}")
        print(f"  Total Trades:     {metrics.get('total_trades', 0)}")
        print(f"  Win Rate:         {metrics.get('win_rate', 0)*100:.1f}%")
        print(f"  Winners:          {metrics.get('winning_trades', 0)}")
        print(f"  Losers:           {metrics.get('losing_trades', 0)}")
        
        # Optimized Parameters
        print(f"\n{Fore.YELLOW}OPTIMIZED PARAMETERS:{Style.RESET_ALL}")
        print(f"  Stop Loss:        {optimal.get('sl_pips', 0):.0f} pips")
        print(f"  Take Profit:      {optimal.get('tp_pips', 0):.0f} pips")
        print(f"  Risk/Reward:      {optimal.get('tp_pips', 0)/optimal.get('sl_pips', 1):.1f}:1")
        print(f"  Breakeven Trigger:{optimal.get('breakeven_trigger', 0):.0f} pips")
        print(f"  Trailing Start:   {optimal.get('trailing_activation', 0):.0f} pips")
        print(f"  Trailing Distance:{optimal.get('trailing_distance', 0):.0f} pips")
        print(f"  Entry Hours:      {optimal.get('entry_hour_start', 0):02d}:00 - {optimal.get('entry_hour_end', 23):02d}:00")
        print(f"  Hard Close:       {optimal.get('hard_close_hour', 22):02d}:30 GMT")
        
        print("\n" + "=" * 80)

    def interactive_mode(self):
        """Interactive mode for symbol selection"""
        while True:
            self.load_data()  # Reload data each time
            self.display_symbol_summary()
            
            print(f"\n{Fore.CYAN}OPTIONS:{Style.RESET_ALL}")
            print("  [symbol] - View detailed parameters (e.g., 'EURUSD')")
            print("  'list'   - Refresh this list")
            print("  'edit'   - Instructions to edit active symbols")
            print("  'quit'   - Exit")
            
            choice = input(f"\n{Fore.GREEN}Enter choice: {Style.RESET_ALL}").strip().upper()
            
            if choice == 'QUIT':
                break
            elif choice == 'LIST':
                continue
            elif choice == 'EDIT':
                print(f"\n{Fore.YELLOW}TO CHANGE ACTIVE SYMBOLS:{Style.RESET_ALL}")
                print(f"  Edit: {self.config_path}")
                print(f"  Modify the 'symbols' list under 'trading' section")
                print(f"  Add or remove symbols as needed")
                print(f"  Save the file and run 'list' to see changes")
                input(f"\n{Fore.GREEN}Press Enter to continue...{Style.RESET_ALL}")
            elif choice in self.get_all_symbols():
                self.display_detailed_view(choice)
                input(f"\n{Fore.GREEN}Press Enter to continue...{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")
                input(f"\n{Fore.GREEN}Press Enter to continue...{Style.RESET_ALL}")


def main():
    """Main entry point"""
    selector = SymbolSelector()
    
    print(f"\n{Fore.CYAN}{'='*59}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}     FX-AI SYMBOL SELECTOR & CONTROL CENTER             {Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*59}{Style.RESET_ALL}")
    
    selector.interactive_mode()
    
    print(f"\n{Fore.CYAN}Thank you for using Symbol Selector!{Style.RESET_ALL}\n")


if __name__ == '__main__':
    main()
