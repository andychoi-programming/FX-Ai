import json
import time
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import logging

class TradingDashboard:
    """Real-time trading system dashboard"""

    def __init__(self):
        self.config_file = Path("config/trading_config.json")
        self.performance_file = Path("logs/performance.json")
        self.backtest_file = Path("backtest_results_ml_optimized.json")
        self.optimization_file = Path("models/parameter_optimization/optimal_parameters.json")

        self.logger = logging.getLogger(__name__)

    def load_data(self) -> Dict:
        """Load all relevant data for dashboard"""
        data = {
            'config': {},
            'performance': {},
            'backtest': {},
            'optimization': {},
            'system_status': {}
        }

        # Load configuration
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    data['config'] = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")

        # Load performance data
        try:
            if self.performance_file.exists():
                with open(self.performance_file, 'r') as f:
                    data['performance'] = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load performance: {e}")

        # Load backtest results
        try:
            if self.backtest_file.exists():
                with open(self.backtest_file, 'r') as f:
                    data['backtest'] = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load backtest: {e}")

        # Load optimization results
        try:
            if self.optimization_file.exists():
                with open(self.optimization_file, 'r') as f:
                    data['optimization'] = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load optimization: {e}")

        # System status
        data['system_status'] = self._get_system_status()

        return data

    def _get_system_status(self) -> Dict:
        """Get current system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'models_trained': 0,
            'total_symbols': 0,
            'optimized_symbols': 0,
            'active_trades': 0  # Would be populated from live system
        }

        # Count trained models
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.glob("*_model.pkl"))
            status['models_trained'] = len(model_files)

        # Count total symbols in config
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    status['total_symbols'] = len(config['trading']['symbols'])
            except:
                pass

        # Count optimized symbols
        if self.optimization_file.exists():
            try:
                with open(self.optimization_file, 'r') as f:
                    opt_data = json.load(f)
                    status['optimized_symbols'] = len(opt_data)
            except:
                pass

        return status

    def display_dashboard(self):
        """Display the main dashboard"""
        data = self.load_data()

        self._clear_screen()
        self._print_header()
        self._print_system_status(data)
        self._print_performance_summary(data)
        self._print_backtest_summary(data)
        self._print_optimization_summary(data)
        self._print_recent_trades(data)
        self._print_footer()

    def _clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def _print_header(self):
        """Print dashboard header"""
        print("="*80)
        print("                    ML FOREX TRADING SYSTEM DASHBOARD")
        print("="*80)
        print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

    def _print_system_status(self, data: Dict):
        """Print system status section"""
        status = data['system_status']

        print("SYSTEM STATUS")
        print("-"*40)
        print(f"Models Trained:     {status['models_trained']}/30")
        print(f"Symbols Configured: {status['total_symbols']}")
        print(f"Symbols Optimized:  {status['optimized_symbols']}")
        print(f"Active Positions:   {status['active_trades']}")
        print(".1f")
        print()

    def _print_performance_summary(self, data: Dict):
        """Print performance summary"""
        perf = data['performance']

        if not perf:
            print("PERFORMANCE SUMMARY")
            print("-"*40)
            print("No performance data available")
            print()
            return

        # Get today's performance
        today = datetime.now().date().isoformat()
        today_stats = perf.get(today, {})

        print("TODAY'S PERFORMANCE")
        print("-"*40)
        print(f"Trades Executed: {today_stats.get('trades', 0)}")
        print(".2f")
        print(f"Winning Trades:  {today_stats.get('winning_trades', 0)}")
        print(f"Losing Trades:   {today_stats.get('losing_trades', 0)}")

        if today_stats.get('trades', 0) > 0:
            win_rate = today_stats['winning_trades'] / today_stats['trades']
            print(".1%")
        print()

    def _print_backtest_summary(self, data: Dict):
        """Print backtest results summary"""
        bt = data['backtest']

        if not bt or 'metrics' not in bt:
            print("BACKTEST RESULTS")
            print("-"*40)
            print("No backtest data available")
            print()
            return

        metrics = bt['metrics']

        print("BACKTEST RESULTS (Last 30 Days)")
        print("-"*40)
        print(".2f")
        print(".2f")
        print(f"Total Trades:     {metrics.get('total_trades', 0)}")
        print(".1%")
        print(".2f")
        print(".2f")
        print()

    def _print_optimization_summary(self, data: Dict):
        """Print parameter optimization summary"""
        opt = data['optimization']

        if not opt:
            print("PARAMETER OPTIMIZATION")
            print("-"*40)
            print("No optimization data available")
            print()
            return

        # Calculate summary stats
        total_symbols = len(opt)
        profitable_symbols = 0
        total_trades = 0
        total_pnl = 0

        for symbol, timeframes in opt.items():
            for tf, results in timeframes.items():
                pnl = results.get('best_pnl', 0)
                trades = results.get('performance_metrics', {}).get('total_trades', 0)

                if pnl > 0 and trades > 10:
                    profitable_symbols += 1

                total_pnl += pnl
                total_trades += trades

        print("PARAMETER OPTIMIZATION SUMMARY")
        print("-"*40)
        print(f"Symbols Optimized:     {total_symbols}")
        print(f"Profitable Symbols:    {profitable_symbols}")
        print(".2f")
        print(f"Total Optimized Trades: {total_trades}")
        print()

    def _print_recent_trades(self, data: Dict):
        """Print recent trades"""
        bt = data['backtest']

        if not bt or 'trades' not in bt:
            print("RECENT TRADES")
            print("-"*40)
            print("No trade data available")
            print()
            return

        trades = bt['trades'][-5:]  # Last 5 trades

        print("RECENT TRADES")
        print("-"*40)

        if not trades:
            print("No trades recorded")
        else:
            for trade in trades:
                pnl_color = "GREEN" if trade['pnl'] > 0 else "RED"
                print("6s")

        print()

    def _print_footer(self):
        """Print dashboard footer"""
        print("="*80)
        print("Commands: [R]efresh | [Q]uit | [B]acktest | [T]rade")
        print("="*80)

    def run_interactive_dashboard(self):
        """Run interactive dashboard"""
        while True:
            self.display_dashboard()

            try:
                command = input("Enter command: ").strip().lower()

                if command == 'q':
                    break
                elif command == 'r':
                    continue  # Just refresh
                elif command == 'b':
                    self._run_backtest()
                elif command == 't':
                    self._start_trading()
                else:
                    print("Invalid command. Use R, Q, B, or T.")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

            time.sleep(2)  # Brief pause before refresh

    def _run_backtest(self):
        """Run a new backtest"""
        print("Running backtest... This may take a few minutes.")
        try:
            # Import and run backtest
            from backtest.ml_backtester import MLBacktester

            with open(self.config_file, 'r') as f:
                config = json.load(f)

            backtester = MLBacktester(config)

            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            end_date = datetime.now().strftime("%Y-%m-%d")

            backtester.run_backtest(start_date, end_date, 'H1')
            backtester.save_results()

            print("Backtest completed! Press R to refresh dashboard.")

        except Exception as e:
            print(f"Backtest failed: {e}")

        input("Press Enter to continue...")

    def _start_trading(self):
        """Start the trading system"""
        print("Starting trading system...")
        try:
            from live_trading.trading_orchestrator import TradingOrchestrator

            orchestrator = TradingOrchestrator()
            print("Trading system started. Press Ctrl+C in terminal to stop.")
            orchestrator.start_trading()

        except Exception as e:
            print(f"Failed to start trading: {e}")

        input("Press Enter to continue...")

def main():
    """Main entry point"""
    dashboard = TradingDashboard()
    dashboard.run_interactive_dashboard()

if __name__ == "__main__":
    main()