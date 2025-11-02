#!/usr/bin/env python3
"""
Run Backtest Script
Executes the backtest using historical data and existing trading system components
"""

import logging
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.backtest_engine import BacktestEngine
from backtest.backtest_config import BacktestConfig
from backtest.performance_metrics import PerformanceMetrics

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('backtest.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main backtest execution function"""
    print("FX-Ai Backtest Runner")
    print("=" * 50)

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Initialize configuration
        config = BacktestConfig()

        # Display configuration
        print(f"Backtest Period: {config.start_date.date()} to {config.end_date.date()}")
        print(f"Symbols: {', '.join(config.symbols)}")
        print(f"Initial Capital: ${config.initial_capital:,.2f}")
        print(f"Timeframe: {config.timeframe}")
        print(f"Max Risk per Trade: {config.max_risk_per_trade*100:.1f}%")
        print()

        # Initialize backtest engine
        logger.info("Initializing backtest engine...")
        engine = BacktestEngine(config)

        # Run backtest
        start_time = datetime.now()
        logger.info("Starting backtest execution...")

        trades_df = engine.run_backtest()

        end_time = datetime.now()
        duration = end_time - start_time

        if trades_df.empty:
            logger.error("Backtest failed - no trades generated")
            print("❌ Backtest failed - no trades were generated")
            return

        # Calculate performance metrics
        logger.info("Calculating performance metrics...")
        metrics = PerformanceMetrics(trades_df, config.initial_capital)

        # Display results
        basic_metrics = metrics.calculate_basic_metrics()
        risk_metrics = metrics.calculate_risk_metrics()

        print("BACKTEST RESULTS")
        print("-" * 30)
        print(f"Duration: {duration}")
        print(f"Total Trades: {basic_metrics['total_trades']}")
        print(f"Win Rate: {basic_metrics['win_rate']:.1%}")
        print(f"Total P&L: ${basic_metrics['total_pnl']:,.2f}")
        print(f"Profit Factor: {basic_metrics['profit_factor']:.2f}")
        print(f"Max Drawdown: {risk_metrics['max_drawdown_pct']:.1%}")
        print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
        print()

        # Save results
        logger.info("Saving results...")
        engine.save_results(trades_df)

        print("✅ Backtest completed successfully!")
        print(f"📊 Results saved to: {config.results_dir}/")
        print(f"📈 Performance report: {config.performance_report_path}")
        if config.save_trades_to_csv:
            print(f"📋 Trades CSV: {config.trades_csv_path}")

    except Exception as e:
        logger.error(f"Backtest failed with error: {e}")
        print(f"❌ Backtest failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()