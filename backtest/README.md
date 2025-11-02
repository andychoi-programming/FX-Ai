# FX-Ai Backtest Module

This module provides backtesting capabilities for the FX-Ai trading system without modifying the main program.

## Overview

The backtest module simulates trading using historical data from MetaTrader 5 and evaluates the performance of your trading strategies. It uses the existing components from your main program (ML predictor, technical analyzer, risk manager, etc.) to generate realistic trading signals and track performance.

## Features

- **Historical Data Integration**: Uses MT5 historical data for accurate backtesting
- **Component Reuse**: Leverages existing ML models, technical analysis, and risk management
- **Comprehensive Metrics**: Calculates win rate, profit factor, drawdown, Sharpe ratio, and more
- **Trade Logging**: Detailed trade records with entry/exit times and P&L
- **Risk Management**: Implements position sizing, stop losses, and take profits
- **Performance Reports**: Generates detailed performance analysis reports

## Files Structure

```
backtest/
├── __init__.py              # Module initialization
├── backtest_config.py       # Configuration settings
├── backtest_engine.py       # Main backtest simulation engine
├── performance_metrics.py   # Performance calculation and reporting
├── run_backtest.py          # Script to execute backtest
└── README.md               # This file
```

## Configuration

Edit `backtest_config.py` to customize:

- **Date Range**: Set `start_date` and `end_date` for backtest period
- **Symbols**: List of currency pairs to trade in `symbols`
- **Capital**: Set `initial_capital` for starting balance
- **Risk Settings**: Configure `max_risk_per_trade`, stop loss/take profit levels
- **ML Thresholds**: Set `min_confidence` for trade signals

## Usage

1. **Ensure MT5 is Connected**: Make sure your main program can connect to MT5
2. **Configure Settings**: Edit `backtest_config.py` as needed
3. **Run Backtest**:

   ```bash
   cd backtest
   python run_backtest.py
   ```

## Output

The backtest generates:

- **Console Output**: Real-time progress and summary results
- **Performance Report**: Detailed metrics in `backtest_results/performance_report.txt`
- **Trades CSV**: Complete trade log in `backtest_results/trades.csv`
- **Log File**: Detailed execution log in `backtest.log`

## Key Metrics Explained

- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit divided by gross loss
- **Max Drawdown**: Largest peak-to-valley decline
- **Sharpe Ratio**: Risk-adjusted return measure
- **Calmar Ratio**: Annual return divided by max drawdown

## Important Notes

- **No Main Program Modification**: This backtest uses your existing components without changing them
- **Historical Data Required**: Ensure MT5 has sufficient historical data for your test period
- **Memory Usage**: Large date ranges may require significant memory
- **Execution Time**: Backtests can take time depending on data size and complexity

## Troubleshooting

- **No Trades Generated**: Check MT5 connection and historical data availability
- **Memory Errors**: Reduce date range or number of symbols
- **Import Errors**: Ensure you're running from the correct directory with proper Python path

## Extending the Backtest

You can extend the backtest by:

- Adding new performance metrics in `performance_metrics.py`
- Implementing different position sizing strategies
- Adding market regime filters
- Incorporating walk-forward analysis
- Adding Monte Carlo simulation for robustness testing

## Dependencies

Requires all dependencies from the main FX-Ai program, plus:

- pandas
- numpy
- matplotlib (optional, for charts)