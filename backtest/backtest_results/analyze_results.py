import pandas as pd
import numpy as np

# Read the trades data
df = pd.read_csv('trades.csv')

print('=== COMPREHENSIVE BACKTEST RESULTS SUMMARY ===')
print('=' * 60)
print(f'Total Trades: {len(df)}')
print(f'Unique Symbols: {df["symbol"].nunique()}')
print(f'Test Period: 2024-11-04 to 2025-10-31 (1 year)')
print()

# Symbol distribution
print('TRADES BY SYMBOL:')
symbol_counts = df['symbol'].value_counts()
for symbol, count in symbol_counts.head(15).items():
    print(f'{symbol}: {count} trades')
if len(symbol_counts) > 15:
    print(f'... and {len(symbol_counts) - 15} more symbols')
print()

# Win/Loss by symbol
print('WIN RATE BY SYMBOL (Top 10):')
win_rates = {}
for symbol in df['symbol'].unique():
    symbol_trades = df[df['symbol'] == symbol]
    wins = len(symbol_trades[symbol_trades['pnl'] > 0])
    total = len(symbol_trades)
    win_rate = (wins / total * 100) if total > 0 else 0
    win_rates[symbol] = win_rate

# Sort by win rate descending
sorted_win_rates = sorted(win_rates.items(), key=lambda x: x[1], reverse=True)
for symbol, win_rate in sorted_win_rates[:10]:
    count = symbol_counts[symbol]
    print(f'{symbol}: {win_rate:.1f}% win rate ({count} trades)')
print()

# Overall P&L
total_pnl = df['pnl'].sum()
winning_trades = len(df[df['pnl'] > 0])
losing_trades = len(df[df['pnl'] < 0])
win_rate = (winning_trades / len(df)) * 100

print('OVERALL PERFORMANCE:')
print(f'Total P&L: ${total_pnl:.2f}')
print(f'Win Rate: {win_rate:.1f}% ({winning_trades} wins, {losing_trades} losses)')
print(f'Average P&L per trade: ${total_pnl/len(df):.4f}')
print()

# Best and worst symbols by P&L
print('P&L BY SYMBOL (Top 5 Best & Worst):')
symbol_pnl = df.groupby('symbol')['pnl'].sum().sort_values(ascending=False)
print('Best Performing:')
for symbol, pnl in symbol_pnl.head(5).items():
    count = symbol_counts[symbol]
    avg_pnl = pnl / count
    print(f'{symbol}: ${pnl:.2f} total (${avg_pnl:.4f} avg, {count} trades)')
print()
print('Worst Performing:')
for symbol, pnl in symbol_pnl.tail(5).items():
    count = symbol_counts[symbol]
    avg_pnl = pnl / count
    print(f'{symbol}: ${pnl:.2f} total (${avg_pnl:.4f} avg, {count} trades)')
print()

# Risk metrics
print('RISK ANALYSIS:')
max_drawdown = abs(df['pnl'].cumsum().min())
print(f'Max Drawdown: ${max_drawdown:.2f}')
print(f'Profit Factor: {(df[df["pnl"] > 0]["pnl"].sum()) / abs(df[df["pnl"] < 0]["pnl"].sum()):.2f}')
print()

print('BACKTEST SETTINGS:')
print('- All 30 symbols included')
print('- Max 30 concurrent positions')
print('- $50 risk per trade')
print('- 1-year historical data (2024-11-04 to 2025-10-31)')
print('- H1 timeframe')
print('- ML confidence threshold: 0.6')