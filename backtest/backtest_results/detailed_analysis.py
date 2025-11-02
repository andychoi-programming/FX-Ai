import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Read the trades data
df = pd.read_csv('trades.csv')

print('=== DETAILED SYMBOL PERFORMANCE ANALYSIS ===')
print('=' * 60)

# Convert pnl to numeric
df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')
df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')

# Basic stats
total_trades = len(df)
unique_symbols = df['symbol'].nunique()
print(f'Total Trades: {total_trades}')
print(f'Unique Symbols: {unique_symbols}')
print()

# Analyze symbol performance in detail
symbol_stats = []
for symbol in df['symbol'].unique():
    symbol_df = df[df['symbol'] == symbol]
    trades = len(symbol_df)
    wins = len(symbol_df[symbol_df['pnl'] > 0])
    losses = len(symbol_df[symbol_df['pnl'] < 0])
    win_rate = (wins / trades * 100) if trades > 0 else 0
    total_pnl = symbol_df['pnl'].sum()
    avg_pnl = total_pnl / trades if trades > 0 else 0
    avg_confidence = symbol_df['confidence'].mean()
    avg_bars = symbol_df['bars_held'].mean()

    symbol_stats.append({
        'symbol': symbol,
        'trades': trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'avg_confidence': avg_confidence,
        'avg_bars': avg_bars,
        'wins': wins,
        'losses': losses
    })

# Sort by total P&L
symbol_stats.sort(key=lambda x: x['total_pnl'], reverse=True)

print('SYMBOL PERFORMANCE RANKED BY TOTAL P&L:')
print('-' * 80)
print(f"{'Symbol':<8} {'Trades':<6} {'Win%':<6} {'Total P&L':<10} {'Avg P&L':<10} {'Avg Conf':<9} {'Avg Bars':<9}")
print('-' * 80)

for stat in symbol_stats:
    print(f"{stat['symbol']:<8} {stat['trades']:<6} {stat['win_rate']:<6.1f} {stat['total_pnl']:<10.2f} {stat['avg_pnl']:<10.4f} {stat['avg_confidence']:<9.2f} {stat['avg_bars']:<9.1f}")

print()
print('=== PROBLEMATIC SYMBOLS ANALYSIS ===')
print('Symbols with significant losses (> $1 loss):')

problem_symbols = [s for s in symbol_stats if s['total_pnl'] < -1.0]
for stat in problem_symbols:
    print(f"\n{stat['symbol']} - {stat['trades']} trades, ${stat['total_pnl']:.2f} loss")
    symbol_df = df[df['symbol'] == stat['symbol']]

    # Direction analysis
    buy_trades = len(symbol_df[symbol_df['direction'] == 'buy'])
    sell_trades = len(symbol_df[symbol_df['direction'] == 'sell'])
    buy_wins = len(symbol_df[(symbol_df['direction'] == 'buy') & (symbol_df['pnl'] > 0)])
    sell_wins = len(symbol_df[(symbol_df['direction'] == 'sell') & (symbol_df['pnl'] > 0)])

    print(f"  Direction: {buy_trades} buys ({buy_wins/buy_trades*100:.1f}% win), {sell_trades} sells ({sell_wins/sell_trades*100:.1f}% win)" if buy_trades > 0 and sell_trades > 0 else f"  Direction: {buy_trades} buys, {sell_trades} sells")

    # Status analysis
    status_counts = symbol_df['status'].value_counts()
    print(f"  Exit reasons: {dict(status_counts)}")

print()
print('=== HIGH VOLUME SYMBOLS ANALYSIS ===')
print('Symbols with >100 trades:')

high_volume = [s for s in symbol_stats if s['trades'] > 100]
for stat in high_volume:
    print(f"\n{stat['symbol']} - {stat['trades']} trades")
    symbol_df = df[df['symbol'] == stat['symbol']]

    # Confidence analysis
    high_conf = len(symbol_df[symbol_df['confidence'] > 0.7])
    low_conf = len(symbol_df[symbol_df['confidence'] <= 0.7])
    print(f"  High confidence (>0.7): {high_conf} trades")
    print(f"  Low confidence (≤0.7): {low_conf} trades")

    # Holding time analysis
    short_trades = len(symbol_df[symbol_df['bars_held'] <= 10])
    long_trades = len(symbol_df[symbol_df['bars_held'] > 48])
    print(f"  Short trades (≤10 bars): {short_trades}")
    print(f"  Long trades (>48 bars): {long_trades}")

print()
print('=== CORRELATION ANALYSIS ===')
print('Checking if poor performance correlates with certain factors...')

# Analyze if JPY crosses have common issues
jpy_symbols = [s for s in symbol_stats if 'JPY' in s['symbol']]
print(f"JPY crosses average win rate: {np.mean([s['win_rate'] for s in jpy_symbols]):.1f}%")
print(f"JPY crosses average P&L: ${np.mean([s['total_pnl'] for s in jpy_symbols]):.4f}")

# Analyze AUD pairs
aud_symbols = [s for s in symbol_stats if 'AUD' in s['symbol'] and s['trades'] > 50]
print(f"AUD pairs average win rate: {np.mean([s['win_rate'] for s in aud_symbols]):.1f}%")
print(f"AUD pairs average P&L: ${np.mean([s['total_pnl'] for s in aud_symbols]):.4f}")

# Analyze CAD pairs (profitable)
cad_symbols = [s for s in symbol_stats if 'CAD' in s['symbol']]
print(f"CAD pairs average win rate: {np.mean([s['win_rate'] for s in cad_symbols]):.1f}%")
print(f"CAD pairs average P&L: ${np.mean([s['total_pnl'] for s in cad_symbols]):.4f}")

print()
print('=== RECOMMENDATIONS ===')
print('Based on analysis:')
print('1. JPY crosses are consistently unprofitable - consider excluding them')
print('2. AUD pairs dominate trading volume but have mixed results')
print('3. CAD crosses show profitability - focus on these patterns')
print('4. Check if model has bias toward certain currency groups')