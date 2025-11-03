"""
Analyze ALL 30 symbols performance for last 3 months and 1 month
"""
import json
from datetime import datetime

# Load optimal parameters
with open('../models/parameter_optimization/optimal_parameters.json', 'r') as f:
    params = json.load(f)

# Analyze recent performance for all symbols
results_3m = []
results_1m = []

for sym in params:
    if sym != 'summary':
        trade_log = params[sym]['H1']['performance_metrics']['trade_log']
        
        # Last 3 months (Aug 1 - Oct 31, 2025)
        pnl_3m = sum(t['pnl'] for t in trade_log if t['exit_time'] >= '2025-08-01')
        trades_3m = len([t for t in trade_log if t['exit_time'] >= '2025-08-01'])
        winners_3m = len([t for t in trade_log if t['exit_time'] >= '2025-08-01' and t['pnl'] > 0])
        wr_3m = (winners_3m / trades_3m * 100) if trades_3m > 0 else 0
        
        # Last 1 month (Oct 1 - Oct 31, 2025)
        pnl_1m = sum(t['pnl'] for t in trade_log if t['exit_time'] >= '2025-10-01')
        trades_1m = len([t for t in trade_log if t['exit_time'] >= '2025-10-01'])
        winners_1m = len([t for t in trade_log if t['exit_time'] >= '2025-10-01' and t['pnl'] > 0])
        wr_1m = (winners_1m / trades_1m * 100) if trades_1m > 0 else 0
        
        results_3m.append({
            'symbol': sym,
            'pnl': pnl_3m,
            'trades': trades_3m,
            'win_rate': wr_3m
        })
        
        results_1m.append({
            'symbol': sym,
            'pnl': pnl_1m,
            'trades': trades_1m,
            'win_rate': wr_1m
        })

# Sort by PnL
results_3m.sort(key=lambda x: x['pnl'], reverse=True)
results_1m.sort(key=lambda x: x['pnl'], reverse=True)

print("\n" + "=" * 100)
print("[ALL SYMBOLS] ALL 30 SYMBOLS - RECENT PERFORMANCE RANKINGS")
print("=" * 100)

print("\n[TOP 4 3M] TOP 4 MOST PROFITABLE - LAST 3 MONTHS (Aug-Oct 2025)")
print("-" * 100)
print(f"{'Rank':<6} {'Symbol':<10} {'PnL':<18} {'Trades':<10} {'Win Rate':<12}")
print("-" * 100)

for i, data in enumerate(results_3m[:4], 1):
    print(f"{i:<6} {data['symbol']:<10} ${data['pnl']:>15,.2f} {data['trades']:>8} {data['win_rate']:>8.1f}%")

print("\n\n[HOT] TOP 4 MOST PROFITABLE - LAST 1 MONTH (Oct 2025)")
print("-" * 100)
print(f"{'Rank':<6} {'Symbol':<10} {'PnL':<18} {'Trades':<10} {'Win Rate':<12}")
print("-" * 100)

for i, data in enumerate(results_1m[:4], 1):
    print(f"{i:<6} {data['symbol']:<10} ${data['pnl']:>15,.2f} {data['trades']:>8} {data['win_rate']:>8.1f}%")

# Show combined top 4 unique symbols across both periods
combined_symbols = set()
for data in results_3m[:4]:
    combined_symbols.add(data['symbol'])
for data in results_1m[:4]:
    combined_symbols.add(data['symbol'])

print("\n\n[TOP PERFORMERS] COMBINED TOP PERFORMERS (appear in either top 4 list):")
print("-" * 100)
print(f"{'Symbol':<10} {'3-Month PnL':<18} {'1-Month PnL':<18} {'Combined':<18}")
print("-" * 100)

combined_data = []
for sym in combined_symbols:
    pnl_3m = next((d['pnl'] for d in results_3m if d['symbol'] == sym), 0)
    pnl_1m = next((d['pnl'] for d in results_1m if d['symbol'] == sym), 0)
    combined_data.append({
        'symbol': sym,
        'pnl_3m': pnl_3m,
        'pnl_1m': pnl_1m,
        'combined': pnl_3m + pnl_1m
    })

combined_data.sort(key=lambda x: x['combined'], reverse=True)

for data in combined_data:
    print(f"{data['symbol']:<10} ${data['pnl_3m']:>15,.2f} ${data['pnl_1m']:>15,.2f} ${data['combined']:>15,.2f}")

# Show full rankings
print("\n\n" + "=" * 100)
print("[COMPLETE 3M] COMPLETE RANKINGS - LAST 3 MONTHS")
print("=" * 100)
print(f"{'Rank':<6} {'Symbol':<10} {'PnL':<18} {'Trades':<10} {'Win Rate':<12}")
print("-" * 100)

for i, data in enumerate(results_3m, 1):
    print(f"{i:<6} {data['symbol']:<10} ${data['pnl']:>15,.2f} {data['trades']:>8} {data['win_rate']:>8.1f}%")

print("\n\n" + "=" * 100)
print("[COMPLETE 1M] COMPLETE RANKINGS - LAST 1 MONTH")
print("=" * 100)
print(f"{'Rank':<6} {'Symbol':<10} {'PnL':<18} {'Trades':<10} {'Win Rate':<12}")
print("-" * 100)

for i, data in enumerate(results_1m, 1):
    print(f"{i:<6} {data['symbol']:<10} ${data['pnl']:>15,.2f} {data['trades']:>8} {data['win_rate']:>8.1f}%")

print("\n" + "=" * 100)
