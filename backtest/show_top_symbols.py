"""
Show top performing symbols from backtest results
"""
import json

# Load optimal parameters
with open('../models/parameter_optimization/optimal_parameters.json', 'r') as f:
    params = json.load(f)

# Extract PnL data
results = []
for sym in params:
    if sym != 'summary':
        metrics = params[sym]['H1']['performance_metrics']
        results.append({
            'symbol': sym,
            'pnl': metrics['pnl'],
            'trades': metrics['total_trades'],
            'win_rate': metrics['win_rate'] * 100,
            'winners': metrics.get('winning_trades', 0),
            'losers': metrics.get('losing_trades', 0)
        })

# Sort by PnL
results.sort(key=lambda x: x['pnl'], reverse=True)

print("\n" + "=" * 80)
print("[TOP PERFORMING] TOP PERFORMING SYMBOLS - 3-YEAR BACKTEST RESULTS")
print("=" * 80)

print("\n[TOP 4] TOP 4 MOST PROFITABLE:")
print("-" * 80)
print(f"{'Rank':<6} {'Symbol':<10} {'Total PnL':<15} {'Trades':<8} {'Win Rate':<12} {'W/L':<10}")
print("-" * 80)

top_4_pnl = 0
for i, data in enumerate(results[:4], 1):
    top_4_pnl += data['pnl']
    print(f"{i:<6} {data['symbol']:<10} ${data['pnl']:>12,.2f} {data['trades']:>6.0f} "
          f"{data['win_rate']:>8.1f}% {data['winners']:.0f}/{data['losers']:.0f}")

print("-" * 80)
print(f"{'TOP 4 COMBINED:':<26} ${top_4_pnl:>12,.2f}")

print("\n\n[ALL SYMBOLS] ALL 30 SYMBOLS RANKED BY PROFITABILITY:")
print("-" * 80)
print(f"{'Rank':<6} {'Symbol':<10} {'Total PnL':<15} {'Trades':<8} {'Win Rate':<12}")
print("-" * 80)

for i, data in enumerate(results, 1):
    print(f"{i:<6} {data['symbol']:<10} ${data['pnl']:>12,.2f} {data['trades']:>6.0f} "
          f"{data['win_rate']:>8.1f}%")

print("\n" + "=" * 80)
print(f"TOTAL ALL 30 SYMBOLS: ${sum(d['pnl'] for d in results):,.2f}")
print("=" * 80)
