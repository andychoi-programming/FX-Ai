"""
Analyze symbol performance by time periods from trade logs
"""
import json
from datetime import datetime
from collections import defaultdict

# Load optimal parameters
with open('../models/parameter_optimization/optimal_parameters.json', 'r') as f:
    params = json.load(f)

def analyze_periods(symbol, trade_log):
    """Analyze trades by year"""
    # Group trades by year
    year_pnl = defaultdict(float)
    year_trades = defaultdict(int)
    year_winners = defaultdict(int)
    
    for trade in trade_log:
        exit_time = trade['exit_time']
        year = exit_time[:4]  # Extract year from timestamp
        
        year_pnl[year] += trade['pnl']
        year_trades[year] += 1
        if trade['pnl'] > 0:
            year_winners[year] += 1
    
    return year_pnl, year_trades, year_winners

# Analyze all symbols
results = {}
for sym in params:
    if sym != 'summary':
        trade_log = params[sym]['H1']['performance_metrics']['trade_log']
        year_pnl, year_trades, year_winners = analyze_periods(sym, trade_log)
        
        # Calculate win rates
        year_wr = {}
        for year in year_trades:
            year_wr[year] = (year_winners[year] / year_trades[year] * 100) if year_trades[year] > 0 else 0
        
        results[sym] = {
            'total_pnl': params[sym]['H1']['performance_metrics']['pnl'],
            'year_pnl': dict(year_pnl),
            'year_trades': dict(year_trades),
            'year_wr': year_wr
        }

# Sort by total PnL
sorted_symbols = sorted(results.items(), key=lambda x: x[1]['total_pnl'], reverse=True)

print("\n" + "=" * 100)
print("[PERIOD ANALYSIS] PERIOD ANALYSIS - TOP 4 SYMBOLS BY YEAR")
print("=" * 100)

for i, (sym, data) in enumerate(sorted_symbols[:4], 1):
    print(f"\n{i}. {sym} - Total: ${data['total_pnl']:,.2f}")
    print("-" * 100)
    print(f"{'Year':<8} {'PnL':<15} {'Trades':<10} {'Win Rate':<12}")
    print("-" * 100)
    
    years = sorted(data['year_pnl'].keys())
    for year in years:
        pnl = data['year_pnl'][year]
        trades = data['year_trades'][year]
        wr = data['year_wr'][year]
        print(f"{year:<8} ${pnl:>12,.2f} {trades:>8} {wr:>8.1f}%")

# Calculate last 3 months and 1 month for top symbols
print("\n\n" + "=" * 100)
print("[RECENT PERFORMANCE] RECENT PERFORMANCE - LAST 3 MONTHS & 1 MONTH (Oct-Nov 2025)")
print("=" * 100)
print("\nNote: Analyzing trades from Aug 2025 onwards for recent performance...")

for i, (sym, data) in enumerate(sorted_symbols[:4], 1):
    trade_log = params[sym]['H1']['performance_metrics']['trade_log']
    
    # Get last 3 months (Aug 1 - Oct 31) and last 1 month (Oct 1 - Oct 31)
    pnl_3m = sum(t['pnl'] for t in trade_log if t['exit_time'] >= '2025-08-01')
    trades_3m = len([t for t in trade_log if t['exit_time'] >= '2025-08-01'])
    winners_3m = len([t for t in trade_log if t['exit_time'] >= '2025-08-01' and t['pnl'] > 0])
    
    pnl_1m = sum(t['pnl'] for t in trade_log if t['exit_time'] >= '2025-10-01')
    trades_1m = len([t for t in trade_log if t['exit_time'] >= '2025-10-01'])
    winners_1m = len([t for t in trade_log if t['exit_time'] >= '2025-10-01' and t['pnl'] > 0])
    
    wr_3m = (winners_3m / trades_3m * 100) if trades_3m > 0 else 0
    wr_1m = (winners_1m / trades_1m * 100) if trades_1m > 0 else 0
    
    print(f"\n{i}. {sym}")
    print("-" * 100)
    print(f"{'Period':<15} {'PnL':<15} {'Trades':<10} {'Win Rate':<12}")
    print("-" * 100)
    print(f"{'Last 3 months':<15} ${pnl_3m:>12,.2f} {trades_3m:>8} {wr_3m:>8.1f}%")
    print(f"{'Last 1 month':<15} ${pnl_1m:>12,.2f} {trades_1m:>8} {wr_1m:>8.1f}%")

print("\n" + "=" * 100)
