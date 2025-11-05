#!/usr/bin/env python3
"""Quick analysis of existing trades in the database"""

import sqlite3
from collections import defaultdict

conn = sqlite3.connect('data/performance_history.db')
c = conn.cursor()

# Get all trades
c.execute('SELECT symbol, duration_minutes, profit FROM trades WHERE duration_minutes IS NOT NULL')
trades = c.fetchall()

print(f"\n{'='*70}")
print(f"ANALYSIS OF {len(trades)} EXISTING TRADES")
print(f"{'='*70}\n")

# Group by duration
duration_groups = {
    'under 30min': [],
    '30-60min': [],
    '1-2hr': [],
    '2-4hr': [],
    'over 4hr': []
}

for symbol, duration, profit in trades:
    if duration < 30:
        group = 'under 30min'
    elif duration < 60:
        group = '30-60min'
    elif duration < 120:
        group = '1-2hr'
    elif duration < 240:
        group = '2-4hr'
    else:
        group = 'over 4hr'
    
    duration_groups[group].append((symbol, duration, profit))

# Print analysis
print("Performance by Duration:")
print("-" * 70)
print(f"{'Duration':<15} | {'Trades':>6} | {'Winners':>7} | {'Win Rate':>8} | {'Avg Profit':>11}")
print("-" * 70)

for group, group_trades in duration_groups.items():
    if group_trades:
        count = len(group_trades)
        winners = sum(1 for _, _, p in group_trades if p > 0)
        win_rate = (winners / count * 100) if count > 0 else 0
        avg_profit = sum(p for _, _, p in group_trades) / count if count > 0 else 0
        
        print(f"{group:<15} | {count:>6} | {winners:>7} | {win_rate:>7.1f}% | ${avg_profit:>10.2f}")

print()

# Symbol-specific recommendations
print("\nRECOMMENDATIONS based on existing trades:")
print("-" * 70)

symbol_data = defaultdict(lambda: {'trades': [], 'profits': []})
for symbol, duration, profit in trades:
    symbol_data[symbol]['trades'].append(duration)
    symbol_data[symbol]['profits'].append(profit)

for symbol in sorted(symbol_data.keys(), key=lambda x: len(symbol_data[x]['trades']), reverse=True)[:5]:
    data = symbol_data[symbol]
    avg_duration = sum(data['trades']) / len(data['trades'])
    avg_profit = sum(data['profits']) / len(data['profits'])
    win_rate = sum(1 for p in data['profits'] if p > 0) / len(data['profits']) * 100
    
    print(f"{symbol}: {len(data['trades'])} trades, avg {avg_duration:.0f}min, "
          f"win rate {win_rate:.1f}%, avg profit ${avg_profit:.2f}")

# Check current defaults
c.execute('SELECT symbol, optimal_holding_hours, max_holding_minutes FROM symbol_optimal_holding LIMIT 1')
default = c.fetchone()
if default:
    print(f"\n{'='*70}")
    print(f"CURRENT DEFAULT SETTINGS: {default[1]}h optimal, {default[2]}min max")
    print(f"{'='*70}")
    print("\nNOTE: These defaults (2h/4h) may be too long based on historical data showing")
    print("      most trades complete in under 30 minutes.")

conn.close()
