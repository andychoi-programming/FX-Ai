"""import sqlite3

Check Adaptive Learning Status

Shows what the system has learned and if it will apply changes to next tradesconn = sqlite3.connect('data/performance_history.db')

"""cursor = conn.cursor()

import sqlite3

import os# Check total trades

from datetime import datetime, timedeltacursor.execute('SELECT COUNT(*) FROM trades')

total = cursor.fetchone()[0]

def check_learning_status():print(f'Total trades recorded: {total}')

    db_path = 'data/adaptive_learning.db'

    if total > 0:

    if not os.path.exists(db_path):    # Top symbols

        print("[WARNING] Adaptive learning database not found!")    cursor.execute('''

        print(f"Expected at: {db_path}")        SELECT symbol, COUNT(*), AVG(profit), SUM(profit) 

        return        FROM trades 

            GROUP BY symbol 

    conn = sqlite3.connect(db_path)        ORDER BY SUM(profit) DESC 

    cursor = conn.cursor()        LIMIT 5

        ''')

    print("=" * 80)    print('\nTop 5 symbols by profit:')

    print("ADAPTIVE LEARNING STATUS CHECK")    for row in cursor.fetchall():

    print("=" * 80)        print(f'  {row[0]}: {row[1]} trades, avg=${row[2]:.2f}, total=${row[3]:.2f}')

    print()    

        # Recent trades

    # Check tables    cursor.execute('''

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")        SELECT timestamp, symbol, direction, profit, profit_pct 

    tables = cursor.fetchall()        FROM trades 

    print(f"Database tables: {[t[0] for t in tables]}")        ORDER BY timestamp DESC 

    print()        LIMIT 5

        ''')

    # Check total trades    print('\nMost recent 5 trades:')

    cursor.execute("SELECT COUNT(*) FROM trades")    for row in cursor.fetchall():

    total_trades = cursor.fetchone()[0]        print(f'  {row[0]} | {row[1]} {row[2]} | ${row[3]:.2f} ({row[4]:.2f}%)')

    print(f"Total trades recorded: {total_trades}")else:

    print()    print('\nNo trades recorded yet. System will learn as trades complete.')

    

    # Check recent trades (last 24 hours)conn.close()

    yesterday = (datetime.now() - timedelta(days=1)).isoformat()
    cursor.execute("""
        SELECT COUNT(*) FROM trades 
        WHERE timestamp > ?
    """, (yesterday,))
    recent_trades = cursor.fetchone()[0]
    print(f"Trades in last 24 hours: {recent_trades}")
    print()
    
    # Show last 10 trades
    print("=" * 80)
    print("LAST 10 TRADES")
    print("=" * 80)
    cursor.execute("""
        SELECT timestamp, symbol, direction, profit, duration_minutes, signal_strength
        FROM trades 
        ORDER BY timestamp DESC 
        LIMIT 10
    """)
    
    trades = cursor.fetchall()
    if trades:
        print(f"{'Timestamp':<20} {'Symbol':<8} {'Dir':<4} {'Profit':>8} {'Duration':>8} {'Signal':>7}")
        print("-" * 80)
        for trade in trades:
            timestamp, symbol, direction, profit, duration, signal = trade
            profit_str = f"${profit:.2f}" if profit else "$0.00"
            duration_str = f"{duration}m" if duration else "N/A"
            signal_str = f"{signal:.3f}" if signal else "N/A"
            print(f"{timestamp:<20} {symbol:<8} {direction:<4} {profit_str:>8} {duration_str:>8} {signal_str:>7}")
    else:
        print("No trades found")
    print()
    
    # Check optimal holding times
    print("=" * 80)
    print("SYMBOL-SPECIFIC OPTIMAL HOLDING TIMES")
    print("=" * 80)
    cursor.execute("""
        SELECT symbol, optimal_holding_hours, max_holding_minutes, confidence_score, sample_size
        FROM symbol_optimal_holding
        ORDER BY confidence_score DESC
        LIMIT 15
    """)
    
    holdings = cursor.fetchall()
    if holdings:
        print(f"{'Symbol':<8} {'Optimal Hours':>13} {'Max Minutes':>12} {'Confidence':>11} {'Trades':>7}")
        print("-" * 80)
        for holding in holdings:
            symbol, opt_hours, max_mins, confidence, samples = holding
            print(f"{symbol:<8} {opt_hours:>13.1f}h {max_mins:>12.0f}m {confidence:>11.2%} {samples:>7}")
    else:
        print("No optimal holding times calculated yet")
    print()
    
    # Check if learning is actually being applied
    print("=" * 80)
    print("LEARNING APPLICATION STATUS")
    print("=" * 80)
    
    # Check if trades are being closed based on optimal times
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            AVG(duration_minutes) as avg_duration,
            MIN(duration_minutes) as min_duration,
            MAX(duration_minutes) as max_duration
        FROM trades
        WHERE timestamp > ?
    """, (yesterday,))
    
    stats = cursor.fetchone()
    if stats and stats[0] > 0:
        total, avg_dur, min_dur, max_dur = stats
        print(f"Recent trade durations (last 24h):")
        print(f"  Average: {avg_dur:.1f} minutes ({avg_dur/60:.1f} hours)")
        print(f"  Minimum: {min_dur:.1f} minutes ({min_dur/60:.1f} hours)")
        print(f"  Maximum: {max_dur:.1f} minutes ({max_dur/60:.1f} hours)")
        print()
        
        if max_dur > 300:  # More than 5 hours
            print("⚠️  WARNING: Some trades held for >5 hours")
            print("   System may not be applying optimal holding times correctly")
    else:
        print("No recent trades to analyze")
    
    print()
    
    # Check win rate by holding time buckets
    print("=" * 80)
    print("WIN RATE BY HOLDING TIME")
    print("=" * 80)
    cursor.execute("""
        SELECT 
            CASE 
                WHEN duration_minutes < 60 THEN '0-1h'
                WHEN duration_minutes < 120 THEN '1-2h'
                WHEN duration_minutes < 180 THEN '2-3h'
                WHEN duration_minutes < 240 THEN '3-4h'
                WHEN duration_minutes < 300 THEN '4-5h'
                ELSE '5h+'
            END as time_bucket,
            COUNT(*) as trades,
            SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as wins,
            AVG(profit) as avg_profit
        FROM trades
        WHERE duration_minutes IS NOT NULL
        GROUP BY time_bucket
        ORDER BY 
            CASE time_bucket
                WHEN '0-1h' THEN 1
                WHEN '1-2h' THEN 2
                WHEN '2-3h' THEN 3
                WHEN '3-4h' THEN 4
                WHEN '4-5h' THEN 5
                ELSE 6
            END
    """)
    
    buckets = cursor.fetchall()
    if buckets:
        print(f"{'Duration':<10} {'Trades':>7} {'Wins':>7} {'Win Rate':>10} {'Avg P/L':>10}")
        print("-" * 80)
        for bucket in buckets:
            time_range, trades, wins, avg_profit = bucket
            win_rate = (wins / trades * 100) if trades > 0 else 0
            print(f"{time_range:<10} {trades:>7} {wins:>7} {win_rate:>9.1f}% ${avg_profit:>9.2f}")
    else:
        print("No duration data available")
    
    conn.close()

if __name__ == "__main__":
    check_learning_status()
