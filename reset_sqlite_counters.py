"""Reset daily trade counters in SQLite database"""

import sqlite3
import os
from datetime import datetime

db_path = "data/performance_history.db"

print("[EMOJI] SQLITE DAILY TRADE COUNTER RESET")
print("=" * 45)

if not os.path.exists(db_path):
    print(f"[FAIL] Database not found: {db_path}")
    exit(1)

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check current daily trade counts
    cursor.execute("SELECT symbol, trade_date, trade_count FROM daily_trade_counts")
    rows = cursor.fetchall()

    print("[CHART] Current trade counts in database:")
    if rows:
        for symbol, trade_date, count in rows:
            print(f"   {symbol}: {count} trades on {trade_date}")
    else:
        print("   No trade count records found")

    # Reset all counts to 0 for today
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute('''
        UPDATE daily_trade_counts
        SET trade_count = 0, last_updated = CURRENT_TIMESTAMP
        WHERE trade_date = ?
    ''', (today,))

    # If no records exist for today, this won't affect anything
    # Let's also delete any old records to be safe
    cursor.execute('''
        DELETE FROM daily_trade_counts
        WHERE trade_date != ?
    ''', (today,))

    conn.commit()

    # Verify the reset
    cursor.execute("SELECT symbol, trade_date, trade_count FROM daily_trade_counts")
    rows = cursor.fetchall()

    print("\n[PASS] After reset:")
    if rows:
        for symbol, trade_date, count in rows:
            print(f"   {symbol}: {count} trades on {trade_date}")
    else:
        print("   All trade counts cleared")

    conn.close()

    print("\n[PASS] DATABASE RESET COMPLETE")
    print("   Daily trade counters have been reset to 0")
    print("   System will allow 5 trades per symbol today")

except Exception as e:
    print(f"[FAIL] Error resetting database: {e}")