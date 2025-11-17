"""Reset daily trade counters to fix trading blocks"""

import os
import json
from datetime import datetime

# Path to the daily trade database
db_path = "data/daily_trade_counts.json"

print("[EMOJI] DAILY TRADE COUNTER RESET")
print("=" * 40)

# Check if database exists
if os.path.exists(db_path):
    print(f"[EMOJI] Found database: {db_path}")

    # Read current data
    try:
        with open(db_path, 'r') as f:
            data = json.load(f)

        print("[CHART] Current trade counts:")
        for symbol, info in data.items():
            print(f"   {symbol}: {info.get('count', 0)} trades on {info.get('date', 'unknown')}")

        # Reset all counters
        reset_data = {}
        for symbol in data.keys():
            reset_data[symbol] = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'count': 0,
                'timestamp': datetime.now().timestamp()
            }

        # Save reset data
        with open(db_path, 'w') as f:
            json.dump(reset_data, f, indent=2)

        print("[PASS] All trade counters reset to 0")

    except Exception as e:
        print(f"[FAIL] Error reading database: {e}")

else:
    print(f"[INFO]  Database not found at {db_path} - this is normal for first run")

print("\n[CYCLE] In-Memory Reset:")
print("   The system will automatically reset in-memory counters on next startup")
print("   All symbols will be allowed 5 trades each for today")

print("\n[PASS] RESET COMPLETE")
print("   Run the system now - daily limits should be cleared!")