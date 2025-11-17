"""Check what session detection thinks is current"""

import MetaTrader5 as mt5
from utils.time_manager import TimeManager

mt5.initialize()

time_manager = TimeManager()
current_session = time_manager.get_current_session()

print(f"[SEARCH] SESSION DETECTION ANALYSIS")
print(f"Current session detected: {current_session}")

# Get current MT5 time
tick = mt5.symbol_info_tick("EURUSD")
if tick:
    from datetime import datetime
    server_time = datetime.fromtimestamp(tick.time)
    hour = server_time.hour
    print(f"MT5 server time: {server_time}")
    print(f"Current hour: {hour}")

    # Check what session this hour should be
    if 22 <= hour or hour < 7:
        expected = "sydney"
    elif 7 <= hour < 15:
        expected = "tokyo"
    elif 15 <= hour < 22:
        expected = "london"
    else:
        expected = "newyork"

    print(f"Expected session for hour {hour}: {expected}")
    print(f"Match: {current_session == expected}")

else:
    print("[FAIL] Cannot get MT5 time")

mt5.shutdown()