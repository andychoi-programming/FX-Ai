#!/usr/bin/env python3
"""
Test script to verify the corrected session detection logic
"""

import MetaTrader5 as mt5
from datetime import datetime

def test_session_now():
    """Test current session detection"""
    print("=" * 60)
    print("Testing Current Session Detection")
    print("=" * 60)

    # Initialize MT5
    if not mt5.initialize():
        print("[FAIL] MT5 initialization failed")
        return False

    # Get current server time
    tick = mt5.symbol_info_tick("EURUSD")
    if not tick:
        print("[FAIL] Failed to get EURUSD tick")
        mt5.shutdown()
        return False

    server_time = datetime.fromtimestamp(tick.time)
    print(f"Server Time: {server_time}")
    print(f"Server Hour: {server_time.hour}")

    # Expected sessions based on corrected logic
    hour = server_time.hour

    if 15 <= hour < 24:
        if 15 <= hour < 19:
            expected = "overlap (London-NY)"
        else:
            expected = "new_york"
    elif 0 <= hour < 11:
        if 10 <= hour < 11:
            expected = "overlap (Tokyo-London)"
        else:
            expected = "tokyo_sydney"
    elif 10 <= hour < 19:
        expected = "london"
    else:
        expected = "closed"

    print(f"Expected Session: {expected}")

    # Test the actual method
    try:
        from modules.schedule_manager import ScheduleManager
        schedule_manager = ScheduleManager()
        actual_session = schedule_manager.get_current_session(server_time)
        print(f"Actual Session: {actual_session}")

        if actual_session == expected.split()[0]:  # Remove overlap description for comparison
            print("[PASS] Session detection CORRECT!")
            success = True
        else:
            print("[FAIL] Session detection INCORRECT!")
            success = False

    except Exception as e:
        print(f"[FAIL] Error testing session detection: {e}")
        success = False

    mt5.shutdown()

    print("\n" + "=" * 60)
    if success:
        print("[SUCCESS] Session Detection Test PASSED!")
    else:
        print("[FAIL] Session Detection Test FAILED!")
    print("=" * 60)

    return success

if __name__ == "__main__":
    test_session_now()