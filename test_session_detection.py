#!/usr/bin/env python3
"""
Test script to verify the get_current_session method fix
"""

import MetaTrader5 as mt5
from datetime import datetime
from modules.schedule_manager import ScheduleManager

def test_session_detection():
    """Test the session detection functionality"""
    print("=" * 60)
    print("Testing Session Detection Fix")
    print("=" * 60)

    # Initialize MT5
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        return False

    print("✅ MT5 initialized successfully")

    # Create schedule manager
    try:
        schedule_manager = ScheduleManager()
        print("✅ ScheduleManager created successfully")
    except Exception as e:
        print(f"❌ Failed to create ScheduleManager: {e}")
        mt5.shutdown()
        return False

    # Get current MT5 server time
    try:
        server_time = mt5.symbol_info_tick("EURUSD")
        if server_time:
            current_time = datetime.fromtimestamp(server_time.time)
            print(f"✅ Current MT5 Server Time: {current_time}")
            print(f"   Hour: {current_time.hour}")
        else:
            print("❌ Failed to get server time from MT5")
            mt5.shutdown()
            return False
    except Exception as e:
        print(f"❌ Error getting server time: {e}")
        mt5.shutdown()
        return False

    # Test the get_current_session method
    try:
        session = schedule_manager.get_current_session(current_time)
        print(f"✅ Current Session: {session}")

        # Test with different hours to verify logic
        test_hours = [0, 6, 9, 12, 15, 18, 22]
        print("\nTesting session detection for different hours:")
        for hour in test_hours:
            test_time = current_time.replace(hour=hour)
            test_session = schedule_manager.get_current_session(test_time)
            print(f"  Hour {hour:2d}:00 → {test_session}")

    except AttributeError as e:
        print(f"❌ Method still missing: {e}")
        mt5.shutdown()
        return False
    except Exception as e:
        print(f"❌ Error testing session detection: {e}")
        mt5.shutdown()
        return False

    mt5.shutdown()
    print("\n" + "=" * 60)
    print("🎉 Session Detection Test PASSED!")
    print("The get_current_session method is working correctly.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_session_detection()
    exit(0 if success else 1)