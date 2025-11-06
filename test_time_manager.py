#!/usr/bin/env python3
"""
TimeManager Test Script
Test the centralized time management system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.time_manager import get_time_manager
import json

def test_time_manager():
    """Test TimeManager functionality"""

    print("🕐 Testing FX-Ai TimeManager")
    print("=" * 50)

    # Initialize TimeManager (without MT5 for testing)
    time_manager = get_time_manager(mt5_connector=None)

    # Test trading allowed check
    is_allowed, reason = time_manager.is_trading_allowed()
    print(f"Trading Allowed: {is_allowed}")
    print(f"Reason: {reason}")

    # Test position closure check
    should_close, close_reason = time_manager.should_close_positions()
    print(f"Should Close Positions: {should_close}")
    print(f"Close Reason: {close_reason}")

    # Test forex session status
    session_status = time_manager.get_forex_session_status()
    print(f"Session Status: {json.dumps(session_status, indent=2, default=str)}")

    # Test comprehensive status
    status = time_manager.get_trading_status_summary()
    print(f"\n📊 Comprehensive Status:")
    print(json.dumps(status, indent=2, default=str))

    print("\n✅ TimeManager test completed!")

if __name__ == "__main__":
    test_time_manager()