#!/usr/bin/env python3
"""
Metal Calculations Test Script
Tests SL/TP calculations for precious metals to ensure proper risk management.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_metal_calculations():
    """Test SL/TP calculations for metals"""
    print("🧪 Testing Metal SL/TP Calculations")
    print("=" * 50)

    # Test data for XAUUSD (Gold)
    print("\n🪙 Testing XAUUSD (Gold):")
    symbol = "XAUUSD"
    current_price = 4050.00  # Current gold price
    atr = 3.0  # Realistic $3 ATR for gold
    sl_multiplier = 2.5
    tp_multiplier = 5.0

    # Calculate distances
    sl_distance = max(atr * sl_multiplier, 5.00)  # Min $5.00
    tp_distance = max(atr * tp_multiplier, 10.00)  # Min $10.00

    # Calculate prices for BUY STOP order
    stop_price = current_price + 10.00  # 10 points above current
    sl_price = stop_price - sl_distance
    tp_price = stop_price + tp_distance

    # Calculate position size for $50 risk
    risk_amount = 50.0
    pip_value = 1.0  # $1 per pip for gold
    sl_pips = sl_distance / 0.01  # Convert dollars to pips (gold pip = 0.01)
    position_size = risk_amount / (sl_pips * pip_value)

    print(f"  Current Price: ${current_price:.2f}")
    print(f"  Stop Order Price: ${stop_price:.2f}")
    print(f"  ATR: ${atr:.2f}")
    print(f"  SL Distance: ${sl_distance:.2f} ({sl_pips:.0f} pips)")
    print(f"  TP Distance: ${tp_distance:.2f} ({tp_distance/0.01:.0f} pips)")
    print(f"  SL Price: ${sl_price:.2f}")
    print(f"  TP Price: ${tp_price:.2f}")
    print(f"  Position Size for $50 risk: {position_size:.3f} lots")
    print(f"  Actual Risk: ${(position_size * sl_pips * pip_value):.2f}")

    # Test data for XAGUSD (Silver)
    print("\n🥈 Testing XAGUSD (Silver):")
    symbol = "XAGUSD"
    current_price = 50.00  # Current silver price
    atr = 0.15  # Realistic $0.15 ATR for silver
    sl_multiplier = 2.5
    tp_multiplier = 5.0

    # Calculate distances
    sl_distance = max(atr * sl_multiplier, 0.25)  # Min $0.25
    tp_distance = max(atr * tp_multiplier, 0.50)  # Min $0.50

    # Calculate prices for BUY STOP order
    stop_price = current_price + 0.50  # 50 points above current
    sl_price = stop_price - sl_distance
    tp_price = stop_price + tp_distance

    # Calculate position size for $50 risk
    risk_amount = 50.0
    pip_value = 5.0  # $5 per pip for silver (per oz)
    sl_pips = sl_distance / 0.001  # Convert dollars to pips (silver pip = 0.001)
    tp_pips = tp_distance / 0.001  # Convert dollars to pips
    position_size = risk_amount / (sl_pips * pip_value)

    print(f"  Current Price: ${current_price:.2f}")
    print(f"  Stop Order Price: ${stop_price:.2f}")
    print(f"  ATR: ${atr:.2f}")
    print(f"  SL Distance: ${sl_distance:.2f} ({sl_pips:.0f} pips)")
    print(f"  TP Distance: ${tp_distance:.2f} ({tp_pips:.0f} pips)")
    print(f"  SL Price: ${sl_price:.2f}")
    print(f"  TP Price: ${tp_price:.2f}")
    print(f"  Position Size for $50 risk: {position_size:.3f} lots")
    print(f"  Actual Risk: ${(position_size * sl_pips * pip_value):.2f}")

    print("\n✅ Metal calculations test completed!")
    print("\n📋 Summary:")
    print("  Gold: SL distances should be $5-15, TP distances $10-30")
    print("  Silver: SL distances should be $0.25-0.75, TP distances $0.50-1.50")
    print("  Position sizes should result in actual $50 risk per trade")

if __name__ == "__main__":
    test_metal_calculations()