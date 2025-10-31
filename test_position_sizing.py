#!/usr/bin/env python3
"""
Test script for position sizing fix - Cross Pairs
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MetaTrader5 as mt5
from core.risk_manager import RiskManager

def test_position_sizing():
    """Test the corrected position sizing for various pair types"""

    # Initialize MT5
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return

    # Create minimal config for testing
    config = {
        'trading': {
            'risk_per_trade': 50.0,
            'max_positions': 20,
            'max_daily_loss': 500.0,
            'max_spread': 3.0
        },
        'risk_management': {
            'symbol_cooldown_minutes': 5
        }
    }

    # Create risk manager
    risk_manager = RiskManager(config)

    # Test symbols and expected results
    test_cases = [
        ("EURUSD", 25, 50, "~0.20 lots"),
        ("GBPUSD", 25, 50, "~0.20 lots"),
        ("EURGBP", 25, 50, "~0.15 lots (NOT 0.23!)"),
        ("EURJPY", 25, 50, "~0.13 lots"),
        ("GBPJPY", 25, 50, "~0.13 lots"),
        ("USDJPY", 25, 50, "~0.13 lots"),
        ("USDCHF", 25, 50, "~0.20 lots"),
    ]

    print("🧪 Testing Position Sizing Fix for Cross Pairs")
    print("=" * 60)

    for symbol, stop_loss_pips, risk_amount, expected in test_cases:
        try:
            lot_size = risk_manager.calculate_position_size(symbol, stop_loss_pips, risk_amount)
            print(f"{symbol:8} | {lot_size:.3f} lots | Expected: {expected}")
        except Exception as e:
            print(f"{symbol:8} | ERROR: {str(e)}")

    print("\n" + "=" * 60)
    print("✅ Test completed!")

    # Shutdown MT5
    mt5.shutdown()

if __name__ == "__main__":
    test_position_sizing()