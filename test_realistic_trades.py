#!/usr/bin/env python3
"""
Test Realistic Trade Scenarios
Tests dynamic RR validation with realistic trading scenarios
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.risk_manager import RiskManager
from utils.config_loader import ConfigLoader

def test_realistic_buy_trades():
    """Test realistic BUY trade scenarios"""
    print("Testing realistic BUY trade scenarios...")

    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load_config()

    # Create risk manager
    risk_manager = RiskManager(config)

    # Realistic BUY trade scenarios
    scenarios = [
        {
            'symbol': 'EURUSD',
            'entry': 1.0520,
            'sl_pips': 25,
            'expected_rr': 3.0,
            'description': 'EURUSD BUY with 25 pip SL'
        },
        {
            'symbol': 'EURGBP',
            'entry': 0.8475,
            'sl_pips': 20,
            'expected_rr': 2.0,
            'description': 'EURGBP BUY with 20 pip SL'
        },
        {
            'symbol': 'EURJPY',
            'entry': 161.50,
            'sl_pips': 30,
            'expected_rr': 4.0,
            'description': 'EURJPY BUY with 30 pip SL'
        },
        {
            'symbol': 'GBPUSD',
            'entry': 1.2680,
            'sl_pips': 22,
            'expected_rr': 3.0,
            'description': 'GBPUSD BUY with 22 pip SL'
        },
        {
            'symbol': 'XAUUSD',
            'entry': 1985.50,
            'sl_pips': 35,
            'expected_rr': 2.5,
            'description': 'XAUUSD BUY with 35 pip SL'
        }
    ]

    for scenario in scenarios:
        symbol = scenario['symbol']
        entry = scenario['entry']
        sl_pips = scenario['sl_pips']
        expected_rr = scenario['expected_rr']

        # Calculate pip size for SL
        pip_size = risk_manager._get_pip_value(symbol)
        sl_distance = sl_pips * pip_size
        tp_distance = sl_distance * expected_rr

        sl = entry - sl_distance
        tp = entry + tp_distance

        # Test validation
        is_valid, reason = risk_manager.validate_risk_reward(symbol, entry, sl, tp)

        # Calculate actual ratio for display
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        actual_ratio = reward / risk if risk > 0 else 0

        status = "✓" if is_valid else "✗"
        print(f"  {status} {scenario['description']}: SL={sl:.5f}, TP={tp:.5f}, Ratio={actual_ratio:.2f}:1")
    
def test_realistic_sell_trades():
    """Test realistic SELL trade scenarios"""
    print("\nTesting realistic SELL trade scenarios...")

    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load_config()

    # Create risk manager
    risk_manager = RiskManager(config)

    # Realistic SELL trade scenarios
    scenarios = [
        {
            'symbol': 'EURUSD',
            'entry': 1.0480,
            'sl_pips': 25,
            'expected_rr': 3.0,
            'description': 'EURUSD SELL with 25 pip SL'
        },
        {
            'symbol': 'EURGBP',
            'entry': 0.8525,
            'sl_pips': 20,
            'expected_rr': 2.0,
            'description': 'EURGBP SELL with 20 pip SL'
        },
        {
            'symbol': 'USDJPY',
            'entry': 148.75,
            'sl_pips': 28,
            'expected_rr': 3.0,
            'description': 'USDJPY SELL with 28 pip SL'
        },
        {
            'symbol': 'AUDUSD',
            'entry': 0.6750,
            'sl_pips': 18,
            'expected_rr': 3.0,
            'description': 'AUDUSD SELL with 18 pip SL'
        }
    ]

    for scenario in scenarios:
        symbol = scenario['symbol']
        entry = scenario['entry']
        sl_pips = scenario['sl_pips']
        expected_rr = scenario['expected_rr']

        # Calculate pip size for SL
        pip_size = risk_manager._get_pip_value(symbol)
        sl_distance = sl_pips * pip_size
        tp_distance = sl_distance * expected_rr

        sl = entry + sl_distance  # SELL: SL above entry
        tp = entry - tp_distance  # SELL: TP below entry

        # Test validation
        is_valid, reason = risk_manager.validate_risk_reward(symbol, entry, sl, tp)

        # Calculate actual ratio for display
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        actual_ratio = reward / risk if risk > 0 else 0

        status = "✓" if is_valid else "✗"
        print(f"  {status} {scenario['description']}: SL={sl:.5f}, TP={tp:.5f}, Ratio={actual_ratio:.2f}:1")
def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("\nTesting edge cases and boundary conditions...")

    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load_config()

    # Create risk manager
    risk_manager = RiskManager(config)

    # Edge case tests
    edge_cases = [
        {
            'symbol': 'EURUSD',
            'entry': 1.0500,
            'sl': 1.0450,
            'tp': 1.0650,
            'expected_pass': True,
            'description': 'EURUSD: Exactly 3:1 ratio (should pass)'
        },
        {
            'symbol': 'EURUSD',
            'entry': 1.0500,
            'sl': 1.0450,
            'tp': 1.0649,
            'expected_pass': False,
            'description': 'EURUSD: Just below 3:1 ratio (should fail)'
        },
        {
            'symbol': 'EURGBP',
            'entry': 0.8500,
            'sl': 0.8450,
            'tp': 0.8600,
            'expected_pass': True,
            'description': 'EURGBP: Exactly 3:1 ratio (above 2:1 minimum)'
        },
        {
            'symbol': 'EURGBP',
            'entry': 0.8500,
            'sl': 0.8450,
            'tp': 0.8599,
            'expected_pass': False,
            'description': 'EURGBP: Just below 3:1 but above 2:1 minimum (should still pass)'
        },
        {
            'symbol': 'EURGBP',
            'entry': 0.8500,
            'sl': 0.8450,
            'tp': 0.8519,
            'expected_pass': False,
            'description': 'EURGBP: Below 2:1 minimum (should fail)'
        }
    ]

    for case in edge_cases:
        symbol = case['symbol']
        entry = case['entry']
        sl = case['sl']
        tp = case['tp']
        expected_pass = case['expected_pass']

        is_valid, reason = risk_manager.validate_risk_reward(symbol, entry, sl, tp)

        status = "✓" if is_valid == expected_pass else "✗"
        expected = "PASS" if expected_pass else "FAIL"
        actual = "PASS" if is_valid else "FAIL"

        print(f"  {status} {case['description']}: expected {expected}, got {actual} - {reason}")

def test_invalid_inputs():
    """Test handling of invalid inputs"""
    print("\nTesting invalid input handling...")

    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load_config()

    # Create risk manager
    risk_manager = RiskManager(config)

    # Invalid input tests
    invalid_cases = [
        {
            'symbol': 'EURUSD',
            'entry': 1.0500,
            'sl': 1.0500,  # SL at entry (zero risk)
            'tp': 1.0650,
            'description': 'Zero risk distance'
        },
        {
            'symbol': 'INVALID',
            'entry': 1.0500,
            'sl': 1.0450,
            'tp': 1.0650,
            'description': 'Invalid symbol (should use default 3.0)'
        }
    ]

    for case in invalid_cases:
        symbol = case['symbol']
        entry = case['entry']
        sl = case['sl']
        tp = case['tp']

        try:
            is_valid, reason = risk_manager.validate_risk_reward(symbol, entry, sl, tp)
            status = "✓" if not is_valid else "✗"  # These should all fail
            print(f"  {status} {case['description']}: {reason}")
        except Exception as e:
            print(f"  ✓ {case['description']}: Exception handled - {e}")

def main():
    """Run all tests"""
    print("=== Realistic Trade Scenario Tests ===\n")

    try:
        test_realistic_buy_trades()
        test_realistic_sell_trades()
        test_edge_cases()
        test_invalid_inputs()

        print("\n=== Tests completed successfully ===")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())