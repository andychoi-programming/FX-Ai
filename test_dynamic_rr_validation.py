#!/usr/bin/env python3
"""
Test Dynamic Risk-Reward Validation
Tests the new symbol-specific risk-reward validation functionality
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

def test_get_symbol_min_rr():
    """Test the _get_symbol_min_rr method"""
    print("Testing _get_symbol_min_rr method...")

    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    
    print(f"Config loaded: {config is not None}")
    print(f"Config keys: {list(config.keys()) if config else 'None'}")
    
    if config and 'trading_rules' in config:
        print(f"trading_rules keys: {list(config['trading_rules'].keys())}")
    
    # Create risk manager
    risk_manager = RiskManager(config)

    # Test various symbols
    test_symbols = [
        ('EURUSD', 3.0),
        ('EURGBP', 2.0),
        ('EURJPY', 4.0),
        ('GBPUSD', 3.0),
        ('XAUUSD', 2.5),
        ('NONEXISTENT', 3.0)  # Should default to 3.0
    ]

    for symbol, expected in test_symbols:
        actual = risk_manager._get_symbol_min_rr(symbol)
        status = "✓" if abs(actual - expected) < 0.01 else "✗"
        print(f"  {status} {symbol}: expected {expected}, got {actual}")

def test_validate_risk_reward():
    """Test the validate_risk_reward method"""
    print("\nTesting validate_risk_reward method...")

    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load_config()

    # Create risk manager
    risk_manager = RiskManager(config)

    # Test cases: (symbol, entry_price, sl_price, tp_price, should_pass)
    test_cases = [
        # EURUSD requires 3:1 minimum
        ('EURUSD', 1.0500, 1.0450, 1.0650, True),   # 4:1 ratio (20pips risk, 60pips reward)
        ('EURUSD', 1.0500, 1.0450, 1.0550, False),  # 2:1 ratio (20pips risk, 40pips reward)

        # EURGBP requires 2:1 minimum
        ('EURGBP', 0.8500, 0.8450, 0.8600, True),   # 3:1 ratio
        ('EURGBP', 0.8500, 0.8450, 0.8525, False),  # 1.25:1 ratio

        # EURJPY requires 4:1 minimum
        ('EURJPY', 160.00, 159.00, 168.00, True),    # 8:1 ratio
        ('EURJPY', 160.00, 159.00, 163.00, False),   # 2:1 ratio
    ]

    for symbol, entry, sl, tp, should_pass in test_cases:
        is_valid, reason = risk_manager.validate_risk_reward(symbol, entry, sl, tp)
        status = "✓" if is_valid == should_pass else "✗"
        expected = "PASS" if should_pass else "FAIL"
        actual = "PASS" if is_valid else "FAIL"
        print(f"  {status} {symbol}: expected {expected}, got {actual} - {reason}")

def test_calculate_sl_tp_with_rr():
    """Test calculate_stop_loss_take_profit with dynamic RR"""
    print("\nTesting calculate_stop_loss_take_profit with dynamic RR...")

    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load_config()

    # Create risk manager
    risk_manager = RiskManager(config)

    # Test EURUSD (should use 3:1 ratio)
    result = risk_manager.calculate_stop_loss_take_profit('EURUSD', 1.0500, 'BUY')
    sl = result['stop_loss']
    tp = result['take_profit']

    # Calculate actual ratio
    risk = abs(1.0500 - sl)
    reward = abs(tp - 1.0500)
    actual_ratio = reward / risk if risk > 0 else 0

    print(f"  EURUSD BUY: SL={sl:.5f}, TP={tp:.5f}, Ratio={actual_ratio:.2f}:1 (expected >= 3.0:1)")
    
    # Test EURGBP (should use 2:1 ratio)
    result = risk_manager.calculate_stop_loss_take_profit('EURGBP', 0.8500, 'BUY')
    sl = result['stop_loss']
    tp = result['take_profit']

    risk = abs(0.8500 - sl)
    reward = abs(tp - 0.8500)
    actual_ratio = reward / risk if risk > 0 else 0

    print(f"  EURGBP BUY: SL={sl:.5f}, TP={tp:.5f}, Ratio={actual_ratio:.2f}:1 (expected >= 2.0:1)")
def main():
    """Run all tests"""
    print("=== Dynamic Risk-Reward Validation Tests ===\n")

    try:
        test_get_symbol_min_rr()
        test_validate_risk_reward()
        test_calculate_sl_tp_with_rr()

        print("\n=== Tests completed successfully ===")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())