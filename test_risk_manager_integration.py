#!/usr/bin/env python3
"""
Verify risk manager is properly integrated after fixes
"""
import sys
sys.path.insert(0, '.')

from core.risk_manager import RiskManager
from utils.config_loader import ConfigLoader

def test_integration():
    print("="*70)
    print("TESTING RISK MANAGER INTEGRATION")
    print("="*70)

    config = ConfigLoader().load_config()
    rm = RiskManager(config)

    # Test symbol-specific minimums
    test_cases = [
        ('EURUSD', 3.0),
        ('GBPUSD', 3.0),
        ('EURGBP', 2.0),
        ('EURJPY', 4.0),
        ('XAUUSD', 2.5),
        ('XAGUSD', 2.5),
        ('AUDNZD', 2.0),
        ('GBPJPY', 4.0),
    ]

    print("\nSymbol-Specific Minimum R:R Ratios:")
    print("-" * 40)

    all_correct = True
    for symbol, expected_rr in test_cases:
        actual_rr = rm._get_symbol_min_rr(symbol)
        status = "✅" if actual_rr == expected_rr else "❌"
        print(f"{symbol:10} | Expected: {expected_rr}:1 | Actual: {actual_rr}:1 {status}")

        if actual_rr != expected_rr:
            all_correct = False

    print("\n" + "="*70)
    if all_correct:
        print("✅ ALL SYMBOLS HAVE CORRECT MINIMUM R:R RATIOS")
        print("   Risk Manager is properly configured!")
        return 0
    else:
        print("❌ SOME SYMBOLS HAVE INCORRECT RATIOS")
        print("   Review configuration!")
        return 1

if __name__ == "__main__":
    sys.exit(test_integration())