#!/usr/bin/env python3
"""
Test All Symbols Risk-Reward Ratios
Tests RR ratios for all configured trading symbols
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

def test_all_symbols_rr_ratios():
    """Test RR ratios for all configured symbols"""
    print("Testing risk-reward ratios for all configured symbols...")

    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load_config()

    # Create risk manager
    risk_manager = RiskManager(config)

    # Get all symbols from config
    trading_config = config.get('trading', {})
    symbols = trading_config.get('symbols', [])

    print(f"Found {len(symbols)} symbols in configuration")

    # Test each symbol
    results = []
    for symbol in symbols:
        rr_ratio = risk_manager._get_symbol_min_rr(symbol)
        results.append((symbol, rr_ratio))

    # Sort by RR ratio for better display
    results.sort(key=lambda x: x[1])

    print("\nSymbol RR Ratios (sorted by ratio):")
    print("-" * 40)

    for symbol, ratio in results:
        print("2.1f")

    # Group by RR ratio
    ratio_groups = {}
    for symbol, ratio in results:
        if ratio not in ratio_groups:
            ratio_groups[ratio] = []
        ratio_groups[ratio].append(symbol)

    print("\nGrouped by RR Ratio:")
    print("-" * 40)

    for ratio in sorted(ratio_groups.keys()):
        symbols_in_group = ratio_groups[ratio]
        print("2.1f")
        for symbol in symbols_in_group:
            print(f"    {symbol}")
        print()

def test_rr_validation_all_symbols():
    """Test RR validation for all symbols with sample trades"""
    print("Testing RR validation for all symbols with sample trades...")

    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load_config()

    # Create risk manager
    risk_manager = RiskManager(config)

    # Get all symbols
    trading_config = config.get('trading', {})
    symbols = trading_config.get('symbols', [])

    # Sample trade parameters (using reasonable defaults)
    sample_prices = {
        'EURUSD': 1.0500,
        'GBPUSD': 1.2500,
        'USDJPY': 150.00,
        'USDCHF': 0.9500,
        'AUDUSD': 0.6800,
        'NZDUSD': 0.6200,
        'USDCAD': 1.3500,
        'EURGBP': 0.8500,
        'EURAUD': 1.6200,
        'EURCAD': 1.4800,
        'EURCHF': 0.9800,
        'EURJPY': 160.00,
        'EURNZD': 1.7500,
        'GBPAUD': 1.8800,
        'GBPCAD': 1.6800,
        'GBPCHF': 1.1200,
        'GBPJPY': 190.00,
        'GBPNZD': 2.0200,
        'AUDCAD': 0.9200,
        'AUDCHF': 0.6200,
        'AUDJPY': 102.00,
        'AUDNZD': 0.9200,
        'NZDCAD': 0.8400,
        'NZDCHF': 0.5600,
        'NZDJPY': 93.00,
        'CADCHF': 0.7000,
        'CADJPY': 111.00,
        'CHFJPY': 158.00,
        'XAUUSD': 1950.00,
        'XAGUSD': 24.00
    }

    print("\nTesting RR validation with sample trades:")
    print("-" * 60)

    passed = 0
    failed = 0

    for symbol in symbols:
        price = sample_prices.get(symbol, 1.0000)  # Default fallback
        min_rr = risk_manager._get_symbol_min_rr(symbol)

        # Create a trade that should pass (good RR)
        sl_distance = 0.0050 if price > 1 else price * 0.005  # Adaptive SL
        tp_distance = sl_distance * min_rr * 1.2  # Slightly better than minimum

        sl = price - sl_distance
        tp = price + tp_distance

        is_valid, reason = risk_manager.validate_risk_reward(symbol, price, sl, tp)

        status = "✓" if is_valid else "✗"
        if is_valid:
            passed += 1
        else:
            failed += 1

        print("2.1f")

    print(f"\nSummary: {passed} passed, {failed} failed")

def main():
    """Run all tests"""
    print("=== All Symbols Risk-Reward Ratio Tests ===\n")

    try:
        test_all_symbols_rr_ratios()
        test_rr_validation_all_symbols()

        print("\n=== Tests completed successfully ===")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())