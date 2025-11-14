#!/usr/bin/env python3
"""
Quick test script to verify position sizing fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import MetaTrader5 as mt5
from core.risk_manager import RiskManager
from utils.config_loader import ConfigLoader

def test_position_sizing():
    """Test the position sizing calculation and validation"""
    print("Testing position sizing fixes...")

    # Initialize MT5
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return

    print("MT5 initialized successfully")

    # Load config
    config_loader = ConfigLoader()
    config_loader.load_config()
    config = config_loader.config

    # Create risk manager
    risk_manager = RiskManager(config)

    # Test symbols that were failing
    test_symbols = ['AUDJPY', 'AUDCHF', 'AUDNZD']
    account_info = mt5.account_info()
    account_balance = account_info.balance if account_info else 6431.33

    print(f"Account balance: ${account_balance:,.2f}")
    print(f"Risk per trade: ${risk_manager.risk_per_trade}")
    print()

    for symbol in test_symbols:
        try:
            # Calculate position size
            position_size = risk_manager.calculate_position_size(symbol, 20)  # 20 pip SL

            # Calculate risk amount
            risk_amount = risk_manager.calculate_risk_for_lot_size(symbol, position_size, 20)

            # Validate position size
            is_valid = risk_manager.validate_position_size(symbol, position_size, account_balance)

            print(f"{symbol}:")
            print(f"  Position size: {position_size:.3f} lots")
            print(f"  Risk amount: ${risk_amount:.2f}")
            print(f"  Risk % of account: {risk_amount/account_balance*100:.2f}%")
            print(f"  Validation: {'PASS' if is_valid else 'FAIL'}")
            print()

        except Exception as e:
            print(f"Error testing {symbol}: {e}")
            print()

    mt5.shutdown()

if __name__ == "__main__":
    test_position_sizing()