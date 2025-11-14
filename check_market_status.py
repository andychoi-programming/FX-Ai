#!/usr/bin/env python3
"""
Check if market is open and why orders are failing
"""
import sys
sys.path.insert(0, '.')

from core.mt5_connector import MT5Connector
import os
from dotenv import load_dotenv
from datetime import datetime

def check_market():
    load_dotenv()

    mt5 = MT5Connector(
        os.getenv('MT5_LOGIN'),
        os.getenv('MT5_PASSWORD'),
        os.getenv('MT5_SERVER')
    )

    if not mt5.connect():
        print("❌ Cannot connect to MT5")
        return

    print("Market Status Check")
    print("="*60)

    # Get server time
    server_time = mt5.get_server_time()
    print(f"MT5 Server Time: {server_time}")
    print(f"Local Time: {datetime.now()}")

    # Check if it's weekend (markets closed)
    if server_time:
        weekday = server_time.weekday()  # 0=Monday, 6=Sunday
        hour = server_time.hour
        print(f"Day of week: {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][weekday]}")
        print(f"Hour (server time): {hour}")

        if weekday >= 5:  # Saturday or Sunday
            print("⚠️  WARNING: It's weekend - Forex markets are closed!")
        elif hour < 1 or hour > 23:  # Outside typical trading hours
            print("⚠️  WARNING: Outside typical trading hours (01:00-23:00 GMT)")
        else:
            print("✓ Within typical trading hours")
    else:
        print("❌ Could not get server time")

    # Check account info
    account = mt5.get_account_info()
    if account:
        print(f"\nAccount: {account.get('login')}")
        print(f"Balance: ${account.get('balance', 0):.2f}")
        print(f"Equity: ${account.get('equity', 0):.2f}")
        print(f"Margin Free: ${account.get('margin_free', 0):.2f}")
        print(f"Margin Level: {account.get('margin_level', 0):.2f}%")

    # Check if trading is allowed
    print("\nTrading Status:")

    test_symbols = ['EURUSD', 'GBPUSD', 'XAGUSD', 'XAUUSD']

    print("\nAttempting to enable symbols...")

    for symbol in test_symbols:
        if mt5.symbol_select(symbol, True):
            print(f"✓ Enabled {symbol}")
        else:
            print(f"✗ Failed to enable {symbol}")

    print("\nRechecking symbols after enabling:")
    print("="*40)

    for symbol in test_symbols:
        symbol_info = mt5.get_symbol_info(symbol)
        if symbol_info:
            print(f"\n{symbol}:")
            print(f"  Visible: {symbol_info.get('visible', False)}")
            print(f"  Selectable: {symbol_info.get('select', False)}")
            print(f"  Trade Mode: {symbol_info.get('trade_mode', 'unknown')}")
            print(f"  Trade Allowed: {symbol_info.get('trade_allowed', False)}")
            print(f"  Spread: {symbol_info.get('spread', 0)} points")
            print(f"  Bid: {symbol_info.get('bid', 0):.5f}")
            print(f"  Ask: {symbol_info.get('ask', 0):.5f}")

            # Check if symbol is properly configured
            if not symbol_info.get('visible', False):
                print("  Status: ✗ Symbol not visible - need to enable in MT5")
            elif not symbol_info.get('select', False):
                print("  Status: ✗ Symbol not selectable - need to select in MT5")
            elif not symbol_info.get('trade_allowed', False):
                print("  Status: ✗ Trading not allowed for this symbol")
            elif symbol_info.get('bid', 0) == 0:
                print("  Status: ✗ Market closed or no liquidity")
            else:
                print("  Status: ✓ Market open and tradable")
        else:
            print(f"\n{symbol}:")
            print("  Status: ✗ Symbol not found in MT5")

    mt5.disconnect()

if __name__ == "__main__":
    check_market()