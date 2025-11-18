# diagnose_mt5.py
import MetaTrader5 as mt5
from datetime import datetime

def diagnose_mt5():
    """Diagnose MT5 connection and trading capabilities"""

    print("=" * 60)
    print("MT5 DIAGNOSTIC TOOL")
    print("=" * 60)

    # Initialize MT5
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        return

    print("✅ MT5 initialized successfully")

    # Get account info
    account_info = mt5.account_info()
    if account_info is None:
        print("❌ Failed to get account info")
        return

    print(f"Account: {account_info.login}")
    print(f"Balance: ${account_info.balance:.2f}")
    print(f"Server: {account_info.server}")
    print(f"Trade allowed: {account_info.trade_allowed}")
    print(f"Trade expert: {account_info.trade_expert}")

    # Check symbols
    print("\n" + "=" * 60)
    print("CHECKING SYMBOLS")
    print("=" * 60)

    test_symbols = ['EURUSD', 'GBPUSD', 'XAUUSD', 'XAGUSD']

    for symbol in test_symbols:
        print(f"\n{symbol}:")

        # Check if symbol exists
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"  ❌ Symbol not found")
            continue

        print(f"  ✅ Symbol found")
        print(f"  Visible: {symbol_info.visible}")
        print(f"  Tradeable: {symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL}")
        print(f"  Min volume: {symbol_info.volume_min}")
        print(f"  Max volume: {symbol_info.volume_max}")
        print(f"  Volume step: {symbol_info.volume_step}")
        print(f"  Digits: {symbol_info.digits}")
        print(f"  Point: {symbol_info.point}")
        print(f"  Stops level: {symbol_info.trade_stops_level}")
        print(f"  Freeze level: {symbol_info.trade_freeze_level}")

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            print(f"  Bid: {tick.bid}, Ask: {tick.ask}")

        # Test order (dry run - not sending)
        # Use proper stop order distances - must be above stops_level
        min_stop_distance = symbol_info.trade_stops_level * symbol_info.point
        pip_distance = max(10 * (10 ** -symbol_info.digits), min_stop_distance * 2)  # At least 2x stops level
        
        if symbol in ['XAUUSD', 'XAGUSD']:
            # For metals, use larger distances
            pip_distance = max(50 * (10 ** -symbol_info.digits), min_stop_distance * 2)
        
        entry_price = tick.ask + pip_distance
        sl_price = tick.ask + pip_distance - pip_distance * 2
        tp_price = tick.ask + pip_distance + pip_distance * 3
        
        print(f"  Min stop distance: {min_stop_distance}")
        print(f"  Test order prices: Entry={entry_price:.{symbol_info.digits}f}, SL={sl_price:.{symbol_info.digits}f}, TP={tp_price:.{symbol_info.digits}f}")
        
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": symbol_info.volume_min,
            "type": mt5.ORDER_TYPE_BUY_STOP,
            "price": entry_price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 20,
            "magic": 234000,
            "comment": "Test",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Check order (without sending)
        result = mt5.order_check(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"  ✅ Order check passed")
        else:
            print(f"  ❌ Order check failed: {result.retcode if result else 'None'}")

    # Check server time
    print("\n" + "=" * 60)
    print("SERVER TIME")
    print("=" * 60)
    server_time = datetime.fromtimestamp(mt5.symbol_info_tick("EURUSD").time)
    local_time = datetime.now()
    print(f"Server time: {server_time}")
    print(f"Local time:  {local_time}")
    print(f"Difference:  {abs((server_time - local_time).total_seconds())} seconds")

    mt5.shutdown()
    print("\n✅ Diagnostic complete")

if __name__ == "__main__":
    diagnose_mt5()