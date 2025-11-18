#!/usr/bin/env python3
"""
MT5 Order Placement Diagnostic Test
Isolates MT5 order_send functionality to debug phantom orders and response issues.
"""

import MetaTrader5 as mt5
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_mt5_connection():
    """Test basic MT5 connection and terminal state"""
    print("🔍 Testing MT5 Connection...")

    # Initialize MT5
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        return False

    print("✅ MT5 initialized successfully")

    # Check terminal info
    terminal_info = mt5.terminal_info()
    if terminal_info is None:
        print("❌ Failed to get terminal info")
        return False

    print(f"📊 Terminal: {terminal_info.name}")
    print(f"📊 Trade Allowed: {terminal_info.trade_allowed}")
    print(f"📊 Connected: {terminal_info.connected}")
    print(f"📊 Server: {getattr(terminal_info, 'server', 'N/A')}")

    if not terminal_info.trade_allowed:
        print("❌ Trading not allowed in terminal")
        return False

    return True

def test_symbol_info(symbol):
    """Test symbol information and selection"""
    print(f"\n🔍 Testing Symbol: {symbol}")

    # Select symbol
    if not mt5.symbol_select(symbol, True):
        print(f"❌ Failed to select symbol {symbol}")
        return None

    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"❌ Failed to get symbol info for {symbol}")
        return None

    print(f"✅ Symbol {symbol} selected")
    print(f"📊 Digits: {symbol_info.digits}")
    print(f"📊 Point: {symbol_info.point}")
    print(f"📊 Volume Min: {symbol_info.volume_min}")
    print(f"📊 Volume Max: {symbol_info.volume_max}")

    # Get current tick
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"❌ Failed to get tick data for {symbol}")
        return None

    print(f"📊 Current Bid: {tick.bid}")
    print(f"📊 Current Ask: {tick.ask}")
    print(f"📊 Spread: {(tick.ask - tick.bid):.5f}")

    return tick

def test_order_placement(symbol, tick):
    """Test placing a simple pending order"""
    print(f"\n🔍 Testing Order Placement for {symbol}")

    # Calculate order price (small distance from current price)
    current_price = (tick.ask + tick.bid) / 2
    pip_size = 0.0001 if symbol.endswith('USD') else 0.01  # Rough pip size
    distance = 10 * pip_size  # 10 pips away

    # Create buy stop order
    request = {
        'action': mt5.TRADE_ACTION_PENDING,
        'symbol': symbol,
        'volume': 0.01,
        'type': mt5.ORDER_TYPE_BUY_STOP,
        'price': current_price + distance,
        'sl': current_price - distance,
        'tp': current_price + (distance * 2),
        'magic': 202411,
        'comment': 'FX-Ai Diagnostic Test'
    }

    print("📤 MT5 Request:")
    for key, value in request.items():
        print(f"   {key}: {value}")

    # Send order
    start_time = time.time()
    result = mt5.order_send(request)
    end_time = time.time()

    print(".3f")
    if result is None:
        print("❌ MT5 Response: None - No response from MT5")
        return False

    print("📥 MT5 Response:")
    print(f"   retcode: {result.retcode}")
    print(f"   order: {getattr(result, 'order', 'N/A')}")
    print(f"   volume: {getattr(result, 'volume', 'N/A')}")
    print(f"   price: {getattr(result, 'price', 'N/A')}")
    print(f"   comment: {getattr(result, 'comment', 'N/A')}")

    # Check success
    success_codes = [mt5.TRADE_RETCODE_PLACED, mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL]
    if result.retcode in success_codes:
        print(f"✅ Order placed successfully! Ticket: {result.order}")

        # Check if order appears in MT5
        time.sleep(0.5)  # Brief wait
        orders = mt5.orders_get(symbol=symbol)
        if orders:
            our_order = None
            for order in orders:
                if order.ticket == result.order:
                    our_order = order
                    break

            if our_order:
                print("✅ Order verified in MT5 orders list")
                print(f"   Status: {our_order.state}")
                print(f"   Type: {our_order.type}")
                print(f"   Price: {our_order.price_open}")
            else:
                print("⚠️ Order not found in MT5 orders list - possible phantom order")
        else:
            print("⚠️ No orders found for symbol - possible phantom order")

        return True
    else:
        print(f"❌ Order failed with retcode {result.retcode}")
        if hasattr(result, 'comment') and result.comment:
            print(f"   Comment: {result.comment}")
        return False

def cleanup_test_orders(symbol):
    """Clean up any test orders"""
    print(f"\n🧹 Cleaning up test orders for {symbol}")

    orders = mt5.orders_get(symbol=symbol)
    if not orders:
        print("✅ No orders to clean up")
        return

    cleaned = 0
    for order in orders:
        if hasattr(order, 'comment') and 'FX-Ai Diagnostic Test' in str(order.comment):
            cancel_request = {
                'action': mt5.TRADE_ACTION_REMOVE,
                'order': order.ticket
            }
            cancel_result = mt5.order_send(cancel_request)
            if cancel_result and cancel_result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ Cancelled test order {order.ticket}")
                cleaned += 1
            else:
                print(f"❌ Failed to cancel order {order.ticket}")

    print(f"🧹 Cleaned up {cleaned} test orders")

def main():
    """Main diagnostic function"""
    print("🚀 MT5 Order Placement Diagnostic Test")
    print("=" * 50)

    # Test symbols
    test_symbols = ['EURUSD', 'GBPUSD', 'XAUUSD']

    try:
        # Test connection
        if not test_mt5_connection():
            return

        # Test each symbol
        for symbol in test_symbols:
            tick = test_symbol_info(symbol)
            if tick:
                success = test_order_placement(symbol, tick)
                if success:
                    print(f"✅ {symbol} order test PASSED")
                else:
                    print(f"❌ {symbol} order test FAILED")
            else:
                print(f"❌ {symbol} symbol test FAILED")

            # Clean up
            cleanup_test_orders(symbol)

        print("\n" + "=" * 50)
        print("🏁 Diagnostic test completed")

    finally:
        # Shutdown MT5
        mt5.shutdown()
        print("🔌 MT5 connection closed")

if __name__ == "__main__":
    main()