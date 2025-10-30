#!/usr/bin/env python3
"""
EURJPY Stop Loss Bug Test Script
Run this to verify the EURJPY SL fix is working
"""

import MetaTrader5 as mt5
import time

def test_eurjpy_sl_fix():
    """Test EURJPY stop loss placement"""

    print("🔍 Testing EURJPY Stop Loss Fix")
    print("=" * 50)

    # Initialize MT5
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        return

    try:
        # Test parameters
        symbol = "EURJPY"
        entry_price = 178.250
        stop_loss_price = 177.650  # 60 pips away
        lot_size = 0.01

        print(f"Test Parameters:")
        print(f"Symbol: {symbol}")
        print(f"Entry: {entry_price}")
        print(f"Stop Loss: {stop_loss_price}")
        print(f"Lot Size: {lot_size}")

        # Calculate expected values
        sl_distance = abs(entry_price - stop_loss_price)
        pip_size = 0.01  # JPY pairs
        expected_pips = sl_distance / pip_size
        print(f"Expected SL Distance: {sl_distance:.5f}")
        print(f"Expected SL Pips: {expected_pips:.1f}")

        # Create test order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY,
            "price": entry_price,
            "sl": stop_loss_price,
            "tp": entry_price + 1.0,  # 100 pips TP
            "deviation": 20,
            "magic": 999999,
            "comment": "EURJPY_SL_TEST",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        print(f"\nSending request to MT5...")
        print(f"SL in request: {request['sl']}")

        # Send the order
        result = mt5.order_send(request)

        if result is None:
            print("❌ Order send failed - no response")
            return

        print(f"Order result: {result}")
        print(f"Retcode: {result.retcode}")

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print("✅ Order placed successfully")

            # Wait and check what was actually set
            time.sleep(1)
            positions = mt5.positions_get(symbol=symbol)

            if positions:
                position = None
                for pos in positions:
                    if pos.comment == "EURJPY_SL_TEST":
                        position = pos
                        break

                if position:
                    actual_sl = position.sl
                    print(f"\nActual position found:")
                    print(f"Entry Price: {position.price_open}")
                    print(f"Stop Loss: {actual_sl}")
                    print(f"Take Profit: {position.tp}")

                    # Check if SL matches
                    sl_mismatch = abs(actual_sl - stop_loss_price)
                    print(f"SL Mismatch: {sl_mismatch:.5f}")

                    if sl_mismatch < 0.01:
                        print("✅ SUCCESS: Stop loss set correctly!")
                    else:
                        print("❌ FAILURE: Stop loss mismatch!")
                        print(f"Expected: {stop_loss_price}")
                        print(f"Got: {actual_sl}")

                        # Calculate what MT5 thinks
                        actual_pips = sl_mismatch / pip_size
                        print(f"Difference: ~{actual_pips:.1f} pips")

                        # Close the test position
                        close_request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": symbol,
                            "volume": lot_size,
                            "type": mt5.ORDER_TYPE_SELL,
                            "price": mt5.symbol_info_tick(symbol).bid,
                            "position": position.ticket,
                        }
                        mt5.order_send(close_request)
                        print("Test position closed")
                else:
                    print("❌ Test position not found")
            else:
                print("❌ No positions found")
        else:
            print(f"❌ Order failed: {result.comment}")

    finally:
        mt5.shutdown()

if __name__ == "__main__":
    test_eurjpy_sl_fix()