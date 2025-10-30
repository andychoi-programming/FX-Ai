#!/usr/bin/env python3
"""
EURJPY Stop Loss Bug Test Script - FIXED VERSION
Uses correct filling mode for the broker
"""

import MetaTrader5 as mt5
import time

def get_filling_mode(symbol):
    """Get the correct filling mode for a symbol"""
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"Failed to get symbol info for {symbol}")
        return mt5.ORDER_FILLING_FOK  # Default fallback

    filling = info.filling_mode
    print(f"Symbol {symbol} filling modes supported: {filling}")

    if filling & 1:  # ORDER_FILLING_FOK
        print("Using ORDER_FILLING_FOK")
        return mt5.ORDER_FILLING_FOK
    elif filling & 2:  # ORDER_FILLING_IOC
        print("Using ORDER_FILLING_IOC")
        return mt5.ORDER_FILLING_IOC
    elif filling & 4:  # ORDER_FILLING_RETURN
        print("Using ORDER_FILLING_RETURN")
        return mt5.ORDER_FILLING_RETURN
    else:
        print("No filling mode detected, using FOK as fallback")
        return mt5.ORDER_FILLING_FOK

def test_eurjpy_sl_fix():
    """Test EURJPY stop loss placement with correct filling mode"""

    print("🔍 Testing EURJPY Stop Loss with Correct Filling Mode")
    print("=" * 60)

    # Initialize MT5
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        return

    try:
        symbol = "EURJPY"

        # Get current market price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"❌ Failed to get tick data for {symbol}")
            return

        # Test parameters
        entry_price = tick.ask
        stop_loss_price = entry_price - 0.60  # 60 pips below (0.60 for JPY)
        take_profit_price = entry_price + 1.20  # 120 pips above
        lot_size = 0.01

        print(f"Market Data:")
        print(f"Current Ask: {tick.ask}")
        print(f"Current Bid: {tick.bid}")
        print(f"Spread: {tick.ask - tick.bid:.5f}")

        print(f"\nTest Parameters:")
        print(f"Symbol: {symbol}")
        print(f"Entry Price: {entry_price:.5f}")
        print(f"Stop Loss: {stop_loss_price:.5f}")
        print(f"Take Profit: {take_profit_price:.5f}")
        print(f"Lot Size: {lot_size}")

        # Calculate expected values
        sl_distance = abs(entry_price - stop_loss_price)
        pip_size = 0.01  # JPY pairs
        expected_pips = sl_distance / pip_size
        print(f"Expected SL Distance: {sl_distance:.5f}")
        print(f"Expected SL Pips: {expected_pips:.1f}")

        # Get correct filling mode
        filling_type = get_filling_mode(symbol)

        # Create test order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY,
            "price": entry_price,
            "sl": stop_loss_price,
            "tp": take_profit_price,
            "deviation": 20,
            "magic": 999999,
            "comment": "EURJPY_SL_TEST",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_type,  # Use correct filling mode
        }

        print(f"\nOrder Request:")
        for key, value in request.items():
            print(f"  {key}: {value}")

        # Send the order
        result = mt5.order_send(request)

        if result is None:
            print("❌ Order send failed - no response from MT5")
            return

        print(f"\nOrder Result:")
        print(f"Retcode: {result.retcode}")
        print(f"Comment: {result.comment}")

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print("✅ Order placed successfully!")

            # Wait and check what was actually set
            time.sleep(1)
            positions = mt5.positions_get(symbol=symbol)

            if positions:
                # Find our test position
                test_position = None
                for pos in positions:
                    if pos.comment == "EURJPY_SL_TEST":
                        test_position = pos
                        break

                if test_position:
                    actual_sl = test_position.sl
                    actual_tp = test_position.tp

                    print(f"\n📊 Position Verification:")
                    print(f"Position Ticket: {test_position.ticket}")
                    print(f"Entry Price: {test_position.price_open:.5f}")
                    print(f"Stop Loss: {actual_sl:.5f}")
                    print(f"Take Profit: {actual_tp:.5f}")

                    # Check SL accuracy
                    sl_mismatch = abs(actual_sl - stop_loss_price)
                    print(f"SL Mismatch: {sl_mismatch:.5f}")

                    if sl_mismatch < 0.01:
                        print("✅ SUCCESS: Stop loss set correctly!")
                        print("🎉 EURJPY stop loss bug is FIXED!")
                    else:
                        print("❌ FAILURE: Stop loss mismatch!")
                        print(f"Expected: {stop_loss_price:.5f}")
                        print(f"Got: {actual_sl:.5f}")

                        # Calculate what MT5 thinks
                        actual_pips = sl_mismatch / pip_size
                        print(f"Difference: ~{actual_pips:.1f} pips")

                    # Check TP accuracy
                    if take_profit_price:
                        tp_mismatch = abs(actual_tp - take_profit_price)
                        if tp_mismatch > 0.01:
                            print(f"⚠️  TP mismatch: Expected {take_profit_price:.5f}, Got {actual_tp:.5f}")

                    # Close the test position
                    print("\n🔄 Closing test position...")
                    close_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": lot_size,
                        "type": mt5.ORDER_TYPE_SELL,
                        "price": tick.bid,
                        "position": test_position.ticket,
                        "deviation": 20,
                    }

                    close_result = mt5.order_send(close_request)
                    if close_result and close_result.retcode == mt5.TRADE_RETCODE_DONE:
                        print("✅ Test position closed successfully")
                    else:
                        print("⚠️  Failed to close test position")

                else:
                    print("❌ Test position not found in positions list")
                    print("Available positions:")
                    for pos in positions:
                        print(f"  {pos.symbol} {pos.comment} SL:{pos.sl}")
            else:
                print("❌ No positions found after order placement")
        else:
            print(f"❌ Order failed: {result.comment}")
            print("This might indicate the filling mode issue is resolved but there's another problem")

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

    finally:
        mt5.shutdown()

if __name__ == "__main__":
    test_eurjpy_sl_fix()