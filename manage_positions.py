#!/usr/bin/env python3
"""
Manual SL/TP Management Script for Existing Positions
Use this to add stop loss and take profit to positions that don't have them.
"""

import MetaTrader5 as mt5
from datetime import datetime

def add_sl_tp_to_positions():
    """Add stop loss and take profit to existing positions"""
    print("=" * 60)
    print("Manual SL/TP Management for Existing Positions")
    print("=" * 60)

    # Initialize MT5
    if not mt5.initialize():
        print("[FAIL] MT5 initialization failed")
        return False

    try:
        # Get all positions
        positions = mt5.positions_get()
        if not positions:
            print("ℹ[EMOJI]  No positions found")
            return True

        print(f"Found {len(positions)} position(s):")
        print("-" * 60)

        for pos in positions:
            symbol = pos.symbol
            direction = 'LONG' if pos.type == 0 else 'SHORT'
            entry = pos.price_open
            current = pos.price_current
            profit = pos.profit
            volume = pos.volume

            print(f"\n{symbol} {direction}")
            print(f"  Entry: {entry:.5f}")
            print(f"  Current: {current:.5f}")
            print(f"  Volume: {volume:.2f} lots")
            print(f"  P&L: ${profit:.2f}")
            print(f"  Current SL: {pos.sl if pos.sl > 0 else 'None'}")
            print(f"  Current TP: {pos.tp if pos.tp > 0 else 'None'}")

            # Skip if already has SL/TP
            if pos.sl > 0 and pos.tp > 0:
                print("  [PASS] Already has SL/TP - skipping")
                continue

            # Calculate suggested SL/TP
            if symbol == 'XAGUSD':
                if direction == 'SHORT':
                    # Current: ~50.659, Entry: 50.758
                    # Suggest SL above entry, TP below for profit target
                    suggested_sl = 50.900  # Above entry to limit loss
                    suggested_tp = 50.300  # Target ~$100 profit
                else:
                    suggested_sl = entry * 0.98  # 2% below
                    suggested_tp = entry * 1.02  # 2% above

            elif symbol == 'XAUUSD':
                if direction == 'SHORT':
                    # Current: ~4083.11, Entry: 4094.49
                    # Suggest SL above entry, TP below for profit target
                    suggested_sl = 4110.00  # Above entry to limit loss
                    suggested_tp = 4050.00  # Target additional profit
                else:
                    suggested_sl = entry * 0.98  # 2% below
                    suggested_tp = entry * 1.02  # 2% above
            else:
                # Generic calculation for other symbols
                pip_size = 0.0001 if len(str(int(entry))) <= 4 else 0.00001  # Rough pip size
                sl_pips = 50 * pip_size  # 50 pip stop
                tp_pips = 100 * pip_size  # 100 pip target

                if direction == 'LONG':
                    suggested_sl = entry - sl_pips
                    suggested_tp = entry + tp_pips
                else:
                    suggested_sl = entry + sl_pips
                    suggested_tp = entry - tp_pips

            print(f"  Suggested SL: {suggested_sl:.5f}")
            print(f"  Suggested TP: {suggested_tp:.5f}")

            # Ask user if they want to apply
            while True:
                response = input(f"  Apply suggested SL/TP to {symbol}? (y/n/custom): ").lower().strip()

                if response == 'n':
                    print("  Skipping this position")
                    break
                elif response == 'y':
                    # Apply the suggested values
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": pos.ticket,
                        "sl": suggested_sl,
                        "tp": suggested_tp
                    }

                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"  [PASS] SL/TP applied successfully")
                        print(f"     SL: {suggested_sl:.5f}, TP: {suggested_tp:.5f}")
                    else:
                        print(f"  [FAIL] Failed to apply SL/TP: {result.comment}")
                    break
                elif response == 'custom':
                    try:
                        custom_sl = float(input("  Enter custom SL: "))
                        custom_tp = float(input("  Enter custom TP: "))

                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": pos.ticket,
                            "sl": custom_sl,
                            "tp": custom_tp
                        }

                        result = mt5.order_send(request)
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            print(f"  [PASS] Custom SL/TP applied successfully")
                            print(f"     SL: {custom_sl:.5f}, TP: {custom_tp:.5f}")
                        else:
                            print(f"  [FAIL] Failed to apply custom SL/TP: {result.comment}")
                    except ValueError:
                        print("  [FAIL] Invalid number format")
                        continue
                    break
                else:
                    print("  Please enter 'y', 'n', or 'custom'")

        print("\n" + "=" * 60)
        print("SL/TP Management Complete")
        print("=" * 60)

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False
    finally:
        mt5.shutdown()

    return True

def close_positions_safely():
    """Option to close positions and secure profits"""
    print("=" * 60)
    print("Safe Position Closure")
    print("=" * 60)

    # Initialize MT5
    if not mt5.initialize():
        print("[FAIL] MT5 initialization failed")
        return False

    try:
        positions = mt5.positions_get()
        if not positions:
            print("ℹ[EMOJI]  No positions to close")
            return True

        total_profit = sum(pos.profit for pos in positions)
        print(f"Total unrealized profit: ${total_profit:.2f}")

        for pos in positions:
            symbol = pos.symbol
            direction = 'LONG' if pos.type == 0 else 'SHORT'
            profit = pos.profit

            print(f"\n{symbol} {direction} - P&L: ${profit:.2f}")

            response = input(f"Close {symbol} position? (y/n): ").lower().strip()
            if response == 'y':
                # Close position at market
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket,
                    "price": mt5.symbol_info_tick(symbol).bid if pos.type == 0 else mt5.symbol_info_tick(symbol).ask,
                    "deviation": 10,
                    "comment": "Manual close - secure profits"
                }

                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"[PASS] {symbol} closed successfully - Realized P&L: ${profit:.2f}")
                else:
                    print(f"[FAIL] Failed to close {symbol}: {result.comment}")

        print("\n" + "=" * 60)
        print("Position closure complete")
        print("=" * 60)

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False
    finally:
        mt5.shutdown()

    return True

if __name__ == "__main__":
    print("FX-Ai Manual SL/TP Management")
    print("Choose an option:")
    print("1. Add SL/TP to existing positions")
    print("2. Close positions safely")
    print("3. Exit")

    while True:
        try:
            choice = input("Enter choice (1-3): ").strip()
            if choice == '1':
                add_sl_tp_to_positions()
                break
            elif choice == '2':
                close_positions_safely()
                break
            elif choice == '3':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break