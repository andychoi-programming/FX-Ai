"""
Emergency Stop Script for FX-Ai Trading System
Immediately closes all open positions and stops the trading system
"""
import MetaTrader5 as mt5
import sys
from datetime import datetime

def emergency_close_all():
    """Close all open positions immediately"""
    print("=" * 80)
    print("               FX-Ai EMERGENCY STOP")
    print("       IMMEDIATELY CLOSING ALL POSITIONS")
    print("=" * 80)
    print()
    
    # Initialize MT5
    if not mt5.initialize():  # type: ignore
        print(f"[ERROR] Failed to initialize MT5: {mt5.last_error()}")  # type: ignore
        return False
    
    try:
        # Get account info
        account_info = mt5.account_info()  # type: ignore
        if account_info:
            print(f"Account: {account_info.login}")
            print(f"Balance: ${account_info.balance:.2f}")
            print(f"Equity: ${account_info.equity:.2f}")
            print()
        
        # Get all open positions
        positions = mt5.positions_get()  # type: ignore
        
        if positions is None or len(positions) == 0:
            print("[INFO] No open positions found")
            return True
        
        print(f"[INFO] Found {len(positions)} open position(s)")
        print()
        
        # Close each position
        success_count = 0
        fail_count = 0
        
        for position in positions:
            symbol = position.symbol
            ticket = position.ticket
            volume = position.volume
            position_type = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
            
            print(f"Closing {position_type} {symbol} (Ticket: {ticket}, Volume: {volume})...")
            
            # Determine close type (opposite of position type)
            if position.type == mt5.ORDER_TYPE_BUY:
                close_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(symbol).bid  # type: ignore
            else:
                close_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(symbol).ask  # type: ignore
            
            # Try different filling modes (brokers have different requirements)
            filling_modes = [
                mt5.ORDER_FILLING_FOK,  # Fill or Kill
                mt5.ORDER_FILLING_IOC,  # Immediate or Cancel
                mt5.ORDER_FILLING_RETURN,  # Return
            ]
            
            closed = False
            for filling_mode in filling_modes:
                # Create close request
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": volume,
                    "type": close_type,
                    "position": ticket,
                    "price": price,
                    "deviation": 20,
                    "magic": 0,
                    "comment": "EMERGENCY_STOP",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": filling_mode,
                }
                
                # Send close order
                result = mt5.order_send(request)  # type: ignore
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"  ✓ Successfully closed (P/L: ${position.profit:.2f})")
                    success_count += 1
                    closed = True
                    break
                elif result.retcode != 10030:  # Not "unsupported filling mode"
                    print(f"  ✗ Failed to close: {result.comment} (Code: {result.retcode})")
                    fail_count += 1
                    closed = True
                    break
            
            if not closed:
                print(f"  ✗ Failed to close: All filling modes rejected")
                fail_count += 1
        
        print()
        print("=" * 80)
        print(f"EMERGENCY STOP COMPLETE")
        print(f"Closed: {success_count}, Failed: {fail_count}")
        print("=" * 80)
        
        return fail_count == 0
        
    except Exception as e:
        print(f"[ERROR] Exception during emergency stop: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        mt5.shutdown()  # type: ignore

if __name__ == "__main__":
    print()
    print(f"Emergency Stop initiated at {datetime.now()}")
    print()
    
    success = emergency_close_all()
    
    print()
    if success:
        print("[SUCCESS] All positions closed successfully")
        sys.exit(0)
    else:
        print("[WARNING] Some positions may not have closed. Check MT5 manually.")
        sys.exit(1)
