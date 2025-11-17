"""
Test script to verify order execution fix
"""

import MetaTrader5 as mt5
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MT5
if not mt5.initialize():
    print("[FAIL] Failed to initialize MT5")
    quit()

# Test with a safe order (far from current price)
symbol = "EURUSD"
current_price = mt5.symbol_info_tick(symbol).ask
test_price = current_price + 0.0500  # 500 pips away (won't trigger)

request = {
    "action": mt5.TRADE_ACTION_PENDING,
    "symbol": symbol,
    "volume": 0.01,  # Minimum size
    "type": mt5.ORDER_TYPE_BUY_STOP,
    "price": round(test_price, 5),
    "sl": round(test_price - 0.0100, 5),
    "tp": round(test_price + 0.0100, 5),
    "deviation": 20,
    "magic": 999999,
    "comment": "TEST ORDER - SAFE TO DELETE",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}

print(f"\n[EMOJI] Sending test order...")
print(f"   Symbol: {symbol}")
print(f"   Price: {test_price}")
print(f"   Current: {current_price}")

result = mt5.order_send(request)

print(f"\n[CHART] RESULT:")
print(f"   retcode: {result.retcode}")
print(f"   comment: {result.comment}")
print(f"   order: {result.order}")

# Check success codes
SUCCESS_CODES = [10008, 10009, 10010]

if result.retcode in SUCCESS_CODES:
    print(f"\n[PASS] SUCCESS! Order placed with ticket {result.order}")
    print(f"   This is the CORRECT interpretation!")
    
    # Clean up - cancel the test order
    cancel_request = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order": result.order
    }
    cancel_result = mt5.order_send(cancel_request)
    print(f"\n[EMOJI] Test order cancelled (ticket {result.order})")
else:
    print(f"\n[FAIL] FAILURE! Order rejected: {result.comment}")

mt5.shutdown()