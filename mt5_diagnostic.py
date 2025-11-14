"""Diagnose MT5 connection issues"""

import MetaTrader5 as mt5
import time
from datetime import datetime

print("🔍 MT5 CONNECTION DIAGNOSTIC")
print("=" * 50)

# Initialize MT5
print("1. Testing MT5 initialization...")
result = mt5.initialize()
print(f"   MT5 initialize result: {result}")

if not result:
    print("   ❌ MT5 initialization FAILED")
    print(f"   Error: {mt5.last_error()}")
    exit(1)

print("   ✅ MT5 initialized successfully")

# Check terminal info
print("\n2. Checking terminal info...")
terminal_info = mt5.terminal_info()
if terminal_info is None:
    print("   ❌ Cannot get terminal info")
    print(f"   Error: {mt5.last_error()}")
else:
    print(f"   ✅ Terminal connected: {terminal_info.connected}")
    print(f"   Trade allowed: {terminal_info.trade_allowed}")
    print(f"   Community account: {terminal_info.community_account}")
    print(f"   Name: {terminal_info.name}")
    print(f"   Company: {terminal_info.company}")

# Check account info
print("\n3. Checking account info...")
account_info = mt5.account_info()
if account_info is None:
    print("   ❌ Cannot get account info")
    print(f"   Error: {mt5.last_error()}")
else:
    print(f"   ✅ Account balance: ${account_info.balance:.2f}")
    print(f"   Account login: {account_info.login}")
    print(f"   Server: {account_info.server}")

# Test symbol info
print("\n4. Testing symbol access...")
symbols_to_test = ["EURUSD", "GBPUSD", "USDJPY"]
for symbol in symbols_to_test:
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"   ❌ {symbol}: No symbol info")
    else:
        print(f"   ✅ {symbol}: Available, spread: {symbol_info.spread}")

# Test tick data
print("\n5. Testing real-time tick data...")
for symbol in symbols_to_test[:1]:  # Test just one
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"   ❌ {symbol}: Cannot get tick data")
        print(f"   Error: {mt5.last_error()}")
    else:
        tick_time = datetime.fromtimestamp(tick.time)
        print(f"   ✅ {symbol}: Tick received at {tick_time}")
        print(f"      Bid: {tick.bid}, Ask: {tick.ask}")

# Test server time synchronization
print("\n6. Testing server time...")
server_time = None
for attempt in range(3):
    tick = mt5.symbol_info_tick("EURUSD")
    if tick:
        server_time = datetime.fromtimestamp(tick.time)
        break
    time.sleep(1)

if server_time:
    print(f"   ✅ Server time: {server_time}")
    print(f"   Local time: {datetime.now()}")
    time_diff = abs((datetime.now() - server_time).total_seconds())
    print(f"   Time difference: {time_diff:.1f} seconds")
    if time_diff > 60:
        print("   ⚠️  Large time difference detected!")
else:
    print("   ❌ Cannot get server time")

print("\n7. Testing position synchronization...")
positions = mt5.positions_get()
if positions is None:
    print("   ❌ Cannot get positions")
    print(f"   Error: {mt5.last_error()}")
else:
    print(f"   ✅ Positions retrieved: {len(positions)} open positions")

print("\n8. Testing order operations...")
# Just test if we can get order history (don't place orders)
history = mt5.history_orders_get(datetime.now().replace(hour=0, minute=0, second=0), datetime.now())
if history is None:
    print("   ❌ Cannot get order history")
    print(f"   Error: {mt5.last_error()}")
else:
    print(f"   ✅ Order history accessible: {len(history)} orders today")

print("\n" + "=" * 50)
print("DIAGNOSTIC COMPLETE")

# Shutdown
mt5.shutdown()
print("MT5 connection closed")