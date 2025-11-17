"""
FX-Ai Pending Orders Manager
Analyzes and optionally clears pending orders from MT5
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, r'C:\Users\andyc\python\FX-Ai')

import MetaTrader5 as mt5
from utils.config_loader import ConfigLoader

def initialize_mt5():
    """Initialize MT5 connection"""
    config = ConfigLoader()
    mt5_config = config.get('mt5', {})
    
    if not mt5.initialize():
        print(f"✗ MT5 initialization failed: {mt5.last_error()}")
        return False
    
    # Login
    login = mt5_config.get('login')
    password = mt5_config.get('password')
    server = mt5_config.get('server')
    
    if not mt5.login(login, password, server):
        print(f"✗ MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False
    
    print(f"✓ Connected to MT5")
    print(f"  Account: {login}")
    print(f"  Server: {server}")
    
    return True

def analyze_pending_orders():
    """Analyze all pending orders"""
    
    print("\n" + "="*80)
    print("PENDING ORDERS ANALYSIS")
    print("="*80)
    
    # Get all pending orders
    orders = mt5.orders_get()
    
    if orders is None or len(orders) == 0:
        print("\n✓ No pending orders found")
        return []
    
    print(f"\n✗ Found {len(orders)} pending orders:\n")
    
    # Group by symbol
    by_symbol = {}
    for order in orders:
        symbol = order.symbol
        if symbol not in by_symbol:
            by_symbol[symbol] = []
        by_symbol[symbol].append(order)
    
    # Display by symbol
    print(f"{'Symbol':8s} | {'Ticket':>10s} | {'Type':>10s} | {'Volume':>6s} | {'Entry':>10s} | {'SL':>10s} | {'TP':>10s} | Age")
    print("-" * 110)
    
    now = datetime.now()
    
    for symbol in sorted(by_symbol.keys()):
        symbol_orders = by_symbol[symbol]
        
        for order in symbol_orders:
            order_type = "BUY_STOP" if order.type == mt5.ORDER_TYPE_BUY_STOP else "SELL_STOP"
            setup_time = datetime.fromtimestamp(order.time_setup)
            age = now - setup_time
            age_str = f"{age.seconds // 3600}h {(age.seconds % 3600) // 60}m"
            
            print(f"{symbol:8s} | {order.ticket:>10d} | {order_type:>10s} | {order.volume:>6.2f} | "
                  f"{order.price_open:>10.5f} | {order.sl:>10.5f} | {order.tp:>10.5f} | {age_str}")
    
    print("\n" + "="*80)
    print("SUMMARY BY SYMBOL")
    print("="*80)
    
    for symbol, symbol_orders in sorted(by_symbol.items()):
        print(f"{symbol:8s}: {len(symbol_orders)} pending order(s)")
    
    return orders

def clear_all_pending_orders(orders):
    """Clear all pending orders"""
    
    print("\n" + "="*80)
    print("CLEARING ALL PENDING ORDERS")
    print("="*80)
    
    if not orders:
        print("\n✓ No orders to clear")
        return
    
    print(f"\nAttempting to cancel {len(orders)} pending orders...")
    
    success_count = 0
    fail_count = 0
    
    for order in orders:
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": order.ticket,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✓ Cancelled {order.symbol} ticket {order.ticket}")
            success_count += 1
        else:
            print(f"✗ Failed to cancel {order.symbol} ticket {order.ticket}: {result.comment}")
            fail_count += 1
    
    print("\n" + "="*80)
    print(f"✓ Successfully cancelled: {success_count}")
    print(f"✗ Failed to cancel: {fail_count}")
    print("="*80)

def clear_symbol_orders(symbol, orders):
    """Clear pending orders for a specific symbol"""
    
    symbol_orders = [o for o in orders if o.symbol == symbol]
    
    if not symbol_orders:
        print(f"\n✓ No pending orders for {symbol}")
        return
    
    print(f"\n" + "="*80)
    print(f"CLEARING PENDING ORDERS FOR {symbol}")
    print("="*80)
    
    print(f"\nAttempting to cancel {len(symbol_orders)} order(s) for {symbol}...")
    
    success_count = 0
    fail_count = 0
    
    for order in symbol_orders:
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": order.ticket,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✓ Cancelled ticket {order.ticket}")
            success_count += 1
        else:
            print(f"✗ Failed to cancel ticket {order.ticket}: {result.comment}")
            fail_count += 1
    
    print(f"\n✓ Successfully cancelled: {success_count}")
    print(f"✗ Failed to cancel: {fail_count}")

def main():
    """Main function"""
    
    print("\n" + "="*80)
    print("FX-Ai Pending Orders Manager")
    print("="*80)
    
    # Initialize MT5
    if not initialize_mt5():
        return
    
    try:
        # Analyze pending orders
        orders = analyze_pending_orders()
        
        if not orders:
            print("\n✓ No action needed - no pending orders found")
            return
        
        # Ask user what to do
        print("\n" + "="*80)
        print("OPTIONS")
        print("="*80)
        print("1. Clear ALL pending orders")
        print("2. Clear orders for specific symbol")
        print("3. Exit (no changes)")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            confirm = input(f"\n⚠️  Are you sure you want to cancel ALL {len(orders)} pending orders? (yes/no): ").strip().lower()
            if confirm == 'yes':
                clear_all_pending_orders(orders)
            else:
                print("✗ Cancelled - no orders removed")
        
        elif choice == '2':
            symbol = input("\nEnter symbol (e.g., USDJPY): ").strip().upper()
            clear_symbol_orders(symbol, orders)
        
        elif choice == '3':
            print("\n✓ Exiting - no changes made")
        
        else:
            print("\n✗ Invalid choice - no changes made")
        
        # Show final state
        print("\n" + "="*80)
        print("FINAL STATE")
        print("="*80)
        
        final_orders = mt5.orders_get()
        if final_orders:
            print(f"\n✗ {len(final_orders)} pending orders remaining")
        else:
            print(f"\n✓ All pending orders cleared")
        
    finally:
        mt5.shutdown()
        print("\n✓ MT5 connection closed")

if __name__ == "__main__":
    main()
