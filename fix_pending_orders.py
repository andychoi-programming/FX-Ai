#!/usr/bin/env python3
"""
FX-Ai Pending Orders Diagnostic and Cleanup Tool
Helps identify and fix stale pending orders issues
"""

import MetaTrader5 as mt5
import logging
from datetime import datetime, timedelta
import json
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PendingOrderManager:
    """Manage and diagnose pending orders issues"""
    
    def __init__(self):
        """Initialize the manager"""
        self.stale_threshold_hours = 2  # Orders older than 2 hours are stale
        self.max_pending_per_symbol = 1  # Maximum pending orders per symbol
        
    def connect_mt5(self):
        """Connect to MT5"""
        if not mt5.initialize():
            logger.error("Failed to initialize MT5")
            return False
        logger.info("MT5 connection successful")
        return True
        
    def get_pending_orders(self):
        """Get all pending orders"""
        orders = mt5.orders_get()
        if orders is None:
            logger.warning("No pending orders found")
            return []
        
        logger.info(f"Found {len(orders)} pending orders")
        return orders
        
    def analyze_pending_orders(self):
        """Analyze pending orders for issues"""
        orders = self.get_pending_orders()
        if not orders:
            return
            
        # Analysis results
        stale_orders = []
        duplicate_symbols = {}
        order_summary = []
        
        current_time = datetime.now()
        
        for order in orders:
            # Get order details
            order_info = {
                'ticket': order.ticket,
                'symbol': order.symbol,
                'type': self._get_order_type_name(order.type),
                'volume': order.volume_initial,
                'price': order.price_open,
                'sl': order.sl,
                'tp': order.tp,
                'time_setup': datetime.fromtimestamp(order.time_setup),
                'comment': order.comment
            }
            
            # Calculate age
            order_age = current_time - order_info['time_setup']
            order_info['age_hours'] = order_age.total_seconds() / 3600
            
            # Check if stale
            if order_info['age_hours'] > self.stale_threshold_hours:
                stale_orders.append(order_info)
                
            # Track duplicates per symbol
            if order.symbol not in duplicate_symbols:
                duplicate_symbols[order.symbol] = []
            duplicate_symbols[order.symbol].append(order_info)
            
            order_summary.append(order_info)
            
        # Display analysis results
        self._display_analysis(order_summary, stale_orders, duplicate_symbols)
        
        return {
            'all_orders': order_summary,
            'stale_orders': stale_orders,
            'duplicate_symbols': duplicate_symbols
        }
        
    def _get_order_type_name(self, order_type):
        """Convert order type to readable name"""
        types = {
            mt5.ORDER_TYPE_BUY_LIMIT: "BUY_LIMIT",
            mt5.ORDER_TYPE_SELL_LIMIT: "SELL_LIMIT",
            mt5.ORDER_TYPE_BUY_STOP: "BUY_STOP",
            mt5.ORDER_TYPE_SELL_STOP: "SELL_STOP",
            mt5.ORDER_TYPE_BUY_STOP_LIMIT: "BUY_STOP_LIMIT",
            mt5.ORDER_TYPE_SELL_STOP_LIMIT: "SELL_STOP_LIMIT"
        }
        return types.get(order_type, f"UNKNOWN_{order_type}")
        
    def _display_analysis(self, all_orders, stale_orders, duplicate_symbols):
        """Display analysis results"""
        print("\n" + "="*80)
        print("PENDING ORDERS ANALYSIS REPORT")
        print("="*80)
        
        print(f"\nTotal Pending Orders: {len(all_orders)}")
        print(f"Stale Orders (>{self.stale_threshold_hours}h): {len(stale_orders)}")
        
        # Display symbols with multiple orders
        print("\n--- SYMBOLS WITH MULTIPLE PENDING ORDERS ---")
        issues_found = False
        for symbol, orders in duplicate_symbols.items():
            if len(orders) > self.max_pending_per_symbol:
                issues_found = True
                print(f"⚠️  {symbol}: {len(orders)} orders (MAX: {self.max_pending_per_symbol})")
                for order in orders:
                    print(f"    - Ticket {order['ticket']}: {order['type']} @ {order['price']:.5f} (age: {order['age_hours']:.1f}h)")
        
        if not issues_found:
            print("✅ No duplicate symbol issues found")
            
        # Display stale orders
        if stale_orders:
            print("\n--- STALE ORDERS ---")
            for order in stale_orders:
                print(f"❌ Ticket {order['ticket']} ({order['symbol']}): {order['age_hours']:.1f} hours old")
                print(f"   Type: {order['type']} | Volume: {order['volume']} | Price: {order['price']:.5f}")
        else:
            print("\n✅ No stale orders found")
            
        # Display all orders summary
        print("\n--- ALL PENDING ORDERS ---")
        for order in sorted(all_orders, key=lambda x: x['age_hours'], reverse=True):
            status = "⚠️ STALE" if order['age_hours'] > self.stale_threshold_hours else "✅"
            print(f"{status} {order['symbol']:8} | Ticket: {order['ticket']:8} | {order['type']:10} | Age: {order['age_hours']:5.1f}h | Vol: {order['volume']:.2f}")
            
    def cleanup_stale_orders(self, confirm=False):
        """Cancel stale pending orders"""
        analysis = self.analyze_pending_orders()
        if not analysis:
            return
            
        stale_orders = analysis['stale_orders']
        
        if not stale_orders:
            logger.info("No stale orders to clean up")
            return
            
        print(f"\n⚠️  Found {len(stale_orders)} stale orders to cancel")
        
        if not confirm:
            response = input("Do you want to cancel these orders? (yes/no): ").lower()
            if response != 'yes':
                print("Cleanup cancelled")
                return
                
        # Cancel stale orders
        cancelled_count = 0
        failed_count = 0
        
        for order in stale_orders:
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order['ticket']
            }
            
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Cancelled order {order['ticket']} ({order['symbol']})")
                cancelled_count += 1
            else:
                logger.error(f"❌ Failed to cancel order {order['ticket']}: {result.comment if result else 'Unknown error'}")
                failed_count += 1
                
        print(f"\nCleanup Complete: {cancelled_count} cancelled, {failed_count} failed")
        
    def cleanup_duplicate_orders(self, confirm=False):
        """Keep only the newest order per symbol"""
        analysis = self.analyze_pending_orders()
        if not analysis:
            return
            
        duplicate_symbols = analysis['duplicate_symbols']
        
        orders_to_cancel = []
        for symbol, orders in duplicate_symbols.items():
            if len(orders) > self.max_pending_per_symbol:
                # Sort by age (newest first)
                sorted_orders = sorted(orders, key=lambda x: x['age_hours'])
                # Keep the newest, cancel the rest
                orders_to_cancel.extend(sorted_orders[self.max_pending_per_symbol:])
                
        if not orders_to_cancel:
            logger.info("No duplicate orders to clean up")
            return
            
        print(f"\n⚠️  Found {len(orders_to_cancel)} duplicate orders to cancel")
        for order in orders_to_cancel:
            print(f"  - {order['symbol']}: Ticket {order['ticket']} (age: {order['age_hours']:.1f}h)")
            
        if not confirm:
            response = input("Do you want to cancel these duplicate orders? (yes/no): ").lower()
            if response != 'yes':
                print("Cleanup cancelled")
                return
                
        # Cancel duplicate orders
        cancelled_count = 0
        failed_count = 0
        
        for order in orders_to_cancel:
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order['ticket']
            }
            
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Cancelled duplicate order {order['ticket']} ({order['symbol']})")
                cancelled_count += 1
            else:
                logger.error(f"❌ Failed to cancel order {order['ticket']}: {result.comment if result else 'Unknown error'}")
                failed_count += 1
                
        print(f"\nDuplicate Cleanup Complete: {cancelled_count} cancelled, {failed_count} failed")
        
    def monitor_orders(self, interval_seconds=30):
        """Monitor pending orders continuously"""
        import time
        
        print(f"\n📊 Starting continuous monitoring (interval: {interval_seconds}s)")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking orders...")
                self.analyze_pending_orders()
                
                # Auto-cleanup if enabled
                if input("\nAuto-cleanup stale orders? (y/n): ").lower() == 'y':
                    self.cleanup_stale_orders(confirm=True)
                    self.cleanup_duplicate_orders(confirm=True)
                    
                print(f"\nNext check in {interval_seconds} seconds...")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped")
            
    def close_all_pending_orders(self, confirm=False):
        """Close all pending orders (emergency cleanup)"""
        orders = self.get_pending_orders()
        
        if not orders:
            logger.info("No pending orders to close")
            return
            
        print(f"\n⚠️  WARNING: This will cancel ALL {len(orders)} pending orders!")
        
        if not confirm:
            response = input("Are you SURE you want to cancel ALL pending orders? (yes/no): ").lower()
            if response != 'yes':
                print("Operation cancelled")
                return
                
        cancelled_count = 0
        failed_count = 0
        
        for order in orders:
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order.ticket
            }
            
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Cancelled order {order.ticket} ({order.symbol})")
                cancelled_count += 1
            else:
                logger.error(f"❌ Failed to cancel order {order.ticket}: {result.comment if result else 'Unknown error'}")
                failed_count += 1
                
        print(f"\nEmergency Cleanup Complete: {cancelled_count} cancelled, {failed_count} failed")

def main():
    """Main function with menu"""
    manager = PendingOrderManager()
    
    # Connect to MT5
    if not manager.connect_mt5():
        print("Failed to connect to MT5. Exiting.")
        return
        
    while True:
        print("\n" + "="*50)
        print("FX-AI PENDING ORDERS MANAGEMENT")
        print("="*50)
        print("1. Analyze pending orders")
        print("2. Cleanup stale orders (>2 hours)")
        print("3. Cleanup duplicate orders per symbol")
        print("4. Monitor orders continuously")
        print("5. EMERGENCY: Cancel ALL pending orders")
        print("6. Exit")
        print("-"*50)
        
        choice = input("Select option (1-6): ")
        
        if choice == '1':
            manager.analyze_pending_orders()
        elif choice == '2':
            manager.cleanup_stale_orders()
        elif choice == '3':
            manager.cleanup_duplicate_orders()
        elif choice == '4':
            manager.monitor_orders()
        elif choice == '5':
            manager.close_all_pending_orders()
        elif choice == '6':
            print("Exiting...")
            mt5.shutdown()
            break
        else:
            print("Invalid option")
            
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
