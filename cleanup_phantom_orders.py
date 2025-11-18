#!/usr/bin/env python3
"""
FX-Ai Phantom Order Cleanup Utility
Cleans up pending orders in MT5 that are not tracked in the system's database
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MetaTrader5 as mt5
from ai.learning_database import LearningDatabase
from core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class PhantomOrderCleaner:
    """Utility to clean up phantom pending orders"""

    def __init__(self, config):
        self.config = config
        self.db = LearningDatabase(config=config)
        self.magic_number = config.get('trading', {}).get('magic_number', 123456)

    def get_mt5_pending_orders(self):
        """Get all pending orders from MT5"""
        orders = mt5.orders_get()
        if orders is None:
            logger.error("Failed to get orders from MT5")
            return []

        # Filter to our system's orders by magic number
        our_orders = []
        for order in orders:
            if hasattr(order, 'magic') and order.magic == self.magic_number:
                our_orders.append(order)

        logger.info(f"Found {len(our_orders)} pending orders with our magic number")
        return our_orders

    def get_tracked_order_tickets(self):
        """Get all order tickets tracked in the database"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()

            # Get tickets from stop_orders table
            cursor.execute("SELECT ticket FROM stop_orders WHERE ticket IS NOT NULL")
            tracked_tickets = {row[0] for row in cursor.fetchall()}

            conn.close()
            logger.info(f"Found {len(tracked_tickets)} orders tracked in database")
            return tracked_tickets

        except Exception as e:
            logger.error(f"Error getting tracked tickets: {e}")
            return set()

    def cancel_phantom_orders(self, phantom_orders):
        """Cancel phantom orders that aren't tracked"""
        cancelled_count = 0
        failed_count = 0

        for order in phantom_orders:
            logger.info(f"Cancelling phantom order {order.ticket} for {order.symbol}")

            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order.ticket
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Successfully cancelled phantom order {order.ticket}")
                cancelled_count += 1
            else:
                error_msg = result.comment if result else "Unknown error"
                logger.error(f"❌ Failed to cancel phantom order {order.ticket}: {error_msg}")
                failed_count += 1

        return cancelled_count, failed_count

    def cleanup_phantom_orders(self):
        """Main cleanup function"""
        logger.info("🔍 Starting phantom order cleanup...")

        # Get orders from MT5
        mt5_orders = self.get_mt5_pending_orders()
        if not mt5_orders:
            logger.info("No pending orders found in MT5")
            return 0, 0

        # Get tracked tickets from database
        tracked_tickets = self.get_tracked_order_tickets()

        # Find phantom orders (in MT5 but not in database)
        phantom_orders = []
        for order in mt5_orders:
            if order.ticket not in tracked_tickets:
                phantom_orders.append(order)
                logger.warning(f"Phantom order detected: {order.ticket} ({order.symbol})")

        if not phantom_orders:
            logger.info("✅ No phantom orders found - all orders are properly tracked")
            return 0, 0

        logger.warning(f"Found {len(phantom_orders)} phantom orders to cancel")

        # Cancel phantom orders
        cancelled, failed = self.cancel_phantom_orders(phantom_orders)

        logger.info(f"Phantom order cleanup complete: {cancelled} cancelled, {failed} failed")
        return cancelled, failed

async def main():
    """Main function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize MT5
    if not mt5.initialize():
        logger.error("Failed to initialize MT5")
        return 1

    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.config

        # Create cleaner
        cleaner = PhantomOrderCleaner(config)

        # Run cleanup
        cancelled, failed = cleaner.cleanup_phantom_orders()

        print(f"\nCleanup Summary:")
        print(f"✅ Cancelled: {cancelled}")
        print(f"❌ Failed: {failed}")

        if failed > 0:
            return 1

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        mt5.shutdown()

    return 0

if __name__ == "__main__":
    import sqlite3  # Import here to avoid import issues
    exit_code = asyncio.run(main())
    sys.exit(exit_code)