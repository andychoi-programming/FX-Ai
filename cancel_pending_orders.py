#!/usr/bin/env python3
"""
Cancel Pending Orders Script
Cancels all pending orders for the FX-Ai system
"""

import sys
import os
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import MetaTrader5 as mt5
    from core.mt5_connector import MT5Connector
    from utils.logger import setup_logger
    from utils.config_loader import ConfigLoader
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cancel_pending_orders():
    """Cancel all pending orders for our magic number"""
    try:
        # Get all orders
        orders = mt5.orders_get()
        if orders is None:
            logger.error("Failed to get orders")
            return 0

        # Filter for our magic number
        magic_number = 123456
        our_orders = [order for order in orders if getattr(order, 'magic', 0) == magic_number]

        if not our_orders:
            logger.info("No pending orders found for our system")
            return 0

        logger.info(f"Found {len(our_orders)} pending orders to cancel")

        cancelled_count = 0
        for order in our_orders:
            try:
                # Cancel the order
                result = mt5.order_send({
                    'action': mt5.TRADE_ACTION_REMOVE,
                    'order': order.ticket,
                })
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Successfully cancelled order {order.ticket} ({order.symbol})")
                    cancelled_count += 1
                else:
                    error_code = result.retcode if result else 'Unknown'
                    logger.error(f"Failed to cancel order {order.ticket}: retcode={error_code}")
            except Exception as e:
                logger.error(f"Error cancelling order {order.ticket}: {e}")

        logger.info(f"Cancelled {cancelled_count}/{len(our_orders)} pending orders")
        return cancelled_count

    except Exception as e:
        logger.error(f"Error in cancel_pending_orders: {e}")
        return 0

def main():
    logger.info("Initializing MT5 connection...")

    # Initialize MT5 directly
    if not mt5.initialize():
        logger.error("Failed to initialize MT5")
        return

    # Check if logged in
    account_info = mt5.account_info()
    if account_info is None:
        logger.error("No MT5 account logged in")
        return

    logger.info(f"Connected to account: {account_info.login}")

    # Cancel pending orders
    cancelled = cancel_pending_orders()

    logger.info(f"Operation complete. Cancelled {cancelled} pending orders.")

    # Shutdown MT5
    mt5.shutdown()
    logger.info("MT5 connection closed")

if __name__ == "__main__":
    main()