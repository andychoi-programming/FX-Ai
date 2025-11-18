"""
Stale Order Cleanup Utility
Cleans up pending orders that are older than specified threshold
"""

import logging
import MetaTrader5 as mt5
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def cleanup_stale_orders(hours_threshold: float = 1.0, magic_number: int = None) -> int:
    """
    Cancel pending orders older than specified hours

    Args:
        hours_threshold: Cancel orders older than this many hours
        magic_number: Only clean orders with this magic number (None for all)

    Returns:
        Number of orders cancelled
    """
    try:
        # Get all pending orders
        orders = mt5.orders_get()
        if orders is None:
            logger.warning("Failed to get orders from MT5")
            return 0

        cancelled_count = 0
        current_time = datetime.now()

        for order in orders:
            # Skip if magic number filter is specified
            if magic_number is not None and order.magic != magic_number:
                continue

            # Calculate order age in hours
            order_time = datetime.fromtimestamp(order.time_setup)
            age_hours = (current_time - order_time).total_seconds() / 3600

            if age_hours > hours_threshold:
                # Cancel the stale order
                cancel_result = mt5.order_delete(order.ticket)
                if cancel_result is not None and cancel_result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Cancelled stale order {order.ticket} for {order.symbol} "
                              f"(age: {age_hours:.1f}h, type: {order.type}, price: {order.price_open})")
                    cancelled_count += 1
                else:
                    logger.error(f"Failed to cancel stale order {order.ticket}: {cancel_result}")

        if cancelled_count > 0:
            logger.info(f"Successfully cancelled {cancelled_count} stale orders")
        else:
            logger.info("No stale orders found to cancel")

        return cancelled_count

    except Exception as e:
        logger.error(f"Error during stale order cleanup: {e}")
        return 0


def cleanup_stale_orders_by_symbol(symbol: str, hours_threshold: float = 1.0, magic_number: int = None) -> int:
    """
    Cancel pending orders for a specific symbol that are older than specified hours

    Args:
        symbol: Symbol to clean orders for
        hours_threshold: Cancel orders older than this many hours
        magic_number: Only clean orders with this magic number (None for all)

    Returns:
        Number of orders cancelled
    """
    try:
        # Get orders for specific symbol
        orders = mt5.orders_get(symbol=symbol)
        if orders is None:
            logger.warning(f"Failed to get orders for {symbol}")
            return 0

        cancelled_count = 0
        current_time = datetime.now()

        for order in orders:
            # Skip if magic number filter is specified
            if magic_number is not None and order.magic != magic_number:
                continue

            # Calculate order age in hours
            order_time = datetime.fromtimestamp(order.time_setup)
            age_hours = (current_time - order_time).total_seconds() / 3600

            if age_hours > hours_threshold:
                # Cancel the stale order
                cancel_result = mt5.order_delete(order.ticket)
                if cancel_result is not None and cancel_result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Cancelled stale order {order.ticket} for {symbol} "
                              f"(age: {age_hours:.1f}h, type: {order.type}, price: {order.price_open})")
                    cancelled_count += 1
                else:
                    logger.error(f"Failed to cancel stale order {order.ticket}: {cancel_result}")

        return cancelled_count

    except Exception as e:
        logger.error(f"Error during stale order cleanup for {symbol}: {e}")
        return 0


if __name__ == "__main__":
    # Test the cleanup function
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Initialize MT5 (you would need to set this up properly)
    if not mt5.initialize():
        print("Failed to initialize MT5")
        sys.exit(1)

    # Clean orders older than 1.4 hours (matching the issue description)
    cancelled = cleanup_stale_orders(hours_threshold=1.4)
    print(f"Cancelled {cancelled} stale orders")

    mt5.shutdown()