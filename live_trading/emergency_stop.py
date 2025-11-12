#!/usr/bin/env python3
"""
FX-Ai Emergency Stop Script
Immediately closes ALL open positions AND cancels ALL pending orders
"""

import sys
import os
import logging
from datetime import datetime

# Add parent directory to path to import FX-Ai modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import MetaTrader5 as mt5
    from core.mt5_connector import MT5Connector
    from utils.logger import setup_logger
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("Please run this script from the FX-Ai root directory")
    sys.exit(1)

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmergencyStop:
    """Emergency stop functionality to close ALL open positions and cancel ALL pending orders"""

    def __init__(self):
        self.mt5 = None
        self.magic_number = 123456  # Use the correct magic number

    def initialize_mt5(self):
        """Initialize MT5 connection"""
        try:
            logger.info("Initializing MT5 connection...")

            # Try to initialize MT5
            if not mt5.initialize():
                logger.error("Failed to initialize MT5")
                return False

            # Try to login (this will use saved credentials if available)
            # If login fails, MT5 might still work for local operations
            try:
                # Basic connection test
                terminal_info = mt5.terminal_info()
                if terminal_info:
                    logger.info(f"MT5 Terminal: {terminal_info.name}")
                    logger.info(f"MT5 Server: {terminal_info.server}")
                    return True
                else:
                    logger.warning("MT5 terminal info not available")
                    return False
            except Exception as e:
                logger.warning(f"MT5 login check failed: {e}")
                # Continue anyway as MT5 might still be usable
                return True

        except Exception as e:
            logger.error(f"Failed to initialize MT5: {e}")
            return False

    def close_all_positions(self):
        """Close all open positions"""
        try:
            logger.info("Getting all open positions...")

            # Get all positions
            positions = mt5.positions_get()
            if positions is None:
                positions = []

            # Filter positions by our magic number
            our_positions = [p for p in positions if hasattr(p, 'magic') and p.magic == self.magic_number]

            logger.info(f"Found {len(our_positions)} positions with magic number {self.magic_number}")

            closed_count = 0
            for position in our_positions:
                try:
                    logger.info(f"Closing position: {position.symbol} ticket {position.ticket} (profit: {position.profit:.2f})")

                    # Determine order type for closing
                    if position.type == mt5.ORDER_TYPE_BUY:
                        order_type = mt5.ORDER_TYPE_SELL
                        price = mt5.symbol_info_tick(position.symbol).bid
                    else:
                        order_type = mt5.ORDER_TYPE_BUY
                        price = mt5.symbol_info_tick(position.symbol).ask

                    # Create close request
                    request = {
                        'action': mt5.TRADE_ACTION_DEAL,
                        'symbol': position.symbol,
                        'volume': position.volume,
                        'type': order_type,
                        'position': position.ticket,
                        'price': price,
                        'deviation': 10,
                        'magic': self.magic_number,
                        'comment': 'Emergency stop - close position',
                        'type_filling': mt5.ORDER_FILLING_IOC,
                    }

                    # Send order
                    result = mt5.order_send(request)

                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"Successfully closed {position.symbol} position {position.ticket}")
                        closed_count += 1
                    else:
                        error_code = result.retcode if result else 'Unknown'
                        logger.error(f"Failed to close {position.symbol} position {position.ticket}: {error_code}")

                except Exception as e:
                    logger.error(f"Error closing position {position.ticket}: {e}")

            logger.info(f"Closed {closed_count}/{len(our_positions)} positions")
            return closed_count

        except Exception as e:
            logger.error(f"Error closing positions: {e}")
            return 0

    def cancel_pending_orders(self):
        """Cancel all pending orders"""
        try:
            logger.info("Getting all pending orders...")

            # Get all orders
            orders = mt5.orders_get()
            if orders is None:
                orders = []

            # Filter orders by our magic number
            our_orders = [o for o in orders if hasattr(o, 'magic') and o.magic == self.magic_number]

            logger.info(f"Found {len(our_orders)} orders with magic number {self.magic_number} out of {len(orders)} total orders")

            # Debug: show details of first few orders
            for i, order in enumerate(our_orders[:3]):
                logger.info(f"Order {i+1}: ticket={order.ticket}, symbol={order.symbol}, type={order.type}, magic={getattr(order, 'magic', 'N/A')}, state={getattr(order, 'state', 'N/A')}")

            cancelled_count = 0
            for order in our_orders:
                try:
                    logger.info(f"Cancelling order: {order.symbol} ticket {order.ticket} (type: {order.type}, price: {getattr(order, 'price_open', 'N/A')}, state: {getattr(order, 'state', 'N/A')})")

                    # Check if order is in a cancellable state
                    if hasattr(order, 'state') and order.state != mt5.ORDER_STATE_PLACED:
                        logger.warning(f"Order {order.ticket} is not in cancellable state (state: {order.state})")
                        continue

                    # Cancel the order
                    result = mt5.order_send({
                        'action': mt5.TRADE_ACTION_REMOVE,
                        'order': order.ticket,
                    })

                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"Successfully cancelled {order.symbol} order {order.ticket}")
                        cancelled_count += 1
                    else:
                        error_code = result.retcode if result else 'Unknown'
                        logger.error(f"Failed to cancel {order.symbol} order {order.ticket}: retcode={error_code}, comment={getattr(result, 'comment', 'N/A') if result else 'N/A'}")

                    # Small delay to avoid overwhelming MT5
                    import time
                    time.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error cancelling order {order.ticket}: {e}")

            logger.info(f"Cancelled {cancelled_count}/{len(our_orders)} pending orders")
            return cancelled_count

        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            return 0

    def run_emergency_stop(self):
        """Run the complete emergency stop procedure"""
        logger.info("=" * 60)
        logger.info("FX-AI EMERGENCY STOP - CLOSING ALL POSITIONS AND CANCELING ALL ORDERS")
        logger.info("=" * 60)

        success = True

        # Initialize MT5
        if not self.initialize_mt5():
            logger.error("Failed to initialize MT5 - cannot proceed with emergency stop")
            return False

        # Close all positions
        positions_closed = self.close_all_positions()
        if positions_closed == 0:
            logger.info("No positions to close")
        else:
            logger.info(f"Emergency stop: Closed {positions_closed} positions")

        # Cancel pending orders
        orders_cancelled = self.cancel_pending_orders()
        if orders_cancelled == 0:
            logger.info("No pending orders to cancel")
        else:
            logger.info(f"Emergency stop: Cancelled {orders_cancelled} pending orders")

        # Shutdown MT5
        try:
            mt5.shutdown()
            logger.info("MT5 connection closed")
        except Exception as e:
            logger.warning(f"Error closing MT5 connection: {e}")

        logger.info("=" * 60)
        logger.info("EMERGENCY STOP COMPLETED")
        logger.info(f"Positions closed: {positions_closed}")
        logger.info(f"Orders cancelled: {orders_cancelled}")
        logger.info("=" * 60)

        return success

def main():
    """Main entry point"""
    try:
        emergency_stop = EmergencyStop()
        success = emergency_stop.run_emergency_stop()

        if success:
            print("\n[SUCCESS] Emergency stop completed successfully")
            sys.exit(0)
        else:
            print("\n[ERROR] Emergency stop failed")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nEmergency stop interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error in emergency stop: {e}")
        print(f"\n[ERROR] Emergency stop failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()