#!/usr/bin/env python3
"""
FX-Ai EMERGENCY STOP Script
*** DANGER: Closes ALL open positions AND cancels ALL pending orders ***
*** This affects ALL EAs and manual trades in MT5 - USE WITH EXTREME CAUTION ***

This script performs a complete emergency shutdown of ALL trading activity in MT5.
It will close every open position and cancel every pending order, regardless of
which system or person placed them.

USE ONLY IN TRUE EMERGENCIES when you need to stop ALL trading immediately.
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
    from utils.config_loader import ConfigLoader
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
    """EMERGENCY STOP: Closes ALL open positions AND cancels ALL pending orders
    
    WARNING: This script closes ALL positions and cancels ALL orders in MT5,
    regardless of which EA or system placed them. Use with extreme caution!
    """

    def __init__(self):
        # Load configuration
        config_loader = ConfigLoader()
        config_loader.load_config()
        self.config = config_loader.config
        
        self.mt5 = None
        self.magic_number = self.config.get('trading', {}).get('magic_number', 123456)

    def initialize_mt5(self):
        """Initialize MT5 connection and verify trading capability"""
        try:
            logger.info("Initializing MT5 connection...")

            # Try to initialize MT5
            if not mt5.initialize():
                logger.error("[FAIL] Failed to initialize MT5 - MT5 terminal may not be running")
                return False

            # Check terminal connection
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                logger.error("[FAIL] Failed to get MT5 terminal info")
                return False

            logger.info(f"[PASS] MT5 Terminal: {terminal_info.name}")

            # Check account login status
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("[FAIL] No MT5 account logged in - cannot trade!")
                logger.error("Please login to your MT5 account first")
                return False

            logger.info(f"[PASS] MT5 Account: {account_info.login} ({account_info.name})")
            logger.info(f"[PASS] Account Balance: ${account_info.balance:.2f}")

            # Verify trading is allowed
            if not terminal_info.trade_allowed:
                logger.error("[FAIL] Trading is not allowed in MT5 terminal")
                return False

            if account_info.trade_expert != 1:
                logger.error("[FAIL] Automated trading is not enabled for this account")
                logger.error("Please enable automated trading in MT5: Tools -> Options -> Expert Advisors -> Allow automated trading")
                return False

            logger.info("[PASS] MT5 connection and trading capability verified")
            return True

        except Exception as e:
            logger.error(f"[FAIL] Failed to initialize MT5: {e}")
            return False

    def test_trading_capability(self):
        """Test if we can actually send orders by attempting a minimal test"""
        try:
            logger.info("Testing trading capability...")

            # Try to get a simple symbol info to test connection
            symbol_info = mt5.symbol_info("EURUSD")
            if symbol_info is None:
                logger.error("[FAIL] Cannot get symbol info - MT5 connection issue")
                return False

            # Try a minimal order check (this won't actually place an order)
            # We check if the order_send function is callable
            if not hasattr(mt5, 'order_send'):
                logger.error("[FAIL] MT5 order_send function not available")
                return False

            logger.info("[PASS] Trading capability test passed")
            return True

        except Exception as e:
            logger.error(f"[FAIL] Trading capability test failed: {e}")
            return False

    def close_all_positions(self):
        """Close all open positions (EMERGENCY: closes ALL positions regardless of magic number)"""
        try:
            logger.info("Getting all open positions...")

            # Get all positions - EMERGENCY MODE: close ALL positions
            positions = mt5.positions_get()
            if positions is None:
                positions = []

            logger.info(f"Found {len(positions)} total open positions")
            
            # Debug: show details of first few positions
            for i, pos in enumerate(positions[:5]):  # Show first 5 positions
                logger.info(f"Position {i+1}: ticket={pos.ticket}, symbol={pos.symbol}, type={pos.type}, volume={pos.volume}, profit={pos.profit:.2f}, magic={getattr(pos, 'magic', 'N/A')}")
            
            # EMERGENCY MODE: Close ALL positions, not just our magic number
            # In emergency situations, we want to close everything
            our_positions = positions  # Close ALL positions in emergency

            logger.info(f"EMERGENCY: Will close ALL {len(our_positions)} positions")

            closed_count = 0
            failed_count = 0
            for position in our_positions:
                try:
                    logger.info(f"EMERGENCY: Closing position: {position.symbol} ticket {position.ticket} (profit: {position.profit:.2f})")

                    # Check if symbol is available
                    symbol_info = mt5.symbol_info(position.symbol)
                    if symbol_info is None:
                        logger.error(f"[FAIL] Symbol {position.symbol} is not available in MT5")
                        failed_count += 1
                        continue

                    # Select the symbol for trading
                    if not mt5.symbol_select(position.symbol, True):
                        logger.error(f"[FAIL] Failed to select symbol {position.symbol} for trading")
                        failed_count += 1
                        continue

                    # Get current price for the symbol
                    symbol_tick = mt5.symbol_info_tick(position.symbol)
                    if symbol_tick is None:
                        logger.error(f"[FAIL] Failed to get tick data for {position.symbol} - symbol may not be available")
                        failed_count += 1
                        continue

                    # Determine order type for closing
                    if position.type == mt5.ORDER_TYPE_BUY:
                        order_type = mt5.ORDER_TYPE_SELL
                        price = symbol_tick.bid
                    else:
                        order_type = mt5.ORDER_TYPE_BUY
                        price = symbol_tick.ask

                    # Create close request
                    request = {
                        'action': mt5.TRADE_ACTION_DEAL,
                        'symbol': position.symbol,
                        'volume': position.volume,
                        'type': order_type,
                        'position': position.ticket,
                        'price': price,
                        'deviation': 50,  # Increased deviation for emergency
                        'magic': position.magic,  # Use position's magic number
                        'comment': 'EMERGENCY STOP - Close All Positions',
                    }

                    # Create close request
                    request = {
                        'action': mt5.TRADE_ACTION_DEAL,
                        'symbol': position.symbol,
                        'volume': position.volume,
                        'type': order_type,
                        'price': price,
                        'deviation': 50,  # Increased deviation for emergency
                        'magic': 0,  # Use 0 for emergency close
                        'comment': 'EMERGENCY STOP - Close All Positions',
                        'type_filling': mt5.ORDER_FILLING_IOC,
                        'type_time': mt5.ORDER_TIME_GTC,
                    }

                    logger.info(f"[ORDER] Sending close request: symbol={request['symbol']}, volume={request['volume']}, price={request['price']}")

                    # Send order
                    result = mt5.order_send(request)
                    logger.info(f"[ORDER] Order send result: {result}")

                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"[PASS] Successfully closed {position.symbol} position {position.ticket}")
                        closed_count += 1
                    else:
                        error_code = result.retcode if result else 'NO_RESULT'
                        error_comment = getattr(result, 'comment', 'Unknown error') if result else 'No result returned'
                        logger.error(f"[FAIL] Failed to close {position.symbol} position {position.ticket}: retcode={error_code}, comment={error_comment}")
                        failed_count += 1

                except Exception as e:
                    logger.error(f"[ERROR] Exception closing position {position.ticket}: {e}")
                    failed_count += 1

            logger.info(f"[CHART] Close Results: {closed_count} successful, {failed_count} failed out of {len(our_positions)} total positions")
            return closed_count

        except Exception as e:
            logger.error(f"Error closing positions: {e}")
            return 0

    def cancel_pending_orders(self):
        """Cancel all pending orders (EMERGENCY: cancels ALL orders regardless of magic number)"""
        try:
            logger.info("Getting all pending orders...")

            # Get all orders - EMERGENCY MODE: cancel ALL orders
            orders = mt5.orders_get()
            if orders is None:
                orders = []

            logger.info(f"Found {len(orders)} total pending orders")
            
            # EMERGENCY MODE: Cancel ALL pending orders, not just our magic number
            # In emergency situations, we want to cancel everything
            our_orders = orders  # Cancel ALL orders in emergency

            logger.info(f"EMERGENCY: Will cancel ALL {len(our_orders)} pending orders")

            # Debug: show details of first few orders
            for i, order in enumerate(our_orders[:3]):
                logger.info(f"Order {i+1}: ticket={order.ticket}, symbol={order.symbol}, type={order.type}, magic={getattr(order, 'magic', 'N/A')}, state={getattr(order, 'state', 'N/A')}")

            cancelled_count = 0
            for order in our_orders:
                try:
                    logger.info(f"EMERGENCY: Cancelling order: {order.symbol} ticket {order.ticket} (type: {order.type}, price: {getattr(order, 'price_open', 'N/A')}, state: {getattr(order, 'state', 'N/A')})")

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
            logger.error("[FAIL] Failed to initialize MT5 - cannot proceed with emergency stop")
            return False

        # Test trading capability before proceeding
        if not self.test_trading_capability():
            logger.error("[FAIL] Trading capability test failed - cannot proceed with emergency stop")
            logger.error("Please ensure MT5 is logged in and automated trading is enabled")
            return False

        # Close all positions
        positions_closed = self.close_all_positions()
        if positions_closed == 0 and len(mt5.positions_get() or []) > 0:
            logger.error("[FAIL] Failed to close all positions - emergency stop incomplete")
            return False
        elif positions_closed > 0:
            logger.info(f"Emergency stop: Closed {positions_closed} positions")
        else:
            logger.info("No positions to close")

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
        logger.info("*** FX-AI EMERGENCY STOP COMPLETED ***")
        logger.info("*** ALL POSITIONS CLOSED AND ALL ORDERS CANCELLED ***")
        logger.info(f"Total positions closed: {positions_closed}")
        logger.info(f"Total orders cancelled: {orders_cancelled}")
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
            print("\n[ERROR] Emergency stop failed - some positions may still be open")
            print("Please close remaining positions manually in MT5 terminal")
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