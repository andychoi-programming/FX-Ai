#!/usr/bin/env python3
"""
Position Monitor Script for FX-Ai Trading System
Monitors existing positions and closes them if they exceed loss thresholds.
Runs continuously to protect capital when SL/TP cannot be set.
"""

import MetaTrader5 as mt5
import time
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Risk management settings
MAX_LOSS_PERCENT = 0.05  # Close position if loses 5% of account balance
CHECK_INTERVAL = 60  # Check every 60 seconds

def initialize_mt5():
    """Initialize MT5 connection"""
    if not mt5.initialize():
        logger.error("Failed to initialize MT5")
        return False
    logger.info("MT5 initialized successfully")
    return True

def get_account_balance():
    """Get current account balance"""
    account = mt5.account_info()
    return account.balance if account else 0

def get_positions():
    """Get all open positions"""
    positions = mt5.positions_get()
    if positions is None:
        logger.error("Failed to get positions")
        return []
    return positions

def should_close_position(position, account_balance):
    """Determine if a position should be closed based on risk rules"""
    current_profit = position.profit

    # Close if loss exceeds threshold
    loss_threshold = -account_balance * MAX_LOSS_PERCENT
    if current_profit < loss_threshold:
        logger.warning(".2f")
        return True, f"Loss exceeded {MAX_LOSS_PERCENT*100}% threshold"

    # For profitable positions, be more conservative
    # Close if profit drops below 50% of peak profit (trailing stop concept)
    # This would require tracking peak profit, for now just use basic loss threshold

    return False, ""

def close_position_safely(position, reason="Risk management"):
    """Close a position safely"""
    # Determine close type
    if position.type == mt5.POSITION_TYPE_BUY:
        close_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(position.symbol).bid
    else:
        close_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(position.symbol).ask

    if not price:
        logger.error(f"Could not get price for {position.symbol}")
        return False

    request = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': position.symbol,
        'volume': position.volume,
        'type': close_type,
        'position': position.ticket,
        'price': price,
        'deviation': 10,
        'magic': 0,
        'comment': reason,
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"✅ Closed {position.symbol} position, profit: {position.profit:.2f}")
        return True
    else:
        logger.error(f"❌ Failed to close {position.symbol}: {result.comment}")
        return False

def monitor_positions():
    """Main monitoring loop"""
    logger.info("Starting position monitoring...")

    if not initialize_mt5():
        return

    try:
        account_balance = get_account_balance()
        logger.info(f"Account balance: {account_balance:.2f}")

        while True:
            try:
                positions = get_positions()
                if not positions:
                    logger.info("No open positions to monitor")
                    time.sleep(CHECK_INTERVAL)
                    continue

                logger.info(f"Monitoring {len(positions)} positions...")

                for position in positions:
                    should_close, reason = should_close_position(position, account_balance)
                    if should_close:
                        logger.warning(f"Closing {position.symbol} position: {reason}")
                        if close_position_safely(position, reason):
                            logger.info(f"Successfully closed {position.symbol}")
                        else:
                            logger.error(f"Failed to close {position.symbol}")
                    else:
                        logger.info(f"{position.symbol}: profit={position.profit:.2f}")

                time.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(CHECK_INTERVAL)

    finally:
        mt5.shutdown()
        logger.info("MT5 connection closed")

if __name__ == "__main__":
    monitor_positions()