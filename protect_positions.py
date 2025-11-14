#!/usr/bin/env python3
"""
Position Protection Script for FX-Ai Trading System
Safely manages SL/TP for existing positions when market allows modifications.
"""

import MetaTrader5 as mt5
import time
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_mt5():
    """Initialize MT5 connection"""
    if not mt5.initialize():
        logger.error("Failed to initialize MT5")
        return False
    logger.info("MT5 initialized successfully")
    return True

def get_positions():
    """Get all open positions"""
    positions = mt5.positions_get()
    if positions is None:
        logger.error("Failed to get positions")
        return []
    return positions

def can_modify_positions():
    """Check if position modifications are currently allowed"""
    # Check trade mode for key symbols
    symbols_to_check = ['XAUUSD', 'XAGUSD', 'EURUSD']

    for symbol in symbols_to_check:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info and symbol_info.trade_mode == 4:  # FULL access
            # Try a test modification on a dummy position
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                # Try to modify SL/TP with current values (should succeed if market is open)
                pos = positions[0]
                request = {
                    'action': mt5.TRADE_ACTION_SLTP,
                    'position': pos.ticket,
                    'sl': pos.sl,
                    'tp': pos.tp
                }
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Market open for modifications on {symbol}")
                    return True
                else:
                    logger.warning(f"Market closed for modifications: {result.comment}")

    logger.warning("Position modifications not currently allowed")
    return False

def calculate_safe_sl_tp(position):
    """Calculate safe SL/TP levels for a position"""
    symbol = position.symbol
    open_price = position.price_open
    current_profit = position.profit

    # Get current market price
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return None, None

    if position.type == mt5.POSITION_TYPE_BUY:
        current_price = tick.bid
        # For long positions, SL below open price, TP above
        sl_price = open_price - (abs(open_price - current_price) * 0.5)  # 50% of current gain as buffer
        tp_price = current_price + (abs(current_price - open_price) * 2)  # 2:1 reward ratio
    else:  # POSITION_TYPE_SELL
        current_price = tick.ask
        # For short positions, SL above open price, TP below
        sl_price = open_price + (abs(open_price - current_price) * 0.5)  # 50% of current gain as buffer
        tp_price = current_price - (abs(current_price - open_price) * 2)  # 2:1 reward ratio

    # Ensure minimum distance (spread * 2)
    spread = tick.ask - tick.bid
    min_distance = spread * 2

    if position.type == mt5.POSITION_TYPE_BUY:
        if sl_price >= current_price - min_distance:
            sl_price = current_price - min_distance
        if tp_price <= current_price + min_distance:
            tp_price = current_price + min_distance
    else:
        if sl_price <= current_price + min_distance:
            sl_price = current_price + min_distance
        if tp_price >= current_price - min_distance:
            tp_price = current_price - min_distance

    return round(sl_price, 5), round(tp_price, 5)

def modify_position_sl_tp(position, sl_price, tp_price):
    """Modify SL/TP for a position"""
    request = {
        'action': mt5.TRADE_ACTION_SLTP,
        'position': position.ticket,
        'sl': sl_price,
        'tp': tp_price
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"✅ Modified {position.symbol} SL={sl_price}, TP={tp_price}")
        return True
    else:
        logger.error(f"❌ Failed to modify {position.symbol}: {result.comment}")
        return False

def close_position_safely(position, reason="Manual close"):
    """Close a position safely"""
    # Determine close type
    if position.type == mt5.POSITION_TYPE_BUY:
        close_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(position.symbol).bid
    else:
        close_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(position.symbol).ask

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
        logger.info(f"✅ Closed {position.symbol} position, profit: {position.profit}")
        return True
    else:
        logger.error(f"❌ Failed to close {position.symbol}: {result.comment}")
        return False

def main():
    """Main position management function"""
    logger.info("Starting position protection script...")

    if not initialize_mt5():
        return

    try:
        positions = get_positions()
        if not positions:
            logger.info("No open positions found")
            return

        logger.info(f"Found {len(positions)} open positions")

        # Check if we can modify positions
        if not can_modify_positions():
            logger.warning("Cannot modify positions right now. Waiting...")
            # Could implement a loop here to wait until market opens
            return

        # Process each position
        for position in positions:
            logger.info(f"Processing {position.symbol} position: profit={position.profit}")

            # Calculate safe SL/TP
            sl_price, tp_price = calculate_safe_sl_tp(position)
            if sl_price is None:
                logger.error(f"Could not calculate SL/TP for {position.symbol}")
                continue

            logger.info(f"Calculated SL={sl_price}, TP={tp_price} for {position.symbol}")

            # Modify SL/TP
            if modify_position_sl_tp(position, sl_price, tp_price):
                logger.info(f"Successfully protected {position.symbol} position")
            else:
                logger.warning(f"Could not modify {position.symbol}, consider manual intervention")

    finally:
        mt5.shutdown()
        logger.info("MT5 connection closed")

if __name__ == "__main__":
    main()