"""
FX-Ai Order Executor Module
Handles order placement and execution through MT5
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Optional
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Handles order execution and validation through MT5"""

    def __init__(self, mt5_connector, config: dict):
        """Initialize order executor"""
        self.mt5 = mt5_connector
        self.config = config
        self.magic_number = config.get('trading', {}).get('magic_number', 123456)
        self.max_slippage = config.get('trading', {}).get('max_slippage', 3)
        self.dry_run = config.get('trading', {}).get('dry_run', False)

    def get_filling_mode(self, symbol):
        """Get appropriate filling mode for symbol"""
        try:
            # First check terminal-wide filling mode support
            terminal_info = mt5.terminal_info()  # type: ignore
            terminal_filling_modes = 0
            try:
                if terminal_info and hasattr(terminal_info, 'trade_filling_flags'):
                    terminal_filling_modes = terminal_info.trade_filling_flags
                    logger.debug(f"Terminal supports filling modes: {terminal_filling_modes}")
                else:
                    logger.debug(f"Terminal filling mode info not available")
            except Exception as e:
                logger.debug(f"Could not get terminal filling modes: {e}")

            symbol_info = mt5.symbol_info(symbol)  # type: ignore
            if symbol_info is None:
                logger.warning(f"No symbol info for {symbol}, using FOK")
                return mt5.ORDER_FILLING_FOK  # Fallback

            # Check available filling modes (both terminal and symbol level)
            filling_modes = symbol_info.filling_mode
            logger.debug(f"{symbol} supports filling modes: {filling_modes}")

            # If terminal has filling mode restrictions, respect them
            if terminal_info and hasattr(terminal_info, 'trade_filling_flags'):
                filling_modes &= terminal_info.trade_filling_flags
                logger.debug(f"{symbol} filling modes after terminal restrictions: {filling_modes}")
            else:
                logger.debug(f"Terminal filling mode restrictions not available, using symbol modes only")

            # Try modes in order of preference, but be more permissive
            if filling_modes & mt5.ORDER_FILLING_IOC:
                return mt5.ORDER_FILLING_IOC
            elif filling_modes & mt5.ORDER_FILLING_FOK:
                return mt5.ORDER_FILLING_FOK
            elif filling_modes & mt5.ORDER_FILLING_RETURN:
                return mt5.ORDER_FILLING_RETURN
            else:
                # If no standard modes supported, try immediate or return
                logger.warning(f"{symbol} doesn't support standard filling modes, trying alternatives")
                if filling_modes & mt5.ORDER_FILLING_IMMEDIATE:
                    return mt5.ORDER_FILLING_IMMEDIATE
                else:
                    logger.error(f"{symbol} has no supported filling modes: {filling_modes}")
                    return mt5.ORDER_FILLING_FOK  # Last resort

        except Exception as e:
            logger.error(f"Error getting filling mode for {symbol}: {e}")
            return mt5.ORDER_FILLING_FOK

    def _calculate_min_stop_distance(self, symbol: str, symbol_info) -> float:
        """Calculate minimum stop distance in price units"""
        stops_level = getattr(symbol_info, 'trade_stops_level', 0)

        # stops_level is in points, convert to price units
        point_size = symbol_info.point

        # Minimum stop distance in price units (broker requirement)
        min_stop_price_distance = stops_level * point_size

        # Ensure minimum stop distances for all symbol types
        if 'XAU' in symbol or 'GOLD' in symbol:
            # Gold: ensure minimum 0.5 price units (5 pips * 0.10) - reasonable for metals
            min_stop_price_distance = max(min_stop_price_distance, 0.5)
        elif 'XAG' in symbol or 'SILVER' in symbol:
            # Silver: ensure minimum 0.05 price units (5 pips * 0.01)
            min_stop_price_distance = max(min_stop_price_distance, 0.05)
        else:
            # For forex pairs, ensure minimum 2 pips stop distance
            # Most brokers require at least 2-3 pips for stops
            if 'JPY' in symbol:
                # JPY pairs: 2 pips minimum (0.02 price units)
                min_stop_price_distance = max(min_stop_price_distance, 0.02)
            else:
                # Other forex pairs: 2 pips minimum (0.0002 price units for 5-digit brokers)
                min_stop_price_distance = max(min_stop_price_distance, 0.0002)

        return min_stop_price_distance

        return min_stop_price_distance

    def _adjust_stop_loss(self, symbol: str, order_type: str, price: float,
                         stop_loss: float, symbol_info) -> float:
        """Adjust stop loss to meet broker requirements"""
        min_stop_distance = self._calculate_min_stop_distance(symbol, symbol_info)

        logger.info(
            f"Symbol {symbol}: min_stop_distance={min_stop_distance}")
        logger.info(
            f"Order {order_type} {symbol}: price={price}, original_sl={stop_loss}")

        if order_type.lower() in ['buy', 'buy_limit', 'buy_stop']:
            # For buy orders, stop loss should be below price
            required_sl = price - min_stop_distance
            if stop_loss >= required_sl:
                logger.info(
                    f"Adjusting BUY stop loss from {stop_loss} to {required_sl}")
                return required_sl
        else:
            # For sell orders, stop loss should be above price
            required_sl = price + min_stop_distance
            if stop_loss <= required_sl:
                logger.info(
                    f"Adjusting SELL stop loss from {stop_loss} to {required_sl}")
                return required_sl

        return stop_loss

    def _adjust_take_profit(self, symbol: str, order_type: str, price: float,
                           take_profit: float, symbol_info) -> float:
        """Adjust take profit to meet broker requirements"""
        min_stop_distance = self._calculate_min_stop_distance(symbol, symbol_info)

        logger.info(
            f"Order {order_type} {symbol}: price={price}, original_tp={take_profit}")

        if order_type.lower() in ['buy', 'buy_limit', 'buy_stop']:
            # For buy orders, take profit should be above price
            required_tp = price + min_stop_distance
            if take_profit <= required_tp:
                logger.info(
                    f"Adjusting BUY take profit from {take_profit} to {required_tp} (broker minimum)")
                return required_tp
        else:
            # For sell orders, take profit should be below price
            required_tp = price - min_stop_distance
            if take_profit >= required_tp:
                logger.info(
                    f"Adjusting SELL take profit from {take_profit} to {required_tp} (broker minimum)")
                return required_tp

        return take_profit

    def _validate_risk_reward_ratio(self, symbol: str, order_type: str, price: float,
                                   stop_loss: float, take_profit: float) -> bool:
        """Validate risk-reward ratio meets minimum requirements"""
        risk_distance = abs(stop_loss - price)
        reward_distance = abs(take_profit - price)
        ratio = reward_distance / risk_distance if risk_distance > 0 else 0

        # Allow for small floating point precision errors
        if ratio < 2.95:  # Slightly below 3.0 to account for precision
            logger.error(
                f"Order rejected: insufficient risk-reward ratio {ratio:.2f}:1 (required: 3.0:1)")
            return False

        logger.info(f"Order validated: {ratio:.1f}:1 risk-reward ratio")
        return True

    def _round_prices_to_symbol_precision(self, symbol: str, price: float = None,
                                        stop_loss: float = None, take_profit: float = None,
                                        symbol_info=None):
        """Round prices to symbol's decimal precision"""
        if symbol_info is None:
            symbol_info = mt5.symbol_info(symbol)  # type: ignore

        if price is not None:
            price = round(price, symbol_info.digits)
        if stop_loss is not None:
            stop_loss = round(stop_loss, symbol_info.digits)
        if take_profit is not None:
            take_profit = round(take_profit, symbol_info.digits)

        return price, stop_loss, take_profit

    def _ensure_broker_minimum_stops(self, symbol: str, order_type: str, price: float,
                                    stop_loss: float = None, take_profit: float = None,
                                    symbol_info=None):
        """Ensure SL/TP meet broker minimum stop distance requirements"""
        if symbol_info is None:
            symbol_info = mt5.symbol_info(symbol)  # type: ignore

        min_stop_points = getattr(symbol_info, 'trade_stops_level', 0)
        min_stop_distance = max(min_stop_points * symbol_info.point, 0.0001)

        # Adjust stop loss if too close
        if stop_loss is not None:
            actual_sl_distance = abs(price - stop_loss)
            if actual_sl_distance < min_stop_distance:
                logger.debug(
                    f"[WARNING] BROKER MINIMUM STOP: Required "
                    f"{min_stop_distance:.5f}, have {actual_sl_distance:.5f}")
                # Adjust to meet broker minimum
                if order_type.lower() in ['buy', 'buy_limit', 'buy_stop']:
                    stop_loss = price - min_stop_distance
                else:
                    stop_loss = price + min_stop_distance
                stop_loss = round(stop_loss, symbol_info.digits)
                logger.debug(f"Adjusted SL to: {stop_loss}")

        # Adjust take profit if too close
        if take_profit is not None:
            actual_tp_distance = abs(price - take_profit)
            if actual_tp_distance < min_stop_distance:
                logger.debug(
                    f"[WARNING] BROKER MINIMUM TP: Required "
                    f"{min_stop_distance:.5f}, have {actual_tp_distance:.5f}")
                # Adjust to meet broker minimum
                if order_type.lower() in ['buy', 'buy_limit', 'buy_stop']:
                    take_profit = price + min_stop_distance
                else:
                    take_profit = price - min_stop_distance
                take_profit = round(take_profit, symbol_info.digits)
                logger.debug(f"Adjusted TP to: {take_profit}")

        return stop_loss, take_profit

    async def place_order(self, symbol: str, order_type: str, volume: float,
                          stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                          price: Optional[float] = None, comment: str = "") -> Dict:
        """Place order through MT5 - ASYNC - ONLY STOP ORDERS ALLOWED"""
        try:
            # Check MT5 connection
            terminal_info = mt5.terminal_info()  # type: ignore
            if terminal_info is None:
                logger.error("MT5 terminal not connected")
                return {
                    'success': False,
                    'error': 'MT5 terminal not connected'}

            # Select symbol for trading
            if not mt5.symbol_select(symbol, True):  # type: ignore
                logger.error(f"Failed to select symbol {symbol}")
                return {
                    'success': False,
                    'error': f'Failed to select symbol {symbol}'}

            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)  # type: ignore
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found")
                return {'success': False, 'error': 'Symbol not found'}

            # CHECK FOR DUPLICATE SYMBOLS: No stop orders or positions for same symbol
            # Check existing positions
            positions = mt5.positions_get(symbol=symbol)  # type: ignore
            if positions and len(positions) > 0:
                for pos in positions:
                    if hasattr(pos, 'magic') and pos.magic == self.magic_number:
                        logger.warning(f"[{symbol}] Cannot place stop order: existing position found (ticket {pos.ticket})")
                        return {
                            'success': False,
                            'error': f'Existing position for {symbol} (ticket {pos.ticket})'}

            # Check existing pending orders
            orders = mt5.orders_get(symbol=symbol)  # type: ignore
            if orders and len(orders) > 0:
                for order in orders:
                    if hasattr(order, 'magic') and order.magic == self.magic_number:
                        logger.warning(f"[{symbol}] Cannot place stop order: existing pending order found (ticket {order.ticket})")
                        return {
                            'success': False,
                            'error': f'Existing pending order for {symbol} (ticket {order.ticket})'}

            # FORCE STOP ORDERS ONLY: Convert any order type to stop order
            if order_type.lower() == 'buy':
                order_type = 'buy_stop'
                logger.info(f"[{symbol}] Converting BUY to BUY_STOP order")
            elif order_type.lower() == 'sell':
                order_type = 'sell_stop'
                logger.info(f"[{symbol}] Converting SELL to SELL_STOP order")
            elif order_type.lower() in ['buy_limit', 'sell_limit']:
                # Convert limit orders to stop orders
                if order_type.lower() == 'buy_limit':
                    order_type = 'buy_stop'
                else:
                    order_type = 'sell_stop'
                logger.info(f"[{symbol}] Converting LIMIT to STOP order")

            # Get current price for stop order placement
            tick = mt5.symbol_info_tick(symbol)  # type: ignore
            if tick is None:
                logger.error(f"Failed to get tick data for {symbol}")
                return {'success': False,
                        'error': f'Failed to get tick data for {symbol}'}

            # Calculate stop order price based on direction and current market
            current_price = (tick.ask + tick.bid) / 2  # Use mid price as reference

            # Get stop order distance configuration
            pending_config = self.config.get('trading', {}).get('pending_order_distances', {})

            # Determine pip ranges based on symbol
            if 'XAU' in symbol or 'GOLD' in symbol:
                min_pips = pending_config.get('xauusd_min_pips', 50)
                max_pips = pending_config.get('xauusd_max_pips', 150)
            elif 'XAG' in symbol or 'SILVER' in symbol:
                min_pips = pending_config.get('xagusd_min_pips', 20)
                max_pips = pending_config.get('xagusd_max_pips', 50)
            else:
                min_pips = pending_config.get('forex_min_pips', 10)
                max_pips = pending_config.get('forex_max_pips', 25)

            # Calculate pip size
            if 'XAU' in symbol or 'GOLD' in symbol or 'XAG' in symbol:
                pip_size = symbol_info.point * 10  # Metals: 1 pip = 10 points
            elif symbol_info.digits == 3 or symbol_info.digits == 5:
                pip_size = symbol_info.point * 10
            else:
                pip_size = symbol_info.point

            # Generate random distance within range for stop order placement
            import random
            random_pips = random.uniform(min_pips, max_pips)
            distance = random_pips * pip_size

            # Calculate stop order price based on direction
            if order_type.lower() == 'buy_stop':
                # BUY STOP: place above current price (breakout buying)
                price = current_price + distance
                logger.info(f"[{symbol}] BUY STOP: {random_pips:.1f} pips above current price {current_price:.5f} -> {price:.5f}")
            elif order_type.lower() == 'sell_stop':
                # SELL STOP: place below current price (breakdown selling)
                price = current_price - distance
                logger.info(f"[{symbol}] SELL STOP: {random_pips:.1f} pips below current price {current_price:.5f} -> {price:.5f}")
            else:
                return {'success': False,
                        'error': f'Invalid order type for stop-only system: {order_type}'}

            # Determine MT5 order type (ONLY STOP ORDERS ALLOWED)
            if order_type.lower() == 'buy_stop':
                mt5_order_type = mt5.ORDER_TYPE_BUY_STOP
            elif order_type.lower() == 'sell_stop':
                mt5_order_type = mt5.ORDER_TYPE_SELL_STOP
            else:
                return {'success': False,
                        'error': f'Only stop orders allowed: {order_type} is not supported'}

            # Recalculate stop loss and take profit based on stop order price
            if stop_loss is not None and price is not None:
                # Get default pips from config
                config = getattr(self, 'config', {})
                default_sl_pips = config.get('trading', {}).get('default_sl_pips', 20)
                default_tp_pips = config.get('trading', {}).get('default_tp_pips', 60)
                
                # Calculate pip size
                if "XAU" in symbol or "GOLD" in symbol or "XAG" in symbol:
                    pip_size = symbol_info.point * 10  # Metals: 1 pip = 10 points
                elif symbol_info.digits == 3 or symbol_info.digits == 5:
                    pip_size = symbol_info.point * 10
                else:
                    pip_size = symbol_info.point
                
                # For stop orders, SL/TP are calculated from the stop order price
                if order_type.lower() == 'buy_stop':
                    # BUY STOP: SL below stop price, TP above stop price
                    stop_loss = price - (default_sl_pips * pip_size)
                    if take_profit is not None:
                        take_profit = price + (default_tp_pips * pip_size)
                elif order_type.lower() == 'sell_stop':
                    # SELL STOP: SL above stop price, TP below stop price
                    stop_loss = price + (default_sl_pips * pip_size)
                    if take_profit is not None:
                        take_profit = price - (default_tp_pips * pip_size)
                
                # Round to symbol precision
                stop_loss = round(stop_loss, symbol_info.digits)
                if take_profit is not None:
                    take_profit = round(take_profit, symbol_info.digits)
                
                tp_display = f"{take_profit:.5f}" if take_profit is not None else "None"
                logger.info(f"[{symbol}] SL/TP for {order_type} @ {price:.5f}: SL={stop_loss:.5f}, TP={tp_display}")

            # Adjust stop loss to meet minimum broker requirements
            if stop_loss is not None and price is not None:
                stop_loss = self._adjust_stop_loss(symbol, order_type, price, stop_loss, symbol_info)

            # Adjust take profit to meet minimum requirements
            if take_profit is not None and price is not None:
                take_profit = self._adjust_take_profit(symbol, order_type, price, take_profit, symbol_info)

            # Final validation: ensure adequate risk-reward ratio
            if stop_loss is not None and take_profit is not None and price is not None:
                if not self._validate_risk_reward_ratio(symbol, order_type, price, stop_loss, take_profit):
                    return {
                        'success': False,
                        'error': f'Insufficient risk-reward ratio'}

            # Check if dry run mode is enabled
            if self.dry_run:
                logger.info(f"[DRY RUN] Would place STOP ORDER for {symbol}: {order_type} @ {price:.5f}")
                logger.info(f"[DRY RUN] SL: {stop_loss}, TP: {take_profit}, Volume: {volume}")
                return {
                    'success': True,
                    'order': 999999,  # Fake order ticket
                    'price': price,
                    'dry_run': True
                }

            # Try different filling modes if the first one fails
            preferred_filling_mode = self.get_filling_mode(symbol)
            filling_modes_to_try = [
                preferred_filling_mode,  # Preferred mode first
                mt5.ORDER_FILLING_IOC,         # Immediate or Cancel
                mt5.ORDER_FILLING_RETURN,      # Return remaining
                mt5.ORDER_FILLING_FOK,         # Fill or Kill
            ]
            
            # Remove duplicates but keep order
            seen = set()
            filling_modes_to_try = [x for x in filling_modes_to_try if not (x in seen or seen.add(x))]
            
            result = None
            last_error = None
            
            for filling_mode in filling_modes_to_try:
                try:
                    # Create order request
                    # Use PENDING action for limit/stop orders, DEAL for market orders
                    if mt5_order_type in [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT, 
                                         mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_SELL_STOP]:
                        trade_action = mt5.TRADE_ACTION_PENDING
                        filling_mode = None  # Pending orders don't use filling modes
                    else:
                        trade_action = mt5.TRADE_ACTION_DEAL
                        # Keep filling mode for market orders
                    
                    request = {
                        "action": trade_action,
                        "symbol": symbol,
                        "volume": volume,
                        "type": mt5_order_type,
                        "price": price,
                        "deviation": self.max_slippage,
                        "magic": self.magic_number,
                        "comment": comment or "FX-Ai",
                        "type_time": mt5.ORDER_TIME_GTC,
                    }

                    # Add filling mode only for market orders
                    if filling_mode is not None and trade_action == mt5.TRADE_ACTION_DEAL:
                        request["type_filling"] = filling_mode

                    # Note: SL/TP will be set after order placement using TRADE_ACTION_SLTP
                    # to avoid broker restrictions on including them in the initial request

                    logger.debug(f"Trying filling mode {filling_mode} ({type(filling_mode).__name__}) for {symbol}")
                    
                    # Send order
                    result = mt5.order_send(request)  # type: ignore

                    # Check if order_send returned None
                    if result is None:
                        last_error = f"Filling mode {filling_mode}: Order send failed - no response from MT5"
                        logger.debug(last_error)
                        continue

                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"Order placed with filling mode {filling_mode}: {symbol} {order_type} @ {price}")
                        break  # Success, exit the retry loop
                    else:
                        error_desc = f"Filling mode {filling_mode}: Retcode {result.retcode}"
                        if hasattr(result, 'comment') and result.comment:
                            error_desc += f", Comment: {result.comment}"
                        logger.debug(f"Filling mode {filling_mode} failed: {error_desc}")
                        last_error = error_desc
                        continue
                        
                except Exception as e:
                    last_error = str(e)
                    logger.debug(f"Error with filling mode {filling_mode}: {e}")
                    continue
            
            # If all filling modes failed, try pending orders as fallback
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.info(f"Market orders failed for {symbol}, trying pending orders as fallback")
                pending_result = await self._place_pending_order(
                    symbol, order_type, volume, stop_loss, take_profit, price, comment
                )
                if pending_result and pending_result.get('success', False):
                    logger.info(f"Pending order placed successfully for {symbol}")
                    return pending_result
            
            # If both market and pending orders failed, return the last error
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"All order types failed for {symbol}. Last error: {last_error}")
                return {
                    'success': False,
                    'error': last_error or 'All order types failed'
                }

            # Order was successful - handle SL/TP and verification
            logger.info(f"Order placed: {symbol} {order_type} @ {price}")

            # Use TRADE_ACTION_SLTP to set SL/TP after order placement
            if stop_loss is not None or take_profit is not None:
                await asyncio.sleep(0.2)  # Brief pause before modifying

                # Get the actual position ticket
                positions = mt5.positions_get(symbol=symbol)  # type: ignore
                if positions:
                    position = positions[-1]  # Last position
                    position_ticket = position.ticket

                    modify_request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position_ticket,
                        "symbol": symbol,
                    }

                    if stop_loss is not None:
                        modify_request["sl"] = stop_loss
                    if take_profit is not None:
                        modify_request["tp"] = take_profit

                    logger.debug(
                        f"Modifying position {position_ticket} with SLTP: {modify_request}")
                    modify_result = mt5.order_send(modify_request)  # type: ignore

                    if modify_result and modify_result.retcode != mt5.TRADE_RETCODE_DONE:
                        logger.warning(
                            f"Failed to set SL/TP for position {position_ticket}: "
                            f"{modify_result.comment}")
                    else:
                        logger.debug("[OK] SLTP modification successful")
                else:
                    logger.debug("[WARNING] No position found to modify SL/TP")

                # Enhanced verification
                await asyncio.sleep(0.5)  # Wait for position to register

                positions = mt5.positions_get(symbol=symbol)  # type: ignore
                if positions:
                    actual_position = positions[-1]
                    actual_sl = actual_position.sl
                    actual_tp = actual_position.tp

                    logger.debug("\n[ORDER VERIFICATION]:")
                    logger.debug(f"Requested SL: {stop_loss}")
                    logger.debug(f"MT5 Set SL: {actual_sl}")
                    logger.debug(f"Requested TP: {take_profit}")
                    logger.debug(f"MT5 Set TP: {actual_tp}")

                    if stop_loss is not None:
                        sl_mismatch = abs(actual_sl - stop_loss)
                        if sl_mismatch > 0.01:
                            logger.error(
                                f"Stop loss mismatch for {symbol}: expected "
                                f"{stop_loss}, got {actual_sl}")
                        else:
                            logger.debug("[OK] SL set correctly")

                    if take_profit is not None:
                        tp_mismatch = abs(actual_tp - take_profit)
                        if tp_mismatch > 0.01:
                            logger.debug(
                                f"[WARNING] TP mismatch! Expected {take_profit}, got {actual_tp}")

                return {
                    'success': True,
                    'order': result.order,
                    'price': result.price
                }
            else:
                logger.error(f"Order failed: {result.comment}")
                return {
                    'success': False,
                    'error': result.comment
                }

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _place_pending_order(self, symbol: str, order_type: str, volume: float,
                                   stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                                   market_price: Optional[float] = None, comment: str = "") -> Optional[Dict]:
        """Place pending order as fallback when market orders fail"""
        try:
            # Get current market price
            tick = mt5.symbol_info_tick(symbol)  # type: ignore
            if tick is None:
                logger.error(f"Failed to get tick data for pending order {symbol}")
                return None

            # Get symbol info for spread and pip calculations
            symbol_info = mt5.symbol_info(symbol)  # type: ignore
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found for pending order")
                return None

            # Calculate pending order price based on order type
            # For buy orders, place below current ask (buy limit)
            # For sell orders, place above current bid (sell limit)
            spread = (tick.ask - tick.bid) / symbol_info.point
            
            # Calculate minimum distance based on broker requirements
            stops_level = getattr(symbol_info, 'trade_stops_level', 20)  # Default 20 points
            min_distance_points = stops_level + 10  # Add margin
            pip_distance = min_distance_points / 10  # Convert points to pips (for 5-digit brokers)
            
            # Ensure minimum distance
            if 'JPY' in symbol:
                pip_distance = max(pip_distance, 20)  # JPY pairs need more
            else:
                pip_distance = max(pip_distance, 20)  # Forex minimum 20 pips away

            if order_type.lower() in ['buy', 'buy_limit']:
                # Buy limit: place below current ask
                pending_price = tick.ask - (pip_distance * symbol_info.point * 10)  # *10 for 5-digit
                mt5_order_type = mt5.ORDER_TYPE_BUY_LIMIT
            elif order_type.lower() in ['sell', 'sell_limit']:
                # Sell limit: place above current bid
                pending_price = tick.bid + (pip_distance * symbol_info.point * 10)
                mt5_order_type = mt5.ORDER_TYPE_SELL_LIMIT
            else:
                logger.error(f"Unsupported order type for pending order: {order_type}")
                return None

            # Round price to symbol precision
            pending_price = round(pending_price, symbol_info.digits)

            logger.info(f"Placing pending {order_type} order for {symbol} @ {pending_price} (market: {tick.ask}/{tick.bid})")

            # Create pending order request
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": volume,
                "type": mt5_order_type,
                "price": pending_price,
                "deviation": self.max_slippage,
                "magic": self.magic_number,
                "comment": comment or "FX-Ai Pending",
                "type_time": mt5.ORDER_TIME_GTC,  # Good till cancelled
                "type_filling": mt5.ORDER_FILLING_RETURN,  # Return remaining
            }

            # Add stop loss and take profit if provided
            if stop_loss is not None:
                request["sl"] = stop_loss
            if take_profit is not None:
                request["tp"] = take_profit

            # Send pending order
            result = mt5.order_send(request)  # type: ignore

            if result is None:
                logger.error(f"Pending order send failed - no response from MT5 for {symbol}")
                return None

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Pending order placed successfully: {symbol} {order_type} @ {pending_price}")
                return {
                    'success': True,
                    'order': result.order,
                    'price': pending_price,
                    'pending': True
                }
            else:
                logger.error(f"Pending order failed: {symbol} - Retcode {result.retcode}, Comment: {result.comment}")
                return None

        except Exception as e:
            logger.error(f"Error placing pending order for {symbol}: {e}")
            return None