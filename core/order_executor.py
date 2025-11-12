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
        self.magic_number = config.get('trading', {}).get('magic_number', 20241029)
        self.max_slippage = config.get('trading', {}).get('max_slippage', 3)

    def get_filling_mode(self, symbol):
        """Get appropriate filling mode for symbol"""
        try:
            symbol_info = mt5.symbol_info(symbol)  # type: ignore
            if symbol_info is None:
                return mt5.ORDER_FILLING_FOK  # Fallback

            # Check available filling modes
            filling_modes = symbol_info.filling_mode

            # Prefer IOC, then FOK, then RETURN
            if filling_modes & mt5.ORDER_FILLING_IOC:
                return mt5.ORDER_FILLING_IOC
            elif filling_modes & mt5.ORDER_FILLING_FOK:
                return mt5.ORDER_FILLING_FOK
            elif filling_modes & mt5.ORDER_FILLING_RETURN:
                return mt5.ORDER_FILLING_RETURN
            else:
                return mt5.ORDER_FILLING_FOK  # Default fallback

        except Exception as e:
            logger.error(f"Error getting filling mode for {symbol}: {e}")
            return mt5.ORDER_FILLING_FOK

    def _calculate_min_stop_distance(self, symbol: str, symbol_info) -> float:
        """Calculate minimum stop distance in price units"""
        stops_level = getattr(symbol_info, 'trade_stops_level', 0)

        # Calculate minimum stop distance in PIPS, not points
        # Gold: 1 pip = 0.10, Silver: 1 pip = 0.001, JPY: 1 pip = 0.01, others: 1 pip = 0.0001
        if 'XAG' in symbol:
            pip_size = 0.001  # Silver
            # Silver: minimum 50 pips due to wider spreads
            min_stop_pips = max(stops_level / 10, 50)
        elif 'XAU' in symbol or 'GOLD' in symbol:
            pip_size = 0.10  # Gold
            # Gold: minimum 50 pips due to wider spreads
            min_stop_pips = max(stops_level / 100, 50)
        elif "JPY" in symbol:
            pip_size = 0.01  # JPY pairs
            # Convert points to pips, minimum 15 pips
            min_stop_pips = max(stops_level / 10, 15)
        else:
            pip_size = 0.0001  # Standard forex
            # Convert points to pips, minimum 15 pips
            min_stop_pips = max(stops_level / 10, 15)

        return min_stop_pips * pip_size

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

        if ratio < 3.0:
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
        """Place order through MT5 - ASYNC"""
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

            # Get current price if not provided
            if price is None:
                tick = mt5.symbol_info_tick(symbol)  # type: ignore
                if tick is None:
                    logger.error(f"Failed to get tick data for {symbol}")
                    return {'success': False,
                            'error': f'Failed to get tick data for {symbol}'}
                if order_type.lower() in ['buy', 'buy_limit', 'buy_stop']:
                    price = tick.ask
                else:
                    price = tick.bid

            # Determine MT5 order type
            if order_type.lower() == 'buy':
                mt5_order_type = mt5.ORDER_TYPE_BUY
            elif order_type.lower() == 'sell':
                mt5_order_type = mt5.ORDER_TYPE_SELL
            elif order_type.lower() == 'buy_limit':
                mt5_order_type = mt5.ORDER_TYPE_BUY_LIMIT
            elif order_type.lower() == 'sell_limit':
                mt5_order_type = mt5.ORDER_TYPE_SELL_LIMIT
            elif order_type.lower() == 'buy_stop':
                mt5_order_type = mt5.ORDER_TYPE_BUY_STOP
            elif order_type.lower() == 'sell_stop':
                mt5_order_type = mt5.ORDER_TYPE_SELL_STOP
            else:
                return {'success': False,
                        'error': f'Unknown order type: {order_type}'}

            # Adjust stop loss to meet minimum requirements
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

            # Round prices to symbol's decimal places
            price, stop_loss, take_profit = self._round_prices_to_symbol_precision(
                symbol, price, stop_loss, take_profit, symbol_info)

            # Ensure broker minimum stop distance restrictions
            stop_loss, take_profit = self._ensure_broker_minimum_stops(
                symbol, order_type, price, stop_loss, take_profit, symbol_info)

            # Debug logging
            logger.debug("=== SENDING TO MT5 ===")
            logger.debug(f"Symbol: {symbol}")
            logger.debug(f"Entry: {price}")
            logger.debug(f"Stop Loss: {stop_loss}")
            logger.debug(f"Take Profit: {take_profit}")
            logger.debug("=" * 30)

            # Create order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5_order_type,
                "price": price,
                "deviation": self.max_slippage,
                "magic": self.magic_number,
                "comment": comment or "FX-Ai",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self.get_filling_mode(symbol),
            }

            # Add stop loss and take profit if provided
            if stop_loss is not None:
                request["sl"] = stop_loss
            if take_profit is not None:
                request["tp"] = take_profit

            # Send order
            result = mt5.order_send(request)  # type: ignore

            # Check if order_send returned None
            if result is None:
                logger.error("Order send failed - no response from MT5")
                return {
                    'success': False,
                    'error': 'Order send failed - no response from MT5'
                }

            if result.retcode == mt5.TRADE_RETCODE_DONE:
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

                    logger.debug("\n✅ ORDER VERIFICATION:")
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