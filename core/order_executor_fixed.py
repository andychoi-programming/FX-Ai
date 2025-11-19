import MetaTrader5 as mt5
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import json
import os
from decimal import Decimal, ROUND_DOWN

from utils.exceptions import OrderExecutionError, MT5ConnectionError
from utils.logger import get_logger
from core.position_manager import PositionManager
from core.risk_manager import RiskManager

logger = get_logger(__name__)

class OrderExecutor:
    """
    Enhanced Order Executor with dynamic filling mode detection
    Fixes Error 10030 by automatically detecting correct filling modes per symbol
    """

    def __init__(self, position_manager: PositionManager, risk_manager: RiskManager):
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.magic_number = 123456
        self.max_retry_attempts = 3
        self.retry_delay = 1.0

        # Initialize MT5 connection
        self._ensure_mt5_connection()

    def _ensure_mt5_connection(self) -> None:
        """Ensure MT5 connection is active"""
        if not mt5.initialize():
            raise MT5ConnectionError("Failed to initialize MT5")

        terminal_info = mt5.terminal_info()
        if not terminal_info or not terminal_info.connected:
            raise MT5ConnectionError("MT5 terminal not connected")

        logger.info(f"MT5 connected to {terminal_info.name}")

    def get_filling_mode(self, symbol: str) -> int:
        """
        Get the correct filling mode for a symbol
        This is the KEY FIX for Error 10030
        """
        try:
            symbol_info = mt5.symbol_info(symbol)

            if symbol_info is None:
                logger.warning(f"No symbol info for {symbol}, using RETURN mode")
                return mt5.ORDER_FILLING_RETURN  # TIOMarkets default

            # Check supported modes in order of preference
            # TIOMarkets typically uses RETURN (Market Execution)
            if symbol_info.filling_mode & 4:  # RETURN (bit 2)
                return mt5.ORDER_FILLING_RETURN
            elif symbol_info.filling_mode & 2:  # IOC (bit 1)
                return mt5.ORDER_FILLING_IOC
            elif symbol_info.filling_mode & 1:  # FOK (bit 0)
                return mt5.ORDER_FILLING_FOK
            else:
                logger.warning(f"No supported filling modes for {symbol}, using RETURN")
                return mt5.ORDER_FILLING_RETURN

        except Exception as e:
            logger.error(f"Error getting filling mode for {symbol}: {e}")
            return mt5.ORDER_FILLING_RETURN  # Safe default

    def _prepare_order_request(self, symbol: str, order_type: int, volume: float,
                             price: float = 0.0, stop_loss: float = 0.0,
                             take_profit: float = 0.0, comment: str = "") -> Dict:
        """Prepare order request with correct filling mode"""

        # Get the correct filling mode for this symbol
        filling_mode = self.get_filling_mode(symbol)

        # Get current price if not provided
        if price == 0.0:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                raise OrderExecutionError(f"Cannot get tick data for {symbol}")

            price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": 10,
            "magic": self.magic_number,
            "comment": comment or "FX-Ai Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,  # THIS IS THE FIX!
        }

        # Add SL/TP if provided
        if stop_loss > 0:
            request["sl"] = stop_loss
        if take_profit > 0:
            request["tp"] = take_profit

        return request

    def _validate_order_request(self, request: Dict) -> bool:
        """Validate order request before sending"""
        try:
            # Check symbol availability
            if not mt5.symbol_select(request["symbol"], True):
                logger.error(f"Symbol {request['symbol']} not available")
                return False

            # Check volume limits
            symbol_info = mt5.symbol_info(request["symbol"])
            if symbol_info:
                min_vol = symbol_info.volume_min
                max_vol = symbol_info.volume_max
                volume = request["volume"]

                if volume < min_vol or volume > max_vol:
                    logger.error(f"Volume {volume} out of range [{min_vol}, {max_vol}]")
                    return False

            # Use order_check for validation
            result = mt5.order_check(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                return True
            else:
                logger.warning(f"Order check failed: {result.comment if result else 'Unknown'}")
                return False

        except Exception as e:
            logger.error(f"Order validation error: {e}")
            return False

    def _execute_order_with_retry(self, request: Dict) -> Optional[Dict]:
        """Execute order with retry logic"""
        for attempt in range(self.max_retry_attempts):
            try:
                logger.info(f"Executing order (attempt {attempt + 1}/{self.max_retry_attempts})")
                logger.info(f"Symbol: {request['symbol']}, Volume: {request['volume']}, "
                          f"Filling Mode: {request['type_filling']}")

                result = mt5.order_send(request)

                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    ticket = getattr(result, 'order', None)
                    logger.info(f"✅ Order executed successfully! Ticket: {ticket}")
                    return {
                        'success': True,
                        'ticket': ticket,
                        'retcode': result.retcode,
                        'comment': getattr(result, 'comment', ''),
                        'request': request
                    }
                else:
                    retcode = result.retcode if result else 'None'
                    comment = getattr(result, 'comment', '') if result else 'No result'

                    logger.warning(f"Order failed (attempt {attempt + 1}): Retcode {retcode}, Comment: {comment}")

                    # Don't retry on certain errors
                    if retcode in [10014, 10015, 10016]:  # Invalid volume, price, etc.
                        break

                    if attempt < self.max_retry_attempts - 1:
                        logger.info(f"Retrying in {self.retry_delay}s...")
                        asyncio.sleep(self.retry_delay)

            except Exception as e:
                logger.error(f"Order execution error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retry_attempts - 1:
                    asyncio.sleep(self.retry_delay)

        return {
            'success': False,
            'retcode': result.retcode if 'result' in locals() and result else 'Unknown',
            'comment': getattr(result, 'comment', '') if 'result' in locals() and result else 'Failed after retries',
            'request': request
        }

    async def execute_market_order(self, symbol: str, order_type: int, volume: float,
                                 stop_loss: float = 0.0, take_profit: float = 0.0,
                                 comment: str = "") -> Dict:
        """
        Execute market order with dynamic filling mode detection
        This replaces the old hardcoded ORDER_FILLING_IOC approach
        """
        try:
            # Risk validation
            if not await self.risk_manager.validate_trade(symbol, volume, order_type):
                raise OrderExecutionError("Risk validation failed")

            # Prepare order request
            request = self._prepare_order_request(
                symbol, order_type, volume, 0.0, stop_loss, take_profit, comment
            )

            # Validate request
            if not self._validate_order_request(request):
                raise OrderExecutionError("Order validation failed")

            # Execute order
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._execute_order_with_retry, request
            )

            if result['success']:
                # Update position manager
                await self.position_manager.add_position(
                    ticket=result['ticket'],
                    symbol=symbol,
                    order_type=order_type,
                    volume=volume,
                    open_price=request['price'],
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )

            return result

        except Exception as e:
            logger.error(f"Market order execution failed: {e}")
            raise OrderExecutionError(f"Market order failed: {e}")

    async def execute_limit_order(self, symbol: str, order_type: int, volume: float,
                                price: float, stop_loss: float = 0.0,
                                take_profit: float = 0.0, comment: str = "") -> Dict:
        """Execute limit order"""
        try:
            # Risk validation
            if not await self.risk_manager.validate_trade(symbol, volume, order_type):
                raise OrderExecutionError("Risk validation failed")

            # Prepare order request
            request = self._prepare_order_request(
                symbol, order_type, volume, price, stop_loss, take_profit, comment
            )

            # Change action to pending order
            request["action"] = mt5.TRADE_ACTION_PENDING
            request["type_time"] = mt5.ORDER_TIME_GTC

            # Validate and execute
            if not self._validate_order_request(request):
                raise OrderExecutionError("Limit order validation failed")

            result = await asyncio.get_event_loop().run_in_executor(
                None, self._execute_order_with_retry, request
            )

            return result

        except Exception as e:
            logger.error(f"Limit order execution failed: {e}")
            raise OrderExecutionError(f"Limit order failed: {e}")

    async def close_position(self, ticket: int, volume: float = 0.0,
                          price: float = 0.0) -> Dict:
        """Close position"""
        try:
            # Get position info
            position = mt5.positions_get(ticket=ticket)
            if not position:
                raise OrderExecutionError(f"Position {ticket} not found")

            position = position[0]
            symbol = position.symbol
            order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

            close_volume = volume or position.volume
            close_price = price or (mt5.symbol_info_tick(symbol).bid if order_type == mt5.ORDER_TYPE_SELL
                                  else mt5.symbol_info_tick(symbol).ask)

            # Prepare close request
            request = self._prepare_order_request(
                symbol, order_type, close_volume, close_price,
                comment=f"Close {ticket}"
            )

            # Execute close
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._execute_order_with_retry, request
            )

            if result['success']:
                await self.position_manager.remove_position(ticket)

            return result

        except Exception as e:
            logger.error(f"Position close failed: {e}")
            raise OrderExecutionError(f"Close failed: {e}")

    def get_supported_filling_modes(self, symbol: str) -> List[str]:
        """Get list of supported filling modes for a symbol"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return []

            modes = []
            if symbol_info.filling_mode & 1:
                modes.append("FOK")
            if symbol_info.filling_mode & 2:
                modes.append("IOC")
            if symbol_info.filling_mode & 4:
                modes.append("RETURN")

            return modes

        except Exception as e:
            logger.error(f"Error getting supported modes for {symbol}: {e}")
            return []

    def diagnose_symbol(self, symbol: str) -> Dict:
        """Diagnose filling mode support for a symbol"""
        return {
            'symbol': symbol,
            'supported_modes': self.get_supported_filling_modes(symbol),
            'recommended_mode': self.get_filling_mode(symbol),
            'mode_name': self._get_mode_name(self.get_filling_mode(symbol))
        }

    def _get_mode_name(self, mode: int) -> str:
        """Convert filling mode constant to name"""
        if mode == mt5.ORDER_FILLING_FOK:
            return "FOK"
        elif mode == mt5.ORDER_FILLING_IOC:
            return "IOC"
        elif mode == mt5.ORDER_FILLING_RETURN:
            return "RETURN"
        return f"UNKNOWN({mode})"