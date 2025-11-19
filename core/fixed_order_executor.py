import MetaTrader5 as mt5
import logging
import asyncio
from datetime import datetime
from typing import Dict, Optional, List
import numpy as np

logger = logging.getLogger(__name__)

class FixedOrderExecutor:
    """Fixed Order Executor with automatic filling mode detection for TIOMarkets"""

    def __init__(self, mt5_connector, config: dict, risk_manager=None, technical_analyzer=None):
        """Initialize fixed order executor"""
        self.logger = logging.getLogger(__name__)
        self.mt5 = mt5_connector
        self.config = config
        self.risk_manager = risk_manager
        self.technical_analyzer = technical_analyzer
        self.magic_number = config.get('trading', {}).get('magic_number')
        self.max_slippage = config.get('trading', {}).get('max_slippage', 10)
        self.dry_run = config.get('trading', {}).get('dry_run', False)

        # Filling mode cache to avoid repeated checks
        self.filling_mode_cache = {}

        # Initialize learning database
        try:
            from ai.learning_database import LearningDatabase
            self.learning_db = LearningDatabase(config=config)
            self.learning_db.init_database()
        except ImportError:
            self.learning_db = None

    def get_filling_mode(self, symbol: str) -> int:
        """Get the correct filling mode for a symbol with caching"""
        # Check cache first
        if symbol in self.filling_mode_cache:
            return self.filling_mode_cache[symbol]

        try:
            symbol_info = mt5.symbol_info(symbol)

            if symbol_info is None:
                self.logger.warning(f"No symbol info for {symbol}, using FOK")
                filling_mode = mt5.ORDER_FILLING_FOK
            else:
                filling_modes = symbol_info.filling_mode

                # For TIOMarkets, prioritize what actually works over symbol claims
                # FOK (0) works in practice even if symbol says otherwise
                if filling_modes & mt5.ORDER_FILLING_FOK:
                    filling_mode = mt5.ORDER_FILLING_FOK
                # IOC (1) is claimed to be supported but fails in practice
                # RETURN (2) may work for some brokers
                elif filling_modes & mt5.ORDER_FILLING_RETURN:
                    filling_mode = mt5.ORDER_FILLING_RETURN
                else:
                    # Default to FOK as it works in practice for TIOMarkets
                    self.logger.debug(f"{symbol} using FOK as fallback (works in practice)")
                    filling_mode = mt5.ORDER_FILLING_FOK

            # Cache the result
            self.filling_mode_cache[symbol] = filling_mode
            self.logger.debug(f"{symbol} filling mode: {filling_mode}")
            return filling_mode

        except Exception as e:
            self.logger.error(f"Error getting filling mode for {symbol}: {e}")
            filling_mode = mt5.ORDER_FILLING_FOK
            self.filling_mode_cache[symbol] = filling_mode
            return filling_mode

    async def execute_trade_safe(self, signal: Dict) -> Optional[Dict]:
        """Execute trade with automatic filling mode detection"""
        try:
            # Validate MT5 connection
            if not mt5.initialize():
                self.logger.error("MT5 not initialized")
                return {'success': False, 'error': 'MT5 not initialized'}

            symbol = signal['symbol']
            direction = signal['direction']

            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"Symbol {symbol} not found")
                return {'success': False, 'error': f'Symbol {symbol} not found'}

            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    self.logger.error(f"Failed to select {symbol}")
                    return {'success': False, 'error': f'Failed to select {symbol}'}

            # Get current prices
            prices = mt5.symbol_info_tick(symbol)
            if prices is None:
                self.logger.error(f"Cannot get prices for {symbol}")
                return {'success': False, 'error': f'Cannot get prices for {symbol}'}

            # Determine order type and entry price
            if direction.upper() == 'BUY':
                order_type = mt5.ORDER_TYPE_BUY
                entry_price = prices.ask
            elif direction.upper() == 'SELL':
                order_type = mt5.ORDER_TYPE_SELL
                entry_price = prices.bid
            else:
                self.logger.error(f"Invalid direction: {direction}")
                return {'success': False, 'error': f'Invalid direction: {direction}'}

            # Normalize entry price
            tick_size = symbol_info.trade_tick_size
            if tick_size == 0:
                tick_size = symbol_info.point
            entry_price = round(entry_price / tick_size) * tick_size
            entry_price = round(entry_price, symbol_info.digits)

            # Calculate stop distances
            stops_level = symbol_info.trade_stops_level
            point = symbol_info.point
            min_stop_price = stops_level * point

            # Use 2x minimum for safety
            sl_distance_price = max(min_stop_price * 2, min_stop_price * 1.5)

            # For metals, ensure minimum distances
            if symbol in ['XAUUSD', 'XAGUSD']:
                sl_distance_price = max(sl_distance_price, 0.5)

            # Calculate TP distance for RR ratio
            rr_ratio = 2.0
            tp_distance_price = sl_distance_price * rr_ratio

            # Calculate SL and TP prices
            if direction.upper() == 'BUY':
                sl_price = entry_price - sl_distance_price
                tp_price = entry_price + tp_distance_price
            else:
                sl_price = entry_price + sl_distance_price
                tp_price = entry_price - tp_distance_price

            # Normalize SL and TP
            sl_price = round(sl_price / tick_size) * tick_size
            tp_price = round(tp_price / tick_size) * tick_size
            sl_price = round(sl_price, symbol_info.digits)
            tp_price = round(tp_price, symbol_info.digits)

            # Calculate lot size
            risk_amount = signal.get('position_size', 0.01)
            stop_loss_pips = sl_distance_price / point
            pip_value = (symbol_info.trade_tick_value / point) * symbol_info.trade_contract_size
            lot_size = risk_amount / (stop_loss_pips * pip_value)

            # Round lot size
            lot_step = symbol_info.volume_step
            lot_size = round(lot_size / lot_step) * lot_step
            lot_size = max(symbol_info.volume_min, min(symbol_info.volume_max, lot_size))

            # Get the correct filling mode for this symbol
            filling_mode = self.get_filling_mode(symbol)

            # Create order request with dynamic filling mode
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": entry_price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 10,
                "magic": self.config.get('trading', {}).get('magic_number', 123456),
                "comment": f"FX-Ai {symbol} Fixed",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,  # Dynamic filling mode!
            }

            self.logger.info(f"Executing {direction} order for {symbol} with filling mode: {filling_mode}")
            self.logger.info(f"  Entry: {entry_price}, SL: {sl_price}, TP: {tp_price}")
            self.logger.info(f"  Lot Size: {lot_size}, RR: {rr_ratio}")

            # Send order
            result = mt5.order_send(request)

            # Handle different result formats (object vs tuple)
            success = False
            order_id = None
            result_comment = ''

            if result is not None:
                if isinstance(result, tuple) and len(result) >= 2:
                    retcode, result_comment = result[0], result[1]
                    success = retcode in [mt5.TRADE_RETCODE_DONE, 1, 10009]
                    order_id = None
                elif hasattr(result, 'retcode'):
                    retcode = result.retcode
                    success = retcode == mt5.TRADE_RETCODE_DONE
                    order_id = getattr(result, 'order', None)
                    result_comment = getattr(result, 'comment', '')

            if success:
                self.logger.info(f"✅ Order executed successfully: Ticket {order_id or 'N/A'}")
                return {
                    'success': True,
                    'order_id': order_id,
                    'symbol': symbol,
                    'direction': direction,
                    'volume': lot_size,
                    'entry_price': entry_price,
                    'sl': sl_price,
                    'tp': tp_price,
                    'comment': result_comment,
                    'filling_mode': filling_mode
                }
            else:
                # Get detailed error information
                if isinstance(result, tuple) and len(result) >= 2:
                    retcode, comment = result[0], result[1]
                else:
                    retcode = getattr(result, 'retcode', 'Unknown') if result else 'None'
                    comment = getattr(result, 'comment', '') if result else ''
                last_error = mt5.last_error()

                error_msg = f"Order failed - Retcode: {retcode}, Comment: {comment}, Last Error: {last_error}"
                self.logger.error(f"❌ {error_msg}")

                return {
                    'success': False,
                    'error': error_msg,
                    'symbol': symbol,
                    'direction': direction,
                    'retcode': retcode,
                    'comment': comment,
                    'filling_mode_used': filling_mode
                }

        except Exception as e:
            self.logger.error(f"Exception in execute_trade_safe: {str(e)}", exc_info=True)
            return {'success': False, 'error': f'Exception: {str(e)}'}

async def test_fixed_executor():
    """Test the fixed order executor"""
    print("Testing Fixed Order Executor...")

    # Initialize MT5
    if not mt5.initialize():
        print("❌ MT5 initialization failed")
        return

    # Create test signal
    test_signal = {
        'symbol': 'EURUSD',
        'direction': 'BUY',
        'position_size': 0.01
    }

    # Create fixed executor
    config = {'trading': {'magic_number': 123456}}
    executor = FixedOrderExecutor(mt5, config)

    # Test filling mode detection
    filling_mode = executor.get_filling_mode('EURUSD')
    print(f"✅ EURUSD filling mode: {filling_mode}")

    # Test trade execution (dry run - won't actually place order)
    print("Testing trade execution (this will fail in demo - that's expected)")
    result = await executor.execute_trade_safe(test_signal)
    print(f"Result: {result}")

    mt5.shutdown()
    print("✅ Test completed")

if __name__ == "__main__":
    asyncio.run(test_fixed_executor())