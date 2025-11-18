#!/usr/bin/env python3
"""
Fixed Order Executor - Resolves MT5 Invalid Price Errors (10015)
Properly handles stop distances, price normalization, and filling modes
"""

import MetaTrader5 as mt5
import json
import os
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

class FixedOrderExecutor:
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), 'config', 'config.json')
        self.config = self.load_config()
        self.logger = self.setup_logger()
        self.mt5_initialized = False

    def load_config(self) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    def setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('FixedOrderExecutor')
        logger.setLevel(logging.INFO)

        # Create console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        if not mt5.initialize():
            error = mt5.last_error()
            self.logger.error(f"MT5 initialization failed: {error}")
            return False

        self.mt5_initialized = True
        self.logger.info("MT5 initialized successfully")
        return True

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        if not self.mt5_initialized:
            return None

        info = mt5.symbol_info(symbol)
        if info is None:
            return None

        return {
            'point': info.point,
            'tick_size': info.trade_tick_size,
            'contract_size': info.trade_contract_size,
            'min_lot': info.volume_min,
            'max_lot': info.volume_max,
            'lot_step': info.volume_step,
            'stops_level': info.trade_stops_level,
            'freeze_level': info.trade_freeze_level,
            'digits': info.digits,
            'spread': info.spread
        }

    def get_current_prices(self, symbol: str) -> Optional[Dict]:
        """Get current bid/ask prices"""
        if not self.mt5_initialized:
            return None

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'spread': tick.ask - tick.bid
        }

    def normalize_price(self, price: float, symbol_info: Dict) -> float:
        """Normalize price to tick size"""
        tick_size = symbol_info.get('tick_size', symbol_info['point'])
        if tick_size == 0:
            tick_size = symbol_info['point']

        # Round to nearest tick
        normalized = round(price / tick_size) * tick_size
        return round(normalized, symbol_info['digits'])

    def calculate_stop_distances(self, symbol: str, symbol_info: Dict, prices: Dict,
                               risk_amount: float = 50.0, rr_ratio: float = 2.0) -> Dict:
        """Calculate proper stop distances (1.5-2x minimum requirements)"""
        stops_level = symbol_info['stops_level']
        point = symbol_info['point']
        tick_value = symbol_info.get('tick_value', 1.0)
        contract_size = symbol_info.get('contract_size', 100000)

        # Minimum stop distance in price units
        min_stop_price = stops_level * point

        # Use 2x minimum for safety
        sl_distance_price = max(min_stop_price * 2, min_stop_price * 1.5)

        # For metals, ensure minimum distances
        if symbol in ['XAUUSD', 'XAGUSD']:
            # Metals often need larger stops
            sl_distance_price = max(sl_distance_price, 0.5)  # At least 50 cents for gold/silver

        # Calculate TP distance for desired RR ratio
        tp_distance_price = sl_distance_price * rr_ratio

        # Calculate lot size based on risk amount
        stop_loss_pips = sl_distance_price / point
        pip_value = (tick_value / point) * contract_size
        lot_size = risk_amount / (stop_loss_pips * pip_value)

        # Round lot size to step
        lot_step = symbol_info['lot_step']
        lot_size = round(lot_size / lot_step) * lot_step
        lot_size = max(symbol_info['min_lot'], min(symbol_info['max_lot'], lot_size))

        return {
            'sl_distance_price': sl_distance_price,
            'tp_distance_price': tp_distance_price,
            'lot_size': lot_size,
            'min_stop_price': min_stop_price,
            'stops_level': stops_level
        }

    def test_filling_modes(self, symbol: str, order_type: int, price: float,
                          sl: float, tp: float, volume: float) -> int:
        """Test filling modes by actually trying order_send since order_check fails"""
        filling_modes = [
            mt5.ORDER_FILLING_IOC,
            mt5.ORDER_FILLING_RETURN,
            mt5.ORDER_FILLING_FOK
        ]

        for filling in filling_modes:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 10,
                "magic": self.config.get('trading', {}).get('magic_number', 123456),
                "comment": f"test_{filling}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }

            # Try order_send directly since order_check returns None
            result = mt5.order_send(request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"✅ Filling mode {filling} works for {symbol}")
                # Close the test order immediately
                if result.order:
                    self.close_position(result.order)
                return filling
            else:
                error = mt5.last_error()
                self.logger.debug(f"❌ Filling mode {filling} failed: {error}")

        self.logger.warning(f"❌ No filling mode worked for {symbol}")
        return mt5.ORDER_FILLING_IOC  # Default fallback

    def execute_order(self, symbol: str, direction: str, risk_amount: float = 50.0,
                     rr_ratio: float = 2.0) -> Optional[Dict]:
        """
        Execute a properly validated order

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            direction: 'BUY' or 'SELL'
            risk_amount: Dollar amount to risk
            rr_ratio: Risk-reward ratio

        Returns:
            Order result dict or None if failed
        """
        if not self.mt5_initialized:
            self.logger.error("MT5 not initialized")
            return None

        try:
            # Get symbol info
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                self.logger.error(f"Cannot get symbol info for {symbol}")
                return None

            # Get current prices
            prices = self.get_current_prices(symbol)
            if not prices:
                self.logger.error(f"Cannot get prices for {symbol}")
                return None

            # Determine order type and entry price
            if direction.upper() == 'BUY':
                order_type = mt5.ORDER_TYPE_BUY
                entry_price = prices['ask']
            elif direction.upper() == 'SELL':
                order_type = mt5.ORDER_TYPE_SELL
                entry_price = prices['bid']
            else:
                self.logger.error(f"Invalid direction: {direction}")
                return None

            # Normalize entry price
            entry_price = self.normalize_price(entry_price, symbol_info)

            # Calculate stop distances and lot size
            distances = self.calculate_stop_distances(symbol, symbol_info, prices, risk_amount, rr_ratio)

            # Calculate SL and TP prices
            if direction.upper() == 'BUY':
                sl_price = entry_price - distances['sl_distance_price']
                tp_price = entry_price + distances['tp_distance_price']
            else:  # SELL
                sl_price = entry_price + distances['sl_distance_price']
                tp_price = entry_price - distances['tp_distance_price']

            # Normalize SL and TP
            sl_price = self.normalize_price(sl_price, symbol_info)
            tp_price = self.normalize_price(tp_price, symbol_info)

            # Test filling modes
            filling_mode = self.test_filling_modes(symbol, order_type, entry_price, sl_price, tp_price, distances['lot_size'])

            # Create order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": distances['lot_size'],
                "type": order_type,
                "price": entry_price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 10,
                "magic": self.config.get('trading', {}).get('magic_number', 123456),
                "comment": f"FixedExecutor_{symbol}_{direction}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }

            self.logger.info(f"Executing {direction} order for {symbol}:")
            self.logger.info(f"  Entry: {entry_price}, SL: {sl_price}, TP: {tp_price}")
            self.logger.info(f"  Lot Size: {distances['lot_size']}, RR: {rr_ratio}")
            self.logger.info(f"  Stops Level: {distances['stops_level']}, Min Distance: {distances['min_stop_price']}")

            # Send order
            result = mt5.order_send(request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"✅ Order executed successfully: Ticket {result.order}")
                return {
                    'success': True,
                    'ticket': result.order,
                    'request': request,
                    'result': result,
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': entry_price,
                    'sl_price': sl_price,
                    'tp_price': tp_price,
                    'lot_size': distances['lot_size']
                }
            else:
                error = mt5.last_error()
                self.logger.error(f"❌ Order failed: {error}")
                return {
                    'success': False,
                    'error': error,
                    'request': request,
                    'result': result,
                    'symbol': symbol,
                    'direction': direction
                }

        except Exception as e:
            self.logger.error(f"Exception in execute_order: {e}")
            return None

    def test_all_symbols(self) -> Dict:
        """Test order execution for all configured symbols"""
        if not self.initialize_mt5():
            return {}

        symbols = self.config.get('trading', {}).get('symbols', [])
        results = {}

        self.logger.info(f"Testing order execution for {len(symbols)} symbols")

        for symbol in symbols:
            self.logger.info(f"\n--- Testing {symbol} ---")

            # Test BUY
            buy_result = self.execute_order(symbol, 'BUY', risk_amount=10.0)  # Small test amount
            if buy_result and buy_result['success']:
                # Close the test position immediately
                self.close_position(buy_result['ticket'])

            # Test SELL
            sell_result = self.execute_order(symbol, 'SELL', risk_amount=10.0)
            if sell_result and sell_result['success']:
                self.close_position(sell_result['ticket'])

            results[symbol] = {
                'buy_test': buy_result,
                'sell_test': sell_result
            }

        mt5.shutdown()
        return results

    def close_position(self, ticket: int) -> bool:
        """Close a position by ticket"""
        try:
            # Get position info
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False

            position = position[0]

            # Create close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
                "deviation": 10,
                "magic": position.magic,
                "comment": "Close test position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            return result and result.retcode == mt5.TRADE_RETCODE_DONE

        except Exception as e:
            self.logger.error(f"Error closing position {ticket}: {e}")
            return False

if __name__ == "__main__":
    # Test the fixed executor
    executor = FixedOrderExecutor()

    print("Testing Fixed Order Executor...")
    print("This will attempt small test trades for all symbols")
    print("Press Ctrl+C to stop if needed\n")

    results = executor.test_all_symbols()

    # Summary
    successful = 0
    total = 0

    for symbol, tests in results.items():
        total += 1
        buy_ok = tests['buy_test'] and tests['buy_test'].get('success', False)
        sell_ok = tests['sell_test'] and tests['sell_test'].get('success', False)

        if buy_ok and sell_ok:
            successful += 1
            print(f"✅ {symbol}: OK")
        else:
            print(f"❌ {symbol}: FAILED")
            if not buy_ok:
                error = tests['buy_test'].get('error', 'Unknown') if tests['buy_test'] else 'No result'
                print(f"    BUY failed: {error}")
            if not sell_ok:
                error = tests['sell_test'].get('error', 'Unknown') if tests['sell_test'] else 'No result'
                print(f"    SELL failed: {error}")

    print(f"\nSummary: {successful}/{total} symbols working")

    # Save results
    with open('fixed_executor_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("Detailed results saved to fixed_executor_test_results.json")