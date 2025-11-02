"""
MT5 Connector Module
Handles all communication with MetaTrader 5 platform
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import pytz

class MT5Connector:
    """MetaTrader 5 connection and data management"""

    def __init__(self, login=None, password=None, server=None, path=None):
        """
        Initialize MT5 connector

        Args:
            login: MT5 account number
            password: MT5 password
            server: Broker server name
            path: Path to terminal64.exe
        """
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.connected = False
        self.logger = logging.getLogger(__name__)

        # Timezone settings
        self.broker_tz = pytz.timezone('Etc/GMT-3')  # Most brokers use GMT+3
        self.local_tz = pytz.timezone('UTC')

        # Symbol information cache
        self.symbol_info_cache = {}

    def connect(self) -> bool:
        """Establish connection to MT5"""
        try:
            # Initialize MT5
            init_params = {}
            if self.path:
                init_params['path'] = self.path

            if not mt5.initialize(**init_params):
                self.logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                return False

            # Login if credentials provided
            if self.login and self.password and self.server:
                if not mt5.login(self.login, password=self.password, server=self.server):
                    self.logger.error(f"MT5 login failed: {mt5.last_error()}")
                    mt5.shutdown()
                    return False

            # Verify connection
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                self.logger.error("Failed to get terminal info")
                mt5.shutdown()
                return False

            account_info = mt5.account_info()
            if account_info is None:
                self.logger.error("Failed to get account info")
                mt5.shutdown()
                return False

            self.connected = True
            self.logger.info(f"Connected to MT5: {account_info.server}")
            self.logger.info(f"Account: {account_info.login}, Balance: ${account_info.balance:.2f}")

            # Cache symbol information
            self._cache_symbol_info()

            return True

        except Exception as e:
            self.logger.error(f"MT5 connection error: {e}")
            return False

    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            self.logger.info("Disconnected from MT5")

    def _cache_symbol_info(self):
        """Cache symbol information for faster access"""
        symbols = mt5.symbols_get()
        if symbols:
            for symbol in symbols:
                if symbol.visible:
                    self.symbol_info_cache[symbol.name] = {
                        'digits': symbol.digits,
                        'point': symbol.point,
                        'tick_size': symbol.trade_tick_size,
                        'tick_value': symbol.trade_tick_value,
                        'min_lot': symbol.volume_min,
                        'max_lot': symbol.volume_max,
                        'lot_step': symbol.volume_step,
                        'contract_size': symbol.trade_contract_size,
                        'swap_long': symbol.swap_long,
                        'swap_short': symbol.swap_short,
                        'margin_initial': symbol.margin_initial
                    }
            self.logger.info(f"Cached info for {len(self.symbol_info_cache)} symbols")

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        if symbol in self.symbol_info_cache:
            return self.symbol_info_cache[symbol]

        info = mt5.symbol_info(symbol)
        if info:
            symbol_dict = info._asdict()
            self.symbol_info_cache[symbol] = symbol_dict
            return symbol_dict
        return None

    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current bid/ask prices"""
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            return {
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'time': datetime.fromtimestamp(tick.time),
                'spread': round((tick.ask - tick.bid) / self.get_symbol_info(symbol)['point'], 1)
            }
        return None

    def get_rates(self, symbol: str, timeframe: int, count: int,
                  start_pos: int = 0) -> Optional[pd.DataFrame]:
        """
        Get historical rates

        Args:
            symbol: Symbol name
            timeframe: MT5 timeframe constant
            count: Number of bars
            start_pos: Starting position (0 = current)
        """
        rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)

        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            # Add additional calculated fields
            df['hl2'] = (df['high'] + df['low']) / 2
            df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
            df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

            return df
        return None

    def get_rates_range(self, symbol: str, timeframe: int,
                       date_from: datetime, date_to: datetime) -> Optional[pd.DataFrame]:
        """Get rates for a specific date range"""
        rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)

        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            return df
        return None

    def get_ticks(self, symbol: str, count: int = 1000) -> Optional[pd.DataFrame]:
        """Get tick data"""
        ticks = mt5.copy_ticks_from(symbol, datetime.now(), count, mt5.COPY_TICKS_ALL)

        if ticks is not None and len(ticks) > 0:
            df = pd.DataFrame(ticks)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df['time_msc'] = pd.to_datetime(df['time_msc'], unit='ms')
            df.set_index('time', inplace=True)
            return df
        return None

    def place_order(self, symbol: str, order_type: str, volume: float,
                   price: float = None, sl: float = None, tp: float = None,
                   deviation: int = 10, comment: str = "", magic: int = 0) -> Dict:
        """
        Place an order

        Args:
            symbol: Trading symbol
            order_type: 'buy', 'sell', 'buy_limit', 'sell_limit', 'buy_stop', 'sell_stop'
            volume: Position size in lots
            price: Order price (for pending orders)
            sl: Stop loss price
            tp: Take profit price
            deviation: Maximum price deviation
            comment: Order comment
            magic: Expert Advisor ID
        """
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {'success': False, 'error': f'Symbol {symbol} not found'}

        # Check if symbol is available for trading
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                return {'success': False, 'error': f'Failed to select {symbol}'}

        # Prepare order request
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': volume,
            'deviation': deviation,
            'magic': magic,
            'comment': comment,
        }

        # Set order type
        order_type_map = {
            'buy': mt5.ORDER_TYPE_BUY,
            'sell': mt5.ORDER_TYPE_SELL,
            'buy_limit': mt5.ORDER_TYPE_BUY_LIMIT,
            'sell_limit': mt5.ORDER_TYPE_SELL_LIMIT,
            'buy_stop': mt5.ORDER_TYPE_BUY_STOP,
            'sell_stop': mt5.ORDER_TYPE_SELL_STOP,
        }

        request['type'] = order_type_map.get(order_type.lower())
        if request['type'] is None:
            return {'success': False, 'error': f'Invalid order type: {order_type}'}

        # Set price for market orders
        if request['type'] in [mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL]:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {'success': False, 'error': 'Failed to get current price'}

            if request['type'] == mt5.ORDER_TYPE_BUY:
                request['price'] = tick.ask
            else:
                request['price'] = tick.bid
        else:
            # Pending order - price required
            if price is None:
                return {'success': False, 'error': 'Price required for pending orders'}
            request['price'] = price
            request['action'] = mt5.TRADE_ACTION_PENDING

        # Set stop loss and take profit
        if sl is not None:
            request['sl'] = sl
        if tp is not None:
            request['tp'] = tp

        # Normalize lot size
        lot_min = symbol_info.volume_min
        lot_max = symbol_info.volume_max
        lot_step = symbol_info.volume_step

        volume = max(lot_min, min(lot_max, volume))
        volume = round(volume / lot_step) * lot_step
        request['volume'] = round(volume, 2)

        # Send order
        result = mt5.order_send(request)

        if result is None:
            return {'success': False, 'error': 'Order send failed - no response'}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                'success': False,
                'error': f'Order failed: {result.comment}',
                'retcode': result.retcode
            }

        return {
            'success': True,
            'order': result.order,
            'deal': result.deal,
            'volume': result.volume,
            'price': result.price,
            'comment': result.comment
        }

    def close_position(self, ticket: int, deviation: int = 10) -> Dict:
        """Close an open position by ticket"""
        position = mt5.positions_get(ticket=ticket)

        if not position:
            return {'success': False, 'error': f'Position {ticket} not found'}

        position = position[0]
        symbol = position.symbol

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {'success': False, 'error': 'Failed to get current price'}

        # Prepare close request
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'position': ticket,
            'symbol': symbol,
            'volume': position.volume,
            'deviation': deviation,
            'magic': position.magic,
            'comment': f'Close #{ticket}',
        }

        # Set opposite order type and price
        if position.type == mt5.ORDER_TYPE_BUY:
            request['type'] = mt5.ORDER_TYPE_SELL
            request['price'] = tick.bid
        else:
            request['type'] = mt5.ORDER_TYPE_BUY
            request['price'] = tick.ask

        # Send order
        result = mt5.order_send(request)

        if result is None:
            return {'success': False, 'error': 'Close order failed - no response'}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                'success': False,
                'error': f'Close failed: {result.comment}',
                'retcode': result.retcode
            }

        return {
            'success': True,
            'order': result.order,
            'deal': result.deal,
            'volume': result.volume,
            'price': result.price
        }

    def modify_position(self, ticket: int, sl: float = None, tp: float = None) -> Dict:
        """Modify stop loss and take profit of a position"""
        position = mt5.positions_get(ticket=ticket)

        if not position:
            return {'success': False, 'error': f'Position {ticket} not found'}

        position = position[0]

        # Prepare modification request
        request = {
            'action': mt5.TRADE_ACTION_SLTP,
            'position': ticket,
            'symbol': position.symbol,
        }

        # Set new SL/TP (use existing if not provided)
        request['sl'] = sl if sl is not None else position.sl
        request['tp'] = tp if tp is not None else position.tp

        # Send modification
        result = mt5.order_send(request)

        if result is None:
            return {'success': False, 'error': 'Modification failed - no response'}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                'success': False,
                'error': f'Modification failed: {result.comment}',
                'retcode': result.retcode
            }

        return {
            'success': True,
            'sl': request['sl'],
            'tp': request['tp']
        }

    def get_positions(self, symbol: str = None) -> List[Dict]:
        """Get open positions"""
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if positions is None:
            return []

        return [position._asdict() for position in positions]

    def get_orders(self, symbol: str = None) -> List[Dict]:
        """Get pending orders"""
        if symbol:
            orders = mt5.orders_get(symbol=symbol)
        else:
            orders = mt5.orders_get()

        if orders is None:
            return []

        return [order._asdict() for order in orders]

    def get_history_orders(self, date_from: datetime, date_to: datetime) -> List[Dict]:
        """Get historical orders"""
        orders = mt5.history_orders_get(date_from, date_to)

        if orders is None:
            return []

        return [order._asdict() for order in orders]

    def get_history_deals(self, date_from: datetime, date_to: datetime) -> List[Dict]:
        """Get historical deals"""
        deals = mt5.history_deals_get(date_from, date_to)

        if deals is None:
            return []

        return [deal._asdict() for deal in deals]

    def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        info = mt5.account_info()
        if info:
            return info._asdict()
        return None

    def get_terminal_info(self) -> Optional[Dict]:
        """Get terminal information"""
        info = mt5.terminal_info()
        if info:
            return info._asdict()
        return None

    def get_server_time(self) -> Optional[datetime]:
        """
        Retrieve the MT5 server time as a timezone-aware UTC datetime.

        Tries several methods depending on the installed MetaTrader5 bindings:
        - mt5.time_current() (returns seconds since epoch)
        - mt5.terminal_info() fields (best-effort fallback)

        Returns:
            datetime (UTC) or None on failure
        """
        try:
            # Preferred: mt5.time_current() -> seconds since epoch (int)
            if hasattr(mt5, 'time_current'):
                ts = mt5.time_current()
                if ts:
                    return datetime.fromtimestamp(ts, tz=timezone.utc)

            # Fallback: try terminal_info attributes
            info = mt5.terminal_info()
            if info is not None:
                # Common attribute names vary by binding/version
                for attr in ('server_time', 'time', 'time_local', 'time_server'):
                    if hasattr(info, attr):
                        try:
                            val = getattr(info, attr)
                            if isinstance(val, (int, float)) and val > 0:
                                return datetime.fromtimestamp(val, tz=timezone.utc)
                        except Exception:
                            continue

            # Last resort: try account/position timestamp via account_info
            acc = mt5.account_info()
            if acc is not None and hasattr(acc, 'timestamp'):
                try:
                    ts = getattr(acc, 'timestamp')
                    if isinstance(ts, (int, float)) and ts > 0:
                        return datetime.fromtimestamp(ts, tz=timezone.utc)
                except Exception:
                    pass

            # Final fallback: use symbol tick time (server time)
            try:
                tick = mt5.symbol_info_tick('EURUSD')
                if tick and hasattr(tick, 'time') and tick.time > 0:
                    return datetime.fromtimestamp(tick.time, tz=timezone.utc)
            except Exception:
                pass

        except Exception as e:
            self.logger.debug(f"get_server_time error: {e}")

        return None

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:
        """Enable or disable a symbol for trading"""
        return mt5.symbol_select(symbol, enable)

    def market_book_add(self, symbol: str) -> bool:
        """Subscribe to market depth"""
        return mt5.market_book_add(symbol)

    def market_book_release(self, symbol: str) -> bool:
        """Unsubscribe from market depth"""
        return mt5.market_book_release(symbol)