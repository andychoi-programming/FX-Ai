"""
MT5 Connector Module
Handles all communication with MetaTrader 5 platform
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Any
import pytz
import threading

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
        self.login = int(login) if login is not None else None
        self.password = password
        self.server = server
        self.path = path
        self.connected = False
        self.logger = logging.getLogger(__name__)
        
        # Thread safety: MT5 API is NOT thread-safe, use lock for all MT5 calls
        self._mt5_lock = threading.RLock()  # Reentrant lock to prevent deadlocks

        # Timezone settings
        self.broker_tz = pytz.timezone('Etc/GMT-3')  # Most brokers use GMT+3
        self.local_tz = pytz.timezone('UTC')

        # Symbol information cache
        self.symbol_info_cache = {}

    def connect(self) -> bool:
        """Establish connection to MT5"""
        with self._mt5_lock:  # Thread-safe MT5 access
            try:
                # Initialize MT5
                init_params = {}
                if self.path:
                    init_params['path'] = self.path

                if not mt5.initialize(**init_params):  # type: ignore
                    self.logger.error(f"MT5 initialize failed: {mt5.last_error()}")  # type: ignore
                    return False

                # Login if credentials provided
                if self.login and self.password and self.server:
                    if not mt5.login(self.login, password=self.password, server=self.server):  # type: ignore
                        self.logger.error(f"MT5 login failed: {mt5.last_error()}")  # type: ignore
                        mt5.shutdown()  # type: ignore
                        return False

                # Verify connection
                terminal_info = mt5.terminal_info()  # type: ignore
                if terminal_info is None:
                    self.logger.error("Failed to get terminal info")
                    mt5.shutdown()  # type: ignore
                    return False

                account_info = mt5.account_info()  # type: ignore
                if account_info is None:
                    self.logger.error("Failed to get account info")
                    mt5.shutdown()  # type: ignore
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
        with self._mt5_lock:  # Thread-safe MT5 access
            if self.connected:
                mt5.shutdown()  # type: ignore
                self.connected = False
                self.logger.info("Disconnected from MT5")

    def _cache_symbol_info(self):
        """Cache symbol information for faster access (called inside lock)"""
        symbols = mt5.symbols_get()  # type: ignore
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

        with self._mt5_lock:
            info = mt5.symbol_info(symbol)  # type: ignore
            if info:
                symbol_dict = info._asdict()
                self.symbol_info_cache[symbol] = symbol_dict
                return symbol_dict
        return None

    def get_current_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current bid/ask prices"""
        with self._mt5_lock:
            tick = mt5.symbol_info_tick(symbol)  # type: ignore
            if tick:
                symbol_info = self.get_symbol_info(symbol)
                spread = 0
                if symbol_info and 'point' in symbol_info:
                    spread = round((tick.ask - tick.bid) / symbol_info['point'], 1)
                
                return {
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'last': tick.last,
                    'volume': tick.volume,
                    'time': datetime.fromtimestamp(tick.time),
                    'spread': spread
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
        with self._mt5_lock:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)  # type: ignore

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
        with self._mt5_lock:  # Thread-safe MT5 access
            rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)  # type: ignore

            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                return df
            return None

    def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        with self._mt5_lock:  # Thread-safe MT5 access
            account_info = mt5.account_info()  # type: ignore
            if account_info is None:
                return None
            
            return {
                'login': account_info.login,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'margin_free': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'server': account_info.server,
                'currency': account_info.currency
            }

    async def place_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        deviation: int = 20,
        magic: int = 234000,
        comment: str = "",
        signal_data: Optional[Dict] = None
    ) -> Dict:
        """
        Place order with proper SL/TP handling for both market and pending orders
        """
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {'success': False, 'error': f'Symbol {symbol} not found'}
            
            # Determine if this is a pending order
            is_pending = 'stop' in order_type.lower() or 'limit' in order_type.lower()
            
            # Map order type
            type_dict = {
                'buy': mt5.ORDER_TYPE_BUY,
                'sell': mt5.ORDER_TYPE_SELL,
                'buy_stop': mt5.ORDER_TYPE_BUY_STOP,
                'sell_stop': mt5.ORDER_TYPE_SELL_STOP,
                'buy_limit': mt5.ORDER_TYPE_BUY_LIMIT,
                'sell_limit': mt5.ORDER_TYPE_SELL_LIMIT
            }
            
            mt5_order_type = type_dict.get(order_type.lower())
            if mt5_order_type is None:
                return {'success': False, 'error': f'Invalid order type: {order_type}'}
            
            # Get current price if not provided
            if price is None:
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    return {'success': False, 'error': f'Failed to get tick for {symbol}'}
                price = tick.ask if 'buy' in order_type.lower() else tick.bid
            
            # Normalize prices to symbol's digit precision
            point = symbol_info.point
            digits = symbol_info.digits
            
            price = round(price / point) * point
            
            # CRITICAL: For pending orders, use 'sl' and 'tp' parameters
            # For market orders, these are applied directly
            request = {
                "action": mt5.TRADE_ACTION_PENDING if is_pending else mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5_order_type,
                "price": price,
                "deviation": deviation,
                "magic": magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_DAY,  # Order valid until end of day
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add SL/TP if provided - NORMALIZED to symbol digits
            if stop_loss is not None:
                request["sl"] = round(stop_loss, digits)
            
            if take_profit is not None:
                request["tp"] = round(take_profit, digits)
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                return {'success': False, 'error': 'order_send returned None'}
            
            # Check result
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    'success': False,
                    'error': f'Order failed: {result.comment}',
                    'retcode': result.retcode
                }
            
            # Log success
            order_info = {
                'success': True,
                'order': result.order,
                'volume': result.volume,
                'price': result.price,
                'sl': stop_loss,
                'tp': take_profit,
                'comment': result.comment
            }
            
            self.logger.info(f"[ORDER PLACED] {symbol} {order_type.upper()} | "
                            f"Vol: {volume:.2f} | Price: {price:.5f} | "
                            f"SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
            
            return order_info
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {'success': False, 'error': str(e)}

    def close_position(self, ticket: int, deviation: int = 10) -> Dict:
        """Close an open position by ticket"""
        with self._mt5_lock:  # Thread-safe MT5 access
            position = mt5.positions_get(ticket=ticket)  # type: ignore

            if not position:
                return {'success': False, 'error': f'Position {ticket} not found'}

            position = position[0]
            symbol = position.symbol

            # Get current price
            tick = mt5.symbol_info_tick(symbol)  # type: ignore
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
            result = mt5.order_send(request)  # type: ignore

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

    def modify_position(self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None) -> Dict:
        """Modify stop loss and take profit of a position"""
        position = mt5.positions_get(ticket=ticket)  # type: ignore

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
        result = mt5.order_send(request)  # type: ignore

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

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open positions"""
        with self._mt5_lock:  # Thread-safe MT5 access
            if symbol:
                positions = mt5.positions_get(symbol=symbol)  # type: ignore
            else:
                positions = mt5.positions_get()  # type: ignore

            if positions is None:
                return []

            return [position._asdict() for position in positions]

    def get_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get pending orders"""
        with self._mt5_lock:  # Thread-safe MT5 access
            if symbol:
                orders = mt5.orders_get(symbol=symbol)  # type: ignore
            else:
                orders = mt5.orders_get()  # type: ignore

            if orders is None:
                return []

            return [order._asdict() for order in orders]

    def get_history_orders(self, date_from: datetime, date_to: datetime) -> List[Dict]:
        """Get historical orders"""
        with self._mt5_lock:  # Thread-safe MT5 access
            orders = mt5.history_orders_get(date_from, date_to)  # type: ignore

            if orders is None:
                return []

            return [order._asdict() for order in orders]

    def get_history_deals(self, date_from: datetime, date_to: datetime) -> List[Dict]:
        """Get historical deals"""
        with self._mt5_lock:  # Thread-safe MT5 access
            deals = mt5.history_deals_get(date_from, date_to)  # type: ignore

            if deals is None:
                return []

            return [deal._asdict() for deal in deals]

    def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        with self._mt5_lock:  # Thread-safe MT5 access
            info = mt5.account_info()  # type: ignore
            if info:
                return info._asdict()
            return None

    def get_terminal_info(self) -> Optional[Dict]:
        """Get terminal information"""
        with self._mt5_lock:  # Thread-safe MT5 access
            info = mt5.terminal_info()  # type: ignore
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
        with self._mt5_lock:  # Thread-safe MT5 access
            try:
                # Preferred: mt5.time_current() -> seconds since epoch (int)
                if hasattr(mt5, 'time_current'):
                    ts = mt5.time_current()  # type: ignore
                    if ts:
                        return datetime.fromtimestamp(ts, tz=timezone.utc)

                # Fallback: try terminal_info attributes
                info = mt5.terminal_info()  # type: ignore
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
                acc = mt5.account_info()  # type: ignore
                if acc is not None and hasattr(acc, 'timestamp'):
                    try:
                        ts = getattr(acc, 'timestamp')
                        if isinstance(ts, (int, float)) and ts > 0:
                            return datetime.fromtimestamp(ts, tz=timezone.utc)
                    except Exception:
                        pass

                # Final fallback: use symbol tick time (server time)
                try:
                    tick = mt5.symbol_info_tick('EURUSD')  # type: ignore
                    if tick and hasattr(tick, 'time') and tick.time > 0:
                        return datetime.fromtimestamp(tick.time, tz=timezone.utc)
                except Exception:
                    pass

            except Exception as e:
                self.logger.debug(f"get_server_time error: {e}")

            return None

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:
        """Enable or disable a symbol for trading"""
        with self._mt5_lock:  # Thread-safe MT5 access
            return mt5.symbol_select(symbol, enable)  # type: ignore

    def market_book_add(self, symbol: str) -> bool:
        """Subscribe to market depth"""
        with self._mt5_lock:  # Thread-safe MT5 access
            return mt5.market_book_add(symbol)  # type: ignore

    def market_book_release(self, symbol: str) -> bool:
        """Unsubscribe from market depth"""
        with self._mt5_lock:  # Thread-safe MT5 access
            return mt5.market_book_release(symbol)  # type: ignore