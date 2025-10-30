﻿# Fixed MarketDataManager - No More Errors!
import logging
import MetaTrader5 as mt5
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

class MarketDataManager:
    """"Fixed market data manager with all required methods"""

    def __init__(self, mt5_connector=None, config=None):
        self.mt5_connector = mt5_connector
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.mt5_connected = False
        self.cached_data = {}
        self.last_update = None

    def initialize(self):
        """"Initialize the market data manager"""
        try:
            if not mt5.initialize():
                self.logger.warning('MT5 initialization failed')
                self.mt5_connected = False
            else:
                self.mt5_connected = True
                self.logger.info('MarketDataManager initialized')
        except Exception as e:
            self.logger.error(f'Error initializing MT5: {e}')
            self.mt5_connected = False
        return self.mt5_connected

    def get_latest_data(self, symbol: str) -> Optional[Dict]:
        """"Get latest market data for a symbol (FIXED METHOD)"""
        try:
            if not self.mt5_connected:
                self.initialize()

            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.warning(f'Symbol {symbol} not found')
                return None

            # Get latest tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                self.logger.warning(f'No tick data for {symbol}')
                return None

            # Get recent bars for additional data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 10)
            if rates is None or len(rates) == 0:
                self.logger.warning(f'No rate data for {symbol}')
                return None

            # Convert to dict
            latest_bar = rates[-1]
            data = {
                'symbol': symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': tick.ask - tick.bid,
                'volume': tick.volume,
                'time': datetime.fromtimestamp(tick.time),
                'open': latest_bar['open'],
                'high': latest_bar['high'],
                'low': latest_bar['low'],
                'close': latest_bar['close'],
                'tick_volume': latest_bar['tick_volume'],
                'real_volume': latest_bar['real_volume'] if 'real_volume' in latest_bar.dtype.names else 0
            }

            self.cached_data[symbol] = data
            self.last_update = datetime.now()
            return data

        except Exception as e:
            self.logger.error(f'Error getting data for {symbol}: {e}')
            return None

    def get_bars(self, symbol: str, timeframe: int, count: int = 100) -> Optional[pd.DataFrame]:
        """"Get historical bars for a symbol"""
        try:
            if not self.mt5_connected:
                self.initialize()

            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None:
                self.logger.warning(f'No historical data for {symbol}')
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Add volume column for compatibility (use tick_volume as volume)
            if 'tick_volume' in df.columns:
                df['volume'] = df['tick_volume']
            else:
                df['volume'] = 1  # fallback
            
            return df

        except Exception as e:
            self.logger.error(f'Error getting bars for {symbol}: {e}')
            return None

    def get_symbols(self) -> List[str]:
        """"Get list of available symbols"""
        try:
            if not self.mt5_connected:
                self.initialize()

            symbols = mt5.symbols_get()
            if symbols is None:
                return []
            return [s.name for s in symbols]

        except Exception as e:
            self.logger.error(f'Error getting symbols: {e}')
            return []

    def is_connected(self) -> bool:
        """"Check if MT5 is connected"""
        return self.mt5_connected

    def disconnect(self):
        """"Disconnect from MT5"""
        try:
            mt5.shutdown()
            self.mt5_connected = False
            self.logger.info('MarketDataManager disconnected')
        except Exception as e:
            self.logger.error(f'Error disconnecting: {e}')
