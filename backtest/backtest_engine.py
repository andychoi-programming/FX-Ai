"""
Backtest Engine Module
Simulates trading using historical data and existing trading system components
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import os
import sys

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mt5_connector import MT5Connector
from data.market_data_manager import MarketDataManager
from ai.ml_predictor import MLPredictor
from analysis.technical_analyzer import TechnicalAnalyzer
from core.risk_manager import RiskManager
from backtest.backtest_config import BacktestConfig
from backtest.performance_metrics import PerformanceMetrics

class Trade:
    """Represents a single trade in the backtest"""

    def __init__(self, symbol: str, direction: str, open_price: float,
                 stop_loss: float, take_profit: float, volume: float,
                 open_time: datetime, confidence: float = 0.0):
        self.symbol = symbol
        self.direction = direction  # 'buy' or 'sell'
        self.open_price = open_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.volume = volume
        self.open_time = open_time
        self.close_time: Optional[datetime] = None
        self.close_price: Optional[float] = None
        self.pnl: Optional[float] = None
        self.confidence = confidence
        self.bars_held = 0
        self.status = 'open'  # 'open', 'closed', 'stopped'

class BacktestEngine:
    """Main backtest engine that simulates trading"""

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine

        Args:
            config: Backtest configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize and connect to MT5
        self.mt5 = MT5Connector()
        if not self.mt5.connect():
            raise ConnectionError("Failed to connect to MT5. Please ensure MT5 is running and properly configured.")

        # Initialize components (similar to main.py)
        self.market_data = MarketDataManager(self.mt5, config.to_dict())
        self.ml_predictor = MLPredictor(config.to_dict())
        self.technical_analyzer = TechnicalAnalyzer(config.to_dict())
        self.risk_manager = RiskManager(config.to_dict())

        # Store timeframe string for ML predictions
        self.timeframe_string = config.get_timeframe_string()

        # Backtest state
        self.current_capital = config.initial_capital
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.current_time: Optional[datetime] = None

        # Historical data cache
        self.historical_data: Dict[str, pd.DataFrame] = {}

        # Performance tracking
        self.equity_curve: List[Tuple[datetime, float]] = [(config.start_date, config.initial_capital)]

    def load_historical_data(self) -> bool:
        """Load historical data for all symbols"""
        self.logger.info("Loading historical data for backtest...")

        # Verify MT5 connection
        if not self.mt5.connected:
            self.logger.error("MT5 is not connected. Cannot load historical data.")
            return False

        timeframe = self.config.get_mt5_timeframe()

        for symbol in self.config.symbols:
            try:
                # Check if symbol is available
                symbol_info = self.mt5.get_symbol_info(symbol)
                if symbol_info is None:
                    self.logger.warning(f"Symbol {symbol} is not available in MT5")
                    continue

                self.logger.info(f"Symbol {symbol} is available")

                # Get historical data
                data = self.market_data.get_bars(symbol, timeframe,
                    count=self.config.max_historical_bars if hasattr(self.config, 'max_historical_bars') else 10000)

                if data is not None and len(data) > 0:
                    df = pd.DataFrame(data)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)

                    self.logger.info(f"Retrieved {len(df)} bars for {symbol}, date range: {df.index.min()} to {df.index.max()}")

                    # Filter by date range
                    mask = (df.index >= self.config.start_date) & (df.index <= self.config.end_date)
                    df_filtered = df[mask]

                    self.logger.info(f"After date filtering: {len(df_filtered)} bars for {symbol}")

                    if not df_filtered.empty:
                        self.historical_data[symbol] = df_filtered
                        self.logger.info(f"Loaded {len(df_filtered)} bars for {symbol}")
                    else:
                        self.logger.warning(f"No data in date range {self.config.start_date} to {self.config.end_date} for {symbol}")
                else:
                    self.logger.warning(f"No historical data available for {symbol} from MT5")

            except Exception as e:
                self.logger.error(f"Error loading data for {symbol}: {e}")
                return False

        return len(self.historical_data) > 0

    def run_backtest(self) -> pd.DataFrame:
        """Run the complete backtest"""
        self.logger.info("Starting backtest...")

        if not self.load_historical_data():
            self.logger.error("Failed to load historical data")
            return pd.DataFrame()

        # Get all unique timestamps across all symbols
        all_timestamps = set()
        for symbol_data in self.historical_data.values():
            all_timestamps.update(symbol_data.index)

        sorted_timestamps = sorted(all_timestamps)

        self.logger.info(f"Running backtest from {sorted_timestamps[0]} to {sorted_timestamps[-1]}")

        # Process each timestamp
        for current_time in sorted_timestamps:
            self.current_time = current_time
            self._process_timestamp(current_time)

            # Update equity curve
            self.equity_curve.append((current_time, self.current_capital))

        # Close any remaining open trades at the end
        self._close_all_trades(sorted_timestamps[-1])

        # Convert trades to DataFrame
        trades_data = []
        for trade in self.closed_trades:
            trades_data.append({
                'symbol': trade.symbol,
                'direction': trade.direction,
                'open_price': trade.open_price,
                'close_price': trade.close_price,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'volume': trade.volume,
                'open_time': trade.open_time,
                'close_time': trade.close_time,
                'pnl': trade.pnl,
                'confidence': trade.confidence,
                'bars_held': trade.bars_held,
                'status': trade.status
            })

        trades_df = pd.DataFrame(trades_data)

        self.logger.info(f"Backtest completed. {len(trades_df)} trades executed.")
        return trades_df

    def _process_timestamp(self, timestamp: datetime):
        """Process a single timestamp in the backtest"""
        # Check for stop loss/take profit hits
        self._check_stop_loss_take_profit(timestamp)

        # Generate signals for each symbol
        for symbol in self.config.symbols:
            if symbol not in self.historical_data:
                continue

            symbol_data = self.historical_data[symbol]

            # Get current bar
            if timestamp not in symbol_data.index:
                continue

            current_bar = symbol_data.loc[timestamp]

            # Generate signals
            self._generate_signals(symbol, current_bar, timestamp)

    def _generate_signals(self, symbol: str, current_bar: pd.Series, timestamp: datetime):
        """Generate trading signals for a symbol"""
        try:
            # Get recent data for analysis (last 100 bars)
            recent_data = self._get_recent_data(symbol, timestamp, 100)
            if recent_data is None or len(recent_data) < 50:
                return

            # Get technical signals
            technical_signals = self.technical_analyzer.analyze_symbol(symbol, recent_data)

            # Get ML prediction
            ml_signal = self.ml_predictor.predict_signal(symbol, recent_data, technical_signals, self.timeframe_string)

            # Check risk management
            risk_check = self.risk_manager.check_trade_risk(symbol, ml_signal, self.current_capital)

            if not risk_check['approved']:
                return

            # Check confidence threshold
            if ml_signal['confidence'] < self.config.min_confidence:
                return

            # Check maximum open positions
            if len(self.open_trades) >= self.config.max_open_positions:
                return

            # Calculate position size
            position_size = self._calculate_position_size(symbol, current_bar['close'], risk_check['max_volume'])

            if position_size <= 0:
                return

            # Calculate stop loss and take profit
            if ml_signal['direction'] == 'bullish':
                stop_loss = current_bar['close'] - (self.config.stop_loss_pips * self._get_pip_value(symbol))
                take_profit = current_bar['close'] + (self.config.take_profit_pips * self._get_pip_value(symbol))
                direction = 'buy'
            else:  # bearish
                stop_loss = current_bar['close'] + (self.config.stop_loss_pips * self._get_pip_value(symbol))
                take_profit = current_bar['close'] - (self.config.take_profit_pips * self._get_pip_value(symbol))
                direction = 'sell'

            # Open trade
            trade = Trade(
                symbol=symbol,
                direction=direction,
                open_price=current_bar['close'],
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume=position_size,
                open_time=timestamp,
                confidence=ml_signal['confidence']
            )

            self.open_trades.append(trade)

            if self.config.enable_detailed_logging:
                self.logger.info(f"Opened {direction} trade for {symbol} at {current_bar['close']:.5f}, "
                               f"SL: {stop_loss:.5f}, TP: {take_profit:.5f}")

        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")

    def _check_stop_loss_take_profit(self, timestamp: datetime):
        """Check if any open trades hit stop loss or take profit"""
        trades_to_close = []

        for trade in self.open_trades:
            if trade.symbol not in self.historical_data:
                continue

            symbol_data = self.historical_data[trade.symbol]

            # Get current price
            if timestamp not in symbol_data.index:
                continue

            current_price = symbol_data.loc[timestamp, 'close']
            trade.bars_held += 1

            # Check stop loss and take profit
            if trade.direction == 'buy':
                if current_price <= trade.stop_loss:
                    self._close_trade(trade, current_price, timestamp, 'stopped')
                    trades_to_close.append(trade)
                elif current_price >= trade.take_profit:
                    self._close_trade(trade, current_price, timestamp, 'profit')
                    trades_to_close.append(trade)
            else:  # sell
                if current_price >= trade.stop_loss:
                    self._close_trade(trade, current_price, timestamp, 'stopped')
                    trades_to_close.append(trade)
                elif current_price <= trade.take_profit:
                    self._close_trade(trade, current_price, timestamp, 'profit')
                    trades_to_close.append(trade)

        # Remove closed trades
        for trade in trades_to_close:
            self.open_trades.remove(trade)

    def _close_trade(self, trade: Trade, close_price: float, close_time: datetime, reason: str):
        """Close a trade and calculate P&L"""
        trade.close_price = close_price
        trade.close_time = close_time
        trade.status = reason

        # Calculate P&L
        pip_value = self._get_pip_value(trade.symbol)
        pips = (close_price - trade.open_price) / pip_value if trade.direction == 'buy' else (trade.open_price - close_price) / pip_value
        trade.pnl = pips * trade.volume * pip_value

        # Update capital
        self.current_capital += trade.pnl

        # Move to closed trades
        self.closed_trades.append(trade)

        if self.config.enable_detailed_logging:
            self.logger.info(f"Closed {trade.direction} trade for {trade.symbol}: "
                           f"P&L = ${trade.pnl:.2f} ({reason})")

    def _close_all_trades(self, timestamp: datetime):
        """Close all remaining open trades at market price"""
        for trade in self.open_trades[:]:  # Copy list to avoid modification during iteration
            if trade.symbol in self.historical_data and timestamp in self.historical_data[trade.symbol].index:
                close_price = self.historical_data[trade.symbol].loc[timestamp, 'close']
                self._close_trade(trade, close_price, timestamp, 'end_of_test')
            else:
                # Close at open price if no data available
                self._close_trade(trade, trade.open_price, timestamp, 'end_of_test')

    def _get_recent_data(self, symbol: str, timestamp: datetime, bars: int) -> Optional[pd.DataFrame]:
        """Get recent historical data up to timestamp"""
        if symbol not in self.historical_data:
            return None

        symbol_data = self.historical_data[symbol]
        mask = symbol_data.index <= timestamp
        recent_data = symbol_data[mask].tail(bars)

        return recent_data if not recent_data.empty else None

    def _calculate_position_size(self, symbol: str, entry_price: float, max_volume: float) -> float:
        """Calculate position size based on risk management"""
        risk_amount = self.current_capital * self.config.max_risk_per_trade
        pip_value = self._get_pip_value(symbol)
        stop_distance_pips = self.config.stop_loss_pips

        # Position size = Risk Amount / (Stop Distance * Pip Value)
        position_size = risk_amount / (stop_distance_pips * pip_value)

        # Apply maximum volume limit
        position_size = min(position_size, max_volume)

        # Minimum position size (0.01 lots)
        position_size = max(position_size, 0.01)

        return round(position_size, 2)

    def _get_pip_value(self, symbol: str) -> float:
        """Get pip value for symbol (simplified)"""
        # For forex pairs, pip value depends on lot size and currency
        # Simplified: assume 0.0001 for most pairs, 0.01 for JPY pairs
        if 'JPY' in symbol:
            return 0.01
        else:
            return 0.0001

    def save_results(self, trades_df: pd.DataFrame):
        """Save backtest results to files"""
        try:
            # Save trades to CSV
            if self.config.save_trades_to_csv and not trades_df.empty:
                trades_df.to_csv(self.config.trades_csv_path, index=False)
                self.logger.info(f"Trades saved to {self.config.trades_csv_path}")

            # Generate performance report
            if not trades_df.empty:
                metrics = PerformanceMetrics(trades_df, self.config.initial_capital)
                report = metrics.generate_report()

                with open(self.config.performance_report_path, 'w') as f:
                    f.write(report)

                self.logger.info(f"Performance report saved to {self.config.performance_report_path}")

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")