import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import joblib
from typing import Dict, List, Tuple
from live_trading.dynamic_parameter_manager import DynamicParameterManager
from ai.ml_predictor import MLPredictor

class MLBacktester:
    """Backtest the ML trading system with optimized parameters"""

    def __init__(self, config: Dict):
        self.config = config
        self.symbols = config['trading']['symbols']
        self.param_manager = DynamicParameterManager(config)

        self.logger = logging.getLogger(__name__)

        # Initialize ML Predictor (loads all timeframe models)
        self.ml_predictor = MLPredictor(config)

        # Backtest settings
        self.initial_balance = 10000
        self.spread_pips = 2  # 2 pips spread
        self.commission_per_lot = 5  # $5 per lot round trip

        # Risk management settings
        self.max_daily_trades_per_symbol = 2  # Maximum trades per symbol per day (reduced)
        self.min_trade_interval_hours = 8  # Minimum hours between trades for same symbol (increased)
        self.max_position_size_pct = 0.05  # Maximum position size as % of account balance

        # Results tracking
        self.results = {
            'balance': self.initial_balance,
            'trades': [],
            'daily_pnl': {},
            'symbol_performance': {}
        }

    def load_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical data for backtesting"""
        try:
            # In a real implementation, you'd load from MT5 or CSV files
            # For now, we'll create synthetic data based on the symbol
            date_range = pd.date_range(start=start_date, end=end_date, freq='H')

            np.random.seed(hash(symbol) % 2**32)  # Deterministic seed per symbol

            # Generate realistic price movements
            base_price = self._get_base_price(symbol)
            prices = [base_price]

            for i in range(1, len(date_range)):
                # Random walk with mean reversion
                change = np.random.normal(0, 0.001)  # 0.1% volatility per hour
                change += (base_price - prices[-1]) * 0.01  # Mean reversion
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 0.0001))  # Prevent negative prices

            # Create OHLC data
            df = pd.DataFrame({
                'time': date_range,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.0005))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.0005))) for p in prices],
                'close': prices[1:] + [prices[-1]],
                'tick_volume': np.random.randint(1000, 10000, len(date_range))
            })

            df['close'] = df['close'].shift(-1).fillna(df['close'].iloc[-1])

            # Calculate technical indicators
            df = self._calculate_indicators(df)

            return df.dropna()

        except Exception as e:
            self.logger.error(f"Failed to load data for {symbol}: {e}")
            return pd.DataFrame()

    def _get_base_price(self, symbol: str) -> float:
        """Get base price for synthetic data generation"""
        base_prices = {
            'EURUSD': 1.0850, 'GBPUSD': 1.2750, 'AUDUSD': 0.6650,
            'USDJPY': 147.50, 'EURJPY': 160.00, 'GBPJPY': 188.00,
            'AUDJPY': 98.00, 'USDCAD': 1.3550, 'EURCAD': 1.4700,
            'GBPCAD': 1.7300, 'AUDCAD': 0.9000, 'USDCHF': 0.9150,
            'EURCHF': 0.9950, 'GBPCHF': 1.1700, 'AUDCHF': 0.6100,
            'NZDUSD': 0.6150, 'EURNZD': 1.7650, 'GBPNZD': 2.0750,
            'AUDNZD': 1.0800, 'NZDJPY': 90.50, 'CADJPY': 109.00,
            'CHFJPY': 161.00, 'NZDCAD': 0.8350, 'CADCHF': 0.6750,
            'NZDCHF': 0.5650
        }
        return base_prices.get(symbol, 1.0)

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for backtesting"""
        # Same indicators as in the trading system
        df['SMA_20'] = df['close'].rolling(20).mean()
        df['SMA_50'] = df['close'].rolling(50).mean()
        df['EMA_12'] = df['close'].ewm(span=12).mean()
        df['EMA_26'] = df['close'].ewm(span=26).mean()

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        df['BB_middle'] = df['close'].rolling(20).mean()
        df['BB_std'] = df['close'].rolling(20).std()
        df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)

        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()

        return df.dropna()

    def get_ml_prediction(self, symbol: str, current_data: pd.DataFrame, data_point: pd.Series, timeframe: str = 'H1') -> Tuple[int, float]:
        """Get ML prediction for a specific data point"""
        try:
            # Use the current window of data for feature preparation
            # Get the last 50 rows up to current point
            if len(current_data) >= 50:
                window_data = current_data.tail(50)
            else:
                window_data = current_data
            
            # Create technical signals from the data_point
            technical_signals = {
                'rsi': data_point.get('RSI', 50),
                'macd': data_point.get('MACD', 0),
                'bb_upper': data_point.get('BB_upper', data_point.get('close', 0)),
                'bb_lower': data_point.get('BB_lower', data_point.get('close', 0)),
                'ema_9': data_point.get('EMA_12', data_point.get('close', 0)),  # Use EMA_12 as approximation
                'ema_21': data_point.get('EMA_26', data_point.get('close', 0)),  # Use EMA_26 as approximation
                'vwap': data_point.get('close', 0)  # Mock VWAP
            }

            # Get prediction from MLPredictor
            ml_signal = self.ml_predictor.predict_signal(symbol, window_data, technical_signals, timeframe)
            
            # Convert to the expected format (1 for bullish, 0 for bearish)
            direction = 1 if ml_signal['direction'] == 'bullish' else 0
            confidence = ml_signal['confidence']

            return direction, confidence

        except Exception as e:
            self.logger.error(f"ML prediction failed for {symbol}: {e}")
            return 0, 0.0

    def run_backtest(self, start_date: str, end_date: str, timeframe: str = 'H1'):
        """Run backtest for all symbols"""
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")

        for symbol in self.symbols:
            if not self.param_manager.should_trade_symbol(symbol, timeframe):
                self.logger.info(f"Skipping {symbol} - not recommended for trading")
                continue

            self.logger.info(f"Backtesting {symbol}...")
            self._backtest_symbol(symbol, start_date, end_date, timeframe)

        self._calculate_final_results()

    def _backtest_symbol(self, symbol: str, start_date: str, end_date: str, timeframe: str):
        """Backtest individual symbol"""
        print(f"Starting backtest for {symbol}...")

        # Load historical data
        df = self.load_historical_data(symbol, start_date, end_date)
        if df.empty:
            print(f"No data for {symbol}")
            return

        print(f"Loaded {len(df)} data points for {symbol}")

        # Get optimal parameters
        params = self.param_manager.get_optimal_parameters(symbol, timeframe)
        entry_start, entry_end = self.param_manager.get_entry_time_windows(symbol, timeframe)
        exit_hour = self.param_manager.get_exit_time(symbol, timeframe)

        # Trading state
        position = None
        entry_price = 0
        position_size = 0
        entry_time = None
        trade_count = 0

        # Breakeven and trailing stop state
        breakeven_activated = False
        trailing_activated = False
        current_sl_price = 0
        highest_price = 0  # For trailing stop

        # Trade frequency control
        daily_trades = {}  # date -> trade count for this symbol
        last_trade_time = None

        for idx, row in df.iterrows():
            current_time = row['time']
            current_hour = current_time.hour
            current_day = current_time.strftime('%A')  # Get day name

            # Check entry conditions
            if position is None:
                # Check trading hours with day-specific adjustments
                base_start = entry_start
                base_end = entry_end

                # Apply Monday entry delay
                if current_day == 'Monday':
                    monday_delay = params.get('monday_entry_delay', base_start)
                    effective_start = max(base_start, monday_delay)
                else:
                    effective_start = base_start

                # Apply Friday early exit constraint (don't enter late Friday)
                if current_day == 'Friday':
                    friday_early_exit = params.get('friday_early_exit', 18)
                    effective_end = min(base_end, friday_early_exit)
                else:
                    effective_end = base_end

                if effective_start <= current_hour <= effective_end:
                    # Check day-of-week entry restrictions
                    best_entry_days = params.get('best_entry_days', ['All'])
                    if best_entry_days != ['All'] and current_day not in best_entry_days:
                        continue
                    # Check trade frequency limits
                    current_date = current_time.date().isoformat()
                    daily_count = daily_trades.get(current_date, 0)
                    
                    # Check daily trade limit
                    if daily_count >= self.max_daily_trades_per_symbol:
                        continue
                        
                    # Check minimum time between trades
                    if last_trade_time is not None:
                        hours_since_last_trade = (current_time - last_trade_time).total_seconds() / 3600
                        if hours_since_last_trade < self.min_trade_interval_hours:
                            continue
                    
                    # Get data up to current point for ML prediction
                    current_window = df.iloc[:idx+1] if idx >= 49 else df.iloc[:idx+1]
                    prediction, confidence = self.get_ml_prediction(symbol, current_window, row, timeframe)

                    if confidence >= 0.6:  # Increased confidence threshold for better selectivity
                        direction = 1 if prediction == 1 else -1
                        position_size = self._calculate_position_size(symbol, params['sl_pips'])

                        # Open position
                        if direction == 1:  # Buy
                            entry_price = row['open'] + (self.spread_pips * 0.00001)
                        else:  # Sell
                            entry_price = row['open'] - (self.spread_pips * 0.00001)

                        position = direction
                        entry_time = current_time
                        last_trade_time = current_time
                        daily_trades[current_date] = daily_count + 1
                        trade_count += 1

                        # Initialize breakeven and trailing stop tracking
                        breakeven_activated = False
                        trailing_activated = False
                        highest_price = entry_price if direction == 1 else entry_price  # Initialize for trailing

            # Check exit conditions
            elif position is not None:
                # Time-based exit with day-of-week restrictions
                avoid_exit_days = params.get('avoid_exit_days', ['None'])

                # Apply Friday early exit
                if current_day == 'Friday':
                    friday_early_exit = params.get('friday_early_exit', 18)
                    effective_exit_hour = min(exit_hour, friday_early_exit)
                else:
                    effective_exit_hour = exit_hour

                if current_hour >= effective_exit_hour:
                    # Check if we should avoid exiting on certain days
                    if avoid_exit_days != ['None'] and current_day in avoid_exit_days:
                        pass  # Don't exit on avoid days
                    else:
                        self._close_position(symbol, position, entry_price, row, "Time exit", params, position_size)
                        position = None
                        continue

                # Stop loss / Take profit with breakeven and trailing stops
                current_price = row['close']
                sl_pips, tp_pips = params['sl_pips'], params['tp_pips']
                breakeven_trigger = params.get('breakeven_trigger', 15)
                trailing_activation = params.get('trailing_activation', 20)
                trailing_distance = params.get('trailing_distance', 15)

                pip_value = 0.0001 if symbol.endswith('JPY') else 0.00001

                if position == 1:  # Buy
                    # Update highest price for trailing stop
                    if row['high'] > highest_price:
                        highest_price = row['high']

                    # Check breakeven activation
                    if not breakeven_activated and (highest_price - entry_price) >= (breakeven_trigger * pip_value):
                        breakeven_activated = True

                    # Check trailing stop activation
                    if not trailing_activated and (highest_price - entry_price) >= (trailing_activation * pip_value):
                        trailing_activated = True

                    # Calculate current stop loss (breakeven or trailing)
                    if breakeven_activated:
                        current_sl_price = entry_price
                    elif trailing_activated:
                        current_sl_price = highest_price - (trailing_distance * pip_value)
                    else:
                        current_sl_price = entry_price - (sl_pips * pip_value)

                    tp_price = entry_price + (tp_pips * pip_value)

                    if current_price <= current_sl_price:
                        exit_reason = "Trailing Stop" if trailing_activated else "Stop Loss" if not breakeven_activated else "Breakeven Stop"
                        self._close_position(symbol, position, entry_price, row, exit_reason, params, position_size)
                        position = None
                    elif current_price >= tp_price:
                        self._close_position(symbol, position, entry_price, row, "Take Profit", params, position_size)
                        position = None

                else:  # Sell
                    # Update lowest price for trailing stop (inverse for sell)
                    if row['low'] < highest_price:
                        highest_price = row['low']

                    # Check breakeven activation
                    if not breakeven_activated and (entry_price - highest_price) >= (breakeven_trigger * pip_value):
                        breakeven_activated = True

                    # Check trailing stop activation
                    if not trailing_activated and (entry_price - highest_price) >= (trailing_activation * pip_value):
                        trailing_activated = True

                    # Calculate current stop loss (breakeven or trailing)
                    if breakeven_activated:
                        current_sl_price = entry_price
                    elif trailing_activated:
                        current_sl_price = highest_price + (trailing_distance * pip_value)
                    else:
                        current_sl_price = entry_price + (sl_pips * pip_value)

                    tp_price = entry_price - (tp_pips * pip_value)

                    if current_price >= current_sl_price:
                        exit_reason = "Trailing Stop" if trailing_activated else "Stop Loss" if not breakeven_activated else "Breakeven Stop"
                        self._close_position(symbol, position, entry_price, row, exit_reason, params, position_size)
                        position = None
                    elif current_price <= tp_price:
                        self._close_position(symbol, position, entry_price, row, "Take Profit", params, position_size)
                        position = None

        # Close any remaining position at end
        if position is not None:
            last_row = df.iloc[-1]
            self._close_position(symbol, position, entry_price, last_row, "End of period", params, position_size)

    def _calculate_position_size(self, symbol: str, sl_pips: int) -> float:
        """Calculate position size based on risk"""
        risk_amount = self.config['trading']['risk_per_trade']
        
        # Pip value per standard lot (100,000 units)
        if symbol.endswith('JPY'):
            pip_value_per_lot = 1000  # ¥1000 per pip for JPY pairs
        else:
            pip_value_per_lot = 10     # $10 per pip for other pairs
        
        # Position size in lots: risk / (stop_loss_pips * pip_value_per_lot)
        position_size = risk_amount / (sl_pips * pip_value_per_lot)
        
        # Apply maximum position size limit (as % of account balance)
        # Max position size ensures max loss <= max_position_value
        max_position_value = self.results['balance'] * self.max_position_size_pct
        max_position_size_from_balance = max_position_value / (sl_pips * pip_value_per_lot)
        position_size = min(position_size, max_position_size_from_balance)
        
        return max(0.01, round(position_size, 2))

    def _close_position(self, symbol: str, direction: int, entry_price: float,
                       exit_row: pd.Series, exit_reason: str, params: Dict, position_size: float):
        """Close a position and record the trade"""
        exit_price = exit_row['close']
        exit_time = exit_row['time']

        # Calculate P&L
        pip_value = 0.0001 if symbol.endswith('JPY') else 0.00001
        pips = (exit_price - entry_price) / pip_value * direction
        
        # Pip value per standard lot (100,000 units)
        pip_value_per_lot = 1000 if symbol.endswith('JPY') else 10
        pnl = pips * pip_value_per_lot * position_size
        pnl -= self.commission_per_lot * position_size  # Commission scales with position size

        # Record trade
        trade = {
            'symbol': symbol,
            'direction': 'BUY' if direction == 1 else 'SELL',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': str(exit_time),
            'exit_time': str(exit_time),
            'pips': pips,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'params': params,
            'position_size': position_size
        }

        self.results['trades'].append(trade)
        self.results['balance'] += pnl

        # Update daily P&L
        date_key = exit_time.date().isoformat()
        if date_key not in self.results['daily_pnl']:
            self.results['daily_pnl'][date_key] = 0
        self.results['daily_pnl'][date_key] += pnl

        # Update symbol performance
        if symbol not in self.results['symbol_performance']:
            self.results['symbol_performance'][symbol] = {
                'trades': 0, 'wins': 0, 'losses': 0, 'total_pnl': 0
            }

        self.results['symbol_performance'][symbol]['trades'] += 1
        self.results['symbol_performance'][symbol]['total_pnl'] += pnl

        if pnl > 0:
            self.results['symbol_performance'][symbol]['wins'] += 1
        else:
            self.results['symbol_performance'][symbol]['losses'] += 1

    def _calculate_final_results(self):
        """Calculate final backtest results"""
        if not self.results['trades']:
            self.logger.warning("No trades executed during backtest")
            return

        trades_df = pd.DataFrame(self.results['trades'])

        # Calculate metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean()
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean()
        profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() /
                           trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else float('inf')

        # Calculate drawdown
        cumulative = trades_df['pnl'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = running_max - cumulative
        max_drawdown = drawdown.max()

        self.results['metrics'] = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win if winning_trades > 0 else 0,
            'avg_loss': avg_loss if losing_trades > 0 else 0,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'final_balance': self.results['balance']
        }

    def print_results(self):
        """Print comprehensive backtest results"""
        if 'metrics' not in self.results:
            print("No backtest results available")
            return

        m = self.results['metrics']

        print("\n" + "="*60)
        print("ML TRADING SYSTEM BACKTEST RESULTS")
        print("="*60)
        print(f"Initial Balance: ${self.initial_balance:.2f}")
        print(f"Final Balance: ${m['final_balance']:.2f}")
        print(f"Total P&L: ${m['total_pnl']:.2f}")
        print(f"Total Trades: {m['total_trades']}")
        print(f"Win Rate: {m['win_rate']:.1%}")
        print(f"Average Win: ${m['avg_win']:.2f}")
        print(f"Average Loss: ${m['avg_loss']:.2f}")
        print(f"Profit Factor: {m['profit_factor']:.2f}")
        print(f"Max Drawdown: ${m['max_drawdown']:.2f}")

        print(f"\nTop 5 Performing Symbols:")
        symbol_perf = self.results['symbol_performance']
        sorted_symbols = sorted(symbol_perf.items(),
                              key=lambda x: x[1]['total_pnl'], reverse=True)

        for symbol, perf in sorted_symbols[:5]:
            win_rate = perf['wins'] / perf['trades'] if perf['trades'] > 0 else 0
            print(f"  {symbol}: ${perf['total_pnl']:.2f} P&L, "
                  f"{win_rate:.1%} win rate, {perf['trades']} trades")

        print(f"\nWorst 5 Performing Symbols:")
        for symbol, perf in sorted_symbols[-5:]:
            win_rate = perf['wins'] / perf['trades'] if perf['trades'] > 0 else 0
            print(f"  {symbol}: ${perf['total_pnl']:.2f} P&L, "
                  f"{win_rate:.1%} win rate, {perf['trades']} trades")

        print("="*60)

    def save_results(self, filename: str = "backtest_results.json"):
        """Save backtest results to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            self.logger.info(f"Backtest results saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")

    def print_detailed_symbol_analysis(self):
        """Print comprehensive analysis of each symbol's performance"""
        if not self.results or 'symbol_performance' not in self.results:
            print("No symbol performance data available")
            return

        symbol_perf = self.results['symbol_performance']

        print("\n" + "="*80)
        print("COMPREHENSIVE 3-YEAR SYMBOL ANALYSIS")
        print("="*80)

        # Sort symbols by total P&L
        sorted_symbols = sorted(symbol_perf.items(),
                              key=lambda x: x[1]['total_pnl'], reverse=True)

        print("<8")
        print("-" * 80)

        for symbol, perf in sorted_symbols:
            trades = perf['trades']
            wins = perf['wins']
            losses = perf['losses']
            total_pnl = perf['total_pnl']

            if trades > 0:
                win_rate = wins / trades
                avg_win = (total_pnl / trades) if trades > 0 else 0
                print("<8")

        print("\n" + "="*80)
        print("BEST SETTINGS SUMMARY")
        print("="*80)

        # Find best and worst performing symbols (handle case with fewer than 5 symbols)
        num_symbols = len(sorted_symbols)
        num_to_show = min(5, num_symbols)

        print(f"\n🏆 TOP {num_to_show} BEST PERFORMING SYMBOLS:")
        for i, (symbol, perf) in enumerate(sorted_symbols[:num_to_show], 1):
            trades = perf['trades']
            if trades > 0:
                win_rate = perf['wins'] / trades
                print(f"{i}. {symbol}: ${perf['total_pnl']:.2f} P&L, "
                      f"{win_rate:.1%} win rate, {trades} trades")

        print(f"\n📉 TOP {num_to_show} WORST PERFORMING SYMBOLS:")
        for i, (symbol, perf) in enumerate(sorted_symbols[-num_to_show:], 1):
            trades = perf['trades']
            if trades > 0:
                win_rate = perf['wins'] / trades
                print(f"{i}. {symbol}: ${perf['total_pnl']:.2f} P&L, "
                      f"{win_rate:.1%} win rate, {trades} trades")

        # Calculate overall statistics
        total_trades = sum(perf['trades'] for perf in symbol_perf.values())
        total_pnl = sum(perf['total_pnl'] for perf in symbol_perf.values())
        profitable_symbols = sum(1 for perf in symbol_perf.values() if perf['total_pnl'] > 0)
        active_symbols = sum(1 for perf in symbol_perf.values() if perf['trades'] > 0)

        print("\n📊 OVERALL STATISTICS:")
        print(f"  Total Symbols: {len(symbol_perf)}")
        print(f"  Active Symbols: {active_symbols}")
        print(f"  Profitable Symbols: {profitable_symbols}")
        print(f"  Total Trades: {total_trades}")
        print(".2f")
        print(".2f")

        if active_symbols > 0:
            avg_pnl_per_symbol = total_pnl / active_symbols
            print(".2f")

        print("\n" + "="*80)
        print("OPTIMAL TRADING SETTINGS RECOMMENDATIONS")
        print("="*80)

        # Provide recommendations based on analysis
        if profitable_symbols > 0:
            print(f"\n✅ RECOMMENDED SYMBOLS TO TRADE: {profitable_symbols}/{len(symbol_perf)} symbols profitable")
            print("Focus on the top performing symbols above for best results.")

        if total_pnl > 0:
            print("\n📈 STRATEGY SHOWS OVERALL PROFITABILITY")
            print("The ML-based approach with optimized parameters shows positive results.")
        else:
            print("\n⚠️  STRATEGY NEEDS FURTHER OPTIMIZATION")
            print("Consider retraining ML models with balanced buy/sell data.")

        print("\n🔧 KEY SETTINGS TO MAINTAIN:")
        print("  - Risk per trade: $50")
        print("  - Max positions: 3")
        print("  - Trading hours: 08:00-20:00 GMT")
        print("  - ML confidence threshold: 60%")
        print("  - Daily loss limit: $200")

        print("\n" + "="*80)

# Example usage
if __name__ == "__main__":
    # Load configuration
    with open("config/trading_config.json", 'r') as f:
        config = json.load(f)

    # Initialize backtester
    backtester = MLBacktester(config)

    # Run backtest for last 3 months
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")  # 3 months = 90 days
    end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Running comprehensive 3-month backtest from {start_date} to {end_date}")
    print("This may take several minutes...")

    backtester.run_backtest(start_date, end_date, 'H1')

    # Print and save results
    backtester.print_results()
    backtester.save_results("backtest_results_3year_ml_optimized.json")

    # Print detailed symbol analysis
    backtester.print_detailed_symbol_analysis()