import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, time
import logging
import json
from pathlib import Path
import joblib
from typing import Dict, Optional, Tuple, List
from live_trading.dynamic_parameter_manager import DynamicParameterManager

class MLTradingSystem:
    """Enhanced trading system combining ML predictions with optimized parameters"""

    def __init__(self, config: Dict):
        self.config = config
        self.symbols = config['trading']['symbols']
        self.timeframes = config['trading']['timeframes']
        self.param_manager = DynamicParameterManager(config)

        # Initialize MT5
        if not mt5.initialize():
            raise Exception("MT5 initialization failed")

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Load ML models
        self.models = {}
        self.scalers = {}
        self._load_ml_models()

        # Trading state
        self.active_positions = {}
        self.daily_pnl = {}
        self.session_start = datetime.now()

        self.logger.info("ML Trading System initialized")

    def _load_ml_models(self):
        """Load trained ML models and scalers for all symbols"""
        models_dir = Path("models")

        for symbol in self.symbols:
            model_file = models_dir / f"{symbol}_model.pkl"
            scaler_file = models_dir / f"{symbol}_scaler.pkl"

            if model_file.exists() and scaler_file.exists():
                try:
                    self.models[symbol] = joblib.load(model_file)
                    self.scalers[symbol] = joblib.load(scaler_file)
                    self.logger.info(f"Loaded ML model for {symbol}")
                except Exception as e:
                    self.logger.error(f"Failed to load model for {symbol}: {e}")
            else:
                self.logger.warning(f"Model files not found for {symbol}")

    def get_market_data(self, symbol: str, timeframe: str, bars: int = 100) -> pd.DataFrame:
        """Get historical market data for ML prediction"""
        timeframe_map = {
            'H1': mt5.TIMEFRAME_H1,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }

        mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)

        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
        if rates is None:
            self.logger.error(f"Failed to get rates for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Calculate technical indicators
        df = self._calculate_indicators(df)

        return df

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for ML features"""
        # Moving averages
        df['SMA_20'] = df['close'].rolling(20).mean()
        df['SMA_50'] = df['close'].rolling(50).mean()
        df['EMA_12'] = df['close'].ewm(span=12).mean()
        df['EMA_26'] = df['close'].ewm(span=26).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(20).mean()
        df['BB_std'] = df['close'].rolling(20).std()
        df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)

        # ATR for volatility
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()

        # Volume indicators (if available)
        if 'tick_volume' in df.columns:
            df['Volume_SMA'] = df['tick_volume'].rolling(20).mean()

        return df.dropna()

    def get_ml_prediction(self, symbol: str, timeframe: str = 'H1') -> Tuple[int, float]:
        """
        Get ML prediction for symbol
        Returns: (prediction, confidence)
        """
        if symbol not in self.models:
            return 0, 0.0  # Neutral if no model

        try:
            # Get recent market data
            df = self.get_market_data(symbol, timeframe, 100)
            if df.empty:
                return 0, 0.0

            # Get latest data point
            latest_data = df.iloc[-1:]

            # Prepare features (same as training)
            feature_columns = [
                'open', 'high', 'low', 'close', 'tick_volume',
                'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI',
                'MACD', 'MACD_signal', 'MACD_hist',
                'BB_middle', 'BB_upper', 'BB_lower', 'ATR'
            ]

            # Ensure all features exist
            available_features = [col for col in feature_columns if col in latest_data.columns]

            if len(available_features) < 10:  # Minimum features needed
                return 0, 0.0

            X = latest_data[available_features].values

            # Scale features
            X_scaled = self.scalers[symbol].transform(X)

            # Get prediction and probability
            prediction = self.models[symbol].predict(X_scaled)[0]
            probabilities = self.models[symbol].predict_proba(X_scaled)[0]

            # Get confidence (probability of predicted class)
            confidence = probabilities[prediction]

            return int(prediction), float(confidence)

        except Exception as e:
            self.logger.error(f"ML prediction failed for {symbol}: {e}")
            return 0, 0.0

    def should_enter_trade(self, symbol: str, timeframe: str = 'H1') -> Tuple[bool, int]:
        """
        Determine if we should enter a trade based on ML prediction and parameters
        Returns: (should_enter, direction) - direction: 1=buy, -1=sell, 0=neutral
        """
        # Check if symbol should be traded
        if not self.param_manager.should_trade_symbol(symbol, timeframe):
            return False, 0

        # Get ML prediction
        prediction, confidence = self.get_ml_prediction(symbol, timeframe)

        # Minimum confidence threshold from config
        min_confidence = self.config['ml']['min_confidence']
        if confidence < min_confidence:
            return False, 0

        # Get current day of week for day-of-week logic
        current_day = datetime.now().strftime('%A')

        # Get optimal entry time window with day-of-week adjustments
        start_hour, end_hour = self.param_manager.get_entry_time_windows(symbol, timeframe)
        current_hour = datetime.now().hour

        # Apply Monday entry delay
        monday_delay = self.param_manager.get_monday_entry_delay(symbol)
        if current_day == 'Monday' and current_hour < monday_delay:
            return False, 0

        # Apply Friday early exit (for entry decisions, avoid entering near early exit time)
        friday_early_exit = self.param_manager.get_friday_early_exit(symbol)
        if current_day == 'Friday' and current_hour >= friday_early_exit:
            return False, 0

        # Check best entry days
        best_entry_days = self.param_manager.get_best_entry_days(symbol)
        if best_entry_days and current_day not in best_entry_days:
            return False, 0

        # Check if within trading hours
        if not (start_hour <= current_hour <= end_hour):
            return False, 0

        # Convert prediction to direction (assuming 1=buy, 0=sell in training)
        direction = 1 if prediction == 1 else -1

        return True, direction

    def calculate_position_size(self, symbol: str, risk_amount: float = 100) -> float:
        """Calculate position size based on risk amount and current price"""
        try:
            # Get current price
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return 0.01  # Default lot size

            current_price = symbol_info.ask
            point = symbol_info.point
            tick_size = symbol_info.trade_tick_size

            # Get optimal SL
            sl_pips, _ = self.param_manager.get_risk_parameters(symbol)

            # Calculate stop loss in price terms
            sl_distance = sl_pips * point * 10  # Assuming 5-digit broker

            # Risk amount per trade
            risk_per_trade = risk_amount

            # Position size = Risk / Stop Loss Distance
            position_size = risk_per_trade / sl_distance

            # Round to lot size increments (0.01 minimum)
            position_size = max(0.01, round(position_size, 2))

            return position_size

        except Exception as e:
            self.logger.error(f"Position size calculation failed: {e}")
            return 0.01

    def open_position(self, symbol: str, direction: int, lot_size: float) -> Optional[int]:
        """Open a trading position"""
        try:
            # Get optimal SL/TP
            sl_pips, tp_pips = self.param_manager.get_risk_parameters(symbol)

            # Get current price
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None

            point = symbol_info.point
            current_price = symbol_info.ask if direction == 1 else symbol_info.bid

            # Calculate SL/TP prices
            if direction == 1:  # Buy
                sl_price = current_price - (sl_pips * point * 10)
                tp_price = current_price + (tp_pips * point * 10)
                order_type = mt5.ORDER_TYPE_BUY
            else:  # Sell
                sl_price = current_price + (sl_pips * point * 10)
                tp_price = current_price - (tp_pips * point * 10)
                order_type = mt5.ORDER_TYPE_SELL

            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": current_price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 10,
                "magic": 123456,
                "comment": "ML Trading System",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Send order
            result = mt5.order_send(request)

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"Opened {symbol} {'BUY' if direction == 1 else 'SELL'} position: {result.order}")
                self.active_positions[symbol] = {
                    'ticket': result.order,
                    'direction': direction,
                    'entry_price': current_price,
                    'sl': sl_price,
                    'tp': tp_price,
                    'lot_size': lot_size,
                    'open_time': datetime.now()
                }
                return result.order
            else:
                self.logger.error(f"Failed to open position: {result.retcode}")
                return None

        except Exception as e:
            self.logger.error(f"Open position failed: {e}")
            return None

    def manage_positions(self):
        """Manage existing positions (breakeven, trailing stops)"""
        try:
            positions = mt5.positions_get()

            if positions is None:
                return

            for position in positions:
                symbol = position.symbol
                ticket = position.ticket

                if symbol not in self.active_positions:
                    continue

                # Get optimal management parameters
                breakeven_trigger = self.param_manager.get_breakeven_settings(symbol)
                trailing_activation, trailing_distance = self.param_manager.get_trailing_settings(symbol)

                current_price = position.price_current
                entry_price = position.price_open
                current_sl = position.sl
                direction = 1 if position.type == mt5.POSITION_TYPE_BUY else -1

                # Calculate current profit in pips
                if 'XAU' in symbol or 'XAG' in symbol:
                    pip_value = 0.10  # Metals: 1 pip = 0.10
                elif symbol.endswith('JPY'):
                    pip_value = 0.01  # JPY pairs: 1 pip = 0.01
                else:
                    pip_value = 0.0001  # Standard forex: 1 pip = 0.0001
                
                current_profit_pips = abs(current_price - entry_price) / pip_value * direction

                # Breakeven management - move SL to breakeven + small buffer
                if current_profit_pips >= breakeven_trigger and current_sl != entry_price:
                    # Set SL slightly above/below entry to avoid validation errors
                    buffer = pip_value * 2  # 2 pip buffer
                    if direction == 1:  # Buy
                        new_sl = entry_price + buffer  # SL slightly above entry for profit lock
                    else:  # Sell
                        new_sl = entry_price - buffer  # SL slightly below entry for profit lock
                    self._modify_position_sl(ticket, new_sl, "Breakeven")

                # Trailing stop management
                elif current_profit_pips >= trailing_activation:
                    # Calculate new trailing SL
                    if direction == 1:  # Buy
                        new_sl = current_price - (trailing_distance * pip_value)
                        if new_sl > current_sl:
                            self._modify_position_sl(ticket, new_sl, "Trailing Stop")
                    else:  # Sell
                        new_sl = current_price + (trailing_distance * pip_value)
                        if new_sl < current_sl:
                            self._modify_position_sl(ticket, new_sl, "Trailing Stop")

        except Exception as e:
            self.logger.error(f"Position management failed: {e}")

    def _modify_position_sl(self, ticket: int, new_sl: float, comment: str):
        """Modify position stop loss"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": new_sl,
            }

            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"Modified SL for ticket {ticket}: {comment}")
            else:
                self.logger.error(f"Failed to modify SL: {result.retcode}")

        except Exception as e:
            self.logger.error(f"Modify SL failed: {e}")

    def close_expired_positions(self):
        """Close positions that have reached exit time"""
        try:
            current_hour = datetime.now().hour
            current_day = datetime.now().strftime('%A')

            for symbol, position in list(self.active_positions.items()):
                exit_hour = self.param_manager.get_exit_time(symbol)

                # Apply Friday early exit
                friday_early_exit = self.param_manager.get_friday_early_exit(symbol)
                if current_day == 'Friday':
                    exit_hour = min(exit_hour, friday_early_exit)

                # Check avoid exit days
                avoid_exit_days = self.param_manager.get_avoid_exit_days(symbol)
                if avoid_exit_days and current_day in avoid_exit_days:
                    continue  # Skip closing on avoid exit days

                if current_hour >= exit_hour:
                    self.close_position(symbol, "Time-based exit")

        except Exception as e:
            self.logger.error(f"Close expired positions failed: {e}")

    def close_position(self, symbol: str, reason: str = "Manual"):
        """Close a position"""
        try:
            if symbol not in self.active_positions:
                return False

            ticket = self.active_positions[symbol]['ticket']

            # Close position
            position = mt5.positions_get(ticket=ticket)
            if position:
                position = position[0]

                # Prepare close request
                if position.type == mt5.POSITION_TYPE_BUY:
                    order_type = mt5.ORDER_TYPE_SELL
                    price = mt5.symbol_info_tick(symbol).bid
                else:
                    order_type = mt5.ORDER_TYPE_BUY
                    price = mt5.symbol_info_tick(symbol).ask

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": position.volume,
                    "type": order_type,
                    "position": ticket,
                    "price": price,
                    "deviation": 10,
                    "magic": 123456,
                    "comment": f"Close: {reason}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                result = mt5.order_send(request)

                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    self.logger.info(f"Closed {symbol} position: {reason}")
                    del self.active_positions[symbol]
                    return True
                else:
                    self.logger.error(f"Failed to close position: {result.retcode}")
                    return False
            else:
                self.logger.warning(f"Position {ticket} not found")
                del self.active_positions[symbol]
                return False

        except Exception as e:
            self.logger.error(f"Close position failed: {e}")
            return False

    def run_trading_cycle(self):
        """Main trading cycle"""
        self.logger.info("Starting trading cycle...")

        try:
            # Check for new trading opportunities
            for symbol in self.symbols:
                if symbol in self.active_positions:
                    continue  # Skip if already have position

                should_enter, direction = self.should_enter_trade(symbol)

                if should_enter:
                    lot_size = self.calculate_position_size(symbol)
                    ticket = self.open_position(symbol, direction, lot_size)

                    if ticket:
                        self.logger.info(f"Opened new position for {symbol}")

            # Manage existing positions
            self.manage_positions()

            # Close expired positions
            self.close_expired_positions()

        except Exception as e:
            self.logger.error(f"Trading cycle failed: {e}")

    def get_system_status(self) -> Dict:
        """Get current system status"""
        status = {
            'active_positions': len(self.active_positions),
            'total_symbols': len(self.symbols),
            'session_duration': str(datetime.now() - self.session_start),
            'positions': []
        }

        for symbol, pos in self.active_positions.items():
            status['positions'].append({
                'symbol': symbol,
                'direction': 'BUY' if pos['direction'] == 1 else 'SELL',
                'entry_price': pos['entry_price'],
                'lot_size': pos['lot_size'],
                'open_time': pos['open_time'].isoformat()
            })

        return status

# Example usage
if __name__ == "__main__":
    # Load configuration
    config = {
        'trading': {
            'symbols': ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDJPY', 'EURJPY'],
            'timeframes': ['H1', 'D1'],
            'risk_per_trade': 100,
            'max_positions': 5
        }
    }

    # Initialize trading system
    trading_system = MLTradingSystem(config)

    # Run a single trading cycle
    trading_system.run_trading_cycle()

    # Print status
    status = trading_system.get_system_status()
    print(f"System Status: {status}")

    # Cleanup
    mt5.shutdown()