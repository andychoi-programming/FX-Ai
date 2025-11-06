"""
FASTER 3-Year Optimization - Reduced parameter space for reasonable runtime
Optimizes parameters on 3 years, validates on 3 months and 1 month
Reduced from 16,384 to 2,048 combinations per symbol (8x faster)
"""

import json
import os
import sys
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from pathlib import Path
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import ParameterGrid

sys.path.append(str(Path(__file__).parent))

class FastParameterOptimizer:
    """Faster optimization with reduced but effective parameter space"""

    def __init__(self, config):
        self.config = config
        
        # All 30 symbols
        self.symbols = [
            "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
            "CADCHF", "CADJPY", "CHFJPY",
            "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
            "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
            "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
            "USDCAD", "USDCHF", "USDJPY",
            "XAGUSD", "XAUUSD"
        ]
        
        self.timeframes = {'H1': mt5.TIMEFRAME_H1}
        
        # Time periods
        self.train_end = datetime(2025, 10, 31)
        self.train_start = datetime(2022, 10, 31)  # 3 years
        self.validate_3m_start = datetime(2025, 8, 2)
        self.validate_1m_start = datetime(2025, 10, 2)
        
        # Symbol specifications
        self.symbol_specs = self._get_symbol_specs()
        
        # REDUCED parameter ranges for faster optimization
        self.parameter_ranges = self._get_parameter_ranges()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('fast_optimization.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        if not mt5.initialize():  # type: ignore
            raise Exception("MT5 initialization failed")
    
    def _get_symbol_specs(self):
        """Get pip values for each symbol type"""
        specs = {}
        for symbol in self.symbols:
            if 'JPY' in symbol:
                specs[symbol] = {'pip_value': 0.01, 'pip_multiplier': 100}
            elif 'XAG' in symbol:
                specs[symbol] = {'pip_value': 50.0, 'pip_multiplier': 100}
            elif 'XAU' in symbol:
                specs[symbol] = {'pip_value': 1.0, 'pip_multiplier': 100}
            else:
                specs[symbol] = {'pip_value': 0.0001, 'pip_multiplier': 10000}
        return specs
    
    def _get_parameter_ranges(self):
        """Fast 3-year backtest parameter ranges
        
        Strategy:
        - Stage 1: Test SL/TP + timing combinations (~1,248 combos)
        - Stage 2: Test BE/trailing variations around Stage 1 best (~12 combos)
        - Stage 3: Fine-tune SL/TP around Stage 2 best (~9 combos)
        - Stage 4: Test ATR & session filters around Stage 3 best (~9 combos)
        
        Total combinations: ~1,278 per symbol
        Total execution time: ~5.5-6 hours for 30 symbols
        Per symbol: ~11 minutes
        """
        return {
            'forex': {
                'sl_pips': [15, 20, 25, 30],  # 4 options (focused range)
                'tp_pips': [60, 75, 90, 120],  # 4 options (focused on 1:3 to 1:4 R/R)
                'breakeven_trigger': [20],  # 1 option (most effective)
                'trailing_activation': [25],  # 1 option (most effective)
                'trailing_distance': [15],  # 1 option (most effective)
                'entry_hour_start': [6, 8],  # 2 options (early vs normal)
                'entry_hour_end': [16, 18],  # 2 options (early vs normal close)
                'max_holding_hours': [6, 12],  # 2 options (short vs medium)
                'monday_entry_delay': [10, 12],  # 2 options (10AM vs 12PM Monday start)
                'friday_early_exit': [10, 13, 16],  # 3 options (before/during/after US news)
                'hard_close_hour': [22],  # Hard close at 22:30
                'best_entry_days': [
                    ['Tuesday', 'Wednesday', 'Thursday', 'Friday'],  # Skip Monday
                    ['Monday', 'Tuesday', 'Wednesday', 'Thursday']  # Include Monday
                ],  # 2 options
            },
            'jpy': {
                'sl_pips': [15, 20, 25, 30],  # 4 options (focused range)
                'tp_pips': [60, 75, 90, 120],  # 4 options (focused on 1:3 to 1:4 R/R)
                'breakeven_trigger': [20],  # 1 option (most effective)
                'trailing_activation': [25],  # 1 option (most effective)
                'trailing_distance': [15],  # 1 option (most effective)
                'entry_hour_start': [6, 8],  # 2 options
                'entry_hour_end': [16, 18],  # 2 options
                'max_holding_hours': [6, 12],  # 2 options
                'monday_entry_delay': [10, 12],  # 2 options (10AM vs 12PM Monday start)
                'friday_early_exit': [10, 13, 16],  # 3 options (before/during/after US news 15:30)
                'hard_close_hour': [22],  # Hard close at 22:30
                'best_entry_days': [
                    ['Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                    ['Monday', 'Tuesday', 'Wednesday', 'Thursday']
                ],  # 2 options
            },
            'metal': {
                'sl_pips': [300, 500, 700, 1000],  # 4 options (focused range)
                'tp_pips': [1200, 1500, 2100, 3000],  # 4 options (focused on proven multiples)
                'breakeven_trigger': [500],  # 1 option (most effective)
                'trailing_activation': [700],  # 1 option (most effective)
                'trailing_distance': [300],  # 1 option (most effective)
                'entry_hour_start': [6, 8],  # 2 options
                'entry_hour_end': [16, 18],  # 2 options
                'max_holding_hours': [4, 8],  # 2 options (metals move faster)
                'monday_entry_delay': [10, 12],  # 2 options (10AM vs 12PM Monday start)
                'friday_early_exit': [10, 13, 16],  # 3 options (before/during/after US news 15:30)
                'hard_close_hour': [22],  # Hard close at 22:30
                'best_entry_days': [
                    ['Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                    ['Monday', 'Tuesday', 'Wednesday', 'Thursday']
                ],  # 2 options
            }
        }
    
    def _get_symbol_type(self, symbol):
        if 'XAG' in symbol or 'XAU' in symbol:
            return 'metal'
        elif 'JPY' in symbol:
            return 'jpy'
        else:
            return 'forex'
    
    def get_historical_data(self, symbol, timeframe, start_date, end_date):
        try:
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)  # type: ignore
            if rates is None or len(rates) == 0:
                return None
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            self.logger.info(f"  Loaded {len(df)} bars ({start_date.date()} to {end_date.date()})")
            return df
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return None
    
    def generate_predictions(self, df):
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        df['sma_fast'] = df['close'].rolling(window=10).mean()
        df['sma_slow'] = df['close'].rolling(window=20).mean()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate ATR for Stage 4 volatility filtering
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_close'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        predictions = np.where(
            (df['sma_fast'] > df['sma_slow']) & 
            (df['rsi'] < 70) & 
            (df['returns'] > 0),
            1.0, 0.0
        )
        return df, predictions
    
    def backtest_parameters(self, df, predictions, params, symbol):
        """Backtest with MT5 time-based logic and 22:30 hard close
        
        Returns detailed trade log with timestamps for period-based analysis
        """
        capital = 10000.0
        position = None
        trades = []  # Will store dict with: {pnl, entry_time, exit_time, direction, pips}
        
        spec = self.symbol_specs[symbol]
        pip_value = spec['pip_value']
        pip_multiplier = spec['pip_multiplier']
        
        sl_pips = params['sl_pips']
        tp_pips = params['tp_pips']
        breakeven_trigger = params.get('breakeven_trigger', 20)
        trailing_activation = params.get('trailing_activation', 20)
        trailing_distance = params.get('trailing_distance', 15)
        entry_start = params['entry_hour_start']
        entry_end = params['entry_hour_end']
        monday_delay = params.get('monday_entry_delay', 12)
        friday_exit = params.get('friday_early_exit', 20)
        hard_close_hour = params.get('hard_close_hour', 22)  # Hard close at 22:30
        hard_close_minute = 30
        best_days = params.get('best_entry_days', ['Tuesday', 'Wednesday', 'Thursday', 'Friday'])
        max_hours = params.get('max_holding_hours', 18)
        atr_multiplier = params.get('atr_multiplier', None)  # Optional for Stage 4
        trading_session = params.get('trading_session', None)  # Optional for Stage 4
        
        # Define trading session hours (GMT) if session filtering enabled
        if trading_session:
            session_hours = {
                'all': list(range(24)),
                'london_ny': list(range(8, 21)),  # London 08:00-16:00, NY 13:00-21:00
                'asian_london': list(range(0, 16))  # Asian 00:00-08:00, London 08:00-16:00
            }
            allowed_hours = session_hours.get(trading_session, list(range(24)))
        else:
            allowed_hours = None
        
        for i in range(len(df)):
            current_time = df.index[i]
            current_day = current_time.strftime('%A')
            current_hour = current_time.hour
            current_minute = current_time.minute
            current_price = df.iloc[i]['close']
            
            # HARD CLOSE at 22:30 MT5 time (before any other logic)
            if position is not None:
                if current_hour == hard_close_hour and current_minute >= hard_close_minute:
                    # Force close at 22:30
                    price_change = (current_price - position['entry_price']) * position['direction']
                    pips_change = price_change * pip_multiplier
                    pnl = pips_change * pip_value
                    capital += pnl
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pips': pips_change,
                        'pnl': pnl,
                        'exit_reason': 'hard_close_2230'
                    })
                    position = None
                    continue
                elif current_hour > hard_close_hour:
                    # Already past 22:30, force close
                    price_change = (current_price - position['entry_price']) * position['direction']
                    pips_change = price_change * pip_multiplier
                    pnl = pips_change * pip_value
                    capital += pnl
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pips': pips_change,
                        'pnl': pnl,
                        'exit_reason': 'hard_close_past_2230'
                    })
                    position = None
                    continue
            
            # Exit logic
            if position is not None:
                price_change = (current_price - position['entry_price']) * position['direction']
                pips_change = price_change * pip_multiplier
                
                # Calculate current SL (may have been moved to breakeven or trailing)
                current_sl = position.get('current_sl', position['entry_price'] - (sl_pips / pip_multiplier * position['direction']))
                sl_distance_pips = abs((current_price - current_sl) * pip_multiplier * position['direction'])
                
                # BREAKEVEN LOGIC: Move SL to breakeven when profit reaches trigger
                if not position.get('breakeven_set', False) and pips_change >= breakeven_trigger:
                    position['current_sl'] = position['entry_price']
                    position['breakeven_set'] = True
                
                # TRAILING STOP LOGIC: Activate when profit reaches threshold
                if pips_change >= trailing_activation:
                    # Calculate where trailing stop should be
                    ideal_trailing_sl = current_price - (trailing_distance / pip_multiplier * position['direction'])
                    
                    # Only move SL up (never down for longs, never up for shorts)
                    if position['direction'] == 1:  # Long
                        if ideal_trailing_sl > position.get('current_sl', position['entry_price'] - (sl_pips / pip_multiplier)):
                            position['current_sl'] = ideal_trailing_sl
                    else:  # Short
                        if ideal_trailing_sl < position.get('current_sl', position['entry_price'] + (sl_pips / pip_multiplier)):
                            position['current_sl'] = ideal_trailing_sl
                
                # Check Stop Loss (using current_sl which may be breakeven or trailing)
                if position['direction'] == 1:  # Long
                    if current_price <= position.get('current_sl', position['entry_price'] - (sl_pips / pip_multiplier)):
                        # Hit stop loss
                        sl_pips_actual = (position.get('current_sl', position['entry_price'] - (sl_pips / pip_multiplier)) - position['entry_price']) * pip_multiplier
                        pnl = sl_pips_actual * pip_value
                        capital += pnl
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': current_time,
                            'direction': position['direction'],
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'pips': sl_pips_actual,
                            'pnl': pnl,
                            'exit_reason': 'stop_loss'
                        })
                        position = None
                        continue
                else:  # Short
                    if current_price >= position.get('current_sl', position['entry_price'] + (sl_pips / pip_multiplier)):
                        # Hit stop loss
                        sl_pips_actual = (position['entry_price'] - position.get('current_sl', position['entry_price'] + (sl_pips / pip_multiplier))) * pip_multiplier
                        pnl = sl_pips_actual * pip_value
                        capital += pnl
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': current_time,
                            'direction': position['direction'],
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'pips': sl_pips_actual,
                            'pnl': pnl,
                            'exit_reason': 'stop_loss'
                        })
                        position = None
                        continue
                
                # Check Take Profit
                if pips_change >= tp_pips:
                    pnl = tp_pips * pip_value
                    capital += pnl
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pips': pips_change,
                        'pnl': pnl,
                        'exit_reason': 'take_profit'
                    })
                    position = None
                    continue
                
                # TIME EXIT: Friday early exit
                holding_hours = (current_time - position['entry_time']).total_seconds() / 3600
                
                if current_day == 'Friday' and current_hour >= friday_exit:
                    pnl = pips_change * pip_value
                    capital += pnl
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pips': pips_change,
                        'pnl': pnl,
                        'exit_reason': 'friday_early_exit'
                    })
                    position = None
                    continue
                
                # TIME EXIT: Max holding duration
                if holding_hours >= max_hours:
                    pnl = pips_change * pip_value
                    capital += pnl
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pips': pips_change,
                        'pnl': pnl,
                        'exit_reason': 'max_holding_duration'
                    })
                    position = None
                    continue
            
            # ENTRY LOGIC with TIME FILTERS
            if position is None and i < len(predictions):
                # Filter 1: Best entry days (avoid weekends, Monday volatility)
                if current_day not in best_days:
                    continue
                
                # Filter 2: Monday entry delay (wait for market to settle)
                if current_day == 'Monday' and current_hour < monday_delay:
                    continue
                
                # Filter 3: Entry time window (avoid overnight and low liquidity hours)
                if not (entry_start <= current_hour <= entry_end):
                    continue
                
                # Filter 4: Trading session restriction (Stage 4 only)
                if allowed_hours is not None and current_hour not in allowed_hours:
                    continue
                
                # Filter 5: ATR volatility filter (Stage 4 only)
                if atr_multiplier is not None:
                    current_atr = df.iloc[i]['atr']
                    if pd.isna(current_atr):  # Skip if ATR not yet calculated
                        continue
                    # Require minimum volatility (ATR * multiplier)
                    min_volatility = current_atr * atr_multiplier
                    # Use current price movement as volatility proxy
                    if i > 0:
                        current_volatility = abs(df.iloc[i]['close'] - df.iloc[i-1]['close'])
                        if current_volatility < min_volatility:
                            continue
                
                # Filter 6: Prediction confidence
                prediction = predictions[i]
                if prediction < 0.5:
                    continue
                
                # Enter position
                direction = 1 if prediction > 0.5 else -1
                position = {
                    'entry_price': current_price,
                    'entry_time': current_time,
                    'direction': direction,
                    'breakeven_set': False,  # Track if breakeven has been activated
                    'current_sl': current_price - (sl_pips / pip_multiplier * direction)  # Initial SL
                }
        
        # Calculate metrics from detailed trade log
        total_pnl = sum([t['pnl'] for t in trades]) if trades else 0
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        total_trades = len(trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        return {
            'pnl': total_pnl,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_trade': avg_trade,
            'capital': capital,
            'winning_trades': winning_trades,
            'trade_log': trades  # Full trade log with timestamps for period analysis
        }
    
    def calculate_metrics_from_trades(self, trades):
        """Calculate performance metrics from trade log"""
        if not trades:
            return {
                'pnl': 0,
                'total_trades': 0,
                'win_rate': 0,
                'avg_trade': 0,
                'winning_trades': 0
            }
        
        total_pnl = sum([t['pnl'] for t in trades])
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        total_trades = len(trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        return {
            'pnl': total_pnl,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_trade': avg_trade,
            'winning_trades': winning_trades
        }
    
    def split_metrics_by_period(self, full_result, train_end, validate_3m_start, validate_1m_start):
        """Split trade log into periods and calculate metrics for each
        
        Args:
            full_result: Result dict with 'trade_log' from backtest_parameters
            train_end: End of training period
            validate_3m_start: Start of 3-month validation
            validate_1m_start: Start of 1-month validation
        
        Returns:
            Dict with metrics for each period
        """
        import pandas as pd
        
        trade_log = full_result.get('trade_log', [])
        
        # Filter trades by exit_time (when trade actually closed)
        train_trades = [t for t in trade_log if t['exit_time'] <= train_end]
        val_3m_trades = [t for t in trade_log if validate_3m_start <= t['exit_time'] <= train_end]
        val_1m_trades = [t for t in trade_log if validate_1m_start <= t['exit_time'] <= train_end]
        
        return {
            'train': self.calculate_metrics_from_trades(train_trades),
            'validation_3m': self.calculate_metrics_from_trades(val_3m_trades),
            'validation_1m': self.calculate_metrics_from_trades(val_1m_trades)
        }
    
    def fine_tune_parameters(self, symbol, train_df, train_predictions, stage1_params, stage1_pnl):
        """STAGE 2: Fine-tune breakeven, trailing stops, and SL/TP around Stage 1 best params"""
        self.logger.info(f"\n{'-' * 70}")
        self.logger.info(f"STAGE 2: FINE-TUNING {symbol}")
        self.logger.info(f"{'-' * 70}")
        
        symbol_type = self._get_symbol_type(symbol)
        
        # Build fine-tuning ranges around Stage 1 optimal parameters
        if symbol_type == 'metal':
            # Test all breakeven and trailing combinations around optimal SL/TP
            sl_base = stage1_params['sl_pips']
            tp_base = stage1_params['tp_pips']
            
            fine_tune_ranges = {
                'sl_pips': [sl_base],  # Keep optimal SL
                'tp_pips': [tp_base],  # Keep optimal TP
                'breakeven_trigger': [300, 500, 700],  # All 3 options
                'trailing_activation': [500, 700],  # Both options
                'trailing_distance': [300, 500],  # Both options
                'entry_hour_start': [stage1_params['entry_hour_start']],
                'entry_hour_end': [stage1_params['entry_hour_end']],
                'max_holding_hours': [stage1_params['max_holding_hours']],
                'monday_entry_delay': [stage1_params['monday_entry_delay']],
                'friday_early_exit': [stage1_params['friday_early_exit']],
                'hard_close_hour': [22],
                'best_entry_days': [stage1_params['best_entry_days']]
            }
        else:
            # Forex/JPY fine-tuning
            sl_base = stage1_params['sl_pips']
            tp_base = stage1_params['tp_pips']
            
            fine_tune_ranges = {
                'sl_pips': [sl_base],  # Keep optimal SL
                'tp_pips': [tp_base],  # Keep optimal TP
                'breakeven_trigger': [15, 20, 25],  # All 3 options
                'trailing_activation': [20, 25],  # Both options
                'trailing_distance': [10, 15],  # Both options
                'entry_hour_start': [stage1_params['entry_hour_start']],
                'entry_hour_end': [stage1_params['entry_hour_end']],
                'max_holding_hours': [stage1_params['max_holding_hours']],
                'monday_entry_delay': [stage1_params['monday_entry_delay']],
                'friday_early_exit': [stage1_params['friday_early_exit']],
                'hard_close_hour': [22],
                'best_entry_days': [stage1_params['best_entry_days']]
            }
        
        param_combinations = list(ParameterGrid(fine_tune_ranges))
        self.logger.info(f"Testing {len(param_combinations)} fine-tuning combinations")
        
        best_params = stage1_params.copy()
        best_pnl = stage1_pnl
        best_metrics = None
        
        for i, params in enumerate(param_combinations):
            result = self.backtest_parameters(train_df.copy(), train_predictions, params, symbol)
            
            # Filter: Minimum 20 trades AND minimum 35% win rate
            if result['pnl'] > best_pnl and result['total_trades'] >= 20 and result['win_rate'] >= 0.35:
                best_pnl = result['pnl']
                best_params = params
                best_metrics = result
                self.logger.info(f"  IMPROVED: P&L=${result['pnl']:.2f}, WinRate={result['win_rate']*100:.1f}%, "
                               f"BE={params['breakeven_trigger']}, Trail={params['trailing_activation']}/{params['trailing_distance']}")
        
        improvement = ((best_pnl - stage1_pnl) / abs(stage1_pnl) * 100) if stage1_pnl != 0 else 0
        self.logger.info(f"\nStage 2 Complete: P&L=${best_pnl:.2f} (Improvement: {improvement:+.1f}%)")
        
        return best_params, best_pnl, best_metrics
    
    def fine_tune_sltp(self, symbol, train_df, train_predictions, stage2_params, stage2_pnl):
        """STAGE 3: Fine-tune SL/TP with ±5 pips around Stage 2 optimal"""
        self.logger.info(f"\n{'-' * 70}")
        self.logger.info(f"STAGE 3: FINE-TUNING SL/TP for {symbol}")
        self.logger.info(f"{'-' * 70}")
        
        symbol_type = self._get_symbol_type(symbol)
        
        # Get Stage 2 optimal SL/TP
        sl_base = stage2_params['sl_pips']
        tp_base = stage2_params['tp_pips']
        
        # Build fine-tuning ranges: ±5 pips SL, ±15 pips TP (9 combos)
        if symbol_type == 'metal':
            # Metal uses larger increments
            sl_step = 100
            tp_step = 300
        else:
            # Forex/JPY uses 5 pip increments
            sl_step = 5
            tp_step = 15
        
        fine_tune_ranges = {
            'sl_pips': [max(10, sl_base - sl_step), sl_base, sl_base + sl_step],  # 3 options
            'tp_pips': [max(30, tp_base - tp_step), tp_base, tp_base + tp_step],  # 3 options
            'breakeven_trigger': [stage2_params['breakeven_trigger']],  # Lock Stage 2
            'trailing_activation': [stage2_params['trailing_activation']],  # Lock Stage 2
            'trailing_distance': [stage2_params['trailing_distance']],  # Lock Stage 2
            'entry_hour_start': [stage2_params['entry_hour_start']],
            'entry_hour_end': [stage2_params['entry_hour_end']],
            'max_holding_hours': [stage2_params['max_holding_hours']],
            'monday_entry_delay': [stage2_params['monday_entry_delay']],
            'friday_early_exit': [stage2_params['friday_early_exit']],
            'hard_close_hour': [22],
            'best_entry_days': [stage2_params['best_entry_days']]
        }
        
        param_combinations = list(ParameterGrid(fine_tune_ranges))
        self.logger.info(f"Testing {len(param_combinations)} SL/TP fine-tuning combinations")
        
        best_params = stage2_params.copy()
        best_pnl = stage2_pnl
        best_metrics = None
        
        for i, params in enumerate(param_combinations):
            # Ensure minimum 1:3 R/R ratio
            rr_ratio = params['tp_pips'] / params['sl_pips']
            if rr_ratio < 3.0:
                continue
                
            result = self.backtest_parameters(train_df.copy(), train_predictions, params, symbol)
            
            # Filter: Minimum 20 trades AND minimum 35% win rate
            if result['pnl'] > best_pnl and result['total_trades'] >= 20 and result['win_rate'] >= 0.35:
                best_pnl = result['pnl']
                best_params = params
                best_metrics = result
                self.logger.info(f"  IMPROVED: P&L=${result['pnl']:.2f}, WinRate={result['win_rate']*100:.1f}%, "
                               f"SL={params['sl_pips']}, TP={params['tp_pips']}, R/R=1:{rr_ratio:.1f}")
        
        improvement = ((best_pnl - stage2_pnl) / abs(stage2_pnl) * 100) if stage2_pnl != 0 else 0
        self.logger.info(f"\nStage 3 Complete: P&L=${best_pnl:.2f} (Improvement: {improvement:+.1f}%)")
        
        return best_params, best_pnl, best_metrics
    
    def optimize_atr_session(self, symbol, train_df, train_predictions, stage3_params, stage3_pnl):
        """STAGE 4: Test ATR and session filters around Stage 3 optimal"""
        self.logger.info(f"\n{'-' * 70}")
        self.logger.info(f"STAGE 4: ATR & SESSION OPTIMIZATION for {symbol}")
        self.logger.info(f"{'-' * 70}")
        
        # Test ATR and session combinations (9 combos: 3 ATR × 3 sessions)
        atr_session_ranges = {
            'sl_pips': [stage3_params['sl_pips']],  # Lock Stage 3
            'tp_pips': [stage3_params['tp_pips']],  # Lock Stage 3
            'breakeven_trigger': [stage3_params['breakeven_trigger']],  # Lock Stage 3
            'trailing_activation': [stage3_params['trailing_activation']],  # Lock Stage 3
            'trailing_distance': [stage3_params['trailing_distance']],  # Lock Stage 3
            'entry_hour_start': [stage3_params['entry_hour_start']],
            'entry_hour_end': [stage3_params['entry_hour_end']],
            'max_holding_hours': [stage3_params['max_holding_hours']],
            'monday_entry_delay': [stage3_params['monday_entry_delay']],
            'friday_early_exit': [stage3_params['friday_early_exit']],
            'hard_close_hour': [22],
            'best_entry_days': [stage3_params['best_entry_days']],
            'atr_multiplier': [1.0, 1.5, 2.0],  # 3 ATR options
            'trading_session': ['all', 'london_ny', 'asian_london']  # 3 session options
        }
        
        param_combinations = list(ParameterGrid(atr_session_ranges))
        self.logger.info(f"Testing {len(param_combinations)} ATR/session combinations")
        
        best_params = stage3_params.copy()
        best_pnl = stage3_pnl
        best_metrics = None
        
        for i, params in enumerate(param_combinations):
            result = self.backtest_parameters(train_df.copy(), train_predictions, params, symbol)
            
            # Filter: Minimum 20 trades AND minimum 35% win rate
            if result['pnl'] > best_pnl and result['total_trades'] >= 20 and result['win_rate'] >= 0.35:
                best_pnl = result['pnl']
                best_params = params
                best_metrics = result
                self.logger.info(f"  IMPROVED: P&L=${result['pnl']:.2f}, WinRate={result['win_rate']*100:.1f}%, "
                               f"ATR={params['atr_multiplier']}x, Session={params['trading_session']}")
        
        improvement = ((best_pnl - stage3_pnl) / abs(stage3_pnl) * 100) if stage3_pnl != 0 else 0
        self.logger.info(f"\nStage 4 Complete: P&L=${best_pnl:.2f} (Improvement: {improvement:+.1f}%)")
        
        return best_params, best_pnl, best_metrics
    
    def optimize_symbol(self, symbol):
        self.logger.info(f"\n{'=' * 70}")
        self.logger.info(f"OPTIMIZING: {symbol}")
        self.logger.info(f"{'=' * 70}")
        
        # Get 3-year training data
        train_df = self.get_historical_data(symbol, self.timeframes['H1'], self.train_start, self.train_end)
        if train_df is None or len(train_df) < 500:
            self.logger.warning(f"Insufficient data for {symbol}")
            return None
        
        train_df, train_predictions = self.generate_predictions(train_df)
        
        symbol_type = self._get_symbol_type(symbol)
        param_ranges = self.parameter_ranges[symbol_type]
        param_combinations = list(ParameterGrid(param_ranges))
        
        # Filter to ensure minimum 1:3 risk/reward ratio
        valid_combinations = []
        for params in param_combinations:
            rr_ratio = params['tp_pips'] / params['sl_pips']
            if rr_ratio >= 3.0:  # Minimum 1:3 ratio
                valid_combinations.append(params)
        
        self.logger.info(f"Symbol type: {symbol_type}")
        self.logger.info(f"Testing {len(valid_combinations)} combinations (filtered for min 1:3 R/R ratio)")
        
        # Optimize on training data
        best_params = None
        best_train_pnl = float('-inf')
        best_train_metrics = None
        
        for i, params in enumerate(valid_combinations):
            if i % 200 == 0:
                self.logger.info(f"  Progress: {i}/{len(valid_combinations)}")
            
            result = self.backtest_parameters(train_df.copy(), train_predictions, params, symbol)
            
            # Filter: Minimum 20 trades AND minimum 35% win rate for consistency
            if result['pnl'] > best_train_pnl and result['total_trades'] >= 20 and result['win_rate'] >= 0.35:
                best_train_pnl = result['pnl']
                best_params = params
                best_train_metrics = result
                rr_ratio = params['tp_pips'] / params['sl_pips']
                self.logger.info(f"  NEW BEST: P&L=${result['pnl']:.2f}, WinRate={result['win_rate']*100:.1f}%, "
                               f"SL={params['sl_pips']}, TP={params['tp_pips']}, R/R=1:{rr_ratio:.1f}")
        
        if best_params is None:
            self.logger.warning(f"No profitable parameters for {symbol}")
            return None
        
        rr_ratio = best_params['tp_pips'] / best_params['sl_pips']
        if best_train_metrics:
            self.logger.info(f"\nSTAGE 1 COMPLETE: P&L=${best_train_pnl:.2f}, Trades={best_train_metrics['total_trades']}, "
                            f"WinRate={best_train_metrics['win_rate']*100:.1f}%")
        else:
            self.logger.info(f"\nSTAGE 1 COMPLETE: P&L=${best_train_pnl:.2f}, Trades=N/A, WinRate=N/A")
        self.logger.info(f"PARAMS: SL={best_params['sl_pips']}, TP={best_params['tp_pips']}, R/R=1:{rr_ratio:.1f}, "
                        f"Entry={best_params['entry_hour_start']}-{best_params['entry_hour_end']}h")
        
        # STAGE 2: Fine-tune breakeven and trailing stops
        stage2_params, stage2_pnl, stage2_metrics = self.fine_tune_parameters(
            symbol, train_df, train_predictions, best_params, best_train_pnl
        )
        
        # Use Stage 2 results if better
        if stage2_pnl > best_train_pnl:
            best_params = stage2_params
            best_train_pnl = stage2_pnl
            best_train_metrics = stage2_metrics
            stage2_improved = True
        else:
            stage2_improved = False
        
        # STAGE 3: Fine-tune SL/TP around Stage 2 optimal
        stage3_params, stage3_pnl, stage3_metrics = self.fine_tune_sltp(
            symbol, train_df, train_predictions, stage2_params, stage2_pnl
        )
        
        # Use Stage 3 results if better
        if stage3_pnl > best_train_pnl:
            best_params = stage3_params
            best_train_pnl = stage3_pnl
            best_train_metrics = stage3_metrics
            stage3_improved = True
        else:
            stage3_params = best_params.copy()
            stage3_pnl = best_train_pnl
            stage3_metrics = best_train_metrics
            stage3_improved = False
        
        # STAGE 4: Test ATR and session filters
        stage4_params, stage4_pnl, stage4_metrics = self.optimize_atr_session(
            symbol, train_df, train_predictions, stage3_params, stage3_pnl
        )
        
        # Use Stage 4 results if better
        if stage4_pnl > best_train_pnl:
            best_params = stage4_params
            best_train_pnl = stage4_pnl
            best_train_metrics = stage4_metrics
            self.logger.info(f"[BEST] Using Stage 4 parameters (ATR/session optimization)")
        elif stage3_improved:
            self.logger.info(f"[BEST] Using Stage 3 parameters (SL/TP optimization)")
        elif stage2_improved:
            self.logger.info(f"[BEST] Using Stage 2 parameters (BE/trailing optimization)")
        else:
            self.logger.info(f"[BEST] Using Stage 1 parameters (already optimal)")
        
        rr_ratio = best_params['tp_pips'] / best_params['sl_pips']
        if best_train_metrics:
            self.logger.info(f"\nFINAL TRAINING (3Y): P&L=${best_train_pnl:.2f}, Trades={best_train_metrics['total_trades']}, "
                            f"WinRate={best_train_metrics['win_rate']*100:.1f}%")
        else:
            self.logger.info(f"\nFINAL TRAINING (3Y): P&L=${best_train_pnl:.2f}, Trades=N/A, WinRate=N/A")
        self.logger.info(f"FINAL PARAMS: SL={best_params['sl_pips']}, TP={best_params['tp_pips']}, R/R=1:{rr_ratio:.1f}, "
                        f"BE={best_params['breakeven_trigger']}, Trail={best_params['trailing_activation']}/{best_params['trailing_distance']}")
        self.logger.info(f"Entry Hours: {best_params['entry_hour_start']}-{best_params['entry_hour_end']}h")
        
        # Show ATR and session filters if Stage 4 was used
        if 'atr_multiplier' in best_params and 'trading_session' in best_params:
            self.logger.info(f"Filters: ATR={best_params['atr_multiplier']}x, Session={best_params['trading_session']}")
        
        # Run single backtest on full period and extract validation metrics from trade log
        full_df = self.get_historical_data(symbol, self.timeframes['H1'], self.validate_3m_start, self.train_end)
        if full_df is not None and len(full_df) >= 100:
            full_df, full_pred = self.generate_predictions(full_df)
            full_result = self.backtest_parameters(full_df, full_pred, best_params, symbol)
            
            # Split metrics by period using trade log timestamps
            period_metrics = self.split_metrics_by_period(
                full_result, self.train_end, self.validate_3m_start, self.validate_1m_start
            )
            
            val_3m_result = period_metrics['validation_3m']
            val_1m_result = period_metrics['validation_1m']
            
            self.logger.info(f"VALIDATE (3M): P&L=${val_3m_result['pnl']:.2f}, WinRate={val_3m_result['win_rate']*100:.1f}%")
            self.logger.info(f"VALIDATE (1M): P&L=${val_1m_result['pnl']:.2f}, WinRate={val_1m_result['win_rate']*100:.1f}%")
        else:
            val_3m_result = None
            val_1m_result = None
        
        # Check profitability
        all_profitable = best_train_pnl > 0
        if val_3m_result:
            all_profitable = all_profitable and val_3m_result['pnl'] > 0
        if val_1m_result:
            all_profitable = all_profitable and val_1m_result['pnl'] > 0
        
        status = "[PASSED]" if all_profitable else "[FAILED]"
        self.logger.info(f"\nSTATUS: {status}\n")
        
        return {
            'symbol': symbol,
            'timeframe': 'H1',
            'best_pnl': best_train_pnl,
            'optimal_params': best_params,
            'performance_metrics': best_train_metrics,
            'validation_3m': val_3m_result,
            'validation_1m': val_1m_result,
            'all_periods_profitable': all_profitable
        }
    
    def optimize_all_symbols(self):
        self.logger.info("\n" + "=" * 70)
        self.logger.info("FOUR-STAGE OPTIMIZATION - 3-YEAR BACKTEST")
        self.logger.info("Stage 1: ~1,248 combos (broad SL/TP search)")
        self.logger.info("Stage 2: ~12 combos (fine-tune BE/trailing)")
        self.logger.info("Stage 3: ~9 combos (fine-tune SL/TP)")
        self.logger.info("Stage 4: ~9 combos (test ATR & session filters)")
        self.logger.info("Total: ~1,278 combos per symbol (~11 min each, ~5.5 hours total)")
        self.logger.info("=" * 70)
        
        results = {}
        passed = 0
        
        for symbol in self.symbols:
            try:
                result = self.optimize_symbol(symbol)
                if result:
                    results[symbol] = {'H1': result}
                    if result['all_periods_profitable']:
                        passed += 1
            except Exception as e:
                self.logger.error(f"Error with {symbol}: {e}")
        
        # Save results - Convert non-serializable objects
        def convert_to_serializable(obj):
            """Convert non-JSON serializable objects to JSON-compatible types"""
            if obj is None:
                return None
            elif isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):  # type: ignore
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):  # type: ignore
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.ndarray,)):
                return convert_to_serializable(obj.tolist())
            elif isinstance(obj, dict):
                return {str(k): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, str):
                return obj
            elif isinstance(obj, (int, float)):
                return obj
            else:
                # Try to convert unknown types to string as fallback
                try:
                    return str(obj)
                except:
                    return None
        
        # Convert results to JSON-serializable format
        try:
            serializable_results = convert_to_serializable(results)
        except Exception as e:
            self.logger.error(f"Error converting results to serializable format: {e}")
            serializable_results = results  # Try with original if conversion fails
        
        # Save to JSON file
        output_file = Path('models/parameter_optimization/optimal_parameters.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            self.logger.info("\n" + "=" * 70)
            self.logger.info(f"COMPLETE: {passed}/{len(self.symbols)} passed all validations")
            self.logger.info(f"Saved to: {output_file}")
            self.logger.info("=" * 70)
        except TypeError as e:
            self.logger.error(f"JSON serialization error: {e}")
            # Try to save a simplified version with just the summary
            try:
                backup_file = Path('models/parameter_optimization/optimal_parameters_backup.json')
                summary_only = {
                    'error': 'Full results failed to serialize',
                    'total_symbols': len(self.symbols),
                    'passed': passed,
                    'failed': len(self.symbols) - passed,
                    'optimization_date': str(datetime.now()),
                    'symbols_processed': list(results.keys())
                }
                with open(backup_file, 'w') as f:
                    json.dump(summary_only, f, indent=2)
                self.logger.warning(f"Saved summary only to: {backup_file}")
            except Exception as backup_error:
                self.logger.error(f"Failed to save backup: {backup_error}")
            
            self.logger.info("\n" + "=" * 70)
            self.logger.info(f"COMPLETE: {passed}/{len(self.symbols)} passed all validations")
            self.logger.info(f"WARNING: Results not saved due to serialization error")
            self.logger.info("=" * 70)
        
        mt5.shutdown()  # type: ignore
        return results

def main():
    # Get parent directory to access config folder
    parent_dir = Path(__file__).parent.parent
    config_path = parent_dir / "config" / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    optimizer = FastParameterOptimizer(config)
    results = optimizer.optimize_all_symbols()
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    for symbol, data in results.items():
        if 'H1' in data and data['H1'].get('all_periods_profitable'):
            params = data['H1']['optimal_params']
            print(f"{symbol:8} - SL:{params['sl_pips']:4} TP:{params['tp_pips']:4} - [VALIDATED]")

if __name__ == "__main__":
    main()
