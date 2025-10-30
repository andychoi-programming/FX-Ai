"""
FX-Ai Trading Engine - Fixed Version
Handles trade execution with proper async/await implementation
"""

import logging
import asyncio
from datetime import datetime, time
from typing import Dict, List, Optional, Any
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)

class TradingEngine:
    """Trading engine with fixed async methods"""

    def __init__(self, mt5_connector, risk_manager, technical_analyzer,
                 sentiment_analyzer, fundamental_collector, ml_predictor, adaptive_learning_manager=None):
        """Initialize trading engine"""
        self.mt5 = mt5_connector
        self.risk_manager = risk_manager
        self.technical = technical_analyzer
        self.sentiment = sentiment_analyzer
        self.fundamental = fundamental_collector
        self.ml_predictor = ml_predictor
        self.adaptive_learning_manager = adaptive_learning_manager
        self.active_positions = {}

        # Get config from risk_manager or create default
        self.config = getattr(risk_manager, 'config', {}) if risk_manager else {}

        # Trading parameters
        self.magic_number = self.config.get('trading', {}).get('magic_number', 20241029)
        self.max_slippage = self.config.get('trading', {}).get('max_slippage', 3)

        logger.info("Trading Engine initialized")

    def debug_stop_loss_calculation(self, symbol: str, order_type: str, stop_loss_pips: float):
        """Debug function to trace stop loss calculation issues"""
        
        print(f"\n=== ORDER DEBUG for {symbol} ===")
        print(f"Stop Loss Pips Input: {stop_loss_pips}")
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print("ERROR: Symbol info not available")
            return
            
        print(f"Symbol Digits: {symbol_info.digits}")
        print(f"Symbol Point: {symbol_info.point}")
        
        # Show pip calculation
        if "JPY" in symbol:
            pip_size = 0.01
            print(f"JPY Pair - Pip Size: {pip_size}")
        else:
            pip_size = 0.0001 if symbol_info.digits == 5 else 0.01
            print(f"Regular Pair - Pip Size: {pip_size}")
        
        sl_distance = stop_loss_pips * pip_size
        print(f"SL Distance: {sl_distance}")
        
        current_price = mt5.symbol_info_tick(symbol)
        if current_price:
            current_price = current_price.ask if order_type.lower() == 'buy' else current_price.bid
            stop_loss_price = current_price - sl_distance if order_type.lower() == 'buy' else current_price + sl_distance
            
            print(f"Current Price: {current_price}")
            print(f"Calculated Stop Loss Price: {stop_loss_price}")
            print(f"Actual SL Distance in Pips: {(abs(current_price - stop_loss_price)) / pip_size}")
        else:
            print("ERROR: Could not get current price")
            
        print("=" * 40)

    async def place_order(self, symbol: str, order_type: str, volume: float,
                         stop_loss: float = None, take_profit: float = None,
                         price: float = None, comment: str = "") -> Dict:
        """Place order through MT5 - ASYNC"""
        try:
            # Check MT5 connection
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                logger.error("MT5 terminal not connected")
                return {'success': False, 'error': 'MT5 terminal not connected'}

            # Select symbol for trading
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}")
                return {'success': False, 'error': f'Failed to select symbol {symbol}'}

            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found")
                return {'success': False, 'error': 'Symbol not found'}

            # Get current price if not provided
            if price is None:
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    logger.error(f"Failed to get tick data for {symbol}")
                    return {'success': False, 'error': f'Failed to get tick data for {symbol}'}
                if order_type.lower() in ['buy', 'buy_limit', 'buy_stop']:
                    price = tick.ask
                else:
                    price = tick.bid

            # Determine MT5 order type
            if order_type.lower() == 'buy':
                mt5_order_type = mt5.ORDER_TYPE_BUY
            elif order_type.lower() == 'sell':
                mt5_order_type = mt5.ORDER_TYPE_SELL
            elif order_type.lower() == 'buy_limit':
                mt5_order_type = mt5.ORDER_TYPE_BUY_LIMIT
            elif order_type.lower() == 'sell_limit':
                mt5_order_type = mt5.ORDER_TYPE_SELL_LIMIT
            elif order_type.lower() == 'buy_stop':
                mt5_order_type = mt5.ORDER_TYPE_BUY_STOP
            elif order_type.lower() == 'sell_stop':
                mt5_order_type = mt5.ORDER_TYPE_SELL_STOP
            else:
                return {'success': False, 'error': f'Unknown order type: {order_type}'}

            # Adjust stop loss to meet minimum requirements
            if stop_loss is not None:
                # Get minimum stop distance from symbol info, with fallback
                stops_level = getattr(symbol_info, 'trade_stops_level', 0)
                min_stop_points = max(stops_level, 100)  # Increased to 100 points minimum
                min_stop_distance = min_stop_points * symbol_info.point
                
                logger.info(f"Symbol {symbol}: stops_level={stops_level}, point={symbol_info.point}, min_stop_points={min_stop_points}, min_stop_distance={min_stop_distance}")
                logger.info(f"Order {order_type} {symbol}: price={price}, original_sl={stop_loss}, min_distance={min_stop_distance}")
                
                if order_type.lower() in ['buy', 'buy_limit', 'buy_stop']:
                    # For buy orders, stop loss should be below price
                    required_sl = price - min_stop_distance
                    if stop_loss >= required_sl:
                        logger.info(f"Adjusting BUY stop loss from {stop_loss} to {required_sl}")
                        stop_loss = required_sl
                else:
                    # For sell orders, stop loss should be above price
                    required_sl = price + min_stop_distance
                    if stop_loss <= required_sl:
                        logger.info(f"Adjusting SELL stop loss from {stop_loss} to {required_sl}")
                        stop_loss = required_sl
                
                logger.info(f"Final stop loss: {stop_loss}")

            # DEBUG: Trace EURJPY stop loss calculation
            if stop_loss is not None and "EURJPY" in symbol:
                print(f"\n=== ORDER DEBUG for {symbol} ===")
                print(f"Order Type: {order_type}")
                print(f"Entry Price: {price}")
                print(f"Stop Loss Price: {stop_loss}")
                
                # Calculate pip size for JPY pairs
                pip_size = 0.01 if "JPY" in symbol else 0.0001
                print(f"Pip Size: {pip_size}")
                
                # Calculate actual pips
                sl_distance = abs(price - stop_loss)
                actual_pips = sl_distance / pip_size
                print(f"SL Distance: {sl_distance}")
                print(f"Actual SL Pips: {actual_pips:.1f}")
                print(f"Symbol Digits: {symbol_info.digits}")
                print(f"Symbol Point: {symbol_info.point}")
                print("=" * 40)

            # Adjust take profit to meet minimum requirements (only if too close to entry)
            if take_profit is not None:
                # Get minimum stop distance from symbol info, with fallback
                stops_level = getattr(symbol_info, 'trade_stops_level', 0)
                min_stop_points = max(stops_level, 100)  # Increased to 100 points minimum
                min_stop_distance = min_stop_points * symbol_info.point
                
                logger.info(f"Order {order_type} {symbol}: price={price}, original_tp={take_profit}, min_distance={min_stop_distance}")
                
                if order_type.lower() in ['buy', 'buy_limit', 'buy_stop']:
                    # For buy orders, take profit should be above price
                    required_tp = price + min_stop_distance
                    if take_profit <= required_tp:
                        logger.info(f"Adjusting BUY take profit from {take_profit} to {required_tp} (broker minimum)")
                        take_profit = required_tp
                else:
                    # For sell orders, take profit should be below price
                    required_tp = price - min_stop_distance
                    if take_profit >= required_tp:
                        logger.info(f"Adjusting SELL take profit from {take_profit} to {required_tp} (broker minimum)")
                        take_profit = required_tp
                
                logger.info(f"Final take profit: {take_profit}")

            # Final validation: ensure adequate risk-reward ratio
            if stop_loss is not None and take_profit is not None:
                risk_distance = abs(stop_loss - price)
                reward_distance = abs(take_profit - price)
                final_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
                
                if final_ratio < 3.0:
                    logger.error(f"Order rejected: insufficient risk-reward ratio {final_ratio:.2f}:1 (required: 3.0:1)")
                    return {'success': False, 'error': f'Insufficient risk-reward ratio {final_ratio:.2f}:1'}
                
                logger.info(f"Order validated: {final_ratio:.1f}:1 risk-reward ratio")

            # CRITICAL FIX: Round prices to symbol's decimal places for JPY pairs
            price = round(price, symbol_info.digits)
            if stop_loss is not None:
                stop_loss = round(stop_loss, symbol_info.digits)
            if take_profit is not None:
                take_profit = round(take_profit, symbol_info.digits)

            # Debug before sending to MT5
            print(f"\n=== SENDING TO MT5 ===")
            print(f"Symbol: {symbol}")
            print(f"Entry: {price}")
            print(f"Stop Loss: {stop_loss}")
            print(f"Take Profit: {take_profit}")
            if stop_loss is not None:
                sl_distance = abs(price - stop_loss)
                print(f"SL Distance: {sl_distance:.5f}")
            print("=" * 30)

            # Create order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5_order_type,
                "price": price,
                "deviation": self.max_slippage,
                "magic": self.magic_number,
                "comment": comment or "FX-Ai",
                "type_time": mt5.ORDER_TIME_GTC,
                # Removed type_filling to avoid "Unsupported filling mode" errors
            }

            # Add stop loss and take profit if provided
            if stop_loss is not None:
                request["sl"] = stop_loss
            if take_profit is not None:
                request["tp"] = take_profit

            # Send order
            result = mt5.order_send(request)

            # Check if order_send returned None
            if result is None:
                logger.error("Order send failed - no response from MT5")
                return {
                    'success': False,
                    'error': 'Order send failed - no response from MT5'
                }

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Order placed: {symbol} {order_type} @ {price}")

                # CRITICAL FIX: Verify what was actually placed
                positions = mt5.positions_get(symbol=symbol)
                if positions:
                    actual_sl = positions[-1].sl
                    print(f"ACTUAL SL SET: {actual_sl}")
                    if stop_loss is not None and abs(actual_sl - stop_loss) > 0.01:
                        print(f"WARNING: SL mismatch! Expected {stop_loss}, got {actual_sl}")
                        logger.warning(f"Stop loss mismatch for {symbol}: expected {stop_loss}, got {actual_sl}")

                # Track the order
                if order_type.lower() in ['buy', 'sell']:
                    self.active_positions[result.order] = {
                        'symbol': symbol,
                        'type': order_type,
                        'volume': volume,
                        'entry': result.price,
                        'sl': stop_loss,
                        'tp': take_profit,
                        'timestamp': datetime.now()
                    }

                return {
                    'success': True,
                    'order': result.order,
                    'price': result.price
                }
            else:
                logger.error(f"Order failed: {result.comment}")
                return {
                    'success': False,
                    'error': result.comment
                }

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def manage_positions(self, symbol: str):
        """Manage open positions for a symbol - ASYNC"""
        try:
            positions = mt5.positions_get(symbol=symbol)

            if positions:
                for position in positions:
                    if position.magic == self.magic_number:
                        # Update take profit based on new analysis first
                        tp_increased = await self.update_take_profit(position)

                        # Only apply breakeven and trailing stops if TP was increased
                        if tp_increased:
                            # Apply breakeven if enabled
                            await self.apply_breakeven(position)

                            # Update trailing stops if needed
                            await self.update_trailing_stop(position)

            # Small delay to prevent CPU overload
            await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error managing positions: {e}")

    async def update_trailing_stop(self, position) -> None:
        """Update trailing stop for a position with adaptive logic - ASYNC"""
        try:
            # Get trailing stop configuration
            trailing_config = self.config.get('risk_management', {}).get('trailing_stop', {})
            if not trailing_config.get('enabled', False):
                return

            base_activation_pips = trailing_config.get('activation_pips', 20)
            base_trail_distance_pips = trailing_config.get('trail_distance_pips', 15)
            
            # Adjust for precious metals (higher volatility)
            if 'XAU' in position.symbol or 'XAG' in position.symbol or 'GOLD' in position.symbol:
                base_activation_pips = max(base_activation_pips, 50)  # Minimum 50 pips activation for metals
                base_trail_distance_pips = max(base_trail_distance_pips, 200)  # Minimum 200 pips for metals

            # Get current price and market data
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                return

            current_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

            # Get recent bars for volatility analysis
            bars = mt5.copy_rates_from_pos(position.symbol, mt5.TIMEFRAME_H1, 0, 50)
            if bars is None or len(bars) < 20:
                # Fallback to basic trailing stop without volatility adjustment
                await self._basic_trailing_stop(position, current_price, base_activation_pips, base_trail_distance_pips)
                return

            # Calculate current ATR for adaptive adjustments
            atr_period = 14
            atr_values = []
            for i in range(atr_period, len(bars)):
                high = bars[i]['high']
                low = bars[i]['low']
                prev_close = bars[i-1]['close']
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                atr_values.append(tr)

            if not atr_values:
                await self._basic_trailing_stop(position, current_price, base_activation_pips, base_trail_distance_pips)
                return

            current_atr = sum(atr_values[-10:]) / len(atr_values[-10:])  # Recent ATR
            avg_atr = sum(atr_values) / len(atr_values)  # Average ATR

            # Adaptive adjustments based on volatility
            volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0

            # Adjust activation and trail distance based on volatility
            if volatility_ratio > 1.2:  # High volatility
                activation_pips = base_activation_pips * 1.5  # Require more profit to activate
                trail_distance_pips = base_trail_distance_pips * 1.3  # Wider trail
            elif volatility_ratio < 0.8:  # Low volatility
                activation_pips = base_activation_pips * 0.8  # Activate earlier
                trail_distance_pips = base_trail_distance_pips * 0.9  # Tighter trail
            else:  # Normal volatility
                activation_pips = base_activation_pips
                trail_distance_pips = base_trail_distance_pips

            # Get correct pip size for the symbol
            symbol_info = mt5.symbol_info(position.symbol)
            if symbol_info:
                point = symbol_info.point
                digits = symbol_info.digits
                if 'XAU' in position.symbol or 'XAG' in position.symbol or 'GOLD' in position.symbol:
                    pip_size = point * 10  # Metals: 1 pip = 10 points (0.1 for 2-digit symbols)
                elif digits == 3 or digits == 5:
                    pip_size = point * 10
                else:
                    pip_size = point
            else:
                pip_size = 0.0001  # Fallback

            # Calculate current profit in pips
            if position.type == mt5.ORDER_TYPE_BUY:
                profit_pips = (current_price - position.price_open) / pip_size
            else:  # SELL
                profit_pips = (position.price_open - current_price) / pip_size

            # Check if trailing stop should be activated
            if profit_pips >= activation_pips:
                # Calculate new stop loss level with adaptive trail distance
                trail_distance = trail_distance_pips * pip_size
                
                # Ensure trail distance meets minimum stop requirements
                min_stop_distance = symbol_info.trade_stops_level * symbol_info.point
                trail_distance = max(trail_distance, min_stop_distance * 2)  # Extra buffer

                if position.type == mt5.ORDER_TYPE_BUY:
                    # For BUY positions, trail below current price
                    new_sl = current_price - trail_distance
                    # Ensure SL doesn't go below entry price (with buffer)
                    min_sl = position.price_open - (2 * symbol_info.trade_stops_level * symbol_info.point)
                    new_sl = max(new_sl, min_sl)
                else:  # SELL
                    # For SELL positions, trail above current price
                    new_sl = current_price + trail_distance
                    # Ensure SL doesn't go above entry price (with buffer)
                    max_sl = position.price_open + (2 * symbol_info.trade_stops_level * symbol_info.point)
                    new_sl = min(new_sl, max_sl)

                # Only update if the new stop loss is better than current stop loss
                current_sl = position.sl
                should_update = False

                if position.type == mt5.ORDER_TYPE_BUY:
                    # For BUY: new SL should be higher than current SL
                    should_update = new_sl > current_sl if current_sl > 0 else True
                else:  # SELL
                    # For SELL: new SL should be lower than current SL
                    should_update = new_sl < current_sl if current_sl > 0 else True

                if should_update:
                    # Update the stop loss
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": position.symbol,
                        "position": position.ticket,
                        "sl": new_sl,
                        "tp": position.tp,  # Keep existing take profit
                        "magic": self.magic_number
                    }

                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"Adaptive trailing stop updated for {position.symbol} position {position.ticket}: "
                                  f"SL moved to {new_sl:.5f} (profit: {profit_pips:.1f} pips, volatility: {volatility_ratio:.2f})")
                    else:
                        logger.warning(f"Failed to update trailing stop for {position.symbol}: {result.comment}")

        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")

    async def apply_breakeven(self, position) -> None:
        """Apply breakeven stop loss once position reaches profit threshold - ASYNC"""
        try:
            # Get breakeven configuration
            breakeven_config = self.config.get('risk_management', {}).get('breakeven', {})
            if not breakeven_config.get('enabled', False):
                return

            activation_pips = breakeven_config.get('activation_pips', 15)

            # Get current price
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                return

            current_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

            # Get symbol info for pip calculation
            symbol_info = mt5.symbol_info(position.symbol)
            if not symbol_info:
                return

            point = symbol_info.point
            digits = symbol_info.digits

            # Determine pip size
            if "XAU" in position.symbol or "XAG" in position.symbol or "GOLD" in position.symbol:
                pip_size = point * 10  # Metals: 1 pip = 10 points (0.1 for 2-digit symbols)
            elif digits == 3 or digits == 5:
                pip_size = point * 10
            else:
                pip_size = point

            # Calculate current profit in pips
            if position.type == mt5.ORDER_TYPE_BUY:
                profit_pips = (current_price - position.price_open) / pip_size
            else:  # SELL
                profit_pips = (position.price_open - current_price) / pip_size

            # Check if breakeven should be applied
            if profit_pips >= activation_pips:
                # Ensure sufficient profit to allow SL at entry without violating broker limits
                min_stop_distance = symbol_info.trade_stops_level * symbol_info.point
                min_stop_pips = min_stop_distance / pip_size
                
                # Need profit >= activation + min_stop to set SL at entry
                required_profit_pips = activation_pips + min_stop_pips
                
                if profit_pips < required_profit_pips:
                    # Not enough profit for breakeven
                    return
                
                # Calculate breakeven SL (entry price)
                breakeven_sl = position.price_open

                # Only update if current SL is worse than breakeven
                current_sl = position.sl
                should_update = False

                if position.type == mt5.ORDER_TYPE_BUY:
                    should_update = breakeven_sl > current_sl if current_sl > 0 else True
                else:  # SELL
                    should_update = breakeven_sl < current_sl if current_sl > 0 else True

                if should_update:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": position.symbol,
                        "position": position.ticket,
                        "sl": breakeven_sl,
                        "tp": position.tp,
                        "magic": self.magic_number
                    }

                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"Breakeven applied for {position.symbol} position {position.ticket}: "
                                  f"SL moved to {breakeven_sl:.5f} (entry price)")
                    else:
                        logger.warning(f"Failed to apply breakeven for {position.symbol}: {result.comment}")

        except Exception as e:
            logger.error(f"Error applying breakeven: {e}")

    async def _basic_trailing_stop(self, position, current_price: float, activation_pips: float, trail_distance_pips: float) -> None:
        """Basic trailing stop fallback when volatility data unavailable"""
        try:
            symbol_info = mt5.symbol_info(position.symbol)
            if not symbol_info:
                return
            
            point = symbol_info.point
            digits = symbol_info.digits
            
            # Determine pip size
            if "XAU" in position.symbol or "XAG" in position.symbol or "GOLD" in position.symbol:
                pip_size = point * 10  # Metals: 1 pip = 10 points (0.1 for 2-digit symbols)
            elif digits == 3 or digits == 5:
                pip_size = point * 10
            else:
                pip_size = point
            
            # Calculate current profit in pips
            if position.type == mt5.ORDER_TYPE_BUY:
                profit_pips = (current_price - position.price_open) / pip_size
            else:  # SELL
                profit_pips = (position.price_open - current_price) / pip_size

            # Check if trailing stop should be activated
            if profit_pips >= activation_pips:
                # Calculate new stop loss level
                trail_distance = trail_distance_pips * pip_size

                if position.type == mt5.ORDER_TYPE_BUY:
                    new_sl = current_price - trail_distance
                    # Ensure SL doesn't go above entry price
                    min_sl = position.price_open - (2 * symbol_info.trade_stops_level * symbol_info.point)
                    new_sl = min(new_sl, min_sl)
                else:  # SELL
                    new_sl = current_price + trail_distance
                    # Ensure SL doesn't go below entry price
                    max_sl = position.price_open + (2 * symbol_info.trade_stops_level * symbol_info.point)
                    new_sl = max(new_sl, max_sl)

                # Only update if the new stop loss is better than current stop loss
                current_sl = position.sl
                should_update = False

                if position.type == mt5.ORDER_TYPE_BUY:
                    should_update = new_sl > current_sl if current_sl > 0 else True
                else:  # SELL
                    should_update = new_sl < current_sl if current_sl > 0 else True

                if should_update:
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": position.symbol,
                        "position": position.ticket,
                        "sl": new_sl,
                        "tp": position.tp
                    }

                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"Basic trailing stop updated for {position.symbol} position {position.ticket}: "
                                  f"SL moved to {new_sl:.5f} (profit: {profit_pips:.1f} pips)")
                    else:
                        logger.warning(f"Failed to update trailing stop for {position.symbol}: {result.comment}")

        except Exception as e:
            logger.error(f"Error in basic trailing stop for {position.symbol}: {e}")

    async def update_take_profit(self, position) -> bool:
        """Update take profit based on new market analysis - ASYNC
        
        Returns:
            bool: True if take profit was increased, False otherwise
        """
        try:
            # Get current market data for analysis
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                return

            current_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

            # Get recent bars for technical analysis
            bars = mt5.copy_rates_from_pos(position.symbol, mt5.TIMEFRAME_H1, 0, 50)
            if bars is None or len(bars) < 20:
                return

            # Calculate current ATR for dynamic TP adjustment
            atr_period = 14
            atr_values = []
            for i in range(atr_period, len(bars)):
                high = bars[i]['high']
                low = bars[i]['low']
                prev_close = bars[i-1]['close']
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                atr_values.append(tr)

            if not atr_values:
                return

            current_atr = sum(atr_values[-10:]) / len(atr_values[-10:])  # Recent ATR

            # Calculate current profit in pips
            if position.type == mt5.ORDER_TYPE_BUY:
                profit_pips = (current_price - position.price_open) / mt5.symbol_info(position.symbol).point
            else:  # SELL
                profit_pips = (position.price_open - current_price) / mt5.symbol_info(position.symbol).point

            # Convert to pips (handle JPY pairs)
            if 'JPY' in position.symbol:
                profit_pips = profit_pips / 100

            # Calculate original risk in pips (distance from entry to SL)
            original_sl = position.sl
            if original_sl > 0:
                if position.type == mt5.ORDER_TYPE_BUY:
                    risk_pips = (original_sl - position.price_open) / mt5.symbol_info(position.symbol).point
                else:  # SELL
                    risk_pips = (position.price_open - original_sl) / mt5.symbol_info(position.symbol).point
                
                if 'JPY' in position.symbol:
                    risk_pips = risk_pips / 100
                
                # Breakeven and trailing stop logic
                # Activate only after reaching 1:3 profit ratio (3x the risk)
                profit_target_pips = risk_pips * 3
                
                if profit_pips >= profit_target_pips:
                    # Activate breakeven and trailing stops
                    await self.apply_breakeven_and_trailing(position, current_price, risk_pips)
                    return True  # Position management activated

            # Dynamic take profit adjustment logic
            current_tp = position.tp
            new_tp = current_tp

            # Base TP on current ATR with adaptive multiplier
            base_tp_multiplier = 4.5  # Start with 3:1 reward ratio (adjusted for minimum SL)

            # Increase multiplier if position is in strong profit (breakeven or better)
            if profit_pips >= 0:
                # Scale up TP multiplier based on profit level
                profit_multiplier = min(1.5, profit_pips / 50)  # Max 50% increase
                base_tp_multiplier *= (1 + profit_multiplier)

                # Calculate new take profit
                if position.type == mt5.ORDER_TYPE_BUY:
                    new_tp = position.price_open + (current_atr * base_tp_multiplier)
                else:  # SELL
                    new_tp = position.price_open - (current_atr * base_tp_multiplier)

                # Ensure new TP meets broker minimum stop requirements
                symbol_info = mt5.symbol_info(position.symbol)
                if symbol_info:
                    min_stop_distance = symbol_info.trade_stops_level * symbol_info.point
                    if position.type == mt5.ORDER_TYPE_BUY:
                        min_tp = current_price + min_stop_distance
                        if new_tp <= min_tp:
                            # TP too close to current price, skip update
                            return False
                    else:  # SELL
                        max_tp = current_price - min_stop_distance
                        if new_tp >= max_tp:
                            # TP too close to current price, skip update
                            return False

                # Only update if new TP is significantly better (at least 10 pips improvement)
                tp_improvement = abs(new_tp - current_tp) / (100 if 'JPY' in position.symbol else 0.0001)

                if tp_improvement >= 10 and new_tp != current_tp:
                    # Update take profit
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": position.symbol,
                        "position": position.ticket,
                        "sl": position.sl,  # Keep existing stop loss
                        "tp": new_tp,
                        "magic": self.magic_number
                    }

                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"Take profit adjusted for {position.symbol} position {position.ticket}: "
                                  f"TP moved to {new_tp:.5f} (ATR: {current_atr:.5f}, multiplier: {base_tp_multiplier:.1f})")
                        return new_tp > current_tp  # Return True if TP was increased
                    else:
                        logger.warning(f"Failed to update take profit for {position.symbol}: {result.comment}")
                        return False

        except Exception as e:
            logger.error(f"Error updating take profit: {e}")
            return False

    async def apply_breakeven_and_trailing(self, position, current_price: float, risk_pips: float) -> None:
        """Apply breakeven and trailing stop logic after reaching 1:3 profit"""
        try:
            symbol_info = mt5.symbol_info(position.symbol)
            if not symbol_info:
                return
            
            new_sl = position.sl
            min_stop_distance = max(symbol_info.trade_stops_level, 10) * symbol_info.point  # Minimum broker distance
            
            # Breakeven: Move SL to entry price + small buffer once at 1:1 profit
            breakeven_pips = risk_pips  # 1:1 profit level
            buffer_pips = max(risk_pips * 0.2, 5)  # At least 5 pips buffer, or 20% of risk
            
            if position.type == mt5.ORDER_TYPE_BUY:
                # For BUY: breakeven at entry + buffer
                breakeven_level = position.price_open + (buffer_pips * symbol_info.point)
                if 'JPY' in position.symbol:
                    breakeven_level = position.price_open + (buffer_pips * symbol_info.point * 100)
                
                # Ensure breakeven level meets broker minimum from current price
                min_breakeven = current_price - min_stop_distance
                breakeven_level = min(breakeven_level, min_breakeven)
                
                if current_price >= breakeven_level and position.sl < breakeven_level:
                    new_sl = breakeven_level
                    logger.info(f"Breakeven activated for {position.symbol} BUY position {position.ticket}: SL moved to {new_sl:.5f}")
            else:  # SELL
                # For SELL: breakeven at entry - buffer
                breakeven_level = position.price_open - (buffer_pips * symbol_info.point)
                if 'JPY' in position.symbol:
                    breakeven_level = position.price_open - (buffer_pips * symbol_info.point * 100)
                
                # Ensure breakeven level meets broker minimum from current price
                max_breakeven = current_price + min_stop_distance
                breakeven_level = max(breakeven_level, max_breakeven)
                
                if current_price <= breakeven_level and position.sl > breakeven_level:
                    new_sl = breakeven_level
                    logger.info(f"Breakeven activated for {position.symbol} SELL position {position.ticket}: SL moved to {new_sl:.5f}")
            
            # Trailing stop: Trail behind current price with adaptive distance
            trail_distance_pips = max(risk_pips * 0.7, 10)  # At least 10 pips, or 70% of original risk
            
            if position.type == mt5.ORDER_TYPE_BUY:
                trail_level = current_price - (trail_distance_pips * symbol_info.point)
                if 'JPY' in position.symbol:
                    trail_level = current_price - (trail_distance_pips * symbol_info.point * 100)
                
                # Ensure trail level is above current SL and meets broker minimum
                trail_level = max(trail_level, position.sl + min_stop_distance)
                
                if trail_level > new_sl:
                    new_sl = trail_level
                    logger.info(f"Trailing stop activated for {position.symbol} BUY position {position.ticket}: SL moved to {new_sl:.5f}")
            else:  # SELL
                trail_level = current_price + (trail_distance_pips * symbol_info.point)
                if 'JPY' in position.symbol:
                    trail_level = current_price + (trail_distance_pips * symbol_info.point * 100)
                
                # Ensure trail level is below current SL and meets broker minimum
                trail_level = min(trail_level, position.sl - min_stop_distance)
                
                if trail_level < new_sl:
                    new_sl = trail_level
                    logger.info(f"Trailing stop activated for {position.symbol} SELL position {position.ticket}: SL moved to {new_sl:.5f}")
            
            # Apply the stop loss update if it changed and meets broker requirements
            # Check that new SL meets minimum distance from current price
            if position.type == mt5.ORDER_TYPE_BUY:
                min_allowed_sl = current_price - min_stop_distance
                sl_valid = new_sl <= min_allowed_sl
            else:  # SELL
                min_allowed_sl = current_price + min_stop_distance
                sl_valid = new_sl >= min_allowed_sl
            
            if abs(new_sl - position.sl) > symbol_info.point and sl_valid:  # Only update if changed and valid
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": position.symbol,
                    "position": position.ticket,
                    "sl": new_sl,
                    "tp": position.tp,  # Keep existing take profit
                    "magic": self.magic_number
                }
                
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Stop loss updated for {position.symbol} position {position.ticket}: SL = {new_sl:.5f}")
                else:
                    logger.warning(f"Failed to update stop loss for {position.symbol}: {result.comment}")
                    
        except Exception as e:
            logger.error(f"Error applying breakeven/trailing stops: {e}")

    async def close_all_positions(self) -> None:
        """Close all open positions - PROPERLY ASYNC"""
        try:
            positions = mt5.positions_get()

            if positions is None or len(positions) == 0:
                logger.info("No positions to close")
                return

            for position in positions:
                if position.magic == self.magic_number:
                    await self.close_position(position)

            logger.info(f"Closed {len(positions)} positions")

        except Exception as e:
            logger.error(f"Error closing all positions: {e}")

    async def close_position(self, position) -> bool:
        """Close a single position - ASYNC"""
        try:
            # Select symbol for trading
            if not mt5.symbol_select(position.symbol, True):
                logger.error(f"Failed to select symbol {position.symbol} for closing")
                return False

            # Determine order type for closing
            if position.type == mt5.ORDER_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(position.symbol).bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(position.symbol).ask

            # Create close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": position.ticket,
                "price": price,
                "deviation": self.max_slippage,
                "magic": self.magic_number,
                "comment": "FX-Ai close",
                "type_time": mt5.ORDER_TIME_GTC,
                # Removed type_filling to avoid "Unsupported filling mode" errors
            }

            # Send close order
            result = mt5.order_send(request)

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Position closed: {position.symbol} ticket {position.ticket}")

                # Remove from active positions
                if position.ticket in self.active_positions:
                    del self.active_positions[position.ticket]

                return True
            else:
                logger.error(f"Failed to close position: {result.comment}")
                return False

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False

    def get_position_by_ticket(self, ticket: int):
        """Get position by ticket number"""
        positions = mt5.positions_get(ticket=ticket)
        if positions and len(positions) > 0:
            return positions[0]
        return None

    def get_trade_history(self, ticket: int) -> Optional[Dict]:
        """Get trade history for a ticket"""
        try:
            # Get deals for this position
            deals = mt5.history_deals_get(position=ticket)

            if deals and len(deals) >= 2:  # Need open and close deals
                open_deal = deals[0]
                close_deal = deals[-1]

                return {
                    'ticket': ticket,
                    'symbol': open_deal.symbol,
                    'entry_price': open_deal.price,
                    'exit_price': close_deal.price,
                    'volume': open_deal.volume,
                    'profit': close_deal.profit,
                    'commission': close_deal.commission,
                    'swap': close_deal.swap,
                    'open_time': datetime.fromtimestamp(open_deal.time),
                    'close_time': datetime.fromtimestamp(close_deal.time)
                }

            return None

        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return None

    def get_performance_metrics(self) -> Dict:
        """Calculate and return performance metrics"""
        # This would analyze trade history
        return {
            'daily_pnl': 0.0,  # Placeholder
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': len(self.active_positions),
            'active_positions': len(self.active_positions)
        }