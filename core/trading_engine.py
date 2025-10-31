"""
FX-Ai Trading Engine - Fixed Version
Handles trade execution with proper async/await implementation
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Optional
import MetaTrader5 as mt5
from utils.position_monitor import PositionMonitor
from utils.risk_validator import RiskValidator

logger = logging.getLogger(__name__)


class TradingEngine:
    """Trading engine with fixed async methods"""

    def __init__(
            self,
            mt5_connector,
            risk_manager,
            technical_analyzer,
            sentiment_analyzer,
            fundamental_collector,
            ml_predictor,
            adaptive_learning_manager=None):
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
        self.config = getattr(risk_manager, 'config', {}
                              ) if risk_manager else {}

        # Trading parameters
        self.magic_number = self.config.get(
            'trading', {}).get('magic_number', 20241029)
        self.max_slippage = self.config.get(
            'trading', {}).get('max_slippage', 3)

        # Initialize position monitor for change detection
        self.position_monitor = PositionMonitor(self.magic_number)
        self.position_monitor.enable_alerts(True)

        logger.info("Trading Engine initialized with position monitoring")

    def get_filling_mode(self, symbol):
        """Get the correct filling mode for a symbol"""
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.warning(
                f"Failed to get symbol info for {symbol}, using FOK as fallback")
            return mt5.ORDER_FILLING_FOK

        filling = info.filling_mode
        logger.debug(f"Symbol {symbol} filling modes supported: {filling}")

        if filling & 1:  # ORDER_FILLING_FOK
            logger.debug(f"Using ORDER_FILLING_FOK for {symbol}")
            return mt5.ORDER_FILLING_FOK
        elif filling & 2:  # ORDER_FILLING_IOC
            logger.debug(f"Using ORDER_FILLING_IOC for {symbol}")
            return mt5.ORDER_FILLING_IOC
        elif filling & 4:  # ORDER_FILLING_RETURN
            logger.debug(f"Using ORDER_FILLING_RETURN for {symbol}")
            return mt5.ORDER_FILLING_RETURN
        else:
            logger.warning(
                f"No filling mode detected for {symbol}, using FOK as fallback")
            return mt5.ORDER_FILLING_FOK

    def debug_stop_loss_calculation(
            self, symbol: str, order_type: str, stop_loss_pips: float):
        """Debug function to trace stop loss calculation issues"""

        logger.debug(f"=== ORDER DEBUG for {symbol} ===")
        logger.debug(f"Stop Loss Pips Input: {stop_loss_pips}")

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error("Symbol info not available")
            return

        logger.debug(f"Symbol Digits: {symbol_info.digits}")
        logger.debug(f"Symbol Point: {symbol_info.point}")

        # Show pip calculation
        if "JPY" in symbol:
            pip_size = 0.01
            logger.debug(f"JPY Pair - Pip Size: {pip_size}")
        else:
            pip_size = 0.0001 if symbol_info.digits == 5 else 0.01
            logger.debug(f"Regular Pair - Pip Size: {pip_size}")

        sl_distance = stop_loss_pips * pip_size
        logger.debug(f"SL Distance: {sl_distance}")

        current_price = mt5.symbol_info_tick(symbol)
        if current_price:
            current_price = current_price.ask if order_type.lower() == 'buy' else current_price.bid
            stop_loss_price = current_price - \
                sl_distance if order_type.lower() == 'buy' else current_price + sl_distance

            logger.debug(f"Current Price: {current_price}")
            logger.debug(f"Calculated Stop Loss Price: {stop_loss_price}")
            logger.debug(
                f"Actual SL Distance in Pips: {(abs(current_price - stop_loss_price)) / pip_size}")
        else:
            logger.error("Could not get current price")

        logger.debug("=" * 40)

    async def place_order(self, symbol: str, order_type: str, volume: float,
                          stop_loss: float = None, take_profit: float = None,
                          price: float = None, comment: str = "") -> Dict:
        """Place order through MT5 - ASYNC"""
        try:
            # Check MT5 connection
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                logger.error("MT5 terminal not connected")
                return {
                    'success': False,
                    'error': 'MT5 terminal not connected'}

            # Select symbol for trading
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}")
                return {
                    'success': False,
                    'error': f'Failed to select symbol {symbol}'}

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
                    return {'success': False,
                            'error': f'Failed to get tick data for {symbol}'}
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
                return {'success': False,
                        'error': f'Unknown order type: {order_type}'}

            # Adjust stop loss to meet minimum requirements
            if stop_loss is not None:
                # Get minimum stop distance from symbol info, with fallback
                stops_level = getattr(symbol_info, 'trade_stops_level', 0)

                # Calculate minimum stop distance in PIPS, not points
                # For JPY pairs: 1 pip = 0.01, for others: 1 pip = 0.0001
                if "JPY" in symbol:
                    pip_size = 0.01
                    # Convert points to pips, minimum 25 pips
                    min_stop_pips = max(stops_level / 10, 25)
                else:
                    pip_size = 0.0001
                    # Convert points to pips, minimum 25 pips
                    min_stop_pips = max(stops_level / 10, 25)

                min_stop_distance = min_stop_pips * pip_size

                logger.info(
                    f"Symbol {symbol}: stops_level={stops_level}, pip_size={pip_size}, min_stop_pips={min_stop_pips}, min_stop_distance={min_stop_distance}")
                logger.info(
                    f"Order {order_type} {symbol}: price={price}, original_sl={stop_loss}, min_distance={min_stop_distance}")

                if order_type.lower() in ['buy', 'buy_limit', 'buy_stop']:
                    # For buy orders, stop loss should be below price
                    required_sl = price - min_stop_distance
                    if stop_loss >= required_sl:
                        logger.info(
                            f"Adjusting BUY stop loss from {stop_loss} to {required_sl}")
                        stop_loss = required_sl
                else:
                    # For sell orders, stop loss should be above price
                    required_sl = price + min_stop_distance
                    if stop_loss <= required_sl:
                        logger.info(
                            f"Adjusting SELL stop loss from {stop_loss} to {required_sl}")
                        stop_loss = required_sl

                logger.info(f"Final stop loss: {stop_loss}")

            # DEBUG: Trace EURJPY stop loss calculation
            if stop_loss is not None and "EURJPY" in symbol:
                logger.debug(f"=== ORDER DEBUG for {symbol} ===")
                logger.debug(f"Order Type: {order_type}")
                logger.debug(f"Entry Price: {price}")
                logger.debug(f"Stop Loss Price: {stop_loss}")

                # Calculate pip size for JPY pairs
                pip_size = 0.01 if "JPY" in symbol else 0.0001
                logger.debug(f"Pip Size: {pip_size}")

                # Calculate actual pips
                sl_distance = abs(price - stop_loss)
                actual_pips = sl_distance / pip_size
                logger.debug(f"SL Distance: {sl_distance}")
                logger.debug(f"Actual SL Pips: {actual_pips:.1f}")
                logger.debug(f"Symbol Digits: {symbol_info.digits}")
                logger.debug(f"Symbol Point: {symbol_info.point}")
                logger.debug("=" * 40)

            # Adjust take profit to meet minimum requirements (only if too
            # close to entry)
            if take_profit is not None:
                # Get minimum stop distance from symbol info, with fallback
                stops_level = getattr(symbol_info, 'trade_stops_level', 0)

                # Calculate minimum stop distance in PIPS, not points
                # For JPY pairs: 1 pip = 0.01, for others: 1 pip = 0.0001
                if "JPY" in symbol:
                    pip_size = 0.01
                    # Convert points to pips, minimum 25 pips
                    min_stop_pips = max(stops_level / 10, 25)
                else:
                    pip_size = 0.0001
                    # Convert points to pips, minimum 25 pips
                    min_stop_pips = max(stops_level / 10, 25)

                min_stop_distance = min_stop_pips * pip_size

                logger.info(
                    f"Order {order_type} {symbol}: price={price}, original_tp={take_profit}, min_distance={min_stop_distance}")

                if order_type.lower() in ['buy', 'buy_limit', 'buy_stop']:
                    # For buy orders, take profit should be above price
                    required_tp = price + min_stop_distance
                    if take_profit <= required_tp:
                        logger.info(
                            f"Adjusting BUY take profit from {take_profit} to {required_tp} (broker minimum)")
                        take_profit = required_tp
                else:
                    # For sell orders, take profit should be below price
                    required_tp = price - min_stop_distance
                    if take_profit >= required_tp:
                        logger.info(
                            f"Adjusting SELL take profit from {take_profit} to {required_tp} (broker minimum)")
                        take_profit = required_tp

                logger.info(f"Final take profit: {take_profit}")

            # Final validation: ensure adequate risk-reward ratio
            if stop_loss is not None and take_profit is not None:
                risk_distance = abs(stop_loss - price)
                reward_distance = abs(take_profit - price)
                final_ratio = reward_distance / risk_distance if risk_distance > 0 else 0

                if final_ratio < 3.0:
                    logger.error(
                        f"Order rejected: insufficient risk-reward ratio {final_ratio:.2f}:1 (required: 3.0:1)")
                    return {
                        'success': False, 'error': f'Insufficient risk-reward ratio {final_ratio:.2f}:1'}

                logger.info(
                    f"Order validated: {final_ratio:.1f}:1 risk-reward ratio")

            # CRITICAL FIX: Round prices to symbol's decimal places for JPY
            # pairs
            price = round(price, symbol_info.digits)
            if stop_loss is not None:
                stop_loss = round(stop_loss, symbol_info.digits)
            if take_profit is not None:
                take_profit = round(take_profit, symbol_info.digits)

            # FIX 2: Check for broker minimum stop distance restrictions
            if stop_loss is not None:
                min_stop_points = getattr(symbol_info, 'trade_stops_level', 0)
                min_stop_distance = max(
                    min_stop_points * symbol_info.point,
                    0.0001)  # Minimum 1 point

                actual_sl_distance = abs(price - stop_loss)
                if actual_sl_distance < min_stop_distance:
                    logger.debug(
                        f"⚠️  BROKER MINIMUM STOP: Required "
                        f"{min_stop_distance:.5f}, have "
                        f"{actual_sl_distance:.5f}")
                    # Adjust to meet broker minimum
                    if order_type.lower() in ['buy', 'buy_limit', 'buy_stop']:
                        stop_loss = price - min_stop_distance
                    else:
                        stop_loss = price + min_stop_distance
                    stop_loss = round(stop_loss, symbol_info.digits)
                    logger.debug(f"Adjusted SL to: {stop_loss}")

            if take_profit is not None:
                min_stop_points = getattr(symbol_info, 'trade_stops_level', 0)
                min_stop_distance = max(
                    min_stop_points * symbol_info.point, 0.0001)

                actual_tp_distance = abs(price - take_profit)
                if actual_tp_distance < min_stop_distance:
                    logger.debug(
                        f"⚠️  BROKER MINIMUM TP: Required "
                        f"{min_stop_distance:.5f}, have "
                        f"{actual_tp_distance:.5f}")
                    # Adjust to meet broker minimum
                    if order_type.lower() in ['buy', 'buy_limit', 'buy_stop']:
                        take_profit = price + min_stop_distance
                    else:
                        take_profit = price - min_stop_distance
                    take_profit = round(take_profit, symbol_info.digits)
                    logger.debug(f"Adjusted TP to: {take_profit}")

            # CRITICAL DEBUG: Enhanced diagnostic for EURJPY SL bug
            if "EURJPY" in symbol and stop_loss is not None:
                logger.debug(f"\n{'='*60}")
                logger.debug("🚨 CRITICAL DEBUG: EURJPY ORDER PLACEMENT")
                logger.debug(f"Symbol: {symbol}")
                logger.debug(f"Order Type: {order_type}")
                logger.debug(f"Entry Price: {price} (type: {type(price)})")
                logger.debug(
                    f"Stop Loss Price: {stop_loss} (type: {type(stop_loss)})")
                logger.debug(
                    f"Take Profit Price: {take_profit} (type: "
                    f"{type(take_profit) if take_profit else None})")

                # Calculate what we expect
                sl_distance = abs(price - stop_loss)
                pip_size = 0.01 if "JPY" in symbol else 0.0001
                expected_pips = sl_distance / pip_size
                logger.debug(f"Expected SL Distance: {sl_distance:.5f}")
                logger.debug(f"Pip Size: {pip_size}")
                logger.debug(f"Expected SL Pips: {expected_pips:.1f}")
                logger.debug(f"Symbol Digits: {symbol_info.digits}")
                logger.debug(f"Symbol Point: {symbol_info.point}")
                logger.debug(f"{'='*60}")

            # Debug before sending to MT5
            logger.debug("=== SENDING TO MT5 ===")
            logger.debug(f"Symbol: {symbol}")
            logger.debug(f"Entry: {price}")
            logger.debug(f"Stop Loss: {stop_loss}")
            logger.debug(f"Take Profit: {take_profit}")
            if stop_loss is not None:
                sl_distance = abs(price - stop_loss)
                logger.debug(f"SL Distance: {sl_distance:.5f}")
            logger.debug("=" * 30)

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
                # Dynamic filling mode
                "type_filling": self.get_filling_mode(symbol),
            }

            # Add stop loss and take profit if provided
            if stop_loss is not None:
                request["sl"] = stop_loss
            if take_profit is not None:
                request["tp"] = take_profit

            # CRITICAL DEBUG: Print the exact request being sent
            logger.debug("REQUEST BEING SENT TO MT5:")
            for key, value in request.items():
                logger.debug(f"  {key}: {value}")
            logger.debug(f"REQUEST SL FIELD TYPE: {type(request.get('sl'))}")
            logger.debug(f"REQUEST SL FIELD VALUE: {request.get('sl')}")
            logger.debug("=" * 50)

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

                # ALTERNATIVE FIX: Use TRADE_ACTION_SLTP to set SL/TP after order placement
                # This ensures SL/TP are set correctly even if initial order
                # had issues
                if stop_loss is not None or take_profit is not None:
                    import time
                    time.sleep(0.2)  # Brief pause before modifying

                    # CRITICAL FIX: Get the actual position ticket, not order
                    # ticket
                    positions = mt5.positions_get(symbol=symbol)
                    if positions:
                        # Get the most recent position for this symbol
                        position = positions[-1]  # Last position
                        position_ticket = position.ticket

                        modify_request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": position_ticket,  # Use actual position ticket
                            "symbol": symbol,
                        }

                        if stop_loss is not None:
                            modify_request["sl"] = stop_loss
                        if take_profit is not None:
                            modify_request["tp"] = take_profit

                        logger.debug(
                            f"Modifying position {position_ticket} with SLTP: {modify_request}")
                        modify_result = mt5.order_send(modify_request)

                        if modify_result and modify_result.retcode != mt5.TRADE_RETCODE_DONE:
                            logger.debug(
                                f"⚠️  WARNING: SLTP modification failed: "
                                f"{modify_result.comment}")
                            logger.warning(
                                f"Failed to set SL/TP for position {position_ticket}: "
                                f"{modify_result.comment}")
                        else:
                            logger.debug("✅ SLTP modification successful")
                    else:
                        logger.debug(
                            "âš ï¸  WARNING: No position found to modify SL/TP")

                # CRITICAL FIX: Enhanced verification with detailed diagnostics
                import time
                time.sleep(0.5)  # Wait for position to register

                positions = mt5.positions_get(symbol=symbol)
                if positions:
                    actual_position = positions[-1]
                    actual_sl = actual_position.sl
                    actual_tp = actual_position.tp

                    logger.debug("\nðŸ” ORDER VERIFICATION:")
                    logger.debug(f"Requested SL: {stop_loss}")
                    logger.debug(f"MT5 Set SL: {actual_sl}")
                    logger.debug(f"Requested TP: {take_profit}")
                    logger.debug(f"MT5 Set TP: {actual_tp}")

                    if stop_loss is not None:
                        sl_mismatch = abs(actual_sl - stop_loss)
                        logger.debug(f"SL Mismatch: {sl_mismatch:.5f}")

                        if sl_mismatch > 0.01:
                            logger.debug(
                                "ðŸš¨ CRITICAL BUG: MT5 ignored our SL!")
                            logger.debug(
                                f"Expected: {stop_loss}, Got: {actual_sl}")
                            logger.debug(
                                f"Difference: {sl_mismatch:.5f} price units")

                            # Calculate what MT5 thinks the pips are
                            if "JPY" in symbol:
                                actual_pips = sl_mismatch / 0.01
                            else:
                                actual_pips = sl_mismatch / 0.0001
                            logger.debug(
                                f"This equals ~{actual_pips:.1f} pips difference")

                            logger.error(
                                f"Stop loss mismatch for {symbol}: expected "
                                f"{stop_loss}, got {actual_sl}")
                        else:
                            logger.debug("✅ SL set correctly")

                    if take_profit is not None:
                        tp_mismatch = abs(actual_tp - take_profit)
                        if tp_mismatch > 0.01:
                            logger.debug(
                                f"ðŸš¨ WARNING: TP mismatch! Expected {take_profit}, got {actual_tp}")

                    logger.debug("=" * 50)
                else:
                    logger.debug(
                        "âš ï¸  WARNING: No positions found after order placement!")

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
                        # FIRST: Check position monitor for unexpected changes
                        alerts = await self.position_monitor.check_positions()
                        if alerts:
                            for alert in alerts:
                                if "VERY TIGHT SL" in alert and str(
                                        position.ticket) in alert:
                                    logger.error(
                                        f"ðŸš¨ PREVENTING AUTOMATED MANAGEMENT: {alert}")
                                    logger.error(
                                        "Position has suspiciously tight SL - skipping automated management")
                                    continue

                        # SECOND: Validate position integrity before management
                        validation = RiskValidator.comprehensive_position_check(
                            position)
                        if not validation['overall_valid']:
                            logger.error(
                                f"🚨 POSITION VALIDATION FAILED for "
                                f"{position.symbol} position "
                                f"{position.ticket}:")
                            for issue in validation['issues']:
                                logger.error(f"  • {issue}")
                            logger.error(
                                "Skipping automated management due to validation failures")
                            continue

                        # THIRD: Check if position was manually modified (skip
                        # automated management)
                        time_since_update = position.time_update - position.time
                        manual_modification_threshold = 600  # 10 minutes in seconds

                        if time_since_update > manual_modification_threshold:
                            logger.info(
                                f"Skipping automated management for "
                                f"{position.symbol} position "
                                f"{position.ticket} "
                                f"(appears manually modified "
                                f"{time_since_update/3600:.1f} hours ago)")
                            continue

                        # Update take profit based on new analysis first
                        tp_increased = await self.update_take_profit(position)

                        # Check for adaptive learning updates to SL/TP
                        await self.check_adaptive_sl_tp_adjustment(position)

                        # Only apply breakeven and trailing stops if TP was
                        # increased
                        if tp_increased:
                            # Apply breakeven if enabled
                            await self.apply_breakeven(position)

                            # Update trailing stops if needed
                            await self.update_trailing_stop(position)

            # Small delay to prevent CPU overload
            await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error managing positions: {e}")

    async def check_adaptive_sl_tp_adjustment(self, position):
        """Check if position SL/TP should be adjusted based on adaptive learning"""
        try:
            if not hasattr(
                    self, 'adaptive_learning') or not self.adaptive_learning:
                return

            # Convert position time to datetime
            trade_timestamp = datetime.fromtimestamp(position.time)

            # Check if there are updated SL/TP parameters for this symbol
            adjustment_needed = self.adaptive_learning.should_adjust_existing_trade(
                position.symbol, position.sl, position.tp, trade_timestamp)

            if not adjustment_needed.get('should_adjust', False):
                return

            logger.info(
                f"Adaptive learning suggests SL/TP adjustment for "
                f"{position.symbol} position "
                f"{position.ticket}")

            # Get current market data to calculate new SL/TP levels
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                return

            current_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

            # Get ATR for calculating new levels
            bars = mt5.copy_rates_from_pos(
                position.symbol, mt5.TIMEFRAME_H1, 0, 20)
            if bars is None or len(bars) < 14:
                return

            # Calculate current ATR
            atr_values = []
            for i in range(1, len(bars)):
                high = bars[i]['high']
                low = bars[i]['low']
                prev_close = bars[i-1]['close']
                tr = max(high - low, abs(high - prev_close),
                         abs(low - prev_close))
                atr_values.append(tr)

            if not atr_values:
                return

            current_atr = sum(atr_values) / len(atr_values)

            # Calculate new SL/TP using optimized multipliers
            sl_multiplier = adjustment_needed['new_sl_atr_multiplier']
            tp_multiplier = adjustment_needed['new_tp_atr_multiplier']

            if position.type == mt5.ORDER_TYPE_BUY:
                new_sl = position.price_open - (current_atr * sl_multiplier)
                new_tp = position.price_open + (current_atr * tp_multiplier)
            else:  # SELL
                new_sl = position.price_open + (current_atr * sl_multiplier)
                new_tp = position.price_open - (current_atr * tp_multiplier)

            # Validate new levels meet broker requirements
            symbol_info = mt5.symbol_info(position.symbol)
            if symbol_info:
                min_stop_distance = symbol_info.trade_stops_level * symbol_info.point

                # Ensure SL is not too close to current price
                if position.type == mt5.ORDER_TYPE_BUY:
                    min_sl = current_price - min_stop_distance
                    if new_sl >= min_sl:
                        logger.debug(
                            f"New SL {new_sl:.5f} too close to current price "
                            f"{current_price:.5f}, skipping adjustment")
                        return
                else:  # SELL
                    max_sl = current_price + min_stop_distance
                    if new_sl <= max_sl:
                        logger.debug(
                            f"New SL {new_sl:.5f} too close to current price "
                            f"{current_price:.5f}, skipping adjustment")
                        return

                # Ensure TP is not too close to current price
                if position.type == mt5.ORDER_TYPE_BUY:
                    min_tp = current_price + min_stop_distance
                    if new_tp <= min_tp:
                        logger.debug(
                            f"New TP {new_tp:.5f} too close to current price "
                            f"{current_price:.5f}, skipping adjustment")
                        return
                else:  # SELL
                    max_tp = current_price - min_stop_distance
                    if new_tp >= max_tp:
                        logger.debug(
                            f"New TP {new_tp:.5f} too close to current price "
                            f"{current_price:.5f}, skipping adjustment")
                        return

            # Only adjust if the new levels are significantly different
            sl_change = abs(new_sl - position.sl)
            tp_change = abs(new_tp - position.tp)

            min_change_pips = 10  # Minimum 10 pips change to warrant adjustment
            min_change_distance = min_change_pips * \
                (100 if 'JPY' in position.symbol else 0.0001)

            if sl_change < min_change_distance and tp_change < min_change_distance:
                logger.debug(
                    f"SL/TP changes too small for {position.symbol}: "
                    f"SL Δ{sl_change:.5f}, TP Δ{tp_change:.5f}")
                return

            # Record the adjustment before making it
            self.adaptive_learning.record_position_adjustment(
                position.ticket,
                position.symbol,
                position.sl,
                position.tp,
                new_sl,
                new_tp,
                f"Adaptive learning update (confidence: {adjustment_needed['confidence']:.2f})"
            )

            # Apply the adjustment
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": position.ticket,
                "sl": new_sl,
                "tp": new_tp,
                "magic": self.magic_number
            }

            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"Adaptive SL/TP adjustment applied for "
                    f"{position.symbol} position "
                    f"{position.ticket}: "
                    f"SL: {position.sl:.5f} → {new_sl:.5f}, "
                    f"TP: {position.tp:.5f} → {new_tp:.5f} "
                    f"(ATR: {current_atr:.5f}, SL mult: {sl_multiplier:.2f}, "
                    f"TP mult: {tp_multiplier:.2f})")
            else:
                logger.warning(
                    f"Failed to apply adaptive SL/TP adjustment for {position.symbol}: {result.comment}")

        except Exception as e:
            logger.error(
                f"Error in adaptive SL/TP adjustment for {position.symbol}: {e}")

    async def update_trailing_stop(self, position) -> None:
        """Update trailing stop for a position with adaptive logic - ASYNC"""
        try:
            # Get trailing stop configuration
            trailing_config = self.config.get(
                'risk_management', {}).get('trailing_stop', {})
            if not trailing_config.get('enabled', False):
                return

            base_activation_pips = trailing_config.get('activation_pips', 20)
            base_trail_distance_pips = trailing_config.get(
                'trail_distance_pips', 15)

            # Adjust for precious metals (higher volatility)
            if 'XAU' in position.symbol or 'XAG' in position.symbol or 'GOLD' in position.symbol:
                # Minimum 50 pips activation for metals
                base_activation_pips = max(base_activation_pips, 50)
                # Minimum 200 pips for metals
                base_trail_distance_pips = max(base_trail_distance_pips, 200)

            # Get current price and market data
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                return

            current_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

            # Get recent bars for volatility analysis
            bars = mt5.copy_rates_from_pos(
                position.symbol, mt5.TIMEFRAME_H1, 0, 50)
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
                tr = max(high - low, abs(high - prev_close),
                         abs(low - prev_close))
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
                    # Metals: 1 pip = 10 points (0.1 for 2-digit symbols)
                    pip_size = point * 10
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
                trail_distance = max(
                    trail_distance, min_stop_distance * 2)  # Extra buffer

                if position.type == mt5.ORDER_TYPE_BUY:
                    # For BUY positions, trail below current price
                    new_sl = current_price - trail_distance
                    # Ensure SL doesn't go below entry price (with buffer)
                    min_sl = position.price_open - \
                        (2 * symbol_info.trade_stops_level * symbol_info.point)
                    new_sl = max(new_sl, min_sl)
                else:  # SELL
                    # For SELL positions, trail above current price
                    new_sl = current_price + trail_distance
                    # Ensure SL doesn't go above entry price (with buffer)
                    max_sl = position.price_open + \
                        (2 * symbol_info.trade_stops_level * symbol_info.point)
                    new_sl = min(new_sl, max_sl)

                # Only update if the new stop loss is better than current stop
                # loss
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
                        logger.info(
                            f"Adaptive trailing stop updated for "
                            f"{position.symbol} position "
                            f"{position.ticket}: "
                            f"SL moved to {new_sl:.5f} "
                            f"(profit: {profit_pips:.1f} pips, "
                            f"volatility: {volatility_ratio:.2f})")
                    else:
                        logger.warning(
                            f"Failed to update trailing stop for "
                            f"{position.symbol}: {result.comment}")

        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")

    async def apply_breakeven(self, position) -> None:
        """Apply breakeven stop loss once position reaches profit threshold - ASYNC"""
        try:
            # Get breakeven configuration
            breakeven_config = self.config.get(
                'risk_management', {}).get('breakeven', {})
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
                # Metals: 1 pip = 10 points (0.1 for 2-digit symbols)
                pip_size = point * 10
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
                # Ensure sufficient profit to allow SL at entry without
                # violating broker limits
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
                        logger.info(
                            f"Breakeven applied for {position.symbol} "
                            f"position {position.ticket}: "
                            f"SL moved to {breakeven_sl:.5f} (entry price)")
                    else:
                        logger.warning(
                            f"Failed to apply breakeven for "
                            f"{position.symbol}: {result.comment}")

        except Exception as e:
            logger.error(f"Error applying breakeven: {e}")

    async def _basic_trailing_stop(
            self,
            position,
            current_price: float,
            activation_pips: float,
            trail_distance_pips: float) -> None:
        """Basic trailing stop fallback when volatility data unavailable"""
        try:
            symbol_info = mt5.symbol_info(position.symbol)
            if not symbol_info:
                return

            point = symbol_info.point
            digits = symbol_info.digits

            # Determine pip size
            if "XAU" in position.symbol or "XAG" in position.symbol or "GOLD" in position.symbol:
                # Metals: 1 pip = 10 points (0.1 for 2-digit symbols)
                pip_size = point * 10
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
                    min_sl = position.price_open - \
                        (2 * symbol_info.trade_stops_level * symbol_info.point)
                    new_sl = min(new_sl, min_sl)
                else:  # SELL
                    new_sl = current_price + trail_distance
                    # Ensure SL doesn't go below entry price
                    max_sl = position.price_open + \
                        (2 * symbol_info.trade_stops_level * symbol_info.point)
                    new_sl = max(new_sl, max_sl)

                # Only update if the new stop loss is better than current stop
                # loss
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
                        logger.info(
                            f"Basic trailing stop updated for {position.symbol} position {position.ticket}: "
                            f"SL moved to {new_sl:.5f} (profit: {profit_pips:.1f} pips)")
                    else:
                        logger.warning(
                            f"Failed to update trailing stop for "
                            f"{position.symbol}: {result.comment}")

        except Exception as e:
            logger.error(
                f"Error in basic trailing stop for {position.symbol}: {e}")

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
            bars = mt5.copy_rates_from_pos(
                position.symbol, mt5.TIMEFRAME_H1, 0, 50)
            if bars is None or len(bars) < 20:
                return

            # Calculate current ATR for dynamic TP adjustment
            atr_period = 14
            atr_values = []
            for i in range(atr_period, len(bars)):
                high = bars[i]['high']
                low = bars[i]['low']
                prev_close = bars[i-1]['close']
                tr = max(high - low, abs(high - prev_close),
                         abs(low - prev_close))
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
                    risk_pips = abs(original_sl - position.price_open) / \
                                    mt5.symbol_info(position.symbol).point
                else:  # SELL
                    risk_pips = abs(position.price_open - original_sl) / \
                                    mt5.symbol_info(position.symbol).point

                if 'JPY' in position.symbol:
                    risk_pips = risk_pips / 100

                # VALIDATION: Ensure risk calculation is positive and
                # reasonable
                if risk_pips <= 0:
                    logger.error(
                        f"🚨 RISK CALCULATION ERROR: Negative risk {risk_pips:.1f} pips for "
                        f"{position.symbol} position {position.ticket}")
                    logger.error(
                        "Skipping position management to prevent corruption")
                    return False

                if risk_pips < 5:
                    logger.warning(
                        f"⚠️ VERY TIGHT RISK: {risk_pips:.1f} pips for "
                        f"{position.symbol} position {position.ticket}")
                elif risk_pips > 500:
                    logger.warning(
                        f"⚠️ VERY WIDE RISK: {risk_pips:.1f} pips for "
                        f"{position.symbol} position {position.ticket}")

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
            # Start with 3:1 reward ratio (adjusted for minimum SL)
            base_tp_multiplier = 4.5

            # Increase multiplier if position is in strong profit (breakeven or
            # better)
            if profit_pips >= 0:
                # Scale up TP multiplier based on profit level
                profit_multiplier = min(
                    1.5, profit_pips / 50)  # Max 50% increase
                base_tp_multiplier *= (1 + profit_multiplier)

                # Calculate new take profit
                if position.type == mt5.ORDER_TYPE_BUY:
                    new_tp = position.price_open + \
                        (current_atr * base_tp_multiplier)
                else:  # SELL
                    new_tp = position.price_open - \
                        (current_atr * base_tp_multiplier)

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

                # Only update if new TP is significantly better (at least 10
                # pips improvement)
                tp_improvement = abs(new_tp - current_tp) / (100 if 'JPY' in position.symbol else 0.0001)

                # Add minimum difference check
                if tp_improvement >= 10 and abs(
                        new_tp - current_tp) > symbol_info.point * 2:
                    logger.debug(
                        f"Updating TP for {position.symbol}: current={current_tp:.5f}, new={new_tp:.5f}, "
                        f"improvement={tp_improvement:.1f} pips")
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
                        logger.info(
                            f"Take profit adjusted for {position.symbol} position {position.ticket}: "
                            f"TP moved to {new_tp:.5f} (ATR: {current_atr:.5f}, multiplier: {base_tp_multiplier:.1f})")
                        return new_tp > current_tp  # Return True if TP was increased
                    else:
                        logger.warning(
                            f"Failed to update take profit for {position.symbol}: {result.comment}")
                        return False
                else:
                    logger.debug(
                        f"Skipping TP update for {position.symbol}: "
                        f"improvement={tp_improvement:.1f} pips, "
                        f"difference={abs(new_tp - current_tp):.5f}")
                    return False

        except Exception as e:
            logger.error(f"Error updating take profit: {e}")
            return False

    async def apply_breakeven_and_trailing(
            self, position, current_price: float, risk_pips: float) -> None:
        """Apply breakeven and trailing stop logic after reaching 1:3 profit"""
        try:
            symbol_info = mt5.symbol_info(position.symbol)
            if not symbol_info:
                return

            new_sl = position.sl
            # Minimum broker distance
            min_stop_distance = max(
                symbol_info.trade_stops_level, 10) * symbol_info.point

            # Breakeven: Move SL to entry price + small buffer once at 1:1 profit
            # At least 5 pips buffer, or 20% of risk
            buffer_pips = max(risk_pips * 0.2, 5)

            if position.type == mt5.ORDER_TYPE_BUY:
                # For BUY: breakeven at entry + buffer
                breakeven_level = position.price_open + \
                    (buffer_pips * symbol_info.point)
                if 'JPY' in position.symbol:
                    breakeven_level = position.price_open + \
                        (buffer_pips * symbol_info.point * 100)

                # Ensure breakeven level meets broker minimum from current
                # price
                min_breakeven = current_price - min_stop_distance
                breakeven_level = min(breakeven_level, min_breakeven)

                if current_price >= breakeven_level and position.sl < breakeven_level:
                    new_sl = breakeven_level
                    logger.info(
                        f"Breakeven activated for {position.symbol} BUY position {position.ticket}: "
                        f"SL moved to {new_sl:.5f}")
            else:  # SELL
                # For SELL: breakeven at entry - buffer
                breakeven_level = position.price_open - \
                    (buffer_pips * symbol_info.point)
                if 'JPY' in position.symbol:
                    breakeven_level = position.price_open - \
                        (buffer_pips * symbol_info.point * 100)

                # Ensure breakeven level meets broker minimum from current
                # price
                max_breakeven = current_price + min_stop_distance
                breakeven_level = max(breakeven_level, max_breakeven)

                if current_price <= breakeven_level and position.sl > breakeven_level:
                    new_sl = breakeven_level
                    logger.info(
                        f"Breakeven activated for {position.symbol} "
                        f"SELL position {position.ticket}: "
                        f"SL moved to {new_sl:.5f}")

            # Trailing stop: Trail behind current price with adaptive distance
            # At least 10 pips, or 70% of original risk
            trail_distance_pips = max(risk_pips * 0.7, 10)

            if position.type == mt5.ORDER_TYPE_BUY:
                trail_level = current_price - \
                    (trail_distance_pips * symbol_info.point)
                if 'JPY' in position.symbol:
                    trail_level = current_price - \
                        (trail_distance_pips * symbol_info.point * 100)

                # Ensure trail level is above current SL and meets broker
                # minimum
                trail_level = max(trail_level, position.sl + min_stop_distance)

                if trail_level > new_sl:
                    new_sl = trail_level
                    logger.info(
                        f"Trailing stop activated for {position.symbol} BUY position {position.ticket}: "
                        f"SL moved to {new_sl:.5f}")
            else:  # SELL
                trail_level = current_price + \
                    (trail_distance_pips * symbol_info.point)
                if 'JPY' in position.symbol:
                    trail_level = current_price + \
                        (trail_distance_pips * symbol_info.point * 100)

                # Ensure trail level is below current SL and meets broker
                # minimum
                trail_level = min(trail_level, position.sl - min_stop_distance)

                if trail_level < new_sl:
                    new_sl = trail_level
                    logger.info(
                        f"Trailing stop activated for {position.symbol} "
                        f"SELL position {position.ticket}: "
                        f"SL moved to {new_sl:.5f}")

            # Apply the stop loss update if it changed and meets broker requirements
            # Check that new SL meets minimum distance from current price
            if position.type == mt5.ORDER_TYPE_BUY:
                min_allowed_sl = current_price - min_stop_distance
                sl_valid = new_sl <= min_allowed_sl
            else:  # SELL
                min_allowed_sl = current_price + min_stop_distance
                sl_valid = new_sl >= min_allowed_sl

            # Only update if changed and valid
            if abs(new_sl - position.sl) > symbol_info.point and sl_valid:
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
                    logger.info(
                        f"Stop loss updated for {position.symbol} position {position.ticket}: SL = {new_sl:.5f}")
                else:
                    logger.warning(
                        f"Failed to update stop loss for {position.symbol}: {result.comment}")

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
                logger.error(
                    f"Failed to select symbol {position.symbol} for closing")
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
                # Removed type_filling to avoid "Unsupported filling mode"
                # errors
            }

            # Send close order
            result = mt5.order_send(request)

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"Position closed: {position.symbol} ticket {position.ticket}")

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
