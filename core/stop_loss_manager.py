"""
FX-Ai Stop Loss Manager Module
Handles stop loss management including trailing stops and breakeven
"""

import logging
import asyncio
from typing import Dict, Optional
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class StopLossManager:
    """Manages stop loss adjustments including trailing stops and breakeven"""

    def __init__(self, mt5_connector, config: dict):
        """Initialize stop loss manager"""
        self.mt5 = mt5_connector
        self.config = config
        self.magic_number = config.get('trading', {}).get('magic_number', 20241029)

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
                # Minimum 50 pips activation for metals
                base_activation_pips = max(base_activation_pips, 50)
                # Minimum 200 pips for metals
                base_trail_distance_pips = max(base_trail_distance_pips, 200)

            # Get current price and market data
            tick = mt5.symbol_info_tick(position.symbol)  # type: ignore
            if tick is None:
                return

            current_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

            # Get recent bars for volatility analysis
            bars = mt5.copy_rates_from_pos(  # type: ignore
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
                prev_close = bars[i - 1]['close']
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
            symbol_info = mt5.symbol_info(position.symbol)  # type: ignore
            if symbol_info:
                point = symbol_info.point
                digits = symbol_info.digits
                if 'XAG' in position.symbol:
                    # Silver: 1 pip = 1 point (0.001 for 3-digit symbols)
                    pip_size = point
                elif 'XAU' in position.symbol or 'GOLD' in position.symbol:
                    # Gold: 1 pip = 10 points (0.1 for 2-digit symbols)
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

                    result = mt5.order_send(request)  # type: ignore
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(
                            f"Adaptive trailing stop updated for {position.symbol} position "
                            f"{position.ticket}: SL moved to {new_sl:.5f} "
                            f"(profit: {profit_pips:.1f} pips, volatility: {volatility_ratio:.2f})")
                    else:
                        logger.warning(
                            f"Failed to update trailing stop for {position.symbol}: {result.comment}")

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
            tick = mt5.symbol_info_tick(position.symbol)  # type: ignore
            if tick is None:
                return

            current_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

            # Get symbol info for pip calculation
            symbol_info = mt5.symbol_info(position.symbol)  # type: ignore
            if not symbol_info:
                return

            point = symbol_info.point
            digits = symbol_info.digits

            # Determine pip size
            if "XAG" in position.symbol:
                # Silver: 1 pip = 1 point (0.001 for 3-digit symbols)
                pip_size = point
            elif "XAU" in position.symbol or "GOLD" in position.symbol:
                # Gold: 1 pip = 10 points (0.1 for 2-digit symbols)
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

                    result = mt5.order_send(request)  # type: ignore
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(
                            f"Breakeven applied for {position.symbol} position "
                            f"{position.ticket}: SL moved to {breakeven_sl:.5f} (entry price)")
                    else:
                        logger.warning(
                            f"Failed to apply breakeven for {position.symbol}: {result.comment}")

        except Exception as e:
            logger.error(f"Error applying breakeven: {e}")

    async def _basic_trailing_stop(self, position, current_price: float,
                                  activation_pips: float, trail_distance_pips: float) -> None:
        """Basic trailing stop fallback when volatility data unavailable"""
        try:
            symbol_info = mt5.symbol_info(position.symbol)  # type: ignore
            if not symbol_info:
                return

            point = symbol_info.point
            digits = symbol_info.digits

            # Determine pip size
            if "XAG" in position.symbol:
                # Silver: 1 pip = 1 point (0.001 for 3-digit symbols)
                pip_size = point
            elif "XAU" in position.symbol or "GOLD" in position.symbol:
                # Gold: 1 pip = 10 points (0.1 for 2-digit symbols)
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

                    result = mt5.order_send(request)  # type: ignore
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(
                            f"Basic trailing stop updated for {position.symbol} position "
                            f"{position.ticket}: SL moved to {new_sl:.5f} (profit: {profit_pips:.1f} pips)")
                    else:
                        logger.warning(
                            f"Failed to update trailing stop for {position.symbol}: {result.comment}")

        except Exception as e:
            logger.error(f"Error in basic trailing stop for {position.symbol}: {e}")

    async def move_sl_to_breakeven(self, position) -> None:
        """Move stop loss to breakeven level"""
        try:
            # Calculate breakeven SL (entry price)
            breakeven_sl = position.price_open

            # Get symbol info to check broker requirements
            symbol_info = mt5.symbol_info(position.symbol)  # type: ignore
            if symbol_info:
                min_stop_distance = symbol_info.trade_stops_level * symbol_info.point

                # Ensure breakeven SL meets broker minimum distance from current price
                tick = mt5.symbol_info_tick(position.symbol)  # type: ignore
                if tick:
                    current_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

                    if position.type == mt5.ORDER_TYPE_BUY:
                        min_allowed_sl = current_price - min_stop_distance
                        if breakeven_sl >= min_allowed_sl:
                            logger.debug(f"Breakeven SL {breakeven_sl} too close to current price, skipping")
                            return
                    else:  # SELL
                        max_allowed_sl = current_price + min_stop_distance
                        if breakeven_sl <= max_allowed_sl:
                            logger.debug(f"Breakeven SL {breakeven_sl} too close to current price, skipping")
                            return

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": position.ticket,
                "sl": breakeven_sl,
                "tp": position.tp,
                "magic": self.magic_number
            }

            result = mt5.order_send(request)  # type: ignore
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"SL moved to breakeven for {position.symbol} position {position.ticket}")
            else:
                logger.warning(f"Failed to move SL to breakeven: {result.comment}")

        except Exception as e:
            logger.error(f"Error moving SL to breakeven: {e}")

    async def tighten_stops_aggressively(self, position) -> None:
        """Aggressively tighten stop loss based on current market conditions"""
        try:
            # Get current price
            tick = mt5.symbol_info_tick(position.symbol)  # type: ignore
            if not tick:
                return

            current_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

            # Get ATR for dynamic adjustment
            bars = mt5.copy_rates_from_pos(position.symbol, mt5.TIMEFRAME_H1, 0, 20)  # type: ignore
            if bars is None or len(bars) < 14:
                return

            # Calculate ATR
            atr_values = []
            for i in range(1, len(bars)):
                high = bars[i]['high']
                low = bars[i]['low']
                prev_close = bars[i - 1]['close']
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                atr_values.append(tr)

            if not atr_values:
                return

            current_atr = sum(atr_values) / len(atr_values)

            # Calculate tighter SL using 1.5 ATR multiplier (more aggressive than default 2.0)
            if position.type == mt5.ORDER_TYPE_BUY:
                new_sl = position.price_open - (current_atr * 1.5)
            else:  # SELL
                new_sl = position.price_open + (current_atr * 1.5)

            # Ensure it meets broker minimums
            symbol_info = mt5.symbol_info(position.symbol)  # type: ignore
            if symbol_info:
                min_stop_distance = symbol_info.trade_stops_level * symbol_info.point

                if position.type == mt5.ORDER_TYPE_BUY:
                    min_allowed_sl = current_price - min_stop_distance
                    new_sl = min(new_sl, min_allowed_sl)  # Don't move SL too close to current price
                else:  # SELL
                    max_allowed_sl = current_price + min_stop_distance
                    new_sl = max(new_sl, max_allowed_sl)  # Don't move SL too close to current price

            # Only tighten if it's an improvement
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
                    "tp": position.tp,
                    "magic": self.magic_number
                }

                result = mt5.order_send(request)  # type: ignore
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Aggressively tightened SL for {position.symbol} position {position.ticket} to {new_sl:.5f}")
                else:
                    logger.warning(f"Failed to tighten SL: {result.comment}")

        except Exception as e:
            logger.error(f"Error tightening stops aggressively: {e}")