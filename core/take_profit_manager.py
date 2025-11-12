"""
FX-Ai Take Profit Manager Module
Handles take profit adjustments and optimizations
"""

import logging
import asyncio
from typing import Dict, Optional, Tuple
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class TakeProfitManager:
    """Manages take profit adjustments and optimizations"""

    def __init__(self, mt5_connector, config: dict):
        """Initialize take profit manager"""
        self.mt5 = mt5_connector
        self.config = config
        self.magic_number = config.get('trading', {}).get('magic_number', 20241029)

    async def update_take_profit(self, position) -> bool:
        """Update take profit based on new market analysis - ASYNC

        Returns:
            bool: True if take profit was increased, False otherwise
        """
        try:
            # Get current market data for analysis
            tick = mt5.symbol_info_tick(position.symbol)  # type: ignore
            if tick is None:
                return False

            current_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

            # Get recent bars for technical analysis
            bars = mt5.copy_rates_from_pos(  # type: ignore
                position.symbol, mt5.TIMEFRAME_H1, 0, 50)
            if bars is None or len(bars) < 20:
                return False

            # Calculate current ATR for dynamic TP adjustment
            atr_period = 14
            atr_values = []
            for i in range(atr_period, len(bars)):
                high = bars[i]['high']
                low = bars[i]['low']
                prev_close = bars[i - 1]['close']
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                atr_values.append(tr)

            if not atr_values:
                return False

            current_atr = sum(atr_values[-10:]) / len(atr_values[-10:])  # Recent ATR

            # Calculate current profit in pips
            symbol_info = mt5.symbol_info(position.symbol)  # type: ignore
            if symbol_info:
                point = symbol_info.point
                if position.type == mt5.ORDER_TYPE_BUY:
                    profit_pips = (current_price - position.price_open) / point
                else:  # SELL
                    profit_pips = (position.price_open - current_price) / point

                # Convert to pips (handle JPY pairs)
                if 'JPY' in position.symbol:
                    profit_pips = profit_pips / 100
            else:
                return False

            # Calculate original risk in pips (distance from entry to SL)
            original_sl = position.sl
            if original_sl > 0:
                if position.type == mt5.ORDER_TYPE_BUY:
                    risk_pips = abs(original_sl - position.price_open) / point
                else:  # SELL
                    risk_pips = abs(position.price_open - original_sl) / point

                if 'JPY' in position.symbol:
                    risk_pips = risk_pips / 100

                # VALIDATION: Ensure risk calculation is positive and reasonable
                if risk_pips <= 0:
                    logger.error(
                        f"[ERROR] RISK CALCULATION ERROR: Negative risk {risk_pips:.1f} pips for "
                        f"{position.symbol} position {position.ticket}")
                    logger.error("Skipping position management to prevent corruption")
                    return False

                if risk_pips < 5:
                    logger.warning(
                        f"[WARNING] VERY TIGHT RISK: {risk_pips:.1f} pips for "
                        f"{position.symbol} position {position.ticket}")
                elif risk_pips > 500:
                    logger.warning(
                        f"[WARNING] VERY WIDE RISK: {risk_pips:.1f} pips for "
                        f"{position.symbol} position {position.ticket}")

                # Breakeven and trailing stop logic
                # Activate only after reaching 1:3 profit ratio (3x the risk)
                profit_target_pips = risk_pips * 3

                if profit_pips >= profit_target_pips:
                    # Activate breakeven and trailing stops
                    await self._apply_breakeven_and_trailing(position, current_price, risk_pips)
                    return True  # Position management activated

            # Dynamic take profit adjustment logic
            current_tp = position.tp
            new_tp = current_tp

            # Base TP on current ATR with adaptive multiplier
            # Start with 3:1 reward ratio (adjusted for minimum SL)
            base_tp_multiplier = 4.5

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

                # Add minimum difference check
                if tp_improvement >= 10 and abs(new_tp - current_tp) > symbol_info.point * 2:
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

                    result = mt5.order_send(request)  # type: ignore
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

            # If position is at loss, don't adjust TP
            return False

        except Exception as e:
            logger.error(f"Error updating take profit: {e}")
            return False

    async def _apply_breakeven_and_trailing(self, position, current_price: float, risk_pips: float) -> None:
        """Apply breakeven and trailing stop logic after reaching 1:3 profit"""
        try:
            symbol_info = mt5.symbol_info(position.symbol)  # type: ignore
            if not symbol_info:
                return

            new_sl = position.sl
            # Minimum broker distance
            min_stop_distance = max(symbol_info.trade_stops_level, 10) * symbol_info.point

            # Breakeven: Move SL to entry price + small buffer once at 1:1 profit
            # At least 5 pips buffer, or 20% of risk
            buffer_pips = max(risk_pips * 0.2, 5)

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
                    logger.info(
                        f"Breakeven activated for {position.symbol} BUY position {position.ticket}: "
                        f"SL moved to {new_sl:.5f}")
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
                    logger.info(
                        f"Breakeven activated for {position.symbol} SELL position {position.ticket}: "
                        f"SL moved to {new_sl:.5f}")

            # Trailing stop: Trail behind current price with adaptive distance
            # Use 15% of current profit as trail distance (more aggressive)
            if position.type == mt5.ORDER_TYPE_BUY:
                profit_distance = current_price - position.price_open
                trail_distance = profit_distance * 0.15  # 15% trail
                new_trail_sl = current_price - trail_distance

                # Ensure trail SL is better than current SL and breakeven
                if new_trail_sl > max(position.sl, breakeven_level):
                    new_sl = max(new_sl, new_trail_sl)
                    logger.info(
                        f"Trailing stop activated for {position.symbol} BUY position {position.ticket}: "
                        f"SL moved to {new_trail_sl:.5f}")
            else:  # SELL
                profit_distance = position.price_open - current_price
                trail_distance = profit_distance * 0.15  # 15% trail
                new_trail_sl = current_price + trail_distance

                # Ensure trail SL is better than current SL and breakeven
                if new_trail_sl < min(position.sl, breakeven_level):
                    new_sl = min(new_sl, new_trail_sl)
                    logger.info(
                        f"Trailing stop activated for {position.symbol} SELL position {position.ticket}: "
                        f"SL moved to {new_trail_sl:.5f}")

            # Apply the combined SL adjustment
            if new_sl != position.sl:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": position.symbol,
                    "position": position.ticket,
                    "sl": new_sl,
                    "tp": position.tp,  # Keep existing TP
                    "magic": self.magic_number
                }

                result = mt5.order_send(request)  # type: ignore
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(
                        f"Combined breakeven/trailing applied for {position.symbol} position {position.ticket}: "
                        f"SL moved to {new_sl:.5f}")
                else:
                    logger.warning(
                        f"Failed to apply combined breakeven/trailing for {position.symbol}: {result.comment}")

        except Exception as e:
            logger.error(f"Error applying breakeven and trailing: {e}")

    async def extend_take_profit(self, position) -> None:
        """Extend take profit based on market conditions and position strength"""
        try:
            # Get current market data
            tick = mt5.symbol_info_tick(position.symbol)  # type: ignore
            if not tick:
                return

            current_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

            # Get ATR for dynamic extension
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

            # Calculate current profit
            symbol_info = mt5.symbol_info(position.symbol)  # type: ignore
            if not symbol_info:
                return

            if position.type == mt5.ORDER_TYPE_BUY:
                profit_pips = (current_price - position.price_open) / symbol_info.point
            else:  # SELL
                profit_pips = (position.price_open - current_price) / symbol_info.point

            if 'JPY' in position.symbol:
                profit_pips /= 100

            # Extend TP if position is in strong profit (2:1 ratio or better)
            if profit_pips >= 40:  # At least 40 pips profit
                # Calculate extended TP using higher ATR multiplier
                extension_multiplier = 6.0  # More aggressive than standard 4.5

                if position.type == mt5.ORDER_TYPE_BUY:
                    extended_tp = position.price_open + (current_atr * extension_multiplier)
                else:  # SELL
                    extended_tp = position.price_open - (current_atr * extension_multiplier)

                # Only extend if it's significantly better than current TP
                current_tp = position.tp
                if position.type == mt5.ORDER_TYPE_BUY and extended_tp > current_tp:
                    improvement = (extended_tp - current_tp) / symbol_info.point
                    if 'JPY' in position.symbol:
                        improvement /= 100

                    if improvement >= 20:  # At least 20 pips improvement
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "symbol": position.symbol,
                            "position": position.ticket,
                            "sl": position.sl,
                            "tp": extended_tp,
                            "magic": self.magic_number
                        }

                        result = mt5.order_send(request)  # type: ignore
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(
                                f"Take profit extended for {position.symbol} position {position.ticket} "
                                f"to {extended_tp:.5f} (+{improvement:.1f} pips)")
                        else:
                            logger.warning(f"Failed to extend take profit: {result.comment}")

                elif position.type == mt5.ORDER_TYPE_SELL and extended_tp < current_tp:
                    improvement = (current_tp - extended_tp) / symbol_info.point
                    if 'JPY' in position.symbol:
                        improvement /= 100

                    if improvement >= 20:  # At least 20 pips improvement
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "symbol": position.symbol,
                            "position": position.ticket,
                            "sl": position.sl,
                            "tp": extended_tp,
                            "magic": self.magic_number
                        }

                        result = mt5.order_send(request)  # type: ignore
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(
                                f"Take profit extended for {position.symbol} position {position.ticket} "
                                f"to {extended_tp:.5f} (+{improvement:.1f} pips)")
                        else:
                            logger.warning(f"Failed to extend take profit: {result.comment}")

        except Exception as e:
            logger.error(f"Error extending take profit: {e}")