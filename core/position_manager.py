"""
FX-Ai Position Manager Module
Handles position monitoring, management, and adaptive adjustments
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Optional
import MetaTrader5 as mt5
from utils.position_monitor import PositionMonitor
from utils.risk_validator import RiskValidator

logger = logging.getLogger(__name__)


class PositionManager:
    """Manages open positions and applies risk management strategies"""

    def __init__(self, mt5_connector, risk_manager, config: dict, adaptive_learning_manager=None, stop_loss_manager=None, take_profit_manager=None, position_closer=None, fundamental_analyzer=None):
        """Initialize position manager"""
        self.mt5 = mt5_connector
        self.risk_manager = risk_manager
        self.config = config
        self.adaptive_learning_manager = adaptive_learning_manager
        self.stop_loss_manager = stop_loss_manager
        self.take_profit_manager = take_profit_manager
        self.position_closer = position_closer
        self.fundamental = fundamental_analyzer
        self.magic_number = config.get('trading', {}).get('magic_number', 20241029)

        # Initialize position monitor for change detection
        self.position_monitor = PositionMonitor(self.magic_number)

    async def manage_positions(self, symbol: str, time_manager=None, adaptive_learning=None):
        """Manage open positions for a symbol - ASYNC"""
        try:
            positions = mt5.positions_get(symbol=symbol)  # type: ignore

            if positions:
                for position in positions:
                    if position.magic == self.magic_number:
                        # FIRST: Check position monitor for unexpected changes
                        alerts = await self.position_monitor.check_positions()
                        if alerts:
                            for alert in alerts:
                                if "VERY TIGHT SL" in alert and str(position.ticket) in alert:
                                    logger.error(f"[ALERT] PREVENTING AUTOMATED MANAGEMENT: {alert}")
                                    logger.error("Position has suspiciously tight SL - skipping automated management")
                                    continue

                        # SECOND: Validate position integrity before management
                        validation = RiskValidator.comprehensive_position_check(position)
                        if not validation['overall_valid']:
                            logger.error(
                                f"[ERROR] POSITION VALIDATION FAILED for "
                                f"{position.symbol} position {position.ticket}:")
                            for issue in validation['issues']:
                                logger.error(f"  - {issue}")
                            logger.error("Skipping automated management due to validation failures")
                            continue

                        # THIRD: Check if position was manually modified (skip automated management)
                        time_since_update = position.time_update - position.time
                        manual_modification_threshold = 600  # 10 minutes in seconds

                        if time_since_update > manual_modification_threshold:
                            logger.info(
                                f"Skipping automated management for {position.symbol} position "
                                f"{position.ticket} (appears manually modified "
                                f"{time_since_update / 3600:.1f} hours ago)")
                            continue

                        # Check for fundamental updates during trade (HIGH PRIORITY)
                        await self._check_fundamental_updates_during_trade(position)

                        # Update take profit based on new analysis first
                        tp_increased = await self._update_take_profit(position)

                        # Check for adaptive learning updates to SL/TP
                        await self._check_adaptive_sl_tp_adjustment(position, adaptive_learning)

                        # Only apply breakeven and trailing stops if TP was increased
                        if tp_increased:
                            # Apply breakeven if enabled
                            await self._apply_breakeven(position)

                            # Update trailing stops if needed
                            await self._update_trailing_stop(position)

            # Small delay to prevent CPU overload
            await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error managing positions: {e}")

    async def _check_adaptive_sl_tp_adjustment(self, position, adaptive_learning=None):
        """Check if position SL/TP should be adjusted based on adaptive learning"""
        try:
            if not adaptive_learning:
                return

            # Convert position time to datetime
            trade_timestamp = datetime.fromtimestamp(position.time)

            # Check if there are updated SL/TP parameters for this symbol
            adjustment_needed = adaptive_learning.should_adjust_existing_trade(
                position.symbol, position.sl, position.tp, trade_timestamp)

            if not adjustment_needed.get('should_adjust', False):
                return

            logger.info(
                f"Adaptive learning suggests SL/TP adjustment for "
                f"{position.symbol} position {position.ticket}")

            # Get current market data to calculate new SL/TP levels
            tick = mt5.symbol_info_tick(position.symbol)  # type: ignore
            if tick is None:
                return

            current_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

            # Get ATR for calculating new levels
            bars = mt5.copy_rates_from_pos(  # type: ignore
                position.symbol, mt5.TIMEFRAME_H1, 0, 20)
            if bars is None or len(bars) < 14:
                return

            # Calculate current ATR
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
            symbol_info = mt5.symbol_info(position.symbol)  # type: ignore
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
            min_change_distance = min_change_pips * (100 if 'JPY' in position.symbol else 0.0001)

            if sl_change < min_change_distance and tp_change < min_change_distance:
                logger.debug(
                    f"SL/TP changes too small for {position.symbol}: "
                    f"SL delta{sl_change:.5f}, TP delta{tp_change:.5f}")
                return

            # Record the adjustment before making it
            if self.adaptive_learning_manager:
                self.adaptive_learning_manager.record_position_adjustment(
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

            result = mt5.order_send(request)  # type: ignore
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"Adaptive SL/TP adjustment applied for {position.symbol} position "
                    f"{position.ticket}: SL: {position.sl:.5f} -> {new_sl:.5f}, "
                    f"TP: {position.tp:.5f} -> {new_tp:.5f} "
                    f"(ATR: {current_atr:.5f}, SL mult: {sl_multiplier:.2f}, "
                    f"TP mult: {tp_multiplier:.2f})")
            else:
                logger.warning(
                    f"Failed to apply adaptive SL/TP adjustment for {position.symbol}: {result.comment}")

        except Exception as e:
            logger.error(f"Error in adaptive SL/TP adjustment for {position.symbol}: {e}")

    async def _check_fundamental_updates_during_trade(self, position) -> None:
        """Check for fundamental updates during active trade and take action"""
        try:
            if not self.fundamental:
                return

            # Get breaking news for the last 5 minutes
            breaking_news = self.fundamental.get_breaking_news(
                symbol=position.symbol,
                minutes=5
            )

            if not breaking_news.get('has_breaking_news', False):
                return

            severity = breaking_news.get('severity', 'low')
            direction = breaking_news.get('direction', 'neutral')
            recommendation = breaking_news.get('recommendation', 'hold')

            logger.warning(
                f"FUNDAMENTAL ALERT for {position.symbol} position {position.ticket}:"
            )
            logger.warning(f"   Severity: {severity.upper()}")
            logger.warning(f"   Direction: {direction}")
            logger.warning(f"   Recommendation: {recommendation}")

            # Execute recommendation
            if recommendation == 'close_position':
                await self.position_closer.close_position(position, reason="Adverse fundamental news")
            elif recommendation == 'lock_profits':
                await self.stop_loss_manager.move_sl_to_breakeven(position)
            elif recommendation == 'tighten_stops':
                await self.stop_loss_manager.tighten_stops_aggressively(position)
            elif recommendation == 'extend_targets':
                await self.take_profit_manager.extend_take_profit(position)

        except Exception as e:
            logger.error(f"Error checking fundamental updates for {position.symbol}: {e}")

    async def _update_take_profit(self, position) -> bool:
        """Update take profit based on new analysis"""
        # This method will be implemented when we extract the take profit logic
        return False

    async def _apply_breakeven(self, position) -> None:
        """Apply breakeven if enabled"""
        # This method will be implemented when we extract the breakeven logic
        pass

    async def _update_trailing_stop(self, position) -> None:
        """Update trailing stops if needed"""
        # This method will be implemented when we extract the trailing stop logic
        pass