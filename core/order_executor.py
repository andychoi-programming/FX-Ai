import logging
import asyncio
from datetime import datetime
from typing import Dict, Optional
import MetaTrader5 as mt5
from ai.learning_database import LearningDatabase

logger = logging.getLogger(__name__)


class OrderManager:
    """Hybrid order manager supporting multiple entry strategies"""

    def __init__(self, mt5, config, order_executor=None):
        self.mt5 = mt5
        self.config = config
        self.order_executor = order_executor  # Reference to parent OrderExecutor
        self.magic_number = self.config.get('trading', {}).get('magic_number')

        # Load order management settings
        order_mgmt_config = self.config.get('trading', {}).get('order_management', {})
        self.default_entry_strategy = order_mgmt_config.get('default_entry_strategy', 'stop')
        self.order_expiration_hours = order_mgmt_config.get('order_expiration_hours', 24)

        pending_config = order_mgmt_config.get('pending_order_management', {})
        self.pending_mgmt_enabled = pending_config.get('enabled', True)
        self.stale_threshold_hours = pending_config.get('stale_order_threshold_hours', 1)
        self.price_movement_threshold = pending_config.get('price_movement_cancel_threshold', 0.02)
        self.max_pending_orders = pending_config.get('max_pending_orders', 10)

        validation_config = order_mgmt_config.get('validation', {})
        self.pre_place_validation = validation_config.get('pre_place_validation', True)

    async def place_order(self, symbol: str, signal: str, entry_strategy: str = None,
                         volume: Optional[float] = None, stop_loss: Optional[float] = None,
                         take_profit: Optional[float] = None, signal_data: Optional[Dict] = None) -> Dict:
        """
        Place order using specified strategy

        entry_strategy options:
        - "stop": Use stop orders (default)
        - "market": Use market orders (fallback)
        - "limit": Use limit orders (advanced)

        Args:
            symbol: Trading symbol
            signal: "BUY" or "SELL"
            entry_strategy: Entry strategy to use (optional, uses config default)
            volume: Order volume (optional)
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            signal_data: Additional signal data

        Returns:
            dict: Order result
        """

        if entry_strategy is None:
            entry_strategy = self.default_entry_strategy

        if entry_strategy == "stop":
            return await self._place_stop_order(symbol, signal, volume, stop_loss, take_profit, signal_data)
        elif entry_strategy == "market":
            return await self._place_market_order(symbol, signal, volume, stop_loss, take_profit, signal_data)
        elif entry_strategy == "limit":
            return await self._place_limit_order(symbol, signal, volume, stop_loss, take_profit, signal_data)
        else:
            return {
                'success': False,
                'error': f'Unknown entry strategy: {entry_strategy}'
            }

    async def _place_stop_order(self, symbol: str, signal: str, volume: Optional[float] = None,
                               stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                               signal_data: Optional[Dict] = None) -> Dict:
        """Place stop order with ATR-based distance calculation"""

        try:
            # Get current price and ATR
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {'success': False, 'error': f'Failed to get tick data for {symbol}'}

            current_price = tick.ask if signal == "BUY" else tick.bid
            atr = self._get_atr(symbol)

            # Calculate optimal stop distance
            stop_distance = self.order_executor._calculate_stop_distance(symbol, signal, atr, signal_data)

            # Calculate stop order price
            if signal == "BUY":
                stop_price = current_price + stop_distance
            else:  # SELL
                stop_price = current_price - stop_distance

            # Validate stop order before placing
            is_valid, validation_error = self.order_executor._validate_stop_order(symbol, signal, stop_price, current_price)
            if not is_valid:
                return {'success': False, 'error': validation_error}

            # Calculate volume if not provided
            if volume is None:
                volume = self.order_executor._calculate_position_size(symbol, stop_price, stop_loss or (stop_price - stop_distance))

            # Calculate SL/TP if not provided
            if stop_loss is None or take_profit is None:
                sl_price, tp_price = self.order_executor._calculate_sl_tp(symbol, signal, stop_price, atr)
                if stop_loss is None:
                    stop_loss = sl_price
                if take_profit is None:
                    take_profit = tp_price

            # Set order expiration
            expiration = self.order_executor._calculate_order_expiration()

            # Place the stop order
            return await self.order_executor.place_order(
                symbol=symbol,
                order_type=f"{signal.lower()}_stop",
                volume=volume,
                stop_loss=stop_loss,
                take_profit=take_profit,
                price=stop_price,
                comment=f"FX-Ai Stop Order ({signal})",
                signal_data=signal_data
            )

        except Exception as e:
            logger.error(f"Error placing stop order for {symbol}: {e}")
            return {'success': False, 'error': str(e)}

    async def _place_market_order(self, symbol: str, signal: str, volume: Optional[float] = None,
                                 stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                                 signal_data: Optional[Dict] = None) -> Dict:
        """Place market order as fallback"""

        try:
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {'success': False, 'error': f'Failed to get tick data for {symbol}'}

            current_price = tick.ask if signal == "BUY" else tick.bid
            atr = self._get_atr(symbol)

            # Calculate volume if not provided
            if volume is None:
                # For market orders, estimate SL distance for position sizing
                estimated_sl_distance = atr * 2  # Conservative estimate
                volume = self.order_executor._calculate_position_size(symbol, current_price, current_price - estimated_sl_distance)

            # Calculate SL/TP if not provided
            if stop_loss is None or take_profit is None:
                sl_price, tp_price = self.order_executor._calculate_sl_tp(symbol, signal, current_price, atr)
                if stop_loss is None:
                    stop_loss = sl_price
                if take_profit is None:
                    take_profit = tp_price

            # Place market order
            return await self.order_executor.place_order(
                symbol=symbol,
                order_type=signal.lower(),
                volume=volume,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=f"FX-Ai Market Order ({signal})",
                signal_data=signal_data
            )

        except Exception as e:
            logger.error(f"Error placing market order for {symbol}: {e}")
            return {'success': False, 'error': str(e)}

    async def _place_limit_order(self, symbol: str, signal: str, volume: Optional[float] = None,
                                stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                                signal_data: Optional[Dict] = None) -> Dict:
        """Place limit order (advanced strategy)"""

        # For now, fall back to market order - limit orders require more sophisticated logic
        logger.info(f"Limit orders not fully implemented, using market order for {symbol}")
        return await self._place_market_order(symbol, signal, volume, stop_loss, take_profit, signal_data)

    def _get_atr(self, symbol: str, period: int = 14) -> float:
        """Get ATR value for symbol - try technical analyzer first, fallback to manual calculation"""
        
        # Try to get ATR from technical analyzer if available
        if hasattr(self.order_executor, 'technical_analyzer') and self.order_executor.technical_analyzer:
            try:
                atr_value = self.order_executor.technical_analyzer.get_atr(symbol, period)
                if atr_value and atr_value > 0:
                    return atr_value
            except Exception as e:
                logger.debug(f"Technical analyzer ATR failed for {symbol}: {e}")
        
        # Fallback to manual calculation
        try:
            # Get recent price data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, period + 1)
            if rates is None or len(rates) < period + 1:
                # Fallback to config-based ATR estimate
                return self.order_executor._get_fallback_atr(symbol)

            # Calculate ATR manually - rates is a numpy structured array
            highs = [rate['high'] for rate in rates]
            lows = [rate['low'] for rate in rates]
            closes = [rate['close'] for rate in rates]

            tr_values = []
            for i in range(1, len(closes)):
                tr = max(
                    highs[i] - lows[i],  # Current high - current low
                    abs(highs[i] - closes[i-1]),  # Current high - previous close
                    abs(lows[i] - closes[i-1])   # Current low - previous close
                )
                tr_values.append(tr)

            # Simple ATR calculation (average of TR values)
            if tr_values:
                atr = sum(tr_values) / len(tr_values)
                # Ensure ATR is reasonable (not too small)
                if atr <= 0:
                    return self._get_fallback_atr(symbol)
                return atr
            else:
                return self._get_fallback_atr(symbol)

        except Exception as e:
            logger.warning(f"Error calculating ATR for {symbol}: {e}")
            return self._get_fallback_atr(symbol)

    def _get_fallback_atr(self, symbol: str) -> float:
        """Get fallback ATR estimate based on symbol type"""

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            atr_fallbacks = self.config.get('atr_fallbacks', {})
            return atr_fallbacks.get('conservative_fallback_pips', 0.001)

        # Estimate ATR based on symbol type and current price
        current_price = symbol_info.ask
        atr_fallbacks = self.config.get('atr_fallbacks', {})

        if 'XAU' in symbol or 'GOLD' in symbol:
            return current_price * atr_fallbacks.get('gold_atr_percentage', 0.01)
        elif 'XAG' in symbol or 'SILVER' in symbol:
            return current_price * atr_fallbacks.get('silver_atr_percentage', 0.015)
        else:
            return current_price * atr_fallbacks.get('forex_atr_percentage', 0.002)

    def _calculate_stop_distance(self, symbol: str, signal: str, atr: float,
                               signal_data: Optional[Dict] = None) -> float:
        """Calculate optimal stop distance using professional ATR multipliers"""

        # Get broker minimum distance first
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return 0.001  # Conservative fallback

        min_distance = self.order_executor._calculate_min_stop_distance(symbol, symbol_info)
        broker_min_distance = min_distance * 1.5  # 1.5x broker minimum for safety

        # Get ATR multipliers from config
        stop_loss_config = self.config.get('trading_rules', {}).get('stop_loss_rules', {})
        atr_multipliers = {
            'major': stop_loss_config.get('sl_atr_multiplier_major', 2.0),
            'minor': stop_loss_config.get('sl_atr_multiplier_minor', 2.0),
            'cross': stop_loss_config.get('sl_atr_multiplier_cross', 2.5),
            'jpy': stop_loss_config.get('sl_atr_multiplier_jpy', 2.5),
            'gold': stop_loss_config.get('sl_atr_multiplier_gold', 1.5),
            'silver': stop_loss_config.get('sl_atr_multiplier_silver', 2.0),
        }

        # Classify symbol for ATR multiplier
        pair_type = self._get_pair_type(symbol)
        atr_multiplier = atr_multipliers.get(pair_type, 2.5)  # Default to cross pair multiplier

        # Calculate ATR-based stop distance
        if atr is None or atr <= 0:
            # If ATR fails, use professional minimum as base
            atr_distance = min_distance
        else:
            atr_distance = atr * atr_multiplier

        # Apply risk adjustments from signal data
        risk_multiplier = 1.0
        if signal_data:
            signal_strength = signal_data.get('signal_strength', 0.5)

            # Increase distance for weaker signals
            if signal_strength < 0.4:
                risk_multiplier *= 1.3  # Less aggressive than before
            elif signal_strength < 0.6:
                risk_multiplier *= 1.1

        atr_distance *= risk_multiplier

        # Use the MAXIMUM of:
        # 1. ATR-based distance (adaptive to volatility)
        # 2. Professional minimum (allows normal market movement)
        # 3. Broker requirement * 1.5 (safety margin)
        final_distance = max(atr_distance, min_distance, broker_min_distance)

        logger.debug(
            f"{symbol} ({pair_type}): Stop calc - ATR: {atr:.5f} [X] {atr_multiplier} = {atr_distance:.5f}, "
            f"Min: {min_distance:.5f}, Broker: {broker_min_distance:.5f}, Final: {final_distance:.5f}"
        )

        return final_distance

    def _get_pair_type(self, symbol: str) -> str:
        """Classify symbol for ATR multiplier selection"""

        majors = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
        minors = ['AUDUSD', 'NZDUSD', 'USDCAD']

        if symbol in majors:
            return 'major'
        elif symbol in minors:
            return 'minor'
        elif 'JPY' in symbol:
            return 'jpy'
        elif symbol == 'XAUUSD':
            return 'gold'
        elif symbol == 'XAGUSD':
            return 'silver'
        else:
            return 'cross'  # All other crosses

    def _get_volatility(self, symbol: str) -> float:
        """Get volatility estimate for symbol"""

        try:
            # Get recent price data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 20)
            if rates is None or len(rates) < 20:
                return 0.01  # Default volatility

            # Calculate price range volatility - rates is numpy structured array
            closes = [rate['close'] for rate in rates]
            if len(closes) < 2:
                return 0.01

            # Calculate standard deviation of returns
            returns = []
            for i in range(1, len(closes)):
                ret = (closes[i] - closes[i-1]) / closes[i-1]
                returns.append(ret)

            if not returns:
                return 0.01

            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            volatility = variance ** 0.5

            return volatility

        except Exception as e:
            logger.warning(f"Error calculating volatility for {symbol}: {e}")
            return 0.01

    def _validate_stop_order(self, symbol: str, signal: str, stop_price: float, current_price: float) -> tuple[bool, str]:
        """Validate stop order before placing"""

        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False, f"Symbol {symbol} not found"

            min_distance = self.order_executor._calculate_min_stop_distance(symbol, symbol_info)

            if signal == "BUY":
                # Buy stop must be above current price
                if stop_price <= current_price:
                    return False, f"Buy stop ({stop_price:.5f}) must be above current price ({current_price:.5f})"

                # Must be minimum distance away
                if stop_price - current_price < min_distance:
                    return False, f"Buy stop too close (min: {min_distance:.5f} price units)"

            elif signal == "SELL":
                # Sell stop must be below current price
                if stop_price >= current_price:
                    return False, f"Sell stop ({stop_price:.5f}) must be below current price ({current_price:.5f})"

                if current_price - stop_price < min_distance:
                    return False, f"Sell stop too close (min: {min_distance:.5f} price units)"

            return True, "Valid"

        except Exception as e:
            logger.error(f"Error validating stop order for {symbol}: {e}")
            return False, f"Validation error: {str(e)}"

    def _calculate_position_size(self, symbol: str, entry_price: float, stop_loss_price: float) -> float:
        """Calculate position size based on risk management"""

        try:
            # Calculate stop loss distance in pips
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return 0.01

            stop_distance = abs(entry_price - stop_loss_price)
            pip_size = symbol_info.point * 10  # Assuming 5-digit broker
            stop_pips = stop_distance / pip_size

            # Use risk manager for position sizing if available
            if hasattr(self, 'risk_manager') and self.risk_manager:
                return self.risk_manager.calculate_position_size(symbol, stop_pips)

            # Fallback calculation
            risk_amount = self.config.get('trading', {}).get('risk_per_trade', 50.0)

            # Simple pip value calculation
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return 0.01

            if symbol.endswith("USD"):
                pip_value_per_lot = 10.0  # $10 per pip for 1 lot
            elif "JPY" in symbol:
                pip_value_per_lot = (0.01 * 100000) / tick.bid
            else:
                pip_value_per_lot = (0.0001 * 100000) / tick.bid

            if pip_value_per_lot > 0 and stop_pips > 0:
                position_size = risk_amount / (pip_value_per_lot * stop_pips)
                # Apply lot size limits
                min_lot = self.config.get('trading', {}).get('min_lot_size', 0.01)
                max_lot = self.config.get('trading', {}).get('max_lot_size', 1.0)
                position_size = max(min_lot, min(max_lot, position_size))
                return round(position_size, 2)

            return 0.01

        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.01

    def _calculate_sl_tp(self, symbol: str, signal: str, entry_price: float, atr: float) -> tuple[float, float]:
        """Calculate stop loss and take profit prices using dynamic RR ratios"""

        # Calculate stop loss distance using ATR-based method
        stop_distance = self._calculate_stop_distance(symbol, signal, atr)

        # Get RR ratio from config for this symbol
        rr_ratios = self.config.get('trading_rules', {}).get('take_profit_rules', {}).get('rr_ratios', {})
        rr_ratio = rr_ratios.get(symbol, 3.0)  # Default to 3.0 if not found

        # Calculate take profit distance: stop_distance * rr_ratio
        tp_distance = stop_distance * rr_ratio

        if signal == "BUY":
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + tp_distance
        else:  # SELL
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - tp_distance

        logger.debug(
            f"{symbol} {signal}: SL distance: {stop_distance:.5f}, RR ratio: {rr_ratio}, "
            f"TP distance: {tp_distance:.5f}, SL: {stop_loss:.5f}, TP: {take_profit:.5f}"
        )

        return stop_loss, take_profit

    def _calculate_order_expiration(self) -> int:
        """Calculate order expiration timestamp"""

        # Expire after configured hours
        expiration = datetime.now().replace(hour=23, minute=59, second=0, microsecond=0)
        # For orders placed during trading hours, expire at end of day
        # For orders placed outside hours, expire after configured hours
        return int(expiration.timestamp())

    def manage_pending_orders(self) -> Dict:
        """Monitor and manage pending stop orders - LESS AGGRESSIVE"""

        if not self.pending_mgmt_enabled:
            return {'managed': 0, 'cancelled': 0, 'errors': 0}

        try:
            orders = mt5.orders_get()
            if orders is None:
                return {'managed': 0, 'cancelled': 0, 'errors': 0}

            # Filter to only our system's orders
            our_orders = [order for order in orders if hasattr(order, 'magic') and order.magic == self.magic_number]
            total_pending = len(our_orders)
            managed = 0
            cancelled = 0
            errors = 0

            # Only cancel if we have WAY too many of OUR pending orders (emergency cleanup)
            if total_pending > self.max_pending_orders * 2:  # Double the limit
                logger.warning(f"Emergency: Too many of our pending orders: {total_pending} (max: {self.max_pending_orders})")
                # Cancel oldest of our orders to reduce count
                orders_to_cancel = sorted(our_orders, key=lambda x: getattr(x, 'time_setup', 0))[:total_pending - self.max_pending_orders]
                for order in orders_to_cancel:
                    if self._cancel_order(order.ticket):
                        cancelled += 1
                        logger.info(f"Emergency cancelled excess pending order: {order.ticket}")

            # Check orders that are older than 1 hour (stale - trigger re-analysis)
            current_time = datetime.now().timestamp()
            stale_threshold = 1 * 3600  # 1 hour
            # Check orders that are older than 2 hours (very stale)
            very_stale_threshold = 2 * 3600  # 2 hours

            for order in our_orders:
                try:
                    managed += 1
                    order_time = getattr(order, 'time_setup', 0)

                    # For orders >1 hour old, trigger re-analysis
                    if (current_time - order_time) > stale_threshold and (current_time - order_time) <= very_stale_threshold:
                        logger.warning(f"Stale pending order detected: {order.ticket} (age: {(current_time - order_time)/3600:.1f}h) - Re-analysis recommended")
                        # TODO: Implement re-analysis logic here
                        # For now, just log - in future, call analysis functions to decide keep/cancel/modify
                        continue

                    # Only cancel VERY stale orders (>2 hours old) that belong to our system
                    if (current_time - order_time) > very_stale_threshold:
                        self._cancel_order(order.ticket)
                        cancelled += 1
                        logger.info(f"Cancelled very stale order: {order.ticket} (age: {(current_time - order_time)/3600:.1f}h)")
                        continue

                    # Don't check signal validity or price movement every cycle
                    # This was causing the death loop

                except Exception as e:
                    logger.error(f"Error managing order {order.ticket}: {e}")
                    errors += 1

            return {
                'managed': managed,
                'cancelled': cancelled,
                'errors': errors
            }

        except Exception as e:
            logger.error(f"Error in pending order management: {e}")
            return {'managed': 0, 'cancelled': 0, 'errors': 1}

    def _is_order_stale(self, order) -> bool:
        """Check if order is stale"""

        current_time = self.mt5.get_server_time().timestamp() if self.mt5 else datetime.now().timestamp()
        order_time = getattr(order, 'time_setup', 0)
        stale_seconds = self.stale_threshold_hours * 3600
        return (current_time - order_time) > stale_seconds

    def _is_signal_still_valid(self, symbol: str, order) -> bool:
        """Check if signal conditions are still valid"""

        # This is a simplified check - in practice, you'd want to re-evaluate
        # the original signal conditions
        try:
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return False

            current_price = tick.ask if order.type == mt5.ORDER_TYPE_BUY_STOP else tick.bid
            order_price = getattr(order, 'price', 0)

            # Check if price has moved significantly away from order
            price_diff = abs(current_price - order_price)
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                # If price has moved more than 5% away from order price, reconsider
                if price_diff / current_price > 0.05:
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking signal validity for {symbol}: {e}")
            return False

    def _price_moved_too_far(self, order) -> bool:
        """Check if price has moved too far from order"""

        try:
            symbol = getattr(order, 'symbol', '')
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return False

            current_price = tick.ask if order.type == mt5.ORDER_TYPE_BUY_STOP else tick.bid
            order_price = getattr(order, 'price', 0)

            return abs(current_price - order_price) / order_price > self.price_movement_threshold

        except Exception:
            return False

    def _cancel_order(self, ticket: int) -> bool:
        """Cancel a pending order"""

        try:
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": ticket
            }

            result = mt5.order_send(request)
            return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE

        except Exception as e:
            logger.error(f"Error cancelling order {ticket}: {e}")
            return False

    def _modify_or_cancel_order(self, order) -> None:
        """Modify order or cancel if modification not possible"""

        # For now, just cancel - modification logic would be more complex
        self._cancel_order(order.ticket)
        logger.info(f"Cancelled order {order.ticket} due to price movement")


class OrderExecutor:
    """Handles order execution and validation through MT5"""

    def __init__(self, mt5_connector, config: dict, risk_manager=None, technical_analyzer=None):
        """Initialize order executor"""
        self.mt5 = mt5_connector
        self.config = config
        self.risk_manager = risk_manager  # Add risk manager
        self.technical_analyzer = technical_analyzer  # Add technical analyzer
        self.magic_number = config.get('trading', {}).get('magic_number')
        self.max_slippage = config.get('trading', {}).get('max_slippage')
        self.min_risk_reward_ratio = config.get('trading', {}).get('min_risk_reward_ratio')
        self.dry_run = config.get('trading', {}).get('dry_run')

        # Initialize learning database for recording stop orders
        self.learning_db = LearningDatabase()

        # Initialize order manager
        self.order_manager = OrderManager(self.mt5, self.config, self)

    def check_pending_orders_health(self) -> Dict:
        """Check health of pending orders for monitoring system"""

        try:
            orders = mt5.orders_get()
            if orders is None:
                return {
                    'total_pending': 0,
                    'issues': [],
                    'cancel_rate': 0.0,
                    'fill_rate': 0.0
                }

            total_pending = len(orders)
            issues = []

            # Check for too many pending orders
            max_pending = self.config.get('trading', {}).get('order_management', {}).get('pending_order_management', {}).get('max_pending_orders', 10)
            if total_pending > max_pending:
                issues.append(f"Too many pending orders: {total_pending}")

            # Check for stale orders (>1 hour old)
            current_time = self.mt5.get_server_time().timestamp() if self.mt5 else datetime.now().timestamp()
            stale_count = 0
            for order in orders:
                order_time = getattr(order, 'time_setup', 0)
                if (current_time - order_time) > 3600:  # 1 hour
                    stale_count += 1

            if stale_count > 0:
                issues.append(f"Stale orders found: {stale_count}")

            # Calculate rates (simplified - would need historical data for accurate rates)
            cancel_rate = 0.0  # Placeholder
            fill_rate = 0.0    # Placeholder

            return {
                'total_pending': total_pending,
                'issues': issues,
                'cancel_rate': cancel_rate,
                'fill_rate': fill_rate
            }

        except Exception as e:
            logger.error(f"Error checking pending orders health: {e}")
            return {
                'total_pending': 0,
                'issues': [f"Error checking orders: {str(e)}"],
                'cancel_rate': 0.0,
                'fill_rate': 0.0
            }

    def get_filling_mode(self, symbol):
        """Get appropriate filling mode for symbol"""
        try:
            # First check terminal-wide filling mode support
            terminal_info = mt5.terminal_info()  # type: ignore
            terminal_filling_modes = 0
            try:
                if terminal_info and hasattr(terminal_info, 'trade_filling_flags'):
                    terminal_filling_modes = terminal_info.trade_filling_flags
                    logger.debug(f"Terminal supports filling modes: {terminal_filling_modes}")
                else:
                    logger.debug(f"Terminal filling mode info not available")
            except Exception as e:
                logger.debug(f"Could not get terminal filling modes: {e}")

            symbol_info = mt5.symbol_info(symbol)  # type: ignore
            if symbol_info is None:
                logger.warning(f"No symbol info for {symbol}, using FOK")
                return mt5.ORDER_FILLING_FOK  # Fallback

            # Check available filling modes (both terminal and symbol level)
            filling_modes = symbol_info.filling_mode
            logger.debug(f"{symbol} supports filling modes: {filling_modes}")

            # If terminal has filling mode restrictions, respect them
            if terminal_info and hasattr(terminal_info, 'trade_filling_flags'):
                filling_modes &= terminal_info.trade_filling_flags
                logger.debug(f"{symbol} filling modes after terminal restrictions: {filling_modes}")
            else:
                logger.debug(f"Terminal filling mode restrictions not available, using symbol modes only")

            # Try modes in order of preference, but be more permissive
            if filling_modes & mt5.ORDER_FILLING_IOC:
                return mt5.ORDER_FILLING_IOC
            elif filling_modes & mt5.ORDER_FILLING_FOK:
                return mt5.ORDER_FILLING_FOK
            elif filling_modes & mt5.ORDER_FILLING_RETURN:
                return mt5.ORDER_FILLING_RETURN
            else:
                # If no standard modes supported, try immediate or return
                logger.warning(f"{symbol} doesn't support standard filling modes, trying alternatives")
                if filling_modes & mt5.ORDER_FILLING_IMMEDIATE:
                    return mt5.ORDER_FILLING_IMMEDIATE
                else:
                    logger.error(f"{symbol} has no supported filling modes: {filling_modes}")
                    return mt5.ORDER_FILLING_FOK  # Last resort

        except Exception as e:
            logger.error(f"Error getting filling mode for {symbol}: {e}")
            return mt5.ORDER_FILLING_FOK

    def _calculate_min_stop_distance(self, symbol: str, symbol_info) -> float:
        """Calculate minimum stop distance in price units - Professional Day Trading Standards"""

        # Get professional minimums from config
        stop_loss_config = self.config.get('trading_rules', {}).get('stop_loss_rules', {})
        professional_minimums = stop_loss_config.get('professional_minimums', {})

        # Get broker's technical requirement
        stops_level = getattr(symbol_info, 'trade_stops_level', 0)
        point_size = symbol_info.point
        broker_minimum = stops_level * point_size

        # Get professional minimum for this symbol (default to 15 pips if not specified)
        professional_minimum = professional_minimums.get(symbol, 0.0015)

        # Use the MAXIMUM of:
        # 1. Professional minimum (allows normal market movement)
        # 2. Broker requirement * 1.5 (safety margin above broker min)
        final_minimum = max(professional_minimum, broker_minimum * 1.5)

        logger.debug(
            f"{symbol}: Min stop - Professional: {professional_minimum:.5f}, "
            f"Broker: {broker_minimum:.5f}, Final: {final_minimum:.5f}"
        )

        return final_minimum

    def _adjust_stop_loss(self, symbol: str, order_type: str, price: float,
                         stop_loss: float, symbol_info) -> float:
        """Adjust stop loss to meet broker requirements - ensure minimum distance"""
        min_stop_distance = self._calculate_min_stop_distance(symbol, symbol_info)

        logger.info(
            f"Symbol {symbol}: min_stop_distance={min_stop_distance}")
        logger.info(
            f"Order {order_type} {symbol}: price={price}, original_sl={stop_loss}")

        if order_type.lower() in ['buy', 'buy_limit', 'buy_stop']:
            # For buy orders, stop loss should be below price
            required_sl = price - min_stop_distance
            if stop_loss >= required_sl:  # SL is too close to price
                logger.info(
                    f"Adjusting BUY stop loss from {stop_loss} to {required_sl} (moving further away)")
                return required_sl
        else:
            # For sell orders, stop loss should be above price
            required_sl = price + min_stop_distance
            if stop_loss <= required_sl:  # SL is too close to price
                logger.info(
                    f"Adjusting SELL stop loss from {stop_loss} to {required_sl} (moving further away)")
                return required_sl

        return stop_loss

    def _adjust_stop_order_price(self, symbol: str, order_type: str, current_price: float,
                                order_price: float, symbol_info) -> float:
        """Adjust stop order price to meet broker minimum distance requirements"""
        min_stop_distance = self._calculate_min_stop_distance(symbol, symbol_info)

        logger.info(
            f"[{symbol}] Checking stop order price: current_price={current_price:.5f}, "
            f"order_price={order_price:.5f}, min_distance={min_stop_distance:.5f}")

        if order_type.lower() == 'buy_stop':
            # BUY STOP must be above current price
            min_allowed_price = current_price + min_stop_distance
            if order_price <= min_allowed_price:
                logger.info(
                    f"[{symbol}] Adjusting BUY STOP price from {order_price:.5f} to {min_allowed_price:.5f} "
                    f"(must be at least {min_stop_distance:.5f} above current price)")
                return min_allowed_price
        elif order_type.lower() == 'sell_stop':
            # SELL STOP must be below current price
            max_allowed_price = current_price - min_stop_distance
            if order_price >= max_allowed_price:
                logger.info(
                    f"[{symbol}] Adjusting SELL STOP price from {order_price:.5f} to {max_allowed_price:.5f} "
                    f"(must be at least {min_stop_distance:.5f} below current price)")
                return max_allowed_price

        return order_price

    def _adjust_take_profit(self, symbol: str, order_type: str, price: float,
                           take_profit: float, symbol_info) -> float:
        """Adjust take profit to meet broker requirements"""
        min_stop_distance = self._calculate_min_stop_distance(symbol, symbol_info)

        logger.info(
            f"Order {order_type} {symbol}: price={price}, original_tp={take_profit}")

        if order_type.lower() in ['buy', 'buy_limit', 'buy_stop']:
            # For buy orders, take profit should be above price
            required_tp = price + min_stop_distance
            if take_profit <= required_tp:
                logger.info(
                    f"Adjusting BUY take profit from {take_profit} to {required_tp} (broker minimum)")
                return required_tp
        else:
            # For sell orders, take profit should be below price
            required_tp = price - min_stop_distance
            if take_profit >= required_tp:
                logger.info(
                    f"Adjusting SELL take profit from {take_profit} to {required_tp} (broker minimum)")
                return required_tp

        return take_profit

    def _validate_risk_reward_ratio(self, symbol: str, order_type: str, price: float,
                                   stop_loss: float, take_profit: float) -> bool:
        """Validate risk-reward ratio meets symbol-specific minimum requirements"""
        try:
            # Use risk manager's dynamic RR validation
            if hasattr(self, 'risk_manager') and self.risk_manager:
                is_valid, reason = self.risk_manager.validate_risk_reward(symbol, price, stop_loss, take_profit)
                if not is_valid:
                    logger.error(f"Order rejected: {reason} for {order_type}")
                    return False
                
                logger.info(f"Order validated: {reason} for {order_type}")
                return True
            else:
                # Fallback to old logic if risk_manager not available
                logger.warning("Risk manager not available, using fallback RR validation")
                min_ratio = self.min_risk_reward_ratio  # 3.0 for all stop orders
                
                risk_distance = abs(stop_loss - price)
                reward_distance = abs(take_profit - price)
                ratio = reward_distance / risk_distance if risk_distance > 0 else 0
                
                # Allow for small floating point precision errors
                if ratio < (min_ratio - 0.05):  # Slightly below minimum to account for precision
                    logger.error(
                        f"Order rejected: insufficient risk-reward ratio {ratio:.2f}:1 (required: {min_ratio}:1 for {order_type})")
                    return False
                
                logger.info(f"Order validated: {ratio:.1f}:1 risk-reward ratio (min: {min_ratio}:1 for {order_type})")
                return True
                
        except Exception as e:
            logger.error(f"Error validating risk-reward ratio for {symbol}: {e}")
            return False

    def _round_prices_to_symbol_precision(self, symbol: str, price: float = None,
                                        stop_loss: float = None, take_profit: float = None,
                                        symbol_info=None):
        """Round prices to symbol's decimal precision"""
        if symbol_info is None:
            symbol_info = mt5.symbol_info(symbol)  # type: ignore

        if price is not None:
            price = round(price, symbol_info.digits)
        if stop_loss is not None:
            stop_loss = round(stop_loss, symbol_info.digits)
        if take_profit is not None:
            take_profit = round(take_profit, symbol_info.digits)

        return price, stop_loss, take_profit

    def _ensure_broker_minimum_stops(self, symbol: str, order_type: str, price: float,
                                    stop_loss: float = None, take_profit: float = None,
                                    symbol_info=None):
        """Ensure SL/TP meet broker minimum stop distance requirements"""
        if symbol_info is None:
            symbol_info = mt5.symbol_info(symbol)  # type: ignore

        min_stop_points = getattr(symbol_info, 'trade_stops_level', 0)
        min_stop_distance = max(min_stop_points * symbol_info.point, 0.0001)

        # Adjust stop loss if too close
        if stop_loss is not None:
            actual_sl_distance = abs(price - stop_loss)
            if actual_sl_distance < min_stop_distance:
                logger.debug(
                    f"[WARNING] BROKER MINIMUM STOP: Required "
                    f"{min_stop_distance:.5f}, have {actual_sl_distance:.5f}")
                # Adjust to meet broker minimum
                if order_type.lower() in ['buy', 'buy_limit', 'buy_stop']:
                    stop_loss = price - min_stop_distance
                else:
                    stop_loss = price + min_stop_distance
                stop_loss = round(stop_loss, symbol_info.digits)
                logger.debug(f"Adjusted SL to: {stop_loss}")

        # Adjust take profit if too close
        if take_profit is not None:
            actual_tp_distance = abs(price - take_profit)
            if actual_tp_distance < min_stop_distance:
                logger.debug(
                    f"[WARNING] BROKER MINIMUM TP: Required "
                    f"{min_stop_distance:.5f}, have {actual_tp_distance:.5f}")
                # Adjust to meet broker minimum
                if order_type.lower() in ['buy', 'buy_limit', 'buy_stop']:
                    take_profit = price + min_stop_distance
                else:
                    take_profit = price - min_stop_distance


                take_profit = round(take_profit, symbol_info.digits)
                logger.debug(f"Adjusted TP to: {take_profit}")

        return stop_loss, take_profit

    async def place_order(self, symbol: str, order_type: str, volume: float,
                          stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                          price: Optional[float] = None, comment: str = "", signal_data: Optional[Dict] = None) -> Dict:
        """Place order through MT5 - ASYNC - ONLY STOP ORDERS ALLOWED"""
        try:
            # Check MT5 connection
            terminal_info = mt5.terminal_info()  # type: ignore
            if terminal_info is None:
                logger.error("MT5 terminal not connected")
                return {
                    'success': False,
                    'error': 'MT5 terminal not connected'}

            # Select symbol for trading
            if not mt5.symbol_select(symbol, True):  # type: ignore
                logger.error(f"Failed to select symbol {symbol}")
                return {
                    'success': False,
                    'error': f'Failed to select symbol {symbol}'}

            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)  # type: ignore
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found")
                return {'success': False, 'error': 'Symbol not found'}

            # CHECK FOR DUPLICATE SYMBOLS: No stop orders or positions for same symbol
            # Check existing positions
            positions = mt5.positions_get(symbol=symbol)  # type: ignore
            if positions and len(positions) > 0:
                for pos in positions:
                    if hasattr(pos, 'magic') and pos.magic == self.magic_number:
                        logger.warning(f"[{symbol}] Cannot place stop order: existing position found (ticket {pos.ticket})")
                        return {
                            'success': False,
                            'error': f'Existing position for {symbol} (ticket {pos.ticket})'}

            # Check existing pending orders
            orders = mt5.orders_get(symbol=symbol)  # type: ignore
            if orders and len(orders) > 0:
                for order in orders:
                    if hasattr(order, 'magic') and order.magic == self.magic_number:
                        logger.warning(f"[{symbol}] Cannot place stop order: existing pending order found (ticket {order.ticket})")
                        return {
                            'success': False,
                            'error': f'Existing pending order for {symbol} (ticket {order.ticket})'}

            # FORCE STOP ORDERS ONLY: Convert any order type to stop order
            if order_type.lower() == 'buy':
                order_type = 'buy_stop'
                logger.info(f"[{symbol}] Converting BUY to BUY_STOP order")
            elif order_type.lower() == 'sell':
                order_type = 'sell_stop'
                logger.info(f"[{symbol}] Converting SELL to SELL_STOP order")
            elif order_type.lower() in ['buy_limit', 'sell_limit']:
                # Convert limit orders to stop orders
                if order_type.lower() == 'buy_limit':
                    order_type = 'buy_stop'
                else:
                    order_type = 'sell_stop'
                logger.info(f"[{symbol}] Converting LIMIT to STOP order")

            # Get current price for stop order placement
            tick = mt5.symbol_info_tick(symbol)  # type: ignore
            if tick is None:
                logger.error(f"Failed to get tick data for {symbol}")
                return {'success': False,
                        'error': f'Failed to get tick data for {symbol}'}

            # Calculate stop order price based on direction and current market
            current_price = (tick.ask + tick.bid) / 2  # Use mid price as reference

            # Get stop order distance configuration - FORCE PIP-BASED
            trading_config = self.config.get('trading', {})
            if not isinstance(trading_config, dict):
                trading_config = {}
            pending_config = trading_config.get('pending_order_distances', {})
            if not isinstance(pending_config, dict):
                pending_config = {}
            distance_type = 'pips'  # Force pip-based distances

            if distance_type == 'percentage':
                # Percentage-based distance calculation
                if 'XAU' in symbol or 'GOLD' in symbol:
                    min_percent = pending_config.get('xauusd_min_percent')
                    max_percent = pending_config.get('xauusd_max_percent')
                elif 'XAG' in symbol or 'SILVER' in symbol:
                    min_percent = pending_config.get('xagusd_min_percent')
                    max_percent = pending_config.get('xagusd_max_percent')
                else:
                    min_percent = pending_config.get('forex_min_percent')
                    max_percent = pending_config.get('forex_max_percent')

                # Adjust stop distance based on signal analysis
                risk_factor = 1.0
                if signal_data:
                    # Apply fundamental risk multiplier
                    risk_multiplier = signal_data.get('risk_multiplier', 1.0)
                    risk_factor *= risk_multiplier

                    # Increase stop distance for low signal strength (higher risk)
                    signal_strength = signal_data.get('signal_strength', 0.5)
                    if signal_strength < 0.4:
                        risk_factor *= 1.5
                    elif signal_strength < 0.6:
                        risk_factor *= 1.2

                    # Adjust based on fundamental analysis
                    fundamental_score = signal_data.get('fundamental_score', 0.5)
                    if order_type.lower() == 'buy_stop' and fundamental_score < 0.4:
                        # Bearish fundamental for buy signal - increase stop
                        risk_factor *= 1.2
                    elif order_type.lower() == 'sell_stop' and fundamental_score > 0.6:
                        # Bullish fundamental for sell signal - increase stop
                        risk_factor *= 1.2

                    # Adjust based on sentiment analysis
                    sentiment_score = signal_data.get('sentiment_score', 0.5)
                    if sentiment_score < 0.4:
                        # Negative sentiment - increase stop distance
                        risk_factor *= 1.1

                    # Cap risk factor to prevent excessive stops
                    max_risk_factor = 2.0 if ('XAU' in symbol or 'GOLD' in symbol or 'XAG' in symbol or 'SILVER' in symbol) else 3.0
                    risk_factor = min(risk_factor, max_risk_factor)

                    logger.info(f"[{symbol}] Risk factor: {risk_factor:.2f} (multiplier: {risk_multiplier:.2f}, strength: {signal_strength:.2f}, fund: {fundamental_score:.2f}, sent: {sentiment_score:.2f})")

                # Apply risk factor to percentage ranges
                min_percent *= risk_factor
                max_percent *= risk_factor

                # Generate random percentage within range
                import random
                random_percent = random.uniform(min_percent, max_percent)
                distance = current_price * (random_percent / 100.0)

                logger.info(f"[{symbol}] Percentage distance: {random_percent:.2f}% of {current_price:.5f} = {distance:.5f}")

            else:
                # Pip-based distance calculation (original logic)
                # Determine pip ranges based on symbol
                if 'XAU' in symbol or 'GOLD' in symbol:
                    min_pips = pending_config.get('xauusd_min_pips')
                    max_pips = pending_config.get('xauusd_max_pips')
                elif 'XAG' in symbol or 'SILVER' in symbol:
                    min_pips = pending_config.get('xagusd_min_pips')
                    max_pips = pending_config.get('xagusd_max_pips')
                else:
                    min_pips = pending_config.get('forex_min_pips')
                    max_pips = pending_config.get('forex_max_pips')

                # Adjust stop distance based on signal analysis
                risk_factor = 1.0
                if signal_data:
                    # Apply fundamental risk multiplier
                    risk_multiplier = signal_data.get('risk_multiplier', 1.0)
                    risk_factor *= risk_multiplier

                    # Increase stop distance for low signal strength (higher risk)
                    signal_strength = signal_data.get('signal_strength', 0.5)
                    if signal_strength < 0.4:
                        risk_factor *= 1.5
                    elif signal_strength < 0.6:
                        risk_factor *= 1.2

                    # Adjust based on fundamental analysis
                    fundamental_score = signal_data.get('fundamental_score', 0.5)
                    if order_type.lower() == 'buy_stop' and fundamental_score < 0.4:
                        # Bearish fundamental for buy signal - increase stop
                        risk_factor *= 1.2
                    elif order_type.lower() == 'sell_stop' and fundamental_score > 0.6:
                        # Bullish fundamental for sell signal - increase stop
                        risk_factor *= 1.2

                    # Adjust based on sentiment analysis
                    sentiment_score = signal_data.get('sentiment_score', 0.5)
                    if sentiment_score < 0.4:
                        # Negative sentiment - increase stop distance
                        risk_factor *= 1.1

                    # Cap risk factor to prevent excessive stops
                    max_risk_factor = 5.0  # Allow higher risk factors, readjustment will handle excessive distances
                    risk_factor = min(risk_factor, max_risk_factor)

                    logger.info(f"[{symbol}] Risk factor: {risk_factor:.2f} (multiplier: {risk_multiplier:.2f}, strength: {signal_strength:.2f}, fund: {fundamental_score:.2f}, sent: {sentiment_score:.2f})")

                # Apply risk factor to pip ranges
                min_pips *= risk_factor
                max_pips *= risk_factor

                # Readjust stop distances if they exceed threshold levels
                if 'XAU' in symbol or 'GOLD' in symbol:
                    if max_pips > 110:
                        max_pips = 55  # Readjust gold stops if over 110 pips
                        logger.info(f"[{symbol}] Readjusted gold stop distance from >110 to 55 pips")
                elif 'XAG' in symbol or 'SILVER' in symbol:
                    if max_pips > 600:
                        max_pips = 300  # Readjust silver stops if over 600 pips
                        logger.info(f"[{symbol}] Readjusted silver stop distance from >600 to 300 pips")
                else:
                    if max_pips > 30:
                        max_pips = 15  # Readjust forex stops if over 30 pips
                        logger.info(f"[{symbol}] Readjusted forex stop distance from >30 to 15 pips")

                # Calculate pip size
                if 'XAU' in symbol or 'GOLD' in symbol:
                    pip_size = symbol_info.point * 10  # Gold: 1 pip = 10 points
                elif symbol_info.digits == 3 or symbol_info.digits == 5:
                    pip_size = symbol_info.point * 10
                else:
                    pip_size = symbol_info.point

                # Generate random distance within range for stop order placement
                import random
                random_pips = random.uniform(min_pips, max_pips)
                distance = random_pips * pip_size

                logger.info(f"[{symbol}] Pip distance: {random_pips:.1f} pips = {distance:.5f}")

            # Store distance parameters for recording
            self._last_min_pips = min_pips
            self._last_max_pips = max_pips
            self._last_actual_pips = random_pips if 'random_pips' in locals() else random_pips
            self._last_risk_factor = risk_factor

            # Calculate stop order price based on direction
            if order_type.lower() == 'buy_stop':
                # BUY STOP: place above current price (breakout buying)
                price = current_price + distance
                if distance_type == 'percentage':
                    distance_display = f"{random_percent:.2f}%"
                else:
                    distance_display = f"{random_pips:.1f} pips"
                logger.info(f"[{symbol}] BUY STOP: {distance_display} above current price {current_price:.5f} -> {price:.5f}")
            elif order_type.lower() == 'sell_stop':
                # SELL STOP: place below current price (breakdown selling)
                price = current_price - distance
                if distance_type == 'percentage':
                    distance_display = f"{random_percent:.2f}%"
                else:
                    distance_display = f"{random_pips:.1f} pips"
                logger.info(f"[{symbol}] SELL STOP: {distance_display} below current price {current_price:.5f} -> {price:.5f}")
            else:
                return {'success': False,
                        'error': f'Invalid order type for stop-only system: {order_type}'}

            # Determine MT5 order type (ONLY STOP ORDERS ALLOWED)
            if order_type.lower() == 'buy_stop':
                mt5_order_type = mt5.ORDER_TYPE_BUY_STOP
            elif order_type.lower() == 'sell_stop':
                mt5_order_type = mt5.ORDER_TYPE_SELL_STOP
            else:
                return {'success': False,
                        'error': f'Only stop orders allowed: {order_type} is not supported'}

            # Recalculate stop loss and take profit based on stop order price
            if stop_loss is not None and price is not None:
                # Get default pips from config
                config = getattr(self, 'config', {})
                default_sl_pips = config.get('trading', {}).get('default_sl_pips')
                default_tp_pips = config.get('trading', {}).get('default_tp_pips')
                
                # Calculate pip size
                if "XAU" in symbol or "GOLD" in symbol:
                    pip_size = symbol_info.point * 10  # Gold: 1 pip = 10 points
                elif "XAG" in symbol or "SILVER" in symbol:
                    pip_size = symbol_info.point  # Silver: 1 pip = 1 point (0.01 price units)
                elif symbol_info.digits == 3 or symbol_info.digits == 5:
                    pip_size = symbol_info.point * 10
                else:
                    pip_size = symbol_info.point
                
                # For stop orders, SL/TP are calculated from the stop order price
                if order_type.lower() == 'buy_stop':
                    # BUY STOP: SL below stop price, TP above stop price
                    stop_loss = price - (default_sl_pips * pip_size)
                    if take_profit is not None:
                        take_profit = price + (default_tp_pips * pip_size)
                elif order_type.lower() == 'sell_stop':
                    # SELL STOP: SL above stop price, TP below stop price
                    stop_loss = price + (default_sl_pips * pip_size)
                    if take_profit is not None:
                        take_profit = price - (default_tp_pips * pip_size)
                
                # Round to symbol precision
                stop_loss = round(stop_loss, symbol_info.digits)
                if take_profit is not None:
                    take_profit = round(take_profit, symbol_info.digits)
                
                tp_display = f"{take_profit:.5f}" if take_profit is not None else "None"
                logger.info(f"[{symbol}] SL/TP for {order_type} @ {price:.5f}: SL={stop_loss:.5f}, TP={tp_display}")

            # Adjust stop order price to meet broker minimum distance requirements
            if price is not None and current_price is not None:
                price = self._adjust_stop_order_price(symbol, order_type, current_price, price, symbol_info)

            # Adjust stop loss to meet minimum broker requirements
            if stop_loss is not None and price is not None:
                stop_loss = self._adjust_stop_loss(symbol, order_type, price, stop_loss, symbol_info)

            # Adjust take profit to meet minimum requirements
            if take_profit is not None and price is not None:
                take_profit = self._adjust_take_profit(symbol, order_type, price, take_profit, symbol_info)

            # Final validation: ensure adequate risk-reward ratio
            if stop_loss is not None and take_profit is not None and price is not None:
                if not self._validate_risk_reward_ratio(symbol, order_type, price, stop_loss, take_profit):
                    return {
                        'success': False,
                        'error': f'Insufficient risk-reward ratio'}

            # Check if dry run mode is enabled
            if self.dry_run:
                tp_display = f"{take_profit:.5f}" if take_profit is not None else "None"
                sl_display = f"{stop_loss:.5f}" if stop_loss is not None else "None"
                logger.info(f"[DRY RUN] Would place STOP ORDER for {symbol}: {order_type} @ {price:.5f}")
                logger.info(f"[DRY RUN] SL: {sl_display}, TP: {tp_display}, Volume: {volume}")
                
                # Record dry run stop order in learning database for AI improvement
                try:
                    # Convert MT5 order type to string
                    order_type_str = 'BUY_STOP' if mt5_order_type == mt5.ORDER_TYPE_BUY_STOP else 'SELL_STOP'
                    
                    # Calculate spread
                    spread = (tick.ask - tick.bid) if tick else None
                    
                    logger.info(f"[{symbol}] Recording dry run stop order: ticket=999999, type={order_type_str}, price={price:.5f}")
                    
                    # Record the dry run stop order
                    self.learning_db.record_stop_order(
                        ticket=999999,  # Fake ticket for dry run
                        symbol=symbol,
                        order_type=order_type_str,
                        order_price=price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        volume=volume,
                        signal_data=signal_data,
                        min_pips=min_pips if 'min_pips' in locals() else None,
                        max_pips=max_pips if 'max_pips' in locals() else None,
                        actual_pips=random_pips if 'random_pips' in locals() else None,
                        risk_factor=risk_factor if 'risk_factor' in locals() else None,
                        market_price=current_price,
                        spread=spread,
                        placement_reason="AI generated stop order (DRY RUN)"
                    )
                    logger.info(f"[{symbol}] Successfully recorded dry run stop order in learning database")
                except Exception as e:
                    logger.error(f"[{symbol}] Failed to record dry run stop order in learning database: {e}")
                    import traceback
                    logger.error(f"[{symbol}] Dry run recording traceback: {traceback.format_exc()}")
                
                return {
                    'success': True,
                    'order': 999999,  # Fake order ticket
                    'price': price,
                    'sl': stop_loss,
                    'tp': take_profit,
                    'dry_run': True
                }

            # Try different filling modes if the first one fails
            preferred_filling_mode = self.get_filling_mode(symbol)
            filling_modes_to_try = [
                preferred_filling_mode,  # Preferred mode first
                mt5.ORDER_FILLING_IOC,         # Immediate or Cancel
                mt5.ORDER_FILLING_RETURN,      # Return remaining
                mt5.ORDER_FILLING_FOK,         # Fill or Kill
            ]
            
            # Remove duplicates but keep order
            seen = set()
            filling_modes_to_try = [x for x in filling_modes_to_try if not (x in seen or seen.add(x))]
            
            result = None
            last_error = None
            
            for filling_mode in filling_modes_to_try:
                try:
                    # Create order request
                    # Use PENDING action for limit/stop orders, DEAL for market orders
                    if mt5_order_type in [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT, 
                                         mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_SELL_STOP]:
                        trade_action = mt5.TRADE_ACTION_PENDING
                        filling_mode = None  # Pending orders don't use filling modes
                    else:
                        trade_action = mt5.TRADE_ACTION_DEAL
                        # Keep filling mode for market orders
                    
                    request = {
                        "action": trade_action,
                        "symbol": symbol,
                        "volume": volume,
                        "type": mt5_order_type,
                        "price": price,
                        "magic": self.magic_number,
                        "comment": comment or "FX-Ai",
                        "type_time": mt5.ORDER_TIME_DAY,  # Expire at end of trading day
                    }

                    # Add SL/TP for pending orders (they are set at order placement)
                    if trade_action == mt5.TRADE_ACTION_PENDING:
                        if stop_loss is not None:
                            request["sl"] = stop_loss
                        if take_profit is not None:
                            request["tp"] = take_profit

                    # Add filling mode only for market orders
                    if trade_action == mt5.TRADE_ACTION_DEAL:
                        request["deviation"] = self.max_slippage
                        if filling_mode is not None:
                            request["type_filling"] = filling_mode
                        else:
                            request["type_filling"] = mt5.ORDER_FILLING_IOC

                    # Note: SL/TP are included in pending order requests and will be applied when order fills
                    # For market orders, SL/TP will be set after order placement using TRADE_ACTION_SLTP

                    logger.debug(f"Trying filling mode {filling_mode} ({type(filling_mode).__name__}) for {symbol}")
                    
                    # Send order
                    result = mt5.order_send(request)  # type: ignore

                    # Check if order_send returned None
                    if result is None:
                        last_error = f"Filling mode {filling_mode}: Order send failed - no response from MT5"
                        logger.debug(last_error)
                        continue

                    # Define success codes
                    SUCCESS_CODES = [mt5.TRADE_RETCODE_PLACED, mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL]

                    if result.retcode in SUCCESS_CODES:
                        logger.info(f"Order placed with filling mode {filling_mode}: {symbol} {order_type} @ {price}")
                        break  # Success, exit the retry loop
                    else:
                        error_desc = f"Filling mode {filling_mode}: Retcode {result.retcode}"
                        if hasattr(result, 'comment') and result.comment:
                            error_desc += f", Comment: {result.comment}"
                        logger.debug(f"Filling mode {filling_mode} failed: {error_desc}")
                        last_error = error_desc
                        continue
                        
                except Exception as e:
                    last_error = str(e)
                    logger.debug(f"Error with filling mode {filling_mode}: {e}")
                    continue
            
            # If all filling modes failed, try pending orders as fallback (only for market orders)
            if result is None or result.retcode not in [10008, 10009, 10010]:
                # Only try pending fallback for market orders, not for stop orders that failed as pending
                if trade_action != mt5.TRADE_ACTION_PENDING:
                    logger.info(f"Market orders failed for {symbol}, trying pending orders as fallback")
                    pending_result = await self._place_pending_order(
                        symbol, order_type, volume, stop_loss, take_profit, price, comment
                    )
                    if pending_result and pending_result.get('success', False):
                        logger.info(f"Pending order placed successfully for {symbol}")
                        return pending_result
            
            # If both market and pending orders failed, return the last error
            if result is None or result.retcode not in [10008, 10009, 10010]:
                logger.error(f"All order types failed for {symbol}. Last error: {last_error}")
                return {
                    'success': False,
                    'error': last_error or 'All order types failed'
                }

            # Order was successful - handle SL/TP and verification
            logger.info(f"Order placed: {symbol} {order_type} @ {price}")

            # For market orders, set SL/TP after order placement using TRADE_ACTION_SLTP
            # For pending orders, SL/TP are already set in the order request
            if trade_action == mt5.TRADE_ACTION_DEAL and (stop_loss is not None or take_profit is not None):
                await asyncio.sleep(0.2)  # Brief pause before modifying

                # Get the actual position ticket
                positions = mt5.positions_get(symbol=symbol)  # type: ignore
                if positions:
                    position = positions[-1]  # Last position
                    position_ticket = position.ticket

                    modify_request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": position_ticket,
                        "symbol": symbol,
                    }

                    if stop_loss is not None:
                        modify_request["sl"] = stop_loss
                    if take_profit is not None:
                        modify_request["tp"] = take_profit

                    logger.debug(
                        f"Modifying position {position_ticket} with SLTP: {modify_request}")
                    modify_result = mt5.order_send(modify_request)  # type: ignore

                    if modify_result and modify_result.retcode != mt5.TRADE_RETCODE_DONE:
                        logger.warning(
                            f"Failed to set SL/TP for position {position_ticket}: "
                            f"{modify_result.comment}")
                    else:
                        logger.debug("[OK] SLTP modification successful")
                else:
                    logger.debug("[WARNING] No position found to modify SL/TP")

                # Enhanced verification
                await asyncio.sleep(0.5)  # Wait for position to register

                positions = mt5.positions_get(symbol=symbol)  # type: ignore
                if positions:
                    actual_position = positions[-1]
                    actual_sl = actual_position.sl
                    actual_tp = actual_position.tp

                    logger.debug("\n[ORDER VERIFICATION]:")
                    logger.debug(f"Requested SL: {stop_loss}")
                    logger.debug(f"MT5 Set SL: {actual_sl}")
                    logger.debug(f"Requested TP: {take_profit}")
                    logger.debug(f"MT5 Set TP: {actual_tp}")

                    if stop_loss is not None:
                        sl_mismatch = abs(actual_sl - stop_loss)
                        if sl_mismatch > 0.01:
                            logger.error(
                                f"Stop loss mismatch for {symbol}: expected "
                                f"{stop_loss}, got {actual_sl}")
                        else:
                            logger.debug("[OK] SL set correctly")

                    if take_profit is not None:
                        tp_mismatch = abs(actual_tp - take_profit)
                        if tp_mismatch > 0.01:
                            logger.debug(
                                f"[WARNING] TP mismatch! Expected {take_profit}, got {actual_tp}")

                # Record stop order in learning database for AI improvement
                try:
                    # Convert MT5 order type to string
                    order_type_str = 'BUY_STOP' if mt5_order_type == mt5.ORDER_TYPE_BUY_STOP else 'SELL_STOP'
                    
                    # Calculate spread
                    spread = (tick.ask - tick.bid) if tick else None
                    
                    # Ensure signal_data is valid
                    if signal_data is None:
                        signal_data = {}
                    
                    logger.info(f"[{symbol}] Recording stop order: ticket={result.order}, type={order_type_str}, price={price:.5f}")
                    
                    # Record the stop order
                    self.learning_db.record_stop_order(
                        ticket=result.order,
                        symbol=symbol,
                        order_type=order_type_str,
                        order_price=price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        volume=volume,
                        signal_data=signal_data,
                        min_pips=getattr(self, '_last_min_pips', None),
                        max_pips=getattr(self, '_last_max_pips', None),
                        actual_pips=getattr(self, '_last_actual_pips', None),
                        risk_factor=getattr(self, '_last_risk_factor', None),
                        market_price=current_price,
                        spread=spread,
                        placement_reason="AI generated stop order"
                    )
                    logger.info(f"[{symbol}] Successfully recorded stop order in learning database: ticket {result.order}")
                except Exception as e:
                    logger.error(f"[{symbol}] Failed to record stop order in learning database: {e}")
                    import traceback
                    logger.error(f"[{symbol}] Recording traceback: {traceback.format_exc()}")
                    logger.error(f"[{symbol}] Recording parameters: ticket={result.order}, symbol={symbol}, order_type={order_type_str}, price={price}, signal_data_keys={list(signal_data.keys()) if signal_data else None}")

                return {
                    'success': True,
                    'order': result.order,
                    'price': result.price
                }

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def record_stop_order_change(self, ticket: int, symbol: str, change_type: str,
                                old_order_price: float = None, new_order_price: float = None,
                                old_sl: float = None, new_sl: float = None,
                                old_tp: float = None, new_tp: float = None,
                                change_reason: str = None, market_price: float = None,
                                signal_update: dict = None, performance_impact: float = None,
                                was_filled: bool = False):
        """Record stop order changes for AI learning"""
        try:
            self.learning_db.record_stop_order_change(
                original_ticket=ticket,
                symbol=symbol,
                change_type=change_type,
                old_order_price=old_order_price,
                new_order_price=new_order_price,
                old_sl=old_sl,
                new_sl=new_sl,
                old_tp=old_tp,
                new_tp=new_tp,
                change_reason=change_reason,
                market_price=market_price,
                signal_update=signal_update,
                performance_impact=performance_impact,
                was_filled=was_filled
            )
            logger.debug(f"Recorded stop order change: {symbol} ticket {ticket} - {change_type}")
        except Exception as e:
            logger.warning(f"Failed to record stop order change: {e}")

    async def _place_pending_order(self, symbol: str, order_type: str, volume: float,
                                   stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                                   market_price: Optional[float] = None, comment: str = "") -> Optional[Dict]:
        """Place pending order as fallback when market orders fail"""
        try:
            # Get current market price
            tick = mt5.symbol_info_tick(symbol)  # type: ignore
            if tick is None:
                logger.error(f"Failed to get tick data for pending order {symbol}")
                return None

            # Get symbol info for spread and pip calculations
            symbol_info = mt5.symbol_info(symbol)  # type: ignore
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found for pending order")
                return None

            # Calculate pending order price based on order type
            # For buy orders, place below current ask (buy limit)
            # For sell orders, place above current bid (sell limit)
            spread = (tick.ask - tick.bid) / symbol_info.point
            
            # Calculate minimum distance based on broker requirements
            stops_level = getattr(symbol_info, 'trade_stops_level', 20)  # Default 20 points
            min_distance_points = stops_level + 10  # Add margin
            pip_distance = min_distance_points / 10  # Convert points to pips (for 5-digit brokers)
            
            # Ensure minimum distance
            if 'JPY' in symbol:
                pip_distance = max(pip_distance, 20)  # JPY pairs need more
            else:
                pip_distance = max(pip_distance, 20)  # Forex minimum 20 pips away

            if order_type.lower() in ['buy', 'buy_limit']:
                # Buy limit: place below current ask
                pending_price = tick.ask - (pip_distance * symbol_info.point * 10)  # *10 for 5-digit
                mt5_order_type = mt5.ORDER_TYPE_BUY_LIMIT
            elif order_type.lower() in ['sell', 'sell_limit']:
                # Sell limit: place above current bid
                pending_price = tick.bid + (pip_distance * symbol_info.point * 10)
                mt5_order_type = mt5.ORDER_TYPE_SELL_LIMIT
            else:
                logger.error(f"Unsupported order type for pending order: {order_type}")
                return None

            # Round price to symbol precision
            pending_price = round(pending_price, symbol_info.digits)

            logger.info(f"Placing pending {order_type} order for {symbol} @ {pending_price} (market: {tick.ask}/{tick.bid})")

            # Create pending order request
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": volume,
                "type": mt5_order_type,
                "price": pending_price,
                "deviation": self.max_slippage,
                "magic": self.magic_number,
                "comment": comment or "FX-Ai Pending",
                "type_time": mt5.ORDER_TIME_GTC,  # Good till cancelled
                "type_filling": mt5.ORDER_FILLING_RETURN,  # Return remaining
            }

            # Add stop loss and take profit if provided
            if stop_loss is not None:
                request["sl"] = stop_loss
            if take_profit is not None:
                request["tp"] = take_profit

            # Send pending order
            result = mt5.order_send(request)  # type: ignore

            if result is None:
                logger.error(f"Pending order send failed - no response from MT5 for {symbol}")
                return None

            if result.retcode in [10008, 10009, 10010]:  # PLACED, DONE, DONE_PARTIAL
                logger.info(f"Pending order placed successfully: {symbol} {order_type} @ {pending_price}")
                return {
                    'success': True,
                    'order': result.order,
                    'price': pending_price,
                    'pending': True
                }
            else:
                logger.error(f"Pending order failed: {symbol} - Retcode {result.retcode}, Comment: {result.comment}")
                return None

        except Exception as e:
            logger.error(f"Error placing pending order for {symbol}: {e}")
            return None