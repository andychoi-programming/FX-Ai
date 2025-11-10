"""
Fundamental Monitor for Ongoing Trades
Monitors breaking news and fundamental changes during active trades
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class FundamentalMonitor:
    """
    Monitors fundamental events and news for ongoing trades
    Takes action when high-impact events occur
    """

    def __init__(self, trading_engine, fundamental_analyzer, risk_manager, config=None):
        """
        Initialize fundamental monitor

        Args:
            trading_engine: TradingEngine instance
            fundamental_analyzer: FundamentalAnalyzer instance
            risk_manager: RiskManager instance
            config: Configuration dictionary
        """
        self.trading_engine = trading_engine
        self.fundamental = fundamental_analyzer
        self.risk_manager = risk_manager
        self.config = config or {}

        # Monitoring settings
        self.check_interval = self.config.get('fundamental_monitor', {}).get(
            'check_interval_seconds', 300  # Default: 5 minutes
        )

        self.high_impact_exit_threshold = self.config.get('fundamental_monitor', {}).get(
            'high_impact_exit_threshold', 15  # Exit if high-impact news within 15 minutes
        )

        self.sl_tighten_threshold = self.config.get('fundamental_monitor', {}).get(
            'sl_tighten_threshold', 30  # Tighten SL if high-impact news within 30 minutes
        )

        self.sl_tighten_percentage = self.config.get('fundamental_monitor', {}).get(
            'sl_tighten_percentage', 0.5  # Tighten SL to 50% of current distance
        )

        # Control flags
        self.running = False
        self.monitor_task = None

        # Last check times per symbol
        self.last_checks = {}

        # Track actions taken to avoid duplicates
        self.actions_taken = {}  # {position_ticket: {'action': 'exit|tighten', 'timestamp': datetime}}

        logger.info("FundamentalMonitor initialized")

    async def start(self):
        """Start the fundamental monitoring loop"""
        if self.running:
            logger.warning("FundamentalMonitor already running")
            return

        self.running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info(f"FundamentalMonitor started (checking every {self.check_interval} seconds)")

    async def stop(self):
        """Stop the fundamental monitoring loop"""
        if not self.running:
            return

        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("FundamentalMonitor stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop - runs continuously"""
        logger.info("Fundamental monitoring loop started")

        while self.running:
            try:
                # Get all open positions
                positions = mt5.positions_get()  # type: ignore

                if positions:
                    for position in positions:
                        # Only monitor positions managed by FX-Ai
                        if position.magic == self.trading_engine.magic_number:
                            await self._monitor_position(position)

                # Wait before next check
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                logger.info("Monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in fundamental monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    async def _monitor_position(self, position):
        """
        Monitor a single position for fundamental changes

        Args:
            position: MT5 position object
        """
        try:
            symbol = position.symbol
            ticket = position.ticket

            # Check if we already took action on this position
            if ticket in self.actions_taken:
                action_info = self.actions_taken[ticket]
                time_since_action = (datetime.now() - action_info['timestamp']).total_seconds()

                # Don't check again for 30 minutes after taking action
                if time_since_action < 1800:
                    return

            # Get breaking news and events
            breaking_events = self._get_breaking_events_for_symbol(symbol)

            if not breaking_events:
                return

            # Analyze impact and decide action
            for event in breaking_events:
                action = await self._determine_action(position, event)

                if action == 'EXIT':
                    await self._exit_position(position, event)
                    # Record action
                    self.actions_taken[ticket] = {
                        'action': 'exit',
                        'timestamp': datetime.now(),
                        'event': event['title']
                    }
                    break  # Position closed, no need to check other events

                elif action == 'TIGHTEN_SL':
                    await self._tighten_stop_loss(position, event)
                    # Record action
                    self.actions_taken[ticket] = {
                        'action': 'tighten_sl',
                        'timestamp': datetime.now(),
                        'event': event['title']
                    }
                    break  # Only tighten once

                elif action == 'LOCK_PROFITS':
                    await self._lock_profits(position, event)
                    # Record action
                    self.actions_taken[ticket] = {
                        'action': 'lock_profits',
                        'timestamp': datetime.now(),
                        'event': event['title']
                    }
                    break

        except Exception as e:
            logger.error(f"Error monitoring position {position.ticket}: {e}")

    def _get_breaking_events_for_symbol(self, symbol: str) -> List[Dict]:
        """
        Get breaking news/events relevant to this symbol

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')

        Returns:
            List of breaking events affecting this symbol
        """
        try:
            # Get high-impact events
            all_events = self.fundamental.get_high_impact_events()

            if not all_events:
                return []

            # Extract currencies from symbol
            if len(symbol) == 6 and symbol not in ['XAUUSD', 'XAGUSD']:
                base_currency = symbol[:3]
                quote_currency = symbol[3:]
            elif symbol in ['XAUUSD', 'XAGUSD']:
                # Gold and Silver affected by USD events
                base_currency = 'XAU' if symbol == 'XAUUSD' else 'XAG'
                quote_currency = 'USD'
            else:
                return []

            # Filter events relevant to this symbol's currencies
            relevant_events = []
            now = datetime.now()

            for event in all_events:
                # Check if event affects base or quote currency
                event_currency = event.get('country', '').upper()

                if event_currency not in [base_currency, quote_currency]:
                    continue

                # Get event time
                event_time = event.get('datetime') or event.get('time')
                if not event_time:
                    continue

                # Parse event time
                try:
                    if isinstance(event_time, str):
                        event_time = datetime.fromisoformat(event_time.replace('Z', '+00:00'))

                    # Check if event is recent (within last 15 minutes) or upcoming (next 60 minutes)
                    time_diff_minutes = (event_time - now).total_seconds() / 60

                    # Recent event (last 15 minutes)
                    if -15 <= time_diff_minutes <= 0:
                        event['urgency'] = 'IMMEDIATE'  # Just happened
                        event['minutes_ago'] = abs(int(time_diff_minutes))
                        relevant_events.append(event)

                    # Upcoming event (next 60 minutes)
                    elif 0 < time_diff_minutes <= 60:
                        event['urgency'] = 'UPCOMING'
                        event['minutes_until'] = int(time_diff_minutes)
                        relevant_events.append(event)

                except Exception as e:
                    logger.debug(f"Error parsing event time: {e}")
                    continue

            return relevant_events

        except Exception as e:
            logger.error(f"Error getting breaking events for {symbol}: {e}")
            return []

    async def _determine_action(self, position, event: Dict) -> Optional[str]:
        """
        Determine what action to take based on event and position

        Args:
            position: MT5 position object
            event: Event dictionary

        Returns:
            Action to take: 'EXIT', 'TIGHTEN_SL', 'LOCK_PROFITS', or None
        """
        try:
            impact = event.get('impact', 'low').lower()
            urgency = event.get('urgency', 'UPCOMING')
            event_currency = event.get('country', '').upper()

            # Extract position currencies
            symbol = position.symbol
            if len(symbol) == 6 and symbol not in ['XAUUSD', 'XAGUSD']:
                base_currency = symbol[:3]
                quote_currency = symbol[3:]
            else:
                base_currency = 'XAU' if symbol == 'XAUUSD' else 'XAG'
                quote_currency = 'USD'

            # Get current profit status
            current_price = mt5.symbol_info_tick(symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask  # type: ignore
            entry_price = position.price_open

            if position.type == mt5.ORDER_TYPE_BUY:
                profit_pips = (current_price - entry_price)
            else:
                profit_pips = (entry_price - current_price)

            # Convert to pips
            symbol_info = mt5.symbol_info(symbol)  # type: ignore
            if symbol_info:
                if 'JPY' in symbol:
                    profit_pips = profit_pips * 100
                else:
                    profit_pips = profit_pips * 10000

            is_profitable = profit_pips > 0

            # DECISION LOGIC:

            # 1. HIGH IMPACT + IMMEDIATE (just happened) → Consider EXIT or TIGHTEN
            if impact == 'high' and urgency == 'IMMEDIATE':
                minutes_ago = event.get('minutes_ago', 0)

                if minutes_ago <= self.high_impact_exit_threshold:
                    # Event just happened - check if it contradicts position

                    # Get event direction (bullish/bearish for currency)
                    event_bias = self._get_event_bias(event, event_currency)

                    # Determine if event contradicts position
                    if event_currency == base_currency:
                        # Event affects base currency
                        if position.type == mt5.ORDER_TYPE_BUY:
                            # We're long base currency
                            if event_bias == 'BEARISH':
                                logger.warning(f"HIGH IMPACT BEARISH news for {base_currency} while LONG {symbol}")
                                return 'EXIT'  # Contradicts position
                        else:
                            # We're short base currency
                            if event_bias == 'BULLISH':
                                logger.warning(f"HIGH IMPACT BULLISH news for {base_currency} while SHORT {symbol}")
                                return 'EXIT'  # Contradicts position

                    elif event_currency == quote_currency:
                        # Event affects quote currency
                        if position.type == mt5.ORDER_TYPE_BUY:
                            # We're long base (short quote)
                            if event_bias == 'BULLISH':
                                logger.warning(f"HIGH IMPACT BULLISH news for {quote_currency} while SHORT it (via LONG {symbol})")
                                return 'EXIT'
                        else:
                            # We're short base (long quote)
                            if event_bias == 'BEARISH':
                                logger.warning(f"HIGH IMPACT BEARISH news for {quote_currency} while LONG it (via SHORT {symbol})")
                                return 'EXIT'

                    # Event doesn't contradict, but if profitable, lock profits
                    if is_profitable and profit_pips > 10:
                        logger.info(f"HIGH IMPACT news, position profitable - locking profits for {symbol}")
                        return 'LOCK_PROFITS'

                    # Event doesn't contradict, but tighten SL as precaution
                    logger.info(f"HIGH IMPACT news, tightening SL for {symbol}")
                    return 'TIGHTEN_SL'

            # 2. HIGH IMPACT + UPCOMING (about to happen) → TIGHTEN SL
            elif impact == 'high' and urgency == 'UPCOMING':
                minutes_until = event.get('minutes_until', 999)

                if minutes_until <= self.sl_tighten_threshold:
                    # High impact news coming soon
                    if is_profitable and profit_pips > 10:
                        logger.info(f"HIGH IMPACT news in {minutes_until} mins, position profitable - locking profits")
                        return 'LOCK_PROFITS'
                    else:
                        logger.info(f"HIGH IMPACT news in {minutes_until} mins, tightening SL for {symbol}")
                        return 'TIGHTEN_SL'

            # 3. MEDIUM IMPACT + IMMEDIATE → TIGHTEN SL if not profitable
            elif impact == 'medium' and urgency == 'IMMEDIATE':
                if not is_profitable:
                    logger.info(f"MEDIUM IMPACT news, position at loss - tightening SL for {symbol}")
                    return 'TIGHTEN_SL'

            return None  # No action needed

        except Exception as e:
            logger.error(f"Error determining action for position {position.ticket}: {e}")
            return None

    def _get_event_bias(self, event: Dict, currency: str) -> str:
        """
        Determine if event is BULLISH or BEARISH for currency

        Args:
            event: Event dictionary
            currency: Currency code

        Returns:
            'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        try:
            # Check actual vs forecast
            actual = event.get('actual')
            forecast = event.get('forecast')

            if actual is None or forecast is None:
                return 'NEUTRAL'

            # Convert to float if strings
            try:
                actual = float(str(actual).replace('%', '').replace('K', '000').replace('M', '000000'))
                forecast = float(str(forecast).replace('%', '').replace('K', '000').replace('M', '000000'))
            except:
                return 'NEUTRAL'

            event_title = event.get('title', '').lower()

            # Determine if beating forecast is bullish or bearish
            # For most indicators, better than expected = bullish
            bearish_indicators = ['unemployment', 'jobless', 'inflation', 'cpi', 'ppi']

            is_bearish_indicator = any(term in event_title for term in bearish_indicators)

            if actual > forecast:
                # Actual beat forecast
                if is_bearish_indicator:
                    return 'BEARISH'  # Higher unemployment/inflation = bearish
                else:
                    return 'BULLISH'  # Higher GDP/employment = bullish
            elif actual < forecast:
                # Actual missed forecast
                if is_bearish_indicator:
                    return 'BULLISH'  # Lower unemployment/inflation = bullish
                else:
                    return 'BEARISH'  # Lower GDP/employment = bearish
            else:
                return 'NEUTRAL'

        except Exception as e:
            logger.debug(f"Error determining event bias: {e}")
            return 'NEUTRAL'

    async def _exit_position(self, position, event: Dict):
        """
        Exit position due to adverse fundamental news

        Args:
            position: MT5 position object
            event: Event that triggered exit
        """
        try:
            logger.warning(
                f"EXITING {position.symbol} position {position.ticket} "
                f"due to: {event.get('title', 'Unknown event')}"
            )

            # Close the position
            success = await self.trading_engine.close_position(position)

            if success:
                logger.info(f"[SUCCESS] Position {position.ticket} closed successfully")

                # Log to adaptive learning if available
                if hasattr(self.trading_engine, 'adaptive_learning_manager') and self.trading_engine.adaptive_learning_manager:
                    self.trading_engine.adaptive_learning_manager.record_early_exit(
                        position.ticket,
                        position.symbol,
                        'fundamental_news',
                        f"High-impact news: {event.get('title', 'Unknown')}"
                    )
            else:
                logger.error(f"Failed to close position {position.ticket}")

        except Exception as e:
            logger.error(f"Error exiting position {position.ticket}: {e}")

    async def _tighten_stop_loss(self, position, event: Dict):
        """
        Tighten stop loss due to upcoming or recent fundamental news

        Args:
            position: MT5 position object
            event: Event that triggered tightening
        """
        try:
            symbol = position.symbol
            current_sl = position.sl

            if current_sl == 0:
                logger.warning(f"Position {position.ticket} has no SL, cannot tighten")
                return

            # Get current price
            tick = mt5.symbol_info_tick(symbol)  # type: ignore
            if not tick:
                logger.error(f"Cannot get price for {symbol}")
                return

            current_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
            entry_price = position.price_open

            # Calculate current SL distance from entry
            current_sl_distance = abs(entry_price - current_sl)

            # Calculate new tightened SL (closer to entry/breakeven)
            tightened_distance = current_sl_distance * self.sl_tighten_percentage

            if position.type == mt5.ORDER_TYPE_BUY:
                new_sl = entry_price - tightened_distance
            else:
                new_sl = entry_price + tightened_distance

            # Ensure new SL meets broker minimums
            symbol_info = mt5.symbol_info(symbol)  # type: ignore
            if symbol_info:
                min_stop_distance = symbol_info.trade_stops_level * symbol_info.point

                if position.type == mt5.ORDER_TYPE_BUY:
                    min_sl = current_price - min_stop_distance
                    if new_sl >= min_sl:
                        new_sl = min_sl
                else:
                    max_sl = current_price + min_stop_distance
                    if new_sl <= max_sl:
                        new_sl = max_sl

            # Only update if new SL is better (closer to current price)
            should_update = False
            if position.type == mt5.ORDER_TYPE_BUY:
                should_update = new_sl > current_sl
            else:
                should_update = new_sl < current_sl

            if should_update:
                logger.info(
                    f"TIGHTENING SL for {symbol} position {position.ticket} "
                    f"from {current_sl:.5f} to {new_sl:.5f} "
                    f"due to: {event.get('title', 'Unknown event')}"
                )

                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": position.ticket,
                    "sl": new_sl,
                    "tp": position.tp,
                    "magic": self.trading_engine.magic_number
                }

                result = mt5.order_send(request)  # type: ignore
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"[SUCCESS] SL tightened successfully for {symbol}")
                else:
                    logger.error(f"Failed to tighten SL: {result.comment}")
            else:
                logger.debug(f"SL already tight enough for {symbol}")

        except Exception as e:
            logger.error(f"Error tightening SL for position {position.ticket}: {e}")

    async def _lock_profits(self, position, event: Dict):
        """
        Move SL to breakeven or better to lock in profits

        Args:
            position: MT5 position object
            event: Event that triggered profit locking
        """
        try:
            symbol = position.symbol
            entry_price = position.price_open

            # Get current price
            tick = mt5.symbol_info_tick(symbol)  # type: ignore
            if not tick:
                return

            current_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

            # Calculate profit
            if position.type == mt5.ORDER_TYPE_BUY:
                profit = current_price - entry_price
            else:
                profit = entry_price - current_price

            if profit <= 0:
                logger.debug(f"Position {position.ticket} not profitable, cannot lock profits")
                return

            # Calculate breakeven + buffer
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return

            # Buffer: 20% of current profit
            buffer = profit * 0.2

            if position.type == mt5.ORDER_TYPE_BUY:
                new_sl = entry_price + buffer
            else:
                new_sl = entry_price - buffer

            # Ensure meets broker minimums
            min_stop_distance = symbol_info.trade_stops_level * symbol_info.point

            if position.type == mt5.ORDER_TYPE_BUY:
                min_sl = current_price - min_stop_distance
                if new_sl >= min_sl:
                    new_sl = min_sl
            else:
                max_sl = current_price + min_stop_distance
                if new_sl <= max_sl:
                    new_sl = max_sl

            # Check if new SL is better
            current_sl = position.sl
            should_update = False

            if position.type == mt5.ORDER_TYPE_BUY:
                should_update = new_sl > current_sl if current_sl > 0 else True
            else:
                should_update = new_sl < current_sl if current_sl > 0 else True

            if should_update:
                logger.info(
                    f"LOCKING PROFITS for {symbol} position {position.ticket} "
                    f"at {new_sl:.5f} (breakeven + buffer) "
                    f"due to: {event.get('title', 'Unknown event')}"
                )

                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": position.ticket,
                    "sl": new_sl,
                    "tp": position.tp,
                    "magic": self.trading_engine.magic_number
                }

                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"[SUCCESS] Profits locked for {symbol}")
                else:
                    logger.error(f"Failed to lock profits: {result.comment}")

        except Exception as e:
            logger.error(f"Error locking profits for position {position.ticket}: {e}")

    def get_status(self) -> Dict:
        """Get monitor status"""
        return {
            'running': self.running,
            'check_interval': self.check_interval,
            'last_checks': self.last_checks,
            'actions_taken_count': len(self.actions_taken),
            'recent_actions': list(self.actions_taken.values())[-5:]  # Last 5 actions
        }
