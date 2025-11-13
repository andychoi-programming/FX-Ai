"""
Trading Orchestrator Module

Handles the core trading loop, trade monitoring, and orchestration logic
for the FX-Ai trading system.
"""

import asyncio
import json
import time as time_module
from typing import Dict, Any, Optional
import MetaTrader5 as mt5
from datetime import datetime

from core.trading_engine import TradingEngine
from core.risk_manager import RiskManager
from ai.adaptive_learning_manager import AdaptiveLearningManager
from ai.reinforcement_learning_agent import RLAgent
from utils.time_manager import TimeManager


class TradingOrchestrator:
    """
    Handles all trading orchestration logic including the main trading loop,
    trade monitoring, correlation actions, and performance tracking.
    """

    def __init__(self, app):
        """
        Initialize the trading orchestrator.

        Args:
            app: The main FXAiApplication instance
        """
        self.app = app
        # Use trading logger if available (MT5 time), otherwise fallback to main logger
        self.logger = getattr(app, 'trading_logger', app.logger)
        self.config = app.config
        self.mt5 = app.mt5
        self.trading_engine = app.trading_engine
        self.risk_manager = app.risk_manager
        self.adaptive_learning = app.adaptive_learning
        self.reinforcement_agent = app.reinforcement_agent
        self.time_manager = app.time_manager
        self.magic_number = app.magic_number
        self.session_stats = app.session_stats
        self.learning_enabled = app.learning_enabled

    async def trading_loop(self):
        """
        Main trading loop that orchestrates all trading activities.
        This is the core heartbeat of the trading system.
        """
        self.logger.info("Starting main trading loop...")

        # Start fundamental monitor background task
        self.app.fundamental_monitor_task = asyncio.create_task(self.fundamental_monitor_loop())

        loop_count = 0
        last_position_log = 0

        while self.app.running:
            try:
                loop_count += 1

                # Log detailed status every 30 loops (5 minutes)
                if loop_count % 30 == 0:
                    self.logger.info(f"=== TRADING CYCLE #{loop_count} ===")

                # 1. Check for time-based closure FIRST (always check after 22:00)
                await self.check_time_based_closure()

                # 2. Log active positions every 6 loops (60 seconds)
                if loop_count - last_position_log >= 6:
                    await self._log_active_positions()
                    last_position_log = loop_count

                # 3. Check for new trading opportunities
                if loop_count % 30 == 0:  # Log opportunity checking every 5 minutes
                    self.logger.debug("Checking for new trading opportunities...")
                await self._check_trading_opportunities()

                # 4. Monitor existing positions and learning systems
                await self._monitor_positions_and_learning(loop_count)

                # 5. Handle correlation-based actions
                await self._process_correlation_actions()

                # 6. Emergency stop check
                await self._check_emergency_conditions()

                # 7. Learning system maintenance
                await self._maintain_learning_systems(loop_count)

                # Sleep before next iteration
                await asyncio.sleep(10)

            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(30)

    async def _check_trading_opportunities(self):
        """Check for new trading opportunities across all symbols."""
        try:
            symbols = self.config.get('trading', {}).get('symbols', [])
            opportunities_found = 0
            symbols_analyzed = 0

            self.logger.info(f"Analyzing {len(symbols)} trading symbols for opportunities...")

            for symbol in symbols:
                try:
                    symbols_analyzed += 1

                    # Check if we can trade this symbol
                    can_trade, reason = self.risk_manager.can_trade(symbol)
                    if not can_trade:
                        self.logger.debug(f"[{symbol}] Cannot trade: {reason}")
                        continue

                    # Check for existing pending orders - prevent duplicates
                    existing_orders = mt5.orders_get(symbol=symbol)
                    if existing_orders and len(existing_orders) > 0:
                        self.logger.debug(f"[{symbol}] Skipping - already has {len(existing_orders)} pending order(s)")
                        continue

                    # Check for existing positions - prevent multiple positions per symbol
                    existing_positions = mt5.positions_get(symbol=symbol)
                    if existing_positions and len(existing_positions) > 0:
                        self.logger.debug(f"[{symbol}] Skipping - already has {len(existing_positions)} open position(s)")
                        continue

                    # Generate trading signal
                    signal = await self._generate_trading_signal(symbol)
                    if signal:
                        opportunities_found += 1
                        self.logger.info(f"[{symbol}] TRADE SIGNAL: {signal['direction']} "
                                       f"(Strength: {signal['signal_strength']:.3f}) | "
                                       f"Size: {signal['position_size']:.2f} lots | "
                                       f"Entry: {signal['entry_price']:.5f}")

                        # Execute the trade
                        trade_result = await self.trading_engine.execute_trade(signal)
                        if trade_result and trade_result.get('success', False):
                            self.logger.info(f"[{symbol}] OK Trade executed successfully - Ticket: {trade_result.get('ticket', 'N/A')}")
                            
                            # Record trade for daily limit tracking
                            self.risk_manager.record_trade(symbol)
                            
                            # Record open trade in database for monitoring
                            if self.adaptive_learning:
                                self.adaptive_learning.record_open_trade(trade_result)
                            
                            # Start monitoring this trade (skip in dry run mode)
                            dry_run = self.config.get('trading', {}).get('dry_run', False)
                            if not dry_run:
                                asyncio.create_task(self.monitor_trade(
                                    trade_result.get('ticket', 0), trade_result))
                            else:
                                # In dry run mode, just log that we're "monitoring"
                                self.logger.info(f"[{symbol}] Dry run: Simulating monitoring for {trade_result.get('ticket', 'N/A')}")
                        else:
                            error_msg = trade_result.get('error', 'Unknown error') if trade_result else 'Trade execution failed'
                            self.logger.warning(f"[{symbol}] [FAILED] Trade execution failed: {error_msg}")
                    else:
                        # Log why no signal was generated (less frequent logging)
                        if symbols_analyzed % 5 == 0:  # Log every 5th symbol to avoid spam
                            self.logger.debug(f"[{symbol}] No trading signal (insufficient strength)")

                except Exception as e:
                    self.logger.error(f"[{symbol}] Error processing: {e}")

            # Summary logging
            if opportunities_found > 0:
                self.logger.info(f"TRADING SUMMARY: {opportunities_found} opportunities found, {symbols_analyzed} symbols analyzed")
            elif symbols_analyzed % 10 == 0:  # Log summary every 10 cycles when no opportunities
                self.logger.info(f"TRADING SUMMARY: No opportunities found, {symbols_analyzed} symbols analyzed")

        except Exception as e:
            self.logger.error(f"Error checking trading opportunities: {e}")

    async def _generate_trading_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Generate a trading signal for the given symbol.

        Args:
            symbol: Trading symbol to analyze

        Returns:
            Trading signal dictionary or None if no signal
        """
        try:
            # Get market data (H1 bars for technical analysis)
            h1_data = self.app.market_data_manager.get_bars(symbol, mt5.TIMEFRAME_H1, 200)
            if h1_data is None or len(h1_data) < 50:
                return None
            
            # Format data as expected by analyzers
            market_data = {'H1': h1_data}
            
            # Get current price from tick data
            tick_data = self.app.market_data_manager.get_market_data(symbol)
            if tick_data:
                # Use mid price (average of bid/ask) for forex symbols where 'last' might be 0
                bid = tick_data.get('bid', 0)
                ask = tick_data.get('ask', 0)
                current_price = (bid + ask) / 2 if bid > 0 and ask > 0 else tick_data.get('last', 0)
                self.logger.info(f"[{symbol}] Tick data: bid={bid}, ask={ask}, last={tick_data.get('last')}, using price={current_price}")
            else:
                current_price = h1_data['close'].iloc[-1] if len(h1_data) > 0 else 0
                self.logger.info(f"[{symbol}] No tick data, using H1 close: {current_price}")

            # Validate current price
            if current_price <= 0 or current_price is None:
                self.logger.info(f"[{symbol}] Invalid current price: {current_price}, skipping signal")
                return None

            # Get analysis from all analyzers
            technical_score = self.app.technical_analyzer.analyze(symbol, market_data)
            fundamental_score = self.app.fundamental_collector.get_news_sentiment(symbol)['score']
            sentiment_score = self.app.sentiment_analyzer.analyze(symbol, market_data)

            # Get ML prediction if available
            ml_score = 0.0
            ml_confidence = 0.0
            ml_prediction = None

            if self.app.ml_predictor:
                try:
                    # Get technical signals for ML prediction using analyze_symbol
                    technical_signals = self.app.technical_analyzer.analyze_symbol(symbol, market_data)

                    # Make ML prediction
                    ml_prediction = await self.app.ml_predictor.predict(
                        symbol, h1_data, technical_signals, 'H1'
                    )

                    if ml_prediction and 'confidence' in ml_prediction:
                        ml_score = ml_prediction['confidence']
                        ml_confidence = ml_prediction.get('confidence', 0.0)
                        self.logger.info(f"[{symbol}] ML Prediction: score={ml_score:.3f}, confidence={ml_confidence:.3f}")
                    else:
                        self.logger.warning(f"[{symbol}] ML prediction failed or returned invalid result")
                except Exception as e:
                    self.logger.error(f"[{symbol}] Error getting ML prediction: {e}")
                    ml_score = 0.0
                    ml_confidence = 0.0

            # Boost fundamental and sentiment scores during active sessions (similar to check_opportunities.py)
            current_session = self.time_manager.get_current_session()
            if current_session == 'london':
                fundamental_score = min(0.6, fundamental_score + 0.1)
                sentiment_score = min(0.6, sentiment_score + 0.1)
            elif current_session == 'new_york':
                fundamental_score = min(0.65, fundamental_score + 0.15)
                sentiment_score = min(0.65, sentiment_score + 0.15)

            # Combine signals using adaptive learning weights (including ML)
            if self.adaptive_learning and current_session not in ['london', 'new_york']:
                # Use adaptive learning only outside active sessions
                signal_strength = self.adaptive_learning.calculate_signal_strength(
                    technical_score, fundamental_score, sentiment_score, ml_score)
            else:
                # Use improved weighting with boosted scores during active sessions
                # Include ML score if confidence is high enough
                if ml_confidence > 0.6:
                    signal_strength = (technical_score * 0.4 + fundamental_score * 0.2 +
                                     sentiment_score * 0.15 + ml_score * 0.25)
                else:
                    signal_strength = (technical_score * 0.6 + fundamental_score * 0.25 + sentiment_score * 0.15)

            self.logger.info(f"[{symbol}] Scores - Tech: {technical_score:.3f}, Fund: {fundamental_score:.3f}, Sent: {sentiment_score:.3f}, ML: {ml_score:.3f} ({ml_confidence:.2f}), Combined: {signal_strength:.3f}")

            # Get session-aware minimum signal strength
            min_strength = self.time_manager.get_session_signal_threshold(self.config)

            # Get session-aware minimum signal strength
            min_strength = self.time_manager.get_session_signal_threshold(self.config)

            # Check session filtering (if enabled)
            session_config = self.config.get('trading_rules', {}).get('session_filter', {})
            if session_config.get('enabled', False):
                if not self.time_manager.is_preferred_session(self.config):
                    # Not in preferred session - require higher threshold
                    preferred_sessions = session_config.get('preferred_sessions', [])
                    current_session = self.time_manager.get_current_session()
                    self.logger.info(f"[{symbol}] Not in preferred session ({current_session}), skipping. Preferred: {preferred_sessions}")
                    return None

                # Check if current hour is optimal for trading
                if not self.time_manager.is_optimal_trading_hour(self.config):
                    current_hour = self.app.get_current_mt5_time().hour
                    hourly_weight = self.time_manager.get_hourly_performance_weight(self.config)
                    self.logger.info(f"[{symbol}] Sub-optimal trading hour ({current_hour:02d}:00, weight: {hourly_weight:.3f}), skipping")
                    return None

            # Check minimum signal strength (session-aware)
            if signal_strength < min_strength:
                current_session = self.time_manager.get_current_session()
                self.logger.info(f"[{symbol}] Signal strength {signal_strength:.3f} below threshold {min_strength:.3f} for {current_session} session")
                return None

            # Determine trade direction
            direction = 'BUY' if technical_score > 0.5 else 'SELL'

            # Calculate position size
            default_sl_pips = self.config.get('trading', {}).get('default_sl_pips', 20)
            position_size = self.risk_manager.calculate_position_size(symbol, default_sl_pips)

            # Get stop loss and take profit levels
            sl_tp = self.risk_manager.calculate_stop_loss_take_profit(
                symbol, current_price, direction)

            # Calculate risk multiplier based on fundamental analysis
            risk_multiplier = 1.0
            if fundamental_score < 0.4:
                risk_multiplier = 1.3  # High risk - increase stops
            elif fundamental_score < 0.6:
                risk_multiplier = 1.1  # Moderate risk

            # Check for high impact events
            if hasattr(self.app, 'fundamental_analyzer') and self.app.fundamental_analyzer:
                if self.app.fundamental_analyzer.should_avoid_trading(symbol):
                    risk_multiplier *= 1.5  # Avoid trading but if we do, very wide stops

            return {
                'symbol': symbol,
                'direction': direction,
                'entry_price': current_price,
                'position_size': position_size,
                'stop_loss': sl_tp['stop_loss'],
                'take_profit': sl_tp['take_profit'],
                'technical_score': technical_score,
                'fundamental_score': fundamental_score,
                'sentiment_score': sentiment_score,
                'ml_score': ml_score,
                'signal_strength': signal_strength,
                'risk_multiplier': risk_multiplier
            }

        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    async def _monitor_positions_and_learning(self, loop_count: int):
        """Monitor existing positions and maintain learning systems."""
        try:
            # Manage pending orders every 60 loops (10 minutes) - less aggressive
            if loop_count % 60 == 0:
                await self._manage_pending_orders()

            # Check learning thread health every hour
            if self.adaptive_learning and loop_count % (360 * 6) == 0:
                await self._check_learning_thread_health()

        except Exception as e:
            self.logger.error(f"Error in position and learning monitoring: {e}")

    async def _manage_pending_orders(self):
        """Manage pending stop orders - cancel stale/invalid orders."""
        try:
            management_result = self.trading_engine.order_executor.order_manager.manage_pending_orders()

            managed = management_result.get('managed', 0)
            cancelled = management_result.get('cancelled', 0)
            errors = management_result.get('errors', 0)

            if cancelled > 0 or errors > 0:
                self.logger.info(f"Pending order management: {managed} checked, {cancelled} cancelled, {errors} errors")
            elif managed > 0 and managed % 10 == 0:  # Log every 10th check when no action taken
                self.logger.debug(f"Pending order management: {managed} orders monitored")

        except Exception as e:
            self.logger.error(f"Error managing pending orders: {e}")

    async def _process_correlation_actions(self):
        """Process correlation-based trading actions."""
        try:
            if hasattr(self.app, 'correlation_manager') and self.app.correlation_manager:
                correlation_actions = self.app.correlation_manager.get_pending_actions()

                for symbol, action in correlation_actions.items():
                    await self._handle_correlation_action(symbol, action)

        except Exception as e:
            self.logger.error(f"Error processing correlation actions: {e}")

    async def _check_emergency_conditions(self):
        """Check for emergency stop conditions."""
        try:
            # Check drawdown limits
            if self.risk_manager.check_emergency_stop():
                self.logger.critical("Emergency stop triggered - shutting down")
                self.running = False
                return

        except Exception as e:
            self.logger.error(f"Error checking emergency conditions: {e}")

    async def _maintain_learning_systems(self, loop_count: int):
        """Maintain and update learning systems."""
        try:
            # Auto-learn from previous day logs (every 6 hours)
            if self.adaptive_learning and loop_count % (360 * 6) == 0:
                try:
                    self.adaptive_learning.auto_learn_from_previous_day_logs()
                except Exception as e:
                    self.logger.error(f"Error in auto log learning: {e}")

        except Exception as e:
            self.logger.error(f"Error maintaining learning systems: {e}")

    async def _check_learning_thread_health(self):
        """Check the health of the learning thread."""
        try:
            thread_status = self.adaptive_learning.get_thread_status()
            self.logger.info(f"Learning thread status: {thread_status}")

            if not thread_status.get('thread_alive', False):
                self.logger.warning("Learning thread is not alive - attempting restart")
                self.adaptive_learning.restart_learning_thread()
                # Check again after restart
                await asyncio.sleep(0.1)
                new_status = self.adaptive_learning.get_thread_status()
                if new_status.get('thread_alive', False):
                    self.logger.info("Learning thread successfully restarted")
                else:
                    self.logger.error("Failed to restart learning thread")
        except Exception as e:
            self.logger.error(f"Error checking learning thread health: {e}")

    async def monitor_trade(self, ticket: int, trade_data: dict):
        """
        Monitor trade outcome for learning with time-based exits.

        Args:
            ticket: MT5 ticket number for the trade
            trade_data: Trade execution data
        """
        try:
            # Get symbol-specific optimal holding times
            symbol = trade_data['symbol']
            if self.adaptive_learning:
                symbol_params = self.adaptive_learning.get_symbol_optimal_holding_time(symbol)
                optimal_holding_hours = symbol_params['optimal_holding_hours']
                max_holding_minutes = symbol_params['max_holding_minutes']
                confidence_score = symbol_params['confidence_score']
            else:
                # Fallback to global parameters
                adaptive_params = self.adaptive_learning.get_adaptive_parameters() if self.adaptive_learning else {}
                optimal_holding_hours = adaptive_params.get('optimal_holding_hours', 4.0)
                max_holding_minutes = adaptive_params.get('max_holding_minutes', 480)
                confidence_score = 0.0

            optimal_holding_minutes = optimal_holding_hours * 60

            # Log symbol-specific parameters
            self.logger.info(
                f"Monitoring {symbol} with optimal holding: "
                f"{optimal_holding_hours:.1f}h "
                f"(max: {max_holding_minutes}min, confidence: "
                f"{confidence_score:.2f})")

            # Wait for trade to complete
            while True:
                await asyncio.sleep(30)  # Check every 30 seconds

                position = self.trading_engine.get_position_by_ticket(ticket)

                if position is None:  # Trade closed
                    await self._process_closed_trade(ticket, trade_data)
                    break
                else:
                    # Position still open - check time-based exit conditions
                    await self._check_open_position_exits(ticket, trade_data, position,
                                                        optimal_holding_minutes, max_holding_minutes)

        except Exception as e:
            self.logger.error(f"Error monitoring trade {ticket}: {e}")

    async def _process_closed_trade(self, ticket: int, trade_data: dict):
        """Process a closed trade and update learning systems."""
        try:
            # Get trade history
            history = self.trading_engine.get_trade_history(ticket)

            if history:
                # Calculate profit metrics
                exit_price = history['exit_price']
                entry_price = trade_data['entry_price']

                if trade_data['direction'] == 'BUY':
                    profit_pips = (exit_price - entry_price) * 10000
                else:
                    profit_pips = (entry_price - exit_price) * 10000

                # Simplified calculation
                profit_pct = (profit_pips / 100)

                # Update trade data
                trade_data['exit_price'] = exit_price
                trade_data['profit'] = history.get('profit', 0)
                trade_data['profit_pct'] = profit_pct
                trade_data['duration_minutes'] = (
                    self.app.get_current_mt5_time() - trade_data['timestamp']).seconds // 60
                trade_data['volume'] = history.get('volume', 0)

                # Add session and day-of-week information for learning
                trade_data['session'] = self.time_manager.get_current_session()
                trade_timestamp = trade_data.get('timestamp', self.app.get_current_mt5_time())
                trade_data['day_of_week'] = trade_timestamp.strftime('%A').lower()
                trade_data['hour_of_day'] = trade_timestamp.hour

                # Record trade result for cooldown management
                actual_profit = history.get('profit', 0)
                self.risk_manager.record_trade_result(trade_data['symbol'], actual_profit)

                # Record for learning
                if self.adaptive_learning:
                    self.adaptive_learning.record_trade(trade_data)

                # Process analyzer evaluation
                await self._process_analyzer_evaluation(trade_data, profit_pct)

                # Update reinforcement learning
                await self._update_reinforcement_learning(ticket, trade_data, exit_price, history)

                # Update session stats
                if profit_pct > 0:
                    self.session_stats['winning_trades'] += 1
                else:
                    self.session_stats['losing_trades'] += 1

                self.session_stats['total_profit'] += profit_pct

                # Log comprehensive trade outcome
                closure_reason = trade_data.get('closure_reason', 'natural_exit')
                self._log_trade_outcome(trade_data, history, closure_reason)

        except Exception as e:
            self.logger.error(f"Error processing closed trade {ticket}: {e}")

    async def _process_analyzer_evaluation(self, trade_data: dict, profit_pct: float):
        """Process analyzer accuracy evaluation for reinforcement learning."""
        try:
            if self.reinforcement_agent and hasattr(self.reinforcement_agent, 'record_trade_with_analyzer_evaluation'):
                # Prepare trade outcome data for analyzer evaluation
                trade_outcome = {
                    'symbol': trade_data['symbol'],
                    'profit_pct': profit_pct,
                    'entry_signals': {
                        'technical_score': trade_data.get('technical_score', 0.5),
                        'fundamental_score': 0.5,  # Placeholder - could be enhanced
                        'sentiment_score': trade_data.get('sentiment_score', 0.5),
                        'signal_strength': trade_data.get('signal_strength', 0.5)
                    }
                }

                # Record trade with analyzer evaluation
                self.reinforcement_agent.record_trade_with_analyzer_evaluation(
                    trade_outcome, self.adaptive_learning)

                self.logger.debug(f"Analyzer evaluation completed for {trade_data['symbol']}")

        except Exception as e:
            self.logger.warning(f"Failed to evaluate analyzer accuracy: {e}")

    async def _update_reinforcement_learning(self, ticket: int, trade_data: dict,
                                           exit_price: float, history: dict):
        """Update reinforcement learning model with trade experience."""
        try:
            if self.reinforcement_agent and self.reinforcement_agent.enabled:
                # Get the stored experience for this ticket
                experience = self.reinforcement_agent.pending_experiences.get(ticket)
                if experience is None:
                    self.logger.debug(f"No RL experience found for ticket {ticket}")
                    return

                # Determine closure reason from trade data
                closure_reason = trade_data.get('closure_reason', 'natural_exit')

                reward = self.reinforcement_agent.calculate_reward(
                    experience['entry_price'],
                    exit_price,
                    trade_data['direction'],
                    trade_data['duration_minutes'],
                    closure_reason=closure_reason
                )

                # Create next state (position closed)
                next_state = experience['initial_state'].copy()
                next_state['position_status'] = 0  # Position closed
                next_state['closure_reason'] = closure_reason  # Add closure reason to state

                # Learn from this experience
                self.reinforcement_agent.update_q_table(
                    experience['initial_state'],
                    experience['action'],
                    reward,
                    next_state
                )

                # Save RL model periodically (every 10 trades)
                self.session_stats['rl_models_saved'] += 1
                if self.session_stats['rl_models_saved'] % 10 == 0:
                    try:
                        self.reinforcement_agent.save_model()
                        self.logger.debug(f"RL model saved after {self.session_stats['rl_models_saved']} trades")
                    except Exception as e:
                        self.logger.error(f"Error saving RL model after trade: {e}")

                # Remove from pending experiences
                del self.reinforcement_agent.pending_experiences[ticket]

                self.logger.debug(f"RL updated: action={experience['action']}, reward={reward:.4f}, reason={closure_reason}")

        except Exception as e:
            self.logger.warning(f"Failed to update RL agent: {e}")

    async def _check_open_position_exits(self, ticket: int, trade_data: dict, position,
                                       optimal_holding_minutes: float, max_holding_minutes: float):
        """Check exit conditions for open positions."""
        try:
            holding_minutes = (
                self.app.get_current_mt5_time() - trade_data['timestamp']).seconds // 60

            # Check for immediate closure after 22:00 MT5 time ONLY if not in active session
            current_time = self.time_manager.get_current_time()
            current_time_only = current_time.time()
            immediate_close_time = self.time_manager.MT5_IMMEDIATE_CLOSE_TIME

            # Only close if after immediate close time AND not in an active trading session
            if current_time_only >= immediate_close_time:
                current_session = self.time_manager.get_current_session()
                # Don't close if we're in Sydney or Tokyo sessions (active after 22:00)
                if current_session not in ['sydney', 'tokyo']:
                    await self._close_position_immediately(ticket, trade_data, immediate_close_time)
                    return

            # No other time-based closures - only SL/TP or manual closure

        except Exception as e:
            self.logger.error(f"Error checking open position exits for ticket {ticket}: {e}")

    async def _close_position_immediately(self, ticket: int, trade_data: dict, close_time):
        """Close position immediately after market close time."""
        try:
            self.logger.info(
                f"Immediately closing {trade_data['symbol']} position - after {close_time.strftime('%H:%M')} MT5 time")
            
            # Set closure reason in trade_data for database recording
            trade_data['closure_reason'] = f"immediate_close_after_{close_time.strftime('%H:%M')}"
            
            close_result = await self.trading_engine.close_position_by_ticket(ticket)
            if close_result:
                self.logger.info("Successfully closed position for immediate time-based exit")
                # Store closure reason in pending experience for RL learning
                if hasattr(self.reinforcement_agent, 'pending_experiences') and ticket in self.reinforcement_agent.pending_experiences:
                    self.reinforcement_agent.pending_experiences[ticket]['closure_reason'] = f"immediate_close_after_{close_time.strftime('%H:%M')}"
            else:
                self.logger.warning("Failed to close position for immediate time-based exit")
        except Exception as e:
            self.logger.error(f"Error closing position for immediate time-based exit: {e}")

    async def _close_position_time_based(self, ticket: int, trade_data: dict, close_reason: str):
        """Close position based on time-based exit conditions."""
        try:
            self.logger.info(
                f"Closing {trade_data['symbol']} position due to time-based exit: {close_reason}")
            close_result = await self.trading_engine.close_position_by_ticket(ticket)
            if close_result:
                self.logger.info(f"Successfully closed position for time-based exit: {close_reason}")
                # Store closure reason in pending experience for RL learning
                if hasattr(self.reinforcement_agent, 'pending_experiences') and ticket in self.reinforcement_agent.pending_experiences:
                    self.reinforcement_agent.pending_experiences[ticket]['closure_reason'] = close_reason
            else:
                self.logger.warning(f"Failed to close position for time-based exit: {close_reason}")
        except Exception as e:
            self.logger.error(f"Error closing position for time-based exit: {e}")

    async def _close_position_max_time(self, ticket: int, trade_data: dict, max_holding_minutes: float):
        """Close position after maximum holding time."""
        try:
            self.logger.info(
                f"Closing {trade_data['symbol']} position due to "
                f"max holding time ({max_holding_minutes} minutes)")
            close_result = await self.trading_engine.close_position_by_ticket(ticket)
            if close_result:
                self.logger.info("Successfully closed position for max time exit")
            else:
                self.logger.warning("Failed to close position for max time exit")
        except Exception as e:
            self.logger.error(f"Error closing position for max time: {e}")

    async def _check_optimal_time_exit(self, ticket: int, trade_data: dict, position, optimal_holding_minutes: float):
        """Check if position should be closed at optimal holding time with profit."""
        try:
            # Check if position is in profit
            current_price = position.get('price_current', 0)
            entry_price = trade_data['entry_price']

            profit_pct = 0.0
            if trade_data['direction'] == 'BUY' and current_price > entry_price:
                profit_pct = ((current_price - entry_price) / entry_price) * 100
            elif trade_data['direction'] == 'SELL' and current_price < entry_price:
                profit_pct = ((entry_price - current_price) / entry_price) * 100

            if profit_pct > 0.1:  # At least 0.1% profit
                self.logger.info(
                    f"Closing {trade_data['symbol']} position at "
                    f"optimal time ({optimal_holding_minutes} min) "
                    f"with {profit_pct:.2f}% profit")
                close_result = await self.trading_engine.close_position_by_ticket(ticket)
                if close_result:
                    self.logger.info("Successfully closed position for optimal time exit")
                else:
                    self.logger.warning("Failed to close position for optimal time exit")
        except Exception as e:
            self.logger.error(f"Error closing position for optimal time: {e}")

    async def _handle_correlation_action(self, symbol: str, correlation_action: Dict):
        """
        Handle correlation-based trading actions

        Args:
            symbol: Base symbol
            correlation_action: Action recommended by correlation manager
        """
        try:
            action_type = correlation_action.get('action')

            if action_type == 'exit_recommended':
                await self._handle_correlation_exit(symbol, correlation_action)
            elif action_type == 'entry_recommended':
                await self._handle_correlation_entry(symbol, correlation_action)
            elif action_type == 'exit_consideration':
                await self._handle_correlation_consideration(symbol, correlation_action)

        except Exception as e:
            self.logger.error(f"Error handling correlation action for {symbol}: {e}")

    async def _handle_correlation_exit(self, symbol: str, correlation_action: Dict):
        """Handle correlation-based exit recommendations."""
        confidence = correlation_action.get('confidence', 0.5)
        correlation = correlation_action.get('correlation', 0)
        correlated_symbol = correlation_action.get('correlated_symbol', '')

        if confidence > 0.7:  # High confidence
            self.logger.info(f"High-confidence correlation exit: Closing {symbol} due to {correlation:.2f} correlation with {correlated_symbol}")
            # Get position and close it
            positions = mt5.positions_get() if self.mt5 else []
            for position in positions:
                if hasattr(position, 'magic') and position.magic == self.magic_number and position.symbol == symbol:
                    close_result = await self.trading_engine.close_position(position)
                    if close_result:
                        self.logger.info(f"Successfully closed {symbol} based on correlation analysis")
                    break
        else:
            self.logger.info(f"Correlation exit suggested for {symbol} (confidence: {confidence:.2f}) - monitoring continues")

    async def _handle_correlation_entry(self, symbol: str, correlation_action: Dict):
        """Handle correlation-based entry recommendations."""
        confidence = correlation_action.get('confidence', 0.5)
        new_symbol = correlation_action.get('symbol', '')
        base_symbol = correlation_action.get('base_symbol', '')

        if confidence > 0.7 and new_symbol:  # High confidence
            self.logger.info(f"High-confidence correlation entry: Opening {new_symbol} based on low correlation with {base_symbol}")

            # Check if we can trade this symbol
            if self.risk_manager and self.risk_manager.can_trade(new_symbol)[0]:
                # Generate a small position for the correlated pair
                signal = {
                    'symbol': new_symbol,
                    'action': 'BUY',  # Default to buy, could be enhanced
                    'entry_price': None,  # Will be set by trading engine
                    'position_size': 0.01,  # Small position
                    'stop_loss': None,
                    'take_profit': None
                }

                # Execute the trade
                trade_result = await self.trading_engine.execute_trade(signal)
                if trade_result:
                    self.logger.info(f"Successfully opened {new_symbol} based on correlation analysis")
                else:
                    self.logger.warning(f"Failed to open {new_symbol} position")
            else:
                self.logger.info(f"Cannot open {new_symbol} - risk limits exceeded")
        else:
            self.logger.info(f"Correlation entry suggested for {new_symbol} (confidence: {confidence:.2f}) - monitoring continues")

    async def _handle_correlation_consideration(self, symbol: str, correlation_action: Dict):
        """Handle correlation considerations for monitoring."""
        correlation = correlation_action.get('correlation', 0)
        correlated_symbol = correlation_action.get('correlated_symbol', '')
        self.logger.info(f"Correlation monitoring: {symbol} correlation with {correlated_symbol} is now {correlation:.2f}")

    async def check_time_based_closure(self):
        """Check if positions should be closed based on time - always check after 22:00"""
        try:
            # Use TimeManager for consistent time handling
            should_close, reason = self.time_manager.should_close_positions()

            if should_close:
                self.logger.info(f"Time-based closure triggered - closing all positions: {reason}")
                self.app._last_closure_date = self.time_manager._last_closure_date  # Sync with TimeManager

                # Safe async handling for close_all_positions
                if self.trading_engine:
                    if hasattr(self.trading_engine, 'close_all_positions'):
                        try:
                            close_method = self.trading_engine.close_all_positions
                            if asyncio.iscoroutinefunction(close_method):
                                await close_method()
                            else:
                                loop = asyncio.get_event_loop()
                                await loop.run_in_executor(None, close_method)
                        except Exception as e:
                            self.logger.error(f"Error closing positions: {e}")
                    else:
                        self.logger.warning("close_all_positions method not found on trading_engine")
                else:
                    self.logger.warning("Trading engine not initialized yet")
            
            # ADDITIONAL CHECK: Close orphaned positions that missed their closure time
            await self._check_orphaned_positions()

        except Exception as e:
            self.logger.error(f"Error in time-based closure: {e}")

    async def _check_orphaned_positions(self):
        """Check for positions that should have been closed but were missed (e.g., due to system restart)"""
        try:
            # Get current MT5 time
            current_time = self.time_manager.get_current_time()
            current_time_only = current_time.time()
            closure_time = self.time_manager.MT5_IMMEDIATE_CLOSE_TIME

            # Only check if we're past closure time
            if current_time_only < closure_time:
                return

            # Get all positions
            positions = mt5.positions_get()
            if positions is None:
                positions = []

            orphaned_positions = []
            for position in positions:
                if hasattr(position, 'magic') and position.magic == self.magic_number:
                    # Check if position was opened before closure time
                    open_time = datetime.fromtimestamp(position.time)
                    if open_time.time() < closure_time:
                        orphaned_positions.append(position)

            if orphaned_positions:
                self.logger.warning(f"Found {len(orphaned_positions)} orphaned positions that missed 22:00 closure")
                
                for pos in orphaned_positions:
                    self.logger.info(f"Force closing orphaned position: {pos.symbol} ticket {pos.ticket}")
                    close_result = await self.trading_engine.close_position_by_ticket(pos.ticket)
                    if close_result:
                        self.logger.info(f"Successfully closed orphaned position {pos.symbol} ticket {pos.ticket}")
                    else:
                        self.logger.error(f"Failed to close orphaned position {pos.symbol} ticket {pos.ticket}")

        except Exception as e:
            self.logger.error(f"Error checking orphaned positions: {e}")

    async def _log_active_positions(self):
        """Log all currently active positions with detailed metrics"""
        try:
            # Get all positions from MT5
            positions = mt5.positions_get()
            if positions is None:
                positions = []

            active_fxai_positions = []
            for position in positions:
                if hasattr(position, 'magic') and position.magic == self.magic_number:
                    active_fxai_positions.append(position)

            if active_fxai_positions:
                total_unrealized_pnl = 0.0
                total_risk = 0.0

                self.logger.info(f"ACTIVE POSITIONS ({len(active_fxai_positions)}):")
                for pos in active_fxai_positions:
                    direction = "LONG" if pos.type == mt5.ORDER_TYPE_BUY else "SHORT"
                    commission = getattr(pos, 'commission', 0.0)
                    swap = getattr(pos, 'swap', 0.0)
                    pnl = pos.profit + swap + commission
                    total_unrealized_pnl += pnl

                    # Calculate duration
                    duration_hours = 0
                    if hasattr(pos, 'time') and pos.time > 0:
                        duration_hours = (self.app.get_current_mt5_time().timestamp() - pos.time) / 3600

                    # Calculate P&L percentage
                    pnl_pct = 0.0
                    if pos.price_open > 0:
                        if direction == "LONG":
                            pnl_pct = ((pos.price_current - pos.price_open) / pos.price_open) * 100
                        else:
                            pnl_pct = ((pos.price_open - pos.price_current) / pos.price_open) * 100

                    # Calculate risk metrics
                    risk_amount = 0.0
                    if pos.sl > 0:
                        pip_size = self._get_pip_size(pos.symbol)
                        if direction == "LONG":
                            risk_pips = (pos.price_open - pos.sl) / pip_size
                        else:
                            risk_pips = (pos.sl - pos.price_open) / pip_size
                        risk_amount = risk_pips * pos.volume * 10  # Approximate dollar risk
                        total_risk += risk_amount

                    sl_display = f"{pos.sl:.5f}" if pos.sl > 0 else "None"
                    tp_display = f"{pos.tp:.5f}" if pos.tp > 0 else "None"

                    self.logger.info(
                        f"  {pos.symbol} {direction} | "
                        f"Size: {pos.volume:.2f} lots | "
                        f"Entry: {pos.price_open:.5f} | "
                        f"Current: {pos.price_current:.5f} | "
                        f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) | "
                        f"Duration: {duration_hours:.1f}h | "
                        f"Risk: ${risk_amount:.2f} | "
                        f"SL: {sl_display} | "
                        f"TP: {tp_display}"
                    )

                # Summary statistics
                avg_pnl_pct = (total_unrealized_pnl / len(active_fxai_positions)) if active_fxai_positions else 0
                self.logger.info(f"PORTFOLIO SUMMARY: Total P&L: ${total_unrealized_pnl:.2f} | "
                               f"Avg P&L: ${avg_pnl_pct:.2f} | Total Risk: ${total_risk:.2f}")

            else:
                self.logger.info("ACTIVE POSITIONS: None")

            # Add system health and market information
            await self._log_system_health()

        except Exception as e:
            self.logger.error(f"Error logging active positions: {e}")

    async def _log_system_health(self):
        """Log comprehensive system health and market information"""
        try:
            # Session statistics
            session_duration = self.app.get_current_mt5_time() - self.app.session_stats['start_time']
            hours_running = session_duration.total_seconds() / 3600

            self.logger.info(f"SESSION: {hours_running:.1f}h | Trades: {self.app.session_stats['total_trades']} | "
                           f"P&L: ${self.app.session_stats['total_profit']:.2f}")

            # Risk manager status
            if hasattr(self.risk_manager, 'daily_trade_count'):
                daily_trades = self.risk_manager.daily_trade_count
                max_daily = self.config.get('risk', {}).get('max_daily_trades', 10)
                self.logger.info(f"RISK: Daily trades {daily_trades}/{max_daily} | "
                               f"Drawdown: ${self.risk_manager.current_drawdown:.2f}")

            # ML/AI status
            ml_status = []
            if self.adaptive_learning and hasattr(self.adaptive_learning, 'is_learning'):
                ml_status.append("Adaptive Learning: Active" if self.adaptive_learning.is_learning else "Adaptive Learning: Idle")

            if self.reinforcement_agent and hasattr(self.reinforcement_agent, 'is_training'):
                ml_status.append("RL Agent: Training" if self.reinforcement_agent.is_training else "RL Agent: Ready")

            if ml_status:
                self.logger.info(f"AI/ML: {' | '.join(ml_status)}")

            # Market regime detection
            if hasattr(self.app, 'market_regime_detector') and self.app.market_regime_detector:
                try:
                    regime = self.app.market_regime_detector.get_current_regime()
                    self.logger.info(f"MARKET REGIME: {regime}")
                except:
                    pass

            # Fundamental monitor status
            if hasattr(self.app, 'fundamental_collector'):
                try:
                    news_count = len(self.app.fundamental_collector.recent_news) if hasattr(self.app.fundamental_collector, 'recent_news') else 0
                    self.logger.info(f"FUNDAMENTAL: {news_count} recent news items monitored")
                except:
                    pass

            # System health indicators
            health_indicators = []
            if self.mt5 and mt5.terminal_info():
                health_indicators.append("MT5: Connected")
            else:
                health_indicators.append("MT5: Disconnected")

            if self.app.running:
                health_indicators.append("System: Running")
            else:
                health_indicators.append("System: Stopped")

            # Check pending orders health
            pending_health = self.trading_engine.order_executor.check_pending_orders_health()
            total_pending = pending_health.get('total_pending', 0)
            issues = pending_health.get('issues', [])

            health_indicators.append(f"Pending Orders: {total_pending}")

            if issues:
                health_indicators.append(f"Issues: {len(issues)}")
                # Log issues separately for visibility
                for issue in issues:
                    self.logger.warning(f"PENDING ORDER ISSUE: {issue}")

            self.logger.info(f"SYSTEM HEALTH: {' | '.join(health_indicators)}")

        except Exception as e:
            self.logger.debug(f"Error logging system health: {e}")

    def _get_pip_size(self, symbol: str) -> float:
        """Get pip size for a symbol"""
        if 'JPY' in symbol:
            return 0.01
        elif 'XAU' in symbol or 'XAG' in symbol:
            return 0.01 if 'XAG' in symbol else 0.1
        else:
            return 0.0001

    def _log_trade_outcome(self, trade_data: dict, history: dict, closure_reason: str = "natural_exit"):
        """Log comprehensive trade outcome with detailed metrics and analysis"""
        try:
            symbol = trade_data['symbol']
            direction = trade_data['direction']
            entry_price = trade_data['entry_price']
            exit_price = trade_data.get('exit_price', history.get('exit_price', 0))
            volume = trade_data.get('volume', history.get('volume', 0))
            duration_minutes = trade_data.get('duration_minutes', 0)
            profit = trade_data.get('profit', history.get('profit', 0))
            profit_pct = trade_data.get('profit_pct', 0)

            # Calculate pip movement
            pip_size = self._get_pip_size(symbol)
            if direction == 'BUY':
                pip_movement = (exit_price - entry_price) / pip_size
            else:
                pip_movement = (entry_price - exit_price) / pip_size

            # Calculate risk metrics
            risk_amount = 0.0
            risk_pct = 0.0
            rr_ratio = 0.0
            if 'sl' in trade_data and trade_data['sl'] > 0:
                if direction == 'BUY':
                    risk_pips = (entry_price - trade_data['sl']) / pip_size
                else:
                    risk_pips = (trade_data['sl'] - entry_price) / pip_size
                risk_amount = abs(risk_pips) * volume * 10  # Approximate dollar risk
                if risk_amount > 0:
                    rr_ratio = abs(profit) / risk_amount

            # Determine trade outcome
            is_win = profit > 0
            outcome = "WIN" if is_win else "LOSS"

            # Calculate performance metrics
            duration_hours = duration_minutes / 60
            profit_per_hour = profit / duration_hours if duration_hours > 0 else 0

            # Log comprehensive trade outcome
            self.logger.info("=" * 80)
            self.logger.info(f"TRADE COMPLETED - {outcome}")
            self.logger.info("=" * 80)

            # Basic trade information
            self.logger.info(f"Symbol: {symbol} | Direction: {direction} | Volume: {volume:.2f} lots")
            self.logger.info(f"Entry Price: {entry_price:.5f} | Exit Price: {exit_price:.5f}")
            self.logger.info(f"Duration: {duration_minutes} minutes ({duration_hours:.1f} hours)")

            # Financial results
            self.logger.info(f"P&L: ${profit:.2f} ({profit_pct:+.2f}%) | Pips: {pip_movement:+.1f}")
            self.logger.info(f"Profit/Hour: ${profit_per_hour:.2f}")

            # Risk metrics
            if risk_amount > 0:
                self.logger.info(f"Risk Amount: ${risk_amount:.2f} | Risk:Reward Ratio: {rr_ratio:.2f}")
            else:
                self.logger.info("Risk Amount: Not set | Risk:Reward Ratio: N/A")

            # Exit analysis
            self.logger.info(f"Exit Reason: {closure_reason}")

            # Signal analysis (if available)
            if 'technical_score' in trade_data:
                self.logger.info(f"Entry Signals - Technical: {trade_data['technical_score']:.2f} | "
                               f"Sentiment: {trade_data.get('sentiment_score', 0.5):.2f} | "
                               f"Signal Strength: {trade_data.get('signal_strength', 0.5):.2f}")

            # Performance analysis
            if is_win:
                self.logger.info("PERFORMANCE: Excellent trade execution!"                if rr_ratio >= 2.0 else
                               "PERFORMANCE: Good trade, consider optimizing risk management"                if rr_ratio >= 1.0 else
                               "PERFORMANCE: Profitable but high risk, review entry criteria")
            else:
                self.logger.info("PERFORMANCE: Loss - analyze entry signals and market conditions"                if abs(rr_ratio) <= 1.0 else
                               "PERFORMANCE: Large loss - review risk management and stop loss placement")

            # Market regime context (if available)
            if hasattr(self.app, 'market_regime_detector') and self.app.market_regime_detector:
                try:
                    regime = self.app.market_regime_detector.get_current_regime(symbol)
                    if regime:
                        self.logger.info(f"Market Regime: {regime['regime']} (Volatility: {regime['volatility']:.2f})")
                except Exception as e:
                    self.logger.debug(f"Could not get market regime: {e}")

            # Adaptive learning insights (if available)
            if self.adaptive_learning:
                try:
                    insights = self.adaptive_learning.get_trade_insights(symbol, is_win)
                    if insights:
                        self.logger.info(f"Learning Insights: {insights}")
                except Exception as e:
                    self.logger.debug(f"Could not get learning insights: {e}")

            self.logger.info("=" * 80)

            # Update session performance tracking
            self._update_performance_metrics(trade_data, is_win, profit, duration_minutes)

        except Exception as e:
            self.logger.error(f"Error logging trade outcome: {e}")

    def _update_performance_metrics(self, trade_data: dict, is_win: bool, profit: float, duration_minutes: int):
        """Update session performance metrics for analysis"""
        try:
            symbol = trade_data['symbol']

            # Initialize symbol metrics if not exists
            if symbol not in self.session_stats['symbol_performance']:
                self.session_stats['symbol_performance'][symbol] = {
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_profit': 0.0,
                    'avg_duration': 0.0,
                    'best_trade': float('-inf'),
                    'worst_trade': float('inf')
                }

            symbol_stats = self.session_stats['symbol_performance'][symbol]
            symbol_stats['total_trades'] += 1
            symbol_stats['wins' if is_win else 'losses'] += 1
            symbol_stats['total_profit'] += profit
            symbol_stats['best_trade'] = max(symbol_stats['best_trade'], profit)
            symbol_stats['worst_trade'] = min(symbol_stats['worst_trade'], profit)

            # Update average duration
            total_duration = symbol_stats['avg_duration'] * (symbol_stats['total_trades'] - 1) + duration_minutes
            symbol_stats['avg_duration'] = total_duration / symbol_stats['total_trades']

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    def _log_performance_summary(self):
        """Log comprehensive performance summary with key metrics"""
        try:
            total_trades = self.session_stats['winning_trades'] + self.session_stats['losing_trades']

            if total_trades == 0:
                self.logger.info("PERFORMANCE SUMMARY: No trades completed yet")
                return

            win_rate = self.session_stats['winning_trades'] / total_trades
            avg_profit = self.session_stats['total_profit'] / total_trades
            total_profit = self.session_stats['total_profit']

            # Calculate additional metrics
            profit_factor = 0.0
            if self.session_stats['losing_trades'] > 0:
                avg_win = self.session_stats.get('avg_win', 0)
                avg_loss = self.session_stats.get('avg_loss', 0)
                if avg_loss != 0:
                    profit_factor = abs(avg_win / avg_loss) if avg_win > 0 else 0

            # Calculate drawdown metrics (simplified)
            max_drawdown = self.session_stats.get('max_drawdown', 0)

            self.logger.info("=" * 80)
            self.logger.info("PERFORMANCE SUMMARY")
            self.logger.info("=" * 80)

            # Overall statistics
            self.logger.info(f"Total Trades: {total_trades}")
            self.logger.info(f"Win Rate: {win_rate:.1%} ({self.session_stats['winning_trades']}W / {self.session_stats['losing_trades']}L)")
            self.logger.info(f"Total P&L: ${total_profit:.2f}")
            self.logger.info(f"Average P&L per Trade: ${avg_profit:.2f}")

            # Advanced metrics
            if profit_factor > 0:
                self.logger.info(f"Profit Factor: {profit_factor:.2f}")
            self.logger.info(f"Max Drawdown: ${max_drawdown:.2f}")

            # Symbol performance
            if self.session_stats['symbol_performance']:
                self.logger.info("SYMBOL PERFORMANCE:")
                for symbol, stats in self.session_stats['symbol_performance'].items():
                    symbol_win_rate = stats['wins'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
                    self.logger.info(f"  {symbol}: {stats['total_trades']} trades | "
                                   f"Win Rate: {symbol_win_rate:.1%} | "
                                   f"P&L: ${stats['total_profit']:.2f} | "
                                   f"Avg Duration: {stats['avg_duration']:.0f}min")

            # System health metrics
            self.logger.info("SYSTEM HEALTH:")
            self.logger.info(f"Models Retrained: {self.session_stats.get('models_retrained', 0)}")
            self.logger.info(f"RL Models Saved: {self.session_stats.get('rl_models_saved', 0)}")
            self.logger.info(f"Emergency Stops: {self.session_stats.get('emergency_stops', 0)}")

            # Performance analysis
            if win_rate >= 0.6:
                performance_rating = "EXCELLENT"
            elif win_rate >= 0.5:
                performance_rating = "GOOD"
            elif win_rate >= 0.4:
                performance_rating = "FAIR"
            else:
                performance_rating = "NEEDS IMPROVEMENT"

            self.logger.info(f"PERFORMANCE RATING: {performance_rating}")

            if profit_factor >= 2.0:
                self.logger.info("ANALYSIS: Strong profit factor indicates good risk management")
            elif profit_factor >= 1.5:
                self.logger.info("ANALYSIS: Moderate profit factor - consider optimizing win/loss ratio")
            else:
                self.logger.info("ANALYSIS: Low profit factor - focus on cutting losses and letting profits run")

            self.logger.info("=" * 80)

        except Exception as e:
            self.logger.error(f"Error logging performance summary: {e}")

    async def fundamental_monitor_loop(self):
        """Background fundamental monitor that checks for breaking news every 5 minutes"""
        self.logger.info("Fundamental Monitor started - checking for breaking news every 5 minutes")

        while self.app.running:
            try:
                # Wait 5 minutes
                await asyncio.sleep(300)

                if not self.app.running:
                    break

                self.logger.info("Fundamental Monitor: Checking for breaking news...")

                # Get all active positions
                active_positions = []
                symbols = self.config.get('trading', {}).get('symbols', [])

                for symbol in symbols:
                    try:
                        positions = mt5.positions_get(symbol=symbol)
                        if positions:
                            for position in positions:
                                if hasattr(position, 'magic') and position.magic == self.magic_number:
                                    active_positions.append(position)
                    except Exception as e:
                        self.logger.debug(f"Error checking positions for {symbol}: {e}")

                if not active_positions:
                    self.logger.debug("Fundamental Monitor: No active positions to monitor")
                    continue

                self.logger.info(f"Fundamental Monitor: Monitoring {len(active_positions)} active positions")

                # Check each position for fundamental updates
                for position in active_positions:
                    try:
                        await self.trading_engine.check_fundamental_updates_during_trade(position)
                    except Exception as e:
                        self.logger.error(f"Error in fundamental check for {position.symbol}: {e}")

                self.logger.info("Fundamental Monitor: Check completed")

            except Exception as e:
                self.logger.error(f"Error in fundamental monitor loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

        self.logger.info("Fundamental Monitor stopped")

    async def _check_orphaned_positions(self):
        """Check for positions that should have been closed but were missed (e.g., due to system restart)"""
        try:
            # Get all open positions
            positions = mt5.positions_get()
            if not positions:
                return
            
            current_time = self.time_manager.get_current_time()
            current_time_only = current_time.time()
            closure_time = self.time_manager.MT5_IMMEDIATE_CLOSE_TIME
            current_session = self.time_manager.get_current_session()
            
            orphaned_positions = []
            
            for pos in positions:
                # Check if this position was opened before closure time
                open_time = datetime.fromtimestamp(pos.time)
                if open_time.time() < closure_time:
                    # This position should have been closed at 22:00, BUT only if we're not in an active session
                    # (Sydney and Tokyo sessions are active after 22:00)
                    if current_session not in ['sydney', 'tokyo']:
                        orphaned_positions.append(pos)
            
            if orphaned_positions:
                self.logger.warning(f"Found {len(orphaned_positions)} orphaned positions that missed 22:00 closure")
                
                for pos in orphaned_positions:
                    self.logger.info(f"Force closing orphaned position: {pos.symbol} ticket {pos.ticket} (opened {datetime.fromtimestamp(pos.time).strftime('%H:%M:%S')})")
                    
                    # Close the position
                    close_result = await self.trading_engine.close_position_by_ticket(pos.ticket)
                    if close_result:
                        self.logger.info(f"Successfully closed orphaned position {pos.symbol} ticket {pos.ticket}")
                    else:
                        self.logger.error(f"Failed to close orphaned position {pos.symbol} ticket {pos.ticket}")
        
        except Exception as e:
            self.logger.error(f"Error checking orphaned positions: {e}")