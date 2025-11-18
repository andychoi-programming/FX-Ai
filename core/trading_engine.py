"""
FX-Ai Trading Engine - Refactored Version
Orchestrates trading operations using modular components
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Optional, Any
import MetaTrader5 as mt5
from utils.position_monitor import PositionMonitor
from utils.risk_validator import RiskValidator

# Import new modular components
from .order_executor import OrderExecutor
from .position_manager import PositionManager
from .stop_loss_manager import StopLossManager
from .take_profit_manager import TakeProfitManager
from .position_closer import PositionCloser

logger = logging.getLogger(__name__)


class TradingEngine:
    """Trading engine orchestrating modular trading components"""

    def __init__(
            self,
            mt5_connector,
            risk_manager,
            technical_analyzer,
            sentiment_analyzer,
            fundamental_collector,
            ml_predictor,
            adaptive_learning_manager=None):
        """Initialize trading engine with modular components"""
        self.logger = logging.getLogger(__name__)
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

        # Store magic number for easy access
        self.magic_number = self.config.get('trading', {}).get('magic_number')

        # Initialize modular components
        self.order_executor = OrderExecutor(mt5_connector, self.config, risk_manager, technical_analyzer)
        self.stop_loss_manager = StopLossManager(mt5_connector, self.config)
        self.take_profit_manager = TakeProfitManager(mt5_connector, self.config)
        self.position_closer = PositionCloser(mt5_connector, self.config, self.active_positions)
        self.position_manager = PositionManager(mt5_connector, risk_manager, self.config, adaptive_learning_manager, self.stop_loss_manager, self.take_profit_manager, self.position_closer, fundamental_collector)

        # Initialize position monitor for change detection
        self.position_monitor = PositionMonitor(self.magic_number)
        self.position_monitor.enable_alerts(True)

        # MT5 Success codes - CRITICAL FIX!
        self.SUCCESS_CODES = [
            mt5.TRADE_RETCODE_DONE,          # 10009 - Request completed
            mt5.TRADE_RETCODE_PLACED,        # 10008 - Order placed
            mt5.TRADE_RETCODE_DONE_PARTIAL,  # 10010 - Request partially completed
        ]
        
        # MT5 Error codes that need special handling
        self.RETRIABLE_ERRORS = [
            mt5.TRADE_RETCODE_REQUOTE,       # 10004 - Requote
            mt5.TRADE_RETCODE_CONNECTION,    # 10018 - No connection
            mt5.TRADE_RETCODE_PRICE_OFF,     # 10015 - Invalid price
        ]

        logger.info("Trading Engine initialized with position monitoring and improved MT5 error handling")

    def sync_with_mt5_positions(self) -> None:
        """Sync internal position tracking with existing MT5 positions"""
        try:
            # Get all positions from MT5
            positions = mt5.positions_get()  # type: ignore
            
            if positions is None:
                logger.warning("Failed to get positions from MT5")
                return
            
            synced_count = 0
            for position in positions:
                # Only sync positions with our magic number
                if position.magic == self.magic_number:
                    # Add to active positions tracking
                    self.active_positions[position.ticket] = {
                        'symbol': position.symbol,
                        'type': 'buy' if position.type == mt5.ORDER_TYPE_BUY else 'sell',
                        'volume': position.volume,
                        'entry': position.price_open,
                        'sl': position.sl,
                        'tp': position.tp,
                        'timestamp': datetime.fromtimestamp(position.time)
                    }
                    synced_count += 1
            
            if synced_count > 0:
                logger.info(f"Synced {synced_count} existing positions from MT5")
            else:
                logger.info("No existing positions found to sync")
                
        except Exception as e:
            logger.error(f"Error syncing with MT5 positions: {e}")

    async def sync_and_record_mt5_positions(self, adaptive_learning=None) -> int:
        """Sync with MT5 positions and record open trades in database for monitoring"""
        try:
            positions = mt5.positions_get()  # type: ignore
            
            if positions is None:
                logger.warning("Failed to get positions from MT5")
                return 0
            
            recorded_count = 0
            for position in positions:
                if position.magic == self.magic_number:
                    # Record open trade in database
                    trade_data = {
                        'id': None,  # Will be auto-assigned
                        'timestamp': datetime.fromtimestamp(position.time).isoformat(),
                        'symbol': position.symbol,
                        'direction': 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL',
                        'entry_price': position.price_open,
                        'exit_price': None,  # Open trade
                        'volume': position.volume,
                        'profit': position.profit,
                        'profit_pct': 0.0,  # Calculate if needed
                        'signal_strength': 0.5,  # Default for manual trades
                        'ml_score': 0.5,
                        'technical_score': 0.5,
                        'sentiment_score': 0.5,
                        'duration_minutes': int((datetime.now() - datetime.fromtimestamp(position.time)).seconds / 60),
                        'model_version': 'manual_sync',
                        'closure_reason': None,
                        'forced_closure': 0
                    }
                    
                    if adaptive_learning and hasattr(adaptive_learning, 'record_open_trade'):
                        adaptive_learning.record_open_trade(trade_data)
                        recorded_count += 1
                    
                    # Also update active positions tracking
                    self.active_positions[position.ticket] = {
                        'symbol': position.symbol,
                        'type': 'buy' if position.type == mt5.ORDER_TYPE_BUY else 'sell',
                        'volume': position.volume,
                        'entry': position.price_open,
                        'sl': position.sl,
                        'tp': position.tp,
                        'timestamp': datetime.fromtimestamp(position.time)
                    }
            
            if recorded_count > 0:
                logger.info(f"Synced and recorded {recorded_count} open positions from MT5")
            
            return recorded_count
                
        except Exception as e:
            logger.error(f"Error syncing and recording MT5 positions: {e}")
            return 0

    def get_filling_mode(self, symbol):
        """Get the correct filling mode for a symbol"""
        info = mt5.symbol_info(symbol)  # type: ignore
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

        symbol_info = mt5.symbol_info(symbol)  # type: ignore
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

        current_price = mt5.symbol_info_tick(symbol)  # type: ignore
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
                          stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                          price: Optional[float] = None, comment: str = "", signal_data: Optional[Dict] = None) -> Dict:
        """Place order through MT5 - delegated to OrderExecutor"""
        result = await self.order_executor.place_order(symbol, order_type, volume, stop_loss, take_profit, price, comment, signal_data)

        # Track the order if successful
        if result.get('success', False) and order_type.lower() in ['buy', 'sell']:
            self.active_positions[result.get('order', 0)] = {
                'symbol': symbol,
                'type': order_type,
                'volume': volume,
                'entry': result.get('price', price),
                'sl': stop_loss,
                'tp': take_profit,
                'timestamp': datetime.now(),
                'signal_data': signal_data or {  # Store signal data for learning
                    'technical_score': 0.5,
                    'fundamental_score': 0.5,
                    'sentiment_score': 0.5,
                    'ml_score': 0.0,
                    'signal_strength': 0.5
                }
            }

        return result

    async def execute_trade_with_validation(self, signal: Dict, orchestrator=None) -> Optional[Dict]:
        """
        Execute trade with comprehensive pre-flight checks

        Args:
            signal: Trading signal dictionary
            orchestrator: TradingOrchestrator instance for validation

        Returns:
            Trade result or None if validation fails
        """
        print(f"🔍 [TradingEngine] execute_trade_with_validation ENTERED for {signal.get('symbol')} {signal.get('direction')}")
        logger.info(f"🔍 [TradingEngine] execute_trade_with_validation ENTERED for {signal.get('symbol')} {signal.get('direction')}")
        try:
            print(f"🔍 [TradingEngine] execute_trade_with_validation called for {signal.get('symbol')} {signal.get('direction')}")
            logger.info(f"🔍 [TradingEngine] execute_trade_with_validation called for {signal.get('symbol')} {signal.get('direction')}")

            # 1. Check data freshness if orchestrator available
            print(f"🔍 [TradingEngine] Checking if orchestrator available: {orchestrator is not None}")
            if orchestrator and hasattr(orchestrator, 'log_system_health'):
                print("🔍 [TradingEngine] Checking system health...")
                self.logger.info("🔍 [TradingEngine] Checking system health...")
                if not orchestrator.log_system_health():
                    print("🔍 [TradingEngine] System health check FAILED")
                    self.logger.error("[FAIL] Trade rejected - System health check failed")
                    return None
                print("🔍 [TradingEngine] System health check PASSED")
                self.logger.info("🔍 [TradingEngine] System health check passed")

            # 2. Check daily limits
            print(f"🔍 [TradingEngine] Checking if daily_limit_tracker available: {orchestrator and hasattr(orchestrator, 'daily_limit_tracker')}")
            if orchestrator and hasattr(orchestrator, 'daily_limit_tracker'):
                print("🔍 [TradingEngine] Checking daily limits...")
                self.logger.info("🔍 [TradingEngine] Checking daily limits...")
                if not orchestrator.daily_limit_tracker.can_trade(signal['symbol']):
                    print(f"🔍 [TradingEngine] Daily limit check FAILED for {signal['symbol']}")
                    self.logger.error(f"[FAIL] Trade rejected - Daily limit reached for {signal['symbol']}")
                    return None
                print("🔍 [TradingEngine] Daily limits check PASSED")
                self.logger.info("🔍 [TradingEngine] Daily limits check passed")

            # 3. Validate position size
            print(f"🔍 [TradingEngine] Checking if risk_manager available: {orchestrator and hasattr(orchestrator, 'risk_manager')}")
            if orchestrator and hasattr(orchestrator, 'risk_manager'):
                print("🔍 [TradingEngine] Validating position size...")
                self.logger.info("🔍 [TradingEngine] Validating position size...")
                account_balance = 0
                if hasattr(orchestrator, 'mt5') and orchestrator.mt5:
                    account_info = orchestrator.mt5.get_account_info()
                    account_balance = account_info.get('balance', 0) if account_info else 0

                position_size = signal.get('position_size', 0)
                print(f"🔍 [TradingEngine] Position size: {position_size}, Account balance: ${account_balance:,.2f}")
                self.logger.info(f"🔍 [TradingEngine] Position size: {position_size}, Account balance: ${account_balance:,.2f}")

                if not orchestrator.risk_manager.validate_position_size(
                    signal['symbol'],
                    position_size,
                    account_balance
                ):
                    print("🔍 [TradingEngine] Position size validation FAILED")
                    self.logger.error("[FAIL] Trade rejected - Position size validation failed")
                    self.logger.error(f"   Symbol: {signal['symbol']}")
                    self.logger.error(f"   Position size: {signal.get('position_size', 0)}")
                    self.logger.error(f"   Account balance: ${account_balance:,.2f}")
                    return None
                print("🔍 [TradingEngine] Position size validation PASSED")
                self.logger.info("🔍 [TradingEngine] Position size validation passed")

            # 4. Execute trade directly (circuit breaker bypassed for now due to async issues)
            print("🔍 [TradingEngine] About to call _execute_trade_safe...")
            self.logger.info("🔍 [TradingEngine] About to call _execute_trade_safe...")
            try:
                result = await self._execute_trade_safe(signal)
                print(f"🔍 [TradingEngine] _execute_trade_safe returned: {result}")
                self.logger.info(f"🔍 [TradingEngine] _execute_trade_safe returned: {result}")
            except Exception as e:
                print(f"🔍 [TradingEngine] _execute_trade_safe EXCEPTION: {e}")
                self.logger.error(f"[FAIL] Trade rejected - Execution error: {e}")
                return None

            # 5. Record trade in daily tracker
            print(f"🔍 [TradingEngine] Checking if should record trade: result={result}, success={result and result.get('success') if result else False}")
            if orchestrator and hasattr(orchestrator, 'daily_limit_tracker') and result and result.get('success'):
                orchestrator.daily_limit_tracker.record_trade(signal['symbol'])

            print(f"🔍 [TradingEngine] Method returning: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Error in trade execution with validation: {e}")
            return None

    async def _execute_trade_safe(self, signal: Dict) -> Optional[Dict]:
        """Execute trade with comprehensive error handling using FixedOrderExecutor logic"""
        try:
            # Validate MT5 connection first
            if not mt5.initialize():
                self.logger.error("MT5 not initialized")
                return {'success': False, 'error': 'MT5 not initialized'}

            symbol = signal['symbol']
            direction = signal['direction']

            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"Symbol {symbol} not found")
                return {'success': False, 'error': f'Symbol {symbol} not found'}

            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    self.logger.error(f"Failed to select {symbol}")
                    return {'success': False, 'error': f'Failed to select {symbol}'}

            # Get current prices
            prices = mt5.symbol_info_tick(symbol)
            if prices is None:
                self.logger.error(f"Cannot get prices for {symbol}")
                return {'success': False, 'error': f'Cannot get prices for {symbol}'}

            # Determine order type and entry price
            if direction.upper() == 'BUY':
                order_type = mt5.ORDER_TYPE_BUY
                entry_price = prices.ask
            elif direction.upper() == 'SELL':
                order_type = mt5.ORDER_TYPE_SELL
                entry_price = prices.bid
            else:
                self.logger.error(f"Invalid direction: {direction}")
                return {'success': False, 'error': f'Invalid direction: {direction}'}

            # Normalize entry price
            tick_size = symbol_info.trade_tick_size
            if tick_size == 0:
                tick_size = symbol_info.point
            entry_price = round(entry_price / tick_size) * tick_size
            entry_price = round(entry_price, symbol_info.digits)

            # Calculate stop distances (1.5-2x minimum requirements)
            stops_level = symbol_info.trade_stops_level
            point = symbol_info.point
            min_stop_price = stops_level * point

            # Use 2x minimum for safety
            sl_distance_price = max(min_stop_price * 2, min_stop_price * 1.5)

            # For metals, ensure minimum distances
            if symbol in ['XAUUSD', 'XAGUSD']:
                sl_distance_price = max(sl_distance_price, 0.5)  # At least 50 cents for gold/silver

            # Calculate TP distance for RR ratio (use 2.0 as default)
            rr_ratio = 2.0
            tp_distance_price = sl_distance_price * rr_ratio

            # Calculate SL and TP prices
            if direction.upper() == 'BUY':
                sl_price = entry_price - sl_distance_price
                tp_price = entry_price + tp_distance_price
            else:  # SELL
                sl_price = entry_price + sl_distance_price
                tp_price = entry_price - tp_distance_price

            # Normalize SL and TP
            sl_price = round(sl_price / tick_size) * tick_size
            tp_price = round(tp_price / tick_size) * tick_size
            sl_price = round(sl_price, symbol_info.digits)
            tp_price = round(tp_price, symbol_info.digits)

            # Calculate lot size based on risk amount
            risk_amount = signal.get('position_size', 0.01)  # Default to 0.01 if not specified
            stop_loss_pips = sl_distance_price / point
            pip_value = (symbol_info.trade_tick_value / point) * symbol_info.trade_contract_size
            lot_size = risk_amount / (stop_loss_pips * pip_value)

            # Round lot size to step
            lot_step = symbol_info.volume_step
            lot_size = round(lot_size / lot_step) * lot_step
            lot_size = max(symbol_info.volume_min, min(symbol_info.volume_max, lot_size))

            # Test filling modes (use IOC as primary, fallback to others)
            filling_modes = [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_FOK]
            working_filling = mt5.ORDER_FILLING_IOC  # Default

            for filling in filling_modes:
                test_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot_size,
                    "type": order_type,
                    "price": entry_price,
                    "sl": sl_price,
                    "tp": tp_price,
                    "deviation": 10,
                    "magic": self.config.get('trading', {}).get('magic_number', 123456),
                    "comment": f"test_fill_{filling}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": filling,
                }

                test_result = mt5.order_send(test_request)
                if test_result and test_result.retcode == mt5.TRADE_RETCODE_DONE:
                    working_filling = filling
                    # Close the test order immediately
                    if test_result.order:
                        close_request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": symbol,
                            "volume": lot_size,
                            "type": mt5.ORDER_TYPE_SELL if order_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                            "position": test_result.order,
                            "price": prices.bid if order_type == mt5.ORDER_TYPE_BUY else prices.ask,
                            "deviation": 10,
                            "magic": test_request["magic"],
                            "comment": "Close test position",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": filling,
                        }
                        mt5.order_send(close_request)
                    break

            # Create final order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": entry_price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 10,
                "magic": self.config.get('trading', {}).get('magic_number', 123456),
                "comment": f"FX-Ai {symbol} Fixed",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": working_filling,
            }

            self.logger.info(f"Executing {direction} order for {symbol}:")
            self.logger.info(f"  Entry: {entry_price}, SL: {sl_price}, TP: {tp_price}")
            self.logger.info(f"  Lot Size: {lot_size}, RR: {rr_ratio}")
            self.logger.info(f"  Stops Level: {stops_level}, Min Distance: {min_stop_price}")

            # Send order
            result = mt5.order_send(request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"✅ Order executed successfully: Ticket {result.order}")
                return {
                    'success': True,
                    'order_id': result.order,
                    'symbol': symbol,
                    'direction': direction,
                    'volume': lot_size,
                    'entry_price': entry_price,
                    'sl': sl_price,
                    'tp': tp_price,
                    'comment': result.comment
                }
            else:
                error = mt5.last_error()
                self.logger.error(f"❌ Order failed: {error}")
                return {
                    'success': False,
                    'error': f'Order failed: {error}',
                    'symbol': symbol,
                    'direction': direction
                }

        except Exception as e:
            self.logger.error(f"Exception in _execute_trade_safe: {str(e)}", exc_info=True)
            return {'success': False, 'error': f'Exception: {str(e)}'}

    async def execute_trade(self, signal: Dict) -> Optional[Dict]:
        """Execute a trading signal - main entry point for trade execution"""
        try:
            print(f"🔍 [TradingEngine] execute_trade called with signal: {signal}")
            self.logger.info(f"🔍 [TradingEngine] execute_trade called with signal: {signal}")
            symbol = signal['symbol']
            direction = signal['direction']
            
            # Calculate position size if not provided
            if 'position_size' not in signal:
                default_sl_pips = self.config.get('trading', {}).get('default_sl_pips', 20)
                print(f"🔍 [TradingEngine] About to calculate position size for {symbol}")
                self.logger.info(f"🔍 [TradingEngine] Calculating position size for {symbol} with default_sl_pips={default_sl_pips}")
                signal['position_size'] = self.risk_manager.calculate_position_size(symbol, default_sl_pips)
                print(f"🔍 [TradingEngine] Position size calculated: {signal['position_size']}")
                self.logger.info(f"🔍 [TradingEngine] Calculated position_size: {signal['position_size']}")
            
            volume = signal['position_size']
            print(f"🔍 [TradingEngine] Volume set to: {volume}")
            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')
            entry_price = signal.get('entry_price')

            # Use OrderManager for hybrid order placement
            self.logger.info(f"🔍 [TradingEngine] About to call order_manager.place_order for {symbol} {direction}")
            try:
                result = await self.order_executor.order_manager.place_order(
                    symbol=symbol,
                    signal=direction,
                    entry_strategy="stop",  # Default to stop orders
                    volume=volume,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    signal_data={
                        'technical_score': signal.get('technical_score', 0.5),
                        'fundamental_score': signal.get('fundamental_score', 0.5),
                        'sentiment_score': signal.get('sentiment_score', 0.5),
                        'ml_score': signal.get('ml_score', 0.0),
                        'signal_strength': signal.get('signal_strength', 0.5),
                        'risk_multiplier': signal.get('risk_multiplier', 1.0)
                    }
                )
                self.logger.info(f"🔍 [TradingEngine] order_manager.place_order returned: {result}")
                self.logger.info(f"🔍 [TradingEngine] Result type: {type(result)}, Result content: {result}")
            except Exception as e:
                self.logger.error(f"🔍 [TradingEngine] Exception in order_manager.place_order: {e}")
                import traceback
                self.logger.error(f"🔍 [TradingEngine] Traceback: {traceback.format_exc()}")
                result = {'success': False, 'error': f'Exception: {str(e)}'}

            if result and result.get('success', False):
                self.logger.info(f"🔍 [TradingEngine] Result is truthy and has success=True, updating result")
                # Add signal data to result for monitoring
                result.update({
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': entry_price,
                    'position_size': volume,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'technical_score': signal.get('technical_score', 0.5),
                    'fundamental_score': signal.get('fundamental_score', 0.5),
                    'sentiment_score': signal.get('sentiment_score', 0.5),
                    'ml_score': signal.get('ml_score', 0.0),
                    'signal_strength': signal.get('signal_strength', 0.5),
                    'timestamp': self.mt5.get_server_time() if self.mt5 else datetime.now()
                })
            else:
                self.logger.info(f"🔍 [TradingEngine] Result is falsy or success=False: result={result}, result.get('success')={result.get('success') if result else 'N/A'}")
                # Ensure we always return a dict
                if not result:
                    result = {'success': False, 'error': 'Unknown error - result is None'}

            return result

        except Exception as e:
            logger.error(f"❌ CRITICAL: Error executing trade for {signal.get('symbol', 'unknown')}: {e}")
            logger.error(f"   Signal details: {signal}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return {'success': False, 'error': f'Critical error: {str(e)}'}

    async def manage_positions(self, symbol: str, time_manager=None, adaptive_learning=None):
        """Manage open positions for a symbol - delegated to PositionManager"""
        await self.position_manager.manage_positions(symbol, time_manager, adaptive_learning)

    async def check_adaptive_sl_tp_adjustment(self, position, adaptive_learning=None):
        """Check if position SL/TP should be adjusted based on adaptive learning - delegated to PositionManager"""
        await self.position_manager._check_adaptive_sl_tp_adjustment(position, adaptive_learning)

    async def update_trailing_stop(self, position) -> None:
        """Update trailing stop for a position - delegated to StopLossManager"""
        await self.stop_loss_manager.update_trailing_stop(position)

    async def apply_breakeven(self, position) -> None:
        """Apply breakeven stop loss - delegated to StopLossManager"""
        await self.stop_loss_manager.apply_breakeven(position)

    async def close_all_positions(self):
        """Close all positions - delegated to PositionCloser"""
        await self.position_closer.close_all_positions()

    async def close_position_by_ticket(self, ticket: int, reason: str = "Manual close") -> bool:
        """Close position by ticket number"""
        try:
            # Get position by ticket
            position = self.get_position_by_ticket(ticket)
            if not position:
                logger.warning(f"Position with ticket {ticket} not found")
                return False
            
            # Close the position
            return await self.position_closer.close_position(position, reason)
            
        except Exception as e:
            logger.error(f"Error closing position by ticket {ticket}: {e}")
            return False

    def get_position_by_ticket(self, ticket: int):
        """Get position by ticket number"""
        positions = mt5.positions_get(ticket=ticket)  # type: ignore
        if positions and len(positions) > 0:
            return positions[0]
        return None

    def get_trade_history(self, ticket: int) -> Optional[Dict]:
        """Get trade history for a ticket"""
        try:
            # Get deals for this position
            deals = mt5.history_deals_get(position=ticket)  # type: ignore

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

    async def check_fundamental_updates_during_trade(self, position) -> None:
        """Check for fundamental updates during active trade and take action - delegated to PositionManager"""
        await self.position_manager._check_fundamental_updates_during_trade(position)

    async def move_sl_to_breakeven(self, position) -> None:
        """Move stop loss to breakeven level - delegated to StopLossManager"""
        await self.stop_loss_manager.move_sl_to_breakeven(position)

    async def tighten_stops_aggressively(self, position) -> None:
        """Tighten stop loss aggressively due to adverse conditions - delegated to StopLossManager"""
        await self.stop_loss_manager.tighten_stops_aggressively(position)

    async def extend_take_profit(self, position) -> None:
        """Extend take profit due to favorable conditions - delegated to TakeProfitManager"""
        await self.take_profit_manager.extend_take_profit(position)

    def execute_order(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute order with CORRECT success/failure detection
        
        Args:
            request: MT5 order request dictionary
            
        Returns:
            Dict with success status and details
        """
        try:
            # Send order to MT5
            result = mt5.order_send(request)
            
            if result is None:
                error = mt5.last_error()
                self.logger.error(f"[FAIL] Order send returned None: {error}")
                return {
                    'success': False,
                    'error': f"MT5 connection error: {error}",
                    'retcode': None
                }
            
            # Log the raw result for debugging
            self.logger.debug(f"MT5 Result - retcode: {result.retcode}, comment: {result.comment}")
            
            # [PASS] CHECK FOR SUCCESS (THIS IS THE CRITICAL FIX!)
            if result.retcode in self.SUCCESS_CODES:
                # SUCCESS - Order was placed or executed
                self.logger.info(f"[PASS] Order placed successfully!")
                self.logger.info(f"   Ticket: {result.order}")
                self.logger.info(f"   Retcode: {result.retcode} ({self._get_retcode_description(result.retcode)})")
                self.logger.info(f"   Comment: {result.comment}")
                
                return {
                    'success': True,
                    'ticket': result.order,
                    'retcode': result.retcode,
                    'comment': result.comment,
                    'price': result.price if hasattr(result, 'price') else None,
                    'volume': result.volume if hasattr(result, 'volume') else None
                }
            
            # [FAIL] FAILURE - Order was rejected or failed
            else:
                error_desc = self._get_retcode_description(result.retcode)
                self.logger.warning(f"[FAIL] Order rejected!")
                self.logger.warning(f"   Retcode: {result.retcode} ({error_desc})")
                self.logger.warning(f"   Comment: {result.comment}")
                
                # Check if error is retriable
                is_retriable = result.retcode in self.RETRIABLE_ERRORS
                
                return {
                    'success': False,
                    'error': result.comment,
                    'retcode': result.retcode,
                    'error_description': error_desc,
                    'retriable': is_retriable
                }
                
        except Exception as e:
            self.logger.error(f"[FAIL] Exception during order execution: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'retcode': None
            }
    
    def _get_retcode_description(self, retcode: int) -> str:
        """Get human-readable description of MT5 return code"""
        
        descriptions = {
            # Success codes
            10008: "Order placed successfully",
            10009: "Request executed successfully", 
            10010: "Request partially completed",
            
            # Common error codes
            10004: "Requote",
            10006: "Request rejected",
            10007: "Request canceled by trader",
            10008: "Order placed",
            10009: "Request completed",
            10010: "Only part of the request was completed",
            10011: "Request processing error",
            10012: "Request canceled by timeout",
            10013: "Invalid request",
            10014: "Invalid volume in the request",
            10015: "Invalid price in the request",
            10016: "Invalid stops in the request",
            10017: "Trade is disabled",
            10018: "Market is closed",
            10019: "Not enough money to complete the request",
            10020: "Prices changed",
            10021: "No quotes to process the request",
            10022: "Invalid order expiration date in the request",
            10023: "Order state changed",
            10024: "Too frequent requests",
            10025: "No changes in request",
            10026: "Autotrading disabled by server",
            10027: "Autotrading disabled by client terminal",
            10028: "Request locked for processing",
            10029: "Order or position frozen",
            10030: "Invalid order filling type",
            10031: "No connection with the trade server",
            10032: "Operation is allowed only for live accounts",
            10033: "The number of pending orders has reached the limit",
            10034: "Volume of orders and positions for the symbol has reached the limit",
            10035: "Incorrect or prohibited order type",
            10036: "Position with the specified POSITION_IDENTIFIER has already been closed",
            10038: "Close volume exceeds the current position volume",
            10039: "Close order already exists for the position",
            10040: "The number of open positions simultaneously present on an account can be limited"
        }
        
        return descriptions.get(retcode, f"Unknown return code: {retcode}")
    
    def place_pending_order(
        self, 
        symbol: str,
        order_type: int,
        volume: float,
        price: float,
        sl: float,
        tp: float,
        comment: str = "",
        magic: int = 234000
    ) -> Dict[str, Any]:
        """
        Place a pending order (buy_stop or sell_stop)
        
        Returns:
            Dict with success status and order details
        """
        
        # Prepare the order request
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        self.logger.info(f"[ORDER] Placing pending order: {symbol} {volume} lots @ {price}")
        self.logger.info(f"   Type: {self._get_order_type_name(order_type)}")
        self.logger.info(f"   SL: {sl}, TP: {tp}")
        
        # Execute the order
        result = self.execute_order(request)
        
        return result
    
    def _get_order_type_name(self, order_type: int) -> str:
        """Get human-readable order type name"""
        types = {
            mt5.ORDER_TYPE_BUY: "BUY (Market)",
            mt5.ORDER_TYPE_SELL: "SELL (Market)",
            mt5.ORDER_TYPE_BUY_LIMIT: "BUY LIMIT",
            mt5.ORDER_TYPE_SELL_LIMIT: "SELL LIMIT",
            mt5.ORDER_TYPE_BUY_STOP: "BUY STOP",
            mt5.ORDER_TYPE_SELL_STOP: "SELL STOP",
            mt5.ORDER_TYPE_BUY_STOP_LIMIT: "BUY STOP LIMIT",
            mt5.ORDER_TYPE_SELL_STOP_LIMIT: "SELL STOP LIMIT",
        }
        return types.get(order_type, f"Unknown type: {order_type}")
