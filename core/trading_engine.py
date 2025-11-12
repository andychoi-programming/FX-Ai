"""
FX-Ai Trading Engine - Refactored Version
Orchestrates trading operations using modular components
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Optional
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
        self.magic_number = self.config.get('trading', {}).get('magic_number', 20241029)

        # Initialize modular components
        self.order_executor = OrderExecutor(mt5_connector, self.config)
        self.stop_loss_manager = StopLossManager(mt5_connector, self.config)
        self.take_profit_manager = TakeProfitManager(mt5_connector, self.config)
        self.position_closer = PositionCloser(mt5_connector, self.config, self.active_positions)
        self.position_manager = PositionManager(mt5_connector, risk_manager, self.config, adaptive_learning_manager, self.stop_loss_manager, self.take_profit_manager, self.position_closer, fundamental_collector)

        # Initialize position monitor for change detection
        self.position_monitor = PositionMonitor(self.magic_number)
        self.position_monitor.enable_alerts(True)

        logger.info("Trading Engine initialized with position monitoring")

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
                          price: Optional[float] = None, comment: str = "") -> Dict:
        """Place order through MT5 - delegated to OrderExecutor"""
        result = await self.order_executor.place_order(symbol, order_type, volume, stop_loss, take_profit, price, comment)

        # Track the order if successful
        if result.get('success', False) and order_type.lower() in ['buy', 'sell']:
            self.active_positions[result.get('order', 0)] = {
                'symbol': symbol,
                'type': order_type,
                'volume': volume,
                'entry': result.get('price', price),
                'sl': stop_loss,
                'tp': take_profit,
                'timestamp': datetime.now()
            }

        return result

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


        """Close all positions - delegated to PositionCloser"""
        await self.position_closer.close_all_positions()

    async def close_position(self, position, reason: str = "Manual close") -> bool:
        """Close position - delegated to PositionCloser"""
        return await self.position_closer.close_position(position, reason)

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
