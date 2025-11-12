"""
FX-Ai Position Closer Module
Handles position closing operations and trade history
"""

import logging
from datetime import datetime
from typing import Dict, Optional
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class PositionCloser:
    """Handles position closing and trade history operations"""

    def __init__(self, mt5_connector, config: dict, active_positions: dict):
        """Initialize position closer"""
        self.mt5 = mt5_connector
        self.config = config
        self.active_positions = active_positions
        self.magic_number = config.get('trading', {}).get('magic_number', 20241029)
        self.max_slippage = config.get('trading', {}).get('max_slippage', 3)

    async def close_all_positions(self) -> None:
        """Close all open positions - PROPERLY ASYNC"""
        try:
            positions = mt5.positions_get()  # type: ignore

            if positions is None or len(positions) == 0:
                logger.info("No positions to close")
                return

            closed_count = 0
            for position in positions:
                if position.magic == self.magic_number:
                    success = await self.close_position(position)
                    if success:
                        closed_count += 1

            logger.info(f"Closed {closed_count} positions")

        except Exception as e:
            logger.error(f"Error closing all positions: {e}")

    async def close_position(self, position, reason: str = "Manual close") -> bool:
        """Close a single position - ASYNC"""
        try:
            # Select symbol for trading
            if not mt5.symbol_select(position.symbol, True):  # type: ignore
                logger.error(f"Failed to select symbol {position.symbol} for closing")
                return False

            # Determine order type for closing
            tick = mt5.symbol_info_tick(position.symbol)  # type: ignore
            if not tick:
                logger.error(f"Failed to get tick data for {position.symbol} - cannot close position")
                return False

            if position.type == mt5.ORDER_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask

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
                "comment": f"FX-Ai close: {reason}",
                "type_time": mt5.ORDER_TIME_GTC,
            }

            # Send close order
            result = mt5.order_send(request)  # type: ignore

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