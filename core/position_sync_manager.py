import asyncio
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import MetaTrader5 as mt5
import logging


class PositionSyncManager:
    """Manages synchronization between MT5 positions and database records"""

    def __init__(self, db_path: str = None, logger: Optional[logging.Logger] = None):
        # Use the same database path as the learning database
        if db_path is None:
            from utils.config_loader import ConfigLoader
            config_loader = ConfigLoader()
            config = config_loader.load_config()
            self.db_path = config.get('data', {}).get('learning_database_path', 'performance_history.db')
        else:
            self.db_path = db_path

        self.logger = logger or logging.getLogger("PositionSyncManager")
        self.sync_interval = 60  # seconds
        self.max_position_age = timedelta(hours=4)  # Force close after 4 hours
        self._running = False

    async def start_monitoring(self):
        """Start the position synchronization monitoring loop"""
        self._running = True
        self.logger.info("🔄 Starting position synchronization monitoring...")

        while self._running:
            try:
                await self.sync_positions()
                await asyncio.sleep(self.sync_interval)
            except Exception as e:
                self.logger.error(f"Error in position sync monitoring: {e}")
                await asyncio.sleep(self.sync_interval)

    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self._running = False
        self.logger.info("🛑 Position synchronization monitoring stopped")

    async def sync_positions(self):
        """Main synchronization method - sync MT5 positions with database"""
        try:
            # Sync closed positions from MT5 history
            await self.sync_closed_positions()

            # Check for overdue positions to force close
            await self.check_overdue_positions()

        except Exception as e:
            self.logger.error(f"Error syncing positions: {e}")

    async def sync_closed_positions(self):
        """Check MT5 history for closed positions and update database"""
        try:
            if not mt5.initialize():
                self.logger.error("MT5 not initialized for position sync")
                return

            # Get all open positions from database that might be closed
            open_positions = self._get_open_positions_from_db()

            if not open_positions:
                return

            self.logger.info(f"🔍 Checking {len(open_positions)} open positions for closure...")

            closed_count = 0
            for position in open_positions:
                ticket = position['ticket']
                symbol = position['symbol']

                # Check if position is still open in MT5
                mt5_positions = mt5.positions_get(ticket=ticket)
                if mt5_positions is None or len(mt5_positions) == 0:
                    # Position not found in MT5 - check history for closure
                    closure_info = self._get_closure_from_history(ticket, symbol)
                    if closure_info:
                        await self._update_database_closure(ticket, closure_info)
                        closed_count += 1
                        self.logger.info(f"✅ Synced closure for ticket {ticket}")

            if closed_count > 0:
                self.logger.info(f"🔄 Synced {closed_count} position closures")

        except Exception as e:
            self.logger.error(f"Error syncing closed positions: {e}")

    def _get_open_positions_from_db(self) -> List[Dict]:
        """Get all positions marked as open in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT ticket, symbol, direction, entry_price, volume, entry_time
                    FROM trades
                    WHERE (status IS NULL OR status = 'open' OR status = 'OPEN') 
                      AND exit_time IS NULL 
                      AND ticket IS NOT NULL
                    ORDER BY entry_time DESC
                """)
                rows = cursor.fetchall()

                positions = []
                for row in rows:
                    positions.append({
                        'ticket': row[0],
                        'symbol': row[1],
                        'direction': row[2],
                        'entry_price': row[3],
                        'volume': row[4],
                        'entry_time': row[5]
                    })

                return positions

        except Exception as e:
            self.logger.error(f"Error getting open positions from DB: {e}")
            return []

    def _get_closure_from_history(self, ticket: int, symbol: str) -> Optional[Dict]:
        """Check MT5 history for position closure details"""
        try:
            # Get history deals for this position
            from_date = datetime.now() - timedelta(days=7)  # Look back 7 days
            to_date = datetime.now()

            history = mt5.history_deals_get(
                position=ticket,
                date_from=int(from_date.timestamp()),
                date_to=int(to_date.timestamp())
            )

            if history is None or len(history) == 0:
                return None

            # Find the closing deal
            for deal in history:
                if deal.position_id == ticket and deal.entry == mt5.DEAL_ENTRY_OUT:
                    # This is the closing deal
                    profit = deal.profit if hasattr(deal, 'profit') else 0
                    commission = deal.commission if hasattr(deal, 'commission') else 0
                    swap = deal.swap if hasattr(deal, 'swap') else 0
                    exit_price = deal.price
                    exit_time = datetime.fromtimestamp(deal.time)

                    return {
                        'exit_price': exit_price,
                        'exit_time': exit_time,
                        'profit': profit,
                        'commission': commission,
                        'swap': swap,
                        'closure_reason': 'mt5_closed'
                    }

            return None

        except Exception as e:
            self.logger.error(f"Error getting closure from history for ticket {ticket}: {e}")
            return None

    async def _update_database_closure(self, ticket: int, closure_info: Dict):
        """Update database record with closure information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE trades
                    SET exit_price = ?,
                        exit_time = ?,
                        profit = ?,
                        commission = ?,
                        swap = ?,
                        closure_reason = ?,
                        status = 'closed',
                        forced_closure = 0
                    WHERE ticket = ?
                """, (
                    closure_info['exit_price'],
                    closure_info['exit_time'].isoformat(),
                    closure_info['profit'],
                    closure_info['commission'],
                    closure_info['swap'],
                    closure_info['closure_reason'],
                    ticket
                ))

                conn.commit()
                self.logger.info(f"✅ Updated closure for ticket {ticket} in database")

        except Exception as e:
            self.logger.error(f"Error updating database closure for ticket {ticket}: {e}")

    async def check_overdue_positions(self):
        """Check for positions that have been open too long and force close them"""
        try:
            open_positions = self._get_open_positions_from_db()
            now = datetime.now()

            for position in open_positions:
                entry_time_str = position['entry_time']
                if entry_time_str is None:
                    self.logger.warning(f"⚠️ Position {position['ticket']} has NULL entry_time, skipping overdue check")
                    continue
                    
                if isinstance(entry_time_str, str):
                    entry_time = datetime.fromisoformat(entry_time_str)
                else:
                    entry_time = entry_time_str

                age = now - entry_time

                if age > self.max_position_age:
                    self.logger.warning(f"⚠️ Position {position['ticket']} overdue ({age.total_seconds()/3600:.1f}h), forcing closure")
                    await self._force_close_position(position)

        except Exception as e:
            self.logger.error(f"Error checking overdue positions: {e}")

    async def _force_close_position(self, position: Dict):
        """Force close an overdue position"""
        try:
            ticket = position['ticket']
            symbol = position['symbol']
            direction = position['direction']
            volume = position['volume']

            if not mt5.initialize():
                self.logger.error("MT5 not initialized for force close")
                return

            # Get current price for closing
            prices = mt5.symbol_info_tick(symbol)
            if prices is None:
                self.logger.error(f"Cannot get prices for force close of {symbol}")
                return

            # Determine close type and price
            if direction.upper() == 'BUY':
                close_type = mt5.ORDER_TYPE_SELL
                close_price = prices.bid
            else:
                close_type = mt5.ORDER_TYPE_BUY
                close_price = prices.ask

            # Send close order
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": close_type,
                "position": ticket,
                "price": close_price,
                "deviation": 10,
                "magic": 123456,  # Use default magic number
                "comment": "Forced close - overdue",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(close_request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"✅ Force closed position {ticket}")

                # Update database with forced closure
                await self._update_database_closure(ticket, {
                    'exit_price': close_price,
                    'exit_time': datetime.now(),
                    'profit': 0,  # Will be updated when MT5 history is synced
                    'commission': 0,
                    'swap': 0,
                    'closure_reason': 'forced_overdue'
                })
            else:
                retcode = result.retcode if result else 'None'
                self.logger.error(f"❌ Failed to force close position {ticket}, retcode: {retcode}")

        except Exception as e:
            self.logger.error(f"Error force closing position {position['ticket']}: {e}")

    def get_sync_stats(self) -> Dict:
        """Get synchronization statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Count open positions
                cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'open'")
                open_count = cursor.fetchone()[0]

                # Count closed positions
                cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'closed'")
                closed_count = cursor.fetchone()[0]

                # Count forced closures
                cursor.execute("SELECT COUNT(*) FROM trades WHERE forced_closure = 1")
                forced_count = cursor.fetchone()[0]

                return {
                    'open_positions': open_count,
                    'closed_positions': closed_count,
                    'forced_closures': forced_count,
                    'total_positions': open_count + closed_count
                }

        except Exception as e:
            self.logger.error(f"Error getting sync stats: {e}")
            return {'error': str(e)}