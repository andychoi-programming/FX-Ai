#!/usr/bin/env python3
"""
Position Synchronization Manager
Synchronizes MT5 positions with database and enforces trade closures
"""

import MetaTrader5 as mt5
import sqlite3
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import os

class PositionSyncManager:
    """Synchronizes MT5 positions with database and enforces closures"""

    def __init__(self, db_path: str = "D:/FX-Ai-Data/databases/performance_history.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.mt5_initialized = False

    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        if not mt5.initialize():
            self.logger.error("MT5 initialization failed")
            return False
        self.mt5_initialized = True
        return True

    def sync_closed_positions(self) -> Dict:
        """Check MT5 history and update database for closed trades"""
        results = {
            'synced': 0,
            'errors': 0,
            'forced_closures': 0,
            'total_checked': 0
        }

        if not self.mt5_initialized and not self.initialize_mt5():
            return results

        try:
            # Get all open trades from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, symbol, direction, entry_price, volume, timestamp
                FROM trades
                WHERE exit_price IS NULL
                ORDER BY timestamp DESC
            """)
            open_db_trades = cursor.fetchall()

            self.logger.info(f"Found {len(open_db_trades)} open trades in database")
            results['total_checked'] = len(open_db_trades)

            # Check each trade in MT5
            for trade in open_db_trades:
                trade_id, symbol, direction, entry_price, volume, entry_time = trade

                # Check if position still exists in MT5
                position = self._get_mt5_position_by_symbol_and_entry(symbol, entry_price, direction)

                if position is None:
                    # Position closed in MT5 - check history
                    closure_info = self._get_closure_from_history(symbol, entry_time, direction)

                    if closure_info:
                        # Update database with closure info
                        self._update_database_closure(
                            trade_id,
                            closure_info['exit_price'],
                            closure_info['profit'],
                            closure_info['exit_time'],
                            closure_info['reason']
                        )
                        results['synced'] += 1
                        self.logger.info(f"✅ Synced closure for {symbol} (ID:{trade_id}): "
                                       f"P&L: ${closure_info['profit']:.2f}, "
                                       f"Reason: {closure_info['reason']}")
                    else:
                        self.logger.warning(f"❌ Could not find closure info for {symbol} (ID:{trade_id})")
                else:
                    # Position still open - check if needs forced closure
                    if self._should_force_close(position, entry_time):
                        success = self._force_close_position(position)
                        if success:
                            results['forced_closures'] += 1

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Position sync error: {e}")
            results['errors'] += 1

        return results

    def _get_mt5_position_by_symbol_and_entry(self, symbol: str, entry_price: float, direction: str):
        """Get MT5 position by symbol and approximate entry price"""
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return None

        # Find position with matching direction and close entry price
        for pos in positions:
            # Check direction
            expected_type = 0 if direction.upper() == 'BUY' else 1  # POSITION_TYPE_BUY = 0, SELL = 1
            if pos.type != expected_type:
                continue

            # Check entry price (allow small tolerance)
            price_diff = abs(pos.price_open - entry_price)
            if price_diff < 0.00001:  # Very small tolerance for price match
                return pos

        return None

    def _get_closure_from_history(self, symbol: str, entry_time: str, direction: str) -> Optional[Dict]:
        """Get closure details from MT5 history"""
        try:
            # Parse entry time and get deals from that time onwards
            entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
            from_date = entry_dt - timedelta(minutes=5)  # Look 5 minutes before entry
            to_date = datetime.now(timezone.utc) + timedelta(minutes=5)  # Look 5 minutes into future

            deals = mt5.history_deals_get(from_date, to_date, symbol=symbol)

            if not deals:
                return None

            # Find exit deal for this position
            for deal in deals:
                # Look for exit deals (entry=1) with matching direction
                if deal.entry == 1:  # DEAL_ENTRY_OUT
                    expected_type = 0 if direction.upper() == 'BUY' else 1  # DEAL_TYPE_BUY = 0, SELL = 1
                    if deal.type == expected_type:
                        reason = self._get_closure_reason(deal.reason)

                        return {
                            'exit_price': deal.price,
                            'profit': deal.profit,
                            'exit_time': datetime.fromtimestamp(deal.time, tz=timezone.utc),
                            'reason': reason,
                            'volume': deal.volume
                        }

        except Exception as e:
            self.logger.error(f"Error getting closure from history: {e}")

        return None

    def _get_closure_reason(self, reason_code: int) -> str:
        """Convert MT5 reason code to readable string"""
        reasons = {
            0: "Manual",
            1: "Expert",
            2: "Expert",  # DEAL_REASON_EXPERT
            3: "SL",      # DEAL_REASON_SL
            4: "TP",      # DEAL_REASON_TP
            5: "SO",      # DEAL_REASON_SO
        }
        return reasons.get(reason_code, f"Unknown({reason_code})")

    def _update_database_closure(self, trade_id: int, exit_price: float,
                                profit: float, exit_time: datetime, reason: str):
        """Update database with trade closure information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE trades
                SET exit_price = ?,
                    profit = ?,
                    exit_time = ?,
                    closure_reason = ?,
                    forced_closure = 0
                WHERE id = ?
            """, (exit_price, profit, exit_time.isoformat(), reason, trade_id))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Error updating database closure: {e}")

    def _should_force_close(self, position, entry_time: str) -> bool:
        """Check if position should be force closed"""
        try:
            # Parse entry time
            entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
            current_time = datetime.now(timezone.utc)

            # Calculate holding time
            holding_hours = (current_time - entry_dt).total_seconds() / 3600

            # Force close if:
            # 1. Held longer than 4 hours
            # 2. Near end of trading day (23:45 UTC)
            current_hour = current_time.hour
            current_minute = current_time.minute

            if holding_hours > 4.0:
                self.logger.warning(f"⚠️ {position.symbol} held for {holding_hours:.1f}h - forcing closure")
                return True

            if current_hour == 23 and current_minute >= 45:
                self.logger.warning(f"⚠️ {position.symbol} - end of day forced closure")
                return True

            # Check for weekend closure (Friday 23:00 UTC)
            if current_time.weekday() == 4 and current_hour >= 23:  # Friday
                self.logger.warning(f"⚠️ {position.symbol} - weekend closure")
                return True

        except Exception as e:
            self.logger.error(f"Error checking force close: {e}")

        return False

    def _force_close_position(self, position) -> bool:
        """Force close a position in MT5"""
        try:
            symbol = position.symbol
            ticket = position.ticket
            volume = position.volume

            # Determine close type (opposite of position type)
            close_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": close_type,
                "position": ticket,
                "deviation": 20,
                "magic": 234000,
                "comment": "Forced closure - overdue",
            }

            result = mt5.order_send(request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"✅ Force closed {symbol} #{ticket} - P&L: ${result.profit:.2f}")

                # Update database immediately
                self._update_database_closure(
                    ticket,  # Assuming ticket matches trade_id
                    result.price,
                    result.profit,
                    datetime.now(timezone.utc),
                    "Forced"
                )
                return True
            else:
                error = mt5.last_error() if result is None else f"Retcode: {result.retcode}"
                self.logger.error(f"❌ Failed to force close {symbol} #{ticket}: {error}")
                return False

        except Exception as e:
            self.logger.error(f"Exception in force close: {e}")
            return False

    async def start_monitoring_loop(self, interval_seconds: int = 60):
        """Start the position monitoring loop"""
        self.logger.info(f"🚀 Starting position sync monitoring (every {interval_seconds}s)")

        while True:
            try:
                results = self.sync_closed_positions()

                if results['synced'] > 0 or results['forced_closures'] > 0:
                    self.logger.info(f"📊 Position Sync: {results['synced']} synced, "
                                   f"{results['forced_closures']} forced closures, "
                                   f"{results['errors']} errors")

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                self.logger.error(f"Position monitoring loop error: {e}")
                await asyncio.sleep(30)  # Shorter wait on error

    def get_position_summary(self) -> Dict:
        """Get summary of current positions"""
        summary = {
            'mt5_positions': 0,
            'db_open_trades': 0,
            'discrepancies': 0,
            'mt5_symbols': [],
            'db_symbols': []
        }

        if not self.mt5_initialized and not self.initialize_mt5():
            return summary

        try:
            # Get MT5 positions
            mt5_positions = mt5.positions_get()
            if mt5_positions:
                summary['mt5_positions'] = len(mt5_positions)
                summary['mt5_symbols'] = list(set([p.symbol for p in mt5_positions]))

            # Get database open trades
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), GROUP_CONCAT(DISTINCT symbol) FROM trades WHERE exit_price IS NULL")
            result = cursor.fetchone()
            summary['db_open_trades'] = result[0] if result[0] else 0
            summary['db_symbols'] = result[1].split(',') if result[1] else []

            conn.close()

            # Calculate discrepancies
            summary['discrepancies'] = abs(summary['mt5_positions'] - summary['db_open_trades'])

        except Exception as e:
            self.logger.error(f"Error getting position summary: {e}")

        return summary

    def cleanup_mt5(self):
        """Cleanup MT5 connection"""
        if self.mt5_initialized:
            mt5.shutdown()
            self.mt5_initialized = False


# Standalone test function
if __name__ == "__main__":
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Test the sync manager
    sync_manager = PositionSyncManager()

    print("🔄 Testing Position Sync Manager...")

    # Get summary
    summary = sync_manager.get_position_summary()
    print(f"MT5 Positions: {summary['mt5_positions']}")
    print(f"DB Open Trades: {summary['db_open_trades']}")
    print(f"Discrepancies: {summary['discrepancies']}")

    # Run sync
    results = sync_manager.sync_closed_positions()
    print(f"Synced: {results['synced']}")
    print(f"Forced Closures: {results['forced_closures']}")
    print(f"Errors: {results['errors']}")

    sync_manager.cleanup_mt5()
    print("✅ Test completed")