"""
Position Monitor - Alerts for SL/TP Changes
Monitors positions and alerts when SL/TP values change unexpectedly
"""
import MetaTrader5 as mt5
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List

logger = logging.getLogger(__name__)

class PositionMonitor:
    def __init__(self, magic_number: int):
        self.magic_number = magic_number
        self.last_positions: Dict[int, Dict] = {}  # ticket -> position data
        self.alerts_enabled = True

    def enable_alerts(self, enabled: bool = True):
        self.alerts_enabled = enabled

    async def check_positions(self) -> List[str]:
        """Check all positions for unexpected changes, return list of alerts"""
        alerts = []

        try:
            positions = mt5.positions_get()  # type: ignore
            if not positions:
                return alerts

            current_positions = {}
            for pos in positions:
                if pos.magic == self.magic_number:
                    current_positions[pos.ticket] = {
                        'symbol': pos.symbol,
                        'sl': pos.sl,
                        'tp': pos.tp,
                        'price_open': pos.price_open,
                        'type': pos.type,
                        'volume': pos.volume,
                        'time': pos.time,
                        'time_update': pos.time_update
                    }

            # Check for changes in existing positions
            for ticket, current_data in current_positions.items():
                if ticket in self.last_positions:
                    previous_data = self.last_positions[ticket]
                    changes = self._detect_changes(previous_data, current_data)

                    if changes:
                        alert_msg = f"Position {ticket} ({current_data['symbol']}) changed: {', '.join(changes)}"
                        alerts.append(alert_msg)

                        if self.alerts_enabled:
                            logger.warning(f"[CRITICAL] POSITION ALERT: {alert_msg}")

            # Check for new positions
            new_tickets = set(current_positions.keys()) - set(self.last_positions.keys())
            for ticket in new_tickets:
                data = current_positions[ticket]
                alert_msg = f"New position opened: {ticket} ({data['symbol']}) SL={data['sl']:.5f}, TP={data['tp']:.5f}"
                alerts.append(alert_msg)

                if self.alerts_enabled:
                    logger.info(f"[NEW POSITION] {alert_msg}")

            # Update last positions
            self.last_positions = current_positions.copy()

        except Exception as e:
            logger.error(f"Error checking positions: {e}")

        return alerts

    def _detect_changes(self, previous: Dict, current: Dict) -> List[str]:
        """Detect significant changes between position states"""
        changes = []

        # Check SL changes
        if previous['sl'] != current['sl']:
            sl_change = current['sl'] - previous['sl']
            if abs(sl_change) > 0.001:  # Significant change
                direction = "tightened" if abs(sl_change) < abs(previous['sl'] - current['price_open']) else "loosened"
                changes.append(f"SL {direction}: {previous['sl']:.5f} → {current['sl']:.5f}")

        # Check TP changes
        if previous['tp'] != current['tp']:
            tp_change = current['tp'] - previous['tp']
            if abs(tp_change) > 0.001:  # Significant change
                changes.append(f"TP changed: {previous['tp']:.5f} → {current['tp']:.5f}")

        # Check for suspicious SL changes (very tight stops)
        if current['sl'] > 0:
            symbol_info = mt5.symbol_info(current['symbol'])  # type: ignore
            if symbol_info:
                point = symbol_info.point
                pip_size = point * 10  # 1 pip = 10 points for all pairs

                # Calculate risk in pips
                if current['type'] == mt5.ORDER_TYPE_BUY:
                    risk_pips = abs(current['sl'] - current['price_open']) / pip_size
                else:
                    risk_pips = abs(current['price_open'] - current['sl']) / pip_size

                if risk_pips < 10:  # Very tight stop
                    changes.append(f"[WARNING] VERY TIGHT SL: {risk_pips:.1f} pips")

        return changes

    async def monitor_loop(self, interval_seconds: int = 30):
        """Continuous monitoring loop"""
        logger.info(f"Starting position monitor (interval: {interval_seconds}s)")

        while True:
            alerts = await self.check_positions()
            if alerts:
                logger.warning(f"\n{'='*50}")
                logger.warning(f"POSITION MONITOR ALERTS ({datetime.now()})")
                for alert in alerts:
                    logger.warning(f"• {alert}")
                logger.warning(f"{'='*50}")

            await asyncio.sleep(interval_seconds)