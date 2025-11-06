#!/usr/bin/env python3
"""
FX-Ai Performance Dashboard
Real-time monitoring of trading performance and system health
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.time_manager import get_time_manager
    from core.mt5_connector import MT5Connector
    import MetaTrader5 as mt5
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("This dashboard requires MT5 connection and project dependencies.")
    sys.exit(1)


class PerformanceDashboard:
    """Real-time performance monitoring dashboard"""

    def __init__(self):
        self.mt5 = None
        self.time_manager = None
        self.last_update = None
        self.cache_duration = 30  # seconds

        # Initialize connections
        self._initialize_connections()

    def _initialize_connections(self):
        """Initialize MT5 and time manager connections"""
        try:
            # Load config
            config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.json')
            with open(config_path, 'r') as f:
                self.config = json.load(f)

            # Initialize MT5
            self.mt5 = MT5Connector(self.config)
            if not self.mt5.connect():
                print("⚠️  MT5 connection failed - some metrics will be unavailable")
                self.mt5 = None

            # Initialize Time Manager
            self.time_manager = get_time_manager(self.mt5)

        except Exception as e:
            print(f"❌ Initialization error: {e}")
            self.mt5 = None
            self.time_manager = None

    def get_system_status(self) -> Dict:
        """Get overall system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'system_health': 'UNKNOWN',
            'mt5_connected': self.mt5 is not None and self.mt5.connected if self.mt5 else False,
            'time_manager_active': self.time_manager is not None
        }

        # Determine system health
        if status['mt5_connected'] and status['time_manager_active']:
            status['system_health'] = 'HEALTHY'
        elif status['mt5_connected'] or status['time_manager_active']:
            status['system_health'] = 'DEGRADED'
        else:
            status['system_health'] = 'CRITICAL'

        return status

    def get_trading_status(self) -> Dict:
        """Get current trading status"""
        if not self.time_manager:
            return {'error': 'TimeManager not available'}

        is_allowed, reason = self.time_manager.is_trading_allowed()
        should_close, close_reason = self.time_manager.should_close_positions()

        return {
            'trading_allowed': is_allowed,
            'trading_reason': reason,
            'should_close_positions': should_close,
            'close_reason': close_reason,
            'time_until_close': str(self.time_manager.get_time_until_close()) if self.time_manager.get_time_until_close() else None,
            'session_info': self.time_manager.get_forex_session_status()
        }

    def get_account_info(self) -> Dict:
        """Get account information and balance"""
        if not self.mt5 or not self.mt5.connected:
            return {'error': 'MT5 not connected'}

        try:
            account_info = self.mt5.get_account_info()  # type: ignore
            if account_info:
                return {
                    'balance': account_info.balance,  # type: ignore
                    'equity': account_info.equity,  # type: ignore
                    'margin': account_info.margin,  # type: ignore
                    'margin_free': account_info.margin_free,  # type: ignore
                    'margin_level': account_info.margin_level,  # type: ignore
                    'profit': account_info.profit,  # type: ignore
                    'leverage': account_info.leverage,  # type: ignore
                    'currency': account_info.currency  # type: ignore
                }
            else:
                return {'error': 'Could not retrieve account info'}
        except Exception as e:
            return {'error': f'Account info error: {e}'}

    def get_positions_info(self) -> Dict:
        """Get information about open positions"""
        if not self.mt5 or not self.mt5.connected:
            return {'error': 'MT5 not connected'}

        try:
            positions = self.mt5.get_positions()  # type: ignore
            if positions is None:
                positions = []

            total_positions = len(positions)
            total_unrealized_pnl = 0
            positions_by_symbol = {}

            for pos in positions:
                symbol = pos.symbol  # type: ignore
                pnl = pos.profit + pos.swap + pos.commission  # type: ignore

                total_unrealized_pnl += pnl

                if symbol not in positions_by_symbol:
                    positions_by_symbol[symbol] = {
                        'count': 0,
                        'total_volume': 0,
                        'total_pnl': 0,
                        'positions': []
                    }

                positions_by_symbol[symbol]['count'] += 1
                positions_by_symbol[symbol]['total_volume'] += pos.volume  # type: ignore
                positions_by_symbol[symbol]['total_pnl'] += pnl
                positions_by_symbol[symbol]['positions'].append({
                    'ticket': pos.ticket,  # type: ignore
                    'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',  # type: ignore
                    'volume': pos.volume,  # type: ignore
                    'price_open': pos.price_open,  # type: ignore
                    'price_current': pos.price_current,  # type: ignore
                    'sl': pos.sl,  # type: ignore
                    'tp': pos.tp,  # type: ignore
                    'pnl': pnl
                })

            return {
                'total_positions': total_positions,
                'total_unrealized_pnl': total_unrealized_pnl,
                'positions_by_symbol': positions_by_symbol,
                'positions_list': [pos.symbol for pos in positions]  # type: ignore
            }
        except Exception as e:
            return {'error': f'Positions info error: {e}'}

    def get_recent_trades(self, hours: int = 24) -> Dict:
        """Get recent trading history"""
        if not self.mt5 or not self.mt5.connected:
            return {'error': 'MT5 not connected'}

        try:
            # Get trades from the last N hours
            from_time = datetime.now() - timedelta(hours=hours)
            to_time = datetime.now()

            deals = mt5.history_deals_get(from_time, to_time)  # type: ignore
            if deals is None:
                deals = []

            # Group by symbol
            trades_by_symbol = {}
            total_trades = len(deals)
            winning_trades = 0
            losing_trades = 0
            total_profit = 0

            for deal in deals:
                if deal.profit != 0:  # Only closed trades  # type: ignore
                    symbol = deal.symbol  # type: ignore
                    profit = deal.profit  # type: ignore

                    total_profit += profit
                    if profit > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1

                    if symbol not in trades_by_symbol:
                        trades_by_symbol[symbol] = {
                            'trades': 0,
                            'wins': 0,
                            'losses': 0,
                            'total_profit': 0,
                            'recent_trades': []
                        }

                    trades_by_symbol[symbol]['trades'] += 1
                    trades_by_symbol[symbol]['total_profit'] += profit
                    if profit > 0:
                        trades_by_symbol[symbol]['wins'] += 1
                    else:
                        trades_by_symbol[symbol]['losses'] += 1

                    # Keep only last 5 trades per symbol
                    if len(trades_by_symbol[symbol]['recent_trades']) < 5:
                        trades_by_symbol[symbol]['recent_trades'].append({
                            'time': deal.time.isoformat() if hasattr(deal.time, 'isoformat') else str(deal.time),  # type: ignore
                            'profit': profit,
                            'volume': deal.volume  # type: ignore
                        })

            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            return {
                'period_hours': hours,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'trades_by_symbol': trades_by_symbol
            }
        except Exception as e:
            return {'error': f'Recent trades error: {e}'}

    def get_risk_metrics(self) -> Dict:
        """Calculate risk metrics"""
        account_info = self.get_account_info()
        positions_info = self.get_positions_info()

        if 'error' in account_info or 'error' in positions_info:
            return {'error': 'Cannot calculate risk metrics - missing account or position data'}

        try:
            balance = account_info.get('balance', 0)
            equity = account_info.get('equity', 0)
            margin = account_info.get('margin', 0)
            margin_free = account_info.get('margin_free', 0)

            # Calculate drawdown
            drawdown_pct = ((balance - equity) / balance * 100) if balance > 0 else 0

            # Calculate margin utilization
            margin_utilization = (margin / equity * 100) if equity > 0 else 0

            # Position risk
            total_positions = positions_info.get('total_positions', 0)
            unrealized_pnl = positions_info.get('total_unrealized_pnl', 0)

            return {
                'account_balance': balance,
                'account_equity': equity,
                'current_drawdown_pct': drawdown_pct,
                'margin_utilization_pct': margin_utilization,
                'margin_free': margin_free,
                'total_open_positions': total_positions,
                'unrealized_pnl': unrealized_pnl,
                'risk_status': self._assess_risk_level(drawdown_pct, margin_utilization, total_positions)
            }
        except Exception as e:
            return {'error': f'Risk metrics calculation error: {e}'}

    def _assess_risk_level(self, drawdown_pct: float, margin_util_pct: float, position_count: int) -> str:
        """Assess overall risk level"""
        risk_score = 0

        # Drawdown risk
        if drawdown_pct > 10:
            risk_score += 3
        elif drawdown_pct > 5:
            risk_score += 2
        elif drawdown_pct > 2:
            risk_score += 1

        # Margin utilization risk
        if margin_util_pct > 80:
            risk_score += 3
        elif margin_util_pct > 60:
            risk_score += 2
        elif margin_util_pct > 40:
            risk_score += 1

        # Position count risk
        if position_count > 20:
            risk_score += 3
        elif position_count > 10:
            risk_score += 2
        elif position_count > 5:
            risk_score += 1

        # Determine risk level
        if risk_score >= 6:
            return 'HIGH'
        elif risk_score >= 3:
            return 'MEDIUM'
        else:
            return 'LOW'

    def display_dashboard(self):
        """Display the complete dashboard"""
        print("\n" + "=" * 80)
        print("📊 FX-AI PERFORMANCE DASHBOARD")
        print("=" * 80)
        print(f"⏰ Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # System Status
        system_status = self.get_system_status()
        print(f"\n🔧 SYSTEM STATUS: {system_status['system_health']}")
        print(f"   MT5 Connected: {system_status['mt5_connected']}")
        print(f"   Time Manager: {system_status['time_manager_active']}")

        # Trading Status
        trading_status = self.get_trading_status()
        if 'error' not in trading_status:
            print("\n🕐 TRADING STATUS")
            print(f"   Trading Allowed: {trading_status['trading_allowed']}")
            print(f"   Reason: {trading_status['trading_reason']}")
            print(f"   Should Close: {trading_status['should_close_positions']}")
            if trading_status['time_until_close']:
                print(f"   Time Until Close: {trading_status['time_until_close']}")

        # Account Info
        account_info = self.get_account_info()
        if 'error' not in account_info:
            print("\n💰 ACCOUNT INFO")
            print(f"   Balance: ${account_info.get('balance', 0):,.2f}")
            print(f"   Equity: ${account_info.get('equity', 0):,.2f}")
            print(f"   Margin Used: ${account_info.get('margin', 0):,.2f}")
            print(f"   Margin Level: {account_info.get('margin_level', 0):.1f}%")

        # Positions Info
        positions_info = self.get_positions_info()
        if 'error' not in positions_info:
            print("\n📈 OPEN POSITIONS")
            print(f"   Total Positions: {positions_info['total_positions']}")
            print(f"   Unrealized P&L: ${positions_info.get('total_unrealized_pnl', 0):,.2f}")

            if positions_info['positions_by_symbol']:
                print("   By Symbol:")
                for symbol, data in positions_info['positions_by_symbol'].items():
                    print(f"     {symbol}: {data['count']} positions, "
                          ",.2f")

        # Recent Performance
        recent_trades = self.get_recent_trades(hours=24)
        if 'error' not in recent_trades:
            print("\n📊 RECENT PERFORMANCE (24h)")
            print(f"   Total Trades: {recent_trades['total_trades']}")
            print(".1f")
            print(f"   Total Profit: ${recent_trades.get('total_profit', 0):,.2f}")

        # Risk Metrics
        risk_metrics = self.get_risk_metrics()
        if 'error' not in risk_metrics:
            print("\n⚠️  RISK METRICS")
            print(f"   Current Drawdown: {risk_metrics.get('current_drawdown_pct', 0):.2f}%")
            print(f"   Margin Utilization: {risk_metrics.get('margin_utilization_pct', 0):.1f}%")
            print(f"   Risk Level: {risk_metrics['risk_status']}")

        print("\n" + "=" * 80)

    def run_continuous_dashboard(self, interval_seconds: int = 60):
        """Run dashboard continuously"""
        print("Starting continuous dashboard monitoring...")
        print("Press Ctrl+C to stop")

        try:
            while True:
                self.display_dashboard()
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\nDashboard stopped by user")


def main():
    """Main dashboard function"""
    dashboard = PerformanceDashboard()

    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        dashboard.run_continuous_dashboard()
    else:
        dashboard.display_dashboard()


if __name__ == "__main__":
    main()
