"""
Test script for new position management features
Tests trailing stops and dynamic take profit adjustments
"""

import sys
import os
import asyncio
import MetaTrader5 as mt5
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.trading_engine import TradingEngine
from core.mt5_connector import MT5Connector
from core.risk_manager import RiskManager
from utils.config_loader import ConfigLoader

async def test_position_management():
    """Test the new position management features"""
    print("=" * 60)
    print("Testing Position Management Features")
    print("=" * 60)

    # Load configuration
    config_loader = ConfigLoader()
    config_loader.load_config()
    config = config_loader.config

    # Initialize MT5
    mt5_connector = MT5Connector(
        login=config.get('mt5', {}).get('login'),
        password=config.get('mt5', {}).get('password'),
        server=config.get('mt5', {}).get('server'),
        path=config.get('mt5', {}).get('path')
    )

    if not mt5_connector.connect():
        print("❌ MT5 connection failed")
        return

    # Initialize components
    risk_manager = RiskManager(config)
    trading_engine = TradingEngine(mt5_connector, risk_manager, None, None, None, None)

    # Get current positions
    positions = mt5.positions_get()
    if positions is None:
        positions = []

    print(f"📊 Current positions: {len(positions)}")

    if not positions:
        print("ℹ️  No open positions to test with")
        print("💡 To test: Open some positions and run this script")
        mt5.shutdown()
        return

    # Test position management for each symbol
    symbols_tested = set()
    for position in positions:
        if position.symbol not in symbols_tested and position.magic == config.get('trading', {}).get('magic_number', 20241029):
            print(f"\n🔄 Testing position management for {position.symbol}")
            symbols_tested.add(position.symbol)

            try:
                # Test trailing stop update
                print("  📈 Testing trailing stop update...")
                await trading_engine.update_trailing_stop(position)

                # Test take profit update
                print("  🎯 Testing take profit update...")
                await trading_engine.update_take_profit(position)

                # Test full position management
                print("  🔧 Testing full position management...")
                await trading_engine.manage_positions(position.symbol)

                print(f"  ✅ Position management completed for {position.symbol}")

            except Exception as e:
                print(f"  ❌ Error testing {position.symbol}: {e}")

    print("\n" + "=" * 60)
    print("Position Management Test Complete")
    print("=" * 60)

    # Show configuration
    trailing_config = config.get('risk_management', {}).get('trailing_stop', {})
    print("\n📋 Configuration:")
    print(f"  Trailing stops enabled: {trailing_config.get('enabled', False)}")
    print(f"  Activation pips: {trailing_config.get('activation_pips', 20)}")
    print(f"  Trail distance pips: {trailing_config.get('trail_distance_pips', 15)}")

    mt5.shutdown()

if __name__ == "__main__":
    asyncio.run(test_position_management())