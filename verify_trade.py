#!/usr/bin/env python3
"""
Verify first trade execution and risk management
"""
import sys
sys.path.insert(0, '.')

from core.mt5_connector import MT5Connector
import os
from dotenv import load_dotenv
import MetaTrader5 as mt5

def verify_trade():
    print("🔍 TRADE VERIFICATION")
    print("=" * 40)

    load_dotenv()

    mt5_conn = MT5Connector(
        os.getenv('MT5_LOGIN'),
        os.getenv('MT5_PASSWORD'),
        os.getenv('MT5_SERVER')
    )

    if not mt5_conn.connect():
        print("❌ Cannot connect to MT5 for verification")
        return

    try:
        # Get positions
        positions = mt5.positions_get()
        if not positions:
            print("No open positions found")
            print("Waiting for first trade to execute...")
            return

        position = positions[0]  # Get first position
        symbol = position.symbol
        entry_price = position.price_open
        sl = position.sl
        tp = position.tp
        lot_size = position.volume

        print(f"\nSymbol: {symbol}")
        print(f"Type: {'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'}")
        print(f"Entry: {entry_price:.5f}")
        print(f"SL: {sl:.5f} ({abs(entry_price - sl)*10000:.0f} pips)" if sl > 0 else "SL: Not set")
        print(f"TP: {tp:.5f} ({abs(tp - entry_price)*10000:.0f} pips)" if tp > 0 else "TP: Not set")
        print(f"Lot Size: {lot_size}")

        # Calculate risk-reward ratio
        if sl > 0 and tp > 0:
            risk = abs(entry_price - sl)
            reward = abs(tp - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            print(f"Risk-Reward Ratio: {rr_ratio:.2f}:1")

            # Get minimum required RR for this symbol
            from core.risk_manager import RiskManager
            from utils.config_loader import ConfigLoader

            config = ConfigLoader().load_config()
            rm = RiskManager(config)
            min_rr = rm._get_symbol_min_rr(symbol)

            print(f"\nVALIDATION RESULTS:")
            print("=" * 40)

            # Validate RR ratio
            if rr_ratio >= min_rr:
                print(f"✅ Risk-Reward: {rr_ratio:.2f}:1 (≥ {min_rr})")
            else:
                print(f"❌ Risk-Reward: {rr_ratio:.2f}:1 (< {min_rr}) - FAILED!")

            # Validate lot size
            if 0.01 <= lot_size <= 1.0:
                print(f"✅ Lot Size: {lot_size} (within 0.01-1.0)")
            else:
                print(f"❌ Lot Size: {lot_size} (outside 0.01-1.0)")

            # Validate SL/TP
            if sl > 0 and tp > 0:
                print("✅ SL/TP: Both set")
            else:
                print("❌ SL/TP: Missing stop loss or take profit")

            if rr_ratio >= min_rr and 0.01 <= lot_size <= 1.0 and sl > 0 and tp > 0:
                print("\n✅ TRADE VALIDATED - System working correctly")
            else:
                print("\n❌ TRADE FAILED VALIDATION - Check system configuration")
        else:
            print("Cannot calculate R:R ratio - missing SL or TP")

    finally:
        mt5_conn.disconnect()

if __name__ == "__main__":
    verify_trade()