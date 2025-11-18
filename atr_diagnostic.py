"""
ATR Diagnostic Script for Metals Trading Issue
Analyzes ATR calculations for XAUUSD and XAGUSD to identify the root cause
"""

import MetaTrader5 as mt5
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_atr_for_metals():
    """Analyze ATR values for metals to understand the scaling issue"""

    if not mt5.initialize():
        logger.error("Failed to initialize MT5")
        return

    symbols = ['XAUUSD', 'XAGUSD', 'EURUSD', 'GBPUSD']

    print("🔍 ATR DIAGNOSTIC ANALYSIS FOR METALS")
    print("=" * 60)

    for symbol in symbols:
        print(f"\n📊 {symbol} Analysis:")
        print("-" * 30)

        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            print(f"❌ Cannot get symbol info for {symbol}")
            continue

        # Get recent rates for ATR calculation
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 14)
        if rates is None or len(rates) < 14:
            print(f"❌ Cannot get rates for {symbol}")
            continue

        # Calculate ATR manually (simple version)
        high_low = rates['high'] - rates['low']
        atr_value = np.mean(high_low[-14:])  # Simple ATR

        current_price = rates[-1]['close']
        pip_size = symbol_info.point

        print(f"  Current Price: ${current_price:.2f}")
        print(f"  Raw ATR Value: ${atr_value:.5f}")
        print(f"  ATR as % of price: {(atr_value/current_price*100):.3f}%")
        print(f"  Pip Size: ${pip_size:.5f}")
        print(f"  ATR in 'pips': {atr_value/pip_size:.1f}")

        # Show what current multipliers would produce
        if symbol in ['XAUUSD', 'XAGUSD']:
            sl_multiplier = 1.5  # From config
            tp_multiplier = 4.0  # From config
        else:
            sl_multiplier = 2.0  # Forex default
            tp_multiplier = 6.0  # Forex default

        sl_distance = atr_value * sl_multiplier
        tp_distance = atr_value * tp_multiplier

        sl_pips = sl_distance / pip_size
        tp_pips = tp_distance / pip_size

        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0

        print(f"  SL Multiplier: {sl_multiplier}")
        print(f"  TP Multiplier: {tp_multiplier}")
        print(f"  SL Distance: ${sl_distance:.5f} ({sl_pips:.1f} pips)")
        print(f"  TP Distance: ${tp_distance:.5f} ({tp_pips:.1f} pips)")
        print(f"  Risk-Reward Ratio: {rr_ratio:.2f}:1")

        # Check required RR ratios from config
        required_rr = 2.0 if symbol == 'XAUUSD' else (1.5 if symbol == 'XAGUSD' else 2.5)
        status = "✅ PASS" if rr_ratio >= required_rr else "❌ FAIL"
        print(f"  Required RR: {required_rr}:1 - {status}")

        # Show percentage-based alternative
        atr_percentage = atr_value / current_price
        sl_percentage = atr_percentage * sl_multiplier
        tp_percentage = atr_percentage * tp_multiplier

        sl_distance_pct = current_price * sl_percentage
        tp_distance_pct = current_price * tp_percentage

        rr_ratio_pct = tp_percentage / sl_percentage if sl_percentage > 0 else 0

        print(f"  📈 Percentage-based alternative:")
        print(f"    SL Distance: ${sl_distance_pct:.5f} ({sl_distance_pct/pip_size:.1f} pips)")
        print(f"    TP Distance: ${tp_distance_pct:.5f} ({tp_distance_pct/pip_size:.1f} pips)")
        print(f"    RR Ratio: {rr_ratio_pct:.2f}:1")

    mt5.shutdown()

if __name__ == "__main__":
    analyze_atr_for_metals()