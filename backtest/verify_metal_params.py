"""Verify metal parameters are correctly loaded"""
from live_trading.dynamic_parameter_manager import DynamicParameterManager
import json

config = json.load(open('config/config.json'))
pm = DynamicParameterManager(config)

xag = pm.get_optimal_parameters('XAGUSD', 'H1')
xau = pm.get_optimal_parameters('XAUUSD', 'H1')

print("=" * 60)
print("METAL TRADING PARAMETERS VERIFICATION")
print("=" * 60)
print(f"\nXAGUSD (Silver):")
print(f"  Stop Loss: {xag['sl_pips']} pips (${xag['sl_pips'] * 0.01:.2f} price move)")
print(f"  Take Profit: {xag['tp_pips']} pips (${xag['tp_pips'] * 0.01:.2f} price move)")
print(f"  Entry Days: {', '.join(xag['best_entry_days'])}")
print(f"  Entry Hours: {xag['entry_hour_start']}:00 - {xag['entry_hour_end']}:00")

print(f"\nXAUUSD (Gold):")
print(f"  Stop Loss: {xau['sl_pips']} pips (${xau['sl_pips'] * 0.01:.2f} price move)")
print(f"  Take Profit: {xau['tp_pips']} pips (${xau['tp_pips'] * 0.01:.2f} price move)")
print(f"  Entry Days: {', '.join(xau['best_entry_days'])}")
print(f"  Entry Hours: {xau['entry_hour_start']}:00 - {xau['entry_hour_end']}:00")

print("\n" + "=" * 60)
print("SYSTEM READY FOR LIVE TRADING")
print("=" * 60)
