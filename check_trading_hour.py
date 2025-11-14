"""Check optimal trading hour status"""

import MetaTrader5 as mt5
from utils.time_manager import TimeManager

mt5.initialize()

time_manager = TimeManager()
current_time = time_manager.get_current_time()
current_hour = current_time.hour

print(f"🔍 OPTIMAL TRADING HOUR CHECK")
print(f"Current MT5 time: {current_time}")
print(f"Current hour: {current_hour}")

# Check if current hour is optimal
# We need to load config to check hourly weights
import json
with open('config/config.json', 'r') as f:
    config = json.load(f)

is_optimal = time_manager.is_optimal_trading_hour(config)
hourly_weight = time_manager.get_hourly_performance_weight(config)

print(f"Is optimal trading hour: {is_optimal}")
print(f"Hourly performance weight: {hourly_weight}")

# Check the hourly weights config
adaptive_config = config.get('adaptive_learning', {}).get('session_time_optimization', {})
hourly_weights = adaptive_config.get('hourly_weights', {})

print(f"\nHourly weights configured: {len(hourly_weights)} hours")
if hourly_weights:
    print("Sample weights:")
    for hour in sorted(hourly_weights.keys())[:5]:  # Show first 5
        print(f"  Hour {hour}: {hourly_weights[hour]}")

    # Calculate average
    avg_weight = sum(hourly_weights.values()) / len(hourly_weights)
    threshold = avg_weight * 0.9
    current_weight = hourly_weights.get(str(current_hour), 1.0)

    print(f"\nCurrent hour {current_hour} weight: {current_weight}")
    print(f"Average weight: {avg_weight:.3f}")
    print(f"90% threshold: {threshold:.3f}")
    print(f"Current >= threshold: {current_weight >= threshold}")

mt5.shutdown()