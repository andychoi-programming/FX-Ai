#!/usr/bin/env python3
"""
Quick Schedule Check
Check if symbol schedules are working with the code
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_schedule_compatibility():
    print("Checking if schedule format is compatible with code...\n")

    try:
        from utils.config_loader import ConfigLoader

        # Load config
        config_loader = ConfigLoader()
        config = config_loader.load_config()

        # Check if symbol_schedules.json exists
        schedule_file = 'config/symbol_schedules.json'
        if not os.path.exists(schedule_file):
            print("⚠️  No symbol_schedules.json found")
            print("   System will use 24-hour trading for all symbols")
            return True

        # Load schedules
        with open(schedule_file, 'r') as f:
            schedules = json.load(f)

        symbol_schedules = schedules.get('symbol_schedules', {})

        if not symbol_schedules:
            print("⚠️  Empty symbol schedules")
            return True

        # Check format
        first_symbol = list(symbol_schedules.keys())[0]
        first_schedule = symbol_schedules[first_symbol]

        print(f"Schedule format detected for {first_symbol}:")
        print(f"  Has start_hour: {'start_hour' in first_schedule}")
        print(f"  Has end_hour: {'end_hour' in first_schedule}")
        print(f"  Has optimal_hours: {'optimal_hours' in first_schedule}")

        # Try to use in a simple way
        if 'start_hour' in first_schedule and 'end_hour' in first_schedule:
            print("\n✅ Using start/end hour format")
            print(f"   Example: {first_symbol} trades {first_schedule['start_hour']}:00 - {first_schedule['end_hour']}:00")
            print("\n⚠️  Ensure your code can handle this format!")
            print("   Check: utils/time_manager.py or wherever schedule is used")
            return True
        elif 'optimal_hours' in first_schedule:
            print("\n✅ Using optimal_hours array format")
            print("   This is the preferred format")
            return True
        else:
            print("\n❌ Unknown schedule format")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    check_schedule_compatibility()