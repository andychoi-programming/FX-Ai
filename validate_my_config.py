#!/usr/bin/env python3
"""
Validate My Configuration
Check for configuration conflicts and issues
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_rr_consistency():
    """Check risk-reward ratio consistency"""
    print("🔍 Checking Risk-Reward Ratio Consistency...")

    try:
        from utils.config_loader import ConfigLoader
        config = ConfigLoader().load_config()

        trading_rules = config.get('trading_rules', {})
        rr_ratios = trading_rules.get('take_profit_rules', {}).get('rr_ratios', {})

        if not rr_ratios:
            print("❌ No RR ratios found in configuration")
            return False

        # Check all symbols have ratios
        symbols = config.get('trading', {}).get('symbols', [])
        missing_ratios = []
        for symbol in symbols:
            if symbol not in rr_ratios:
                missing_ratios.append(symbol)

        if missing_ratios:
            print(f"❌ Missing RR ratios for: {missing_ratios}")
            return False

        # Check ratio ranges are reasonable
        invalid_ratios = []
        for symbol, ratio in rr_ratios.items():
            if not (1.5 <= ratio <= 5.0):
                invalid_ratios.append(f"{symbol}({ratio})")

        if invalid_ratios:
            print(f"⚠️  Unusual RR ratios: {invalid_ratios}")
            print("   Expected range: 1.5:1 to 5.0:1")

        print(f"✅ All {len(rr_ratios)} symbols have RR ratios configured")
        return True

    except Exception as e:
        print(f"❌ RR consistency check failed: {e}")
        return False

def check_schedule_format():
    """Check symbol schedule format"""
    print("\n🔍 Checking Symbol Schedule Format...")

    try:
        # Check if symbol_schedules.json exists
        schedule_file = 'config/symbol_schedules.json'
        if not os.path.exists(schedule_file):
            print("⚠️  No symbol_schedules.json found")
            print("   System will use default 24-hour trading")
            return True

        with open(schedule_file, 'r') as f:
            schedules = json.load(f)

        symbol_schedules = schedules.get('symbol_schedules', {})
        if not symbol_schedules:
            print("⚠️  Empty symbol schedules")
            return True

        # Check format
        first_symbol = list(symbol_schedules.keys())[0]
        first_schedule = symbol_schedules[first_symbol]

        if 'start_hour' in first_schedule and 'end_hour' in first_schedule:
            print("⚠️  Using start_hour/end_hour format")
            print("   This format is supported but optimal_hours array is preferred")
            return True
        elif 'optimal_hours' in first_schedule:
            print("✅ Using optimal_hours array format (preferred)")
            return True
        else:
            print("❌ Unknown schedule format")
            return False

    except Exception as e:
        print(f"❌ Schedule format check failed: {e}")
        return False

def check_24_hour_trading():
    """Check 24-hour trading setup"""
    print("\n🔍 Checking 24-Hour Trading Setup...")

    try:
        from utils.config_loader import ConfigLoader
        config = ConfigLoader().load_config()

        time_restrictions = config.get('trading_rules', {}).get('time_restrictions', {})

        # Check if 24-hour trading is enabled
        day_trading_only = time_restrictions.get('day_trading_only', True)
        if day_trading_only:
            print("✅ Day trading only: True")
        else:
            print("⚠️  24-hour trading enabled")

        # Check force close time
        mt5_times = time_restrictions.get('mt5_trading_times', {})
        force_close = mt5_times.get('mt5_force_close_time', '23:55')
        print(f"✅ Force close time: {force_close} GMT")

        return True

    except Exception as e:
        print(f"❌ 24-hour trading check failed: {e}")
        return False

def check_trading_rules():
    """Check trading rules configuration"""
    print("\n🔍 Checking Trading Rules Configuration...")

    try:
        from utils.config_loader import ConfigLoader
        config = ConfigLoader().load_config()

        trading_rules = config.get('trading_rules', {})

        # Check position limits
        position_limits = trading_rules.get('position_limits', {})
        max_positions = position_limits.get('max_positions', 30)
        max_per_symbol = position_limits.get('max_positions_per_symbol', 1)

        print(f"✅ Max positions: {max_positions}")
        print(f"✅ Max per symbol: {max_per_symbol}")

        # Check risk limits
        risk_limits = trading_rules.get('risk_limits', {})
        risk_per_trade = risk_limits.get('risk_per_trade', 50.0)
        max_daily_loss = risk_limits.get('max_daily_loss', 500.0)

        print(f"✅ Risk per trade: ${risk_per_trade}")
        print(f"✅ Max daily loss: ${max_daily_loss}")

        return True

    except Exception as e:
        print(f"❌ Trading rules check failed: {e}")
        return False

def main():
    """Run configuration validation"""
    print("🔍 VALIDATING YOUR CONFIGURATION")
    print("=" * 50)

    checks = [
        ("Risk-Reward Ratio Consistency", check_rr_consistency),
        ("Symbol Schedule Format", check_schedule_format),
        ("24-Hour Trading Setup", check_24_hour_trading),
        ("Trading Rules Configuration", check_trading_rules)
    ]

    passed = 0
    warnings = 0
    total = len(checks)

    for name, check_func in checks:
        if check_func():
            passed += 1
        else:
            print(f"❌ {name} check failed")

    print("\n" + "=" * 50)

    if passed == total:
        print("✅ No critical issues - Configuration is valid")
        print("   Warnings are acceptable and don't prevent deployment")
        return 0
    else:
        print("❌ Critical configuration issues found")
        print("   Please resolve before deployment")
        return 1

if __name__ == "__main__":
    sys.exit(main())