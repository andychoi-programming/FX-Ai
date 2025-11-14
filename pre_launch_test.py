#!/usr/bin/env python3
"""
FX-Ai Pre-Launch Validation Script
Run this 30 minutes before Sydney session to ensure everything is ready.
"""

import MetaTrader5 as mt5
from datetime import datetime
import json
import os
import sys

def pre_launch_test():
    """Quick system validation before Sydney session"""
    print("="*60)
    print("FX-Ai Pre-Launch Validation")
    print("="*60)

    validation_passed = True

    try:
        # 1. MT5 Connection
        print("\n🔗 Testing MT5 Connection...")
        if not mt5.initialize():
            print("❌ MT5 connection failed!")
            print(f"   Error: {mt5.last_error()}")
            return False
        print("✅ MT5 connected successfully")

        # 2. Server Time Check
        print("\n🕐 Checking Time Synchronization...")
        try:
            # Get server time from MT5
            server_time = datetime.fromtimestamp(mt5.symbol_info_tick("EURUSD").time)
            local_time = datetime.now()

            time_diff_hours = (local_time - server_time).total_seconds() / 3600
            print(f"✅ MT5 Server Time: {server_time}")
            print(f"   Local Time: {local_time}")
            print(f"   Time Difference: {time_diff_hours:.1f} hours")

            if abs(time_diff_hours) > 3:  # Allow up to 3 hours difference for timezone variations
                print("⚠️  WARNING: Time difference > 3 hours - check system clock!")
                validation_passed = False
            else:
                print("✅ Time synchronization looks good")

        except Exception as e:
            print(f"❌ Failed to get server time: {e}")
            validation_passed = False

        # 3. Config Validation
        print("\n⚙️  Validating Configuration...")
        config_path = 'config/config.json'
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            validation_passed = False
        else:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)

                # Check trading interval
                trading_interval = config.get('trading', {}).get('trading_opportunity_check_interval_seconds')
                if trading_interval == 120:
                    print("✅ Trading check interval: 120 seconds (2 minutes)")
                else:
                    print(f"⚠️  Trading check interval: {trading_interval} seconds (expected: 120)")
                    validation_passed = False

                # Check schedule interval
                schedule_interval = config.get('trading', {}).get('schedule_check_interval_seconds')
                if schedule_interval == 600:
                    print("✅ Schedule check interval: 600 seconds (10 minutes)")
                else:
                    print(f"⚠️  Schedule check interval: {schedule_interval} seconds (expected: 600)")
                    validation_passed = False

                # Check symbols
                symbols = config.get('trading', {}).get('symbols', [])
                expected_symbols = ['AUDUSD', 'NZDUSD', 'AUDJPY', 'NZDJPY', 'AUDNZD', 'USDJPY']
                if symbols == expected_symbols:
                    print(f"✅ Sydney test symbols configured: {symbols}")
                else:
                    print(f"⚠️  Symbols: {symbols} (expected: {expected_symbols})")
                    validation_passed = False

                # Check threshold
                threshold = config.get('trading', {}).get('min_signal_strength')
                if threshold == 0.250:
                    print("✅ Signal threshold: 0.250 (conservative)")
                else:
                    print(f"⚠️  Signal threshold: {threshold} (expected: 0.250)")
                    validation_passed = False

            except Exception as e:
                print(f"❌ Failed to read config: {e}")
                validation_passed = False

        # 4. Account Info
        print("\n💰 Checking Account Status...")
        try:
            account = mt5.account_info()
            if account:
                print(f"✅ Account Balance: ${account.balance:.2f}")
                print(f"✅ Account Equity: ${account.equity:.2f}")
                print(f"✅ Margin Free: ${account.margin_free:.2f}")

                if account.margin_free < 100:
                    print("⚠️  WARNING: Low margin free - consider adding funds")
                    validation_passed = False
            else:
                print("❌ Failed to get account info")
                validation_passed = False
        except Exception as e:
            print(f"❌ Account check failed: {e}")
            validation_passed = False

        # 5. Check for existing orders/positions
        print("\n📊 Checking Existing Orders/Positions...")
        try:
            positions = mt5.positions_get()
            orders = mt5.orders_get()

            positions_count = len(positions) if positions else 0
            orders_count = len(orders) if orders else 0

            print(f"✅ Existing Positions: {positions_count}")
            print(f"✅ Existing Orders: {orders_count}")

            if positions_count > 0:
                print("ℹ️  INFO: Existing positions detected - review in MT5 terminal")
            if orders_count > 0:
                print("ℹ️  INFO: Existing orders detected - review in MT5 terminal")

        except Exception as e:
            print(f"❌ Order/position check failed: {e}")
            validation_passed = False

        # 6. Symbol Availability
        print("\n📈 Checking Symbol Availability...")
        test_symbols = ['AUDUSD', 'NZDUSD', 'AUDJPY', 'NZDJPY', 'AUDNZD', 'USDJPY']

        for symbol in test_symbols:
            try:
                info = mt5.symbol_info(symbol)
                if info is None:
                    print(f"❌ {symbol} not available!")
                    validation_passed = False
                else:
                    spread = getattr(info, 'spread', 'N/A')
                    print(f"✅ {symbol} available (spread: {spread})")
            except Exception as e:
                print(f"❌ {symbol} check failed: {e}")
                validation_passed = False

        # 7. Git Status Check
        print("\n📝 Checking Git Status...")
        try:
            import subprocess
            result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, cwd='.')
            if result.returncode == 0:
                if result.stdout.strip():
                    print("⚠️  WARNING: Uncommitted changes detected")
                    print(f"   Changes: {result.stdout.strip()}")
                    validation_passed = False
                else:
                    print("✅ Working directory clean")
            else:
                print("⚠️  WARNING: Git check failed")
        except Exception as e:
            print(f"⚠️  Git check failed: {e}")

        mt5.shutdown()

    except Exception as e:
        print(f"❌ Pre-launch test failed with error: {e}")
        import traceback
        traceback.print_exc()
        validation_passed = False

    print("\n" + "="*60)
    if validation_passed:
        print("🎉 PRE-LAUNCH VALIDATION PASSED!")
        print("   System is ready for Sydney session testing.")
    else:
        print("⚠️  PRE-LAUNCH VALIDATION ISSUES DETECTED!")
        print("   Please resolve the issues above before starting.")
    print("="*60)

    return validation_passed

if __name__ == "__main__":
    success = pre_launch_test()
    sys.exit(0 if success else 1)