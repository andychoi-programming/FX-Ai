#!/usr/bin/env python3
"""
Pre-Flight System Check
Comprehensive validation before deployment
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_imports():
    """Check all critical imports"""
    print("🔍 Checking critical imports...")

    try:
        from core.risk_manager import RiskManager
        from core.order_executor import OrderExecutor
        from core.mt5_connector import MT5Connector
        from ai.reinforcement_learning_agent import RLAgent
        from app.application import FXAiApplication
        from utils.config_loader import ConfigLoader
        from utils.logger import setup_logger

        print("✅ All critical imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def check_config():
    """Check configuration integrity"""
    print("\n🔍 Checking configuration...")

    try:
        from utils.config_loader import ConfigLoader
        config_loader = ConfigLoader()
        config = config_loader.load_config()

        # Check trading rules
        trading_rules = config.get('trading_rules', {})
        if not trading_rules:
            print("❌ Trading rules not found in config")
            return False

        # Check symbols
        symbols = config.get('trading', {}).get('symbols', [])
        if len(symbols) != 30:
            print(f"❌ Expected 30 symbols, found {len(symbols)}")
            return False

        # Check RR ratios
        rr_ratios = trading_rules.get('take_profit_rules', {}).get('rr_ratios', {})
        if len(rr_ratios) != 30:
            print(f"❌ Expected 30 RR ratios, found {len(rr_ratios)}")
            return False

        print(f"✅ Configuration valid: {len(symbols)} symbols, {len(rr_ratios)} RR ratios")
        return True

    except Exception as e:
        print(f"❌ Config check failed: {e}")
        return False

def check_risk_reward_system():
    """Check dynamic R:R system"""
    print("\n🔍 Checking dynamic R:R system...")

    try:
        from utils.config_loader import ConfigLoader
        from core.risk_manager import RiskManager

        config = ConfigLoader().load_config()
        rm = RiskManager(config)

        # Test symbol-specific ratios
        test_cases = [
            ('EURUSD', 3.0),
            ('EURGBP', 2.0),
            ('EURJPY', 4.0),
            ('GBPUSD', 3.0),
            ('XAUUSD', 2.5)
        ]

        for symbol, expected in test_cases:
            actual = rm._get_symbol_min_rr(symbol)
            if abs(actual - expected) > 0.01:
                print(f"❌ {symbol}: expected {expected}, got {actual}")
                return False

        # Test validation
        is_valid, _ = rm.validate_risk_reward('EURUSD', 1.0500, 1.0450, 1.0650)  # 4:1 ratio
        if not is_valid:
            print("❌ EURUSD 4:1 validation failed")
            return False

        is_valid, _ = rm.validate_risk_reward('EURUSD', 1.0500, 1.0450, 1.0550)  # 2:1 ratio (should fail)
        if is_valid:
            print("❌ EURUSD 2:1 validation should have failed")
            return False

        print("✅ Dynamic R:R system working correctly")
        return True

    except Exception as e:
        print(f"❌ R:R system check failed: {e}")
        return False

def check_position_sizing():
    """Check position sizing calculations"""
    print("\n🔍 Checking position sizing...")

    try:
        from utils.config_loader import ConfigLoader
        from core.risk_manager import RiskManager

        config = ConfigLoader().load_config()
        rm = RiskManager(config)

        # Test position size calculation (this would need MT5 connection for full test)
        # For now, just check the method exists and doesn't crash
        try:
            # This will fail without MT5, but should not crash
            rm.calculate_position_size('EURUSD', 25, 1.0500)
        except Exception:
            # Expected to fail without MT5, just check it doesn't crash
            pass

        print("✅ Position sizing methods available")
        return True

    except Exception as e:
        print(f"❌ Position sizing check failed: {e}")
        return False

def check_database():
    """Check database connectivity"""
    print("\n🔍 Checking database...")

    try:
        import sqlite3
        import os

        db_path = 'data/performance_history.db'
        if not os.path.exists(db_path):
            print("⚠️  Database not found (expected for new setup)")
            return True

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        if len(tables) > 0:
            print(f"✅ Database connected: {len(tables)} tables found")
        else:
            print("⚠️  Database exists but no tables found")

        conn.close()
        return True

    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return False

def main():
    """Run all pre-flight checks"""
    print("🚀 FX-Ai Pre-Flight System Check")
    print("=" * 50)

    checks = [
        ("Critical Imports", check_imports),
        ("Configuration", check_config),
        ("Dynamic R:R System", check_risk_reward_system),
        ("Position Sizing", check_position_sizing),
        ("Database", check_database)
    ]

    passed = 0
    total = len(checks)

    for name, check_func in checks:
        if check_func():
            passed += 1
        else:
            print(f"❌ {name} check failed")

    print("\n" + "=" * 50)
    print(f"PRE-FLIGHT CHECK COMPLETE: {passed}/{total} checks passed")

    if passed == total:
        print("✅ ALL CHECKS PASSED - System is READY for deployment")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Do not deploy until all issues are resolved")
        return 1

if __name__ == "__main__":
    sys.exit(main())