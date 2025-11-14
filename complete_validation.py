"""
Complete system validation after fixes
"""
import sys
sys.path.insert(0, '.')

def run_all_tests():
    print("🔍 RUNNING COMPLETE VALIDATION SUITE\n")
    
    tests = []
    
    # Test 1: Imports
    print("Test 1: Import validation...")
    try:
        from core.risk_manager import RiskManager
        from ai.reinforcement_learning_agent import RLAgent
        from app.application import FXAiApplication
        print("✅ All imports successful\n")
        tests.append(True)
    except Exception as e:
        print(f"❌ Import failed: {e}\n")
        tests.append(False)
    
    # Test 2: Risk-Reward Calculations
    print("Test 2: Risk-reward validation...")
    try:
        from utils.config_loader import ConfigLoader
        cl = ConfigLoader()
        cl.load_config()
        config = cl.config
        rm = RiskManager(config)
        
        # Test EURUSD BUY
        result = rm.calculate_stop_loss_take_profit('EURUSD', 1.10000, 'BUY')
        sl_price = result['stop_loss']
        tp_price = result['take_profit']
        risk_pips = abs(1.10000 - sl_price) / 0.0001
        reward_pips = abs(tp_price - 1.10000) / 0.0001
        ratio = reward_pips / risk_pips
        
        if abs(ratio - 3.0) < 0.01:
            print(f"✅ Risk-reward ratio correct: {ratio:.2f}:1\n")
            tests.append(True)
        else:
            print(f"❌ Risk-reward ratio wrong: {ratio:.2f}:1 (expected 3.0:1)\n")
            tests.append(False)
    except Exception as e:
        print(f"❌ Risk-reward test failed: {e}\n")
        tests.append(False)
    
    # Test 3: Position Sizing
    print("Test 3: Position sizing validation...")
    try:
        lot_size = rm.calculate_position_size('EURUSD', 20)
        if 0.01 <= lot_size <= 1.0:
            print(f"✅ Position size valid: {lot_size} lots\n")
            tests.append(True)
        else:
            print(f"❌ Position size out of range: {lot_size}\n")
            tests.append(False)
    except Exception as e:
        print(f"❌ Position sizing test failed: {e}\n")
        tests.append(False)
    
    # Test 4: MT5 Connection
    print("Test 4: MT5 connection...")
    try:
        from core.mt5_connector import MT5Connector
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        mt5 = MT5Connector(
            os.getenv('MT5_LOGIN'),
            os.getenv('MT5_PASSWORD'),
            os.getenv('MT5_SERVER')
        )
        
        if mt5.connect():
            print("✅ MT5 connection successful\n")
            tests.append(True)
            # No shutdown method, perhaps mt5.shutdown()
        else:
            print("❌ MT5 connection failed\n")
            tests.append(False)
    except Exception as e:
        print(f"❌ MT5 test failed: {e}\n")
        tests.append(False)
    
    # Summary
    passed = sum(tests)
    total = len(tests)
    
    print("=" * 60)
    print(f"VALIDATION COMPLETE: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED - System ready for demo trading")
        print("⚠️  Still recommend 24-48 hours of demo monitoring before live")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - DO NOT resume trading")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())