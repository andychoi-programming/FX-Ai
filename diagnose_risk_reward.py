"""
Diagnostic script to find risk-reward calculation bug
"""
import sys
sys.path.insert(0, '.')

from core.risk_manager import RiskManager
from utils.config_loader import ConfigLoader

def test_risk_reward():
    cl = ConfigLoader()
    cl.load_config()
    config = cl.config
    rm = RiskManager(config)
    
    print("Config trading section:", config.get('trading', {}))
    print("Default SL pips:", config.get('trading', {}).get('default_sl_pips'))
    print("Default TP pips:", config.get('trading', {}).get('default_tp_pips'))
    print()
    
    print("=" * 60)
    print("TESTING RISK-REWARD CALCULATIONS")
    print("=" * 60)
    
    # Test Case 1: EURUSD BUY
    test_cases = [
        {
            'symbol': 'EURUSD',
            'direction': 'BUY',
            'entry': 1.10000,
            'sl_pips': 20,
            'tp_pips': 60,
            'expected_ratio': 3.0
        },
        {
            'symbol': 'EURUSD',
            'direction': 'SELL',
            'entry': 1.10000,
            'sl_pips': 20,
            'tp_pips': 60,
            'expected_ratio': 3.0
        },
        {
            'symbol': 'USDJPY',
            'direction': 'BUY',
            'entry': 149.500,
            'sl_pips': 20,
            'tp_pips': 60,
            'expected_ratio': 3.0
        },
        {
            'symbol': 'XAUUSD',
            'direction': 'BUY',
            'entry': 2000.00,
            'sl_pips': 20,
            'tp_pips': 60,
            'expected_ratio': 3.0
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test['symbol']} {test['direction']} ---")
        print(f"Entry: {test['entry']}")
        print(f"SL Pips: {test['sl_pips']}, TP Pips: {test['tp_pips']}")
        
        try:
            print(f"Calling calculate_stop_loss_take_profit with symbol={test['symbol']}, entry={test['entry']}, direction={test['direction']}")
            # Calculate SL/TP prices
            result = rm.calculate_stop_loss_take_profit(
                test['symbol'],
                test['entry'],
                test['direction']
            )
            
            sl_price = result['stop_loss']
            tp_price = result['take_profit']
            
            print(f"SL Price: {sl_price}")
            print(f"TP Price: {tp_price}")
            
            # Calculate actual risk-reward
            if test['direction'] == 'BUY':
                risk_pips = abs(test['entry'] - sl_price) / rm._get_pip_value(test['symbol'])
                reward_pips = abs(tp_price - test['entry']) / rm._get_pip_value(test['symbol'])
            else:  # SELL
                risk_pips = abs(sl_price - test['entry']) / rm._get_pip_value(test['symbol'])
                reward_pips = abs(test['entry'] - tp_price) / rm._get_pip_value(test['symbol'])
            
            actual_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
            
            print(f"Risk (pips): {risk_pips:.2f}")
            print(f"Reward (pips): {reward_pips:.2f}")
            print(f"Actual Ratio: {actual_ratio:.2f}:1")
            print(f"Expected Ratio: {test['expected_ratio']:.2f}:1")
            
            # Validate
            if abs(actual_ratio - test['expected_ratio']) < 0.01:
                print("✅ PASS")
                passed += 1
            else:
                print(f"❌ FAIL - Expected {test['expected_ratio']:.2f}, got {actual_ratio:.2f}")
                failed += 1
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

if __name__ == "__main__":
    test_risk_reward()