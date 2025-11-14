"""
Test position sizing calculations across symbol types
"""
import sys
sys.path.insert(0, '.')

from core.risk_manager import RiskManager
from utils.config_loader import ConfigLoader

def test_position_sizing():
    cl = ConfigLoader()
    cl.load_config()
    config = cl.config
    rm = RiskManager(config)
    
    # Risk amount from config (should be $50)
    risk_amount = config.get('risk_management', {}).get('risk_per_trade', 50)
    
    test_cases = [
        {'symbol': 'EURUSD', 'sl_pips': 20, 'price': 1.10000},
        {'symbol': 'GBPUSD', 'sl_pips': 20, 'price': 1.27000},
        {'symbol': 'USDJPY', 'sl_pips': 20, 'price': 149.500},
        {'symbol': 'XAUUSD', 'sl_pips': 20, 'price': 2000.00},
        {'symbol': 'XAGUSD', 'sl_pips': 100, 'price': 25.00},
    ]
    
    print("=" * 60)
    print("POSITION SIZING VALIDATION")
    print(f"Risk per trade: ${risk_amount}")
    print("=" * 60)
    
    for test in test_cases:
        print(f"\n{test['symbol']}:")
        print(f"  Current Price: {test['price']}")
        print(f"  SL Distance: {test['sl_pips']} pips")
        
        try:
            lot_size = rm.calculate_position_size(
                test['symbol'],
                test['sl_pips']
            )
            
            print(f"  Calculated Lot Size: {lot_size}")
            
            # Validate range
            if 0.01 <= lot_size <= 1.0:
                print(f"  ✅ Within valid range (0.01-1.0)")
            else:
                print(f"  ❌ OUT OF RANGE!")
                
            # Calculate actual risk (rough estimate)
            # For standard lot: 1 pip = $10 for EURUSD
            # For mini lot (0.01): 1 pip = $0.10
            pip_value_per_lot = 10  # Simplified
            actual_risk = lot_size * test['sl_pips'] * pip_value_per_lot
            
            print(f"  Estimated Risk: ${actual_risk:.2f}")
            
            if abs(actual_risk - risk_amount) / risk_amount > 0.2:  # >20% off
                print(f"  ⚠️  Risk calculation may be off target")
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")

if __name__ == "__main__":
    test_position_sizing()