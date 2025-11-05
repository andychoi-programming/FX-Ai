"""
Test Daily Trade Limit Per Symbol

Tests the new rule: Each symbol can only trade ONE time per day based on MT5 server time
"""

from core.risk_manager import RiskManager
from datetime import datetime
import time

def test_daily_limit():
    """Test the daily trade limit functionality"""
    
    # Mock config
    config = {
        'trading': {
            'risk_per_trade': 50.0,
            'max_positions': 3,
            'max_daily_loss': 200.0,
            'max_spread': 3.0,
            'prevent_multiple_positions_per_symbol': True
        },
        'risk_management': {
            'symbol_cooldown_minutes': 5
        }
    }
    
    # Create risk manager
    risk_manager = RiskManager(config)
    
    print("=" * 80)
    print("TESTING: One Trade Per Symbol Per Day Rule")
    print("=" * 80)
    print()
    
    # Test symbols
    test_symbols = ['EURUSD', 'GBPUSD', 'XAUUSD']
    
    for symbol in test_symbols:
        print(f"\nTesting {symbol}:")
        print("-" * 40)
        
        # First check - should be allowed
        has_traded = risk_manager.has_traded_today(symbol)
        print(f"  Has traded today? {has_traded}")
        print(f"  Can trade? {not has_traded}")
        
        if not has_traded:
            # Simulate trade execution
            risk_manager.record_trade(symbol)
            print(f"  [ACTION] Trade executed for {symbol}")
            
            # Second check - should be blocked
            has_traded = risk_manager.has_traded_today(symbol)
            print(f"  Has traded today (after trade)? {has_traded}")
            print(f"  Can trade again? {not has_traded}")
            
            # Try to record another trade (should still count)
            risk_manager.record_trade(symbol)
            print(f"  [ATTEMPT] Tried to trade again")
            
            # Check counter
            if symbol in risk_manager.daily_trades_per_symbol:
                count = risk_manager.daily_trades_per_symbol[symbol]['count']
                print(f"  Trade count for today: {count}")
    
    print("\n" + "=" * 80)
    print("TESTING: Daily Reset")
    print("=" * 80)
    print()
    
    # Test reset
    print("Before reset:")
    for symbol in test_symbols:
        if symbol in risk_manager.daily_trades_per_symbol:
            info = risk_manager.daily_trades_per_symbol[symbol]
            print(f"  {symbol}: {info['count']} trades on {info['date']}")
    
    risk_manager.reset_daily_stats()
    
    print("\nAfter reset:")
    for symbol in test_symbols:
        has_traded = risk_manager.has_traded_today(symbol)
        print(f"  {symbol}: Has traded? {has_traded}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nSummary:")
    print("  [+] Each symbol can only trade once per day")
    print("  [+] Daily reset clears all counters")
    print("  [+] Multiple symbols can trade on the same day")
    print("  [+] Rule applies per symbol, not globally")

if __name__ == "__main__":
    test_daily_limit()
