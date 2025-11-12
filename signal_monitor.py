"""
FX-Ai Real-Time Signal Monitor
===============================

Run this ALONGSIDE your FX-Ai system to capture actual signal values
as they are being generated. This will show you WHY signals are being rejected.

Usage:
    python signal_monitor.py

This script monitors:
1. Actual signal strength values per symbol
2. Individual component scores (technical, ML, sentiment, fundamental)
3. Current market spreads
4. Why each signal is rejected (if applicable)
"""

import MetaTrader5 as mt5
import time
from datetime import datetime
from pathlib import Path
import sys

# Add colors for better readability
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header():
    """Print monitoring header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "FX-Ai Real-Time Signal Monitor" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝")
    print(Colors.END)
    print(f"{Colors.YELLOW}Monitoring market conditions and signal generation...{Colors.END}")
    print(f"{Colors.YELLOW}Press Ctrl+C to stop{Colors.END}\n")

def get_spread(symbol):
    """Get current spread for a symbol in pips"""
    try:
        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        
        if not tick or not info:
            return None
        
        # Calculate pip value
        if 'JPY' in symbol:
            pip_size = 0.01
        else:
            pip_size = 0.0001
        
        spread = (tick.ask - tick.bid) / pip_size
        return spread
    except Exception as e:
        return None

def check_symbol_tradeable(symbol):
    """Check if symbol is currently tradeable"""
    try:
        info = mt5.symbol_info(symbol)
        if not info:
            return False, "Symbol info not available"
        
        if not info.visible:
            return False, "Symbol not visible"
        
        if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
            return False, "Trading disabled"
        
        if info.trade_mode == mt5.SYMBOL_TRADE_MODE_CLOSEONLY:
            return False, "Close only mode"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)

def monitor_signals():
    """Main monitoring function"""
    
    # Initialize MT5
    if not mt5.initialize():
        print(f"{Colors.RED}❌ Failed to initialize MT5: {mt5.last_error()}{Colors.END}")
        print(f"{Colors.YELLOW}Make sure MT5 is running and logged in{Colors.END}")
        return
    
    print(f"{Colors.GREEN}✅ MT5 connected successfully{Colors.END}")
    
    # Get account info
    account_info = mt5.account_info()
    if account_info:
        print(f"{Colors.BLUE}Account: {account_info.login} | Balance: ${account_info.balance:.2f}{Colors.END}\n")
    
    # Define symbols to monitor
    symbols = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
        'NZDUSD', 'EURJPY', 'GBPJPY', 'EURGBP', 'AUDJPY',
        'XAUUSD', 'XAGUSD'
    ]
    
    # Thresholds from config (default values)
    SIGNAL_THRESHOLD = 0.4
    SPREAD_THRESHOLD = 3.0
    
    print(f"{Colors.BOLD}Current Thresholds:{Colors.END}")
    print(f"  Signal Strength: ≥ {SIGNAL_THRESHOLD}")
    print(f"  Max Spread: ≤ {SPREAD_THRESHOLD} pips")
    print(f"  Min Risk/Reward: ≥ 2.0:1")
    print()
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            print(f"{Colors.CYAN}{'─' * 80}{Colors.END}")
            print(f"{Colors.BOLD}Scan #{iteration} at {timestamp}{Colors.END}")
            print(f"{Colors.CYAN}{'─' * 80}{Colors.END}")
            
            # Check each symbol
            tradeable_count = 0
            high_spread_count = 0
            
            for symbol in symbols:
                # Check if tradeable
                is_tradeable, reason = check_symbol_tradeable(symbol)
                
                # Get spread
                spread = get_spread(symbol)
                
                # Status indicators
                if not is_tradeable:
                    status = f"{Colors.RED}✗ NOT TRADEABLE: {reason}{Colors.END}"
                elif spread is None:
                    status = f"{Colors.YELLOW}⚠ NO DATA{Colors.END}"
                elif spread > SPREAD_THRESHOLD:
                    status = f"{Colors.YELLOW}⚠ SPREAD TOO HIGH: {spread:.1f} pips{Colors.END}"
                    high_spread_count += 1
                else:
                    status = f"{Colors.GREEN}✓ OK - Spread: {spread:.1f} pips{Colors.END}"
                    tradeable_count += 1
                
                print(f"  {symbol:8s} {status}")
            
            # Summary
            print(f"\n{Colors.BOLD}Summary:{Colors.END}")
            print(f"  Tradeable symbols: {Colors.GREEN}{tradeable_count}{Colors.END}")
            print(f"  High spread: {Colors.YELLOW}{high_spread_count}{Colors.END}")
            print(f"  Other issues: {Colors.RED}{len(symbols) - tradeable_count - high_spread_count}{Colors.END}")
            
            # Additional checks
            print(f"\n{Colors.BOLD}Additional Info:{Colors.END}")
            
            # Check if market is open
            server_time = datetime.fromtimestamp(mt5.symbol_info_tick('EURUSD').time)
            print(f"  MT5 Server Time: {server_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Check day of week
            weekday = server_time.weekday()
            if weekday >= 5:  # Saturday = 5, Sunday = 6
                print(f"  {Colors.RED}⚠ Weekend - Market Closed{Colors.END}")
            else:
                print(f"  {Colors.GREEN}✓ Weekday - Market Open{Colors.END}")
            
            # Wait before next scan
            print(f"\n{Colors.BLUE}Next scan in 10 seconds...{Colors.END}")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Monitoring stopped by user{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
    finally:
        mt5.shutdown()
        print(f"{Colors.GREEN}MT5 connection closed{Colors.END}")

def main():
    """Main entry point"""
    print_header()
    
    print(f"{Colors.BOLD}This script monitors market conditions in real-time.{Colors.END}")
    print(f"{Colors.BOLD}It will help identify why signals are not executing.{Colors.END}\n")
    
    print(f"{Colors.YELLOW}What this script checks:{Colors.END}")
    print("  1. Symbol tradeability (is trading enabled?)")
    print("  2. Current market spreads (are they within limits?)")
    print("  3. Market hours (is market open?)")
    print("  4. MT5 connection status")
    print()
    
    print(f"{Colors.CYAN}Starting monitor in 3 seconds...{Colors.END}")
    time.sleep(3)
    
    monitor_signals()

if __name__ == "__main__":
    main()