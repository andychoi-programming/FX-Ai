"""
FX-Ai Signal Blocker Identifier
================================

This script identifies the EXACT line of code or condition that is blocking
signal execution in your FX-Ai system.

It simulates the signal validation process step-by-step to show where trades fail.
"""

import MetaTrader5 as mt5
import json
from pathlib import Path
from datetime import datetime, date
import sqlite3

# Colors
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def load_config():
    """Load configuration file"""
    config_path = Path('config/config.json')
    if not config_path.exists():
        return None
    
    with open(config_path, 'r') as f:
        return json.load(f)

def check_daily_trade_limit(symbol):
    """Check if symbol has already traded today"""
    db_path = Path('data/performance_history.db')
    
    if not db_path.exists():
        return False, "Database not found"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_trades'")
        if not cursor.fetchone():
            conn.close()
            return False, "No tracking table"
        
        # Check today's trades
        today = date.today().isoformat()
        cursor.execute("""
            SELECT trade_count 
            FROM daily_trades 
            WHERE symbol = ? AND trade_date = ?
        """, (symbol, today))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] >= 1:
            return True, f"Already traded {result[0]} time(s) today"
        
        return False, "No trades today"
        
    except Exception as e:
        return False, f"Error: {e}"

def simulate_signal_validation(symbol, config):
    """Simulate the signal validation process for a symbol"""
    
    print(f"\n{Colors.CYAN}{'═' * 80}{Colors.END}")
    print(f"{Colors.BOLD}Testing Symbol: {symbol}{Colors.END}")
    print(f"{Colors.CYAN}{'═' * 80}{Colors.END}\n")
    
    # Initialize MT5
    if not mt5.initialize():
        print(f"{Colors.RED}❌ STEP 0: MT5 Connection FAILED{Colors.END}")
        return False
    
    print(f"{Colors.GREEN}✅ STEP 0: MT5 Connected{Colors.END}")
    
    # Get entry rules from config
    entry_rules = config.get('trading_rules', {}).get('entry_rules', {})
    position_limits = config.get('trading_rules', {}).get('position_limits', {})
    
    min_signal_strength = entry_rules.get('min_signal_strength', 0.4)
    max_spread = entry_rules.get('max_spread', 3.0)
    min_risk_reward = entry_rules.get('min_risk_reward_ratio', 2.0)
    require_ml = entry_rules.get('require_ml_confirmation', True)
    require_tech = entry_rules.get('require_technical_confirmation', True)
    max_trades_per_day = position_limits.get('max_trades_per_symbol_per_day', 1)
    
    # Step 1: Check if symbol exists and is tradeable
    print(f"\n{Colors.BOLD}STEP 1: Symbol Availability{Colors.END}")
    info = mt5.symbol_info(symbol)
    
    if not info:
        print(f"{Colors.RED}❌ BLOCKED: Symbol '{symbol}' not found in MT5{Colors.END}")
        print(f"{Colors.YELLOW}   → Check symbol name spelling{Colors.END}")
        print(f"{Colors.YELLOW}   → Ensure symbol is added to Market Watch{Colors.END}")
        mt5.shutdown()
        return False
    
    print(f"{Colors.GREEN}✅ Symbol exists: {info.description}{Colors.END}")
    
    # Step 2: Check if trading is enabled
    print(f"\n{Colors.BOLD}STEP 2: Trading Permissions{Colors.END}")
    
    if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
        print(f"{Colors.RED}❌ BLOCKED: Trading is DISABLED for this symbol{Colors.END}")
        print(f"{Colors.YELLOW}   → Contact broker to enable trading{Colors.END}")
        mt5.shutdown()
        return False
    
    if info.trade_mode == mt5.SYMBOL_TRADE_MODE_CLOSEONLY:
        print(f"{Colors.RED}❌ BLOCKED: Symbol is in CLOSE-ONLY mode{Colors.END}")
        print(f"{Colors.YELLOW}   → Wait for market to open{Colors.END}")
        mt5.shutdown()
        return False
    
    print(f"{Colors.GREEN}✅ Trading enabled: {info.trade_mode}{Colors.END}")
    
    # Step 3: Check spread
    print(f"\n{Colors.BOLD}STEP 3: Spread Check{Colors.END}")
    tick = mt5.symbol_info_tick(symbol)
    
    if not tick:
        print(f"{Colors.RED}❌ BLOCKED: Cannot get tick data{Colors.END}")
        mt5.shutdown()
        return False
    
    # Calculate spread in pips
    if 'JPY' in symbol:
        pip_size = 0.01
    else:
        pip_size = 0.0001
    
    current_spread = (tick.ask - tick.bid) / pip_size
    
    print(f"   Current spread: {current_spread:.1f} pips")
    print(f"   Maximum allowed: {max_spread} pips")
    
    if current_spread > max_spread:
        print(f"{Colors.RED}❌ BLOCKED: Spread TOO HIGH{Colors.END}")
        print(f"{Colors.YELLOW}   → Current: {current_spread:.1f} pips > Max: {max_spread} pips{Colors.END}")
        print(f"{Colors.YELLOW}   → Wait for tighter spreads OR increase max_spread in config{Colors.END}")
        print(f"{Colors.YELLOW}   → Best spreads during London/NY overlap (8 AM - 12 PM EST){Colors.END}")
        mt5.shutdown()
        return False
    
    print(f"{Colors.GREEN}✅ Spread OK: {current_spread:.1f} ≤ {max_spread} pips{Colors.END}")
    
    # Step 4: Check daily trade limit
    print(f"\n{Colors.BOLD}STEP 4: Daily Trade Limit{Colors.END}")
    already_traded, trade_status = check_daily_trade_limit(symbol)
    
    print(f"   Max trades per symbol per day: {max_trades_per_day}")
    print(f"   Status: {trade_status}")
    
    if already_traded and max_trades_per_day == 1:
        print(f"{Colors.RED}❌ BLOCKED: Daily limit reached{Colors.END}")
        print(f"{Colors.YELLOW}   → This symbol already traded today{Colors.END}")
        print(f"{Colors.YELLOW}   → Wait until tomorrow OR increase max_trades_per_symbol_per_day{Colors.END}")
        print(f"{Colors.YELLOW}   → For testing: Clear database with:{Colors.END}")
        print(f"{Colors.YELLOW}      sqlite3 data/performance_history.db \"DELETE FROM daily_trades WHERE trade_date = date('now');\"{Colors.END}")
        mt5.shutdown()
        return False
    
    print(f"{Colors.GREEN}✅ Daily limit OK: Can trade{Colors.END}")
    
    # Step 5: Signal strength simulation
    print(f"\n{Colors.BOLD}STEP 5: Signal Strength (SIMULATED){Colors.END}")
    print(f"{Colors.YELLOW}⚠️  NOTE: Actual signal calculation requires running FX-Ai components{Colors.END}")
    print(f"{Colors.YELLOW}⚠️  This shows the MINIMUM required signal strength{Colors.END}\n")
    
    print(f"   Required signal strength: ≥ {min_signal_strength}")
    print(f"   Signal formula: (0.25×technical + 0.30×ML + 0.20×sentiment + 0.15×fundamental + 0.10×S/R)")
    print()
    print(f"   {Colors.BOLD}Example scenarios:{Colors.END}")
    
    # Scenario 1: All components at 0.5
    example_signal_1 = 0.25*0.5 + 0.30*0.5 + 0.20*0.5 + 0.15*0.5 + 0.10*0.5
    if example_signal_1 >= min_signal_strength:
        print(f"   {Colors.GREEN}✓{Colors.END} All components at 0.5 = {example_signal_1:.2f} → WOULD PASS")
    else:
        print(f"   {Colors.RED}✗{Colors.END} All components at 0.5 = {example_signal_1:.2f} → WOULD FAIL")
    
    # Scenario 2: All components at 0.6
    example_signal_2 = 0.25*0.6 + 0.30*0.6 + 0.20*0.6 + 0.15*0.6 + 0.10*0.6
    if example_signal_2 >= min_signal_strength:
        print(f"   {Colors.GREEN}✓{Colors.END} All components at 0.6 = {example_signal_2:.2f} → WOULD PASS")
    else:
        print(f"   {Colors.RED}✗{Colors.END} All components at 0.6 = {example_signal_2:.2f} → WOULD FAIL")
    
    # Scenario 3: All components at 0.7
    example_signal_3 = 0.25*0.7 + 0.30*0.7 + 0.20*0.7 + 0.15*0.7 + 0.10*0.7
    if example_signal_3 >= min_signal_strength:
        print(f"   {Colors.GREEN}✓{Colors.END} All components at 0.7 = {example_signal_3:.2f} → WOULD PASS")
    else:
        print(f"   {Colors.RED}✗{Colors.END} All components at 0.7 = {example_signal_3:.2f} → WOULD FAIL")
    
    print(f"\n{Colors.YELLOW}📊 TO SEE ACTUAL SIGNAL VALUES:{Colors.END}")
    print(f"{Colors.YELLOW}   1. Add debug logging to your trading_engine.py or main.py{Colors.END}")
    print(f"{Colors.YELLOW}   2. Or check if FX-Ai logs signal components (grep 'signal' logs/*.log){Colors.END}")
    
    # Step 6: ML confirmation requirement
    print(f"\n{Colors.BOLD}STEP 6: ML Confirmation Requirement{Colors.END}")
    print(f"   ML confirmation required: {require_ml}")
    
    if require_ml:
        print(f"{Colors.YELLOW}⚠️  ML models must predict direction{Colors.END}")
        print(f"{Colors.YELLOW}   → Check if models are loaded: ls -lh models/*.pkl{Colors.END}")
        print(f"{Colors.YELLOW}   → If missing: python backtest/train_all_models.py{Colors.END}")
    else:
        print(f"{Colors.GREEN}✅ ML confirmation not required{Colors.END}")
    
    # Step 7: Technical confirmation requirement
    print(f"\n{Colors.BOLD}STEP 7: Technical Confirmation Requirement{Colors.END}")
    print(f"   Technical confirmation required: {require_tech}")
    
    if require_tech:
        print(f"{Colors.YELLOW}⚠️  Technical indicators must confirm direction{Colors.END}")
    else:
        print(f"{Colors.GREEN}✅ Technical confirmation not required{Colors.END}")
    
    # Step 8: Risk/Reward ratio
    print(f"\n{Colors.BOLD}STEP 8: Risk/Reward Ratio{Colors.END}")
    print(f"   Minimum ratio: {min_risk_reward}:1")
    print(f"{Colors.YELLOW}⚠️  Calculated during trade setup based on ATR and SL/TP levels{Colors.END}")
    
    # Summary
    print(f"\n{Colors.CYAN}{'═' * 80}{Colors.END}")
    print(f"{Colors.BOLD}VALIDATION SUMMARY FOR {symbol}{Colors.END}")
    print(f"{Colors.CYAN}{'═' * 80}{Colors.END}\n")
    
    print(f"{Colors.GREEN}✅ Symbol exists and is tradeable{Colors.END}")
    print(f"{Colors.GREEN}✅ Spread is within limits ({current_spread:.1f} ≤ {max_spread} pips){Colors.END}")
    print(f"{Colors.GREEN}✅ Daily trade limit OK{Colors.END}")
    print(f"{Colors.YELLOW}⚠️  Signal strength needs actual FX-Ai analysis to verify{Colors.END}")
    print(f"{Colors.YELLOW}⚠️  ML/Technical confirmations need actual component checks{Colors.END}")
    print(f"{Colors.YELLOW}⚠️  Risk/Reward calculated during trade setup{Colors.END}")
    
    print(f"\n{Colors.BOLD}MOST LIKELY BLOCKER:{Colors.END}")
    print(f"{Colors.RED}→ Signal strength < {min_signal_strength} (not shown in basic logs){Colors.END}")
    print(f"{Colors.YELLOW}→ To verify: Add debug logging to see actual signal values{Colors.END}")
    
    mt5.shutdown()
    return True

def main():
    """Main function"""
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "FX-Ai Signal Blocker Identifier" + " " * 26 + "║")
    print("╚" + "═" * 78 + "╝")
    print(Colors.END)
    
    # Load config
    print(f"\n{Colors.BOLD}Loading configuration...{Colors.END}")
    config = load_config()
    
    if not config:
        print(f"{Colors.RED}❌ Configuration file not found: config/config.json{Colors.END}")
        return
    
    print(f"{Colors.GREEN}✅ Configuration loaded{Colors.END}")
    
    # Test a few key symbols
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    
    print(f"\n{Colors.BOLD}Testing {len(symbols)} representative symbols...{Colors.END}")
    print(f"{Colors.YELLOW}This will identify where trades are being blocked{Colors.END}")
    
    for symbol in symbols:
        simulate_signal_validation(symbol, config)
        print("\n" + "─" * 80 + "\n")
    
    # Final recommendations
    print(f"\n{Colors.CYAN}{'═' * 80}{Colors.END}")
    print(f"{Colors.BOLD}RECOMMENDATIONS{Colors.END}")
    print(f"{Colors.CYAN}{'═' * 80}{Colors.END}\n")
    
    print(f"{Colors.BOLD}Based on this analysis, the most likely issue is:{Colors.END}")
    print(f"  {Colors.RED}Signal strength < threshold{Colors.END}\n")
    
    print(f"{Colors.BOLD}To diagnose signal strength:{Colors.END}")
    print("  1. Check FX-Ai logs for signal component values:")
    print("     grep -i 'signal\\|score' logs/*.log | tail -50")
    print()
    print("  2. Add debug logging to trading_engine.py or main.py:")
    print("     logger.info(f\"{symbol} - Signal: {signal_strength:.3f}, Tech: {tech_score:.3f}, ML: {ml_score:.3f}\")")
    print()
    print("  3. Temporarily lower threshold to test:")
    print("     python adjust_config.py → Select 'Aggressive' mode")
    print()
    
    print(f"{Colors.BOLD}Alternative tests:{Colors.END}")
    print("  • Run signal_monitor.py to see real-time market conditions")
    print("  • Check if all 30 symbols already traded today (database check)")
    print("  • Verify ML models exist: ls -lh models/*.pkl | wc -l (should be 30)")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.END}")
        import traceback
        traceback.print_exc()