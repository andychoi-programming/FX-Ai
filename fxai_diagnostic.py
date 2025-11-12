#!/usr/bin/env python3
"""
FX-Ai Live Trading Diagnostic Tool
Checks for common issues preventing live trading signals
"""

import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta
import logging
import time

# Setup diagnostic logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FXAiDiagnostic:
    def __init__(self):
        self.issues_found = []
        self.warnings = []
        self.successes = []
        
    def run_all_checks(self):
        """Run all diagnostic checks"""
        print("\n" + "="*60)
        print("FX-Ai LIVE TRADING DIAGNOSTIC")
        print("="*60 + "\n")
        
        # Check 1: Configuration
        self.check_configuration()
        
        # Check 2: Database and Trade History
        self.check_database()
        
        # Check 3: Signal Generation
        self.check_signal_generation()
        
        # Check 4: MT5 Time Synchronization
        self.check_time_sync()
        
        # Check 5: Log Files
        self.check_log_files()
        
        # Check 6: Adaptive Learning Requirements
        self.check_adaptive_learning()
        
        # Check 7: Trading Rules
        self.check_trading_rules()
        
        # Check 8: MT5 Connection
        self.check_mt5_connection()
        
        # Report results
        self.print_report()
        
    def check_configuration(self):
        """Check configuration files"""
        print("🔍 Checking Configuration...")
        
        config_files = [
            'config/config.json',
            'config/adaptive_weights.json',
            'models/parameter_optimization/optimal_parameters.json'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as f:
                        data = json.load(f)
                        self.successes.append(f"✓ {config_file} exists and is valid JSON")
                        
                        # Check specific settings
                        if 'config.json' in config_file:
                            self.check_config_details(data)
                except Exception as e:
                    self.issues_found.append(f"✗ {config_file} is corrupted: {e}")
            else:
                self.issues_found.append(f"✗ {config_file} not found")
    
    def check_config_details(self, config):
        """Check specific configuration details"""
        # Check trading mode
        if config.get('trading', {}).get('mode') == 'paper':
            self.warnings.append("⚠ Trading mode is set to 'paper' - change to 'live' for real trading")
        
        # Check signal strength threshold
        signal_threshold = config.get('trading', {}).get('min_signal_strength', 0.4)
        if signal_threshold > 0.7:
            self.warnings.append(f"⚠ Signal threshold is high ({signal_threshold}) - may prevent trades")
        
        # Check day trading settings
        if config.get('trading_rules', {}).get('time_restrictions', {}).get('day_trading_only', True):
            close_hour = config.get('trading_rules', {}).get('time_restrictions', {}).get('close_hour', 22)
            close_minute = config.get('trading_rules', {}).get('time_restrictions', {}).get('close_minute', 30)
            self.successes.append(f"✓ Day trading mode active (closes at {close_hour:02d}:{close_minute:02d})")
    
    def check_database(self):
        """Check database and trade history"""
        print("\n🔍 Checking Database...")
        
        db_path = 'data/performance_history.db'
        if not os.path.exists(db_path):
            self.issues_found.append("✗ Database not found at data/performance_history.db")
            return
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check trades table
            cursor.execute("SELECT COUNT(*) FROM trades")
            trade_count = cursor.fetchone()[0]
            
            if trade_count == 0:
                self.issues_found.append("✗ No trades in database - adaptive learning needs historical data")
            else:
                self.successes.append(f"✓ Found {trade_count} trades in database")
                
                # Check recent trades
                cursor.execute("""
                    SELECT timestamp, symbol, profit_pct 
                    FROM trades 
                    ORDER BY timestamp DESC 
                    LIMIT 5
                """)
                recent_trades = cursor.fetchall()
                
                if recent_trades:
                    last_trade_time = datetime.fromisoformat(recent_trades[0][0])
                    days_since_trade = (datetime.now() - last_trade_time).days
                    
                    if days_since_trade > 7:
                        self.warnings.append(f"⚠ No trades in last {days_since_trade} days")
            
            # Check daily trades tracking
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='daily_trades'
            """)
            if cursor.fetchone():
                cursor.execute("SELECT symbol, trade_date FROM daily_trades WHERE trade_date = date('now')")
                today_trades = cursor.fetchall()
                if today_trades:
                    symbols_traded = [t[0] for t in today_trades]
                    self.warnings.append(f"⚠ Already traded today: {', '.join(symbols_traded)}")
            
            conn.close()
            
        except Exception as e:
            self.issues_found.append(f"✗ Database error: {e}")
    
    def check_signal_generation(self):
        """Check signal generation components"""
        print("\n🔍 Checking Signal Generation...")
        
        # Check ML models
        model_files = [
            'models/EURUSD_H1_model.pkl',
            'models/EURUSD_H1_scaler.pkl'
        ]
        
        models_found = 0
        for model_file in model_files:
            if os.path.exists(model_file):
                models_found += 1
        
        if models_found == 0:
            self.issues_found.append("✗ No ML models found - run train_all_models.py")
        else:
            self.successes.append(f"✓ Found {models_found} ML model files")
        
        # Check signal weights
        weights_file = 'data/signal_weights.json'
        if os.path.exists(weights_file):
            with open(weights_file, 'r') as f:
                weights = json.load(f)
                total_weight = sum(weights.values())
                if abs(total_weight - 1.0) > 0.01:
                    self.warnings.append(f"⚠ Signal weights don't sum to 1.0 (sum={total_weight:.2f})")
                self.successes.append(f"✓ Signal weights configured: ML={weights.get('ml_score', 0):.2f}, Tech={weights.get('technical_score', 0):.2f}")
        else:
            self.warnings.append("⚠ No signal_weights.json found - using defaults")
    
    def check_time_sync(self):
        """Check time synchronization issues"""
        print("\n🔍 Checking Time Synchronization...")
        
        # Check for common MT5 time sync issues
        current_time = datetime.now()
        current_hour = current_time.hour
        
        # Check if it's weekend
        if current_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            self.warnings.append("⚠ It's weekend - Forex market is closed")
        
        # Check if past closing time (22:30)
        if current_hour >= 22 and current_time.minute >= 30:
            self.issues_found.append("✗ Past 22:30 - trading is blocked by day trading rules")
        
        # Check for MT5 server time file (if using EA)
        time_sync_file = 'FXAi_TimeSync.csv'
        if os.path.exists(time_sync_file):
            try:
                with open(time_sync_file, 'r') as f:
                    content = f.read()
                    if content:
                        # Parse last sync time
                        last_line = content.strip().split('\n')[-1]
                        self.successes.append(f"✓ MT5 time sync file found: {last_line[:50]}")
                    else:
                        self.warnings.append("⚠ MT5 time sync file is empty")
            except Exception as e:
                self.warnings.append(f"⚠ Cannot read time sync file: {e}")
        else:
            self.warnings.append("⚠ No MT5 time sync file found - EA may not be running")
        
        self.successes.append(f"✓ Current system time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def check_log_files(self):
        """Check log files for errors"""
        print("\n🔍 Checking Log Files...")
        
        current_time = datetime.now()
        
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            self.issues_found.append("✗ No logs directory found")
            return
        
        # Find today's log file
        today = datetime.now().strftime('%Y-%m-%d')
        log_file = f"{log_dir}/fxai_{today}.log"
        
        if os.path.exists(log_file):
            self.successes.append(f"✓ Today's log file found: {log_file}")
            
            # Check for common errors
            error_patterns = {
                "Cannot get MT5 server time": "MT5 time sync issue",
                "No trading allowed": "Trading blocked",
                "Signal strength too low": "Weak signals",
                "Spread too high": "High spreads preventing trades",
                "Daily trade limit reached": "Already traded today",
                "RiskLimitExceededError": "Risk limits exceeded",
                "MT5 connection lost": "MT5 connection issues",
                "Model prediction failed": "ML model errors"
            }
            
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    for pattern, description in error_patterns.items():
                        count = content.count(pattern)
                        if count > 0:
                            if "Cannot get MT5 server time" in pattern and current_time.weekday() >= 5:
                                self.warnings.append(f"⚠ {description} ({count} occurrences) - Expected on weekends")
                            else:
                                self.issues_found.append(f"✗ {description} ({count} occurrences)")
                    
                    # Check for successful trades
                    if "Trade opened successfully" in content:
                        trade_count = content.count("Trade opened successfully")
                        self.successes.append(f"✓ {trade_count} successful trades logged today")
                    else:
                        self.warnings.append("⚠ No successful trades logged today")
                        
            except Exception as e:
                self.warnings.append(f"⚠ Cannot read log file: {e}")
        else:
            self.warnings.append(f"⚠ No log file for today ({log_file})")
    
    def check_adaptive_learning(self):
        """Check adaptive learning requirements"""
        print("\n🔍 Checking Adaptive Learning...")
        
        # Adaptive learning needs minimum trades
        MIN_TRADES_REQUIRED = 50
        
        db_path = 'data/performance_history.db'
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM trades")
                trade_count = cursor.fetchone()[0]
                
                if trade_count < MIN_TRADES_REQUIRED:
                    self.warnings.append(f"⚠ Adaptive learning needs {MIN_TRADES_REQUIRED} trades, have {trade_count}")
                    self.warnings.append(f"  Consider running in paper mode first to build history")
                else:
                    self.successes.append(f"✓ Sufficient trades for adaptive learning ({trade_count})")
                
                conn.close()
            except:
                pass
    
    def check_trading_rules(self):
        """Check trading rule constraints"""
        print("\n🔍 Checking Trading Rules...")
        
        config_path = 'config/config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            rules = config.get('trading_rules', {})
            
            # Check critical rules
            if rules.get('position_limits', {}).get('max_trades_per_symbol_per_day') == 1:
                self.successes.append("✓ One trade per symbol per day rule active")
            
            risk = rules.get('risk_limits', {}).get('risk_per_trade', 50)
            self.successes.append(f"✓ Risk per trade: ${risk}")
            
            min_rr = rules.get('entry_rules', {}).get('min_risk_reward_ratio', 2.0)
            self.successes.append(f"✓ Minimum risk/reward ratio: {min_rr}:1")
    
    def check_mt5_connection(self):
        """Check MT5 connection"""
        print("\n🔍 Checking MT5 Connection...")
        
        try:
            import MetaTrader5 as mt5
            
            if mt5.initialize():
                self.successes.append("✓ MT5 module initialized successfully")
                
                account_info = mt5.account_info()
                if account_info:
                    self.successes.append(f"✓ Connected to account: {account_info.login}")
                    self.successes.append(f"✓ Balance: ${account_info.balance:.2f}")
                    
                    if account_info.trade_allowed:
                        self.successes.append("✓ Trading is allowed on this account")
                    else:
                        self.issues_found.append("✗ Trading is NOT allowed on this account")
                    
                    # Check if market is open
                    symbol_info = mt5.symbol_info("EURUSD")
                    if symbol_info:
                        if symbol_info.trade_mode == 0:
                            self.issues_found.append("✗ EURUSD market is CLOSED")
                        else:
                            self.successes.append("✓ EURUSD market is OPEN")
                
                mt5.shutdown()
            else:
                self.issues_found.append("✗ Cannot initialize MT5")
                error = mt5.last_error()
                if error:
                    self.issues_found.append(f"  MT5 Error: {error}")
                    
        except ImportError:
            self.issues_found.append("✗ MetaTrader5 module not installed")
        except Exception as e:
            self.issues_found.append(f"✗ MT5 connection error: {e}")
    
    def print_report(self):
        """Print diagnostic report"""
        print("\n" + "="*60)
        print("DIAGNOSTIC REPORT")
        print("="*60)
        
        if self.successes:
            print("\n✅ WORKING CORRECTLY:")
            for item in self.successes:
                print(f"  {item}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for item in self.warnings:
                print(f"  {item}")
        
        if self.issues_found:
            print("\n❌ ISSUES FOUND:")
            for item in self.issues_found:
                print(f"  {item}")
            
            print("\n🔧 RECOMMENDED FIXES:")
            self.print_recommendations()
        else:
            print("\n✅ No critical issues found!")
        
        print("\n" + "="*60)
    
    def print_recommendations(self):
        """Print specific recommendations based on issues found"""
        recommendations = []
        
        for issue in self.issues_found:
            if "MT5 time sync issue" in issue:
                recommendations.append("""
  📍 MT5 TIME SYNC FIX:
     1. Ensure MT5 is running and logged in
     2. Check if EA is attached to a chart
     3. Verify AutoTrading is enabled
     4. Run during market hours (not weekends)
                """)
            
            elif "No trades in database" in issue:
                recommendations.append("""
  📍 NO TRADE HISTORY FIX:
     1. Run in paper mode first: Set config.json trading.mode = "paper"
     2. Let it run for a few days to build history
     3. Once you have 50+ trades, switch to live mode
     4. Or manually add test trades to database
                """)
            
            elif "Past 22:30" in issue:
                recommendations.append("""
  📍 TRADING TIME FIX:
     1. Wait until next trading session (after midnight)
     2. Or modify close_hour in config.json if needed
     3. System blocks trades after 22:30 by design
                """)
            
            elif "No ML models found" in issue:
                recommendations.append("""
  📍 ML MODELS FIX:
     1. Run: python backtest/train_all_models.py
     2. This will create the required model files
     3. Models are stored in models/ directory
                """)
            
            elif "Trading is NOT allowed" in issue:
                recommendations.append("""
  📍 MT5 TRADING PERMISSION FIX:
     1. Check MT5: Tools -> Options -> Expert Advisors
     2. Enable "Allow automated trading"
     3. Enable "Allow DLL imports" if needed
     4. Verify account has trading permissions
                """)
            
            elif "market is CLOSED" in issue:
                recommendations.append("""
  📍 MARKET CLOSED FIX:
     1. Wait for market to open (Sunday 5PM EST)
     2. Forex market hours: Sun 5PM - Fri 5PM EST
     3. Check for holidays that might close markets
                """)
        
        # Remove duplicates and print
        seen = set()
        for rec in recommendations:
            if rec not in seen:
                print(rec)
                seen.add(rec)

if __name__ == "__main__":
    diagnostic = FXAiDiagnostic()
    diagnostic.run_all_checks()
    
    print("\n💡 QUICK TEST COMMANDS:")
    print("  1. Test MT5 connection:  python mt5_diagnostic.py")
    print("  2. Check risk settings:  python risk_display.py")
    print("  3. Train ML models:      python backtest/train_all_models.py")
    print("  4. Start trading:        python main.py")
    print("  5. View dashboard:       python live_trading/trading_dashboard.py")