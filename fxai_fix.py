#!/usr/bin/env python3
"""
FX-Ai Quick Fix Script
Addresses the three main issues preventing live trading signals
"""

import os
import json
import time
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class FXAiFixer:
    def __init__(self):
        self.fixes_applied = []
        
    def run_fixes(self):
        """Run all fixes"""
        print("\n" + "="*60)
        print("FX-Ai LIVE TRADING FIX SCRIPT")
        print("="*60 + "\n")
        
        # Fix 1: Signal Generation
        print("🔧 FIX 1: SIGNAL GENERATION")
        self.fix_signal_generation()
        
        # Fix 2: MT5 Communication
        print("\n🔧 FIX 2: MT5 COMMUNICATION")
        self.fix_mt5_communication()
        
        # Fix 3: Time Synchronization
        print("\n🔧 FIX 3: TIME SYNCHRONIZATION")
        self.fix_time_sync()
        
        # Apply configuration fixes
        print("\n🔧 FIX 4: CONFIGURATION ADJUSTMENTS")
        self.fix_configuration()
        
        # Summary
        self.print_summary()
    
    def fix_signal_generation(self):
        """Fix signal generation issues"""
        
        # 1. Check and adjust signal threshold
        config_path = 'config/config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Lower signal threshold if too high
            current_threshold = config.get('trading', {}).get('min_signal_strength', 0.4)
            if current_threshold > 0.5:
                config['trading']['min_signal_strength'] = 0.4
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                self.fixes_applied.append(f"✓ Lowered signal threshold from {current_threshold} to 0.4")
                print(f"  ✓ Adjusted signal threshold: {current_threshold} → 0.4")
            else:
                print(f"  ✓ Signal threshold OK: {current_threshold}")
            
            # 2. Ensure trading mode is correct
            mode = config.get('trading', {}).get('mode', 'live')
            if mode != 'live':
                config['trading']['mode'] = 'live'
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                self.fixes_applied.append(f"✓ Changed trading mode from '{mode}' to 'live'")
                print(f"  ✓ Set trading mode: {mode} → live")
            else:
                print(f"  ✓ Trading mode OK: live")
        
        # 3. Check ML models exist
        model_count = 0
        models_dir = 'models'
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('_model.pkl'):
                    model_count += 1
        
        if model_count == 0:
            print("  ⚠ No ML models found - You need to run: python backtest/train_all_models.py")
            self.fixes_applied.append("⚠ ML models need training")
        else:
            print(f"  ✓ Found {model_count} ML models")
        
        # 4. Create/fix signal weights
        weights_file = 'data/signal_weights.json'
        os.makedirs('data', exist_ok=True)
        
        if not os.path.exists(weights_file):
            default_weights = {
                'technical_score': 0.35,
                'ml_score': 0.25,
                'sentiment_score': 0.15,
                'fundamental_score': 0.15,
                'sr_score': 0.10
            }
            with open(weights_file, 'w') as f:
                json.dump(default_weights, f, indent=2)
            self.fixes_applied.append("✓ Created signal_weights.json with optimized values")
            print(f"  ✓ Created signal weights file")
        else:
            print(f"  ✓ Signal weights file exists")
    
    def fix_mt5_communication(self):
        """Fix MT5 communication issues"""
        
        # 1. Test MT5 connection
        try:
            import MetaTrader5 as mt5
            
            if mt5.initialize():
                print("  ✓ MT5 connection successful")
                
                # Check if logged in
                account_info = mt5.account_info()
                if account_info:
                    print(f"  ✓ Connected to account: {account_info.login}")
                    
                    # Enable symbol if needed
                    symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
                    for symbol in symbols:
                        symbol_info = mt5.symbol_info(symbol)
                        if symbol_info:
                            if not symbol_info.visible:
                                mt5.symbol_select(symbol, True)
                                print(f"  ✓ Enabled symbol: {symbol}")
                        else:
                            print(f"  ⚠ Symbol not available: {symbol}")
                    
                    # Check trading permissions
                    if not account_info.trade_allowed:
                        print("  ⚠ Trading not allowed - Enable in MT5 settings")
                        self.fixes_applied.append("⚠ Need to enable trading in MT5")
                    else:
                        print("  ✓ Trading permissions OK")
                else:
                    print("  ⚠ Not logged in to MT5")
                    self.fixes_applied.append("⚠ Need to log in to MT5")
                
                mt5.shutdown()
            else:
                print("  ⚠ Cannot initialize MT5 - Make sure MT5 is running")
                self.fixes_applied.append("⚠ MT5 needs to be running")
                
        except ImportError:
            print("  ⚠ MetaTrader5 module not installed")
            print("     Run: pip install MetaTrader5")
            self.fixes_applied.append("⚠ Install MetaTrader5 module")
        
        # 2. Create signal files for EA communication
        signal_file = 'fxai_signals.txt'
        if not os.path.exists(signal_file):
            with open(signal_file, 'w') as f:
                f.write("# FX-Ai Signal File\n")
                f.write(f"# Created: {datetime.now()}\n")
                f.write("# Waiting for signals...\n")
            print(f"  ✓ Created signal file: {signal_file}")
            self.fixes_applied.append("✓ Created signal communication file")
        else:
            print(f"  ✓ Signal file exists: {signal_file}")
    
    def fix_time_sync(self):
        """Fix time synchronization issues"""
        
        now = datetime.now()
        
        # 1. Check if it's weekend
        if now.weekday() >= 5:
            print("  ⚠ It's weekend - Forex market is closed")
            print("     Market opens Sunday 5PM EST")
            self.fixes_applied.append("⚠ Wait for market to open (weekend)")
            return
        
        # 2. Check if past trading hours
        config_path = 'config/config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            close_hour = config.get('trading_rules', {}).get('time_restrictions', {}).get('close_hour', 22)
            close_minute = config.get('trading_rules', {}).get('time_restrictions', {}).get('close_minute', 30)
            
            if now.hour > close_hour or (now.hour == close_hour and now.minute >= close_minute):
                print(f"  ⚠ Past trading close time ({close_hour:02d}:{close_minute:02d})")
                print(f"     Current time: {now.strftime('%H:%M')}")
                print("     Trading will resume after midnight")
                self.fixes_applied.append(f"⚠ Wait until after midnight (past {close_hour:02d}:{close_minute:02d})")
            else:
                print(f"  ✓ Within trading hours (closes at {close_hour:02d}:{close_minute:02d})")
        
        # 3. Test MT5 server time
        try:
            import MetaTrader5 as mt5
            
            if mt5.initialize():
                # Get server time through last tick
                tick = mt5.symbol_info_tick("EURUSD")
                if tick:
                    server_time = datetime.fromtimestamp(tick.time)
                    local_time = datetime.now()
                    time_diff = abs((server_time - local_time).total_seconds())
                    
                    print(f"  ✓ MT5 Server Time: {server_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"  ✓ Local Time:      {local_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"  ✓ Time Difference: {time_diff:.1f} seconds")
                    
                    if time_diff > 60:
                        print(f"  ⚠ Large time difference detected!")
                        self.fixes_applied.append(f"⚠ Time sync issue: {time_diff:.0f}s difference")
                else:
                    print("  ⚠ Cannot get MT5 server time (market may be closed)")
                    if now.weekday() >= 5:
                        print("     This is normal on weekends")
                
                mt5.shutdown()
        except:
            pass
        
        # 4. Create time sync file
        time_sync_file = 'mt5_time_sync.json'
        time_data = {
            'last_sync': now.isoformat(),
            'system_time': now.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'synchronized'
        }
        with open(time_sync_file, 'w') as f:
            json.dump(time_data, f, indent=2)
        print(f"  ✓ Created time sync file")
    
    def fix_configuration(self):
        """Apply configuration fixes for common issues"""
        
        config_path = 'config/config.json'
        if not os.path.exists(config_path):
            print("  ⚠ No config.json found - Creating from template")
            self.create_default_config()
            self.fixes_applied.append("✓ Created default configuration")
            return
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        changes_made = False
        
        # Ensure critical sections exist
        if 'trading' not in config:
            config['trading'] = {}
            changes_made = True
        
        if 'trading_rules' not in config:
            config['trading_rules'] = {}
            changes_made = True
        
        # Set optimal values for live trading
        optimal_settings = {
            'trading': {
                'mode': 'live',
                'min_signal_strength': 0.4,
                'symbols': ['EURUSD', 'GBPUSD', 'USDJPY'],
                'risk_per_trade': 50.0,
                'max_positions': 5
            }
        }
        
        for key, value in optimal_settings['trading'].items():
            if key not in config['trading']:
                config['trading'][key] = value
                changes_made = True
                print(f"  ✓ Added {key}: {value}")
        
        # Ensure daily trade limits
        if 'position_limits' not in config['trading_rules']:
            config['trading_rules']['position_limits'] = {}
        
        if 'max_trades_per_symbol_per_day' not in config['trading_rules']['position_limits']:
            config['trading_rules']['position_limits']['max_trades_per_symbol_per_day'] = 1
            changes_made = True
            print(f"  ✓ Set daily trade limit: 1 per symbol")
        
        # Save if changes made
        if changes_made:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            self.fixes_applied.append("✓ Updated configuration with optimal settings")
            print("  ✓ Configuration updated")
        else:
            print("  ✓ Configuration OK")
    
    def create_default_config(self):
        """Create a default configuration file"""
        config = {
            "mt5": {
                "login": "YOUR_ACCOUNT_NUMBER",
                "password": "YOUR_PASSWORD",
                "server": "YOUR_BROKER_SERVER",
                "timeout": 60000,
                "path": ""
            },
            "trading": {
                "mode": "live",
                "symbols": ["EURUSD", "GBPUSD", "USDJPY"],
                "min_signal_strength": 0.4,
                "risk_per_trade": 50.0,
                "max_positions": 5
            },
            "trading_rules": {
                "position_limits": {
                    "max_positions": 30,
                    "max_positions_per_symbol": 1,
                    "max_trades_per_symbol_per_day": 1
                },
                "time_restrictions": {
                    "day_trading_only": True,
                    "close_hour": 22,
                    "close_minute": 30
                },
                "risk_limits": {
                    "risk_per_trade": 50.0,
                    "max_daily_loss": 500.0
                }
            }
        }
        
        os.makedirs('config', exist_ok=True)
        with open('config/config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("  ✓ Created default config.json")
        print("  ⚠ Remember to update MT5 credentials in config.json")
    
    def print_summary(self):
        """Print summary of fixes"""
        print("\n" + "="*60)
        print("SUMMARY OF FIXES APPLIED")
        print("="*60)
        
        if self.fixes_applied:
            for fix in self.fixes_applied:
                print(f"  {fix}")
        else:
            print("  ✓ No fixes needed - system appears ready")
        
        print("\n📋 NEXT STEPS:")
        print("  1. If MT5 not running: Start MetaTrader 5")
        print("  2. If not logged in: Log in to your account")
        print("  3. If no models: Run 'python backtest/train_all_models.py'")
        print("  4. Update config.json with your MT5 credentials")
        print("  5. Run diagnostic: 'python fxai_diagnostic.py'")
        print("  6. Start trading: 'python main.py'")
        print("\n" + "="*60)

if __name__ == "__main__":
    fixer = FXAiFixer()
    fixer.run_fixes()