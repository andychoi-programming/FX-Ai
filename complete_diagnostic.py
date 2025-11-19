"""
FX-Ai Complete System Diagnostic
Identifies ALL root causes of Error 10030 and other issues
"""

import MetaTrader5 as mt5
import json
import os
import sys
from datetime import datetime
import traceback

class SystemDiagnostic:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.config = {}
        self.test_results = {}
        
    def run_complete_diagnostic(self):
        """Run all diagnostic tests"""
        print("=" * 80)
        print("FX-Ai COMPLETE SYSTEM DIAGNOSTIC")
        print("=" * 80)
        print(f"Diagnostic Started: {datetime.now()}")
        print("-" * 80)
        
        # Test 1: MT5 Connection
        self.test_mt5_connection()
        
        # Test 2: Configuration
        self.test_configuration()
        
        # Test 3: Order Types
        self.test_order_types()
        
        # Test 4: File Structure
        self.test_file_structure()
        
        # Test 5: Trading Engine
        self.test_trading_engine()
        
        # Generate Report
        self.generate_report()
        
    def test_mt5_connection(self):
        """Test MT5 connection and account"""
        print("\n[1] Testing MT5 Connection...")
        
        try:
            if not mt5.initialize():
                self.issues.append("CRITICAL: Cannot initialize MT5")
                return False
                
            account_info = mt5.account_info()
            terminal_info = mt5.terminal_info()
            
            if account_info is None:
                self.issues.append("CRITICAL: Cannot get account info")
                return False
                
            self.test_results['mt5'] = {
                'connected': True,
                'account': account_info.login,
                'balance': account_info.balance,
                'server': account_info.server,
                'trade_allowed': terminal_info.trade_allowed,
                'trade_expert': account_info.trade_expert,
                'trade_mode': account_info.trade_mode
            }
            
            print(f"   ✅ MT5 Connected: {account_info.server}")
            print(f"   ✅ Account: {account_info.login}")
            print(f"   ✅ Balance: ${account_info.balance:.2f}")
            
            # Check trading permissions
            if not terminal_info.trade_allowed:
                self.issues.append("CRITICAL: Trading not allowed in terminal")
            if not account_info.trade_expert:
                self.issues.append("CRITICAL: Expert Advisors not allowed")
                
            return True
            
        except Exception as e:
            self.issues.append(f"CRITICAL: MT5 test failed - {str(e)}")
            return False
            
    def test_configuration(self):
        """Test configuration files"""
        print("\n[2] Testing Configuration...")
        
        config_paths = [
            'config.json',
            'C:\\Users\\andyc\\python\\FX-Ai\\config.json',
            '../config.json',
            'config/config.json'
        ]
        
        config_found = False
        for path in config_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        self.config = json.load(f)
                        config_found = True
                        print(f"   ✅ Config found: {path}")
                        
                        # Check critical settings
                        trading_config = self.config.get('trading', {})
                        
                        # Check order management
                        order_mgmt = trading_config.get('order_management', {})
                        entry_strategy = order_mgmt.get('default_entry_strategy', 'unknown')
                        
                        if entry_strategy == 'stop':
                            self.warnings.append(f"WARNING: Default strategy is 'stop' - may cause Error 10030")
                            print(f"   ⚠️  Default order type: STOP (problematic)")
                        else:
                            print(f"   ✅ Default order type: {entry_strategy}")
                            
                        break
                        
                except Exception as e:
                    self.issues.append(f"ERROR: Cannot read config - {str(e)}")
                    
        if not config_found:
            self.issues.append("CRITICAL: No config.json found")
            
    def test_order_types(self):
        """Test different order types with the broker"""
        print("\n[3] Testing Order Types...")
        
        if not mt5.initialize():
            self.issues.append("SKIP: MT5 not initialized for order test")
            return
            
        test_symbol = "EURUSD"
        test_volume = 0.01
        
        # Get symbol info
        symbol_info = mt5.symbol_info(test_symbol)
        if symbol_info is None:
            self.issues.append(f"ERROR: Cannot get info for {test_symbol}")
            return
            
        tick = mt5.symbol_info_tick(test_symbol)
        if tick is None:
            self.issues.append(f"ERROR: Cannot get tick for {test_symbol}")
            return
            
        # Test 1: Market Order with different filling modes
        print(f"   Testing market orders...")
        
        filling_modes = [
            (mt5.ORDER_FILLING_FOK, "FOK"),
            (mt5.ORDER_FILLING_IOC, "IOC"),
            (mt5.ORDER_FILLING_RETURN, "RETURN")
        ]
        
        for filling_mode, mode_name in filling_modes:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": test_symbol,
                "volume": test_volume,
                "type": mt5.ORDER_TYPE_BUY,
                "price": tick.ask,
                "deviation": 10,
                "magic": 12345,
                "comment": f"Test {mode_name}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode
            }
            
            # Check only, don't send
            result = mt5.order_check(request)
            if result and result.retcode == 0:
                print(f"   ✅ Market order with {mode_name}: VALID")
                self.test_results[f'market_{mode_name}'] = 'valid'
            else:
                retcode = result.retcode if result else 'None'
                print(f"   ❌ Market order with {mode_name}: INVALID (Error {retcode})")
                self.test_results[f'market_{mode_name}'] = f'invalid_{retcode}'
                if retcode == 10030:
                    self.issues.append(f"ERROR 10030: Market order with {mode_name} filling")
                    
        # Test 2: Stop Orders
        print(f"   Testing stop orders...")
        
        stop_price = tick.ask + 0.0020  # 20 pips above
        
        stop_request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": test_symbol,
            "volume": test_volume,
            "type": mt5.ORDER_TYPE_BUY_STOP,
            "price": stop_price,
            "sl": stop_price - 0.0030,
            "tp": stop_price + 0.0060,
            "magic": 12345,
            "comment": "Test Stop",
            "type_time": mt5.ORDER_TIME_GTC,
            # NO filling mode for pending orders!
        }
        
        result = mt5.order_check(stop_request)
        if result and result.retcode == 0:
            print(f"   ✅ Stop order: VALID")
            self.test_results['stop_order'] = 'valid'
        else:
            retcode = result.retcode if result else 'None'
            print(f"   ❌ Stop order: INVALID (Error {retcode})")
            self.test_results['stop_order'] = f'invalid_{retcode}'
            if retcode == 10030:
                self.issues.append(f"ERROR 10030: Stop orders not working")
                
    def test_file_structure(self):
        """Test if all required files exist"""
        print("\n[4] Testing File Structure...")
        
        required_files = [
            'core/trading_engine.py',
            'core/risk_manager.py',
            'core/mt5_connector.py',
            'core/signal_generator.py',
            'core/orchestrator.py',
            'order_executor.py',
            'config.json'
        ]
        
        base_paths = [
            '.',
            'C:\\Users\\andyc\\python\\FX-Ai',
            '..',
            'src'
        ]
        
        for base in base_paths:
            files_found = 0
            for file in required_files:
                path = os.path.join(base, file)
                if os.path.exists(path):
                    files_found += 1
                    
            if files_found > 0:
                print(f"   Found {files_found}/{len(required_files)} files in {base}")
                
                # Check which files are missing
                for file in required_files:
                    path = os.path.join(base, file)
                    if not os.path.exists(path):
                        self.warnings.append(f"MISSING: {file}")
                        
    def test_trading_engine(self):
        """Test the trading engine execution path"""
        print("\n[5] Testing Trading Engine...")
        
        # Try to import and test
        try:
            # Check if we can import the trading engine
            sys.path.append('.')
            sys.path.append('C:\\Users\\andyc\\python\\FX-Ai')
            
            # Try to import core modules
            try:
                from core.trading_engine import TradingEngine
                print("   ✅ TradingEngine imported")
            except ImportError as e:
                self.issues.append(f"CRITICAL: Cannot import TradingEngine - {str(e)}")
                print(f"   ❌ Cannot import TradingEngine")
                
            try:
                from order_executor import OrderExecutor
                print("   ✅ OrderExecutor imported")
                
                # Check the actual implementation
                import inspect
                
                # Find _execute_trade_safe method
                if hasattr(OrderExecutor, '_execute_trade_safe'):
                    print("   ✅ _execute_trade_safe method exists")
                else:
                    self.issues.append("CRITICAL: _execute_trade_safe method not found")
                    print("   ❌ _execute_trade_safe method missing")
                    
            except ImportError as e:
                self.issues.append(f"CRITICAL: Cannot import OrderExecutor - {str(e)}")
                print(f"   ❌ Cannot import OrderExecutor")
                
        except Exception as e:
            self.issues.append(f"ERROR: Trading engine test failed - {str(e)}")
            
    def generate_report(self):
        """Generate final diagnostic report"""
        print("\n" + "=" * 80)
        print("DIAGNOSTIC REPORT")
        print("=" * 80)
        
        # Critical Issues
        if self.issues:
            print("\n🔴 CRITICAL ISSUES FOUND:")
            print("-" * 40)
            for i, issue in enumerate(self.issues, 1):
                print(f"{i}. {issue}")
                
        # Warnings
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            print("-" * 40)
            for i, warning in enumerate(self.warnings, 1):
                print(f"{i}. {warning}")
                
        # Root Cause Analysis
        print("\n🎯 ROOT CAUSE ANALYSIS:")
        print("-" * 40)
        
        if any('10030' in str(issue) for issue in self.issues):
            print("\nError 10030 is caused by:")
            
            # Check order type issues
            if self.config.get('trading', {}).get('order_management', {}).get('default_entry_strategy') == 'stop':
                print("1. System is using STOP orders by default")
                print("   - Stop orders may not be configured correctly")
                print("   - Broker may require specific parameters")
                
            # Check filling mode issues  
            if 'invalid_10030' in str(self.test_results.values()):
                print("2. Filling mode mismatch")
                print("   - Broker requires specific filling mode")
                print("   - Current implementation may use wrong mode")
                
            print("\n📝 RECOMMENDED FIXES:")
            print("1. Switch to MARKET orders:")
            print("   - Change config: 'default_entry_strategy': 'market'")
            print("2. Fix OrderExecutor implementation:")
            print("   - Ensure correct filling mode for your broker")
            print("3. Verify _execute_trade_safe method:")
            print("   - This is where actual order is sent")
            print("   - Must handle broker-specific requirements")
            
        # Save report
        report_file = f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_data = {
            'timestamp': str(datetime.now()),
            'issues': self.issues,
            'warnings': self.warnings,
            'test_results': self.test_results,
            'config': self.config
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        print(f"\n💾 Report saved to: {report_file}")
        print("=" * 80)

if __name__ == "__main__":
    diagnostic = SystemDiagnostic()
    diagnostic.run_complete_diagnostic()
    
    # Shutdown MT5
    mt5.shutdown()
