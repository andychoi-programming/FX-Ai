#!/usr/bin/env python3
"""
FX-Ai OrderManager Fix and Diagnostic Tool
This script will find and fix the _calculate_min_stop_distance issue
"""

import os
import re
import sys
import glob
from datetime import datetime

def find_order_manager_file():
    """Find the file containing OrderManager class"""
    print("\nSearching for OrderManager class...")
    
    # Common locations to search
    search_paths = [
        'core/*.py',
        '*.py',
        'trading/*.py',
        'engine/*.py'
    ]
    
    files_found = []
    
    for pattern in search_paths:
        for filepath in glob.glob(pattern, recursive=True):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'class OrderManager' in content or 'OrderManager(' in content:
                        files_found.append(filepath)
                        print(f"  Found OrderManager in: {filepath}")
            except:
                continue
                
    return files_found

def check_missing_methods(filepath):
    """Check which methods are missing from OrderManager"""
    print(f"\nChecking {filepath} for missing methods...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    missing_methods = []
    
    # Check for various method names that might be missing
    methods_to_check = [
        '_calculate_min_stop_distance',
        '_validate_stop_distance',
        '_calculate_position_size_with_min_stop',
        '_get_minimum_stops'
    ]
    
    for method in methods_to_check:
        if f'def {method}' not in content:
            missing_methods.append(method)
            print(f"  ❌ Missing: {method}")
        else:
            print(f"  ✅ Found: {method}")
            
    return missing_methods

def add_missing_methods(filepath, missing_methods):
    """Add missing methods to the OrderManager class"""
    
    if not missing_methods:
        print("\n✅ All methods already exist!")
        return True
        
    print(f"\nAdding {len(missing_methods)} missing method(s) to {filepath}...")
    
    # Create backup
    backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(filepath, 'r', encoding='utf-8') as f:
        original_content = f.read()
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original_content)
    print(f"  Backup created: {backup_path}")
    
    # Methods to add
    methods_code = '''
    def _calculate_min_stop_distance(self, symbol: str) -> float:
        """Calculate minimum stop distance based on broker requirements"""
        try:
            import MetaTrader5 as mt5
            
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                if 'logger' in dir(self):
                    self.logger.error(f"Cannot get symbol info for {symbol}")
                if symbol in ['XAUUSD', 'XAGUSD']:
                    return 0.50
                return 0.0010
                    
            stops_level = symbol_info.stops_level * symbol_info.point
            
            if symbol in ['XAUUSD', 'XAGUSD']:
                min_pips = 50
            elif 'JPY' in symbol:
                min_pips = 15
            else:
                min_pips = 10
                
            min_distance_from_pips = min_pips * symbol_info.point
            min_stop_distance = max(stops_level, min_distance_from_pips) * 1.1
            
            if 'logger' in dir(self):
                self.logger.info(f"{symbol}: Min stop distance: {min_stop_distance:.5f}")
            
            return min_stop_distance
            
        except Exception as e:
            if 'logger' in dir(self):
                self.logger.error(f"Error calculating min stop distance: {e}")
            if symbol in ['XAUUSD', 'XAGUSD']:
                return 0.50
            elif 'JPY' in symbol:
                return 0.15
            return 0.0010
    
    def _validate_stop_distance(self, symbol: str, stop_distance: float) -> float:
        """Validate and adjust stop distance to meet minimum requirements"""
        try:
            min_distance = self._calculate_min_stop_distance(symbol)
            
            if stop_distance < min_distance:
                if 'logger' in dir(self):
                    self.logger.warning(f"{symbol}: Adjusting stop distance to minimum")
                return min_distance
                
            max_sl_pips = 50
            if hasattr(self, 'config'):
                max_sl_pips = self.config.get('trading_rules', {}).get('stop_loss_rules', {}).get('max_sl_pips', 50)
            
            import MetaTrader5 as mt5
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                max_distance = max_sl_pips * symbol_info.point
                if stop_distance > max_distance:
                    return max_distance
                    
            return stop_distance
            
        except Exception as e:
            if 'logger' in dir(self):
                self.logger.error(f"Error validating stop distance: {e}")
            return stop_distance
    
    def _calculate_position_size_with_min_stop(self, symbol: str, stop_distance: float, risk_amount: float) -> float:
        """Calculate position size considering minimum stop distance requirements"""
        try:
            import MetaTrader5 as mt5
            
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return 0.01
                
            validated_stop_distance = self._validate_stop_distance(symbol, stop_distance)
            
            tick_value = symbol_info.trade_tick_value
            tick_size = symbol_info.trade_tick_size
            
            if tick_size == 0:
                return 0.01
                
            pip_value = (tick_value * symbol_info.point) / tick_size
            
            if validated_stop_distance > 0 and pip_value > 0:
                stop_in_pips = validated_stop_distance / symbol_info.point
                position_size = risk_amount / (stop_in_pips * pip_value)
                
                lot_step = symbol_info.volume_step
                position_size = round(position_size / lot_step) * lot_step
                
                position_size = max(symbol_info.volume_min, min(position_size, symbol_info.volume_max))
                
                return position_size
            else:
                return symbol_info.volume_min
                
        except Exception as e:
            if 'logger' in dir(self):
                self.logger.error(f"Error calculating position size: {e}")
            return 0.01
    
    def _get_minimum_stops(self, symbol: str) -> dict:
        """Get minimum stop and target distances for a symbol"""
        try:
            import MetaTrader5 as mt5
            
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                if symbol in ['XAUUSD', 'XAGUSD']:
                    return {'stop_distance': 0.50, 'target_distance': 1.00}
                else:
                    return {'stop_distance': 0.0010, 'target_distance': 0.0020}
                    
            min_stop_points = symbol_info.stops_level if symbol_info.stops_level > 0 else 10
            min_stop_distance = min_stop_points * symbol_info.point
            
            min_rr_ratio = 2.0
            if hasattr(self, 'config'):
                min_rr_ratio = self.config.get('trading_rules', {}).get('entry_rules', {}).get('min_risk_reward_ratio', 2.0)
            
            min_target_distance = min_stop_distance * min_rr_ratio
            
            return {
                'stop_distance': min_stop_distance,
                'target_distance': min_target_distance
            }
            
        except Exception as e:
            if 'logger' in dir(self):
                self.logger.error(f"Error getting minimum stops: {e}")
            return {'stop_distance': 0.0010, 'target_distance': 0.0020}
'''
    
    # Find the class definition and insert point
    lines = original_content.split('\n')
    insert_line = -1
    indent = '    '  # Default 4 spaces
    
    # Find OrderManager class and its indentation
    for i, line in enumerate(lines):
        if 'class OrderManager' in line:
            # Find the next method definition to determine indentation
            for j in range(i+1, min(i+50, len(lines))):
                if 'def ' in lines[j] and 'def __' in lines[j]:
                    # Found a method, get its indentation
                    indent = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
                    # Find end of __init__ or first method
                    for k in range(j+1, len(lines)):
                        if lines[k].strip() and not lines[k].startswith(indent):
                            if 'def ' in lines[k]:
                                insert_line = k
                                break
                        elif lines[k].strip() == '' and k+1 < len(lines) and 'def ' in lines[k+1]:
                            insert_line = k+1
                            break
                    break
            break
    
    if insert_line == -1:
        # Try to find any method in the file
        for i, line in enumerate(lines):
            if 'def place_order' in line or 'def execute_order' in line:
                insert_line = i
                break
    
    if insert_line == -1:
        print("  ❌ Could not find insertion point. Adding to end of file.")
        insert_line = len(lines) - 1
    
    # Add only missing methods
    methods_to_add = []
    for method in missing_methods:
        # Extract the specific method from methods_code
        pattern = f'def {method}.*?(?=\\n    def |$)'
        import re
        match = re.search(pattern, methods_code, re.DOTALL)
        if match:
            methods_to_add.append(match.group(0))
    
    # Insert the methods
    if methods_to_add:
        methods_text = '\n'.join(methods_to_add)
        lines.insert(insert_line, methods_text)
        
        # Write updated content
        updated_content = '\n'.join(lines)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(updated_content)
            
        print(f"  ✅ Added {len(missing_methods)} method(s) successfully!")
        return True
    
    return False

def find_and_fix_all_issues():
    """Main function to find and fix all OrderManager issues"""
    print("="*60)
    print("FX-Ai OrderManager Diagnostic and Fix Tool")
    print("="*60)
    
    # First, try to find OrderManager files
    files = find_order_manager_file()
    
    if not files:
        print("\n⚠️  Could not find OrderManager class automatically.")
        print("Please check these common locations:")
        print("  - core/order_manager.py")
        print("  - core/trading_engine.py")
        print("  - trading_engine.py")
        print("  - main.py")
        
        # Ask user for file path
        filepath = input("\nEnter the file path containing OrderManager class: ").strip()
        if os.path.exists(filepath):
            files = [filepath]
        else:
            print(f"❌ File not found: {filepath}")
            return False
    
    # Process each file found
    for filepath in files:
        print(f"\nProcessing: {filepath}")
        print("-"*40)
        
        # Check for missing methods
        missing = check_missing_methods(filepath)
        
        if missing:
            # Ask to fix
            response = input(f"\nFound {len(missing)} missing method(s). Add them? (y/n): ")
            if response.lower() == 'y':
                if add_missing_methods(filepath, missing):
                    print("\n✅ Fix applied successfully!")
                    print("Please restart FX-Ai to test the changes.")
                else:
                    print("\n❌ Failed to apply fix.")
        else:
            print("\n✅ All required methods already exist!")
    
    return True

def test_methods():
    """Test if the methods work correctly with MT5"""
    print("\nTesting methods with MT5...")
    
    try:
        import MetaTrader5 as mt5
        
        if not mt5.initialize():
            print("❌ MT5 not initialized. Please ensure MT5 is running.")
            return
            
        # Create a simple test class
        class TestOrderManager:
            def __init__(self):
                self.config = {'trading_rules': {'stop_loss_rules': {'max_sl_pips': 50}}}
                
            exec(open('fix_order_manager.py').read())  # Load our methods
            
        # Test the methods
        manager = TestOrderManager()
        
        # Test with EURUSD
        symbol = 'EURUSD'
        min_dist = manager._calculate_min_stop_distance(symbol)
        print(f"  {symbol} min stop distance: {min_dist:.5f}")
        
        # Test with XAUUSD
        symbol = 'XAUUSD'
        min_dist = manager._calculate_min_stop_distance(symbol)
        print(f"  {symbol} min stop distance: {min_dist:.5f}")
        
        print("\n✅ Methods tested successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            test_methods()
        elif sys.argv[1] == '--help':
            print("Usage:")
            print("  python fix_order_manager_tool.py        - Find and fix issues")
            print("  python fix_order_manager_tool.py --test - Test the methods")
            print("  python fix_order_manager_tool.py --help - Show this help")
        else:
            # Assume it's a file path
            if os.path.exists(sys.argv[1]):
                missing = check_missing_methods(sys.argv[1])
                if missing:
                    add_missing_methods(sys.argv[1], missing)
            else:
                print(f"File not found: {sys.argv[1]}")
    else:
        find_and_fix_all_issues()
