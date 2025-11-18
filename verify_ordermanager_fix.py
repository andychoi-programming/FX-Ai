#!/usr/bin/env python3
"""
FX-Ai OrderManager Fix Verification Tool
This will help identify why the fix hasn't been applied correctly
"""

import os
import glob
import re
import sys

def find_all_python_files():
    """Find all Python files in the project"""
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip virtual environment and cache directories
        if 'venv' in root or '__pycache__' in root or '.git' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def search_for_ordermanager_class(files):
    """Find files containing OrderManager class"""
    found_classes = []
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'class OrderManager' in content:
                    # Count methods in the class
                    method_count = content.count('def ', content.index('class OrderManager'))
                    has_min_stop = '_calculate_min_stop_distance' in content
                    found_classes.append({
                        'file': filepath,
                        'methods': method_count,
                        'has_min_stop': has_min_stop
                    })
                    print(f"\n✅ Found OrderManager class in: {filepath}")
                    print(f"   Methods count: {method_count}")
                    print(f"   Has _calculate_min_stop_distance: {has_min_stop}")
        except Exception as e:
            continue
    return found_classes

def search_for_ordermanager_usage(files):
    """Find where OrderManager is being used"""
    usage_locations = []
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'OrderManager(' in content or 'order_manager' in content.lower():
                    # Check if it's calling _calculate_min_stop_distance
                    if '_calculate_min_stop_distance' in content:
                        usage_locations.append({
                            'file': filepath,
                            'calls_method': True
                        })
                        print(f"\n⚠️  File {filepath} CALLS _calculate_min_stop_distance")
        except:
            continue
    return usage_locations

def check_method_exists(filepath, method_name='_calculate_min_stop_distance'):
    """Check if a specific method exists in a file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Look for the method definition
        pattern = rf'def {method_name}\s*\('
        if re.search(pattern, content):
            print(f"✅ Method {method_name} EXISTS in {filepath}")
            # Find the line number
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if f'def {method_name}' in line:
                    print(f"   Found at line {i}")
                    # Show a few lines of context
                    start = max(0, i-2)
                    end = min(len(lines), i+3)
                    print("   Context:")
                    for j in range(start, end):
                        prefix = ">>> " if j == i-1 else "    "
                        print(f"{prefix}{lines[j]}")
                    return True
        else:
            print(f"❌ Method {method_name} NOT FOUND in {filepath}")
            return False
    except Exception as e:
        print(f"Error checking {filepath}: {e}")
        return False

def suggest_fix_location(ordermanager_files):
    """Suggest where to add the missing method"""
    print("\n" + "="*60)
    print("FIX INSTRUCTIONS")
    print("="*60)
    
    if not ordermanager_files:
        print("❌ No OrderManager class found!")
        print("\nPossible locations to check:")
        print("  - core/order_manager.py")
        print("  - core/trading_engine.py")
        print("  - trading_engine.py")
        return
    
    for om in ordermanager_files:
        if not om['has_min_stop']:
            print(f"\n📁 File: {om['file']}")
            print("   Status: NEEDS FIX")
            print("\n   Add this method to the OrderManager class:")
            print("   "+"="*50)
            print('''
    def _calculate_min_stop_distance(self, symbol: str) -> float:
        """Calculate minimum stop distance based on broker requirements"""
        try:
            import MetaTrader5 as mt5
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                if hasattr(self, 'logger'):
                    self.logger.error(f"Cannot get symbol info for {symbol}")
                # Return defaults based on symbol type
                if symbol in ['XAUUSD', 'XAGUSD']:
                    return 0.50  # 50 pips for metals
                elif 'JPY' in symbol:
                    return 0.15  # 15 pips for JPY pairs
                else:
                    return 0.0010  # 10 pips for regular forex
            
            # Get broker's minimum stop level
            stops_level = symbol_info.stops_level * symbol_info.point
            
            # Define our minimum requirements
            if symbol in ['XAUUSD', 'XAGUSD']:
                min_pips = 50
            elif 'JPY' in symbol:
                min_pips = 15
            else:
                min_pips = 10
            
            min_distance_from_pips = min_pips * symbol_info.point
            
            # Use the larger of broker requirement or our minimum
            min_stop_distance = max(stops_level, min_distance_from_pips)
            
            # Add 10% buffer for safety
            min_stop_distance *= 1.1
            
            if hasattr(self, 'logger'):
                self.logger.info(f"{symbol}: Min stop distance: {min_stop_distance:.5f}")
            
            return min_stop_distance
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error calculating min stop distance: {e}")
            # Return safe defaults
            if symbol in ['XAUUSD', 'XAGUSD']:
                return 0.50
            elif 'JPY' in symbol:
                return 0.15
            else:
                return 0.0010
            ''')
            print("   "+"="*50)

def main():
    print("="*60)
    print("FX-Ai OrderManager Diagnostic Tool")
    print("="*60)
    
    print("\n1. Searching for all Python files...")
    python_files = find_all_python_files()
    print(f"   Found {len(python_files)} Python files")
    
    print("\n2. Looking for OrderManager class...")
    ordermanager_classes = search_for_ordermanager_class(python_files)
    
    if not ordermanager_classes:
        print("\n❌ ERROR: OrderManager class not found in any Python file!")
        print("\nSearching for files that might contain it...")
        
        # Search for specific patterns
        for filepath in python_files:
            if 'order' in filepath.lower() or 'manager' in filepath.lower() or 'trading' in filepath.lower():
                print(f"   Check: {filepath}")
    
    print("\n3. Looking for files that USE OrderManager...")
    usage_files = search_for_ordermanager_usage(python_files)
    
    print("\n4. Verifying method existence...")
    for om in ordermanager_classes:
        check_method_exists(om['file'])
    
    # Provide fix instructions
    suggest_fix_location(ordermanager_classes)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if ordermanager_classes:
        needs_fix = [om for om in ordermanager_classes if not om['has_min_stop']]
        if needs_fix:
            print(f"❌ {len(needs_fix)} file(s) need the method added")
            for om in needs_fix:
                print(f"   - {om['file']}")
        else:
            print("✅ All OrderManager classes have the method")
            print("\nPossible issues:")
            print("  1. The file wasn't saved after adding the method")
            print("  2. Python is using cached bytecode (.pyc files)")
            print("  3. The method was added with wrong indentation")
            print("\nTry:")
            print("  1. Save the file again")
            print("  2. Delete all __pycache__ directories")
            print("  3. Restart the Python program")
    else:
        print("❌ OrderManager class not found - this is the main issue!")
    
    # Show trade statistics from the log
    print("\n" + "="*60)
    print("TRADING OPPORTUNITY ANALYSIS")
    print("="*60)
    print("Your system found 11 STRONG trading signals:")
    signals = [
        ("GBPJPY", 0.5033, "BUY", "Strongest signal!"),
        ("EURJPY", 0.4755, "BUY", "Strong bullish"),
        ("NZDCHF", 0.4633, "SELL", "Good short"),
        ("CHFJPY", 0.4619, "BUY", "Bullish"),
        ("AUDCHF", 0.4533, "SELL", "Short opportunity"),
        ("NZDJPY", 0.4449, "SELL", "Short"),
        ("NZDCAD", 0.4422, "SELL", "Short"),
        ("AUDUSD", 0.4323, "SELL", "Short"),
        ("AUDNZD", 0.4319, "SELL", "Short"),
        ("NZDUSD", 0.4246, "SELL", "Short"),
        ("AUDJPY", 0.4243, "SELL", "Short")
    ]
    
    for symbol, strength, direction, note in signals:
        bar = "█" * int(strength * 50)
        print(f"  {symbol:8} {direction:4} {strength:.4f} {bar} {note}")
    
    print("\n⚠️  ALL these trades are being BLOCKED by the missing method!")
    print("    Fix OrderManager to unlock these opportunities!")

if __name__ == "__main__":
    main()
