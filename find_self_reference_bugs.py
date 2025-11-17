#!/usr/bin/env python3
"""
Diagnostic script to find self.order_executor bugs in FX-Ai codebase
"""

import os
import re
from pathlib import Path

def find_self_order_executor_bugs():
    """Find all instances of self.order_executor in the codebase"""

    # Get the project root directory
    project_root = Path(__file__).parent

    # Files to search
    search_files = [
        'core/order_executor.py',
        'core/trading_engine.py',
        'app/application.py',
        'main.py'
    ]

    print("🔍 SEARCHING FOR self.order_executor BUGS")
    print("=" * 50)

    total_bugs = 0

    for file_path in search_files:
        full_path = project_root / file_path
        if not full_path.exists():
            continue

        print(f"\n📁 Checking {file_path}:")

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            bugs_in_file = 0

            for line_num, line in enumerate(lines, 1):
                if 'self.order_executor' in line:
                    # Check if this is in the OrderManager class or OrderExecutor class
                    # Look for class context by checking previous lines
                    class_context = "Unknown"
                    for i in range(max(0, line_num-20), line_num):
                        if 'class OrderManager:' in lines[i]:
                            class_context = "OrderManager (OK)"
                            break
                        elif 'class OrderExecutor:' in lines[i]:
                            class_context = "OrderExecutor (BUG!)"
                            break

                    print(f"  Line {line_num}: {line.strip()}")
                    print(f"    Context: {class_context}")

                    if "BUG!" in class_context:
                        bugs_in_file += 1
                        total_bugs += 1

            if bugs_in_file == 0:
                print("  ✅ No bugs found in this file")

        except Exception as e:
            print(f"  ❌ Error reading {file_path}: {e}")

    print("\n" + "=" * 50)
    print(f"🎯 TOTAL BUGS FOUND: {total_bugs}")

    if total_bugs > 0:
        print("\n🔧 FIX REQUIRED:")
        print("In OrderExecutor class, change:")
        print("  self.order_executor.method()  →  self.method()")
        print("  OR")
        print("  self.order_executor.method()  →  self.order_manager.method()")
    else:
        print("\n✅ NO BUGS FOUND - System should work correctly!")

    return total_bugs

if __name__ == "__main__":
    find_self_order_executor_bugs()