import os
import re
import glob

search_dir = r"C:\Users\andyc\python\FX-Ai\core"
target_file = "order_executor.py"

print("="*70)
print("SEARCHING FOR ALL SELF METHOD CALLS IN ORDER_EXECUTOR.PY")
print("="*70)

# Read order_executor.py
filepath = os.path.join(search_dir, target_file)
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()
    lines = content.split('\n')

# Find all self.method_name() calls
method_calls = set()
for line in lines:
    # Find all self._method() or self.method() patterns
    matches = re.findall(r'self\.([_a-zA-Z][_a-zA-Z0-9]*)\(', line)
    method_calls.update(matches)

# Find all method definitions in the same file
method_definitions = set()
for line in lines:
    match = re.match(r'\s*(async\s+)?def\s+([_a-zA-Z][_a-zA-Z0-9]*)\(', line)
    if match:
        method_definitions.add(match.group(2))

# Find methods that are called but not defined
missing_methods = method_calls - method_definitions

print(f"\nSTATISTICS:")
print(f"  Total method calls found: {len(method_calls)}")
print(f"  Methods defined in file: {len(method_definitions)}")
print(f"  Potentially missing: {len(missing_methods)}")

print(f"\nMISSING METHODS (called but not defined):")
for method in sorted(missing_methods):
    print(f"  - self.{method}()")

    # Find where it's called
    print(f"    Called in lines:")
    for line_num, line in enumerate(lines, 1):
        if f'self.{method}(' in line and not line.strip().startswith('#'):
            print(f"      Line {line_num}: {line.strip()[:80]}")

print("\n" + "="*70)
print("RECOMMENDED FIXES:")
print("="*70)

common_fixes = {
    '_get_atr': 'Call from self.technical_analyzer.get_atr()',
    '_calculate_min_stop_distance': 'Add method or call from risk_manager',
    '_validate_order': 'Add method or call from risk_manager.validate_position_size()',
    '_get_symbol_info': 'Call mt5.symbol_info() directly',
    '_calculate_position_size': 'Call from self.risk_manager.calculate_position_size()',
}

for method in sorted(missing_methods):
    if method in common_fixes:
        print(f"\n{method}:")
        print(f"  -> {common_fixes[method]}")
    else:
        print(f"\n{method}:")
        print(f"  -> Add this method to OrderExecutor class")