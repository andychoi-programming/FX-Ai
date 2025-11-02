"""
Monitor optimization progress in real-time
Shows current symbol, progress, and latest results
"""

import time
from pathlib import Path

def monitor_progress():
    log_file = Path("robust_optimization.log")
    
    if not log_file.exists():
        print("Log file not found yet...")
        return
    
    print("\n" + "=" * 100)
    print("OPTIMIZATION PROGRESS MONITOR")
    print("=" * 100 + "\n")
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find current symbol
    current_symbol = None
    progress = None
    best_found = []
    
    for line in lines:
        if "OPTIMIZING:" in line:
            current_symbol = line.split("OPTIMIZING:")[-1].strip()
        elif "Training progress:" in line:
            progress = line.split("Training progress:")[-1].strip()
        elif "NEW BEST @" in line:
            best_found.append(line)
    
    if current_symbol:
        print(f"Current Symbol: {current_symbol}")
    
    if progress:
        print(f"Progress: {progress}")
        completed, total = progress.split('/')
        pct = (int(completed) / int(total)) * 100
        print(f"Percentage: {pct:.1f}%")
    
    if best_found:
        print(f"\nBest Results Found So Far: {len(best_found)}")
        print("\nLatest Improvements:")
        for line in best_found[-5:]:  # Show last 5
            print("  " + line.split(" - INFO - ")[-1].strip())
    
    # Count completed symbols
    completed_symbols = []
    for line in lines:
        if "PASSED ALL VALIDATIONS" in line or "FAILED VALIDATION" in line:
            symbol = line.split("✓✓✓")[-1].split("✗✗✗")[-1].split("PASSED")[0].split("FAILED")[0].strip()
            completed_symbols.append(symbol)
    
    if completed_symbols:
        print(f"\n{'=' * 100}")
        print(f"Completed Symbols: {len(completed_symbols)}/30")
        print(f"{'=' * 100}")
        for sym in completed_symbols:
            print(f"  ✓ {sym}")
    
    print("\n" + "=" * 100)
    print(f"Total Log Lines: {len(lines)}")
    print(f"Log File: {log_file}")
    print("=" * 100)

if __name__ == "__main__":
    monitor_progress()
