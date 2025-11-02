"""
Quick status check for optimization progress
Run this periodically to see how far along the optimization is
"""

from pathlib import Path
import json

def quick_status():
    log_file = Path("robust_optimization.log")
    
    if not log_file.exists():
        print("❌ Optimization not started yet")
        return
    
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find current symbol
    lines = content.split('\n')
    current_symbol = None
    current_progress = None
    latest_best = None
    completed = []
    
    for line in reversed(lines):
        if not current_progress and "Training progress:" in line:
            current_progress = line.split("Training progress:")[-1].strip()
        if not latest_best and "NEW BEST @" in line:
            latest_best = line
        if not current_symbol and "OPTIMIZING:" in line:
            current_symbol = line.split("OPTIMIZING:")[-1].strip()
            break
    
    # Count completed
    for line in lines:
        if "PASSED ALL VALIDATIONS" in line:
            sym = line.split("✓✓✓")[1].split("PASSED")[0].strip()
            completed.append((sym, "✓ PASSED"))
        elif "FAILED VALIDATION" in line:
            sym = line.split("✗✗✗")[1].split("FAILED")[0].strip()
            completed.append((sym, "✗ FAILED"))
    
    print("\n" + "=" * 80)
    print("🔄 OPTIMIZATION STATUS")
    print("=" * 80)
    
    print(f"\n📊 Overall Progress: {len(completed)}/30 symbols completed")
    
    if current_symbol:
        print(f"\n🎯 Current Symbol: {current_symbol}")
        
        if current_progress:
            done, total = current_progress.split('/')
            pct = (int(done) / int(total)) * 100
            bar_length = 40
            filled = int(bar_length * int(done) / int(total))
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"   Progress: [{bar}] {pct:.1f}%")
            print(f"   Parameters: {done}/{total}")
            
            # Estimate time remaining
            if int(done) > 0:
                # Each symbol takes about 4 minutes for 16K combinations
                symbols_remaining = 30 - len(completed)
                mins_per_symbol = 4
                est_mins = symbols_remaining * mins_per_symbol
                est_hours = est_mins / 60
                print(f"   Estimated time remaining: ~{est_hours:.1f} hours ({est_mins:.0f} mins)")
    
    if latest_best:
        print(f"\n💎 Latest Best Result:")
        info = latest_best.split(" - INFO - ")[-1] if " - INFO - " in latest_best else latest_best
        print(f"   {info.strip()}")
    
    if completed:
        print(f"\n✅ Completed Symbols ({len(completed)}):")
        for sym, status in completed[-10:]:  # Show last 10
            print(f"   {status} {sym}")
        if len(completed) > 10:
            print(f"   ... and {len(completed) - 10} more")
    
    print("\n" + "=" * 80)
    print("💡 To see full details, check: robust_optimization.log")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    quick_status()
