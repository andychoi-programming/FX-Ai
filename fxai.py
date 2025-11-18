#!/usr/bin/env python3
"""
FX-Ai Unified Launcher
Single entry point for all FX-Ai operations
"""

import argparse
import sys
import os
from pathlib import Path

def activate_venv():
    """Activate virtual environment"""
    venv_path = Path('venv/Scripts/activate.bat')
    if venv_path.exists():
        os.system(f'call {venv_path}')
        return True
    return False

def run_command(command: str, description: str):
    """Run a command with description"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print('='*60)
    exit_code = os.system(command)
    return exit_code == 0

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description='FX-Ai Unified Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fxai.py run live          # Run live trading
  python fxai.py run backtest      # Run backtesting
  python fxai.py train             # Train ML models
  python fxai.py emergency-stop    # Emergency stop all trades
  python fxai.py status            # Show system status
        """
    )

    parser.add_argument('command', choices=['run', 'train', 'emergency-stop', 'status'],
                       help='Command to execute')
    parser.add_argument('mode', nargs='?', choices=['live', 'backtest'],
                       help='Mode for run command')
    parser.add_argument('--no-venv', action='store_true',
                       help='Skip virtual environment activation')

    args = parser.parse_args()

    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Activate virtual environment unless disabled
    if not args.no_venv and not activate_venv():
        print("Warning: Could not activate virtual environment")

    # Execute commands
    if args.command == 'run':
        if not args.mode:
            print("Error: Mode required for run command (live or backtest)")
            sys.exit(1)

        print(f"Starting FX-Ai in {args.mode} mode...")

        if args.mode == 'live':
            success = run_command('python main.py', 'FX-AI LIVE TRADING')
        else:
            success = run_command('python main.py --mode backtest', 'FX-AI BACKTESTING')

    elif args.command == 'train':
        success = run_command('python scripts/train_models.py', 'FX-AI MODEL TRAINING')

    elif args.command == 'emergency-stop':
        print("\n" + "!"*60)
        print("*** FX-AI EMERGENCY STOP ***")
        print("*** IMMEDIATELY CLOSING ALL POSITIONS AND CANCELING ALL ORDERS ***")
        print("*** THIS AFFECTS ALL TRADING SYSTEMS AND MANUAL TRADES ***")
        print("!"*60)

        confirm = input("\nAre you sure you want to continue? (type 'YES' to confirm): ")
        if confirm.upper() == 'YES':
            success = run_command('python live_trading/emergency_stop.py', 'EMERGENCY STOP')
        else:
            print("Emergency stop cancelled.")
            success = True

    elif args.command == 'status':
        success = run_command('python scripts/system_status.py', 'FX-AI SYSTEM STATUS')

    if success:
        print("\n✅ Command completed successfully")
    else:
        print("\n❌ Command failed")
        sys.exit(1)

if __name__ == "__main__":
    main()