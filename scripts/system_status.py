#!/usr/bin/env python3
"""
FX-Ai System Status Checker
Unified utility for checking system status and parameters
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config_manager import ConfigManager
from core.parameter_manager import ParameterManager

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)

def check_config_status(config: ConfigManager):
    """Check configuration status"""
    print_header("CONFIGURATION STATUS")

    print(f"Mode: {config.mode.upper()}")
    print(f"Dry Run: {config.get('trading.dry_run', False)}")
    print(f"Symbols: {len(config.get_symbols())}")
    print(f"Risk per Trade: ${config.get('trading.risk_per_trade', 0)}")
    print(f"Max Daily Loss: ${config.get('trading.max_daily_loss', 0)}")

    # Check MT5 config
    mt5_config = config.get_mt5_config()
    print(f"MT5 Login: {'✓ Set' if mt5_config.get('login') else '✗ Not set'}")
    print(f"MT5 Server: {'✓ Set' if mt5_config.get('server') else '✗ Not set'}")

def check_parameter_status(param_manager: ParameterManager):
    """Check parameter optimization status"""
    print_header("PARAMETER OPTIMIZATION STATUS")

    summary = param_manager.get_parameter_summary()

    print(f"Total Symbols with Parameters: {summary['total_symbols']}")

    if summary['symbols_by_timeframe']:
        print("\nParameters by Timeframe:")
        for timeframe, symbols in summary['symbols_by_timeframe'].items():
            print(f"  {timeframe}: {len(symbols)} symbols")

    if summary['last_updates']:
        print("\nRecent Updates:")
        # Sort by last updated and show top 5
        recent = sorted(summary['last_updates'],
                       key=lambda x: x['last_updated'],
                       reverse=True)[:5]
        for update in recent:
            print(f"  {update['symbol']} {update['timeframe']}: {update['last_updated'][:19]}")

def check_model_status():
    """Check ML model status"""
    print_header("ML MODEL STATUS")

    models_dir = Path('models')
    if not models_dir.exists():
        print("❌ Models directory not found")
        return

    # Check for model files
    model_files = list(models_dir.glob('*.h5')) + list(models_dir.glob('*.pkl'))
    print(f"Model Files Found: {len(model_files)}")

    if model_files:
        print("\nModel Files:")
        for model_file in sorted(model_files)[:10]:  # Show first 10
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(".2f")
        if len(model_files) > 10:
            print(f"  ... and {len(model_files) - 10} more")

def check_database_status():
    """Check database status"""
    print_header("DATABASE STATUS")

    # Check for database files
    db_files = ['performance_history.db', 'learning_database.db']

    for db_file in db_files:
        if Path(db_file).exists():
            size_mb = Path(db_file).stat().st_size / (1024 * 1024)
            print(".2f")
        else:
            print(f"❌ {db_file} not found")

def check_system_health():
    """Check overall system health"""
    print_header("SYSTEM HEALTH CHECK")

    issues = []

    # Check required directories
    required_dirs = ['config', 'models', 'logs', 'data']
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            issues.append(f"Missing directory: {dir_name}")

    # Check required files
    required_files = ['config/config.json', 'main.py']
    for file_name in required_files:
        if not Path(file_name).exists():
            issues.append(f"Missing file: {file_name}")

    # Check Python environment
    try:
        import MetaTrader5
        print("✓ MetaTrader5 library available")
    except ImportError:
        issues.append("MetaTrader5 library not installed")

    try:
        import tensorflow
        print("✓ TensorFlow library available")
    except ImportError:
        issues.append("TensorFlow library not installed")

    if issues:
        print("❌ Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ All system checks passed")

def show_optimization_progress():
    """Show parameter optimization progress"""
    print_header("OPTIMIZATION PROGRESS")

    param_file = Path('models/parameter_optimization/optimal_parameters.json')
    if not param_file.exists():
        print("❌ Parameter file not found")
        return

    try:
        with open(param_file, 'r') as f:
            params = json.load(f)

        total_symbols = len(params)
        optimized_symbols = 0
        total_timeframes = 0

        for symbol, timeframes in params.items():
            if timeframes:  # Has at least one timeframe
                optimized_symbols += 1
                total_timeframes += len(timeframes)

        print(f"Symbols with Parameters: {optimized_symbols}/{total_symbols}")
        print(f"Total Timeframes Optimized: {total_timeframes}")
        print(".1f")

        if optimized_symbols > 0:
            print("\nOptimization Completion by Timeframe:")
            timeframe_counts = {}
            for symbol, timeframes in params.items():
                for timeframe in timeframes.keys():
                    timeframe_counts[timeframe] = timeframe_counts.get(timeframe, 0) + 1

            for timeframe, count in sorted(timeframe_counts.items()):
                percentage = (count / optimized_symbols) * 100
                print(".1f")

    except Exception as e:
        print(f"❌ Error reading parameter file: {e}")

def main():
    """Main status checker function"""
    parser = argparse.ArgumentParser(description='FX-Ai System Status Checker')
    parser.add_argument('--mode', choices=['live', 'backtest'], default='live',
                       help='System mode (default: live)')
    parser.add_argument('--check', choices=['all', 'config', 'params', 'models', 'db', 'health', 'progress'],
                       default='all', help='Specific check to run')

    args = parser.parse_args()

    print(f"FX-Ai System Status - {args.mode.upper()} MODE")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        config = ConfigManager(mode=args.mode)
        param_manager = ParameterManager(config)

        if args.check in ['all', 'config']:
            check_config_status(config)

        if args.check in ['all', 'params']:
            check_parameter_status(param_manager)

        if args.check in ['all', 'models']:
            check_model_status()

        if args.check in ['all', 'db']:
            check_database_status()

        if args.check in ['all', 'health']:
            check_system_health()

        if args.check in ['all', 'progress']:
            show_optimization_progress()

    except Exception as e:
        print(f"❌ Error checking system status: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()