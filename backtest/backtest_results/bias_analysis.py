import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

def analyze_training_data_bias():
    """Analyze potential training data biases that could cause the model issues"""

    print("=== TRAINING DATA BIAS ANALYSIS ===")
    print("=" * 40)

    # Check if we have training data or model files
    model_dir = Path("../../models")
    data_dir = Path("../../data")

    print("\nChecking for model and training data files...")

    # Look for model files
    if model_dir.exists():
        model_files = list(model_dir.glob("*"))
        print(f"Found model files: {[f.name for f in model_files]}")

        # Look for training data
        training_files = list(model_dir.glob("*training*")) + list(model_dir.glob("*train*"))
        if training_files:
            print(f"Training data files: {[f.name for f in training_files]}")
        else:
            print("No training data files found in models directory")
    else:
        print("Models directory not found")

    # Look for data directory
    if data_dir.exists():
        data_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.pkl"))
        print(f"Data files: {[f.name for f in data_files[:5]]}")  # Show first 5
    else:
        print("Data directory not found")

    # Try to load any available training data
    training_data = None
    for potential_file in ["training_data.csv", "train.csv", "features.csv"]:
        if (model_dir / potential_file).exists():
            try:
                training_data = pd.read_csv(model_dir / potential_file)
                print(f"\nLoaded training data from {potential_file}")
                break
            except Exception as e:
                print(f"Failed to load {potential_file}: {e}")

    if training_data is None:
        print("\nNo training data found. Analyzing based on trade patterns...")

        # Load trade data to infer training data issues
        trades_file = Path("trades.csv")
        if trades_file.exists():
            trades_df = pd.read_csv(trades_file)

            # Analyze direction distribution in trades
            print("\nTRADE DIRECTION ANALYSIS:")
            print("-" * 25)

            direction_dist = trades_df['direction'].value_counts()
            total_trades = len(trades_df)
            print(f"Total trades: {total_trades}")
            print(f"Direction distribution: {direction_dist.to_dict()}")

            if 'sell' in direction_dist and direction_dist['sell'] == total_trades:
                print("⚠️  CRITICAL: ALL TRADES ARE SELLS - MODEL HAS SEVERE DIRECTION BIAS")
            elif 'buy' in direction_dist and direction_dist['buy'] == total_trades:
                print("⚠️  CRITICAL: ALL TRADES ARE BUYS - MODEL HAS SEVERE DIRECTION BIAS")

            # Analyze by symbol groups
            jpy_symbols = ['USDJPY', 'CHFJPY', 'AUDJPY', 'CADJPY', 'EURJPY', 'GBPJPY', 'NZDJPY']
            cad_symbols = ['CADJPY', 'CADCHF', 'USDCAD', 'GBPCAD', 'AUDCAD', 'EURCAD', 'NZDCAD']

            for group_name, symbols in [("JPY", jpy_symbols), ("CAD", cad_symbols)]:
                group_trades = trades_df[trades_df['symbol'].isin(symbols)]
                if not group_trades.empty:
                    group_directions = group_trades['direction'].value_counts()
                    print(f"\n{group_name} crosses direction distribution: {group_directions.to_dict()}")

                    if len(group_directions) == 1:
                        print(f"⚠️  {group_name} crosses: ONLY {list(group_directions.keys())[0].upper()} TRADES")

    else:
        # Analyze actual training data
        print(f"\nTraining data shape: {training_data.shape}")
        print(f"Columns: {list(training_data.columns)}")

        # Check for target/label column
        target_cols = [col for col in training_data.columns if 'target' in col.lower() or 'label' in col.lower() or 'direction' in col.lower()]
        if target_cols:
            target_col = target_cols[0]
            print(f"\nTarget column: {target_col}")
            target_dist = training_data[target_col].value_counts()
            print(f"Target distribution: {target_dist.to_dict()}")

            # Check class balance
            if len(target_dist) == 2:
                minority_class = target_dist.min()
                majority_class = target_dist.max()
                imbalance_ratio = majority_class / minority_class
                print(".2f")

                if imbalance_ratio > 3:
                    print("⚠️  SEVERE CLASS IMBALANCE - This could cause bias toward majority class")
        else:
            print("No obvious target column found")

    # Check model configuration
    config_files = list(Path("../../config").glob("*.json")) if Path("../../config").exists() else []
    if config_files:
        print("\nMODEL CONFIGURATION:")
        print("-" * 22)
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)

                if 'model' in config:
                    model_config = config['model']
                    print(f"\n{config_file.name} model config:")
                    for key, value in model_config.items():
                        if not key.startswith('_'):  # Skip private keys
                            print(f"  {key}: {value}")
            except Exception as e:
                print(f"Failed to read {config_file.name}: {e}")

    # Generate recommendations
    print("\n=== DIAGNOSIS & RECOMMENDATIONS ===")
    print("=" * 38)

    print("\nLIKELY CAUSES OF THE JPY CROSS PROBLEM:")
    print("1. Training data bias: Model trained predominantly on sell signals")
    print("2. Feature inadequacy: Technical indicators don't work well for JPY pairs")
    print("3. Currency-specific patterns: JPY has unique market characteristics")
    print("4. Overfitting: Model learned patterns that don't generalize to JPY")

    print("\nRECOMMENDED FIXES:")
    print("1. Retrain model with balanced buy/sell training data")
    print("2. Add JPY-specific features or separate JPY model")
    print("3. Exclude JPY crosses from trading universe")
    print("4. Implement currency-specific confidence thresholds")
    print("5. Add more diverse training data including various market conditions")

if __name__ == "__main__":
    analyze_training_data_bias()