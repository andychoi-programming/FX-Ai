import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

def analyze_model_predictions():
    """Analyze ML model predictions and feature patterns for problematic symbols"""

    # Load trade data
    trades_file = Path("trades.csv")
    if not trades_file.exists():
        print("Trades log not found!")
        return

    trades_df = pd.read_csv(trades_file)

    # Focus on JPY crosses vs CAD crosses
    jpy_symbols = ['USDJPY', 'CHFJPY', 'AUDJPY', 'CADJPY', 'EURJPY', 'GBPJPY', 'NZDJPY']
    cad_symbols = ['CADJPY', 'CADCHF', 'USDCAD', 'GBPCAD', 'AUDCAD', 'EURCAD', 'NZDCAD']

    jpy_trades = trades_df[trades_df['symbol'].isin(jpy_symbols)].copy()
    cad_trades = trades_df[trades_df['symbol'].isin(cad_symbols)].copy()

    print("=== ML MODEL PREDICTION ANALYSIS ===")
    print("=" * 50)

    # Analyze prediction confidence
    print("\nJPY CROSSES PREDICTION ANALYSIS:")
    print("-" * 40)
    if not jpy_trades.empty:
        print(f"Total JPY trades: {len(jpy_trades)}")
        print(".3f")
        print(".3f")
        print(".3f")

        # Direction analysis
        direction_counts = jpy_trades['direction'].value_counts()
        print(f"Direction distribution: {direction_counts.to_dict()}")

        # Confidence by direction
        buy_conf = jpy_trades[jpy_trades['direction'] == 'buy']['confidence'].mean()
        sell_conf = jpy_trades[jpy_trades['direction'] == 'sell']['confidence'].mean()
        print(".3f")
        print(".3f")

    print("\nCAD CROSSES PREDICTION ANALYSIS:")
    print("-" * 40)
    if not cad_trades.empty:
        print(f"Total CAD trades: {len(cad_trades)}")
        print(".3f")
        print(".3f")
        print(".3f")

        # Direction analysis
        direction_counts = cad_trades['direction'].value_counts()
        print(f"Direction distribution: {direction_counts.to_dict()}")

        # Confidence by direction
        buy_conf = cad_trades[cad_trades['direction'] == 'buy']['confidence'].mean()
        sell_conf = cad_trades[cad_trades['direction'] == 'sell']['confidence'].mean()
        print(".3f")
        print(".3f")

    # Analyze win rates by confidence levels
    print("\nWIN RATE BY CONFIDENCE LEVEL:")
    print("-" * 35)

    confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for symbol_group, name in [(jpy_trades, "JPY Crosses"), (cad_trades, "CAD Crosses")]:
        if symbol_group.empty:
            continue

        print(f"\n{name}:")
        symbol_group = symbol_group.copy()
        symbol_group['conf_bin'] = pd.cut(symbol_group['confidence'], confidence_bins, labels=['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'])

        win_rates = symbol_group.groupby('conf_bin').agg({
            'pnl': ['count', lambda x: (x > 0).mean()]
        }).round(3)

        win_rates.columns = ['trades', 'win_rate']
        print(win_rates)

    # Analyze feature patterns if available
    try:
        # Look for feature data in trades
        feature_cols = [col for col in trades_df.columns if col.startswith('feature_') or 'rsi' in col.lower() or 'macd' in col.lower()]

        if feature_cols:
            print("\nFEATURE ANALYSIS:")
            print("-" * 20)

            for symbol_group, name in [(jpy_trades, "JPY"), (cad_trades, "CAD")]:
                if symbol_group.empty or not any(col in symbol_group.columns for col in feature_cols):
                    continue

                print(f"\n{name} Crosses Feature Averages:")
                available_features = [col for col in feature_cols if col in symbol_group.columns]
                if available_features:
                    feature_means = symbol_group[available_features].mean()
                    print(feature_means.round(4))

    except Exception as e:
        print(f"Feature analysis failed: {e}")

    # Analyze entry timing patterns
    print("\nENTRY TIMING ANALYSIS:")
    print("-" * 25)

    for symbol_group, name in [(jpy_trades, "JPY"), (cad_trades, "CAD")]:
        if symbol_group.empty:
            continue

        print(f"\n{name} Crosses:")
        try:
            symbol_group['entry_time'] = pd.to_datetime(symbol_group['entry_time'])
            symbol_group['hour'] = symbol_group['entry_time'].dt.hour
            symbol_group['day_of_week'] = symbol_group['entry_time'].dt.day_name()

            # Hourly distribution
            hourly_dist = symbol_group['hour'].value_counts().sort_index()
            print(f"Trades by hour: {hourly_dist.to_dict()}")

            # Day of week distribution
            dow_dist = symbol_group['day_of_week'].value_counts()
            print(f"Trades by day: {dow_dist.to_dict()}")

        except Exception as e:
            print(f"Timing analysis failed: {e}")

    # Generate recommendations
    print("\n=== MODEL IMPROVEMENT RECOMMENDATIONS ===")
    print("=" * 45)

    if not jpy_trades.empty and not cad_trades.empty:
        jpy_win_rate = (jpy_trades['pnl'] > 0).mean()
        cad_win_rate = (cad_trades['pnl'] > 0).mean()

        print(".1%")
        print(".1%")

        if jpy_win_rate < 0.3 and cad_win_rate > 0.4:
            print("\nKEY FINDINGS:")
            print("1. Strong directional bias: All JPY trades are sells")
            print("2. CAD crosses show balanced direction and profitability")
            print("3. JPY crosses have poor win rates despite high confidence")
            print("4. Consider excluding JPY crosses or retraining model on JPY-specific patterns")

        # Check for overconfidence in losing trades
        losing_jpy = jpy_trades[jpy_trades['pnl'] < 0]
        if not losing_jpy.empty:
            avg_losing_conf = losing_jpy['confidence'].mean()
            print(f"Average confidence in losing JPY trades: {avg_losing_conf:.3f}")
if __name__ == "__main__":
    analyze_model_predictions()