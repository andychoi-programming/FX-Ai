"""
FX-Ai ML Model Status Checker
Verifies which ML models exist and which are missing
"""

import os
import sys

# Add project root to path
sys.path.insert(0, r'C:\Users\andyc\python\FX-Ai')

def check_ml_models():
    """Check which ML models are trained and which are missing"""
    
    print("="*80)
    print("ML MODEL STATUS CHECK")
    print("="*80)
    
    # Expected symbols
    symbols = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD',
        'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD',
        'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPNZD',
        'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD',
        'NZDCAD', 'NZDCHF', 'NZDJPY',
        'CADCHF', 'CADJPY', 'CHFJPY',
        'XAUUSD', 'XAGUSD'
    ]
    
    # Model directory
    models_dir = r'C:\Users\andyc\python\FX-Ai\models'
    
    if not os.path.exists(models_dir):
        print(f"✗ ERROR: Models directory not found: {models_dir}")
        return
    
    print(f"\n✓ Models directory: {models_dir}\n")
    
    # Check for H1 timeframe models
    timeframe = 'H1'
    
    found_models = []
    missing_models = []
    
    for symbol in symbols:
        model_file = os.path.join(models_dir, f'{symbol}_{timeframe}_model.pkl')
        scaler_file = os.path.join(models_dir, f'{symbol}_{timeframe}_scaler.pkl')
        
        has_model = os.path.exists(model_file)
        has_scaler = os.path.exists(scaler_file)
        
        status = "✓" if (has_model and has_scaler) else "✗"
        
        if has_model and has_scaler:
            # Get file sizes
            model_size = os.path.getsize(model_file) / 1024  # KB
            scaler_size = os.path.getsize(scaler_file) / 1024  # KB
            print(f"{status} {symbol:8s} - Model: {model_size:>6.1f} KB | Scaler: {scaler_size:>6.1f} KB")
            found_models.append(symbol)
        else:
            missing = []
            if not has_model:
                missing.append("model")
            if not has_scaler:
                missing.append("scaler")
            print(f"{status} {symbol:8s} - MISSING: {', '.join(missing)}")
            missing_models.append(symbol)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total symbols: {len(symbols)}")
    print(f"✓ Models found: {len(found_models)}")
    print(f"✗ Models missing: {len(missing_models)}")
    
    if missing_models:
        print(f"\n✗ Missing models for: {', '.join(missing_models)}")
        print(f"\nTo train missing models, run:")
        print(f"  python backtest/train_all_models.py")
        print(f"\nOr train specific symbols:")
        for symbol in missing_models:
            print(f"  python backtest/train_model.py --symbol {symbol} --timeframe H1")
    else:
        print(f"\n✓ All models trained successfully!")
    
    return found_models, missing_models

def check_model_performance():
    """Check if models have performance data"""
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE DATA CHECK")
    print("="*80)
    
    import sqlite3
    
    db_path = r'C:\Users\andyc\python\FX-Ai\data\performance_history.db'
    
    if not os.path.exists(db_path):
        print(f"✗ Database not found: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if model_performance table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='model_performance'
        """)
        
        if not cursor.fetchone():
            print("✗ model_performance table does not exist")
            conn.close()
            return
        
        # Get model performance data
        cursor.execute("""
            SELECT symbol, accuracy, precision, recall, f1_score, last_updated
            FROM model_performance
            ORDER BY symbol
        """)
        
        rows = cursor.fetchall()
        
        if not rows:
            print("✗ No model performance data found")
            print("  Models have not been evaluated yet")
        else:
            print(f"\n✓ Found performance data for {len(rows)} models:\n")
            print(f"{'Symbol':8s} | {'Accuracy':>8s} | {'Precision':>9s} | {'Recall':>6s} | {'F1':>6s} | Last Updated")
            print("-" * 80)
            
            for row in rows:
                symbol, accuracy, precision, recall, f1, updated = row
                print(f"{symbol:8s} | {accuracy:>7.2%} | {precision:>8.2%} | {recall:>5.2%} | {f1:>5.2%} | {updated}")
        
        conn.close()
        
    except Exception as e:
        print(f"✗ Error checking performance data: {e}")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("FX-Ai ML Model Status Checker")
    print("="*80)
    print("\nChecking which ML models are trained and ready...\n")
    
    found, missing = check_ml_models()
    check_model_performance()
    
    print("\n" + "="*80)
    print("CHECK COMPLETE")
    print("="*80)
