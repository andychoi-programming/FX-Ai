"""
Initialize Adaptive Learning Database
Creates all necessary tables for trade tracking and learning
"""
import sqlite3
import os

def initialize_database():
    """Initialize the adaptive learning database with all required tables"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    db_path = 'data/performance_history.db'
    
    print(f"Initializing database at: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Trades table
    print("Creating trades table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            symbol TEXT,
            direction TEXT,
            entry_price REAL,
            exit_price REAL,
            volume REAL,
            profit REAL,
            profit_pct REAL,
            signal_strength REAL,
            ml_score REAL,
            technical_score REAL,
            sentiment_score REAL,
            duration_minutes INTEGER,
            model_version TEXT
        )
    ''')
    
    # Symbol optimal holding times table
    print("Creating symbol_optimal_holding table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS symbol_optimal_holding (
            symbol TEXT PRIMARY KEY,
            optimal_holding_hours REAL,
            max_holding_minutes INTEGER,
            confidence_score REAL,
            sample_size INTEGER,
            last_updated DATETIME
        )
    ''')
    
    # Insert default holding times for all symbols
    print("Setting default holding times...")
    symbols = [
        'AUDCAD', 'AUDCHF', 'AUDNZD', 'AUDUSD', 'CADCHF', 'CADJPY',
        'CHFJPY', 'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY',
        'EURNZD', 'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY',
        'GBPNZD', 'GBPUSD', 'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD',
        'USDCAD', 'USDCHF', 'USDJPY', 'XAUUSD', 'XAGUSD'
    ]
    
    for symbol in symbols:
        # Default: 2 hour optimal, 4 hour max
        cursor.execute('''
            INSERT OR REPLACE INTO symbol_optimal_holding 
            (symbol, optimal_holding_hours, max_holding_minutes, confidence_score, sample_size, last_updated)
            VALUES (?, 2.0, 240, 0.0, 0, datetime('now'))
        ''', (symbol,))
    
    # Model performance table
    print("Creating model_performance table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            symbol TEXT,
            model_version TEXT,
            accuracy REAL,
            precision_val REAL,
            recall_val REAL,
            f1_score REAL,
            win_rate REAL,
            avg_profit REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            total_trades INTEGER
        )
    ''')
    
    # Parameter optimization table
    print("Creating parameter_optimization table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parameter_optimization (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            parameter_name TEXT,
            old_value REAL,
            new_value REAL,
            performance_improvement REAL
        )
    ''')
    
    conn.commit()
    conn.close()
    
    print("\n" + "="*60)
    print("✓ Database initialized successfully!")
    print(f"  Location: {db_path}")
    print(f"  Tables created: trades, symbol_optimal_holding, model_performance, parameter_optimization")
    print(f"  Default holding times: 2h optimal, 4h max for all symbols")
    print("="*60)

if __name__ == "__main__":
    initialize_database()
