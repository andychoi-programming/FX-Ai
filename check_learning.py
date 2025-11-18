import sqlite3
import os
from datetime import datetime

def check_trading_activity():
    # Check performance history database
    db_path = 'data/performance_history.db'
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check what tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print('Available tables in performance_history.db:')
        for table in tables:
            print(f'  - {table[0]}')

        # Get recent trades
        if ('trades',) in tables:
            cursor.execute("SELECT COUNT(*) FROM trades WHERE DATE(timestamp) >= '2025-11-17'")
            recent_trades = cursor.fetchone()[0]

            # Get total trades
            cursor.execute('SELECT COUNT(*) FROM trades')
            total_trades = cursor.fetchone()[0]

            print(f'\nPerformance Database Status:')
            print(f'Total trades recorded: {total_trades}')
            print(f'Trades on 2025-11-17: {recent_trades}')

            if recent_trades > 0:
                cursor.execute("SELECT symbol, pnl, timestamp FROM trades WHERE DATE(timestamp) >= '2025-11-17' ORDER BY timestamp DESC LIMIT 5")
                recent_trades_data = cursor.fetchall()
                print('\nRecent trades:')
                for trade in recent_trades_data:
                    print(f'  {trade[0]}: ${trade[1]:.2f} at {trade[2]}')
        else:
            print('No trades table found')

        conn.close()
    else:
        print('Performance database not found')    # Check learning database
    learning_db_path = 'data/learning_database.db'
    if os.path.exists(learning_db_path):
        conn = sqlite3.connect(learning_db_path)
        cursor = conn.cursor()

        # Get recent learning updates
        cursor.execute("SELECT COUNT(*) FROM model_updates WHERE DATE(timestamp) >= '2025-11-17'")
        recent_updates = cursor.fetchone()[0]

        print(f'\nLearning Database Status:')
        print(f'Model updates on 2025-11-17: {recent_updates}')

        if recent_updates > 0:
            cursor.execute("SELECT symbol, update_type, timestamp FROM model_updates WHERE DATE(timestamp) >= '2025-11-17' ORDER BY timestamp DESC LIMIT 5")
            recent_updates_data = cursor.fetchall()
            print('\nRecent model updates:')
            for update in recent_updates_data:
                print(f'  {update[0]}: {update[1]} at {update[2]}')

        conn.close()
    else:
        print('Learning database not found')

if __name__ == '__main__':
    check_trading_activity()