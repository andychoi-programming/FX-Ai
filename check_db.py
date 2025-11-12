import sqlite3
import os

db_path = 'data/performance_history.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print('Tables in performance_history.db:')
    for table in tables:
        print(f'  {table[0]}')
        
        # Get column info for each table
        cursor.execute(f'PRAGMA table_info({table[0]})')
        columns = cursor.fetchall()
        print(f'    Columns: {[col[1] for col in columns]}')
        
        # Get row count
        cursor.execute(f'SELECT COUNT(*) FROM {table[0]}')
        count = cursor.fetchone()[0]
        print(f'    Rows: {count}')
        
        # Check for open trades
        col_names = [col[1] for col in columns]
        if 'status' in col_names:
            cursor.execute(f'SELECT COUNT(*) FROM {table[0]} WHERE status = "open" OR status = "OPEN"')
            open_count = cursor.fetchone()[0]
            print(f'    Open trades: {open_count}')
        elif 'position_status' in col_names:
            cursor.execute(f'SELECT COUNT(*) FROM {table[0]} WHERE position_status = "open" OR position_status = "OPEN"')
            open_count = cursor.fetchone()[0]
            print(f'    Open trades: {open_count}')
        print()
    
    conn.close()
else:
    print('performance_history.db does not exist')