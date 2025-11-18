import sqlite3
import os

def check_learning_activity():
    # Check if there are any learning-related tables with recent activity
    db_path = 'data/performance_history.db'
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check for any tables that might contain learning data
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND (name LIKE '%learn%' OR name LIKE '%model%' OR name LIKE '%param%')")
        learning_tables = cursor.fetchall()

        print('Learning/Model/Parameter tables:')
        for table in learning_tables:
            table_name = table[0]
            print(f'  - {table_name}')

            # Check if table has recent entries
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE DATE(timestamp) >= '2025-11-17'")
                count = cursor.fetchone()[0]
                if count > 0:
                    print(f'    ✅ {count} entries today in {table_name}')
            except Exception as e:
                print(f'    ❌ Error checking {table_name}: {e}')

        conn.close()
    else:
        print('Database not found')

if __name__ == '__main__':
    check_learning_activity()