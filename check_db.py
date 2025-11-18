#!/usr/bin/env python3
import os
import sqlite3

db_path = 'data/performance_history.db'
if os.path.exists(db_path):
    print(f'Database exists: {db_path}')
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT name FROM sqlite_master WHERE type="table" AND name="stop_orders"')
        result = cursor.fetchone()
        if result:
            print('stop_orders table exists')
            cursor.execute('SELECT COUNT(*) FROM stop_orders')
            count = cursor.fetchone()[0]
            print(f'Records in stop_orders: {count}')

            # Show recent records
            cursor.execute('SELECT ticket, symbol, order_type, timestamp FROM stop_orders ORDER BY timestamp DESC LIMIT 5')
            recent = cursor.fetchall()
            print('Recent stop orders:')
            for record in recent:
                print(f'  {record}')
        else:
            print('stop_orders table does not exist')
        conn.close()
    except Exception as e:
        print(f'Database error: {e}')
else:
    print(f'Database does not exist: {db_path}')