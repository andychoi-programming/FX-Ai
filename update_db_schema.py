import sqlite3

# Connect to the database
conn = sqlite3.connect('data/performance_history.db')
cursor = conn.cursor()

# Add missing columns
try:
    cursor.execute('ALTER TABLE trades ADD COLUMN ticket INTEGER')
    print("Added ticket column")
except sqlite3.OperationalError as e:
    print(f"ticket column: {e}")

try:
    cursor.execute('ALTER TABLE trades ADD COLUMN entry_time DATETIME')
    print("Added entry_time column")
except sqlite3.OperationalError as e:
    print(f"entry_time column: {e}")

try:
    cursor.execute('ALTER TABLE trades ADD COLUMN exit_time DATETIME')
    print("Added exit_time column")
except sqlite3.OperationalError as e:
    print(f"exit_time column: {e}")

try:
    cursor.execute('ALTER TABLE trades ADD COLUMN status TEXT DEFAULT "open"')
    print("Added status column")
except sqlite3.OperationalError as e:
    print(f"status column: {e}")

try:
    cursor.execute('ALTER TABLE trades ADD COLUMN commission REAL DEFAULT 0')
    print("Added commission column")
except sqlite3.OperationalError as e:
    print(f"commission column: {e}")

try:
    cursor.execute('ALTER TABLE trades ADD COLUMN swap REAL DEFAULT 0')
    print("Added swap column")
except sqlite3.OperationalError as e:
    print(f"swap column: {e}")

# Commit and close
conn.commit()
conn.close()

print("Database schema update completed")