#!/usr/bin/env python3
"""
Database Verification Script
Checks for order duplicates and trading system health after test runs.
"""

import sqlite3
import os
from datetime import datetime, date
import sys

def check_database_duplicates():
    """Check database for duplicate orders"""
    try:
        # Try common database locations
        db_paths = [
            'D:/FX-Ai-Data/trades.db',
            './trades.db',
            '../trades.db'
        ]

        db_found = False
        for db_path in db_paths:
            if os.path.exists(db_path):
                print(f"Found database at: {db_path}")
                db_found = True
                break

        if not db_found:
            print("❌ Database not found. Checked locations:")
            for path in db_paths:
                print(f"  - {path}")
            return

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check today's orders
        today = date.today()
        print(f"\n📊 ORDERS PLACED TODAY ({today}):")
        print("-" * 60)

        cursor.execute("""
            SELECT
                symbol,
                COUNT(*) as total_orders,
                SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as successful_orders,
                SUM(CASE WHEN status = 'PENDING' THEN 1 ELSE 0 END) as pending_orders,
                SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed_orders,
                MAX(timestamp) as last_order_time
            FROM orders
            WHERE DATE(timestamp) = DATE(?)
            GROUP BY symbol
            ORDER BY total_orders DESC
        """, (today,))

        results = cursor.fetchall()

        if not results:
            print("✅ No orders found in database today")
            print("   This is expected if running in test mode or no trades executed")
        else:
            total_orders = 0
            duplicate_symbols = 0
            successful_orders = 0

            for row in results:
                symbol, count, success, pending, failed, last_time = row
                total_orders += count
                successful_orders += success

                status = "✅" if count <= 1 else "❌ DUPLICATE"
                if count > 1:
                    duplicate_symbols += 1

                print(f"{symbol:8s}: {count:2d} orders ({success} success, {pending} pending, {failed} failed) {status}")
                if count > 1:
                    print(f"         ⚠️  Last order: {last_time}")

            print("-" * 60)
            print(f"Total Orders: {total_orders}")
            print(f"Successful Orders: {successful_orders}")
            print(f"Symbols with Orders: {len(results)}")
            print(f"Symbols with Duplicates: {duplicate_symbols}")

            # Calculate success rate
            if total_orders > 0:
                success_rate = (successful_orders / total_orders) * 100
                print(f"Success Rate: {success_rate:.1f}%")
                if success_rate < 90:
                    print("⚠️  WARNING: Low success rate - check MT5 connection!")
                else:
                    print("✅ Success rate looks good!")

            if duplicate_symbols > 0:
                print(f"\n❌ ISSUE DETECTED: {duplicate_symbols} symbols have duplicate orders!")
                print("   This indicates the loop timing fixes may not be working properly.")
            else:
                print("\n✅ No duplicate orders detected - loop timing fixes working!")

        # Check for recent orders (last 24 hours)
        print(f"\n📈 RECENT ORDERS (Last 24 hours):")
        cursor.execute("""
            SELECT symbol, status, timestamp
            FROM orders
            WHERE timestamp > datetime('now', '-1 day')
            ORDER BY timestamp DESC
            LIMIT 10
        """)

        recent_orders = cursor.fetchall()
        if recent_orders:
            for symbol, status, timestamp in recent_orders:
                print(f"  {timestamp}: {symbol} - {status}")
        else:
            print("  No orders in the last 24 hours")

        conn.close()

    except Exception as e:
        print(f"❌ Database check failed: {e}")
        import traceback
        traceback.print_exc()

def check_log_files():
    """Check log files for key indicators"""
    print(f"\n📋 LOG FILE ANALYSIS:")
    print("-" * 40)

    log_files = ['sydney_test.log', 'logs/trading.log', 'logs/main.log']

    success_indicators = [
        '✅ ORDER SUCCESS',
        'Order placed successfully',
        'Request executed'
    ]

    failure_indicators = [
        'Order failed: Request executed',
        'Multiple pending orders per symbol',
        '❌ ORDER FAILED'
    ]

    for log_file in log_files:
        if os.path.exists(log_file):
            print(f"\nAnalyzing {log_file}:")
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    # Check for success indicators
                    success_count = 0
                    for indicator in success_indicators:
                        count = content.count(indicator)
                        if count > 0:
                            print(f"  ✅ {indicator}: {count} occurrences")
                            success_count += count

                    # Check for failure indicators
                    failure_count = 0
                    for indicator in failure_indicators:
                        count = content.count(indicator)
                        if count > 0:
                            print(f"  ❌ {indicator}: {count} occurrences")
                            failure_count += count

                    if success_count > 0 and failure_count == 0:
                        print("  🎉 Log analysis: SUCCESS - No failure indicators found!")
                    elif failure_count > 0:
                        print("  ⚠️  Log analysis: ISSUES DETECTED - Check failure messages above")

            except Exception as e:
                print(f"  Error reading {log_file}: {e}")
        else:
            print(f"  Log file not found: {log_file}")

def main():
    print("🔍 FX-Ai Database and Log Verification")
    print("=" * 50)
    print(f"Timestamp: {datetime.now()}")
    print()

    check_database_duplicates()
    check_log_files()

    print(f"\n" + "=" * 50)
    print("🎯 VERIFICATION COMPLETE")
    print("=" * 50)
    print("Summary:")
    print("  ✅ No duplicates = Loop timing fixes working")
    print("  ✅ High success rate = MT5 retcode fixes working")
    print("  ✅ Orders in database = Tracking working")
    print("  ✅ No 'Request executed' failures = Success code fixes working")
    print("=" * 50)

if __name__ == "__main__":
    main()