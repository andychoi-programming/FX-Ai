import sqlite3
import shutil
from datetime import datetime, timezone
import MetaTrader5 as mt5
import os

class DatabaseCleanup:
    """Clean up corrupted and orphaned positions in the database"""

    def __init__(self, db_path="data/performance_history.db"):
        self.db_path = db_path
        self.backup_path = None

    def create_backup(self):
        """Create database backup"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.backup_path = f"{self.db_path}.backup_{timestamp}"

        print(f"📦 Creating backup: {self.backup_path}")
        shutil.copy2(self.db_path, self.backup_path)
        print("✅ Backup created successfully")
        return self.backup_path

    def analyze_database(self):
        """Analyze the current database state"""
        if not os.path.exists(self.db_path):
            print(f"❌ Database not found: {self.db_path}")
            return None

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get total trades
        cursor.execute("SELECT COUNT(*) FROM trades")
        total_trades = cursor.fetchone()[0]

        # Get open positions (NULL exit_price)
        cursor.execute("SELECT COUNT(*) FROM trades WHERE exit_price IS NULL")
        open_positions = cursor.fetchone()[0]

        # Get corrupted positions (NULL entry_time)
        cursor.execute("SELECT COUNT(*) FROM trades WHERE exit_price IS NULL AND entry_time IS NULL")
        corrupted_positions = cursor.fetchone()[0]

        # Get positions with tickets
        cursor.execute("SELECT COUNT(*) FROM trades WHERE ticket IS NOT NULL")
        positions_with_tickets = cursor.fetchone()[0]

        # Get sample corrupted positions
        cursor.execute("""
            SELECT id, symbol, direction, volume, ticket
            FROM trades
            WHERE exit_price IS NULL AND entry_time IS NULL
            LIMIT 5
        """)
        samples = cursor.fetchall()

        conn.close()

        stats = {
            'total_trades': total_trades,
            'open_positions': open_positions,
            'corrupted_positions': corrupted_positions,
            'positions_with_tickets': positions_with_tickets,
            'samples': samples
        }

        return stats

    def verify_mt5_positions(self):
        """Check current MT5 positions"""
        print("🔌 Connecting to MT5...")
        if not mt5.initialize():
            print(f"❌ MT5 connection failed: {mt5.last_error()}")
            return set()

        positions = mt5.positions_get()
        mt5_tickets = set()

        if positions:
            mt5_tickets = {pos.ticket for pos in positions}
            print(f"✅ Found {len(mt5_tickets)} positions in MT5")
        else:
            print("✅ No open positions in MT5")

        mt5.shutdown()
        return mt5_tickets

    def clean_corrupted_positions(self, dry_run=True):
        """Clean corrupted positions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        mode = "DRY RUN" if dry_run else "CLEANUP"
        print(f"\n🧹 {mode} MODE")

        # Find corrupted positions
        cursor.execute("""
            SELECT COUNT(*) FROM trades
            WHERE exit_time IS NULL AND (entry_time IS NULL OR ticket IS NULL)
        """)
        corrupted_count = cursor.fetchone()[0]

        if corrupted_count == 0:
            print("✅ No corrupted positions found")
            conn.close()
            return 0

        print(f"Found {corrupted_count} corrupted positions")

        if not dry_run:
            # Close corrupted positions
            cursor.execute("""
                UPDATE trades
                SET exit_time = ?,
                    closure_reason = 'DATABASE_CLEANUP',
                    status = 'ERROR'
                WHERE exit_time IS NULL AND (entry_time IS NULL OR ticket IS NULL)
            """, (datetime.now(timezone.utc),))

            conn.commit()
            print(f"✅ Closed {corrupted_count} corrupted positions")

        conn.close()
        return corrupted_count

    def clean_orphaned_positions(self, mt5_tickets, dry_run=True):
        """Clean positions that exist in DB but not in MT5"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get DB positions with tickets
        cursor.execute("""
            SELECT ticket FROM trades
            WHERE exit_price IS NULL AND ticket IS NOT NULL
        """)
        db_tickets = {row[0] for row in cursor.fetchall()}

        # Find orphaned positions
        orphaned_tickets = db_tickets - mt5_tickets

        if not orphaned_tickets:
            print("✅ No orphaned positions found")
            conn.close()
            return 0

        print(f"Found {len(orphaned_tickets)} orphaned positions")

        if not dry_run:
            # Close orphaned positions
            for ticket in orphaned_tickets:
                cursor.execute("""
                    UPDATE trades
                    SET exit_price = entry_price,
                        profit = 0,
                        exit_time = ?,
                        closure_reason = 'ORPHANED_CLEANUP',
                        status = 'ERROR'
                    WHERE ticket = ? AND exit_price IS NULL
                """, (datetime.now(timezone.utc), ticket))

            conn.commit()
            print(f"✅ Closed {len(orphaned_tickets)} orphaned positions")

        conn.close()
        return len(orphaned_tickets)

    def show_final_stats(self):
        """Show final database statistics"""
        stats = self.analyze_database()
        if not stats:
            return

        print("\n" + "=" * 60)
        print("FINAL DATABASE STATISTICS")
        print("=" * 60)
        print(f"Total trades: {stats['total_trades']}")
        print(f"Open positions: {stats['open_positions']}")
        print(f"Positions with tickets: {stats['positions_with_tickets']}")

        if stats['open_positions'] == 0:
            print("✅ Database is clean - no open positions!")
        else:
            print(f"⚠️  {stats['open_positions']} positions still open")

    def run_cleanup(self):
        """Run the complete cleanup process"""
        print("=" * 60)
        print("FX-AI DATABASE CLEANUP TOOL")
        print("=" * 60)

        # Initial analysis
        print("\n📊 INITIAL ANALYSIS")
        initial_stats = self.analyze_database()
        if not initial_stats:
            return

        print(f"Total trades: {initial_stats['total_trades']}")
        print(f"Open positions: {initial_stats['open_positions']}")
        print(f"Corrupted positions: {initial_stats['corrupted_positions']}")

        if initial_stats['samples']:
            print("Sample corrupted positions:")
            for sample in initial_stats['samples']:
                print(f"  ID: {sample[0]}, Symbol: {sample[1]}, Dir: {sample[2]}, Vol: {sample[3]}, Ticket: {sample[4]}")

        # Create backup
        self.create_backup()

        # Get MT5 positions
        mt5_tickets = self.verify_mt5_positions()

        # Dry run
        print("\n" + "=" * 60)
        corrupted_cleaned = self.clean_corrupted_positions(dry_run=True)
        orphaned_cleaned = self.clean_orphaned_positions(mt5_tickets, dry_run=True)

        # Ask for confirmation
        total_to_clean = corrupted_cleaned + orphaned_cleaned
        if total_to_clean == 0:
            print("\n✅ Database is already clean!")
            return

        print(f"\n⚠️  Will clean {total_to_clean} positions ({corrupted_cleaned} corrupted, {orphaned_cleaned} orphaned)")

        response = input("Proceed with cleanup? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Cleanup cancelled")
            return

        # Real cleanup
        print("\n🧹 Starting cleanup...")
        corrupted_cleaned = self.clean_corrupted_positions(dry_run=False)
        orphaned_cleaned = self.clean_orphaned_positions(mt5_tickets, dry_run=False)

        # Final stats
        self.show_final_stats()

        print("\n✅ Cleanup complete!")
        print(f"📦 Backup saved: {self.backup_path}")

def main():
    cleanup = DatabaseCleanup()
    cleanup.run_cleanup()

if __name__ == "__main__":
    main()