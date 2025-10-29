#!/usr/bin/env python3
"""
Debug script to check betting_analysis table schema.
"""

import duckdb
from pathlib import Path

def debug_table_schema():
    """Check betting_analysis table schema."""

    db_path = Path(__file__).parent / "data" / "nba_data.duckdb"

    try:
        conn = duckdb.connect(str(db_path))

        print("🔍 Checking betting_analysis table schema...")

        # Get table schema
        schema = conn.execute("DESCRIBE betting_analysis").fetchall()
        print("betting_analysis table schema:")
        for col in schema:
            print(f"  {col[0]}: {col[1]}")

        # Check bankroll_history schema
        schema2 = conn.execute("DESCRIBE bankroll_history").fetchall()
        print("\nbankroll_history table schema:")
        for col in schema2:
            print(f"  {col[0]}: {col[1]}")

        # Check placed_bets schema
        schema3 = conn.execute("DESCRIBE placed_bets").fetchall()
        print("\nplaced_bets table schema:")
        for col in schema3:
            print(f"  {col[0]}: {col[1]}")

        # Check existing data in bankroll_history
        existing_data = conn.execute("SELECT COUNT(*) FROM bankroll_history").fetchone()[0]
        print(f"\nExisting records in bankroll_history: {existing_data}")

        if existing_data > 0:
            max_id = conn.execute("SELECT MAX(history_id) FROM bankroll_history").fetchone()[0]
            print(f"Maximum history_id: {max_id}")

        conn.close()

    except Exception as e:
        print(f"❌ Error checking schema: {e}")

if __name__ == "__main__":
    debug_table_schema()