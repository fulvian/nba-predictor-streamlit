#!/usr/bin/env python3
"""
Debug script to check existing data in the betting_analysis table.
"""

import duckdb
from pathlib import Path

def debug_existing_data():
    """Check existing data in betting_analysis table."""

    db_path = Path(__file__).parent / "data" / "nba_data.duckdb"

    try:
        conn = duckdb.connect(str(db_path))

        print("🔍 Checking existing betting_analysis data...")

        # Check if table exists
        tables = conn.execute("SHOW TABLES").fetchall()
        print(f"Tables in database: {[t[0] for t in tables]}")

        if 'betting_analysis' in [t[0] for t in tables]:
            # Count records
            count = conn.execute("SELECT COUNT(*) FROM betting_analysis").fetchone()[0]
            print(f"Records in betting_analysis: {count}")

            if count > 0:
                # Show sample records
                records = conn.execute("SELECT analysis_id, roi, risk_level FROM betting_analysis LIMIT 5").fetchall()
                print("Sample records:")
                for record in records:
                    print(f"  ID: {record[0]}, ROI: {record[1]} (type: {type(record[1])}), Risk Level: {record[2]}")

                # Check for any ROI values that are strings
                bad_records = conn.execute("""
                    SELECT analysis_id, roi, risk_level
                    FROM betting_analysis
                    WHERE TYPEOF(roi) != 'DOUBLE' OR roi IS NULL
                """).fetchall()

                if bad_records:
                    print("❌ Found problematic ROI records:")
                    for record in bad_records:
                        print(f"  ID: {record[0]}, ROI: {record[1]} (type: {conn.execute('SELECT TYPEOF(?)', [record[1]]).fetchone()[0]}), Risk Level: {record[2]}")
                else:
                    print("✅ All ROI values appear to be numeric")
        else:
            print("❌ betting_analysis table does not exist")

        conn.close()

    except Exception as e:
        print(f"❌ Error checking database: {e}")

if __name__ == "__main__":
    debug_existing_data()