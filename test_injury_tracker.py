#!/usr/bin/env python3
"""
🏀 Test NBA Injury Tracking System
"""

import sys
import logging
from datetime import date, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from nba_predictor.core.injury_tracker import NBAInjuryTracker

def main():
    """Main function to test injury tracker."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("🏀 NBA INJURY TRACKING SYSTEM TEST")
    print("=" * 80)

    # Initialize tracker
    tracker = NBAInjuryTracker()

    # Test with recent dates
    end_date = date.today()
    start_date = end_date - timedelta(days=3)  # Last 3 days for testing

    print(f"Testing injury tracking for {start_date} to {end_date}")
    print("-" * 40)

    # Run injury tracking
    result = tracker.download_injuries_for_period(start_date, end_date)

    # Display results
    if result['success']:
        print(f"\n✅ INJURY TRACKING COMPLETED SUCCESSFULLY")
        print(f"   Period: {result['period']}")
        print(f"   Total Injuries: {result['total_injuries']}")
        print(f"   Unique Players: {result['unique_players']}")
        print(f"   Sources Used: {', '.join(result['sources_used'])}")
        print(f"   Session Time: {result['session_time_seconds']:.2f} seconds")
        print(f"   Errors: {result['errors']}")

        # Get current injuries from database
        print(f"\n📋 CURRENT INJURIES FROM DATABASE:")
        current_injuries = tracker.get_current_injuries()

        if current_injuries:
            print(f"   Found {len(current_injuries)} recent injuries")
            print(f"\n   Sample injuries:")
            for i, injury in enumerate(current_injuries[:5], 1):
                print(f"   {i}. {injury['player_name']} ({injury['team_abbreviation']})")
                print(f"      Status: {injury['injury_status']}")
                print(f"      Availability: {injury.get('availability_probability', 0):.0%}")
                print(f"      Source: {injury['source']}")
                print(f"      Date: {injury['injury_date']}")
                print()
        else:
            print("   No recent injuries found in database")

        # Check data files
        print(f"📁 DATA FILES:")
        injury_dir = Path("data/injuries")
        if injury_dir.exists():
            files = list(injury_dir.glob("*.parquet"))
            print(f"   Created {len(files)} Parquet files")
            if files:
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                size_kb = latest_file.stat().st_size / 1024
                print(f"   Latest file: {latest_file.name} ({size_kb:.1f} KB)")

        # Database verification
        print(f"💾 DATABASE VERIFICATION:")
        try:
            import sqlite3
            conn = sqlite3.connect("data/nba_data.db")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM player_injuries")
            count = cursor.fetchone()[0]
            print(f"   Total injury records in database: {count}")
            conn.close()
        except Exception as e:
            print(f"   Database check failed: {e}")

    else:
        print(f"\n❌ INJURY TRACKING FAILED")
        print(f"   Errors encountered: {result['errors']}")
        print(f"   Session time: {result['session_time_seconds']:.2f} seconds")

    print(f"\n🎯 INJURY TRACKING SYSTEM TEST COMPLETED!")


if __name__ == "__main__":
    main()