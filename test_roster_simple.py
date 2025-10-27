#!/usr/bin/env python3
"""
🏀 Simple Roster Downloader Test
"""

import polars as pl
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict, Optional, Any

from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams

logger = logging.getLogger(__name__)

def test_nba_roster_api():
    """Test basic NBA roster API functionality."""
    print("🏀 NBA ROSTER API TEST")
    print("=" * 60)

    # Get all NBA teams
    print("1. Getting NBA teams...")
    try:
        nba_teams = teams.get_teams()
        active_teams = [team for team in nba_teams if team.get('is_nba_franchise', True)]
        print(f"✅ Found {len(active_teams)} NBA teams")

        # Show sample teams
        sample_teams = active_teams[:3]
        for team in sample_teams:
            print(f"   {team['full_name']} ({team['abbreviation']}) - ID: {team['id']}")

    except Exception as e:
        print(f"❌ Error getting teams: {e}")
        return

    # Test with a specific team
    print(f"\n2. Testing roster download for sample team...")

    # Use first team as sample
    sample_team = active_teams[0]
    team_id = sample_team['id']
    team_name = sample_team['full_name']
    season = "2024-25"

    print(f"Team: {team_name} (ID: {team_id})")
    print(f"Season: {season}")

    try:
        # Make API request
        print("Making API request...")
        roster_endpoint = commonteamroster.CommonTeamRoster(
            team_id=team_id,
            season=season,
            timeout=30
        )

        # Get roster data
        roster_data = roster_endpoint.get_data_frames()

        if not roster_data or len(roster_data) == 0:
            print("❌ No data returned")
            return

        # Process player data
        players_df = roster_data[0] if len(roster_data) > 0 else pd.DataFrame()

        if players_df.empty:
            print("❌ Empty roster data")
            return

        print(f"✅ Successfully retrieved roster data")
        print(f"   Players found: {len(players_df)}")

        # Show sample players
        print(f"\n3. Sample roster data:")
        print(f"{'#':<3} {'Name':<25} {'Pos':<5} {'Jersey':<8} {'Height':<8} {'Weight':<8} {'Exp':<5}")
        print("-" * 70)

        for idx, (_, row) in enumerate(players_df.head(10).iterrows()):
            name = str(row.get('PLAYER', 'Unknown'))[:24]
            position = str(row.get('POSITION', ''))[:4]
            jersey = str(row.get('NUM', ''))[:7]
            height = str(row.get('HEIGHT', ''))[:7]
            weight = str(row.get('WEIGHT', ''))[:7]
            exp = str(row.get('EXP', ''))[:4]

            print(f"{idx+1:<3} {name:<25} {position:<5} {jersey:<8} {height:<8} {weight:<8} {exp:<5}")

        # Check data quality
        print(f"\n4. Data quality analysis:")

        # Position distribution
        pos_counts = players_df['POSITION'].value_counts()
        print(f"   Position distribution:")
        for pos, count in pos_counts.head().items():
            print(f"     {pos}: {count} players")

        # Experience distribution
        exp_counts = players_df['EXP'].value_counts()
        print(f"   Experience distribution:")
        for exp, count in exp_counts.head().items():
            print(f"     {exp}: {count} players")

        # Save sample data
        print(f"\n5. Saving sample data...")
        output_dir = Path("data/roster_test")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert to Polars and save
        pl_df = pl.from_pandas(players_df)
        output_file = output_dir / f"roster_{team_name.replace(' ', '_')}_2024-25.parquet"
        pl_df.write_parquet(output_file)

        print(f"✅ Data saved to: {output_file}")
        print(f"   File size: {output_file.stat().st_size} bytes")

        print(f"\n🎉 ROSTER API TEST COMPLETED SUCCESSFULLY!")
        print(f"✅ API working correctly")
        print(f"✅ Data format validated")
        print(f"✅ File storage working")

    except Exception as e:
        print(f"❌ Error downloading roster: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    test_nba_roster_api()

if __name__ == "__main__":
    main()