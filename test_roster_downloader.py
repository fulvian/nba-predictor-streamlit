#!/usr/bin/env python3
"""
🏀 Test Enhanced Roster Downloader
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from nba_predictor.core.roster_downloader import EnhancedRosterDownloader
from nba_predictor.core.data_store import UnifiedDataStore

def main():
    """Main function to test roster downloader."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("🏀 NBA ENHANCED ROSTER DOWNLOADER TEST")
    print("=" * 80)

    # Initialize data store
    data_store = UnifiedDataStore()

    # Create downloader
    downloader = EnhancedRosterDownloader(data_store)

    # Test with 2024-25 season
    season = "2024-25"

    print(f"Testing roster download for season {season}")
    print("-" * 40)

    # Download all rosters
    rosters = downloader.download_all_rosters(season)

    if rosters:
        print(f"\n✅ Successfully downloaded {len(rosters)} team rosters")

        # Show sample
        if rosters:
            sample_roster = rosters[0]
            print(f"\n📝 Sample Roster:")
            print(f"   Team: {sample_roster.team_name} ({sample_roster.team_abbreviation})")
            print(f"   Players: {sample_roster.total_players}")
            print(f"   Active: {sample_roster.active_players}")
            print(f"   Injured: {sample_roster.injured_players}")

            if sample_roster.players:
                sample_player = sample_roster.players[0]
                print(f"\n   Sample Player:")
                print(f"   Name: Player ID {sample_player.player_id}")
                print(f"   Jersey: #{sample_player.jersey_number}")
                print(f"   Position: {sample_player.position}")
                print(f"   Status: {sample_player.roster_status}")
    else:
        print("❌ No rosters downloaded")

    # Show statistics
    stats = downloader.get_download_statistics()
    print(f"\n📊 Download Statistics:")
    print(f"   Teams: {stats['download_stats']['teams_processed']}")
    print(f"   Players: {stats['download_stats']['players_processed']}")
    print(f"   Errors: {stats['download_stats']['errors']}")
    print(f"   Retries: {stats['download_stats']['retries']}")

if __name__ == "__main__":
    main()