#!/usr/bin/env python3
"""
🏀 NBA Enhanced Roster Data Downloader
Context7-compliant roster data system using CommonTeamRoster API.
Complete implementation for 2024-25 and 2025-26 seasons.
"""

import polars as pl
import pandas as pd
import time
from datetime import datetime, date
from pathlib import Path
import logging
from typing import List, Dict, Optional, Any, Tuple
import sqlite3

from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NBARosterDownloader:
    """Enhanced NBA roster data downloader using CommonTeamRoster API."""

    def __init__(self, data_dir: str = "data"):
        """Initialize roster downloader with data directory."""
        self.data_dir = Path(data_dir)
        self.roster_dir = self.data_dir / "rosters"
        self.roster_dir.mkdir(parents=True, exist_ok=True)

        # Database connection for summary data
        self.db_path = self.data_dir / "nba_data.db"
        self._init_database()

        # Configuration
        self.rate_limit = 0.6  # 600ms between requests
        self.max_retries = 3
        self.timeout = 30

        # Statistics
        self.session_start = datetime.now()
        self.stats = {
            'teams_processed': 0,
            'players_processed': 0,
            'errors': 0,
            'retries': 0
        }

    def _init_database(self):
        """Initialize database tables for roster data."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Create team rosters table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS team_rosters (
                team_id INTEGER,
                team_name TEXT,
                team_abbreviation TEXT,
                season TEXT,
                season_type TEXT,
                total_players INTEGER,
                active_players INTEGER,
                injured_players INTEGER,
                last_updated TEXT,
                source TEXT,
                file_path TEXT,
                PRIMARY KEY (team_id, season)
            )
        ''')

        # Create player roster details table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_roster_details (
                player_id INTEGER,
                team_id INTEGER,
                season TEXT,
                jersey_number TEXT,
                position TEXT,
                height TEXT,
                weight TEXT,
                birth_date TEXT,
                age INTEGER,
                experience TEXT,
                school TEXT,
                last_updated TEXT,
                source TEXT,
                PRIMARY KEY (player_id, team_id, season)
            )
        ''')

        conn.commit()
        conn.close()

    def get_all_nba_teams(self) -> List[Dict[str, Any]]:
        """Get all NBA teams."""
        try:
            logger.info("📋 Retrieving NBA teams...")
            nba_teams = teams.get_teams()
            active_teams = [team for team in nba_teams if team.get('is_nba_franchise', True)]
            logger.info(f"✅ Found {len(active_teams)} NBA teams")
            return active_teams
        except Exception as e:
            logger.error(f"Error retrieving NBA teams: {e}")
            return []

    def download_team_roster(self, team_id: int, season: str) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Download roster for a specific team and season."""
        try:
            # Rate limiting
            time.sleep(self.rate_limit)

            logger.info(f"📥 Downloading roster for team {team_id}, season {season}")

            # Get team info
            team_info = None
            all_teams = self.get_all_nba_teams()
            for team in all_teams:
                if team['id'] == team_id:
                    team_info = team
                    break

            if not team_info:
                logger.error(f"Team {team_id} not found")
                return None

            # Make API request
            roster_endpoint = commonteamroster.CommonTeamRoster(
                team_id=team_id,
                season=season,
                timeout=self.timeout
            )

            # Get roster data
            roster_data = roster_endpoint.get_data_frames()

            if not roster_data or len(roster_data) == 0:
                logger.warning(f"No roster data for team {team_id}")
                return None

            players_df = roster_data[0]

            if players_df.empty:
                logger.warning(f"Empty roster for team {team_id}")
                return None

            # Create summary data
            summary = {
                'team_id': team_id,
                'team_name': team_info['full_name'],
                'team_abbreviation': team_info['abbreviation'],
                'season': season,
                'season_type': 'Regular Season',
                'total_players': len(players_df),
                'active_players': len(players_df),  # Assume all are active for now
                'injured_players': 0,  # Would need injury API for this
                'last_updated': datetime.now().isoformat(),
                'source': 'NBA_CommonTeamRoster_API'
            }

            self.stats['teams_processed'] += 1
            self.stats['players_processed'] += len(players_df)

            logger.info(f"✅ Downloaded roster for {team_info['full_name']}: {len(players_df)} players")
            return players_df, summary

        except Exception as e:
            logger.error(f"Error downloading roster for team {team_id}: {e}")
            self.stats['errors'] += 1
            return None

    def save_roster_data(self, team_id: int, season: str, players_df: pd.DataFrame, summary: Dict[str, Any]) -> bool:
        """Save roster data to both Parquet and database."""
        try:
            # Save to Parquet
            filename = f"roster_team_{team_id}_{season.replace('-', '_')}.parquet"
            file_path = self.roster_dir / filename

            # Convert to Polars and save
            pl_df = pl.from_pandas(players_df)
            pl_df.write_parquet(file_path)

            # Save summary to database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO team_rosters
                (team_id, team_name, team_abbreviation, season, season_type,
                 total_players, active_players, injured_players, last_updated,
                 source, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', [
                summary['team_id'], summary['team_name'], summary['team_abbreviation'],
                summary['season'], summary['season_type'], summary['total_players'],
                summary['active_players'], summary['injured_players'],
                summary['last_updated'], summary['source'], str(file_path)
            ])

            # Save player details to database
            for _, player in players_df.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO player_roster_details
                    (player_id, team_id, season, jersey_number, position, height,
                     weight, birth_date, age, experience, school, last_updated, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', [
                    player['PLAYER_ID'], team_id, season,
                    player.get('NUM', ''), player.get('POSITION', ''),
                    player.get('HEIGHT', ''), player.get('WEIGHT', ''),
                    player.get('BIRTH_DATE', ''), player.get('AGE', 0),
                    player.get('EXP', ''), player.get('SCHOOL', ''),
                    datetime.now().isoformat(), 'NBA_CommonTeamRoster_API'
                ])

            conn.commit()
            conn.close()

            logger.info(f"✅ Saved roster data for {summary['team_name']}")
            return True

        except Exception as e:
            logger.error(f"Error saving roster data: {e}")
            return False

    def download_all_rosters(self, season: str) -> Dict[str, Any]:
        """Download rosters for all NBA teams for a given season."""
        logger.info(f"🏀 Starting roster download for season {season}")

        # Get all NBA teams
        nba_teams = self.get_all_nba_teams()

        if not nba_teams:
            logger.error("No NBA teams found")
            return {'success': False, 'message': 'No teams found'}

        successful_teams = []
        failed_teams = []

        for team in nba_teams:
            team_id = team['id']
            team_name = team['full_name']

            logger.info(f"Processing {team_name} (ID: {team_id})")

            # Download team roster
            result = self.download_team_roster(team_id, season)

            if result:
                players_df, summary = result

                # Save data
                success = self.save_roster_data(team_id, season, players_df, summary)

                if success:
                    successful_teams.append(summary)
                else:
                    failed_teams.append({'team_id': team_id, 'team_name': team_name, 'error': 'Save failed'})
            else:
                failed_teams.append({'team_id': team_id, 'team_name': team_name, 'error': 'Download failed'})

        # Generate summary
        session_time = datetime.now() - self.session_start

        final_summary = {
            'season': season,
            'successful_teams': len(successful_teams),
            'failed_teams': len(failed_teams),
            'total_players': self.stats['players_processed'],
            'errors': self.stats['errors'],
            'retries': self.stats['retries'],
            'session_time_seconds': session_time.total_seconds(),
            'successful_team_details': successful_teams,
            'failed_team_details': failed_teams
        }

        self._log_final_summary(final_summary)
        return final_summary

    def _log_final_summary(self, summary: Dict[str, Any]):
        """Log comprehensive final summary."""
        logger.info("=" * 80)
        logger.info("🎯 ROSTER DOWNLOAD FINAL SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Season: {summary['season']}")
        logger.info(f"✅ Successful Teams: {summary['successful_teams']}")
        logger.info(f"❌ Failed Teams: {summary['failed_teams']}")
        logger.info(f"👥 Total Players: {summary['total_players']}")
        logger.info(f"⚠️  Errors: {summary['errors']}")
        logger.info(f"🔄 Retries: {summary['retries']}")
        logger.info(f"⏱️  Total Time: {summary['session_time_seconds']:.2f} seconds")

        if summary['successful_teams'] > 0:
            avg_players = summary['total_players'] / summary['successful_teams']
            logger.info(f"📊 Average Players per Team: {avg_players:.1f}")

        logger.info("=" * 80)

    def get_roster_summary(self, season: str) -> List[Dict[str, Any]]:
        """Get roster summary from database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute('''
                SELECT team_id, team_name, team_abbreviation, season,
                       total_players, active_players, injured_players,
                       last_updated, source
                FROM team_rosters
                WHERE season = ?
                ORDER BY team_name
            ''', [season])

            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]

            conn.close()
            return results

        except Exception as e:
            logger.error(f"Error retrieving roster summary: {e}")
            return []

    def get_team_roster_details(self, team_id: int, season: str) -> List[Dict[str, Any]]:
        """Get detailed roster for a specific team."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute('''
                SELECT player_id, jersey_number, position, height, weight,
                       birth_date, age, experience, school, last_updated
                FROM player_roster_details
                WHERE team_id = ? AND season = ?
                ORDER BY jersey_number
            ''', [team_id, season])

            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]

            conn.close()
            return results

        except Exception as e:
            logger.error(f"Error retrieving team roster details: {e}")
            return []


def main():
    """Main function to download NBA rosters."""
    print("🏀 NBA ENHANCED ROSTER DOWNLOADER")
    print("=" * 80)

    downloader = NBARosterDownloader()

    # Download rosters for both seasons
    seasons = ["2024-25", "2025-26"]

    all_results = {}

    for season in seasons:
        print(f"\n📅 Processing season: {season}")
        print("-" * 40)

        result = downloader.download_all_rosters(season)
        all_results[season] = result

        if result['successful_teams'] > 0:
            print(f"✅ Successfully downloaded rosters for {result['successful_teams']} teams")
            print(f"   Total players: {result['total_players']}")

            # Show sample teams
            if result['successful_team_details']:
                print(f"\n📝 Sample teams:")
                for team in result['successful_team_details'][:3]:
                    print(f"   {team['team_name']}: {team['total_players']} players")
        else:
            print(f"❌ Failed to download rosters for season {season}")

    # Show final summary
    print(f"\n{'='*80}")
    print("🎉 OVERALL SUMMARY")
    print("="*80)

    total_teams = sum(result['successful_teams'] for result in all_results.values())
    total_players = sum(result['total_players'] for result in all_results.values())

    print(f"📊 Total Teams Processed: {total_teams}")
    print(f"👥 Total Players Processed: {total_players}")
    print(f"📁 Data saved to: {downloader.roster_dir}")
    print(f"💾 Database: {downloader.db_path}")

    # Verify data by querying one season
    if "2024-25" in all_results and all_results["2024-25"]['successful_teams'] > 0:
        print(f"\n🔍 Verification query for 2024-25:")
        summary = downloader.get_roster_summary("2024-25")
        print(f"   Teams in database: {len(summary)}")

        if summary:
            sample_team = summary[0]
            details = downloader.get_team_roster_details(sample_team['team_id'], "2024-25")
            print(f"   Sample team {sample_team['team_name']}: {len(details)} players in database")

    print(f"\n✅ NBA ROSTER DOWNLOAD COMPLETED!")
    print("🚀 Ready for advanced analytics!")


if __name__ == "__main__":
    main()