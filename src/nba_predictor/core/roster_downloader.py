#!/usr/bin/env python3
"""
🏀 NBA Enhanced Roster Data Downloader
Context7-compliant roster data system using CommonTeamRoster API.
"""

import polars as pl
import pandas as pd
import time
from datetime import datetime, date
from pathlib import Path
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from nba_api.stats.endpoints import commonteamroster, commonteamyears
from nba_api.stats.static import teams

from .roster_injury_schemas import (
    PlayerInfo, RosterInfo, TeamRoster, RosterStatus, ContractStatus,
    Position, ROSTER_COLUMN_MAPPING
)
from .roster_injury_store_extensions import RosterInjuryStoreExtensions
from .data_store import UnifiedDataStore

logger = logging.getLogger(__name__)


@dataclass
class RosterDownloadConfig:
    """Configuration for roster download operations."""
    rate_limit: float = 0.5  # 500ms between requests
    max_retries: int = 3
    retry_delay: float = 2.0
    timeout: int = 60
    validate_data: bool = True


class EnhancedRosterDownloader:
    """Enhanced roster data downloader using CommonTeamRoster API."""

    def __init__(self, data_store: UnifiedDataStore, config: Optional[RosterDownloadConfig] = None):
        """Initialize roster downloader with data store and configuration."""
        self.data_store = data_store
        self.config = config or RosterDownloadConfig()
        self.roster_extensions = RosterInjuryStoreExtensions(data_store)
        self.session_start = datetime.now()
        self.download_stats = {
            'teams_processed': 0,
            'players_processed': 0,
            'errors': 0,
            'retries': 0
        }

    def get_all_nba_teams(self, season: str) -> List[Dict[str, Any]]:
        """Get all NBA teams for a specific season."""
        try:
            logger.info(f"📋 Retrieving NBA teams for season {season}")

            # Get all teams from static data
            nba_teams = teams.get_teams()

            # Filter for active NBA teams
            active_teams = [team for team in nba_teams if team.get('is_nba_franchise', True)]

            logger.info(f"✅ Found {len(active_teams)} NBA teams")
            return active_teams

        except Exception as e:
            logger.error(f"Error retrieving NBA teams: {e}")
            return []

    def download_team_roster(self, team_id: int, season: str) -> Optional[TeamRoster]:
        """Download complete roster for a specific team and season."""
        try:
            # Rate limiting
            time.sleep(self.config.rate_limit)

            logger.info(f"📥 Downloading roster for team {team_id}, season {season}")

            # Get team info
            team_info = None
            all_teams = self.get_all_nba_teams(season)
            for team in all_teams:
                if team['id'] == team_id:
                    team_info = team
                    break

            if not team_info:
                logger.error(f"Team {team_id} not found in team list")
                return None

            # Make API request
            roster_endpoint = commonteamroster.CommonTeamRoster(
                team_id=team_id,
                season=season,
                timeout=self.config.timeout
            )

            # Get roster data
            roster_data = roster_endpoint.get_data_frames()

            if not roster_data or len(roster_data) == 0:
                logger.warning(f"No roster data returned for team {team_id}")
                return None

            # Process player data
            players_df = roster_data[0] if len(roster_data) > 0 else pd.DataFrame()

            if players_df.empty:
                logger.warning(f"Empty roster data for team {team_id}")
                return None

            # Convert to roster objects
            players = []
            active_count = 0
            injured_count = 0

            for _, row in players_df.iterrows():
                try:
                    # Map position
                    position = self._map_position(row.get('POSITION', ''))

                    # Determine roster status
                    roster_status = self._determine_roster_status(row)
                    if roster_status == RosterStatus.ACTIVE:
                        active_count += 1
                    else:
                        injured_count += 1

                    # Create roster info
                    roster_info = RosterInfo(
                        player_id=int(row['PLAYER_ID']),
                        team_id=int(row['TeamID']),
                        season=season,
                        jersey_number=str(row.get('NUM', '')),
                        position=position,
                        roster_status=roster_status,
                        contract_status=self._estimate_contract_status(row),
                        salary=None,  # Not available from this API
                        acquisition_date=None,  # Not available from this API
                        trade_deadline_eligible=None,  # Not available from this API
                        waive_deadline_eligible=None,  # Not available from this API
                        last_updated=datetime.now(),
                        source="NBA_CommonTeamRoster_API"
                    )

                    players.append(roster_info)

                except Exception as e:
                    logger.error(f"Error processing player {row.get('PLAYER', 'Unknown')}: {e}")
                    self.download_stats['errors'] += 1

            # Create team roster object
            team_roster = TeamRoster(
                team_id=team_id,
                team_name=team_info['full_name'],
                team_abbreviation=team_info['abbreviation'],
                season=season,
                season_type="Regular Season",  # Default, can be updated later
                players=players,
                total_players=len(players),
                active_players=active_count,
                injured_players=injured_count,
                total_salary=None,  # Not available from this API
                salary_cap_space=None,  # Not available from this API
                last_updated=datetime.now(),
                source="NBA_CommonTeamRoster_API"
            )

            self.download_stats['teams_processed'] += 1
            self.download_stats['players_processed'] += len(players)

            logger.info(f"✅ Downloaded roster for {team_info['full_name']}: {len(players)} players")
            return team_roster

        except Exception as e:
            logger.error(f"Error downloading roster for team {team_id}: {e}")
            self.download_stats['errors'] += 1
            return None

    def download_all_rosters(self, season: str) -> List[TeamRoster]:
        """Download rosters for all NBA teams for a given season."""
        logger.info(f"🏀 Starting enhanced roster download for season {season}")

        # Get all NBA teams
        nba_teams = self.get_all_nba_teams(season)

        if not nba_teams:
            logger.error("No NBA teams found")
            return []

        all_rosters = []

        for team in nba_teams:
            team_id = team['id']
            team_name = team['full_name']

            logger.info(f"Processing {team_name} (ID: {team_id})")

            # Download team roster with retry logic
            roster = self._download_with_retry(team_id, season)

            if roster:
                all_rosters.append(roster)

                # Store in data store
                success = self.roster_extensions.store_team_roster(roster)
                if success:
                    logger.info(f"✅ Stored roster for {team_name}")
                else:
                    logger.error(f"❌ Failed to store roster for {team_name}")
            else:
                logger.warning(f"⚠️  No roster data for {team_name}")

        # Generate summary
        self._log_download_summary(season, len(all_rosters))

        return all_rosters

    def _download_with_retry(self, team_id: int, season: str) -> Optional[TeamRoster]:
        """Download roster with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                roster = self.download_team_roster(team_id, season)
                if roster:
                    return roster

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for team {team_id}: {e}")

                if attempt < self.config.max_retries - 1:
                    retry_delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    self.download_stats['retries'] += 1
                else:
                    logger.error(f"All retries failed for team {team_id}")
                    self.download_stats['errors'] += 1

        return None

    def _map_position(self, position_str: str) -> Optional[Position]:
        """Map position string to Position enum."""
        if not position_str or pd.isna(position_str):
            return None

        position_str = str(position_str).upper().strip()

        position_mapping = {
            'PG': Position.PG,
            'SG': Position.SG,
            'SF': Position.SF,
            'PF': Position.PF,
            'C': Position.C,
            'G': Position.G,
            'F': Position.F,
            'FC': Position.FC,
            'GF': Position.G,  # Guard-Forward maps to Guard
            'FORWARD': Position.F,
            'GUARD': Position.G,
            'CENTER': Position.C
        }

        return position_mapping.get(position_str)

    def _determine_roster_status(self, row: pd.Series) -> RosterStatus:
        """Determine roster status from row data."""
        # CommonTeamRoster doesn't provide explicit status, so we estimate based on available data
        exp = str(row.get('EXP', ''))

        # If player has experience, assume active
        if exp and exp != '' and exp != 'R' and exp != '0':
            return RosterStatus.ACTIVE

        # For rookies, determine based on other factors
        # This is a simplified estimation - in reality, you'd need injury data
        return RosterStatus.ACTIVE  # Default assumption

    def _estimate_contract_status(self, row: pd.Series) -> ContractStatus:
        """Estimate contract status from available data."""
        # CommonTeamRoster doesn't provide contract details
        # This is a simplified estimation
        exp = str(row.get('EXP', ''))

        if exp == 'R' or exp == '0':
            return ContractStatus.ROOKIE_SCALE

        # For veterans, assume guaranteed (simplified)
        return ContractStatus.GUARANTEED

    def _log_download_summary(self, season: str, successful_teams: int):
        """Log comprehensive download summary."""
        elapsed_time = datetime.now() - self.session_start

        logger.info("=" * 80)
        logger.info("🎯 ROSTER DOWNLOAD SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Season: {season}")
        logger.info(f"Teams Successfully Processed: {successful_teams}")
        logger.info(f"Total Players Processed: {self.download_stats['players_processed']}")
        logger.info(f"Errors Encountered: {self.download_stats['errors']}")
        logger.info(f"Retry Attempts: {self.download_stats['retries']}")
        logger.info(f"Total Time: {elapsed_time.total_seconds():.2f} seconds")

        if successful_teams > 0:
            avg_players_per_team = self.download_stats['players_processed'] / successful_teams
            logger.info(f"Average Players per Team: {avg_players_per_team:.1f}")

        logger.info("=" * 80)

    def get_download_statistics(self) -> Dict[str, Any]:
        """Get comprehensive download statistics."""
        return {
            'session_start': self.session_start,
            'download_stats': self.download_stats.copy(),
            'config': {
                'rate_limit': self.config.rate_limit,
                'max_retries': self.config.max_retries,
                'timeout': self.config.timeout
            }
        }


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