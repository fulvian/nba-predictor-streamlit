#!/usr/bin/env python3
"""
🏀 NBA Data Store Extensions - Metodi Mancanti e Column Mapping

Estensione del UnifiedDataStore con metodi mancanti per:
- Teams data storage
- Roster data storage
- Column mapping corretto per NBA API schema
- Data validation e transformation

Basato su Context7 best practice per NBA API schema e Polars optimization.
"""

import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Any, Optional
import polars as pl

from .data_store import UnifiedDataStore
from ..utils.exceptions import DatabaseError, ValidationError

logger = logging.getLogger(__name__)


class NBADataStoreExtensions:
    """
    Extension class providing additional methods for NBA data storage.
    """

    def __init__(self, data_store: UnifiedDataStore):
        """
        Initialize extensions with existing data store.

        Args:
            data_store: Existing UnifiedDataStore instance
        """
        self.data_store = data_store

    def store_teams_data(self, teams_df: pl.DataFrame, date_str: str) -> str:
        """
        Store NBA teams data with proper schema validation and mapping.

        Args:
            teams_df: Polars DataFrame containing teams data
            date_str: Date string for file naming

        Returns:
            Path to stored Parquet file

        Raises:
            ValidationError: If required columns are missing
            DatabaseError: If storage operation fails
        """
        if teams_df is None or teams_df.height == 0:
            raise ValidationError("Teams DataFrame is empty or None")

        try:
            # Validate and map columns based on NBA API schema
            mapped_df = self._map_teams_columns(teams_df)

            # Validate required columns after mapping
            required_columns = {'team_id', 'team_name', 'abbreviation', 'season'}
            missing_columns = required_columns - set(mapped_df.columns)

            if missing_columns:
                raise ValidationError(f"Missing required teams columns: {missing_columns}")

            # Create teams directory if not exists
            teams_dir = self.data_store.base_path / "teams"
            teams_dir.mkdir(exist_ok=True)

            # Create file path
            file_path = teams_dir / f"teams_{date_str}.parquet"

            # Store data with compression
            mapped_df.write_parquet(file_path, compression="snappy")

            # Update metadata
            self.data_store._update_metadata(
                table_name=f"teams_{date_str}",
                record_count=len(mapped_df),
                file_path=str(file_path)
            )

            logger.info(
                f"Stored {len(mapped_df)} teams data to {file_path}",
                extra={"file_path": str(file_path), "record_count": len(mapped_df)}
            )

            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to store teams data: {e}")
            raise DatabaseError(f"Failed to store teams data: {e}") from e

    def store_roster_data(self, roster_df: pl.DataFrame, team_id: int, date_str: str) -> str:
        """
        Store NBA team roster data with proper schema validation.

        Args:
            roster_df: Polars DataFrame containing roster data
            team_id: Team identifier
            date_str: Date string for file naming

        Returns:
            Path to stored Parquet file

        Raises:
            ValidationError: If required columns are missing
            DatabaseError: If storage operation fails
        """
        if roster_df is None or roster_df.height == 0:
            logger.warning(f"Empty roster data for team {team_id}")
            return ""

        try:
            # Validate and map columns based on NBA API roster schema
            mapped_df = self._map_roster_columns(roster_df, team_id)

            # Validate required columns after mapping
            required_columns = {'player_id', 'player_name', 'team_id', 'season', 'position'}
            missing_columns = required_columns - set(mapped_df.columns)

            if missing_columns:
                raise ValidationError(f"Missing required roster columns: {missing_columns}")

            # Create rosters directory if not exists
            rosters_dir = self.data_store.base_path / "rosters"
            rosters_dir.mkdir(exist_ok=True)

            # Create file path
            file_path = rosters_dir / f"roster_team_{team_id}_{date_str}.parquet"

            # Store data with compression
            mapped_df.write_parquet(file_path, compression="snappy")

            # Update metadata
            self.data_store._update_metadata(
                table_name=f"roster_team_{team_id}_{date_str}",
                record_count=len(mapped_df),
                file_path=str(file_path)
            )

            logger.info(
                f"Stored {len(mapped_df)} roster entries for team {team_id} to {file_path}",
                extra={
                    "team_id": team_id,
                    "file_path": str(file_path),
                    "record_count": len(mapped_df)
                }
            )

            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to store roster data for team {team_id}: {e}")
            raise DatabaseError(f"Failed to store roster data: {e}") from e

    def _map_teams_columns(self, teams_df: pl.DataFrame) -> pl.DataFrame:
        """
        Map NBA API teams data to standardized schema.

        Args:
            teams_df: Raw teams DataFrame from NBA API

        Returns:
            Mapped DataFrame with standardized columns
        """
        try:
            # Column mapping based on NBA API static teams schema
            column_mapping = {
                'id': 'team_id',
                'full_name': 'team_name',
                'abbreviation': 'abbreviation',
                'nickname': 'nickname',
                'city': 'city',
                'state': 'state',
                'year_founded': 'year_founded',
                'owner': 'owner',
                'general_manager': 'general_manager',
                'head_coach': 'head_coach',
                'd_league_id': 'd_league_id'
            }

            # Apply column mapping
            mapped_df = teams_df.rename(column_mapping)

            # Add metadata columns
            current_date = date.today().strftime('%Y-%m-%d')
            mapped_df = mapped_df.with_columns([
                pl.lit(2024).alias('season'),
                pl.lit(current_date).alias('last_updated'),
                pl.lit("NBA_API_STATIC").alias('source'),
                pl.lit("Active").alias('status')
            ])

            # Select and order final columns
            final_columns = [
                'team_id', 'team_name', 'abbreviation', 'nickname', 'city', 'state',
                'year_founded', 'owner', 'general_manager', 'head_coach',
                'd_league_id', 'season', 'status', 'last_updated', 'source'
            ]

            # Only include columns that exist
            available_columns = [col for col in final_columns if col in mapped_df.columns]
            mapped_df = mapped_df.select(available_columns)

            return mapped_df

        except Exception as e:
            logger.error(f"Error mapping teams columns: {e}")
            raise ValidationError(f"Failed to map teams columns: {e}") from e

    def _map_roster_columns(self, roster_df: pl.DataFrame, team_id: int) -> pl.DataFrame:
        """
        Map NBA API roster data to standardized schema.

        Args:
            roster_df: Raw roster DataFrame from NBA API
            team_id: Team identifier to ensure consistency

        Returns:
            Mapped DataFrame with standardized columns
        """
        try:
            # Column mapping based on NBA API CommonTeamRoster schema
            column_mapping = {
                'PLAYER_ID': 'player_id',
                'PLAYER': 'player_name',
                'PLAYER_SLUG': 'player_slug',
                'NUM': 'jersey_number',
                'POSITION': 'position',
                'HEIGHT': 'height',
                'WEIGHT': 'weight',
                'BIRTH_DATE': 'birth_date',
                'AGE': 'age',
                'EXP': 'experience_years',
                'SCHOOL': 'school',
                'PLAYER_ID_LEGACY': 'player_id_legacy',
                'HOW_ACQUIRED': 'how_acquired',
                'START_DATE': 'start_date',
                'ROSTER_STATUS': 'roster_status'
            }

            # Apply column mapping only for columns that exist
            existing_columns = {}
            for old_name, new_name in column_mapping.items():
                if old_name in roster_df.columns:
                    existing_columns[old_name] = new_name

            mapped_df = roster_df.rename(existing_columns)

            # Ensure team_id is correct
            if 'team_id' not in mapped_df.columns:
                mapped_df = mapped_df.with_columns([
                    pl.lit(team_id).alias('team_id')
                ])

            # Add metadata columns
            current_date = date.today().strftime('%Y-%m-%d')
            mapped_df = mapped_df.with_columns([
                pl.lit(2024).alias('season'),
                pl.lit(current_date).alias('last_updated'),
                pl.lit("NBA_API_ROSTER").alias('source')
            ])

            # Data validation and cleaning
            if 'position' in mapped_df.columns:
                # Standardize position values
                mapped_df = mapped_df.with_columns([
                    pl.col('position').str.to_uppercase().alias('position')
                ])

            # Select and order final columns
            final_columns = [
                'player_id', 'player_name', 'player_slug', 'jersey_number', 'position',
                'height', 'weight', 'birth_date', 'age', 'experience_years',
                'school', 'team_id', 'how_acquired', 'start_date', 'roster_status',
                'season', 'last_updated', 'source'
            ]

            # Only include columns that exist
            available_columns = [col for col in final_columns if col in mapped_df.columns]
            mapped_df = mapped_df.select(available_columns)

            return mapped_df

        except Exception as e:
            logger.error(f"Error mapping roster columns for team {team_id}: {e}")
            raise ValidationError(f"Failed to map roster columns: {e}") from e

    def _map_players_columns(self, players_df: pl.DataFrame) -> pl.DataFrame:
        """
        Map NBA API players data to standardized schema for store_players_data.

        Args:
            players_df: Raw players DataFrame from NBA API

        Returns:
            Mapped DataFrame with standardized columns
        """
        try:
            # Column mapping based on NBA API static players schema
            column_mapping = {
                'id': 'player_id',
                'full_name': 'player_name',
                'first_name': 'first_name',
                'last_name': 'last_name',
                'is_active': 'is_active'
            }

            # Apply column mapping
            mapped_df = players_df.rename(column_mapping)

            # Add required columns that might be missing
            if 'position' not in mapped_df.columns:
                # For static players data, position might not be available
                mapped_df = mapped_df.with_columns([
                    pl.lit("N/A").alias('position')
                ])

            if 'team_id' not in mapped_df.columns:
                mapped_df = mapped_df.with_columns([
                    pl.lit(0).alias('team_id')  # Default placeholder
                ])

            # Add metadata columns
            current_date = date.today().strftime('%Y-%m-%d')
            mapped_df = mapped_df.with_columns([
                pl.lit(2024).alias('season'),
                pl.lit(current_date).alias('last_updated'),
                pl.lit("NBA_API_STATIC").alias('source')
            ])

            # Select and order final columns
            final_columns = [
                'player_id', 'player_name', 'first_name', 'last_name', 'position',
                'team_id', 'is_active', 'season', 'last_updated', 'source'
            ]

            # Only include columns that exist
            available_columns = [col for col in final_columns if col in mapped_df.columns]
            mapped_df = mapped_df.select(available_columns)

            return mapped_df

        except Exception as e:
            logger.error(f"Error mapping players columns: {e}")
            raise ValidationError(f"Failed to map players columns: {e}") from e

    def _map_games_columns(self, games_df: pl.DataFrame) -> pl.DataFrame:
        """
        Map games data to standardized schema for store_games_data.

        Args:
            games_df: Raw games DataFrame from API

        Returns:
            Mapped DataFrame with standardized columns
        """
        try:
            # Determine the source and apply appropriate mapping
            if 'home_team_id' in games_df.columns:
                # BallDontLie API format
                column_mapping = {
                    'id': 'game_id',
                    'home_team_id': 'home_team_id',
                    'visitor_team_id': 'visitor_team_id',
                    'home_team': 'home_team',
                    'visitor_team': 'visitor_team',
                    'date': 'game_date',
                    'status': 'status',
                    'period': 'period',
                    'time': 'game_time',
                    'postseason': 'postseason',
                    'home_team_score': 'home_score',
                    'visitor_team_score': 'visitor_score'
                }
            else:
                # NBA API format (BoxScoreTraditionalV2)
                column_mapping = {
                    'GAME_ID': 'game_id',
                    'TEAM_ID': 'team_id',  # Will need processing
                    'TEAM_ABBREVIATION': 'team_abbreviation',
                    'PLAYER_ID': 'player_id',
                    'PLAYER_NAME': 'player_name',
                    'START_POSITION': 'start_position',
                    'MIN': 'minutes',
                    'PTS': 'points',
                    'REB': 'rebounds',
                    'AST': 'assists'
                }

            # Apply column mapping
            mapped_df = games_df.rename(column_mapping)

            # Add metadata columns
            mapped_df = mapped_df.with_columns([
                pl.lit(2024).alias('season'),
                pl.lit("API").alias('source'),
                pl.lit(date.today().isoformat()).alias('last_updated')
            ])

            # For NBA API format, we need to process differently
            if 'player_id' in mapped_df.columns:
                # This is box score data, not schedule data
                logger.warning("Received box score data instead of schedule data")
                return None

            # Ensure required columns for games
            required_columns = {'game_id', 'game_date', 'away_team', 'season'}
            if not all(col in mapped_df.columns for col in required_columns):
                # Try to derive missing columns
                if 'visitor_team' in mapped_df.columns and 'away_team' not in mapped_df.columns:
                    mapped_df = mapped_df.rename({'visitor_team': 'away_team'})

            return mapped_df

        except Exception as e:
            logger.error(f"Error mapping games columns: {e}")
            raise ValidationError(f"Failed to map games columns: {e}") from e


def enhance_data_store(data_store: UnifiedDataStore) -> UnifiedDataStore:
    """
    Enhance an existing UnifiedDataStore with NBA-specific methods.

    Args:
        data_store: Existing UnifiedDataStore instance

    Returns:
        Enhanced data store with NBA methods
    """
    extensions = NBADataStoreExtensions(data_store)

    # Add new methods to the data store class
    data_store.store_teams_data = extensions.store_teams_data
    data_store.store_roster_data = extensions.store_roster_data
    data_store._map_teams_columns = extensions._map_teams_columns
    data_store._map_roster_columns = extensions._map_roster_columns
    data_store._map_players_columns = extensions._map_players_columns
    data_store._map_games_columns = extensions._map_games_columns

    # Override existing methods to use new column mapping
    original_store_players = data_store.store_players_data

    def enhanced_store_players_data(players_df: pl.DataFrame, season: str) -> str:
        """Enhanced store_players_data with column mapping."""
        try:
            # Map columns before validation
            mapped_df = extensions._map_players_columns(players_df)
            return original_store_players(mapped_df, season)
        except Exception as e:
            logger.error(f"Enhanced store_players_data failed: {e}")
            raise

    data_store.store_players_data = enhanced_store_players_data

    return data_store


# Standalone test function
def test_data_store_extensions():
    """Test the NBA data store extensions."""
    from .data_store import UnifiedDataStore

    print("🏀 Testing NBA Data Store Extensions")

    # Create enhanced data store
    base_store = UnifiedDataStore("data/test_extensions")
    base_store.initialize()

    enhanced_store = enhance_data_store(base_store)

    print("✅ Data store enhanced successfully")
    print("✅ Available methods:")

    methods = [
        'store_teams_data',
        'store_roster_data',
        'store_players_data',
        'store_games_data'
    ]

    for method in methods:
        if hasattr(enhanced_store, method):
            print(f"  - {method}")
        else:
            print(f"  - ❌ {method} (missing)")


if __name__ == "__main__":
    test_data_store_extensions()