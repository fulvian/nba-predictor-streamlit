#!/usr/bin/env python3
"""
🏀 NBA Statistics Store Extensions - Context7 Compliant Data Storage

Estensione del UnifiedDataStore per dati statistici NBA basata su:
- Context7 best practice per data modeling e validation
- NBA API LeagueGameLog integration
- Polars optimization per large datasets
- Predictive analytics ready schema

Features:
- Game results storage con advanced metrics
- Player statistics con validation
- Bulk operations e batch processing
- Metadata tracking e versioning
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import polars as pl

from .data_store import UnifiedDataStore
from .statistics_data_schemas import (
    validate_game_results_data,
    map_league_game_log_columns,
    calculate_derived_metrics,
    GameResultModel
)
from ..utils.exceptions import DatabaseError, ValidationError

logger = logging.getLogger(__name__)

class NBAStatisticsStoreExtensions:
    """
    Extension class providing NBA statistics storage methods.
    Context7-compliant implementation for advanced analytics.
    """

    def __init__(self, data_store: UnifiedDataStore):
        """
        Initialize statistics extensions with existing data store.

        Args:
            data_store: Existing UnifiedDataStore instance
        """
        self.data_store = data_store

    def store_game_results(self, game_results_df: pl.DataFrame, season: str, season_type: str = "Regular Season") -> str:
        """
        Store NBA game results with Context7 validation and enhancement.

        Args:
            game_results_df: Polars DataFrame containing game results
            season: NBA season identifier (e.g., '2024-25')
            season_type: Type of season ('Regular Season', 'Playoffs', etc.)

        Returns:
            Path to stored Parquet file

        Raises:
            ValidationError: If data validation fails
            DatabaseError: If storage operation fails
        """
        if game_results_df is None or game_results_df.height == 0:
            raise ValidationError("Game results DataFrame is empty or None")

        try:
            logger.info(f"Storing {len(game_results_df)} game results for season {season}")

            # Validate data quality
            is_valid, errors = validate_game_results_data(game_results_df)
            if not is_valid:
                raise ValidationError(f"Data validation failed: {errors}")

            # Apply Context7 column mapping if needed
            if 'SEASON_ID' in game_results_df.columns:
                # Raw NBA API data - needs mapping
                processed_df = map_league_game_log_columns(game_results_df)
                logger.debug("Applied NBA API column mapping")
            else:
                # Already processed data
                processed_df = game_results_df.clone()

            # Add/override season metadata
            processed_df = processed_df.with_columns([
                pl.lit(season).alias("season"),
                pl.lit(season_type).alias("season_type"),
                pl.lit("NBA_API_LeagueGameLog").alias("source"),
                pl.lit(datetime.now()).alias("created_at"),
                pl.lit(datetime.now()).alias("updated_at")
            ])

            # Calculate Context7 derived metrics
            enhanced_df = calculate_derived_metrics(processed_df)
            logger.debug("Calculated Context7 derived metrics")

            # Create game results directory
            game_results_dir = self.data_store.base_path / "game_results"
            game_results_dir.mkdir(exist_ok=True)

            # Create file path with Context7 naming convention
            file_path = game_results_dir / f"game_results_{season}_{season_type.replace(' ', '_')}.parquet"

            # Store with Context7 optimization
            enhanced_df.write_parquet(
                file_path,
                compression="snappy",
                statistics=True,
                row_group_size=1000  # Context7 best practice
            )

            # Update metadata
            self.data_store._update_metadata(
                table_name=f"game_results_{season}_{season_type.replace(' ', '_')}",
                record_count=len(enhanced_df),
                file_path=str(file_path)
            )

            logger.info(
                f"Successfully stored {len(enhanced_df)} game results to {file_path}",
                extra={
                    "file_path": str(file_path),
                    "record_count": len(enhanced_df),
                    "season": season,
                    "season_type": season_type
                }
            )

            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to store game results: {e}")
            raise DatabaseError(f"Failed to store game results: {e}") from e

    def store_player_game_stats(self, player_stats_df: pl.DataFrame, game_date: date) -> str:
        """
        Store NBA player game statistics with Context7 validation.

        Args:
            player_stats_df: Polars DataFrame containing player game stats
            game_date: Date of the games

        Returns:
            Path to stored Parquet file

        Raises:
            ValidationError: If data validation fails
            DatabaseError: If storage operation fails
        """
        if player_stats_df is None or player_stats_df.height == 0:
            logger.warning(f"No player stats data for date {game_date}")
            return ""

        try:
            logger.info(f"Storing {len(player_stats_df)} player game stats for {game_date}")

            # Validate required columns for player stats
            required_columns = ['player_id', 'player_name', 'team_id', 'game_id', 'points']
            missing_columns = set(required_columns) - set(player_stats_df.columns)

            if missing_columns:
                raise ValidationError(f"Missing required player stats columns: {missing_columns}")

            # Add Context7 metadata
            date_str = game_date.strftime('%Y-%m-%d')
            enhanced_df = player_stats_df.with_columns([
                pl.lit(date_str).alias("game_date"),
                pl.lit("NBA_API_PlayerStats").alias("source"),
                pl.lit(datetime.now()).alias("created_at")
            ])

            # Create player stats directory
            player_stats_dir = self.data_store.base_path / "player_stats"
            player_stats_dir.mkdir(exist_ok=True)

            # Create file path
            file_path = player_stats_dir / f"player_stats_{date_str}.parquet"

            # Store with optimization
            enhanced_df.write_parquet(
                file_path,
                compression="snappy",
                statistics=True,
                row_group_size=500  # Context7 best practice for player data
            )

            # Update metadata
            self.data_store._update_metadata(
                table_name=f"player_stats_{date_str}",
                record_count=len(enhanced_df),
                file_path=str(file_path)
            )

            logger.info(
                f"Successfully stored {len(enhanced_df)} player stats to {file_path}",
                extra={
                    "file_path": str(file_path),
                    "record_count": len(enhanced_df),
                    "game_date": date_str
                }
            )

            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to store player game stats: {e}")
            raise DatabaseError(f"Failed to store player game stats: {e}") from e

    def get_game_results(self, season: str, season_type: str = "Regular Season", team_id: Optional[int] = None) -> pl.DataFrame:
        """
        Retrieve game results with optional filtering.

        Args:
            season: NBA season identifier
            season_type: Type of season
            team_id: Optional team ID filter

        Returns:
            Polars DataFrame with game results
        """
        try:
            file_name = f"game_results_{season}_{season_type.replace(' ', '_')}.parquet"
            file_path = self.data_store.base_path / "game_results" / file_name

            if not file_path.exists():
                logger.warning(f"Game results file not found: {file_path}")
                return pl.DataFrame()

            df = pl.read_parquet(file_path)

            # Apply team filter if specified
            if team_id is not None:
                df = df.filter(pl.col("team_id") == team_id)

            logger.info(f"Retrieved {len(df)} game results for season {season}")
            return df

        except Exception as e:
            logger.error(f"Failed to retrieve game results: {e}")
            return pl.DataFrame()

    def get_player_stats_by_game(self, game_date: date, team_id: Optional[int] = None) -> pl.DataFrame:
        """
        Retrieve player statistics for a specific game date.

        Args:
            game_date: Date of the game
            team_id: Optional team ID filter

        Returns:
            Polars DataFrame with player statistics
        """
        try:
            date_str = game_date.strftime('%Y-%m-%d')
            file_name = f"player_stats_{date_str}.parquet"
            file_path = self.data_store.base_path / "player_stats" / file_name

            if not file_path.exists():
                logger.warning(f"Player stats file not found: {file_path}")
                return pl.DataFrame()

            df = pl.read_parquet(file_path)

            # Apply team filter if specified
            if team_id is not None:
                df = df.filter(pl.col("team_id") == team_id)

            logger.info(f"Retrieved {len(df)} player stats for {date_str}")
            return df

        except Exception as e:
            logger.error(f"Failed to retrieve player stats: {e}")
            return pl.DataFrame()

    def get_team_season_summary(self, season: str, team_id: int) -> Dict[str, Any]:
        """
        Get comprehensive season summary for a specific team.

        Args:
            season: NBA season identifier
            team_id: Team ID to summarize

        Returns:
            Dictionary with team season statistics
        """
        try:
            # Get game results for the team
            games_df = self.get_game_results(season, "Regular Season", team_id)

            if games_df.height == 0:
                return {"error": f"No game data found for team {team_id} in season {season}"}

            # Calculate Context7 team metrics
            summary = {
                "team_id": team_id,
                "season": season,
                "games_played": len(games_df),
                "wins": len(games_df.filter(pl.col("result") == "W")),
                "losses": len(games_df.filter(pl.col("result") == "L")),
                "win_percentage": (games_df.filter(pl.col("result") == "W").height / len(games_df)) * 100,

                # Scoring averages (Context7 standard)
                "avg_points": games_df["points"].mean(),
                "avg_field_goal_percentage": games_df["field_goal_percentage"].mean(),
                "avg_three_point_percentage": games_df["three_point_percentage"].mean(),
                "avg_free_throw_percentage": games_df["free_throw_percentage"].mean(),

                # Advanced averages (Context7 analytics)
                "avg_offensive_rating": games_df["offensive_rating"].mean(),
                "avg_true_shooting_percentage": games_df["true_shooting_pct"].mean(),
                "avg_effective_fg_percentage": games_df["effective_fg_pct"].mean(),

                # Team performance averages
                "avg_assists": games_df["assists"].mean(),
                "avg_rebounds": games_df["total_rebounds"].mean(),
                "avg_steals": games_df["steals"].mean(),
                "avg_blocks": games_df["blocks"].mean(),
                "avg_turnovers": games_df["turnovers"].mean(),
                "avg_plus_minus": games_df["plus_minus"].mean(),

                # Best and worst performances
                "highest_score": games_df["points"].max(),
                "lowest_score": games_df["points"].min(),
                "best_plus_minus": games_df["plus_minus"].max(),
                "worst_plus_minus": games_df["plus_minus"].min(),

                "data_source": "NBA_API_LeagueGameLog",
                "generated_at": datetime.now().isoformat()
            }

            logger.info(f"Generated season summary for team {team_id}, season {season}")
            return summary

        except Exception as e:
            logger.error(f"Failed to generate team season summary: {e}")
            return {"error": f"Failed to generate summary: {e}"}

    def bulk_store_season_data(self, seasons: List[str], game_data_by_season: Dict[str, pl.DataFrame]) -> Dict[str, str]:
        """
        Bulk store game results for multiple seasons.

        Args:
            seasons: List of season identifiers
            game_data_by_season: Dictionary mapping seasons to DataFrames

        Returns:
            Dictionary mapping seasons to file paths
        """
        results = {}

        for season in seasons:
            if season in game_data_by_season:
                try:
                    file_path = self.store_game_results(game_data_by_season[season], season)
                    results[season] = file_path
                    logger.info(f"Successfully stored {len(game_data_by_season[season])} games for season {season}")
                except Exception as e:
                    logger.error(f"Failed to store data for season {season}: {e}")
                    results[season] = f"ERROR: {e}"
            else:
                logger.warning(f"No data provided for season {season}")
                results[season] = "NO_DATA"

        return results

def enhance_data_store_with_statistics(data_store: UnifiedDataStore) -> UnifiedDataStore:
    """
    Enhance an existing UnifiedDataStore with NBA statistics methods.

    Args:
        data_store: Existing UnifiedDataStore instance

    Returns:
        Enhanced data store with statistics methods
    """
    extensions = NBAStatisticsStoreExtensions(data_store)

    # Add new statistics methods to the data store class
    data_store.store_game_results = extensions.store_game_results
    data_store.store_player_game_stats = extensions.store_player_game_stats
    data_store.get_game_results = extensions.get_game_results
    data_store.get_player_stats_by_game = extensions.get_player_stats_by_game
    data_store.get_team_season_summary = extensions.get_team_season_summary
    data_store.bulk_store_season_data = extensions.bulk_store_season_data

    logger.info("Enhanced data store with NBA statistics methods")
    return data_store

# Test function
def test_statistics_store_extensions():
    """Test the NBA statistics store extensions."""
    from .data_store import UnifiedDataStore

    print("🏀 Testing NBA Statistics Store Extensions")

    # Create enhanced data store
    base_store = UnifiedDataStore("data/test_statistics")
    base_store.initialize()

    enhanced_store = enhance_data_store_with_statistics(base_store)

    # Verify methods are available
    methods = [
        'store_game_results',
        'store_player_game_stats',
        'get_game_results',
        'get_team_season_summary'
    ]

    print("✅ Statistics methods available:")
    for method in methods:
        has_method = hasattr(enhanced_store, method)
        print(f"  - {method}: {'✅' if has_method else '❌'}")

    print("🎉 Statistics store extensions test completed!")

if __name__ == "__main__":
    test_statistics_store_extensions()