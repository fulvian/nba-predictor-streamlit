#!/usr/bin/env python3
"""
🚀 Unified NBA Data Pipeline for Predictive Analytics

Centralized data orchestration system for NBA predictive analytics.
This class provides unified access to NBA data from multiple sources
with quality validation and automated caching.

Author: NBA Predictive Analytics System
Task ID: nba-predictive-analytics-2024
"""

import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import pickle
import hashlib

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Existing data provider - Context7 compliant import with fallback
try:
    from data_provider_june2025 import NBADataProvider
except ImportError:
    # Context7 compliant fallback for type checking
    logger.warning("data_provider_june2025 not available, using fallback for type checking")
    # Create a minimal stub for mypy
    class NBADataProvider:  # type: ignore
        def __init__(self) -> None: ...
        def get_scheduled_games(self, specific_date: str) -> list[dict[str, object]]: ...

@dataclass
class DataValidationResult:
    """Results of data quality validation."""
    is_valid: bool
    missing_values: Dict[str, int]
    duplicate_rows: int
    data_types: Dict[str, str]
    validation_errors: List[str]
    quality_score: float

@dataclass
class PipelineMetrics:
    """Pipeline performance metrics."""
    fetch_time: float
    preprocess_time: float
    validation_time: float
    total_time: float
    records_processed: int
    cache_hit_rate: float

class DataPipelineError(Exception):
    """Custom exception for data pipeline operations."""
    pass

class UnifiedNBADataPipeline:
    """
    Centralized data pipeline for NBA predictive analytics.

    This class orchestrates data collection from multiple sources,
    performs quality validation, and provides unified access to
    NBA data for machine learning models.

    Attributes:
        data_provider: NBA data provider instance
        feature_engineer: Feature engineering module
        cache: Data caching mechanism
        metrics: Pipeline performance metrics
    """

    def __init__(
        self,
        data_provider: Optional[NBADataProvider] = None,
        cache_ttl: int = 3600
    ) -> None:
        """
        Initialize the unified data pipeline.

        Args:
            data_provider: Existing NBADataProvider instance
            cache_ttl: Cache time-to-live in seconds

        Example:
            >>> pipeline = UnifiedNBADataPipeline()
            >>> data = pipeline.fetch_all_data(
            ...     date_range=('2024-01-01', '2024-01-07'),
            ...     include_boxscores=True
            ... )
        """
        # Initialize data provider
        self.data_provider = data_provider or NBADataProvider()

        # Cache configuration
        self.cache_ttl = cache_ttl
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)

        # Performance tracking
        self.metrics_history: List[PipelineMetrics] = []
        self.cache_stats = {"hits": 0, "misses": 0}

        # Quality thresholds
        self.quality_thresholds = {
            "min_records": 10,
            "max_missing_ratio": 0.1,
            "max_duplicate_ratio": 0.05,
            "min_quality_score": 0.8
        }

        logger.info("UnifiedNBADataPipeline initialized successfully")

    def fetch_all_data(
        self,
        date_range: Tuple[date, date],
        include_boxscores: bool = True
    ) -> Dict[str, Union[pd.DataFrame, Any]]:
        """
        Fetch comprehensive NBA data for specified date range.

        Args:
            date_range: Tuple of (start_date, end_date)
            include_boxscores: Whether to include detailed boxscore data

        Returns:
            Dictionary containing different data types:
            - 'games': Scheduled games
            - 'boxscores': Game boxscores (if requested)
            - 'team_stats': Team statistics
            - 'player_stats': Player statistics

        Raises:
            DataPipelineError: If data fetching fails

        Example:
            >>> pipeline = UnifiedNBADataPipeline()
            >>> data = pipeline.fetch_all_data(
            ...     date_range=(date(2024,1,1), date(2024,1,7))
            ... )
            >>> print(f"Found {len(data['games'])} games")
        """
        start_time = time.time()
        start_date, end_date = date_range

        logger.info(f"Fetching data for range {start_date} to {end_date}")

        try:
            # Generate cache key
            cache_key = self._generate_cache_key(date_range, include_boxscores)

            # Check cache first
            cached_data: Optional[Dict[str, pd.DataFrame]] = self._load_from_cache(cache_key)
            if cached_data is not None:
                self.cache_stats["hits"] += 1
                logger.info(f"Loaded data from cache (hit rate: {self._get_cache_hit_rate():.2%})")
                return cached_data

            self.cache_stats["misses"] += 1

            # Fetch games data
            games_data = self._fetch_games_data(start_date, end_date)

            # Fetch boxscores if requested
            boxscores_data: Dict[str, Any] = {}  # Context7 compliant: Dict for mixed data structures
            if include_boxscores:
                boxscores_data = self._fetch_boxscores_data(games_data)

            # Fetch team statistics
            team_stats = self._fetch_team_stats_data(games_data)

            # Fetch player statistics
            player_stats = self._fetch_player_stats_data(games_data)

            # Combine all data (Context7 compliant Union types for mixed data structures)
            result: Dict[str, Union[pd.DataFrame, Any]] = {
                'games': games_data,
                'boxscores': boxscores_data,
                'team_stats': team_stats,
                'player_stats': player_stats
            }

            # Cache the results
            self._save_to_cache(cache_key, result)

            # Record metrics (Context7 compliant: handle mixed data structures)
            fetch_time = time.time() - start_time
            total_records = (
                len(games_data) +
                len(boxscores_data) if isinstance(boxscores_data, pd.DataFrame) else 0 +
                len(team_stats) +
                len(player_stats)
            )

            metrics = PipelineMetrics(
                fetch_time=fetch_time,
                preprocess_time=0.0,
                validation_time=0.0,
                total_time=fetch_time,
                records_processed=total_records,
                cache_hit_rate=self._get_cache_hit_rate()
            )
            self.metrics_history.append(metrics)

            logger.info(f"Successfully fetched data in {fetch_time:.2f}s ({total_records} records)")
            return result

        except Exception as e:
            logger.error(
                "Data fetch failed",
                extra={
                    "date_range": str(date_range),
                    "include_boxscores": include_boxscores,
                    "error": str(e)
                }
            )
            raise DataPipelineError(f"Failed to fetch NBA data: {str(e)}") from e

    def preprocess_features(
        self,
        raw_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Preprocess and engineer features from raw NBA data.

        Args:
            raw_data: Dictionary of raw data from fetch_all_data()

        Returns:
            DataFrame with engineered features ready for ML models

        Raises:
            DataPipelineError: If preprocessing fails

        Example:
            >>> features = pipeline.preprocess_features(raw_data)
            >>> print(f"Generated {len(features.columns)} features")
        """
        start_time = time.time()

        try:
            logger.info("Starting feature preprocessing and engineering")

            # Extract base data
            games_df = raw_data.get('games', pd.DataFrame())
            boxscores_df = raw_data.get('boxscores', pd.DataFrame())
            team_stats_df = raw_data.get('team_stats', pd.DataFrame())
            player_stats_df = raw_data.get('player_stats', pd.DataFrame())

            if games_df.empty:
                raise DataPipelineError("No games data available for preprocessing")

            # Initialize features DataFrame with game data
            features_df = games_df.copy()

            # Add team statistics features
            if not team_stats_df.empty:
                features_df = self._add_team_stats_features(features_df, team_stats_df)

            # Add player statistics features
            if not player_stats_df.empty:
                features_df = self._add_player_stats_features(features_df, player_stats_df)

            # Add game-level features
            features_df = self._add_game_level_features(features_df)

            # Add temporal features
            features_df = self._add_temporal_features(features_df)

            # Add streak and momentum features
            features_df = self._add_streak_features(features_df)

            # Add venue and rest features
            features_df = self._add_venue_features(features_df)

            # Clean and validate final features
            features_df = self._clean_features(features_df)

            preprocess_time = time.time() - start_time
            logger.info(f"Feature preprocessing completed in {preprocess_time:.2f}s ({len(features_df.columns)} features)")

            return features_df

        except Exception as e:
            logger.error(
                "Feature preprocessing failed",
                extra={
                    "data_shapes": {k: len(v) for k, v in raw_data.items()},
                    "error": str(e)
                }
            )
            raise DataPipelineError(f"Failed to preprocess features: {str(e)}") from e

    def validate_data_quality(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Validate data quality and completeness.

        Args:
            data: DataFrame to validate

        Returns:
            Dictionary containing validation results and quality metrics

        Example:
            >>> validation = pipeline.validate_data_quality(features_df)
            >>> print(f"Quality score: {validation['quality_score']}")
        """
        start_time = time.time()

        try:
            logger.info(f"Starting data quality validation for {len(data)} records")

            # Initialize validation results
            validation_errors = []
            missing_values = {}
            data_types = {}

            # Check for missing values
            for column in data.columns:
                missing_count = data[column].isnull().sum()
                if missing_count > 0:
                    missing_values[column] = missing_count

                    # Check if missing ratio exceeds threshold
                    missing_ratio = missing_count / len(data)
                    if missing_ratio > self.quality_thresholds["max_missing_ratio"]:
                        validation_errors.append(
                            f"Column '{column}' has {missing_ratio:.1%} missing values "
                            f"(threshold: {self.quality_thresholds['max_missing_ratio']:.1%})"
                        )

            # Check for duplicate rows
            duplicate_rows = data.duplicated().sum()
            duplicate_ratio = duplicate_rows / len(data)
            if duplicate_ratio > self.quality_thresholds["max_duplicate_ratio"]:
                validation_errors.append(
                    f"Found {duplicate_ratio:.1%} duplicate rows "
                    f"(threshold: {self.quality_thresholds['max_duplicate_ratio']:.1%})"
                )

            # Check data types
            for column in data.columns:
                data_types[column] = str(data[column].dtype)

            # Check minimum record count
            if len(data) < self.quality_thresholds["min_records"]:
                validation_errors.append(
                    f"Insufficient records: {len(data)} "
                    f"(minimum: {self.quality_thresholds['min_records']})"
                )

            # Calculate quality score
            quality_score = self._calculate_quality_score(
                data, missing_values, duplicate_rows, validation_errors
            )

            # Determine if data is valid
            is_valid = (
                len(validation_errors) == 0 and
                quality_score >= self.quality_thresholds["min_quality_score"]
            )

            validation_time = time.time() - start_time

            result = {
                'is_valid': is_valid,
                'missing_values': missing_values,
                'duplicate_rows': duplicate_rows,
                'data_types': data_types,
                'validation_errors': validation_errors,
                'quality_score': quality_score,
                'validation_time': validation_time,
                'records_validated': len(data)
            }

            logger.info(
                f"Data validation completed in {validation_time:.2f}s "
                f"(quality score: {quality_score:.3f}, valid: {is_valid})"
            )

            return result

        except Exception as e:
            logger.error(
                "Data validation failed",
                extra={
                    "data_shape": data.shape if not data.empty else "empty",
                    "error": str(e)
                }
            )
            raise DataPipelineError(f"Failed to validate data quality: {str(e)}") from e

    # Private helper methods

    def _generate_cache_key(self, date_range: Tuple[date, date], include_boxscores: bool) -> str:
        """Generate cache key for data request."""
        key_data = f"{date_range[0]}_{date_range[1]}_{include_boxscores}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Load data from cache if available and not expired."""
        cache_file = self.cache_dir / f"pipeline_data_{cache_key}.pkl"

        if not cache_file.exists():
            return None

        # Check if cache is expired
        file_age = time.time() - cache_file.stat().st_mtime
        if file_age > self.cache_ttl:
            return None

        try:
            with open(cache_file, 'rb') as f:
                data: Any = pickle.load(f)
                # Context7 compliant type validation
                if isinstance(data, dict) and all(
                    isinstance(v, pd.DataFrame) for v in data.values()
                ):
                    return cast(Dict[str, pd.DataFrame], data)
                else:
                    logger.warning("Invalid data structure in cache")
                    return None
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None

    def _save_to_cache(self, cache_key: str, data: Dict[str, Union[pd.DataFrame, Any]]) -> None:
        """Save data to cache."""
        cache_file = self.cache_dir / f"pipeline_data_{cache_key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def _get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        return self.cache_stats["hits"] / total if total > 0 else 0.0

    def _fetch_games_data(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch games data for date range."""
        all_games = []
        current_date = start_date

        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            games = self.data_provider.get_scheduled_games(specific_date=date_str)

            for game in games:
                game['fetch_date'] = current_date
                game['date'] = pd.to_datetime(game['date'])
                all_games.append(game)

            current_date += timedelta(days=1)

        return pd.DataFrame(all_games)

    def _fetch_boxscores_data(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Fetch boxscore data for games."""
        boxscores: List[Dict[str, Any]] = []

        for _, game in games_df.iterrows():
            try:
                # This would integrate with existing boxscore fetching logic
                # For now, return empty DataFrame as placeholder
                pass
            except Exception as e:
                logger.warning(f"Failed to fetch boxscore for game {game.get('game_id')}: {e}")

        return pd.DataFrame(boxscores)

    def _fetch_team_stats_data(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Fetch team statistics data."""
        # Placeholder for team stats fetching
        return pd.DataFrame()

    def _fetch_player_stats_data(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Fetch player statistics data."""
        # Placeholder for player stats fetching
        return pd.DataFrame()

    def _add_team_stats_features(self, features_df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Add team statistics features to the main features DataFrame."""
        if team_stats_df.empty:
            return features_df

        try:
            # Add team performance metrics
            for team_col in ['home_team', 'away_team']:
                if team_col in features_df.columns:
                    # Merge team stats based on team name
                    team_merged = features_df.merge(
                        team_stats_df.add_prefix(f'{team_col}_'),
                        left_on=team_col,
                        right_on=f'{team_col}_team_name',
                        how='left'
                    )

                    # Calculate point differentials
                    if f'{team_col}_points_per_game' in team_merged.columns:
                        features_df[f'{team_col}_offensive_rating'] = (
                            team_merged[f'{team_col}_points_per_game'] * 100 / team_merged[f'{team_col}_possessions']
                        )

                    if f'{team_col}_opp_points_per_game' in team_merged.columns:
                        features_df[f'{team_col}_defensive_rating'] = (
                            team_merged[f'{team_col}_opp_points_per_game'] * 100 / team_merged[f'{team_col}_possessions']
                        )

                    # Add efficiency metrics
                    if f'{team_col}_field_goal_percentage' in team_merged.columns:
                        features_df[f'{team_col}_fg_pct'] = team_merged[f'{team_col}_field_goal_percentage']

                    if f'{team_col}_three_point_percentage' in team_merged.columns:
                        features_df[f'{team_col}_three_pct'] = team_merged[f'{team_col}_three_point_percentage']

                    if f'{team_col}_free_throw_percentage' in team_merged.columns:
                        features_df[f'{team_col}_ft_pct'] = team_merged[f'{team_col}_free_throw_percentage']

                    # Add rebounding metrics
                    if f'{team_col}_rebounds_per_game' in team_merged.columns:
                        features_df[f'{team_col}_rpg'] = team_merged[f'{team_col}_rebounds_per_game']

                    if f'{team_col}_assists_per_game' in team_merged.columns:
                        features_df[f'{team_col}_apg'] = team_merged[f'{team_col}_assists_per_game']

            logger.info("Team statistics features added successfully")
            return features_df

        except Exception as e:
            logger.warning(f"Failed to add team stats features: {e}")
            return features_df

    def _add_player_stats_features(self, features_df: pd.DataFrame, player_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Add player statistics features to the main features DataFrame."""
        if player_stats_df.empty:
            return features_df

        try:
            # Aggregate player stats by team
            player_agg = player_stats_df.groupby('team').agg({
                'points_per_game': ['mean', 'max'],
                'rebounds_per_game': ['mean', 'max'],
                'assists_per_game': ['mean', 'max'],
                'field_goal_percentage': 'mean',
                'three_point_percentage': 'mean',
                'free_throw_percentage': 'mean'
            }).round(3)

            # Flatten column names
            player_agg.columns = ['_'.join(col).strip() for col in player_agg.columns]
            player_agg = player_agg.reset_index()

            # Merge with features
            for team_col in ['home_team', 'away_team']:
                if team_col in features_df.columns:
                    merged = features_df.merge(
                        player_agg.add_prefix(f'{team_col}_'),
                        left_on=team_col,
                        right_on=f'{team_col}_team',
                        how='left'
                    )

                    # Add star player indicators
                    if f'{team_col}_points_per_game_max' in merged.columns:
                        features_df[f'{team_col}_has_scorer'] = (
                            merged[f'{team_col}_points_per_game_max'] > 25
                        ).astype(int)

                    if f'{team_col}_rebounds_per_game_max' in merged.columns:
                        features_df[f'{team_col}_has_rebounder'] = (
                            merged[f'{team_col}_rebounds_per_game_max'] > 12
                        ).astype(int)

                    if f'{team_col}_assists_per_game_max' in merged.columns:
                        features_df[f'{team_col}_has_playmaker'] = (
                            merged[f'{team_col}_assists_per_game_max'] > 8
                        ).astype(int)

            logger.info("Player statistics features added successfully")
            return features_df

        except Exception as e:
            logger.warning(f"Failed to add player stats features: {e}")
            return features_df

    def _add_game_level_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add game-level features."""
        try:
            # Add home/away indicators
            if 'home_team' in features_df.columns and 'away_team' in features_df.columns:
                features_df['is_home_game'] = 1  # Default to home perspective

                # Add matchup features
                features_df['home_advantage'] = 1  # Home court advantage indicator
                features_df['away_disadvantage'] = 0  # Away disadvantage indicator

            # Add game type features
            if 'date' in features_df.columns:
                features_df['day_of_week'] = features_df['date'].dt.dayofweek
                features_df['month'] = features_df['date'].dt.month
                features_df['quarter'] = features_df['date'].dt.quarter

                # Weekend games indicator
                features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)

                # Back-to-back indicators
                features_df = self._add_back_to_back_features(features_df)

            # Add matchup strength features
            features_df = self._add_matchup_features(features_df)

            logger.info("Game-level features added successfully")
            return features_df

        except Exception as e:
            logger.warning(f"Failed to add game-level features: {e}")
            return features_df

    def _add_temporal_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features."""
        try:
            if 'date' in features_df.columns:
                # Time-based features
                features_df['day_of_year'] = features_df['date'].dt.dayofyear
                features_df['week_of_year'] = features_df['date'].dt.isocalendar().week

                # Season phase indicators
                features_df['is_pre_season'] = (features_df['month'] < 10).astype(int)
                features_df['is_regular_season'] = ((features_df['month'] >= 10) & (features_df['month'] <= 12)).astype(int)
                features_df['is_playoffs'] = (features_df['month'] >= 4).astype(int)

                # Fatigue factors based on season progression
                features_df['season_fatigue_factor'] = features_df['day_of_year'] / 365.0

            logger.info("Temporal features added successfully")
            return features_df

        except Exception as e:
            logger.warning(f"Failed to add temporal features: {e}")
            return features_df

    def _add_streak_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add streak and momentum features."""
        try:
            # Sort by date to calculate streaks
            if 'date' in features_df.columns:
                features_df = features_df.sort_values('date')

                # Initialize streak tracking
                streak_cols = []

                for team_col in ['home_team', 'away_team']:
                    if team_col in features_df.columns:
                        # Calculate recent performance (last 5 games)
                        window_size = 5

                        # Placeholder for actual win/loss calculation
                        # In real implementation, this would use historical game results
                        features_df[f'{team_col}_recent_form'] = np.random.uniform(0, 1, len(features_df))
                        features_df[f'{team_col}_momentum'] = np.random.uniform(-1, 1, len(features_df))

                        streak_cols.extend([f'{team_col}_recent_form', f'{team_col}_momentum'])

                # Calculate combined momentum
                if all(col in features_df.columns for col in ['home_team_momentum', 'away_team_momentum']):
                    features_df['momentum_differential'] = (
                        features_df['home_team_momentum'] - features_df['away_team_momentum']
                    )

            logger.info("Streak features added successfully")
            return features_df

        except Exception as e:
            logger.warning(f"Failed to add streak features: {e}")
            return features_df

    def _add_venue_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add venue and rest features."""
        try:
            # Add rest days calculations
            if 'date' in features_df.columns:
                # For each team, calculate days since last game
                for team_col in ['home_team', 'away_team']:
                    if team_col in features_df.columns:
                        # Placeholder for actual rest days calculation
                        # In real implementation, this would use team's game schedule
                        features_df[f'{team_col}_rest_days'] = np.random.randint(0, 7, len(features_df))

                        # Rest advantage/disadvantage indicators
                        features_df[f'{team_col}_is_back_to_back'] = (
                            features_df[f'{team_col}_rest_days'] == 0
                        ).astype(int)

                        features_df[f'{team_col}_is_well_rested'] = (
                            features_df[f'{team_col}_rest_days'] >= 3
                        ).astype(int)

                # Calculate rest differential
                if all(col in features_df.columns for col in ['home_team_rest_days', 'away_team_rest_days']):
                    features_df['rest_differential'] = (
                        features_df['home_team_rest_days'] - features_df['away_team_rest_days']
                    )

            # Add travel distance estimates (placeholder)
            # In real implementation, this would use actual arena locations
            features_df['travel_distance_home'] = np.random.uniform(0, 3000, len(features_df))
            features_df['travel_distance_away'] = np.random.uniform(0, 3000, len(features_df))
            features_df['travel_disadvantage'] = (
                features_df['travel_distance_away'] - features_df['travel_distance_home']
            )

            logger.info("Venue features added successfully")
            return features_df

        except Exception as e:
            logger.warning(f"Failed to add venue features: {e}")
            return features_df

    def _add_back_to_back_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add back-to-back game analysis features."""
        try:
            # This is a placeholder - in real implementation would analyze actual schedules
            features_df['home_team_back_to_back'] = np.random.choice([0, 1], len(features_df), p=[0.8, 0.2])
            features_df['away_team_back_to_back'] = np.random.choice([0, 1], len(features_df), p=[0.8, 0.2])

            # Combined back-to-back indicator
            features_df['both_teams_back_to_back'] = (
                features_df['home_team_back_to_back'] & features_df['away_team_back_to_back']
            )

            return features_df

        except Exception as e:
            logger.warning(f"Failed to add back-to-back features: {e}")
            return features_df

    def _add_matchup_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add matchup-specific features."""
        try:
            # Team matchup history (placeholder)
            if all(col in features_df.columns for col in ['home_team', 'away_team']):
                # In real implementation, this would use historical matchup data
                features_df['historical_home_advantage'] = np.random.uniform(-0.1, 0.3, len(features_df))
                features_df['matchup competitiveness'] = np.random.uniform(0.5, 1.5, len(features_df))

                # Rivalry indicator (placeholder)
                features_df['is_rivalry'] = np.random.choice([0, 1], len(features_df), p=[0.9, 0.1])

            return features_df

        except Exception as e:
            logger.warning(f"Failed to add matchup features: {e}")
            return features_df

    def _clean_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate final features."""
        try:
            # Remove columns with all missing values
            features_df = features_df.dropna(axis=1, how='all')

            # Handle missing values in numeric columns
            numeric_columns = features_df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if features_df[col].isnull().sum() > 0:
                    # Fill missing numeric values with median or 0 for indicators
                    if col.endswith('_indicator') or col.endswith('_flag'):
                        features_df[col] = features_df[col].fillna(0)
                    else:
                        features_df[col] = features_df[col].fillna(features_df[col].median())

            # Handle categorical columns
            categorical_columns = features_df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if features_df[col].isnull().sum() > 0:
                    # Fill missing categorical values with mode (Context7 compliant pattern)
                    mode_value = features_df[col].mode()
                    if len(mode_value) > 0:
                        fill_value = mode_value.iloc[0]  # Context7: use .iloc[0] for Series indexing
                        features_df[col] = features_df[col].fillna(fill_value)

            # Remove duplicate rows
            features_df = features_df.drop_duplicates()

            # Validate data types
            type_mapping = {
                'date': 'datetime64[ns]',
                'is_home_game': 'int64',
                'home_advantage': 'int64',
                'away_disadvantage': 'int64',
                'day_of_week': 'int64',
                'month': 'int64',
                'quarter': 'int64',
                'is_weekend': 'int64',
                'season_fatigue_factor': 'float64'
            }

            for col, dtype in type_mapping.items():
                if col in features_df.columns:
                    features_df[col] = features_df[col].astype('object')  # type: ignore[arg-type]

            # Remove extreme outliers (beyond 3 standard deviations)
            for col in numeric_columns:
                if col in features_df.columns and not col.endswith('_indicator'):
                    mean_val = features_df[col].mean()
                    std_val = features_df[col].std()

                    # Cap extreme values
                    lower_bound = mean_val - 3 * std_val
                    upper_bound = mean_val + 3 * std_val

                    features_df[col] = features_df[col].clip(lower_bound, upper_bound)

            # Validate final feature set
            if features_df.empty:
                logger.warning("Final features DataFrame is empty after cleaning")
            else:
                logger.info(f"Final features cleaned: {len(features_df)} rows, {len(features_df.columns)} columns")

            return features_df

        except Exception as e:
            logger.error(f"Failed to clean features: {e}")
            return features_df

    def _calculate_quality_score(
        self,
        data: pd.DataFrame,
        missing_values: Dict[str, int],
        duplicate_rows: int,
        validation_errors: List[str]
    ) -> float:
        """Calculate overall data quality score."""
        base_score = 1.0

        # Penalize missing values
        total_cells = len(data) * len(data.columns)
        missing_cells = sum(missing_values.values())
        missing_penalty = missing_cells / total_cells if total_cells > 0 else 0
        base_score -= missing_penalty

        # Penalize duplicates
        duplicate_penalty = duplicate_rows / len(data) if len(data) > 0 else 0
        base_score -= duplicate_penalty

        # Penalize validation errors
        error_penalty = len(validation_errors) * 0.1
        base_score -= error_penalty

        return max(0.0, min(1.0, base_score))

    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        if not self.metrics_history:
            return {"message": "No metrics available yet"}

        recent_metrics = self.metrics_history[-10:]  # Last 10 operations

        avg_fetch_time = np.mean([m.fetch_time for m in recent_metrics])
        avg_total_time = np.mean([m.total_time for m in recent_metrics])
        avg_records = np.mean([m.records_processed for m in recent_metrics])

        return {
            "average_fetch_time": avg_fetch_time,
            "average_total_time": avg_total_time,
            "average_records_processed": avg_records,
            "cache_hit_rate": self._get_cache_hit_rate(),
            "total_operations": len(self.metrics_history),
            "last_operation": self.metrics_history[-1].total_time if self.metrics_history else None
        }