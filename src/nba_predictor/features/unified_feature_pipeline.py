"""Unified Feature Engineering Pipeline for NBA Predictor.

This module implements a standardized feature engineering pipeline that consolidates
all feature creation into a single, optimized system following the refactoring plan.

Implements the 5 main feature categories:
- Four Factors Features (8): eFG%, TOV%, OREB%, FT_RATE%
- Team Differentials (10): Score, rebounds, assists differentials
- Pace Features (5): HOME_PACE, AWAY_PACE, GAME_PACE
- Efficiency Features (4): Offensive/Defensive ratings
- Situational Features (6): Home/away, rest days, schedule
"""

import logging
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from ..core.data_store import UnifiedDataStore
from ..utils.exceptions import FeatureEngineeringError
from .feature_validator import NBAFeatureValidator, create_feature_validator

logger = logging.getLogger(__name__)


@dataclass
class FeatureExtractionResult:
    """Result of feature extraction from unified pipeline."""

    features_df: pd.DataFrame
    feature_metadata: Dict[str, Any]
    extraction_stats: Dict[str, Any]
    quality_metrics: Dict[str, float]
    processing_time: float


class FourFactorsEngineer:
    """Engineer Dean Oliver's Four Factors features."""

    def __init__(self):
        self.feature_names = ["efg_pct", "tov_pct", "orb_pct", "ftr"]

    def extract_features(self, game_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract Four Factors features from game data.

        Args:
            game_data: Dictionary containing game statistics

        Returns:
            Dictionary with Four Factors features
        """
        try:
            features = {}

            # Effective Field Goal Percentage
            if "field_goals_made" in game_data and "field_goals_attempted" in game_data:
                fgm = game_data["field_goals_made"]
                fga = game_data["field_goals_attempted"]
                fg3m = game_data.get("three_pointers_made", 0)
                efg_pct = ((fgm + 0.5 * fg3m) / fga) if fga > 0 else 0.450
                features["efg_pct"] = np.clip(efg_pct, 0.450, 0.580)
            else:
                features["efg_pct"] = 0.492  # NBA average

            # Turnover Percentage
            if "turnovers" in game_data and "field_goals_attempted" in game_data:
                tov = game_data["turnovers"]
                fga = game_data["field_goals_attempted"]
                fta = game_data.get("free_throws_attempted", 0)
                possessions = fga + 0.44 * fta - game_data.get("offensive_rebounds", 0)
                tov_pct = (tov / possessions) if possessions > 0 else 0.138
                features["tov_pct"] = np.clip(tov_pct, 0.100, 0.180)
            else:
                features["tov_pct"] = 0.138  # NBA average

            # Offensive Rebound Percentage
            if (
                "offensive_rebounds" in game_data
                and "field_goals_attempted" in game_data
            ):
                orb = game_data["offensive_rebounds"]
                fga = game_data["field_goals_attempted"]
                fta = game_data.get("free_throws_attempted", 0)
                possessions = fga + 0.44 * fta
                orb_pct = (orb / (possessions - orb)) if possessions > 0 else 0.217
                features["orb_pct"] = np.clip(orb_pct, 0.200, 0.320)
            else:
                features["orb_pct"] = 0.217  # NBA average

            # Free Throw Rate
            if "free_throws_made" in game_data and "field_goals_attempted" in game_data:
                ftm = game_data["free_throws_made"]
                fga = game_data["field_goals_attempted"]
                fta = game_data.get("free_throws_attempted", 0)
                ftr = (ftm / fga) if fga > 0 else 0.197
                features["ftr"] = np.clip(ftr, 0.150, 0.300)
            else:
                features["ftr"] = 0.197  # NBA average

            return features

        except Exception as e:
            logger.error(f"Error extracting Four Factors features: {e}")
            raise FeatureEngineeringError(f"Failed to extract Four Factors: {e}") from e


class TeamDifferentialEngineer:
    """Engineer team differential features."""

    def __init__(self):
        self.differential_features = [
            "score_differential",
            "rebounds_differential",
            "assists_differential",
            "steals_differential",
            "blocks_differential",
            "turnovers_differential",
            "field_goal_pct_differential",
            "three_point_pct_differential",
            "free_throw_pct_differential",
            "fouls_differential",
        ]

    def extract_features(
        self, home_stats: Dict[str, Any], away_stats: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract team differential features.

        Args:
            home_stats: Home team statistics
            away_stats: Away team statistics

        Returns:
            Dictionary with differential features
        """
        try:
            features = {}

            # Score differential
            home_score = home_stats.get("score", 0)
            away_score = away_stats.get("score", 0)
            features["score_differential"] = home_score - away_score

            # Rebounds differential
            home_reb = home_stats.get("rebounds", 0)
            away_reb = away_stats.get("rebounds", 0)
            features["rebounds_differential"] = home_reb - away_reb

            # Assists differential
            home_ast = home_stats.get("assists", 0)
            away_ast = away_stats.get("assists", 0)
            features["assists_differential"] = home_ast - away_ast

            # Steals differential
            home_stl = home_stats.get("steals", 0)
            away_stl = away_stats.get("steals", 0)
            features["steals_differential"] = home_stl - away_stl

            # Blocks differential
            home_blk = home_stats.get("blocks", 0)
            away_blk = away_stats.get("blocks", 0)
            features["blocks_differential"] = home_blk - away_blk

            # Turnovers differential
            home_tov = home_stats.get("turnovers", 0)
            away_tov = away_stats.get("turnovers", 0)
            features["turnovers_differential"] = home_tov - away_tov

            # Field goal percentage differential
            home_fgm = home_stats.get("field_goals_made", 0)
            home_fga = home_stats.get("field_goals_attempted", 1)
            away_fgm = away_stats.get("field_goals_made", 0)
            away_fga = away_stats.get("field_goals_attempted", 1)

            home_fg_pct = home_fgm / home_fga if home_fga > 0 else 0
            away_fg_pct = away_fgm / away_fga if away_fga > 0 else 0
            features["field_goal_pct_differential"] = home_fg_pct - away_fg_pct

            # Three point percentage differential
            home_fg3m = home_stats.get("three_pointers_made", 0)
            home_fg3a = home_stats.get("three_pointers_attempted", 1)
            away_fg3m = away_stats.get("three_pointers_made", 0)
            away_fg3a = away_stats.get("three_pointers_attempted", 1)

            home_3p_pct = home_fg3m / home_fg3a if home_fg3a > 0 else 0
            away_3p_pct = away_fg3m / away_fg3a if away_fg3a > 0 else 0
            features["three_point_pct_differential"] = home_3p_pct - away_3p_pct

            # Free throw percentage differential
            home_ftm = home_stats.get("free_throws_made", 0)
            home_fta = home_stats.get("free_throws_attempted", 1)
            away_ftm = away_stats.get("free_throws_made", 0)
            away_fta = away_stats.get("free_throws_attempted", 1)

            home_ft_pct = home_ftm / home_fta if home_fta > 0 else 0
            away_ft_pct = away_ftm / away_fta if away_fta > 0 else 0
            features["free_throw_pct_differential"] = home_ft_pct - away_ft_pct

            # Fouls differential (negative is bad for home team)
            home_pf = home_stats.get("fouls", 0)
            away_pf = away_stats.get("fouls", 0)
            features["fouls_differential"] = home_pf - away_pf

            return features

        except Exception as e:
            logger.error(f"Error extracting team differential features: {e}")
            raise FeatureEngineeringError(
                f"Failed to extract team differentials: {e}"
            ) from e


class PaceFeatureEngineer:
    """Engineer pace-related features."""

    def __init__(self):
        self.pace_features = [
            "home_pace",
            "away_pace",
            "game_pace",
            "pace_differential",
            "avg_pace",
        ]

    def extract_features(
        self, home_stats: Dict[str, Any], away_stats: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract pace features from team statistics.

        Args:
            home_stats: Home team statistics
            away_stats: Away team statistics

        Returns:
            Dictionary with pace features
        """
        try:
            features = {}

            # Calculate possessions for each team
            home_possessions = self._calculate_possessions(home_stats)
            away_possessions = self._calculate_possessions(away_stats)

            # Calculate pace (possessions per 48 minutes)
            home_minutes = home_stats.get("minutes", 48)
            away_minutes = away_stats.get("minutes", 48)

            features["home_pace"] = (
                (home_possessions / home_minutes) * 48 if home_minutes > 0 else 100.0
            )
            features["away_pace"] = (
                (away_possessions / away_minutes) * 48 if away_minutes > 0 else 100.0
            )

            # Game pace (average of both teams)
            features["game_pace"] = (features["home_pace"] + features["away_pace"]) / 2

            # Pace differential
            features["pace_differential"] = (
                features["home_pace"] - features["away_pace"]
            )

            # Average pace (normalized to NBA average)
            features["avg_pace"] = (
                features["game_pace"] / 100.0
            )  # Normalized to NBA average of 100

            return features

        except Exception as e:
            logger.error(f"Error extracting pace features: {e}")
            raise FeatureEngineeringError(
                f"Failed to extract pace features: {e}"
            ) from e

    def _calculate_possessions(self, team_stats: Dict[str, Any]) -> float:
        """Calculate estimated possessions from team statistics."""
        try:
            fga = team_stats.get("field_goals_attempted", 0)
            fta = team_stats.get("free_throws_attempted", 0)
            orb = team_stats.get("offensive_rebounds", 0)
            tov = team_stats.get("turnovers", 0)

            # Standard possession formula
            possessions = fga + 0.44 * fta - orb + tov
            return max(possessions, 80.0)  # Minimum realistic possessions

        except Exception as e:
            logger.warning(f"Error calculating possessions: {e}")
            return 100.0  # Default to NBA average


class EfficiencyFeatureEngineer:
    """Engineer efficiency rating features."""

    def __init__(self):
        self.efficiency_features = [
            "home_offensive_rating",
            "away_offensive_rating",
            "home_defensive_rating",
            "away_defensive_rating",
            "home_net_rating",
            "away_net_rating",
        ]

    def extract_features(
        self, home_stats: Dict[str, Any], away_stats: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract efficiency rating features.

        Args:
            home_stats: Home team statistics
            away_stats: Away team statistics

        Returns:
            Dictionary with efficiency features
        """
        try:
            features = {}

            # Calculate possessions
            home_possessions = self._calculate_possessions(home_stats)
            away_possessions = self._calculate_possessions(away_stats)

            # Offensive rating (points per 100 possessions)
            home_score = home_stats.get("score", 0)
            away_score = away_stats.get("score", 0)

            features["home_offensive_rating"] = (
                (home_score / home_possessions) * 100 if home_possessions > 0 else 110.0
            )
            features["away_offensive_rating"] = (
                (away_score / away_possessions) * 100 if away_possessions > 0 else 110.0
            )

            # Defensive rating (points allowed per 100 possessions)
            features["home_defensive_rating"] = (
                (away_score / home_possessions) * 100 if home_possessions > 0 else 110.0
            )
            features["away_defensive_rating"] = (
                (home_score / away_possessions) * 100 if away_possessions > 0 else 110.0
            )

            # Net rating (offensive - defensive)
            features["home_net_rating"] = (
                features["home_offensive_rating"] - features["home_defensive_rating"]
            )
            features["away_net_rating"] = (
                features["away_offensive_rating"] - features["away_defensive_rating"]
            )

            return features

        except Exception as e:
            logger.error(f"Error extracting efficiency features: {e}")
            raise FeatureEngineeringError(
                f"Failed to extract efficiency features: {e}"
            ) from e

    def _calculate_possessions(self, team_stats: Dict[str, Any]) -> float:
        """Calculate estimated possessions from team statistics."""
        try:
            fga = team_stats.get("field_goals_attempted", 0)
            fta = team_stats.get("free_throws_attempted", 0)
            orb = team_stats.get("offensive_rebounds", 0)
            tov = team_stats.get("turnovers", 0)

            # Standard possession formula
            possessions = fga + 0.44 * fta - orb + tov
            return max(possessions, 80.0)  # Minimum realistic possessions

        except Exception as e:
            logger.warning(f"Error calculating possessions: {e}")
            return 100.0  # Default to NBA average


class SituationalFeatureEngineer:
    """Engineer situational and context features."""

    def __init__(self):
        self.situational_features = [
            "home_court_advantage",
            "rest_days_home",
            "rest_days_away",
            "back_to_back_home",
            "back_to_back_away",
            "travel_distance_factor",
            "schedule_density",
        ]

    def extract_features(self, game_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract situational features from game context.

        Args:
            game_context: Dictionary containing game context information

        Returns:
            Dictionary with situational features
        """
        try:
            features = {}

            # Home court advantage (standard NBA value)
            features["home_court_advantage"] = 3.5  # NBA average home advantage

            # Rest days
            home_rest_days = game_context.get("home_rest_days", 2)
            away_rest_days = game_context.get("away_rest_days", 2)

            features["rest_days_home"] = float(home_rest_days)
            features["rest_days_away"] = float(away_rest_days)

            # Back to back games
            features["back_to_back_home"] = (
                1.0 if game_context.get("home_back_to_back", False) else 0.0
            )
            features["back_to_back_away"] = (
                1.0 if game_context.get("away_back_to_back", False) else 0.0
            )

            # Travel distance (simplified estimation)
            features["travel_distance_factor"] = (
                game_context.get("travel_distance_miles", 0) / 1000.0
            )

            # Schedule density (games in last week)
            features["schedule_density"] = game_context.get("games_last_week", 3) / 7.0

            return features

        except Exception as e:
            logger.error(f"Error extracting situational features: {e}")
            raise FeatureEngineeringError(
                f"Failed to extract situational features: {e}"
            ) from e


class UnifiedFeaturePipeline:
    """
    Unified Feature Engineering Pipeline for NBA Predictor.

    This class implements the complete feature engineering pipeline as specified
    in the refactoring plan, consolidating all feature creation into a single,
    optimized system.

    Feature Categories:
    - Four Factors Features (8): eFG%, TOV%, OREB%, FT_RATE%
    - Team Differentials (10): Score, rebounds, assists differentials
    - Pace Features (5): HOME_PACE, AWAY_PACE, GAME_PACE
    - Efficiency Features (4): Offensive/Defensive ratings
    - Situational Features (6): Home/away, rest days, schedule
    """

    def __init__(self, data_store: Optional[UnifiedDataStore] = None):
        """
        Initialize unified feature pipeline.

        Args:
            data_store: UnifiedDataStore instance for data access
        """
        self.data_store = data_store

        # Initialize feature engineers
        self.feature_engineers = {
            "four_factors": FourFactorsEngineer(),
            "team_differentials": TeamDifferentialEngineer(),
            "pace_features": PaceFeatureEngineer(),
            "efficiency_features": EfficiencyFeatureEngineer(),
            "situational_features": SituationalFeatureEngineer(),
        }

        # Initialize feature validator
        self.validator = create_feature_validator(data_store)

        # Performance tracking
        self.extraction_stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "average_processing_time": 0.0,
        }

        logger.info("🏀 Unified Feature Pipeline initialized")

    def extract_features(self, game_data: Dict[str, Any]) -> FeatureExtractionResult:
        """
        Extract all features for a single game.

        Args:
            game_data: Dictionary containing game data and context

        Returns:
            FeatureExtractionResult with all engineered features
        """
        start_time = datetime.now()

        try:
            logger.info("🔧 Extracting unified features for game")

            # Extract features from each engineer
            all_features = {}
            feature_metadata = {}

            # Four Factors features
            four_factors = self.feature_engineers["four_factors"].extract_features(
                game_data
            )
            all_features.update({f"home_{k}": v for k, v in four_factors.items()})
            all_features.update({f"away_{k}": v for k, v in four_factors.items()})
            feature_metadata["four_factors_count"] = len(four_factors)

            # Team differential features
            home_stats = game_data.get("home_stats", {})
            away_stats = game_data.get("away_stats", {})
            differentials = self.feature_engineers[
                "team_differentials"
            ].extract_features(home_stats, away_stats)
            all_features.update(differentials)
            feature_metadata["differentials_count"] = len(differentials)

            # Pace features
            pace_features = self.feature_engineers["pace_features"].extract_features(
                home_stats, away_stats
            )
            all_features.update(pace_features)
            feature_metadata["pace_features_count"] = len(pace_features)

            # Efficiency features
            efficiency_features = self.feature_engineers[
                "efficiency_features"
            ].extract_features(home_stats, away_stats)
            all_features.update(efficiency_features)
            feature_metadata["efficiency_features_count"] = len(efficiency_features)

            # Situational features
            game_context = game_data.get("context", {})
            situational_features = self.feature_engineers[
                "situational_features"
            ].extract_features(game_context)
            all_features.update(situational_features)
            feature_metadata["situational_features_count"] = len(situational_features)

            # Create feature DataFrame
            features_df = pd.DataFrame([all_features])

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(features_df)

            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.extraction_stats["total_extractions"] += 1
            self.extraction_stats["successful_extractions"] += 1
            self.extraction_stats["average_processing_time"] = (
                self.extraction_stats["average_processing_time"]
                * (self.extraction_stats["successful_extractions"] - 1)
                + processing_time
            ) / self.extraction_stats["successful_extractions"]

            result = FeatureExtractionResult(
                features_df=features_df,
                feature_metadata=feature_metadata,
                extraction_stats=self.extraction_stats.copy(),
                quality_metrics=quality_metrics,
                processing_time=processing_time,
            )

            logger.info(
                f"✅ Feature extraction completed: {len(features_df.columns)} features in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            self.extraction_stats["failed_extractions"] += 1
            raise FeatureEngineeringError(f"Failed to extract features: {e}") from e

    def extract_features_batch(
        self, games_data: List[Dict[str, Any]]
    ) -> FeatureExtractionResult:
        """
        Extract features for multiple games (batch processing).

        Args:
            games_data: List of game data dictionaries

        Returns:
            FeatureExtractionResult with batch features
        """
        start_time = datetime.now()

        try:
            logger.info(f"🔧 Extracting features for {len(games_data)} games (batch)")

            all_features_list = []
            total_metadata = {
                "four_factors_count": 0,
                "differentials_count": 0,
                "pace_features_count": 0,
                "efficiency_features_count": 0,
                "situational_features_count": 0,
            }

            for i, game_data in enumerate(games_data):
                try:
                    # Extract features for this game
                    game_result = self.extract_features(game_data)
                    all_features_list.append(game_result.features_df.iloc[0].to_dict())

                    # Accumulate metadata
                    for key, value in game_result.feature_metadata.items():
                        if key.endswith("_count"):
                            total_metadata[key] += value

                    if (i + 1) % 100 == 0:
                        logger.info(f"Processed {i + 1} games...")

                except Exception as e:
                    logger.warning(f"Failed to extract features for game {i}: {e}")
                    self.extraction_stats["failed_extractions"] += 1
                    continue

            # Create batch DataFrame
            if all_features_list:
                features_df = pd.DataFrame(all_features_list)
            else:
                features_df = pd.DataFrame()

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(features_df)

            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.extraction_stats["total_extractions"] += len(games_data)
            self.extraction_stats["successful_extractions"] += len(all_features_list)

            result = FeatureExtractionResult(
                features_df=features_df,
                feature_metadata=total_metadata,
                extraction_stats=self.extraction_stats.copy(),
                quality_metrics=quality_metrics,
                processing_time=processing_time,
            )

            logger.info(
                f"✅ Batch feature extraction completed: {len(features_df)} features from {len(games_data)} games in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"❌ Batch feature extraction failed: {e}")
            raise FeatureEngineeringError(
                f"Failed to extract batch features: {e}"
            ) from e

    def _calculate_quality_metrics(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate quality metrics for extracted features."""
        try:
            if features_df.empty:
                return {
                    "completeness_score": 0.0,
                    "variance_score": 0.0,
                    "correlation_score": 0.0,
                    "overall_quality": 0.0,
                }

            # Completeness score (percentage of non-null values)
            completeness_score = 1 - features_df.isnull().sum().sum() / (
                len(features_df) * len(features_df.columns)
            )

            # Variance score (average feature variance)
            numeric_columns = features_df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                variance_score = np.mean(
                    [features_df[col].var() for col in numeric_columns]
                )
                variance_score = min(variance_score / 1000.0, 1.0)  # Normalized
            else:
                variance_score = 0.0

            # Correlation score (average absolute correlation between features)
            if len(numeric_columns) > 1:
                corr_matrix = features_df[numeric_columns].corr().abs()
                # Remove diagonal (self-correlation)
                np.fill_diagonal(corr_matrix.values, np.nan)
                correlation_score = np.nanmean(corr_matrix.values)
                correlation_score = min(correlation_score, 1.0)
            else:
                correlation_score = 1.0

            # Overall quality score
            overall_quality = (
                completeness_score + variance_score + correlation_score
            ) / 3

            return {
                "completeness_score": float(completeness_score),
                "variance_score": float(variance_score),
                "correlation_score": float(correlation_score),
                "overall_quality": float(overall_quality),
            }

        except Exception as e:
            logger.warning(f"Error calculating quality metrics: {e}")
            return {
                "completeness_score": 0.5,
                "variance_score": 0.5,
                "correlation_score": 0.5,
                "overall_quality": 0.5,
            }

    def validate_and_optimize_features(
        self,
        features_df: pd.DataFrame,
        target_column: str = "total_score",
        optimization_level: str = "moderate",
    ) -> Tuple[pd.DataFrame, Any]:
        """
        Validate and optimize extracted features.

        Args:
            features_df: DataFrame with extracted features
            target_column: Target column name
            optimization_level: Optimization level ('conservative', 'moderate', 'aggressive')

        Returns:
            Tuple of (optimized DataFrame, validation report)
        """
        try:
            logger.info(
                f"🔍 Validating and optimizing features (level: {optimization_level})"
            )

            # Use feature validator for validation and optimization
            optimized_df, validation_report = self.validator.optimize_feature_set(
                features_df, target_column, optimization_level
            )

            logger.info(
                f"✅ Feature validation completed: "
                f"{len(optimized_df.columns)} features retained from {len(features_df.columns)}"
            )

            return optimized_df, validation_report

        except Exception as e:
            logger.error(f"❌ Feature validation failed: {e}")
            raise FeatureEngineeringError(f"Failed to validate features: {e}") from e

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status and statistics."""
        try:
            return {
                "pipeline_type": "Unified Feature Pipeline",
                "feature_engineers": list(self.feature_engineers.keys()),
                "extraction_stats": self.extraction_stats.copy(),
                "success_rate": (
                    self.extraction_stats["successful_extractions"]
                    / max(self.extraction_stats["total_extractions"], 1)
                ),
                "average_processing_time": self.extraction_stats[
                    "average_processing_time"
                ],
                "last_update": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {
                "pipeline_type": "Unified Feature Pipeline",
                "status": "error",
                "error": str(e),
            }

    def save_features(
        self, features_df: pd.DataFrame, filepath: str, format: str = "parquet"
    ) -> bool:
        """
        Save engineered features to file.

        Args:
            features_df: DataFrame with features to save
            filepath: Output file path
            format: Output format ('parquet', 'csv', 'json')

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(filepath)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if format.lower() == "parquet":
                features_df.to_parquet(file_path, index=False)
            elif format.lower() == "csv":
                features_df.to_csv(file_path, index=False)
            elif format.lower() == "json":
                features_df.to_json(file_path, orient="records", indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"💾 Features saved to {file_path} ({format})")
            return True

        except Exception as e:
            logger.error(f"Error saving features: {e}")
            return False


def create_unified_feature_pipeline(
    data_store: Optional[UnifiedDataStore] = None,
) -> UnifiedFeaturePipeline:
    """
    Create and configure unified feature pipeline.

    Args:
        data_store: Optional UnifiedDataStore instance

    Returns:
        Configured UnifiedFeaturePipeline instance
    """
    return UnifiedFeaturePipeline(data_store)
