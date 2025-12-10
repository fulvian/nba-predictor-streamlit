#!/usr/bin/env python3
"""
🏀 Research-Based Feature Engineering - Context7 Compliant
Advanced feature engineering for NBA predictions based on academic research.

This module implements:
- Four Factors calculations (eFG%, TOV%, ORB%, FTR%)
- Pace explosion features and momentum indicators
- Team advantage calculations and differential features
- Research-based feature combinations
- Proper error handling and validation
"""

import logging
from typing import Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def enhance_nba_features(
    df: pd.DataFrame,
    four_factors_columns: List[str],
    momentum_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Enhance NBA dataset with research-based features.

    Args:
        df: Base NBA dataset
        four_factors_columns: List of Four Factors column names
        momentum_data: Optional player momentum data

    Returns:
        Enhanced DataFrame with research features

    Raises:
        ValueError: If required columns missing
        KeyError: If column names invalid

    Example:
        >>> enhanced = enhance_nba_features(df, ['eFG%', 'TOV%', 'ORB%', 'FTR%'])
        >>> enhanced.columns.tolist()
        ['eFG%', 'TOV%', 'ORB%', 'FTR%', 'efg_advantage', 'pace_explosion', ...]
    """
    try:
        # Validate input
        validate_input_data(df, four_factors_columns)

        # Create working copy
        enhanced_df = df.copy()

        # Apply research-based feature engineering
        enhanced_df = calculate_four_factors_features(enhanced_df, four_factors_columns)
        enhanced_df = calculate_team_differentials(enhanced_df)
        enhanced_df = calculate_pace_features(enhanced_df)
        enhanced_df = calculate_efficiency_features(enhanced_df)
        enhanced_df = calculate_situational_features(enhanced_df)

        # Add momentum features if data provided
        if momentum_data is not None:
            enhanced_df = integrate_momentum_features(enhanced_df, momentum_data)

        # Calculate interaction features (NEW)
        enhanced_df = calculate_interaction_features(enhanced_df)

        logger.info(
            "Research features enhanced successfully",
            extra={
                "original_columns": len(df.columns),
                "enhanced_columns": len(enhanced_df.columns),
                "features_added": len(enhanced_df.columns) - len(df.columns),
            },
        )

        return enhanced_df

    except (ValueError, KeyError) as e:
        logger.error(
            "Feature enhancement failed",
            extra={
                "input_shape": df.shape,
                "four_factors_columns": four_factors_columns,
                "error": str(e),
            },
        )
        raise ValueError(f"Feature enhancement failed: {e}") from e


def validate_input_data(df: pd.DataFrame, four_factors_columns: List[str]) -> None:
    """
    Validate input data for feature enhancement.

    Args:
        df: Input DataFrame
        four_factors_columns: List of required column names

    Raises:
        ValueError: If validation fails
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    if len(four_factors_columns) < 4:
        raise ValueError("At least 4 Four Factors columns required")

    # Check for required columns
    missing_cols = [col for col in four_factors_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Validate data types
    for col in four_factors_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} must be numeric")


def calculate_four_factors_features(
    df: pd.DataFrame, four_factors_columns: List[str]
) -> pd.DataFrame:
    """
    Calculate Four Factors based features and advantages.

    Args:
        df: Input DataFrame
        four_factors_columns: List of Four Factors column names

    Returns:
        DataFrame with Four Factors enhancements
    """
    enhanced_df = df.copy()

    # Calculate Four Factors combinations
    if len(four_factors_columns) >= 4:
        efg_col, tov_col, orb_col, ftr_col = four_factors_columns[:4]

        # Four Factors product (overall efficiency measure)
        enhanced_df["four_factors_product"] = (
            enhanced_df[efg_col]
            * (1 - enhanced_df[tov_col])
            * enhanced_df[orb_col]
            * enhanced_df[ftr_col]
        )

        # Four Factors weighted sum (Dean Oliver's formula approximation)
        enhanced_df["four_factors_weighted"] = (
            0.4 * enhanced_df[efg_col]
            + 0.25 * (1 - enhanced_df[tov_col])
            + 0.2 * enhanced_df[orb_col]
            + 0.15 * enhanced_df[ftr_col]
        )

        # Shooting efficiency score
        enhanced_df["shooting_efficiency"] = enhanced_df[efg_col] * 0.4

        # Possession efficiency
        enhanced_df["possession_efficiency"] = (1 - enhanced_df[tov_col]) * 0.25

        # Rebounding contribution
        enhanced_df["rebounding_contribution"] = enhanced_df[orb_col] * 0.2

        # Free throw contribution
        enhanced_df["free_throw_contribution"] = enhanced_df[ftr_col] * 0.15

    logger.debug("Four Factors features calculated", extra={"features_count": 6})

    return enhanced_df


def calculate_team_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate team differential features.

    Args:
        df: Input DataFrame with team statistics

    Returns:
        DataFrame with differential features
    """
    enhanced_df = df.copy()

    # Common differential patterns (assuming naming conventions)
    differential_pairs = [
        ("team1_score", "team2_score", "score_differential"),
        ("team1_field_goals_made", "team2_field_goals_made", "fg_differential"),
        (
            "team1_three_pointers_made",
            "team2_three_pointers_made",
            "threes_differential",
        ),
        ("team1_free_throws_made", "team2_free_throws_made", "ft_differential"),
        ("team1_rebounds", "team2_rebounds", "rebounds_differential"),
        ("team1_assists", "team2_assists", "assists_differential"),
        ("team1_steals", "team2_steals", "steals_differential"),
        ("team1_blocks", "team2_blocks", "blocks_differential"),
        ("team1_turnovers", "team2_turnovers", "turnovers_differential"),
        ("team1_fouls", "team2_fouls", "fouls_differential"),
    ]

    for team1_col, team2_col, diff_col in differential_pairs:
        if team1_col in enhanced_df.columns and team2_col in enhanced_df.columns:
            enhanced_df[diff_col] = enhanced_df[team1_col] - enhanced_df[team2_col]

    # Calculate scoring ratios
    if "team1_score" in enhanced_df.columns and "team2_score" in enhanced_df.columns:
        enhanced_df["scoring_ratio"] = enhanced_df["team1_score"] / (
            enhanced_df["team2_score"] + 1e-6  # Avoid division by zero
        )

    logger.debug(
        "Team differential features calculated",
        extra={"differentials": len(differential_pairs)},
    )

    return enhanced_df


def calculate_pace_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate pace-related features.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with pace features
    """
    enhanced_df = df.copy()

    # Total possessions estimation
    if all(
        col in enhanced_df.columns
        for col in [
            "team1_field_goals_attempted",
            "team2_field_goals_attempted",
            "team1_free_throws_attempted",
            "team2_free_throws_attempted",
            "team1_offensive_rebounds",
            "team2_offensive_rebounds",
            "team1_turnovers",
            "team2_turnovers",
        ]
    ):
        # Dean Oliver's possessions formula
        team1_possessions = (
            enhanced_df["team1_field_goals_attempted"]
            + enhanced_df["team1_free_throws_attempted"] * 0.44
            + enhanced_df["team1_offensive_rebounds"]
            - enhanced_df["team1_turnovers"]
        )

        team2_possessions = (
            enhanced_df["team2_field_goals_attempted"]
            + enhanced_df["team2_free_throws_attempted"] * 0.44
            + enhanced_df["team2_offensive_rebounds"]
            - enhanced_df["team2_turnovers"]
        )

        enhanced_df["total_possessions"] = team1_possessions + team2_possessions
        enhanced_df["pace_possessions"] = (
            enhanced_df["total_possessions"] / 2
        )  # Average per team
        enhanced_df["pace_explosion"] = (
            enhanced_df["total_possessions"] > 200
        )  # High pace indicator

    # Shooting volume indicators
    if all(
        col in enhanced_df.columns
        for col in ["team1_field_goals_attempted", "team2_field_goals_attempted"]
    ):
        enhanced_df["total_shot_attempts"] = (
            enhanced_df["team1_field_goals_attempted"]
            + enhanced_df["team2_field_goals_attempted"]
        )
        enhanced_df["shooting_volume"] = (
            enhanced_df["total_shot_attempts"] / 100
        )  # Normalized

    # Three point shooting volume
    if all(
        col in enhanced_df.columns
        for col in ["team1_three_pointers_attempted", "team2_three_pointers_attempted"]
    ):
        enhanced_df["total_three_attempts"] = (
            enhanced_df["team1_three_pointers_attempted"]
            + enhanced_df["team2_three_pointers_attempted"]
        )
        enhanced_df["three_point_volume"] = (
            enhanced_df["total_three_attempts"] / 50
        )  # Normalized

    logger.debug("Pace features calculated", extra={"pace_features": 6})

    return enhanced_df


def calculate_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate shooting and efficiency features.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with efficiency features
    """
    enhanced_df = df.copy()

    # True Shooting Percentage (TS%)
    if all(
        col in enhanced_df.columns
        for col in [
            "team1_points",
            "team1_field_goals_attempted",
            "team1_free_throws_attempted",
        ]
    ):
        enhanced_df["team1_ts_percentage"] = enhanced_df["team1_points"] / (
            2
            * (
                enhanced_df["team1_field_goals_attempted"]
                + 0.44 * enhanced_df["team1_free_throws_attempted"]
            )
        )

    if all(
        col in enhanced_df.columns
        for col in [
            "team2_points",
            "team2_field_goals_attempted",
            "team2_free_throws_attempted",
        ]
    ):
        enhanced_df["team2_ts_percentage"] = enhanced_df["team2_points"] / (
            2
            * (
                enhanced_df["team2_field_goals_attempted"]
                + 0.44 * enhanced_df["team2_free_throws_attempted"]
            )
        )

    # Effective Field Goal Percentage (eFG%)
    if all(
        col in enhanced_df.columns
        for col in [
            "team1_field_goals_made",
            "team1_three_pointers_made",
            "team1_field_goals_attempted",
        ]
    ):
        enhanced_df["team1_efg_percentage"] = (
            enhanced_df["team1_field_goals_made"]
            + 0.5 * enhanced_df["team1_three_pointers_made"]
        ) / enhanced_df["team1_field_goals_attempted"]

    if all(
        col in enhanced_df.columns
        for col in [
            "team2_field_goals_made",
            "team2_three_pointers_made",
            "team2_field_goals_attempted",
        ]
    ):
        enhanced_df["team2_efg_percentage"] = (
            enhanced_df["team2_field_goals_made"]
            + 0.5 * enhanced_df["team2_three_pointers_made"]
        ) / enhanced_df["team2_field_goals_attempted"]

    # Efficiency differentials
    if (
        "team1_ts_percentage" in enhanced_df.columns
        and "team2_ts_percentage" in enhanced_df.columns
    ):
        enhanced_df["ts_percentage_differential"] = (
            enhanced_df["team1_ts_percentage"] - enhanced_df["team2_ts_percentage"]
        )

    if (
        "team1_efg_percentage" in enhanced_df.columns
        and "team2_efg_percentage" in enhanced_df.columns
    ):
        enhanced_df["efg_percentage_differential"] = (
            enhanced_df["team1_efg_percentage"] - enhanced_df["team2_efg_percentage"]
        )

    logger.debug("Efficiency features calculated", extra={"efficiency_features": 5})

    return enhanced_df


def calculate_situational_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate situational and context-based features.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with situational features
    """
    enhanced_df = df.copy()

    # Scoring balance (inside vs outside)
    if all(
        col in enhanced_df.columns
        for col in ["team1_two_pointers_made", "team1_three_pointers_made"]
    ):
        total_field_goals = (
            enhanced_df["team1_two_pointers_made"]
            + enhanced_df["team1_three_pointers_made"]
        )
        enhanced_df["team1_three_point_ratio"] = enhanced_df[
            "team1_three_pointers_made"
        ] / (total_field_goals + 1e-6)

    if all(
        col in enhanced_df.columns
        for col in ["team2_two_pointers_made", "team2_three_pointers_made"]
    ):
        total_field_goals = (
            enhanced_df["team2_two_pointers_made"]
            + enhanced_df["team2_three_pointers_made"]
        )
        enhanced_df["team2_three_point_ratio"] = enhanced_df[
            "team2_three_pointers_made"
        ] / (total_field_goals + 1e-6)

    # Assists to field goals ratio (ball movement)
    if all(
        col in enhanced_df.columns
        for col in ["team1_assists", "team1_field_goals_made"]
    ):
        enhanced_df["team1_assist_ratio"] = enhanced_df["team1_assists"] / (
            enhanced_df["team1_field_goals_made"] + 1e-6
        )

    if all(
        col in enhanced_df.columns
        for col in ["team2_assists", "team2_field_goals_made"]
    ):
        enhanced_df["team2_assist_ratio"] = enhanced_df["team2_assists"] / (
            enhanced_df["team2_field_goals_made"] + 1e-6
        )

    # Turnover rate
    if all(
        col in enhanced_df.columns for col in ["team1_turnovers", "team1_possessions"]
    ):
        enhanced_df["team1_turnover_rate"] = enhanced_df["team1_turnovers"] / (
            enhanced_df["team1_possessions"] + 1e-6
        )

    if all(
        col in enhanced_df.columns for col in ["team2_turnovers", "team2_possessions"]
    ):
        enhanced_df["team2_turnover_rate"] = enhanced_df["team2_turnovers"] / (
            enhanced_df["team2_possessions"] + 1e-6
        )

    # Rebounding rates
    if all(
        col in enhanced_df.columns
        for col in ["team1_defensive_rebounds", "team1_possessions"]
    ):
        enhanced_df["team1_defensive_rebound_rate"] = enhanced_df[
            "team1_defensive_rebounds"
        ] / (enhanced_df["team1_possessions"] + 1e-6)

    if all(
        col in enhanced_df.columns
        for col in ["team2_defensive_rebounds", "team2_possessions"]
    ):
        enhanced_df["team2_defensive_rebound_rate"] = enhanced_df[
            "team2_defensive_rebounds"
        ] / (enhanced_df["team2_possessions"] + 1e-6)

    logger.debug("Situational features calculated", extra={"situational_features": 8})

    return enhanced_df


def integrate_momentum_features(
    df: pd.DataFrame, momentum_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Integrate momentum-based features.

    Args:
        df: Base game data
        momentum_data: Player/team momentum indicators

    Returns:
        DataFrame with momentum features
    """
    enhanced_df = df.copy()

    # Add momentum indicators if available
    if "team_momentum" in momentum_data.columns:
        # Simple momentum integration (would need proper team matching in real implementation)
        enhanced_df["team1_momentum"] = (
            momentum_data["team_momentum"].iloc[: len(enhanced_df)].values
        )
        enhanced_df["team2_momentum"] = (
            momentum_data["team_momentum"].iloc[: len(enhanced_df)].values
        )

    if "player_form" in momentum_data.columns:
        # Player form aggregation (simplified)
        enhanced_df["avg_player_form"] = (
            momentum_data["player_form"].iloc[: len(enhanced_df)].values
        )

    logger.info(
        "Momentum features integrated",
        extra={"momentum_columns": momentum_data.columns.tolist()},
    )

    return enhanced_df


def get_feature_importance_ranking(
    df: pd.DataFrame, target_column: str = "total_score"
) -> Dict[str, float]:
    """
    Calculate basic feature importance using correlation with target.

    Args:
        df: Enhanced DataFrame with features
        target_column: Target variable column name

    Returns:
        Dictionary mapping feature names to importance scores

    Raises:
        ValueError: If target column not found
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    # Calculate correlation with target
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = (
        df[numeric_cols].corrwith(df[target_column]).abs().sort_values(ascending=False)
    )

    # Convert to dictionary
    importance_dict = correlations.drop(target_column).to_dict()

    logger.info(
        "Feature importance calculated",
        extra={"target_column": target_column, "features_ranked": len(importance_dict)},
    )

    return importance_dict


def validate_feature_engineering_pipeline(
    df_original: pd.DataFrame, df_enhanced: pd.DataFrame, expected_feature_count: int
) -> bool:
    """
    Validate that feature engineering pipeline worked correctly.

    Args:
        df_original: Original DataFrame
        df_enhanced: Enhanced DataFrame
        expected_feature_count: Expected number of new features

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails
    """
    # Check that we have more features
    if len(df_enhanced.columns) <= len(df_original.columns):
        raise ValueError("No new features were created")

    # Check expected feature count (with tolerance)
    actual_feature_count = len(df_enhanced.columns) - len(df_original.columns)
    if abs(actual_feature_count - expected_feature_count) > 5:
        logger.warning(
            "Feature count differs from expected",
            extra={
                "expected": expected_feature_count,
                "actual": actual_feature_count,
                "tolerance": 5,
            },
        )

    # Check for NaN values in new features
    new_features = set(df_enhanced.columns) - set(df_original.columns)
    for feature in new_features:
        if df_enhanced[feature].isna().all():
            raise ValueError(f"Feature '{feature}' contains only NaN values")

    logger.info(
        "Feature engineering validation passed",
        extra={
            "original_features": len(df_original.columns),
            "enhanced_features": len(df_enhanced.columns),
            "new_features": len(new_features),
        },
    )

    return True

def calculate_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate interaction features (e.g., Pace * Efficiency).

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with interaction features
    """
    enhanced_df = df.copy()

    # Define minimal epsilon to avoid errors
    epsilon = 1e-6

    # 1. Pace * Efficiency (Context7 / Perplexity Suggestion)
    # Allows model to distinguish "Fast & Good" vs "Fast & Bad"
    # We need Pace and ORtg.
    # If they are not present, we try to calculate them on the fly if underlying metrics exist.

    # Check for Pace (pace_possessions)
    if 'pace_possessions' in enhanced_df.columns:
        pace = enhanced_df['pace_possessions']
    elif 'pace' in enhanced_df.columns:
        pace = enhanced_df['pace']
    else:
        # Cannot calc interactions involving pace
        pace = None

    # Helper for team-specific interactions
    for team_prefix in ['team1', 'team2']:
        if pace is not None:
            # Efficiency Interaction
            # Check for ORtg
            ortg_col = f'{team_prefix}_offensive_rating'
            if ortg_col in enhanced_df.columns:
                enhanced_df[f'{team_prefix}_pace_x_ortg'] = pace * enhanced_df[ortg_col]
            
            # Alternative: approximate ORtg using Score / Pace if ORtg missing but Score exists
            elif f'{team_prefix}_score' in enhanced_df.columns:
                 # Score per possession * 100 ~= ORtg
                 approx_ortg = (enhanced_df[f'{team_prefix}_score'] / (pace + epsilon)) * 100
                 enhanced_df[f'{team_prefix}_pace_x_ortg'] = pace * approx_ortg

            # Defense Interaction
            drtg_col = f'{team_prefix}_defensive_rating'
            if drtg_col in enhanced_df.columns:
                enhanced_df[f'{team_prefix}_pace_x_drtg'] = pace * enhanced_df[drtg_col]

        # 2. Shooting Profile * Defense Interaction (3P Rate * Opponent 3P Allowed?)
        # 3P Rate
        t3p_rate_col = f'{team_prefix}_three_point_rate'
        # Opponent prefix
        opp_prefix = 'team2' if team_prefix == 'team1' else 'team1'
        
        # We don't strictly have "Opponent 3P Allowed" as a pre-calc column here usually,
        # but we can assume opponent's defensive rating is a proxy for general defense,
        # or if we had opponent 3P% allowed.
        # Let's stick to what we have:
        # Interaction: Own 3P Rate * Opponent Defensive Rating?
        # A team that shoots many 3s vs a bad defense -> Good?
        
        # Let's just do Volume * Efficiency interaction
        # 3P Volume * 3P%
        if f'{team_prefix}_three_pointers_attempted' in enhanced_df.columns and            f'{team_prefix}_three_pointers_made' in enhanced_df.columns:
               
           attempts = enhanced_df[f'{team_prefix}_three_pointers_attempted']
           # Avoid division by zero
           pct = enhanced_df[f'{team_prefix}_three_pointers_made'] / (attempts + epsilon)
           enhanced_df[f'{team_prefix}_volume_x_efficiency_3p'] = attempts * pct

    logger.debug("Interaction features calculated")
    return enhanced_df
