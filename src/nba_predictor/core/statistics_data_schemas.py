#!/usr/bin/env python3
"""
🏀 NBA Statistics Data Schemas - Context7 Best Practice Design

Schema design per dati statistici NBA basato su:
- Context7 best practice per data modeling
- NBA API LeagueGameLog structure reale
- Polars optimization e validation
- Predictive analytics requirements

Data Sources:
- LeagueGameLog: Game results storici con statistiche complete
- ScoreboardV2: Box scores e player performance
- TeamGameLogs: Team-specific analysis
"""

from datetime import date, datetime
from typing import Dict, List, Any, Optional, Union
import polars as pl
from pydantic import BaseModel, Field, validator
from enum import Enum

# Context7-based Enums
class SeasonType(str, Enum):
    """NBA season types - Context7 standard"""
    REGULAR = "Regular Season"
    PLAYOFFS = "Playoffs"
    PRESEASON = "Pre Season"
    ALLSTAR = "All Star"

class GameResult(str, Enum):
    """Game result status - Context7 standard"""
    WIN = "W"
    LOSS = "L"
    TIED = "T"  # Rare but possible
    POSTPONED = "P"

class StatisticType(str, Enum):
    """Types of statistics available - Context7 standard"""
    BASE = "Base"
    ADVANCED = "Advanced"
    MISC = "Misc"
    FOUR_FACTORS = "Four Factors"
    SCORING = "Scoring"
    DEFENSE = "Defense"

# Core Data Models - Context7 Pattern
class GameResultModel(BaseModel):
    """
    Game result data model based on NBA API LeagueGameLog.
    Context7 best practice: comprehensive yet flexible schema.
    """
    # Primary identifiers
    game_id: str = Field(..., description="NBA unique game identifier")
    season_id: str = Field(..., description="NBA season identifier")
    team_id: int = Field(..., description="NBA team identifier")

    # Game metadata
    game_date: date = Field(..., description="Game date")
    matchup: str = Field(..., description="Team matchup format")
    season: str = Field(..., description="Season string (e.g., '2024-25')")
    season_type: SeasonType = Field(..., description="Type of season")

    # Game outcome
    result: GameResult = Field(..., description="Game result")
    minutes: int = Field(..., description="Minutes played")

    # Basic statistics - Context7 Core Metrics
    points: int = Field(..., ge=0, description="Points scored")
    field_goals_made: int = Field(..., ge=0, description="Field goals made")
    field_goals_attempted: int = Field(..., ge=0, description="Field goals attempted")
    field_goal_percentage: float = Field(..., ge=0, le=1, description="FG percentage")

    three_points_made: int = Field(..., ge=0, description="3-point FG made")
    three_points_attempted: int = Field(..., ge=0, description="3-point FG attempted")
    three_point_percentage: float = Field(..., ge=0, le=1, description="3P percentage")

    free_throws_made: int = Field(..., ge=0, description="Free throws made")
    free_throws_attempted: int = Field(..., ge=0, description="Free throws attempted")
    free_throw_percentage: float = Field(..., ge=0, le=1, description="FT percentage")

    # Advanced team statistics - Context7 Analytics
    offensive_rebounds: int = Field(..., ge=0, description="Offensive rebounds")
    defensive_rebounds: int = Field(..., ge=0, description="Defensive rebounds")
    total_rebounds: int = Field(..., ge=0, description="Total rebounds")

    assists: int = Field(..., ge=0, description="Assists")
    steals: int = Field(..., ge=0, description="Steals")
    blocks: int = Field(..., ge=0, description="Blocks")
    turnovers: int = Field(..., ge=0, description="Turnovers")
    personal_fouls: int = Field(..., ge=0, description="Personal fouls")

    plus_minus: int = Field(..., description="Plus/minus rating")
    video_available: bool = Field(default=True, description="Game video availability")

    # Metadata - Context7 Standard
    source: str = Field(default="NBA_API", description="Data source")
    created_at: datetime = Field(default_factory=datetime.now, description="Record creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Record update timestamp")

    @validator('field_goal_percentage')
    def validate_fg_percentage(cls, v, values):
        """Validate FG percentage calculation - Context7 validation"""
        if 'field_goals_attempted' in values and values['field_goals_attempted'] > 0:
            expected = values['field_goals_made'] / values['field_goals_attempted']
            if abs(v - expected) > 0.001:  # Allow small rounding differences
                raise ValueError(f"FG percentage mismatch: {v} vs calculated {expected}")
        return v

    @validator('three_point_percentage')
    def validate_three_point_percentage(cls, v, values):
        """Validate 3P percentage calculation - Context7 validation"""
        if 'three_points_attempted' in values and values['three_points_attempted'] > 0:
            expected = values['three_points_made'] / values['three_points_attempted']
            if abs(v - expected) > 0.001:
                raise ValueError(f"3P percentage mismatch: {v} vs calculated {expected}")
        return v

    @validator('free_throw_percentage')
    def validate_free_throw_percentage(cls, v, values):
        """Validate FT percentage calculation - Context7 validation"""
        if 'free_throws_attempted' in values and values['free_throws_attempted'] > 0:
            expected = values['free_throws_made'] / values['free_throws_attempted']
            if abs(v - expected) > 0.001:
                raise ValueError(f"FT percentage mismatch: {v} vs calculated {expected}")
        return v

class PlayerGameStatsModel(BaseModel):
    """
    Player game statistics model - Context7 design.
    Extends game results with individual player performance.
    """
    # Player identifiers
    player_id: int = Field(..., description="NBA player ID")
    player_name: str = Field(..., description="Player full name")
    team_id: int = Field(..., description="Team ID for this game")
    game_id: str = Field(..., description="Game ID")

    # Performance metrics - Context7 Core
    minutes: float = Field(..., ge=0, description="Minutes played")
    points: int = Field(..., ge=0, description="Points scored")

    # Shooting statistics
    field_goals_made: int = Field(..., ge=0)
    field_goals_attempted: int = Field(..., ge=0)
    field_goal_percentage: float = Field(..., ge=0, le=1)

    three_points_made: int = Field(..., ge=0)
    three_points_attempted: int = Field(..., ge=0)
    three_point_percentage: float = Field(..., ge=0, le=1)

    free_throws_made: int = Field(..., ge=0)
    free_throws_attempted: int = Field(..., ge=0)
    free_throw_percentage: float = Field(..., ge=0, le=1)

    # Advanced metrics
    offensive_rebounds: int = Field(..., ge=0)
    defensive_rebounds: int = Field(..., ge=0)
    total_rebounds: int = Field(..., ge=0)
    assists: int = Field(..., ge=0)
    steals: int = Field(..., ge=0)
    blocks: int = Field(..., ge=0)
    turnovers: int = Field(..., ge=0)
    personal_fouls: int = Field(..., ge=0)
    plus_minus: int = Field(..., description="Player plus/minus")

    # Metadata
    game_date: date = Field(..., description="Game date")
    source: str = Field(default="NBA_API", description="Data source")
    created_at: datetime = Field(default_factory=datetime.now)

# Polars Schema Definitions - Context7 Optimization
GAME_RESULTS_SCHEMA = pl.Schema({
    "game_id": pl.Utf8,
    "season_id": pl.Utf8,
    "team_id": pl.Int64,
    "team_abbreviation": pl.Utf8,
    "team_name": pl.Utf8,
    "game_date": pl.Date,
    "matchup": pl.Utf8,
    "result": pl.Utf8,
    "minutes": pl.Int32,
    "points": pl.Int32,
    "field_goals_made": pl.Int32,
    "field_goals_attempted": pl.Int32,
    "field_goal_percentage": pl.Float64,
    "three_points_made": pl.Int32,
    "three_points_attempted": pl.Int32,
    "three_point_percentage": pl.Float64,
    "free_throws_made": pl.Int32,
    "free_throws_attempted": pl.Int32,
    "free_throw_percentage": pl.Float64,
    "offensive_rebounds": pl.Int32,
    "defensive_rebounds": pl.Int32,
    "total_rebounds": pl.Int32,
    "assists": pl.Int32,
    "steals": pl.Int32,
    "blocks": pl.Int32,
    "turnovers": pl.Int32,
    "personal_fouls": pl.Int32,
    "plus_minus": pl.Int32,
    "video_available": pl.Boolean,
    "season": pl.Utf8,
    "season_type": pl.Utf8,
    "source": pl.Utf8,
    "created_at": pl.Datetime(time_unit='us', time_zone='UTC'),
    "updated_at": pl.Datetime(time_unit='us', time_zone='UTC')
})

PLAYER_GAME_STATS_SCHEMA = pl.Schema({
    "player_id": pl.Int64,
    "player_name": pl.Utf8,
    "team_id": pl.Int64,
    "game_id": pl.Utf8,
    "minutes": pl.Float64,
    "points": pl.Int32,
    "field_goals_made": pl.Int32,
    "field_goals_attempted": pl.Int32,
    "field_goal_percentage": pl.Float64,
    "three_points_made": pl.Int32,
    "three_points_attempted": pl.Int32,
    "three_point_percentage": pl.Float64,
    "free_throws_made": pl.Int32,
    "free_throws_attempted": pl.Int32,
    "free_throw_percentage": pl.Float64,
    "offensive_rebounds": pl.Int32,
    "defensive_rebounds": pl.Int32,
    "total_rebounds": pl.Int32,
    "assists": pl.Int32,
    "steals": pl.Int32,
    "blocks": pl.Int32,
    "turnovers": pl.Int32,
    "personal_fouls": pl.Int32,
    "plus_minus": pl.Int32,
    "game_date": pl.Date,
    "source": pl.Utf8,
    "created_at": pl.Datetime(time_unit='us', time_zone='UTC')
})

# Data Validation Functions - Context7 Best Practice
def validate_game_results_data(df: pl.DataFrame) -> tuple[bool, List[str]]:
    """
    Validate game results data against Context7 standards.

    Args:
        df: Polars DataFrame with game results

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check for empty DataFrame
    if df.height == 0:
        errors.append("DataFrame is empty")
        return False, errors

    # Flexible required columns - check for any game identifier
    game_id_candidates = ['game_id', 'GAME_ID', 'Game_ID']
    team_id_candidates = ['team_id', 'TEAM_ID', 'Team_ID']
    points_candidates = ['points', 'PTS', 'Points']

    # Find actual column names
    game_id_col = next((col for col in game_id_candidates if col in df.columns), None)
    team_id_col = next((col for col in team_id_candidates if col in df.columns), None)
    points_col = next((col for col in points_candidates if col in df.columns), None)

    if not game_id_col:
        errors.append("Missing game identifier (game_id, GAME_ID, or Game_ID)")
    if not team_id_col:
        errors.append("Missing team identifier (team_id, TEAM_ID, or Team_ID)")
    if not points_col:
        errors.append("Missing points column (points, PTS, or Points)")

    if errors:
        return False, errors

    # Validate data consistency
    try:
        # Check for negative values in numeric columns
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Int64, pl.Int32, pl.Float64]]
        for col in numeric_cols:
            if col in df.columns and (df[col] < 0).any():
                # Allow negative values only for plus_minus column
                if 'plus_minus' not in col.lower() and 'PLUS_MINUS' not in col.upper():
                    errors.append(f"Negative values found in {col}")

        # Check percentage ranges for common percentage columns
        percentage_patterns = ['percentage', 'pct', 'PCT', '_PCT']
        percentage_cols = [col for col in df.columns if any(pattern in col.upper() for pattern in percentage_patterns)]

        for col in percentage_cols:
            if col in df.columns:
                # Convert to float if it's a string
                if df[col].dtype == pl.Utf8:
                    df = df.with_columns([
                        pl.col(col).cast(pl.Float64).alias(col)
                    ])

                invalid_pct = (df[col] < 0) | (df[col] > 1)
                if invalid_pct.any():
                    # Check if these might be decimal percentages (e.g., 43.6 instead of 0.436)
                    max_val = df[col].max()
                    if max_val > 1:
                        df = df.with_columns([
                            (pl.col(col) / 100).alias(col)
                        ])
                        logger.warning(f"Converted {col} from percentage to decimal format")
                    else:
                        errors.append(f"Invalid percentage values in {col}")

    except Exception as e:
        errors.append(f"Validation error: {e}")

    return len(errors) == 0, errors

def calculate_derived_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate derived metrics for Context7 analytics.

    Args:
        df: Base game results DataFrame

    Returns:
        DataFrame with additional calculated metrics
    """
    # Calculate team efficiency ratings
    df = df.with_columns([
        # Offensive Rating (points per 100 possessions)
        (pl.col('points') * 100 / (pl.col('field_goals_attempted') + 0.44 * pl.col('free_throws_attempted') + pl.col('turnovers'))).alias('offensive_rating'),

        # True Shooting Percentage
        (pl.col('points') / (2 * (pl.col('field_goals_attempted') + 0.44 * pl.col('free_throws_attempted')))).alias('true_shooting_pct'),

        # Effective Field Goal Percentage
        ((pl.col('points') - pl.col('free_throws_made')) / (2 * pl.col('field_goals_attempted'))).alias('effective_fg_pct'),

        # Assist to Turnover Ratio
        (pl.col('assists') / pl.col('turnovers')).alias('assist_turnover_ratio'),

        # Points per Game equivalent (for context)
        (pl.col('points') / (pl.col('minutes') / 48)).alias('points_per_48_min')
    ])

    return df

# Column Mapping Functions - Context7 Standard
def map_league_game_log_columns(raw_df: pl.DataFrame) -> pl.DataFrame:
    """
    Map NBA API LeagueGameLog columns to Context7 standard schema.

    Args:
        raw_df: Raw DataFrame from NBA API

    Returns:
        Mapped DataFrame with standard column names
    """
    # NBA API to Context7 mapping
    column_mapping = {
        'SEASON_ID': 'season_id',
        'TEAM_ID': 'team_id',
        'TEAM_ABBREVIATION': 'team_abbreviation',
        'TEAM_NAME': 'team_name',
        'GAME_ID': 'game_id',
        'GAME_DATE': 'game_date',
        'MATCHUP': 'matchup',
        'WL': 'result',
        'MIN': 'minutes',
        'PTS': 'points',
        'FGM': 'field_goals_made',
        'FGA': 'field_goals_attempted',
        'FG_PCT': 'field_goal_percentage',
        'FG3M': 'three_points_made',
        'FG3A': 'three_points_attempted',
        'FG3_PCT': 'three_point_percentage',
        'FTM': 'free_throws_made',
        'FTA': 'free_throws_attempted',
        'FT_PCT': 'free_throw_percentage',
        'OREB': 'offensive_rebounds',
        'DREB': 'defensive_rebounds',
        'REB': 'total_rebounds',
        'AST': 'assists',
        'STL': 'steals',
        'BLK': 'blocks',
        'TOV': 'turnovers',
        'PF': 'personal_fouls',
        'PLUS_MINUS': 'plus_minus',
        'VIDEO_AVAILABLE': 'video_available'
    }

    # Apply mapping only for existing columns
    existing_mapping = {k: v for k, v in column_mapping.items() if k in raw_df.columns}
    mapped_df = raw_df.rename(existing_mapping)

    # Add Context7 metadata columns
    mapped_df = mapped_df.with_columns([
        pl.lit("2024-25").alias("season"),
        pl.lit("Regular Season").alias("season_type"),
        pl.lit("NBA_API").alias("source"),
        pl.lit(datetime.now()).alias("created_at"),
        pl.lit(datetime.now()).alias("updated_at")
    ])

    # Convert date column if needed
    if 'game_date' in mapped_df.columns:
        mapped_df = mapped_df.with_columns([
            pl.col('game_date').str.strptime(pl.Date, '%Y-%m-%d')
        ])

    return mapped_df

if __name__ == "__main__":
    # Test schema validation
    print("🏀 Testing NBA Statistics Data Schemas")

    # Test with sample data
    sample_data = {
        "game_id": ["0022401186"],
        "season_id": ["22024"],
        "team_id": [1610612747],
        "team_abbreviation": ["LAL"],
        "team_name": ["Los Angeles Lakers"],
        "game_date": [date(2025, 4, 13)],
        "matchup": ["LAL vs. BOS"],
        "result": ["W"],
        "minutes": [48],
        "points": [117],
        "field_goals_made": [42],
        "field_goals_attempted": [94],
        "field_goal_percentage": [0.436],
        "assists": [28],
        "rebounds": [48],
        "source": ["TEST"]
    }

    # Create DataFrame and validate
    df = pl.DataFrame(sample_data)
    is_valid, errors = validate_game_results_data(df)

    print(f"✅ Schema validation: {'PASS' if is_valid else 'FAIL'}")
    if errors:
        print(f"❌ Errors: {errors}")

    # Test column mapping
    mapped_df = calculate_derived_metrics(df)
    print(f"✅ Derived metrics calculated: {len(mapped_df.columns)} columns")

    print("🎉 Statistics schemas ready for implementation!")