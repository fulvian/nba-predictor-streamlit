#!/usr/bin/env python3
"""
🏀 NBA Roster & Injury Data Schemas
Context7-compliant Pydantic models for comprehensive roster, injury, and lineup data.
"""

from datetime import date, datetime
from typing import Optional, List, Dict, Union
from pydantic import BaseModel, Field, validator, field_validator
from enum import Enum


class Position(str, Enum):
    """NBA player positions."""
    PG = "PG"  # Point Guard
    SG = "SG"  # Shooting Guard
    SF = "SF"  # Small Forward
    PF = "PF"  # Power Forward
    C = "C"    # Center
    G = "G"    # Guard (PG/SG)
    F = "F"    # Forward (SF/PF)
    FC = "FC"  # Forward-Center
    UTIL = "UTIL"  # Utility


class InjuryStatus(str, Enum):
    """NBA injury status classifications."""
    ACTIVE = "Active"
    OUT = "Out"
    DAY_TO_DAY = "Day-to-Day"
    QUESTIONABLE = "Questionable"
    DOUBTFUL = "Doubtful"
    PROBABLE = "Probable"
    SUSPENDED = "Suspended"
    INACTIVE = "Inactive"
    NOT_WITH_TEAM = "Not With Team"


class InjurySeverity(str, Enum):
    """Injury severity levels."""
    MINOR = "Minor"
    MODERATE = "Moderate"
    SIGNIFICANT = "Significant"
    SEVERE = "Severe"
    SEASON_ENDING = "Season-Ending"


class RosterStatus(str, Enum):
    """Player roster status."""
    ACTIVE = "Active"
    INACTIVE = "Inactive"
    TWO_WAY = "Two-Way"
    THREE_DAY = "Three-Day"
    TEN_DAY = "Ten-Day"
    ASSIGNED = "Assigned"
    WAIVED = "Waived"
    TRADED = "Traded"
    FREE_AGENT = "Free Agent"


class ContractStatus(str, Enum):
    """Contract status categories."""
    GUARANTEED = "Guaranteed"
    PARTIALLY_GUARANTEED = "Partially Guaranteed"
    NON_GUARANTEED = "Non-Guaranteed"
    EXPIRING = "Expiring"
    RESTRICTED_FREE_AGENT = "RFA"
    UNRESTRICTED_FREE_AGENT = "UFA"
    ROOKIE_SCALE = "Rookie Scale"
    TWO_WAY = "Two-Way"
    TEN_DAY = "Ten-Day"


class PlayerInfo(BaseModel):
    """Core player information model."""
    player_id: int = Field(..., description="NBA official player ID")
    player_name: str = Field(..., description="Full player name")
    first_name: str = Field(..., description="Player first name")
    last_name: str = Field(..., description="Player last name")
    slug: Optional[str] = Field(None, description="URL-friendly player name slug")
    position: Optional[Position] = Field(None, description="Primary position")
    height: Optional[str] = Field(None, description="Height in feet-inches format")
    height_cm: Optional[int] = Field(None, ge=150, le=250, description="Height in centimeters")
    weight_lbs: Optional[int] = Field(None, ge=150, le=400, description="Weight in pounds")
    weight_kg: Optional[float] = Field(None, ge=68, le=181, description="Weight in kilograms")
    birth_date: Optional[date] = Field(None, description="Date of birth")
    age: Optional[int] = Field(None, ge=15, le=50, description="Current age")
    college: Optional[str] = Field(None, description="College attended")
    draft_year: Optional[int] = Field(None, ge=2000, le=2030, description="Draft year")
    draft_round: Optional[int] = Field(None, ge=1, le=3, description="Draft round")
    draft_pick: Optional[int] = Field(None, ge=1, le=60, description="Draft pick number")
    experience_years: Optional[float] = Field(None, ge=0, le=25, description="Years of NBA experience")
    nationality: Optional[str] = Field(None, description="Country of origin")


class RosterInfo(BaseModel):
    """Team roster information for a player."""
    player_id: int = Field(..., description="NBA player ID")
    team_id: int = Field(..., description="NBA team ID")
    season: str = Field(..., description="NBA season (e.g., '2024-25')")
    jersey_number: Optional[str] = Field(None, pattern=r"^\d{1,2}$", description="Jersey number")
    position: Optional[Position] = Field(None, description="Primary position")
    secondary_position: Optional[Position] = Field(None, description="Secondary position")
    roster_status: RosterStatus = Field(..., description="Current roster status")
    contract_status: ContractStatus = Field(..., description="Contract status type")
    salary: Optional[float] = Field(None, ge=0, description="Annual salary in USD")
    acquisition_date: Optional[date] = Field(None, description="Date acquired by team")
    trade_deadline_eligible: Optional[bool] = Field(None, description="Trade deadline eligibility")
    waive_deadline_eligible: Optional[bool] = Field(None, description="Waive deadline eligibility")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    source: str = Field(..., description="Data source")

    @field_validator('salary')
    @classmethod
    def validate_salary(cls, v):
        if v is not None and v < 0:
            raise ValueError("Salary must be non-negative")
        return v


class InjuryInfo(BaseModel):
    """Comprehensive injury and availability information."""
    player_id: int = Field(..., description="NBA player ID")
    team_id: int = Field(..., description="NBA team ID")
    season: str = Field(..., description="NBA season")
    injury_status: InjuryStatus = Field(..., description="Current injury status")
    previous_status: Optional[InjuryStatus] = Field(None, description="Previous injury status")
    injury_type: Optional[str] = Field(None, description="Type of injury")
    injury_description: Optional[str] = Field(None, description="Detailed injury description")
    injury_date: Optional[date] = Field(None, description="Date injury occurred")
    expected_return: Optional[date] = Field(None, description="Expected return date")
    return_date: Optional[date] = Field(None, description="Actual return date")
    severity: Optional[InjurySeverity] = Field(None, description="Injury severity level")
    games_missed: Optional[int] = Field(0, ge=0, description="Games missed due to injury")
    consecutive_games_missed: Optional[int] = Field(0, ge=0, description="Consecutive games missed")
    game_time_decision: Optional[bool] = Field(None, description="Game-time decision status")
    availability_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Playing probability")
    practice_status: Optional[str] = Field(None, description="Practice participation status")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    source: str = Field(..., description="Data source")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Data confidence score")
    notes: Optional[str] = Field(None, description="Additional injury notes")

    @field_validator('availability_probability')
    @classmethod
    def validate_probability(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Availability probability must be between 0.0 and 1.0")
        return v


class LineupGroup(BaseModel):
    """Lineup group information."""
    group_set: str = Field(..., description="Group set identifier (e.g., '5-man')")
    group_id: int = Field(..., description="Unique group identifier")
    group_name: str = Field(..., description="Group name or player combination")
    player_ids: List[int] = Field(..., description="List of player IDs in the lineup")
    team_id: int = Field(..., description="NBA team ID")
    team_abbreviation: str = Field(..., description="Team abbreviation")


class LineupStats(BaseModel):
    """Comprehensive lineup performance statistics."""
    group_id: int = Field(..., description="Lineup group ID")
    team_id: int = Field(..., description="NBA team ID")
    season: str = Field(..., description="NBA season")
    season_type: str = Field(..., description="Season type (Regular Season, Playoffs)")
    games_played: int = Field(..., ge=0, description="Games played together")
    wins: int = Field(..., ge=0, description="Games won")
    losses: int = Field(..., ge=0, description="Games lost")
    win_percentage: float = Field(..., ge=0.0, le=1.0, description="Win percentage")
    minutes: float = Field(..., ge=0.0, description="Minutes played")
    offensive_rating: Optional[float] = Field(None, description="Points per 100 possessions")
    defensive_rating: Optional[float] = Field(None, description="Points allowed per 100 possessions")
    net_rating: Optional[float] = Field(None, description="Net rating")
    plus_minus: Optional[float] = Field(0.0, description="Plus/minus rating")
    effective_field_goal_percentage: Optional[float] = Field(None, description="Effective field goal percentage")
    true_shooting_percentage: Optional[float] = Field(None, description="True shooting percentage")
    pace: Optional[float] = Field(None, description="Pace (possessions per 48 minutes)")
    lineup_type: Optional[str] = Field(None, description="Lineup type (starting, bench, etc.)")

    # Traditional stats
    field_goals_made: Optional[float] = Field(None, ge=0.0, description="Field goals made")
    field_goals_attempted: Optional[float] = Field(None, ge=0.0, description="Field goals attempted")
    field_goal_percentage: Optional[float] = Field(None, ge=0.0, le=1.0, description="Field goal percentage")
    three_pointers_made: Optional[float] = Field(None, ge=0.0, description="3-pointers made")
    three_pointers_attempted: Optional[float] = Field(None, ge=0.0, description="3-pointers attempted")
    three_point_percentage: Optional[float] = Field(None, ge=0.0, le=1.0, description="3-point percentage")
    free_throws_made: Optional[float] = Field(None, ge=0.0, description="Free throws made")
    free_throws_attempted: Optional[float] = Field(None, ge=0.0, description="Free throws attempted")
    free_throw_percentage: Optional[float] = Field(None, ge=0.0, le=1.0, description="Free throw percentage")

    # Advanced stats
    offensive_rebounds: Optional[float] = Field(None, ge=0.0, description="Offensive rebounds")
    defensive_rebounds: Optional[float] = Field(None, ge=0.0, description="Defensive rebounds")
    total_rebounds: Optional[float] = Field(None, ge=0.0, description="Total rebounds")
    assists: Optional[float] = Field(None, ge=0.0, description="Assists")
    steals: Optional[float] = Field(None, ge=0.0, description="Steals")
    blocks: Optional[float] = Field(None, ge=0.0, description="Blocks")
    turnovers: Optional[float] = Field(None, ge=0.0, description="Turnovers")
    personal_fouls: Optional[float] = Field(None, ge=0.0, description="Personal fouls")
    points: Optional[float] = Field(None, ge=0.0, description="Points scored")

    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    source: str = Field(..., description="Data source")


class TeamRoster(BaseModel):
    """Complete team roster for a season."""
    team_id: int = Field(..., description="NBA team ID")
    team_name: str = Field(..., description="Team full name")
    team_abbreviation: str = Field(..., description="Team abbreviation")
    season: str = Field(..., description="NBA season")
    season_type: str = Field(..., description="Season type")
    players: List[RosterInfo] = Field(..., description="List of players on roster")
    total_players: int = Field(..., ge=0, description="Total number of players")
    active_players: int = Field(..., ge=0, description="Number of active players")
    injured_players: int = Field(..., ge=0, description="Number of injured players")
    total_salary: Optional[float] = Field(None, ge=0.0, description="Total team salary")
    salary_cap_space: Optional[float] = Field(None, description="Salary cap space available")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    source: str = Field(..., description="Data source")

    @field_validator('players')
    @classmethod
    def validate_players_unique(cls, v):
        player_ids = [p.player_id for p in v]
        if len(player_ids) != len(set(player_ids)):
            raise ValueError("Player IDs must be unique in roster")
        return v


class LineupAnalysis(BaseModel):
    """Team lineup analysis and effectiveness metrics."""
    team_id: int = Field(..., description="NBA team ID")
    season: str = Field(..., description="NBA season")
    season_type: str = Field(..., description="Season type")
    most_effective_lineup: Optional[LineupStats] = Field(None, description="Most effective lineup by net rating")
    most_used_lineup: Optional[LineupStats] = Field(None, description="Most frequently used lineup")
    starting_lineup: Optional[LineupStats] = Field(None, description="Starting lineup data")
    bench_lineup: Optional[LineupStats] = Field(None, description="Bench lineup data")
    total_lineups: int = Field(..., ge=0, description="Total unique lineup combinations used")
    effective_lineups: int = Field(..., ge=0, description="Number of lineups with positive net rating")
    average_lineup_size: Optional[float] = Field(None, description="Average number of players per lineup")
    lineup_diversity_score: Optional[float] = Field(None, ge=0.0, description="Lineup rotation diversity metric")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    source: str = Field(..., description="Data source")


# Context7-compliant schema mappings for API integration
ROSTER_COLUMN_MAPPING = {
    'PLAYER_ID': 'player_id',
    'PLAYER': 'player_name',
    'PLAYER_SLUG': 'slug',
    'NUM': 'jersey_number',
    'POSITION': 'position',
    'HEIGHT': 'height',
    'WEIGHT': 'weight_lbs',
    'BIRTH_DATE': 'birth_date',
    'AGE': 'age',
    'EXP': 'experience_years',
    'SCHOOL': 'college',
    'DRAFT_YEAR': 'draft_year',
    'DRAFT_ROUND': 'draft_round',
    'DRAFT_NUMBER': 'draft_pick',
    'TEAM_ID': 'team_id',
    'SEASON': 'season',
    'ROSTERSTATUS': 'roster_status',
    'CONTRACT_TYPE': 'contract_status',
    'SALARY': 'salary',
    'ACQUISITION_DATE': 'acquisition_date'
}

INJURY_COLUMN_MAPPING = {
    'PLAYER_ID': 'player_id',
    'TEAM_ID': 'team_id',
    'STATUS': 'injury_status',
    'INJURY_TYPE': 'injury_type',
    'INJURY_DATE': 'injury_date',
    'EXPECTED_RETURN': 'expected_return',
    'RETURN_DATE': 'return_date',
    'SEVERITY': 'severity',
    'GAMES_MISSED': 'games_missed',
    'CONSECUTIVE_GAMES_MISSED': 'consecutive_games_missed',
    'GAME_TIME_DECISION': 'game_time_decision',
    'AVAILABILITY_PROBABILITY': 'availability_probability',
    'PRACTICE_STATUS': 'practice_status',
    'NOTES': 'notes'
}

LINEUP_COLUMN_MAPPING = {
    'GROUP_SET': 'group_set',
    'GROUP_ID': 'group_id',
    'GROUP_NAME': 'group_name',
    'TEAM_ID': 'team_id',
    'TEAM_ABBREVIATION': 'team_abbreviation',
    'GP': 'games_played',
    'W': 'wins',
    'L': 'losses',
    'W_PCT': 'win_percentage',
    'MIN': 'minutes',
    'OFF_RATING': 'offensive_rating',
    'DEF_RATING': 'defensive_rating',
    'NET_RATING': 'net_rating',
    'PLUS_MINUS': 'plus_minus',
    'EFG_PCT': 'effective_field_goal_percentage',
    'TS_PCT': 'true_shooting_percentage',
    'PACE': 'pace',
    'FGM': 'field_goals_made',
    'FGA': 'field_goals_attempted',
    'FG_PCT': 'field_goal_percentage',
    'FG3M': 'three_pointers_made',
    'FG3A': 'three_pointers_attempted',
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
    'PTS': 'points'
}