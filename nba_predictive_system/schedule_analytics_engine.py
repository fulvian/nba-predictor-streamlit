#!/usr/bin/env python3
"""
🏀 NBA Schedule Analytics Engine - DevStream SuperPowered Implementation

Advanced schedule analysis system with rest days calculation, back-to-back detection,
travel fatigue analysis, and comprehensive schedule-based predictive features.

Author: NBA Predictive Analytics System
Task ID: 1.2.3 - Rest days and schedule analysis features
Date: 2025-01-10
Architecture: DevStream SuperPowered with Context Set Patterns
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

class RestDayCategory(Enum):
    """Enumeration of rest day categories"""
    ZERO_DAYS = 0      # Back-to-back
    ONE_DAY = 1        # Normal rest
    TWO_DAYS = 2       # Good rest
    THREE_PLUS_DAYS = 3 # Excellent rest

class FatigueLevel(Enum):
    """Fatigue level enumeration"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class RestDayMetrics:
    """Data class for rest day metrics with predictive insights"""
    days_since_last_game: int = 0
    rest_category: RestDayCategory = RestDayCategory.ONE_DAY
    days_until_next_game: int = 0
    is_back_to_back: bool = False
    is_three_in_four: bool = False
    is_four_in_five: bool = False
    cumulative_rest_days: int = 0
    rest_advantage_score: float = 0.0
    fatigue_level: FatigueLevel = FatigueLevel.MODERATE
    travel_fatigue_score: float = 0.0
    schedule_density_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class TeamScheduleProfile:
    """Comprehensive team schedule profile with advanced analytics"""
    team_id: int
    team_name: str
    current_metrics: RestDayMetrics
    schedule_patterns: pd.DataFrame = field(default_factory=pd.DataFrame)
    travel_patterns: Dict[str, float] = field(default_factory=dict)
    back_to_back_frequency: float = 0.0
    avg_rest_days: float = 0.0
    schedule_density: float = 0.0
    rest_advantage_rating: float = 0.0

class ScheduleAnalyticsEngine:
    """
    🚀 SuperPowered Schedule Analytics Engine

    Features:
    - Advanced rest day calculations with business day awareness
    - Back-to-back and compressed schedule detection
    - Travel fatigue analysis with distance calculations
    - Schedule density and advantage scoring
    - Context-aware predictive features
    - Real-time caching system with TTL
    - DevStream compliant architecture
    """

    def __init__(self, cache_ttl_minutes: int = 15):
        """
        Initialize the schedule analytics engine

        Args:
            cache_ttl_minutes: Cache time-to-live in minutes
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)

        # Cache for schedule calculations
        self._schedule_cache: Dict[int, TeamScheduleProfile] = {}
        self._cache_timestamps: Dict[int, datetime] = {}

        # Data storage
        self.games_df: Optional[pd.DataFrame] = None
        self.teams_df: Optional[pd.DataFrame] = None

        # NBA team locations for travel calculations (simplified)
        self.team_locations = {
            1: ("Lakers", "Los Angeles", "CA"),
            2: ("Celtics", "Boston", "MA"),
            3: ("Warriors", "San Francisco", "CA"),
            4: ("Heat", "Miami", "FL"),
            5: ("Nets", "Brooklyn", "NY"),
            6: ("Knicks", "New York", "NY"),
            7: ("Bulls", "Chicago", "IL"),
            8: ("Cavaliers", "Cleveland", "OH"),
            9: ("Suns", "Phoenix", "AZ"),
            10: ("Mavericks", "Dallas", "TX")
        }

        # Performance metrics
        self._calculation_stats = {
            'total_calculations': 0,
            'cache_hits': 0,
            'avg_calculation_time_ms': 0.0
        }

        self.logger.info("🏀 ScheduleAnalyticsEngine initialized with SuperPowered features")

    def load_games_data(self, games_df: pd.DataFrame) -> None:
        """
        Load and preprocess games data for schedule analysis

        Args:
            games_df: DataFrame containing NBA games data
        """
        start_time = datetime.now()

        # Validate required columns
        required_cols = ['game_id', 'game_date', 'home_team', 'away_team', 'season']
        missing_cols = [col for col in required_cols if col not in games_df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Preprocess data using vectorized pandas operations
        self.games_df = self._preprocess_games_data(games_df.copy())

        # Calculate schedule analytics for all teams
        self._calculate_all_team_schedules()

        load_time = (datetime.now() - start_time).total_seconds() * 1000
        self.logger.info(f"✅ Games data loaded and processed in {load_time:.2f}ms")

    def _preprocess_games_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess games data for schedule analysis

        Args:
            df: Raw games DataFrame

        Returns:
            Preprocessed DataFrame with additional features
        """
        # Convert date column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['game_date']):
            df['game_date'] = pd.to_datetime(df['game_date'])

        # Sort by date for proper schedule analysis
        df = df.sort_values(['game_date']).reset_index(drop=True)

        # Add location information
        df = self._add_location_info(df)

        # Add travel information
        df = self._calculate_travel_info(df)

        self.logger.info(f"📊 Preprocessed {len(df)} games with schedule information")
        return df

    def _add_location_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team location information to games DataFrame"""

        # Add home team location
        df['home_team_name'] = df['home_team'].map(
            lambda x: self.team_locations.get(x, ("Unknown", "Unknown", "Unknown"))[0]
        )
        df['home_city'] = df['home_team'].map(
            lambda x: self.team_locations.get(x, ("Unknown", "Unknown", "Unknown"))[1]
        )
        df['home_state'] = df['home_team'].map(
            lambda x: self.team_locations.get(x, ("Unknown", "Unknown", "Unknown"))[2]
        )

        # Add away team location
        df['away_team_name'] = df['away_team'].map(
            lambda x: self.team_locations.get(x, ("Unknown", "Unknown", "Unknown"))[0]
        )
        df['away_city'] = df['away_team'].map(
            lambda x: self.team_locations.get(x, ("Unknown", "Unknown", "Unknown"))[1]
        )
        df['away_state'] = df['away_team'].map(
            lambda x: self.team_locations.get(x, ("Unknown", "Unknown", "Unknown"))[2]
        )

        return df

    def _calculate_travel_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate travel information for each game"""

        # Simple travel distance calculation (simplified)
        def calculate_distance(row):
            # Simplified distance calculation - in real implementation would use actual distances
            home_city = row.get('home_city', 'Unknown')
            away_city = row.get('away_city', 'Unknown')
            home_state = row.get('home_state', 'Unknown')
            away_state = row.get('away_state', 'Unknown')

            if home_city == away_city:
                return 0  # Same city
            elif home_state == away_state:
                return 200  # Same state
            else:
                return 1000  # Different state (simplified)

        df['travel_distance'] = df.apply(calculate_distance, axis=1)

        # Calculate travel time (simplified: distance / 500 mph average)
        df['travel_time_hours'] = df['travel_distance'] / 500

        return df

    def _calculate_all_team_schedules(self) -> None:
        """
        Calculate schedule analytics for all teams using optimized pandas operations
        """
        if self.games_df is None:
            raise ValueError("No games data loaded")

        start_time = datetime.now()

        # Get all unique team IDs
        all_teams = set(self.games_df['home_team'].unique()) | set(self.games_df['away_team'].unique())

        # Calculate schedules for each team
        for team_id in all_teams:
            self._calculate_team_schedule(team_id)

        calculation_time = (datetime.now() - start_time).total_seconds() * 1000
        self.logger.info(f"⚡ All team schedules calculated in {calculation_time:.2f}ms")

    def _calculate_team_schedule(self, team_id: int) -> None:
        """
        Calculate comprehensive schedule analytics for a single team

        Args:
            team_id: Team identifier
        """
        # Get all games for this team (both home and away)
        team_games = self._get_team_games(team_id)

        if team_games.empty:
            return

        # Calculate rest days and schedule features
        team_games = self._calculate_rest_days(team_games, team_id)
        team_games = self._calculate_schedule_patterns(team_games, team_id)
        team_games = self._calculate_travel_patterns(team_games, team_id)

        # Create team schedule profile
        profile = self._create_team_schedule_profile(team_id, team_games)

        # Update cache
        self._update_cache(team_id, profile)

    def _get_team_games(self, team_id: int) -> pd.DataFrame:
        """Get all games for a specific team"""

        if self.games_df is None:
            return pd.DataFrame()

        # Filter games where team is either home or away
        team_mask = (self.games_df['home_team'] == team_id) | (self.games_df['away_team'] == team_id)
        team_games = self.games_df[team_mask].copy().sort_values('game_date').reset_index(drop=True)

        return team_games

    def _calculate_rest_days(self, team_games: pd.DataFrame, team_id: int) -> pd.DataFrame:
        """
        Calculate rest days and schedule density for each game

        Args:
            team_games: DataFrame of team games
            team_id: Team identifier

        Returns:
            DataFrame with rest day calculations
        """
        # Calculate days between games using pandas shift operations
        team_games = team_games.sort_values('game_date')
        team_games['prev_game_date'] = team_games['game_date'].shift(1)
        team_games['next_game_date'] = team_games['game_date'].shift(-1)

        # Calculate days since last game (rest days)
        team_games['days_since_last_game'] = (
            team_games['game_date'] - team_games['prev_game_date']
        ).dt.days.fillna(99)  # Large number for first game

        # Calculate days until next game
        team_games['days_until_next_game'] = (
            team_games['next_game_date'] - team_games['game_date']
        ).dt.days.fillna(99)  # Large number for last game

        # Determine rest category
        team_games['rest_category'] = team_games['days_since_last_game'].apply(
            lambda x: self._get_rest_category(x)
        )

        # Detect back-to-back games
        team_games['is_back_to_back'] = team_games['days_since_last_game'] == 0

        # Detect compressed schedules
        team_games = self._detect_compressed_schedules(team_games)

        return team_games

    def _get_rest_category(self, days: int) -> RestDayCategory:
        """Convert rest days to category"""
        if days == 0:
            return RestDayCategory.ZERO_DAYS
        elif days == 1:
            return RestDayCategory.ONE_DAY
        elif days == 2:
            return RestDayCategory.TWO_DAYS
        else:
            return RestDayCategory.THREE_PLUS_DAYS

    def _detect_compressed_schedules(self, team_games: pd.DataFrame) -> pd.DataFrame:
        """Detect compressed schedule patterns"""

        # Initialize columns
        team_games['is_three_in_four'] = False
        team_games['is_four_in_five'] = False

        # Check for 3 games in 4 nights
        for i in range(len(team_games)):
            if i >= 2:  # Need at least 3 games to check
                games_window = team_games.iloc[i-2:i+1]
                date_range = (games_window['game_date'].max() - games_window['game_date'].min()).days

                if date_range <= 3:  # 3 games in 3-4 days
                    team_games.loc[team_games.index[i], 'is_three_in_four'] = True

            if i >= 3:  # Need at least 4 games to check
                games_window = team_games.iloc[i-3:i+1]
                date_range = (games_window['game_date'].max() - games_window['game_date'].min()).days

                if date_range <= 4:  # 4 games in 4-5 days
                    team_games.loc[team_games.index[i], 'is_four_in_five'] = True

        return team_games

    def _calculate_schedule_patterns(self, team_games: pd.DataFrame, team_id: int) -> pd.DataFrame:
        """
        Calculate schedule pattern analytics

        Args:
            team_games: DataFrame of team games
            team_id: Team identifier

        Returns:
            DataFrame with schedule pattern calculations
        """
        # Calculate cumulative rest days
        team_games['cumulative_rest_days'] = team_games['days_since_last_game'].cumsum()

        # Calculate rest advantage score
        team_games['rest_advantage_score'] = team_games.apply(
            lambda row: self._calculate_rest_advantage(row), axis=1
        )

        # Calculate fatigue level
        team_games['fatigue_level'] = team_games.apply(
            lambda row: self._determine_fatigue_level(row), axis=1
        )

        # Calculate schedule density score
        team_games = self._calculate_schedule_density(team_games)

        return team_games

    def _calculate_rest_advantage(self, row: pd.Series) -> float:
        """
        Calculate rest advantage score based on rest days and opponent context

        Args:
            row: Game data row

        Returns:
            Rest advantage score (-1 to 1)
        """
        days_rest = row['days_since_last_game']

        # Base rest advantage
        if days_rest == 0:
            base_score = -0.8  # Big disadvantage (back-to-back)
        elif days_rest == 1:
            base_score = -0.2  # Slight disadvantage
        elif days_rest == 2:
            base_score = 0.1   # Slight advantage
        elif days_rest == 3:
            base_score = 0.3   # Good advantage
        else:  # 4+ days
            base_score = 0.5   # Strong advantage, but diminishing returns

        # Adjust for compressed schedules
        if row['is_three_in_four']:
            base_score -= 0.2
        if row['is_four_in_five']:
            base_score -= 0.3

        return np.clip(base_score, -1, 1)

    def _determine_fatigue_level(self, row: pd.Series) -> FatigueLevel:
        """Determine fatigue level based on schedule context"""

        days_rest = row['days_since_last_game']
        compressed_penalty = 0

        if row['is_three_in_four']:
            compressed_penalty += 1
        if row['is_four_in_five']:
            compressed_penalty += 1
        if row['is_back_to_back']:
            compressed_penalty += 2

        fatigue_score = compressed_penalty + (3 - min(days_rest, 3))

        if fatigue_score >= 4:
            return FatigueLevel.EXTREME
        elif fatigue_score >= 3:
            return FatigueLevel.HIGH
        elif fatigue_score >= 1:
            return FatigueLevel.MODERATE
        else:
            return FatigueLevel.LOW

    def _calculate_schedule_density(self, team_games: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate schedule density metrics

        Args:
            team_games: DataFrame of team games

        Returns:
            DataFrame with schedule density calculations
        """
        window_size = 7  # 7-day window

        for i in range(len(team_games)):
            # Get games in the past 7 days
            current_date = team_games.iloc[i]['game_date']
            window_start = current_date - timedelta(days=window_size)

            recent_games = team_games[
                (team_games['game_date'] >= window_start) &
                (team_games['game_date'] <= current_date)
            ]

            # Calculate density score
            games_in_window = len(recent_games)
            density_score = min(games_in_window / window_size, 1.0)  # Normalize to 0-1

            team_games.loc[team_games.index[i], 'schedule_density_score'] = density_score

        return team_games

    def _calculate_travel_patterns(self, team_games: pd.DataFrame, team_id: int) -> pd.DataFrame:
        """
        Calculate travel fatigue patterns

        Args:
            team_games: DataFrame of team games
            team_id: Team identifier

        Returns:
            DataFrame with travel pattern calculations
        """
        # Calculate travel fatigue score
        for i in range(len(team_games)):
            if i == 0:
                # First game of season - assume minimal travel fatigue
                travel_fatigue = 0.1
            else:
                # Calculate travel from previous game
                prev_game = team_games.iloc[i-1]
                curr_game = team_games.iloc[i]

                # Determine if team traveled
                if (curr_game['home_team'] == team_id and prev_game['away_team'] == team_id):
                    # Returning home after away game
                    travel_fatigue = 0.2
                elif (curr_game['away_team'] == team_id and prev_game['home_team'] == team_id):
                    # Going away after home game
                    travel_fatigue = 0.3 + min(curr_game['travel_distance'] / 2000, 0.4)
                elif (curr_game['away_team'] == team_id and prev_game['away_team'] == team_id):
                    # Away to away - calculate distance between cities
                    travel_fatigue = 0.4 + min(curr_game['travel_distance'] / 1500, 0.5)
                else:
                    # Home to home - minimal travel
                    travel_fatigue = 0.05

                # Adjust for back-to-back travel (harsh penalty)
                if curr_game['is_back_to_back']:
                    travel_fatigue *= 1.5

            team_games.loc[team_games.index[i], 'travel_fatigue_score'] = np.clip(travel_fatigue, 0, 1)

        return team_games

    def _create_team_schedule_profile(self, team_id: int, team_games: pd.DataFrame) -> TeamScheduleProfile:
        """
        Create comprehensive team schedule profile

        Args:
            team_id: Team identifier
            team_games: DataFrame of team games with all calculations

        Returns:
            TeamScheduleProfile with comprehensive analytics
        """
        # Get team name
        team_name = self.team_locations.get(team_id, ("Unknown", "Unknown", "Unknown"))[0]

        # Calculate team-level metrics
        back_to_back_freq = team_games['is_back_to_back'].mean()
        avg_rest_days = team_games['days_since_last_game'].mean()
        schedule_density = team_games['schedule_density_score'].mean()
        rest_advantage = team_games['rest_advantage_score'].mean()

        # Current metrics (from most recent game)
        if len(team_games) > 0:
            latest_game = team_games.iloc[-1]
            current_metrics = RestDayMetrics(
                days_since_last_game=int(latest_game['days_since_last_game']),
                rest_category=latest_game['rest_category'],
                days_until_next_game=int(latest_game['days_until_next_game']),
                is_back_to_back=bool(latest_game['is_back_to_back']),
                is_three_in_four=bool(latest_game['is_three_in_four']),
                is_four_in_five=bool(latest_game['is_four_in_five']),
                cumulative_rest_days=int(latest_game['cumulative_rest_days']),
                rest_advantage_score=float(latest_game['rest_advantage_score']),
                fatigue_level=latest_game['fatigue_level'],
                travel_fatigue_score=float(latest_game['travel_fatigue_score']),
                schedule_density_score=float(latest_game['schedule_density_score'])
            )
        else:
            current_metrics = RestDayMetrics()

        # Travel patterns summary
        travel_patterns = {
            'avg_travel_distance': team_games['travel_distance'].mean() if len(team_games) > 0 else 0,
            'total_travel_distance': team_games['travel_distance'].sum() if len(team_games) > 0 else 0,
            'max_consecutive_away': self._calculate_max_consecutive_away(team_games)
        }

        return TeamScheduleProfile(
            team_id=team_id,
            team_name=team_name,
            current_metrics=current_metrics,
            schedule_patterns=team_games,
            travel_patterns=travel_patterns,
            back_to_back_frequency=back_to_back_freq,
            avg_rest_days=avg_rest_days,
            schedule_density=schedule_density,
            rest_advantage_rating=rest_advantage
        )

    def _calculate_max_consecutive_away(self, team_games: pd.DataFrame) -> int:
        """Calculate maximum consecutive away games"""
        if team_games.empty:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for _, game in team_games.iterrows():
            if game['away_team'] == game.get('team_id', 0):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def get_team_schedule_profile(self, team_id: int, as_of_date: Optional[datetime] = None) -> TeamScheduleProfile:
        """
        Get comprehensive schedule profile for a specific team

        Args:
            team_id: Team identifier
            as_of_date: Calculate schedule as of this date (default: latest)

        Returns:
            TeamScheduleProfile with comprehensive metrics
        """
        # Check cache first
        if self._is_cache_valid(team_id):
            self._calculation_stats['cache_hits'] += 1
            return self._schedule_cache[team_id]

        start_time = datetime.now()

        if self.games_df is None:
            raise ValueError("No games data loaded")

        # Filter data for the specific team and date
        team_games = self._get_team_games(team_id)

        if as_of_date:
            team_games = team_games[team_games['game_date'] <= as_of_date]

        if team_games.empty:
            # Return empty profile for team with no games
            return self._create_empty_profile(team_id)

        # Calculate comprehensive metrics
        team_games = self._calculate_rest_days(team_games, team_id)
        team_games = self._calculate_schedule_patterns(team_games, team_id)
        team_games = self._calculate_travel_patterns(team_games, team_id)

        # Create schedule profile
        profile = self._create_team_schedule_profile(team_id, team_games)

        # Update cache
        self._update_cache(team_id, profile)

        calculation_time = (datetime.now() - start_time).total_seconds() * 1000
        self._calculation_stats['total_calculations'] += 1
        self._calculation_stats['avg_calculation_time_ms'] = (
            (self._calculation_stats['avg_calculation_time_ms'] * (self._calculation_stats['total_calculations'] - 1) + calculation_time) /
            self._calculation_stats['total_calculations']
        )

        return profile

    def _create_empty_profile(self, team_id: int) -> TeamScheduleProfile:
        """Create empty profile for team with no games"""
        team_name = self.team_locations.get(team_id, ("Unknown", "Unknown", "Unknown"))[0]

        return TeamScheduleProfile(
            team_id=team_id,
            team_name=team_name,
            current_metrics=RestDayMetrics(),
            schedule_patterns=pd.DataFrame(),
            travel_patterns={},
            back_to_back_frequency=0.0,
            avg_rest_days=0.0,
            schedule_density=0.0,
            rest_advantage_rating=0.0
        )

    def _is_cache_valid(self, team_id: int) -> bool:
        """Check if cached entry is still valid"""
        if team_id not in self._schedule_cache or team_id not in self._cache_timestamps:
            return False

        cache_age = datetime.now() - self._cache_timestamps[team_id]
        return cache_age < self.cache_ttl

    def _update_cache(self, team_id: int, profile: TeamScheduleProfile) -> None:
        """Update cache with new profile"""
        self._schedule_cache[team_id] = profile
        self._cache_timestamps[team_id] = datetime.now()

        # Clean old cache entries
        self._clean_cache()

    def _clean_cache(self) -> None:
        """Remove expired cache entries"""
        current_time = datetime.now()
        expired_teams = [
            team_id for team_id, timestamp in self._cache_timestamps.items()
            if current_time - timestamp > self.cache_ttl
        ]

        for team_id in expired_teams:
            self._schedule_cache.pop(team_id, None)
            self._cache_timestamps.pop(team_id, None)

    def get_league_schedule_analytics(self) -> Dict[str, Dict]:
        """
        Get league-wide schedule analytics summary

        Returns:
            Dictionary with league-level schedule insights
        """
        if self.games_df is None:
            return {}

        # Get all unique team IDs
        all_teams = set(self.games_df['home_team'].unique()) | set(self.games_df['away_team'].unique())

        # Calculate profiles for all teams
        team_profiles = {}
        for team_id in all_teams:
            profile = self.get_team_schedule_profile(team_id)
            team_profiles[team_id] = {
                'team_name': profile.team_name,
                'back_to_back_frequency': profile.back_to_back_frequency,
                'avg_rest_days': profile.avg_rest_days,
                'schedule_density': profile.schedule_density,
                'rest_advantage_rating': profile.rest_advantage_rating,
                'current_fatigue': profile.current_metrics.fatigue_level.value,
                'travel_fatigue': profile.current_metrics.travel_fatigue_score
            }

        # Calculate league statistics
        league_stats = {
            'avg_back_to_back_freq': np.mean([p['back_to_back_frequency'] for p in team_profiles.values()]),
            'avg_rest_days': np.mean([p['avg_rest_days'] for p in team_profiles.values()]),
            'avg_schedule_density': np.mean([p['schedule_density'] for p in team_profiles.values()]),
            'total_teams': len(team_profiles),
            'most_rested_team': max(team_profiles.items(), key=lambda x: x[1]['avg_rest_days']) if team_profiles else None,
            'most_fatigued_team': max(team_profiles.items(), key=lambda x: x[1]['schedule_density']) if team_profiles else None
        }

        return {
            'league_statistics': league_stats,
            'team_profiles': team_profiles
        }

    def get_performance_statistics(self) -> Dict[str, Union[int, float]]:
        """Get engine performance statistics"""
        cache_hit_rate = (self._calculation_stats['cache_hits'] /
                         max(self._calculation_stats['total_calculations'], 1)) * 100

        return {
            **self._calculation_stats,
            'cache_hit_rate_percent': round(cache_hit_rate, 2),
            'cached_teams': len(self._schedule_cache)
        }

    def clear_cache(self) -> None:
        """Clear all cached data"""
        self._schedule_cache.clear()
        self._cache_timestamps.clear()
        self.logger.info("🗑️ Schedule cache cleared")

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create realistic sample data for testing
    np.random.seed(42)  # For reproducible results

    # Create games data with realistic schedule patterns
    games_data = []
    teams = {
        1: 'Lakers',
        2: 'Celtics',
        3: 'Warriors'
    }

    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    for i, date in enumerate(dates):
        if i % 3 == 0:
            home, away = 1, 2
        elif i % 3 == 1:
            home, away = 2, 3
        else:
            home, away = 3, 1

        # Add some compressed schedules
        if i in [5, 6, 10, 11]:  # Back-to-back games
            pass  # Keep normal spacing
        elif i in [7, 8]:  # Three in four nights
            dates[i] = dates[i-1] + timedelta(days=1)

        games_data.append({
            'game_id': f'00{i+1:010d}',
            'game_date': date,
            'home_team': home,
            'away_team': away,
            'season': 2024
        })

    sample_games = pd.DataFrame(games_data)

    # Create schedule analytics engine
    engine = ScheduleAnalyticsEngine()

    # Load data
    engine.load_games_data(sample_games)

    # Test team profile retrieval
    profile = engine.get_team_schedule_profile(1)
    print(f"Team 1 Schedule Profile: {profile.current_metrics}")

    # Get league analytics
    league_analytics = engine.get_league_schedule_analytics()
    print(f"League Analytics: {len(league_analytics['team_profiles'])} teams analyzed")

    # Performance stats
    stats = engine.get_performance_statistics()
    print(f"Performance Stats: {stats}")