#!/usr/bin/env python3
"""
🔥 REAL-TIME NBA STREAK ANALYZER - DevStream SuperPowered Implementation

Advanced real-time streak detection system using pandas rolling calculations
and context-driven patterns for NBA predictive analytics.

Author: NBA Predictive Analytics System
Task ID: 1.2.1 - Real-time streak detection
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

class StreakType(Enum):
    """Enumeration of streak types"""
    WINNING = 'W'
    LOSING = 'L'
    NEUTRAL = 'N'

@dataclass
class StreakMetrics:
    """Data class for streak metrics with context-aware scoring"""
    current_streak: int = 0
    streak_type: StreakType = StreakType.NEUTRAL
    season_longest_win: int = 0
    season_longest_loss: int = 0
    recent_form: float = 0.0
    momentum_score: float = 0.0
    streak_strength: float = 0.0
    consistency_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class TeamStreakProfile:
    """Comprehensive team streak profile with advanced metrics"""
    team_id: int
    team_name: str
    current_metrics: StreakMetrics
    historical_context: Dict[str, float] = field(default_factory=dict)
    streak_trends: pd.DataFrame = field(default_factory=pd.DataFrame)
    performance_volatility: float = 0.0

class RealTimeStreakAnalyzer:
    """
    🚀 SuperPowered Real-Time Streak Detection System

    Features:
    - Vectorized pandas rolling calculations with groupby optimization
    - Context-aware momentum scoring
    - Real-time streak updates with caching
    - Advanced statistical metrics
    - DevStream compliant architecture
    """

    def __init__(self, cache_ttl_minutes: int = 15):
        """
        Initialize the real-time streak analyzer

        Args:
            cache_ttl_minutes: Cache time-to-live in minutes
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)

        # Cache for streak calculations
        self._streak_cache: Dict[int, TeamStreakProfile] = {}
        self._cache_timestamps: Dict[int, datetime] = {}

        # Data storage
        self.games_df: Optional[pd.DataFrame] = None
        self.teams_df: Optional[pd.DataFrame] = None

        # Performance metrics
        self._calculation_stats = {
            'total_calculations': 0,
            'cache_hits': 0,
            'avg_calculation_time_ms': 0.0
        }

        self.logger.info("🔥 RealTimeStreakAnalyzer initialized with SuperPowered features")

    def load_games_data(self, games_df: pd.DataFrame) -> None:
        """
        Load and preprocess games data for streak analysis

        Args:
            games_df: DataFrame containing NBA games data
        """
        start_time = datetime.now()

        # Validate required columns (team_id can be derived from home_team/away_team)
        required_cols = ['game_id', 'game_date', 'home_team', 'away_team',
                        'home_score', 'away_score', 'season']
        missing_cols = [col for col in required_cols if col not in games_df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Add team_id column if missing (for compatibility)
        if 'team_id' not in games_df.columns:
            games_df['team_id'] = games_df['home_team']

        # Preprocess data using vectorized pandas operations
        self.games_df = self._preprocess_games_data(games_df.copy())

        # Calculate all streaks using optimized pandas operations
        self._calculate_all_team_streaks()

        load_time = (datetime.now() - start_time).total_seconds() * 1000
        self.logger.info(f"✅ Games data loaded and processed in {load_time:.2f}ms")

    def _preprocess_games_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess games data using vectorized pandas operations

        Args:
            df: Raw games DataFrame

        Returns:
            Preprocessed DataFrame with additional features
        """
        # Convert date column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['game_date']):
            df['game_date'] = pd.to_datetime(df['game_date'])

        # Create team-specific game records (each game appears twice, once per team)
        home_games = df[['game_id', 'game_date', 'home_team', 'home_score', 'away_score', 'season']].copy()
        home_games['team_id'] = home_games['home_team']
        home_games['opponent_id'] = home_games['away_team']
        home_games['team_score'] = home_games['home_score']
        home_games['opponent_score'] = home_games['away_score']
        home_games['is_home'] = True

        away_games = df[['game_id', 'game_date', 'away_team', 'away_score', 'home_score', 'season']].copy()
        away_games['team_id'] = away_games['away_team']
        away_games['opponent_id'] = away_games['home_team']
        away_games['team_score'] = away_games['away_score']
        away_games['opponent_score'] = away_games['home_score']
        away_games['is_home'] = False

        # Combine home and away games
        team_games = pd.concat([home_games, away_games], ignore_index=True)

        # Calculate win/loss result using vectorized operations
        team_games['won'] = (team_games['team_score'] > team_games['opponent_score']).astype(int)
        team_games['margin'] = team_games['team_score'] - team_games['opponent_score']

        # Sort by team and date for proper streak calculation
        team_games = team_games.sort_values(['team_id', 'game_date']).reset_index(drop=True)

        # Add game number for each team
        team_games['team_game_number'] = team_games.groupby('team_id').cumcount() + 1

        self.logger.info(f"📊 Preprocessed {len(team_games)} team-game records")
        return team_games

    def _calculate_all_team_streaks(self) -> None:
        """
        Calculate streaks for all teams using optimized pandas groupby operations
        """
        if self.games_df is None:
            raise ValueError("No games data loaded")

        start_time = datetime.now()

        # Use pandas groupby with rolling for efficient streak calculation
        def calculate_team_streaks(group: pd.DataFrame) -> pd.DataFrame:
            """Calculate streaks for a single team using vectorized operations"""

            # Calculate rolling wins (streaks)
            group['win_streak'] = self._calculate_rolling_streaks(group['won'])

            # Calculate recent form using rolling mean (last 5 games)
            group['recent_form_5'] = group['won'].rolling(window=5, min_periods=1).mean()

            # Calculate momentum using exponential weighted average
            group['momentum_ewm'] = group['won'].ewm(span=10, adjust=False).mean()

            # Calculate consistency (standard deviation of recent performance)
            group['consistency'] = group['won'].rolling(window=10, min_periods=3).std().fillna(0)

            # Calculate streak strength (weighted by margin of victory)
            group['streak_strength'] = self._calculate_streak_strength(group)

            return group

        # Apply function to each team group
        self.games_df = self.games_df.groupby('team_id', group_keys=False).apply(calculate_team_streaks)

        calculation_time = (datetime.now() - start_time).total_seconds() * 1000
        self.logger.info(f"⚡ All team streaks calculated in {calculation_time:.2f}ms")

    def _calculate_rolling_streaks(self, wins: pd.Series) -> pd.Series:
        """
        Calculate rolling streaks using vectorized pandas operations

        Args:
            wins: Series of win/loss results (1 for win, 0 for loss)

        Returns:
            Series of current streak values (positive for wins, negative for losses)
        """
        # Create streak direction (1 for win, -1 for loss)
        directions = np.where(wins == 1, 1, -1)

        # Use cumsum to identify streak changes
        streak_groups = (directions != directions.shift()).cumsum()

        # Calculate streak lengths using groupby cumsum
        streak_lengths = wins.groupby(streak_groups).cumsum() - wins.groupby(streak_groups).cummin() + 1

        # Apply direction to get positive/negative streak values
        return streak_lengths * directions

    def _calculate_streak_strength(self, group: pd.DataFrame) -> pd.Series:
        """
        Calculate streak strength based on margin of victory and consistency

        Args:
            group: Team games DataFrame

        Returns:
            Series of streak strength values
        """
        # Normalize margin of victory to [-1, 1] range
        max_margin = group['margin'].abs().max()
        if max_margin > 0:
            normalized_margin = (group['margin'] / max_margin).clip(-1, 1)
        else:
            normalized_margin = pd.Series(0, index=group.index)

        # Combine with win/loss result
        win_multiplier = group['won'] * 2 - 1  # 1 for win, -1 for loss

        return normalized_margin * win_multiplier

    def get_team_streak_profile(self, team_id: int, as_of_date: Optional[datetime] = None) -> TeamStreakProfile:
        """
        Get comprehensive streak profile for a specific team

        Args:
            team_id: Team identifier
            as_of_date: Calculate streaks as of this date (default: latest)

        Returns:
            TeamStreakProfile with comprehensive metrics
        """
        # Check cache first
        if self._is_cache_valid(team_id):
            self._calculation_stats['cache_hits'] += 1
            return self._streak_cache[team_id]

        start_time = datetime.now()

        if self.games_df is None:
            raise ValueError("No games data loaded")

        # Filter data for the specific team
        team_games = self.games_df[self.games_df['team_id'] == team_id].copy()

        if as_of_date:
            team_games = team_games[team_games['game_date'] <= as_of_date]

        if team_games.empty:
            # Return empty profile for team with no games
            return self._create_empty_profile(team_id)

        # Get latest streak information
        latest_game = team_games.iloc[-1]

        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(team_games, latest_game)

        # Create streak profile
        profile = TeamStreakProfile(
            team_id=team_id,
            team_name=latest_game.get('team_name', f'Team {team_id}'),
            current_metrics=metrics,
            historical_context=self._calculate_historical_context(team_games),
            streak_trends=self._calculate_streak_trends(team_games),
            performance_volatility=team_games['won'].std() if len(team_games) > 1 else 0.0
        )

        # Update cache
        self._update_cache(team_id, profile)

        calculation_time = (datetime.now() - start_time).total_seconds() * 1000
        self._calculation_stats['total_calculations'] += 1
        self._calculation_stats['avg_calculation_time_ms'] = (
            (self._calculation_stats['avg_calculation_time_ms'] * (self._calculation_stats['total_calculations'] - 1) + calculation_time) /
            self._calculation_stats['total_calculations']
        )

        return profile

    def _calculate_comprehensive_metrics(self, team_games: pd.DataFrame, latest_game: pd.Series) -> StreakMetrics:
        """Calculate comprehensive streak metrics using latest pandas features"""

        current_streak = int(latest_game['win_streak'])
        streak_type = StreakType.WINNING if current_streak > 0 else StreakType.LOSING if current_streak < 0 else StreakType.NEUTRAL

        # Calculate season extremes
        season_longest_win = int(team_games[team_games['win_streak'] > 0]['win_streak'].max()) if (team_games['win_streak'] > 0).any() else 0
        season_longest_loss = abs(int(team_games[team_games['win_streak'] < 0]['win_streak'].min())) if (team_games['win_streak'] < 0).any() else 0

        # Get recent performance metrics
        recent_form = float(latest_game['recent_form_5'])
        momentum_score = float(latest_game['momentum_ewm'])
        streak_strength = float(latest_game['streak_strength'])
        consistency_score = 1.0 - float(latest_game['consistency'])  # Invert so higher is better

        return StreakMetrics(
            current_streak=current_streak,
            streak_type=streak_type,
            season_longest_win=season_longest_win,
            season_longest_loss=season_longest_loss,
            recent_form=recent_form,
            momentum_score=momentum_score,
            streak_strength=streak_strength,
            consistency_score=consistency_score,
            last_updated=datetime.now()
        )

    def _calculate_historical_context(self, team_games: pd.DataFrame) -> Dict[str, float]:
        """Calculate historical context metrics using pandas aggregations"""

        if len(team_games) < 5:
            return {}

        context = {}

        # Overall win rate
        context['overall_win_rate'] = float(team_games['won'].mean())

        # Home vs away performance
        home_win_rate = team_games[team_games['is_home']]['won'].mean()
        away_win_rate = team_games[~team_games['is_home']]['won'].mean()
        context['home_advantage'] = float(home_win_rate - away_win_rate)

        # Recent performance trends (last 10 vs previous 10)
        if len(team_games) >= 20:
            recent_10 = team_games.tail(10)['won'].mean()
            previous_10 = team_games.iloc[-20:-10]['won'].mean()
            context['recent_trend'] = float(recent_10 - previous_10)

        # Margin statistics
        context['avg_margin'] = float(team_games['margin'].mean())
        context['margin_std'] = float(team_games['margin'].std())

        return context

    def _calculate_streak_trends(self, team_games: pd.DataFrame) -> pd.DataFrame:
        """Calculate streak trends using rolling windows"""

        if len(team_games) < 10:
            return pd.DataFrame()

        trends = team_games[['game_date', 'win_streak', 'recent_form_5', 'momentum_ewm']].copy()

        # Calculate trend indicators
        trends['streak_trend'] = trends['win_streak'].diff()
        trends['momentum_trend'] = trends['momentum_ewm'].diff()

        return trends.tail(20)  # Return last 20 games

    def _create_empty_profile(self, team_id: int) -> TeamStreakProfile:
        """Create empty profile for team with no games"""
        return TeamStreakProfile(
            team_id=team_id,
            team_name=f'Team {team_id}',
            current_metrics=StreakMetrics(),
            historical_context={},
            streak_trends=pd.DataFrame(),
            performance_volatility=0.0
        )

    def _is_cache_valid(self, team_id: int) -> bool:
        """Check if cached entry is still valid"""
        if team_id not in self._streak_cache or team_id not in self._cache_timestamps:
            return False

        cache_age = datetime.now() - self._cache_timestamps[team_id]
        return cache_age < self.cache_ttl

    def _update_cache(self, team_id: int, profile: TeamStreakProfile) -> None:
        """Update cache with new profile"""
        self._streak_cache[team_id] = profile
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
            self._streak_cache.pop(team_id, None)
            self._cache_timestamps.pop(team_id, None)

    def get_league_streak_leaderboard(self, limit: int = 10) -> List[TeamStreakProfile]:
        """
        Get league-wide streak leaderboard

        Args:
            limit: Maximum number of teams to return

        Returns:
            List of TeamStreakProfile sorted by current streak
        """
        if self.games_df is None:
            return []

        # Get all unique team IDs
        team_ids = self.games_df['team_id'].unique()

        # Calculate profiles for all teams
        profiles = [self.get_team_streak_profile(team_id) for team_id in team_ids]

        # Sort by absolute current streak (longest streaks first)
        profiles.sort(key=lambda p: abs(p.current_metrics.current_streak), reverse=True)

        return profiles[:limit]

    def get_performance_statistics(self) -> Dict[str, Union[int, float]]:
        """Get analyzer performance statistics"""
        cache_hit_rate = (self._calculation_stats['cache_hits'] /
                         max(self._calculation_stats['total_calculations'], 1)) * 100

        return {
            **self._calculation_stats,
            'cache_hit_rate_percent': round(cache_hit_rate, 2),
            'cached_teams': len(self._streak_cache)
        }

    def clear_cache(self) -> None:
        """Clear all cached data"""
        self._streak_cache.clear()
        self._cache_timestamps.clear()
        self.logger.info("🗑️ Cache cleared")

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create realistic sample data for testing
    np.random.seed(42)  # For reproducible results

    # Create games data in proper format
    games_data = []
    teams = {
        1: 'Lakers',
        2: 'Celtics',
        3: 'Warriors'
    }

    dates = pd.date_range('2024-01-01', periods=15, freq='D')
    for i, date in enumerate(dates):
        if i % 3 == 0:
            home, away = 1, 2
        elif i % 3 == 1:
            home, away = 2, 3
        else:
            home, away = 3, 1

        # Create some predictable winning patterns
        if i < 5:  # Lakers win early games
            home_score, away_score = (120, 105) if home == 1 else (95, 110)
        elif i < 10:  # Celtics win middle games
            home_score, away_score = (115, 100) if home == 2 else (90, 105)
        else:  # Warriors win late games
            home_score, away_score = (125, 110) if home == 3 else (85, 100)

        games_data.append({
            'game_id': f'00{i+1:010d}',  # NBA game ID format
            'game_date': date,
            'home_team': home,
            'away_team': away,
            'home_score': home_score,
            'away_score': away_score,
            'season': 2024
        })

    sample_games = pd.DataFrame(games_data)

    # Create analyzer
    analyzer = RealTimeStreakAnalyzer()

    # Load data
    analyzer.load_games_data(sample_games)

    # Test team profile retrieval
    profile = analyzer.get_team_streak_profile(1)
    print(f"Team 1 Streak Profile: {profile.current_metrics}")

    # Get leaderboard
    leaderboard = analyzer.get_league_streak_leaderboard()
    print(f"League Leaderboard: {len(leaderboard)} teams")

    # Performance stats
    stats = analyzer.get_performance_statistics()
    print(f"Performance Stats: {stats}")