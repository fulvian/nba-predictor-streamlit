#!/usr/bin/env python3
"""
🚀 ADVANCED MOMENTUM ENGINE - DevStream SuperPowered Implementation

Next-generation momentum calculation system using pandas EWMA and rolling averages
for NBA predictive analytics with Context Set patterns.

Author: NBA Predictive Analytics System
Task ID: 1.2.2 - Momentum calculations using rolling averages
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

class MomentumCalculationMethod(Enum):
    """Enumeration of momentum calculation methods"""
    EWMA = 'ewma'                    # Exponential Weighted Moving Average
    SIMPLE_ROLLING = 'simple_rolling'      # Simple rolling average
    WEIGHTED_ROLLING = 'weighted_rolling'    # Weighted rolling average
    HYBRID_EWM = 'hybrid_ewm'            # Hybrid EWMA with multiple spans
    ADAPTIVE_EWM = 'adaptive_ewm'          # Adaptive EWMA based on volatility

@dataclass
class MomentumMetrics:
    """Data class for comprehensive momentum metrics"""
    current_momentum: float = 0.0
    momentum_trend: float = 0.0
    momentum_strength: float = 0.0
    momentum_consistency: float = 0.0
    momentum_velocity: float = 0.0
    volatility_adjusted_momentum: float = 0.0
    momentum_signal_strength: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class TeamMomentumProfile:
    """Comprehensive team momentum profile with advanced analytics"""
    team_id: int
    team_name: str
    current_metrics: MomentumMetrics
    historical_momentum: pd.DataFrame = field(default_factory=pd.DataFrame)
    momentum_regimes: Dict[str, int] = field(default_factory=dict)
    performance_cycles: Dict[str, float] = field(default_factory=dict)
    momentum_predictions: Dict[str, float] = field(default_factory=dict)
    momentum_volatility: float = 0.0

class AdvancedMomentumEngine:
    """
    🚀 SuperPowered Advanced Momentum Engine

    Features:
    - Multiple EWMA configurations for different time horizons
    - Adaptive momentum based on volatility analysis
    - Hybrid momentum calculations combining multiple methods
    - Real-time momentum prediction capabilities
    - Performance tracking and monitoring
    - DevStream compliant architecture
    """

    def __init__(self, cache_ttl_minutes: int = 15):
        """
        Initialize the advanced momentum engine

        Args:
            cache_ttl_minutes: Cache time-to-live in minutes
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)

        # Cache for momentum calculations
        self._momentum_cache: Dict[int, TeamMomentumProfile] = {}
        self._cache_timestamps: Dict[int, datetime] = {}

        # Data storage
        self.games_df: Optional[pd.DataFrame] = None
        self.teams_df: Optional[pd.DataFrame] = None

        # Momentum calculation parameters
        self.ewm_params = {
            'short_span': 5,      # Short-term momentum (5 games)
            'medium_span': 10,    # Medium-term momentum (10 games)
            'long_span': 20,       # Long-term momentum (20 games)
            'season_span': 82     # Season-long momentum (full season)
        }

        # Performance metrics
        self._calculation_stats = {
            'total_calculations': 0,
            'cache_hits': 0,
            'avg_calculation_time_ms': 0.0,
            'method_usage': {
                'ewma': 0,
                'simple_rolling': 0,
                'weighted_rolling': 0,
                'hybrid_ewm': 0,
                'adaptive_ewm': 0
            }
        }

        self.logger.info("🚀 AdvancedMomentumEngine initialized with SuperPowered features")

    def load_games_data(self, games_df: pd.DataFrame) -> None:
        """
        Load and preprocess games data for momentum analysis

        Args:
            games_df: DataFrame containing NBA games data
        """
        start_time = datetime.now()

        # Validate required columns
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

        # Calculate all momentum indicators using optimized pandas operations
        self._calculate_all_momentum_indicators()

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
        home_games['point_differential'] = home_games['team_score'] - home_games['opponent_score']

        away_games = df[['game_id', 'game_date', 'away_team', 'away_score', 'home_score', 'season']].copy()
        away_games['team_id'] = away_games['away_team']
        away_games['opponent_id'] = away_games['home_team']
        away_games['team_score'] = away_games['away_score']
        away_games['opponent_score'] = away_games['home_score']
        away_games['is_home'] = False
        away_games['point_differential'] = away_games['team_score'] - away_games['opponent_score']

        # Combine home and away games
        team_games = pd.concat([home_games, away_games], ignore_index=True)

        # Calculate win/loss result using vectorized operations
        team_games['won'] = (team_games['team_score'] > team_games['opponent_score']).astype(int)
        team_games['margin'] = team_games['point_differential']

        # Sort by team and date for proper momentum calculation
        team_games = team_games.sort_values(['team_id', 'game_date']).reset_index(drop=True)

        # Add game number for each team
        team_games['team_game_number'] = team_games.groupby('team_id').cumcount() + 1

        # Calculate advanced metrics for momentum
        team_games['strength_of_victory'] = np.where(
            team_games['won'] == 1,
            np.clip(team_games['margin'] / 20, 0, 1),  # Normalize margin (20 point win = max strength)
            np.clip(-team_games['margin'] / 20, -1, 0)  # Normalize loss (20 point loss = min strength)
        )

        self.logger.info(f"📊 Preprocessed {len(team_games)} team-game records")
        return team_games

    def _calculate_all_momentum_indicators(self) -> None:
        """
        Calculate momentum indicators for all teams using optimized pandas operations
        """
        if self.games_df is None:
            raise ValueError("No games data loaded")

        start_time = datetime.now()

        # Use pandas groupby with EWM for efficient momentum calculation
        def calculate_team_momentum(group: pd.DataFrame) -> pd.DataFrame:
            """Calculate comprehensive momentum for a single team using vectorized operations"""

            # Calculate various EWMA-based momentum indicators
            group['momentum_ewm_short'] = group['won'].ewm(
                span=self.ewm_params['short_span'],
                adjust=True
            ).mean()

            group['momentum_ewm_medium'] = group['won'].ewm(
                span=self.ewm_params['medium_span'],
                adjust=True
            ).mean()

            group['momentum_ewm_long'] = group['won'].ewm(
                span=self.ewm_params['long_span'],
                adjust=True
            ).mean()

            group['momentum_ewm_season'] = group['won'].ewm(
                span=min(self.ewm_params['season_span'], len(group)),
                adjust=True
            ).mean()

            # Calculate weighted momentum (emphasizes recent performance)
            weights = np.exp(np.linspace(-2, 0, min(10, len(group))))  # Exponential decay weights
            weights = weights / weights.sum()  # Normalize weights
            group['momentum_weighted'] = (group['won'] * weights).rolling(
                window=min(10, len(group)),
                min_periods=1
            ).sum()

            # Calculate momentum volatility (how volatile the momentum is)
            group['momentum_volatility'] = group['momentum_ewm_medium'].rolling(window=10, min_periods=3).std().fillna(0)

            # Calculate momentum velocity (rate of change)
            group['momentum_velocity'] = group['momentum_ewm_short'].diff()

            # Calculate momentum acceleration (rate of change of velocity)
            group['momentum_acceleration'] = group['momentum_velocity'].diff()

            # Calculate momentum signal strength (consistency of positive/negative momentum)
            group['momentum_signal'] = np.where(
                group['momentum_ewm_short'] > 0.6,
                group['momentum_ewm_short'],
                np.where(group['momentum_ewm_short'] < 0.4,
                    -group['momentum_ewm_short'], 0)
            )

            # Calculate hybrid momentum (combines multiple time horizons)
            group['momentum_hybrid'] = (
                0.4 * group['momentum_ewm_short'] +
                0.3 * group['momentum_ewm_medium'] +
                0.2 * group['momentum_ewm_long'] +
                0.1 * group['momentum_ewm_season']
            )

            # Calculate volatility-adjusted momentum
            volatility_adjustment = 1.0 / (1.0 + group['momentum_volatility'])
            group['momentum_volatility_adjusted'] = group['momentum_hybrid'] * volatility_adjustment

            # Calculate momentum strength based on winning margin
            group['momentum_strength'] = (
                group['momentum_ewm_short'] *
                (1 + group['strength_of_victory'].rolling(window=5, min_periods=1).mean())
            )

            # Calculate momentum consistency (how consistent the momentum is)
            group['momentum_consistency'] = 1.0 - group['momentum_volatility'].rolling(
                window=20,
                min_periods=5
            ).mean().fillna(0)

            return group

        # Apply function to each team group
        self.games_df = self.games_df.groupby('team_id', group_keys=False).apply(calculate_team_momentum)

        calculation_time = (datetime.now() - start_time).total_seconds() * 1000
        self.logger.info(f"⚡ All momentum indicators calculated in {calculation_time:.2f}ms")

    def get_team_momentum_profile(self, team_id: int, as_of_date: Optional[datetime] = None) -> TeamMomentumProfile:
        """
        Get comprehensive momentum profile for a specific team

        Args:
            team_id: Team identifier
            as_of_date: Calculate momentum as of this date (default: latest)

        Returns:
            TeamMomentumProfile with comprehensive metrics
        """
        # Check cache first
        if self._is_cache_valid(team_id):
            self._calculation_stats['cache_hits'] += 1
            return self._momentum_cache[team_id]

        start_time = datetime.now()

        if self.games_df is None:
            raise ValueError("No games data loaded")

        # Filter data for the specific team
        team_games = self.games_df[self.games_df['team_id'] == team_id].copy()

        if as_of_date:
            team_games = team_games[team_games['game_date'] <= as_of_date]

        if team_games.empty:
            # Return empty profile for team with no games
            return self._create_empty_momentum_profile(team_id)

        # Get latest momentum information
        latest_game = team_games.iloc[-1]

        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_momentum_metrics(team_games, latest_game)

        # Create momentum profile
        profile = TeamMomentumProfile(
            team_id=team_id,
            team_name=latest_game.get('team_name', f'Team {team_id}'),
            current_metrics=metrics,
            historical_momentum=self._calculate_historical_momentum_trends(team_games),
            momentum_regimes=self._identify_momentum_regimes(team_games),
            performance_cycles=self._analyze_performance_cycles(team_games),
            momentum_predictions=self._generate_momentum_predictions(team_games),
            momentum_volatility=team_games['momentum_volatility'].iloc[-5:].mean() if len(team_games) >= 5 else 0.0
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

    def _calculate_comprehensive_momentum_metrics(self, team_games: pd.DataFrame, latest_game: pd.Series) -> MomentumMetrics:
        """Calculate comprehensive momentum metrics using latest pandas features"""

        current_momentum = float(latest_game['momentum_hybrid'])
        momentum_trend = float(latest_game['momentum_velocity'] if not pd.isna(latest_game['momentum_velocity']) else 0.0)
        momentum_strength = float(latest_game['momentum_strength'])
        momentum_consistency = float(latest_game['momentum_consistency'])
        momentum_velocity = float(latest_game['momentum_velocity'] if not pd.isna(latest_game['momentum_velocity']) else 0.0)
        volatility_adjusted_momentum = float(latest_game['momentum_volatility_adjusted'])
        momentum_signal_strength = float(latest_game['momentum_signal'])

        return MomentumMetrics(
            current_momentum=current_momentum,
            momentum_trend=momentum_trend,
            momentum_strength=momentum_strength,
            momentum_consistency=momentum_consistency,
            momentum_velocity=momentum_velocity,
            volatility_adjusted_momentum=volatility_adjusted_momentum,
            momentum_signal_strength=momentum_signal_strength,
            last_updated=datetime.now()
        )

    def _calculate_historical_momentum_trends(self, team_games: pd.DataFrame) -> pd.DataFrame:
        """Calculate historical momentum trends using rolling windows"""

        if len(team_games) < 10:
            return pd.DataFrame()

        trends = team_games[['game_date', 'momentum_ewm_short', 'momentum_ewm_medium', 'momentum_ewm_long']].copy()

        # Calculate trend indicators
        trends['short_momentum_trend'] = trends['momentum_ewm_short'].diff()
        trends['medium_momentum_trend'] = trends['momentum_ewm_medium'].diff()
        trends['long_momentum_trend'] = trends['momentum_ewm_long'].diff()

        # Calculate momentum crossovers (signal changes)
        trends['momentum_crossover_signal'] = (
            (trends['momentum_ewm_short'] > 0.5) !=
            (trends['momentum_ewm_short'].shift(1) > 0.5)
        ).astype(int)

        return trends.tail(20)  # Return last 20 games

    def _identify_momentum_regimes(self, team_games: pd.DataFrame) -> Dict[str, int]:
        """Identify different momentum regimes based on statistical analysis"""

        if len(team_games) < 20:
            return {}

        regimes = {}

        # Analyze momentum levels
        current_momentum = team_games['momentum_ewm_medium'].iloc[-1]

        if current_momentum > 0.7:
            regimes['current'] = 'strong_bullish'
        elif current_momentum > 0.55:
            regimes['current'] = 'moderate_bullish'
        elif current_momentum > 0.45:
            regimes['current'] = 'neutral'
        elif current_momentum > 0.3:
            regimes['current'] = 'moderate_bearish'
        else:
            regimes['current'] = 'strong_bearish'

        # Count regime transitions in last 20 games
        momentum_changes = team_games['momentum_ewm_medium'].tail(20).diff().dropna()
        regimes['transition_frequency'] = len(momentum_changes)

        return regimes

    def _analyze_performance_cycles(self, team_games: pd.DataFrame) -> Dict[str, float]:
        """Analyze performance cycles and patterns"""

        if len(team_games) < 10:
            return {}

        cycles = {}

        # Analyze winning cycles
        wins = team_games['won'].tail(20)
        win_cycles = self._identify_cycles(wins, threshold=0.6)
        cycles['win_cycle_length_avg'] = np.mean([len(cycle) for cycle in win_cycles]) if win_cycles else 0

        # Analyze momentum cycles
        momentum = team_games['momentum_ewm_medium'].tail(20)
        momentum_cycles = self._identify_cycles(momentum, threshold=0.1)
        cycles['momentum_cycle_length_avg'] = np.mean([len(cycle) for cycle in momentum_cycles]) if momentum_cycles else 0

        # Calculate cycle strength
        cycles['cycle_consistency'] = 1.0 - (momentum.rolling(window=10, min_periods=5).std().mean() if len(momentum) >= 10 else 0)

        return cycles

    def _identify_cycles(self, series: pd.Series, threshold: float = 0.5) -> List[List[int]]:
        """Identify cycles in a time series using threshold-based approach"""

        cycles = []
        current_cycle = []

        for i, value in enumerate(series):
            if value > threshold:
                current_cycle.append(i)
            else:
                if current_cycle:
                    cycles.append(current_cycle)
                    current_cycle = []

        if current_cycle:  # Add the last cycle if it ends with positive values
            cycles.append(current_cycle)

        return cycles

    def _generate_momentum_predictions(self, team_games: pd.DataFrame) -> Dict[str, float]:
        """Generate momentum-based predictions using EWMA extrapolation"""

        if len(team_games) < 10:
            return {}

        predictions = {}

        # Use EWMA to predict next game momentum
        recent_momentum = team_games['momentum_ewm_medium'].tail(5)
        predicted_momentum = recent_momentum.ewm(span=3, adjust=True).iloc[-1] if len(recent_momentum) > 0 else 0.5

        predictions['next_game_momentum'] = predicted_momentum
        predictions['next_game_win_probability'] = self._momentum_to_win_probability(predicted_momentum)

        # Calculate confidence based on momentum consistency
        momentum_consistency = team_games['momentum_consistency'].tail(10).mean() if len(team_games) >= 10 else 0.5
        predictions['prediction_confidence'] = min(0.95, max(0.05, momentum_consistency))

        return predictions

    def _momentum_to_win_probability(self, momentum: float) -> float:
        """Convert momentum score to win probability using logistic transformation"""
        # Logistic function: probability = 1 / (1 + exp(-k * momentum))
        # Adjust k to map typical momentum range [-1, 1] to [0.2, 0.8]
        k = 2.5  # Steepness parameter
        return 1.0 / (1.0 + np.exp(-k * momentum))

    def _create_empty_momentum_profile(self, team_id: int) -> TeamMomentumProfile:
        """Create empty momentum profile for team with no games"""
        return TeamMomentumProfile(
            team_id=team_id,
            team_name=f'Team {team_id}',
            current_metrics=MomentumMetrics(),
            historical_momentum=pd.DataFrame(),
            momentum_regimes={},
            performance_cycles={},
            momentum_predictions={},
            momentum_volatility=0.0
        )

    def _is_cache_valid(self, team_id: int) -> bool:
        """Check if cached entry is still valid"""
        if team_id not in self._momentum_cache or team_id not in self._cache_timestamps:
            return False

        cache_age = datetime.now() - self._cache_timestamps[team_id]
        return cache_age < self.cache_ttl

    def _update_cache(self, team_id: int, profile: TeamMomentumProfile) -> None:
        """Update cache with new profile"""
        self._momentum_cache[team_id] = profile
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
            self._momentum_cache.pop(team_id, None)
            self._cache_timestamps.pop(team_id, None)

    def get_league_momentum_leaderboard(self, limit: int = 10) -> List[TeamMomentumProfile]:
        """
        Get league-wide momentum leaderboard

        Args:
            limit: Maximum number of teams to return

        Returns:
            List of TeamMomentumProfile sorted by current momentum
        """
        if self.games_df is None:
            return []

        # Get all unique team IDs
        team_ids = self.games_df['team_id'].unique()

        # Calculate profiles for all teams
        profiles = [self.get_team_momentum_profile(team_id) for team_id in team_ids]

        # Sort by absolute current momentum (strongest momentum first)
        profiles.sort(key=lambda p: abs(p.current_metrics.current_momentum), reverse=True)

        return profiles[:limit]

    def get_performance_statistics(self) -> Dict[str, Union[int, float]]:
        """Get engine performance statistics"""
        cache_hit_rate = (self._calculation_stats['cache_hits'] /
                         max(self._calculation_stats['total_calculations'], 1)) * 100

        return {
            **self._calculation_stats,
            'cache_hit_rate_percent': round(cache_hit_rate, 2),
            'cached_teams': len(self._momentum_cache)
        }

    def clear_cache(self) -> None:
        """Clear all cached data"""
        self._momentum_cache.clear()
        self._cache_timestamps.clear()
        self.logger.info("🗑️ Cache cleared")

    def calculate_momentum_correlations(self, window_days: int = 30) -> Dict[str, float]:
        """
        Calculate momentum correlations between teams using rolling windows

        Args:
            window_days: Time window in days for correlation analysis

        Returns:
            Dictionary of correlation coefficients
        """
        if self.games_df is None:
            return {}

        # Get recent games within time window
        cutoff_date = datetime.now() - timedelta(days=window_days)
        recent_games = self.games_df[self.games_df['game_date'] >= cutoff_date]

        if recent_games.empty():
            return {}

        # Create team momentum matrix for correlation analysis
        team_momentum_pivot = recent_games.pivot_table(
            index='game_date',
            columns='team_id',
            values='momentum_ewm_medium'
        )

        # Calculate correlation matrix
        correlation_matrix = team_momentum_pivot.corr()

        # Extract significant correlations (|correlation| > 0.7)
        significant_correlations = {}
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.7:
                    key = f"{correlation_matrix.columns[i]}_vs_{correlation_matrix.columns[j]}"
                    significant_correlations[key] = correlation_matrix.iloc[i, j]

        return significant_correlations

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create sample data for testing
    np.random.seed(42)  # For reproducible results

    # Create games data with realistic momentum patterns
    games_data = []
    teams = {
        1: 'Lakers',
        2: 'Celtics',
        3: 'Warriors',
        4: 'Heat',
        5: 'Spurs'
    }

    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    for i, date in enumerate(dates):
        # Create realistic momentum patterns
        team_pairings = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]
        home, away = team_pairings[i % 5]

        # Create realistic scoring patterns with momentum
        if i < 15:  # Lakers strong period
            home_score, away_score = (125, 105) if home == 1 else (95, 115)
        elif i < 30:  # Celtics strong period
            home_score, away_score = (120, 100) if home == 2 else (90, 110)
        elif i < 40: # Warriors strong period
            home_score, away_score = (130, 110) if home == 3 else (80, 100)
        else:  # More competitive games
            home_score, away_score = np.random.randint(85, 130, 2)

        games_data.append({
            'game_id': f'00{i+1:010d}',
            'game_date': date,
            'home_team': home,
            'away_team': away,
            'home_score': home_score,
            'away_score': away_score,
            'season': 2024,
            'team_name': teams[home]
        })

    sample_games = pd.DataFrame(games_data)

    # Create engine
    engine = AdvancedMomentumEngine()

    # Load data
    engine.load_games_data(sample_games)

    # Test team profile retrieval
    profile = engine.get_team_momentum_profile(1)
    print(f"Team 1 Momentum Profile: {profile.current_metrics}")

    # Get leaderboard
    leaderboard = engine.get_league_momentum_leaderboard()
    print(f"League Leaderboard: {len(leaderboard)} teams")

    # Performance stats
    stats = engine.get_performance_statistics()
    print(f"Performance Stats: {stats}")

    # Test momentum correlations
    correlations = engine.calculate_momentum_correlations(window_days=20)
    print(f"Momentum Correlations: {len(correlations)} significant correlations found")