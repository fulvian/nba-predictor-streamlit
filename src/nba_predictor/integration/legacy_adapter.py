#!/usr/bin/env python3
"""
🏀 NBA Legacy System Integration Adapter
Context7-compliant adapter pattern for integrating deprecated NBA prediction system with UnifiedDataStore.

This module implements:
- Adapter pattern for legacy data provider integration
- Dependency injection following Context7 best practices
- Interface segregation for modular component structure
- Cache optimization for performance
"""

from typing import Optional, Dict, List, Tuple
import pandas as pd
from datetime import datetime, date, timedelta
import logging

# Context7-compliant imports
from nba_predictor.core.data_store import UnifiedDataStore
from nba_predictor.core.roster_injury_schemas import TeamRoster, InjuryInfo


logger = logging.getLogger(__name__)


class UnifiedDataStoreAdapter:
    """
    Adapter pattern per integrare NBAHybridDataProvider con UnifiedDataStore.

    Context7-compliant implementation following:
    - Adapter pattern for system integration
    - Dependency injection principles
    - Interface segregation
    - Single responsibility principle
    """

    def __init__(self, unified_store: UnifiedDataStore):
        """
        Initialize adapter with UnifiedDataStore dependency.

        Args:
            unified_store: UnifiedDataStore instance for data access
        """
        self.unified_store = unified_store
        self._team_cache = {}
        self._game_cache = {}
        self._player_cache = {}

        logger.info("UnifiedDataStoreAdapter initialized with dependency injection")

    def _find_team_by_name(self, team_name: str) -> Optional[Dict]:
        """
        Find team by name - method missing from original system.

        Context7-compliant: Implements caching strategy for performance optimization.

        Args:
            team_name: Name of the team to find

        Returns:
            Team information dict or None if not found
        """
        if team_name not in self._team_cache:
            try:
                # Use hardcoded team data as fallback - NBA teams for 2025-26 season
                nba_teams = {
                    "Utah Jazz": {"team_id": 1610612762, "team_name": "Utah Jazz", "team_abbreviation": "UTA"},
                    "Golden State Warriors": {"team_id": 1610612747, "team_name": "Golden State Warriors", "team_abbreviation": "GSW"},
                    "Los Angeles Lakers": {"team_id": 1610612744, "team_name": "Los Angeles Lakers", "team_abbreviation": "LAL"},
                    "Boston Celtics": {"team_id": 1610612738, "team_name": "Boston Celtics", "team_abbreviation": "BOS"},
                    "Brooklyn Nets": {"team_id": 1610612751, "team_name": "Brooklyn Nets", "team_abbreviation": "BKN"},
                    "New York Knicks": {"team_id": 1610612752, "team_name": "New York Knicks", "team_abbreviation": "NYK"},
                    "Philadelphia 76ers": {"team_id": 1610612755, "team_name": "Philadelphia 76ers", "team_abbreviation": "PHI"},
                    "Toronto Raptors": {"team_id": 1610612761, "team_name": "Toronto Raptors", "team_abbreviation": "TOR"},
                    "Chicago Bulls": {"team_id": 1610612741, "team_name": "Chicago Bulls", "team_abbreviation": "CHI"},
                    "Cleveland Cavaliers": {"team_id": 1610612739, "team_name": "Cleveland Cavaliers", "team_abbreviation": "CLE"},
                    "Detroit Pistons": {"team_id": 1610612765, "team_name": "Detroit Pistons", "team_abbreviation": "DET"},
                    "Indiana Pacers": {"team_id": 1610612754, "team_name": "Indiana Pacers", "team_abbreviation": "IND"},
                    "Milwaukee Bucks": {"team_id": 1610612749, "team_name": "Milwaukee Bucks", "team_abbreviation": "MIL"},
                    "Atlanta Hawks": {"team_id": 1610612737, "team_name": "Atlanta Hawks", "team_abbreviation": "ATL"},
                    "Charlotte Hornets": {"team_id": 1610612766, "team_name": "Charlotte Hornets", "team_abbreviation": "CHA"},
                    "Miami Heat": {"team_id": 1610612748, "team_name": "Miami Heat", "team_abbreviation": "MIA"},
                    "Orlando Magic": {"team_id": 1610612753, "team_name": "Orlando Magic", "team_abbreviation": "ORL"},
                    "Washington Wizards": {"team_id": 1610612764, "team_name": "Washington Wizards", "team_abbreviation": "WAS"},
                    "Denver Nuggets": {"team_id": 1610612743, "team_name": "Denver Nuggets", "team_abbreviation": "DEN"},
                    "Minnesota Timberwolves": {"team_id": 1610612750, "team_name": "Minnesota Timberwolves", "team_abbreviation": "MIN"},
                    "Oklahoma City Thunder": {"team_id": 1610612760, "team_name": "Oklahoma City Thunder", "team_abbreviation": "OKC"},
                    "Portland Trail Blazers": {"team_id": 1610612757, "team_name": "Portland Trail Blazers", "team_abbreviation": "POR"},
                    "Seattle SuperSonics": {"team_id": 1610612758, "team_name": "Seattle SuperSonics", "team_abbreviation": "SEA"},
                    "Los Angeles Clippers": {"team_id": 1610612746, "team_name": "Los Angeles Clippers", "team_abbreviation": "LAC"},
                    "Phoenix Suns": {"team_id": 1610612756, "team_name": "Phoenix Suns", "team_abbreviation": "PHX"},
                    "Sacramento Kings": {"team_id": 1610612758, "team_name": "Sacramento Kings", "team_abbreviation": "SAC"},
                    "Dallas Mavericks": {"team_id": 1610612742, "team_name": "Dallas Mavericks", "team_abbreviation": "DAL"},
                    "Houston Rockets": {"team_id": 1610612745, "team_name": "Houston Rockets", "team_abbreviation": "HOU"},
                    "Memphis Grizzlies": {"team_id": 1610612763, "team_name": "Memphis Grizzlies", "team_abbreviation": "MEM"},
                    "New Orleans Pelicans": {"team_id": 1610612740, "team_name": "New Orleans Pelicans", "team_abbreviation": "NOP"},
                    "San Antonio Spurs": {"team_id": 1610612759, "team_name": "San Antonio Spurs", "team_abbreviation": "SAS"}
                }

                # Direct match first
                if team_name in nba_teams:
                    self._team_cache[team_name] = nba_teams[team_name]
                    logger.debug(f"Team '{team_name}' found directly and cached")
                else:
                    # Fuzzy matching for team names
                    found_team = None
                    for team_key, team_data in nba_teams.items():
                        if (team_name.lower() in team_key.lower() or
                            team_key.lower() in team_name.lower() or
                            team_name.lower() in team_data['team_abbreviation'].lower()):
                            found_team = team_data
                            break

                    if found_team:
                        self._team_cache[team_name] = found_team
                        logger.debug(f"Team '{team_name}' found via fuzzy matching and cached")
                    else:
                        logger.warning(f"Team '{team_name}' not found in team registry")
                        self._team_cache[team_name] = None

            except Exception as e:
                logger.error(f"Error finding team '{team_name}': {e}")
                self._team_cache[team_name] = None

        return self._team_cache.get(team_name)

    def get_team_id(self, team_name: str) -> Optional[int]:
        """
        Get team ID by name.

        Args:
            team_name: Team name

        Returns:
            Team ID or None if not found
        """
        team_info = self._find_team_by_name(team_name)
        return team_info.get('team_id') if team_info else None

    def get_historical_games(self, team1: str, team2: str, season: str, limit: int = 100) -> pd.DataFrame:
        """
        Get historical games between two teams.

        Context7-compliant: Bridge between legacy interface and UnifiedDataStore.

        Args:
            team1: First team name
            team2: Second team name
            season: NBA season
            limit: Maximum number of games to return

        Returns:
            DataFrame with historical games
        """
        try:
            # Get team IDs
            team1_id = self.get_team_id(team1)
            team2_id = self.get_team_id(team2)

            if not team1_id or not team2_id:
                logger.error(f"Could not find team IDs for {team1} and {team2}")
                return pd.DataFrame()

            # Create cache key
            cache_key = f"{team1_id}_{team2_id}_{season}_{limit}"
            if cache_key in self._game_cache:
                logger.debug(f"Returning cached games for {team1} vs {team2}")
                return self._game_cache[cache_key]

            # Get games from UnifiedDataStore
            games_df = self.unified_store.get_games_for_teams(
                team_ids=[team1_id, team2_id],
                seasons=[season],
                limit=limit
            )

            if games_df is not None and not games_df.empty:
                # Filter for games between the two teams
                games_between = games_df[
                    ((games_df['home_team_id'] == team1_id) & (games_df['away_team_id'] == team2_id)) |
                    ((games_df['home_team_id'] == team2_id) & (games_df['away_team_id'] == team1_id))
                ]

                # Cache the result
                self._game_cache[cache_key] = games_between
                logger.info(f"Found {len(games_between)} historical games between {team1} and {team2}")
                return games_between
            else:
                logger.warning(f"No historical games found between {team1} and {team2}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error getting historical games: {e}")
            return pd.DataFrame()

    def get_team_recent_games(self, team_name: str, days: int = 10) -> pd.DataFrame:
        """
        Get recent games for a team.

        Args:
            team_name: Team name
            days: Number of days to look back

        Returns:
            DataFrame with recent games
        """
        try:
            team_id = self.get_team_id(team_name)
            if not team_id:
                return pd.DataFrame()

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            games_df = self.unified_store.get_games_for_teams(
                team_ids=[team_id],
                start_date=start_date,
                end_date=end_date
            )

            if games_df is not None and not games_df.empty:
                # Sort by date descending
                games_df = games_df.sort_values('game_date', ascending=False)
                logger.info(f"Found {len(games_df)} recent games for {team_name}")

            return games_df or pd.DataFrame()

        except Exception as e:
            logger.error(f"Error getting recent games for {team_name}: {e}")
            return pd.DataFrame()

    def get_team_roster(self, team_name: str, season: str) -> Optional[TeamRoster]:
        """
        Get team roster information.

        Args:
            team_name: Team name
            season: NBA season

        Returns:
            TeamRoster object or None
        """
        try:
            team_id = self.get_team_id(team_name)
            if not team_id:
                return None

            roster_data = self.unified_store.get_team_roster(team_id, season)
            if roster_data:
                return TeamRoster.parse_obj(roster_data)

            return None

        except Exception as e:
            logger.error(f"Error getting roster for {team_name}: {e}")
            return None

    def get_team_injuries(self, team_name: str) -> List[InjuryInfo]:
        """
        Get injury information for a team.

        Args:
            team_name: Team name

        Returns:
            List of InjuryInfo objects
        """
        try:
            team_id = self.get_team_id(team_name)
            if not team_id:
                return []

            injuries_data = self.unified_store.get_team_injuries(team_id)
            if injuries_data:
                return [InjuryInfo.parse_obj(injury) for injury in injuries_data]

            return []

        except Exception as e:
            logger.error(f"Error getting injuries for {team_name}: {e}")
            return []

    def get_player_stats(self, team_name: str, season: str, recent_games: int = 10) -> Dict[str, Dict]:
        """
        Get player statistics for a team.

        Args:
            team_name: Team name
            season: NBA season
            recent_games: Number of recent games to consider

        Returns:
            Dictionary mapping player names to player stats dictionaries
        """
        try:
            team_id = self.get_team_id(team_name)
            if not team_id:
                return {}

            player_stats = self.unified_store.get_player_stats(
                team_id=team_id,
                season=season,
                recent_games=recent_games
            )

            if player_stats:
                return {
                    stats.get('player_name', f'Player_{stats.get("player_id", "unknown")}'): stats
                    for stats in player_stats
                }

            return {}

        except Exception as e:
            logger.error(f"Error getting player stats for {team_name}: {e}")
            return {}

    def get_team_momentum_metrics(self, team_name: str, days: int = 10) -> Dict[str, float]:
        """
        Calculate team momentum metrics.

        Context7-compliant: Implements momentum calculation using recent performance.

        Args:
            team_name: Team name
            days: Number of days to analyze

        Returns:
            Dictionary with momentum metrics
        """
        try:
            recent_games = self.get_team_recent_games(team_name, days)

            if recent_games.empty:
                return {
                    'win_rate': 0.0,
                    'avg_points_scored': 0.0,
                    'avg_points_allowed': 0.0,
                    'momentum_score': 0.0,
                    'games_analyzed': 0
                }

            # Calculate team performance metrics
            team_id = self.get_team_id(team_name)
            wins = 0
            points_scored = []
            points_allowed = []

            for _, game in recent_games.iterrows():
                if game['home_team_id'] == team_id:
                    # Team is home team
                    is_win = game['home_score'] > game['away_score']
                    points_scored.append(game['home_score'])
                    points_allowed.append(game['away_score'])
                else:
                    # Team is away team
                    is_win = game['away_score'] > game['home_score']
                    points_scored.append(game['away_score'])
                    points_allowed.append(game['home_score'])

                if is_win:
                    wins += 1

            total_games = len(recent_games)
            win_rate = wins / total_games if total_games > 0 else 0.0
            avg_points_scored = sum(points_scored) / len(points_scored) if points_scored else 0.0
            avg_points_allowed = sum(points_allowed) / len(points_allowed) if points_allowed else 0.0

            # Calculate momentum score (weighted recent performance)
            momentum_score = win_rate * 0.6 + (avg_points_scored - avg_points_allowed) * 0.4

            return {
                'win_rate': round(win_rate, 3),
                'avg_points_scored': round(avg_points_scored, 1),
                'avg_points_allowed': round(avg_points_allowed, 1),
                'momentum_score': round(momentum_score, 3),
                'games_analyzed': total_games
            }

        except Exception as e:
            logger.error(f"Error calculating momentum metrics for {team_name}: {e}")
            return {
                'win_rate': 0.0,
                'avg_points_scored': 0.0,
                'avg_points_allowed': 0.0,
                'momentum_score': 0.0,
                'games_analyzed': 0
            }

    def clear_cache(self):
        """Clear all cached data"""
        self._team_cache.clear()
        self._game_cache.clear()
        self._player_cache.clear()
        logger.info("Adapter cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'team_cache_size': len(self._team_cache),
            'game_cache_size': len(self._game_cache),
            'player_cache_size': len(self._player_cache)
        }


class LegacySystemBridge:
    """
    Bridge class for integrating legacy NBA prediction system with UnifiedDataStore.

    Context7-compliant: Implements bridge pattern for system integration.
    """

    def __init__(self, unified_store: UnifiedDataStore):
        """
        Initialize bridge with UnifiedDataStore.

        Args:
            unified_store: UnifiedDataStore instance
        """
        self.adapter = UnifiedDataStoreAdapter(unified_store)
        logger.info("LegacySystemBridge initialized")

    def create_legacy_data_provider(self):
        """
        Create a data provider that mimics the legacy interface.

        Returns:
            Data provider object compatible with legacy prediction system
        """
        return self.adapter

    def get_legacy_prediction_data(self, team1: str, team2: str, season: str) -> Dict:
        """
        Get all data needed for legacy prediction system.

        Args:
            team1: First team name
            team2: Second team name
            season: NBA season

        Returns:
            Dictionary with all necessary data for prediction
        """
        return {
            'team1_data': {
                'name': team1,
                'id': self.adapter.get_team_id(team1),
                'roster': self.adapter.get_team_roster(team1, season),
                'injuries': self.adapter.get_team_injuries(team1),
                'player_stats': self.adapter.get_player_stats(team1, season),
                'momentum': self.adapter.get_team_momentum_metrics(team1),
                'recent_games': self.adapter.get_team_recent_games(team1)
            },
            'team2_data': {
                'name': team2,
                'id': self.adapter.get_team_id(team2),
                'roster': self.adapter.get_team_roster(team2, season),
                'injuries': self.adapter.get_team_injuries(team2),
                'player_stats': self.adapter.get_player_stats(team2, season),
                'momentum': self.adapter.get_team_momentum_metrics(team2),
                'recent_games': self.adapter.get_team_recent_games(team2)
            },
            'historical_games': self.adapter.get_historical_games(team1, team2, season)
        }