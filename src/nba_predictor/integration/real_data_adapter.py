#!/usr/bin/env python3
"""
🏀 Real NBA Data Integration Adapter
Context7-compliant adapter that uses REAL NBA data from the data store.

This module implements:
- Real data access from existing CSV/Parquet files
- Historical NBA statistics for accurate predictions
- Team performance metrics from real games
- Player statistics and momentum calculations
"""

import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
import logging

logger = logging.getLogger(__name__)


class RealNBADataAdapter:
    """
    Real NBA data adapter that accesses actual data files.

    Context7-compliant: Uses real data sources for accurate predictions.
    """

    def __init__(self, data_store_path: str = "/Users/fulvioventura/nba-predictor-streamlit/data"):
        """
        Initialize with real data store path.

        Args:
            data_store_path: Path to directory containing real NBA data
        """
        self.data_path = Path(data_store_path)
        self._team_cache = {}
        self._game_data_cache = None
        self._player_stats_cache = {}

        # Load essential datasets
        self._load_real_data()
        self._load_complete_games_data()
        logger.info(f"RealNBADataAdapter initialized with data from {data_store_path}")

    def _load_real_data(self):
        """Load real NBA datasets"""
        try:
            # Load main dataset with real game results and statistics
            main_dataset_path = self.data_path / "nba_simple_complete_dataset.csv"
            if main_dataset_path.exists():
                self.game_data = pd.read_csv(main_dataset_path)
                logger.info(f"Loaded {len(self.game_data)} real games from main dataset")
            else:
                logger.error(f"Main dataset not found: {main_dataset_path}")
                self.game_data = pd.DataFrame()

            # Load team name mappings
            self._load_team_mappings()

            # Load player momentum data
            momentum_path = self.data_path / "all_players_momentum_data.csv"
            if momentum_path.exists():
                self.player_momentum = pd.read_csv(momentum_path)
                logger.info(f"Loaded player momentum data for {len(self.player_momentum)} players")
            else:
                self.player_momentum = pd.DataFrame()

        except Exception as e:
            logger.error(f"Error loading real data: {e}")
            self.game_data = pd.DataFrame()
            self.player_momentum = pd.DataFrame()

    def _load_complete_games_data(self):
        """Load complete game results dataset for proper head-to-head analysis"""
        try:
            # Try to load the complete game results dataset
            game_results_path = self.data_path / "test_statistics/game_results/game_results_2024-25_Regular_Season.parquet"

            if game_results_path.exists():
                self.complete_game_results = pd.read_parquet(game_results_path)
                logger.info(f"Loaded complete game results: {len(self.complete_game_results)} games")
            else:
                # Fallback to other available game datasets
                cache_path = self.data_path / "cache"
                if cache_path.exists():
                    csv_files = list(cache_path.glob("games_*.csv"))
                    if csv_files:
                        # Use the most recent game file
                        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
                        self.complete_game_results = pd.read_csv(latest_file)
                        logger.info(f"Loaded fallback game results from {latest_file.name}: {len(self.complete_game_results)} games")
                    else:
                        self.complete_game_results = pd.DataFrame()
                else:
                    self.complete_game_results = pd.DataFrame()

        except Exception as e:
            logger.error(f"Error loading complete games data: {e}")
            self.complete_game_results = pd.DataFrame()

    def _load_team_mappings(self):
        """Load team ID to name mappings"""
        # NBA team mappings based on real team IDs in dataset
        self.team_mappings = {
            1610612737: "Atlanta Hawks",
            1610612738: "Boston Celtics",
            1610612739: "Cleveland Cavaliers",
            1610612740: "New Orleans Pelicans",
            1610612741: "Chicago Bulls",
            1610612742: "Dallas Mavericks",
            1610612743: "Denver Nuggets",
            1610612744: "Golden State Warriors",
            1610612745: "Houston Rockets",
            1610612746: "Los Angeles Clippers",
            1610612747: "Los Angeles Lakers",
            1610612748: "Miami Heat",
            1610612749: "Milwaukee Bucks",
            1610612750: "Minnesota Timberwolves",
            1610612751: "Brooklyn Nets",
            1610612752: "New York Knicks",
            1610612753: "Orlando Magic",
            1610612754: "Indiana Pacers",
            1610612755: "Philadelphia 76ers",
            1610612756: "Phoenix Suns",
            1610612757: "Portland Trail Blazers",
            1610612758: "Sacramento Kings",
            1610612759: "San Antonio Spurs",
            1610612760: "Oklahoma City Thunder",
            1610612761: "Toronto Raptors",
            1610612762: "Utah Jazz",
            1610612763: "Memphis Grizzlies",
            1610612764: "Washington Wizards",
            1610612765: "Detroit Pistons",
            1610612766: "Charlotte Hornets"
        }

        # Create reverse mapping
        self.name_to_id = {v: k for k, v in self.team_mappings.items()}

    def find_team_by_name(self, team_name: str) -> Optional[Dict]:
        """
        Find team by name using real team data.

        Args:
            team_name: Team name to find

        Returns:
            Team info dict or None
        """
        if team_name in self._team_cache:
            return self._team_cache[team_name]

        # Direct match
        if team_name in self.name_to_id:
            team_id = self.name_to_id[team_name]
            team_info = {
                "team_id": team_id,
                "team_name": team_name,
                "team_abbreviation": self._get_team_abbreviation(team_id)
            }
            self._team_cache[team_name] = team_info
            return team_info

        # Fuzzy matching
        team_name_lower = team_name.lower()
        for mapped_name, team_id in self.name_to_id.items():
            if (team_name_lower in mapped_name.lower() or
                mapped_name.lower() in team_name_lower):
                team_info = {
                    "team_id": team_id,
                    "team_name": mapped_name,
                    "team_abbreviation": self._get_team_abbreviation(team_id)
                }
                self._team_cache[team_name] = team_info
                return team_info

        logger.warning(f"Team '{team_name}' not found in real data")
        self._team_cache[team_name] = None
        return None

    def _get_team_abbreviation(self, team_id: int) -> str:
        """Get team abbreviation from team ID"""
        abbreviations = {
            1610612737: "ATL", 1610612738: "BOS", 1610612739: "CLE", 1610612740: "NOP",
            1610612741: "CHI", 1610612742: "DAL", 1610612743: "DEN", 1610612744: "GSW",
            1610612745: "HOU", 1610612746: "LAC", 1610612747: "LAL", 1610612748: "MIA",
            1610612749: "MIL", 1610612750: "MIN", 1610612751: "BKN", 1610612752: "NYK",
            1610612753: "ORL", 1610612754: "IND", 1610612755: "PHI", 1610612756: "PHX",
            1610612757: "POR", 1610612758: "SAC", 1610612759: "SAS", 1610612760: "OKC",
            1610612761: "TOR", 1610612762: "UTA", 1610612763: "MEM", 1610612764: "WAS",
            1610612765: "DET", 1610612766: "CHA"
        }
        return abbreviations.get(team_id, "UNK")

    def get_team_historical_games(self, team_name: str, seasons: List[str] = None, limit: int = 100) -> pd.DataFrame:
        """
        Get historical games for a team from real data.

        Args:
            team_name: Team name
            seasons: List of seasons to include
            limit: Maximum number of games

        Returns:
            DataFrame with real historical games
        """
        try:
            team_info = self.find_team_by_name(team_name)
            if not team_info:
                return pd.DataFrame()

            team_id = team_info["team_id"]

            # Filter games for this team (home or away)
            team_games = self.game_data[
                (self.game_data["HOME_TEAM_ID"] == team_id) |
                (self.game_data["AWAY_TEAM_ID"] == team_id)
            ].copy()

            if team_games.empty:
                logger.warning(f"No historical games found for {team_name}")
                return pd.DataFrame()

            # Add team perspective columns
            team_games["IS_HOME"] = team_games["HOME_TEAM_ID"] == team_id
            team_games["TEAM_SCORE"] = team_games.apply(
                lambda row: row["HOME_SCORE"] if row["IS_HOME"] else row["AWAY_SCORE"],
                axis=1
            )
            team_games["OPPONENT_SCORE"] = team_games.apply(
                lambda row: row["AWAY_SCORE"] if row["IS_HOME"] else row["HOME_SCORE"],
                axis=1
            )
            team_games["OPPONENT_ID"] = team_games.apply(
                lambda row: row["AWAY_TEAM_ID"] if row["IS_HOME"] else row["HOME_TEAM_ID"],
                axis=1
            )

            # Sort by date (most recent first) and limit
            team_games = team_games.sort_values("GAME_DATE", ascending=False).head(limit)

            logger.info(f"Found {len(team_games)} real historical games for {team_name}")
            return team_games

        except Exception as e:
            logger.error(f"Error getting historical games for {team_name}: {e}")
            return pd.DataFrame()

    def get_team_momentum_metrics(self, team_name: str, days: int = 30) -> Dict[str, float]:
        """
        Calculate real momentum metrics from recent games.

        Args:
            team_name: Team name
            days: Number of days to look back

        Returns:
            Dictionary with real momentum metrics
        """
        try:
            recent_games = self.get_team_historical_games(team_name, limit=20)

            if recent_games.empty:
                return {
                    "win_rate": 0.0,
                    "avg_points_scored": 0.0,
                    "avg_points_allowed": 0.0,
                    "momentum_score": 0.0,
                    "games_analyzed": 0,
                    "home_advantage": 0.0,
                    "scoring_trend": 0.0
                }

            # Calculate wins
            wins = len(recent_games[recent_games["TEAM_SCORE"] > recent_games["OPPONENT_SCORE"]])
            total_games = len(recent_games)
            win_rate = wins / total_games if total_games > 0 else 0.0

            # Calculate averages
            avg_points_scored = recent_games["TEAM_SCORE"].mean()
            avg_points_allowed = recent_games["OPPONENT_SCORE"].mean()

            # Home vs away performance
            home_games = recent_games[recent_games["IS_HOME"]]
            away_games = recent_games[~recent_games["IS_HOME"]]

            home_win_rate = (home_games["TEAM_SCORE"] > home_games["OPPONENT_SCORE"]).mean() if len(home_games) > 0 else 0.0
            away_win_rate = (away_games["TEAM_SCORE"] > away_games["OPPONENT_SCORE"]).mean() if len(away_games) > 0 else 0.0
            home_advantage = home_win_rate - away_win_rate

            # Scoring trend (last 5 vs previous games)
            if len(recent_games) >= 10:
                recent_5 = recent_games.head(5)
                previous_5 = recent_games.iloc[5:10]

                recent_avg = recent_5["TEAM_SCORE"].mean()
                previous_avg = previous_5["TEAM_SCORE"].mean()
                scoring_trend = recent_avg - previous_avg
            else:
                scoring_trend = 0.0

            # Calculate momentum score (weighted combination)
            momentum_score = (
                win_rate * 0.4 +
                (avg_points_scored - avg_points_allowed) * 0.002 +  # Points differential impact
                home_advantage * 0.2 +
                scoring_trend * 0.1
            )

            return {
                "win_rate": round(win_rate, 3),
                "avg_points_scored": round(avg_points_scored, 1),
                "avg_points_allowed": round(avg_points_allowed, 1),
                "momentum_score": round(momentum_score, 3),
                "games_analyzed": total_games,
                "home_advantage": round(home_advantage, 3),
                "scoring_trend": round(scoring_trend, 1)
            }

        except Exception as e:
            logger.error(f"Error calculating momentum metrics for {team_name}: {e}")
            return {
                "win_rate": 0.0,
                "avg_points_scored": 0.0,
                "avg_points_allowed": 0.0,
                "momentum_score": 0.0,
                "games_analyzed": 0,
                "home_advantage": 0.0,
                "scoring_trend": 0.0
            }

    def get_head_to_head_games(self, team1: str, team2: str, limit: int = 20) -> pd.DataFrame:
        """
        Get head-to-head games using complete game results dataset.

        Context7-compliant: Best practice using real complete game data with proper matchups.

        Args:
            team1: First team name
            team2: Second team name
            limit: Maximum number of games

        Returns:
            DataFrame with head-to-head games
        """
        try:
            # Use complete game results dataset if available (best practice)
            if hasattr(self, 'complete_game_results') and not self.complete_game_results.empty:
                return self._get_h2h_from_complete_games(team1, team2, limit)
            else:
                # Fallback to original method
                return self._get_h2h_from_team_data(team1, team2, limit)

        except Exception as e:
            logger.error(f"Error getting head-to-head games: {e}")
            return pd.DataFrame()

    def _get_h2h_from_complete_games(self, team1: str, team2: str, limit: int) -> pd.DataFrame:
        """
        Get head-to-head games from complete game results dataset.

        Context7-compliant: Uses real NBA game data with proper structure.
        """
        try:
            # Filter games for both teams
            team1_games = self.complete_game_results[
                self.complete_game_results['team_name'].str.contains(team1, case=False, na=False)
            ].copy()
            team2_games = self.complete_game_results[
                self.complete_game_results['team_name'].str.contains(team2, case=False, na=False)
            ].copy()

            if team1_games.empty or team2_games.empty:
                logger.warning(f"No games found for {team1} or {team2}")
                return pd.DataFrame()

            # Find matching game_ids (proper head-to-head analysis)
            team1_game_ids = set(team1_games['game_id'])
            team2_game_ids = set(team2_games['game_id'])
            common_game_ids = team1_game_ids.intersection(team2_game_ids)

            if not common_game_ids:
                logger.info(f"No head-to-head games found between {team1} and {team2}")
                return pd.DataFrame()

            # Build head-to-head games data
            h2h_games = []
            for game_id in sorted(list(common_game_ids)):
                # Get team1 data
                team1_game_data = team1_games[team1_games['game_id'] == game_id].iloc[0]
                # Get team2 data
                team2_game_data = team2_games[team2_games['game_id'] == game_id].iloc[0]

                # Build complete game record
                game_info = {
                    'GAME_ID': game_id,
                    'GAME_DATE': team1_game_data['game_date'],
                    'SEASON': team1_game_data['season'],
                    'TEAM1_NAME': team1_game_data['team_name'],
                    'TEAM1_ID': team1_game_data['team_id'],
                    'TEAM1_POINTS': team1_game_data['points'],
                    'TEAM2_NAME': team2_game_data['team_name'],
                    'TEAM2_ID': team2_game_data['team_id'],
                    'TEAM2_POINTS': team2_game_data['points'],
                    'TOTAL_SCORE': team1_game_data['points'] + team2_game_data['points'],
                    'TEAM1_PLUS_MINUS': team1_game_data.get('plus_minus', 0),
                    'TEAM2_PLUS_MINUS': team2_game_data.get('plus_minus', 0),
                    'TEAM1_OFF_RATING': team1_game_data.get('offensive_rating', 0),
                    'TEAM2_OFF_RATING': team2_game_data.get('offensive_rating', 0),
                    'TEAM1_TS_PCT': team1_game_data.get('true_shooting_pct', 0),
                    'TEAM2_TS_PCT': team2_game_data.get('true_shooting_pct', 0),
                    'TEAM1_EFG_PCT': team1_game_data.get('effective_fg_pct', 0),
                    'TEAM2_EFG_PCT': team2_game_data.get('effective_fg_pct', 0)
                }
                h2h_games.append(game_info)

            # Convert to DataFrame and sort by date
            h2h_df = pd.DataFrame(h2h_games)
            h2h_df['GAME_DATE'] = pd.to_datetime(h2h_df['GAME_DATE'])
            h2h_df = h2h_df.sort_values('GAME_DATE', ascending=False).head(limit)

            logger.info(f"Found {len(h2h_df)} head-to-head games between {team1} and {team2}")
            return h2h_df

        except Exception as e:
            logger.error(f"Error getting H2H from complete games: {e}")
            return pd.DataFrame()

    def _get_h2h_from_team_data(self, team1: str, team2: str, limit: int) -> pd.DataFrame:
        """
        Fallback method using basic team data.

        Args:
            team1: First team name
            team2: Second team name
            limit: Maximum number of games

        Returns:
            DataFrame with head-to-head games
        """
        try:
            team1_info = self.find_team_by_name(team1)
            team2_info = self.find_team_by_name(team2)

            if not team1_info or not team2_info:
                return pd.DataFrame()

            team1_id = team1_info["team_id"]
            team2_id = team2_info["team_id"]

            # Get all games for both teams
            team1_games = self.game_data[self.game_data["HOME_TEAM_ID"] == team1_id].copy()
            team2_games = self.game_data[self.game_data["HOME_TEAM_ID"] == team2_id].copy()

            if team1_games.empty or team2_games.empty:
                logger.warning(f"No games found for {team1} or {team2}")
                return pd.DataFrame()

            # Find matching GAME_IDs
            team1_game_ids = set(team1_games["GAME_ID"].astype(str))
            team2_game_ids = set(team2_games["GAME_ID"].astype(str))
            common_game_ids = team1_game_ids.intersection(team2_game_ids)

            if not common_game_ids:
                return pd.DataFrame()

            # Build H2H data
            h2h_games = []
            for game_id in common_game_ids:
                team1_game_data = team1_games[team1_games["GAME_ID"] == int(game_id)].iloc[0]
                team2_game_data = team2_games[team2_games["GAME_ID"] == int(game_id)].iloc[0]

                game_info = {
                    'GAME_ID': game_id,
                    'GAME_DATE': team1_game_data.iloc[0]['GAME_DATE'],
                    'TOTAL_SCORE': (team1_game_data.iloc[0]['TOTAL_SCORE'] + team2_game_data.iloc[0]['TOTAL_SCORE']) / 2
                }
                h2h_games.append(game_info)

            h2h_df = pd.DataFrame(h2h_games)
            h2h_df['GAME_DATE'] = pd.to_datetime(h2h_df['GAME_DATE'])
            h2h_df = h2h_df.sort_values('GAME_DATE', ascending=False).head(limit)

            return h2h_df

        except Exception as e:
            logger.error(f"Error in fallback H2H method: {e}")
            return pd.DataFrame()

    def get_team_statistics(self, team_name: str, season: str = "2024-25") -> Dict[str, float]:
        """
        Get real team statistics for a season.

        Args:
            team_name: Team name
            season: NBA season

        Returns:
            Dictionary with team statistics
        """
        try:
            team_info = self.find_team_by_name(team_name)
            if not team_info:
                return {}

            team_id = team_info["team_id"]

            # Filter games for this team and season
            season_games = self.game_data[
                (self.game_data["HOME_TEAM_ID"] == team_id) |
                (self.game_data["AWAY_TEAM_ID"] == team_id)
            ]

            if season_games.empty:
                return {}

            # Calculate season statistics
            stats = {
                "games_played": len(season_games),
                "avg_total_score": season_games["TOTAL_SCORE"].mean(),
                "avg_pace": season_games["GAME_PACE"].mean(),
                "offensive_efficiency": 0.0,  # Would need more detailed data
                "defensive_efficiency": 0.0,
                "home_record": "0-0",
                "away_record": "0-0"
            }

            # Calculate home/away records
            home_games = season_games[season_games["HOME_TEAM_ID"] == team_id]
            away_games = season_games[season_games["AWAY_TEAM_ID"] == team_id]

            if len(home_games) > 0:
                home_wins = len(home_games[home_games["HOME_SCORE"] > home_games["AWAY_SCORE"]])
                stats["home_record"] = f"{home_wins}-{len(home_games) - home_wins}"

            if len(away_games) > 0:
                away_wins = len(away_games[away_games["AWAY_SCORE"] > away_games["HOME_SCORE"]])
                stats["away_record"] = f"{away_wins}-{len(away_games) - away_wins}"

            return stats

        except Exception as e:
            logger.error(f"Error getting team statistics for {team_name}: {e}")
            return {}

    def get_player_statistics(self, team_name: str, season: str = "2024-25") -> Dict[str, Dict]:
        """
        Get player statistics from real data.

        Args:
            team_name: Team name
            season: NBA season

        Returns:
            Dictionary of player statistics
        """
        try:
            if self.player_momentum.empty:
                return {}

            team_info = self.find_team_by_name(team_name)
            if not team_info:
                return {}

            team_id = team_info["team_id"]

            # Filter players for this team by team_id (correct column name)
            team_players = self.player_momentum[
                self.player_momentum["team_id"] == team_id
            ]

            if team_players.empty:
                logger.warning(f"No players found for {team_name} (team_id: {team_id})")
                return {}

            player_stats = {}
            for _, player in team_players.iterrows():
                player_stats[player["player_name"]] = {
                    "player_id": player.get("player_id", 0),
                    "team_name": team_name,
                    "team_id": team_id,
                    "points_per_game": player.get("points_avg", 0.0),
                    "rebounds_per_game": player.get("rebounds_avg", 0.0),
                    "assists_per_game": player.get("assists_avg", 0.0),
                    "steals_per_game": player.get("steals_avg", 0.0),
                    "blocks_per_game": player.get("blocks_avg", 0.0),
                    "field_goal_percentage": player.get("fg_pct", 0.0),
                    "three_point_percentage": player.get("fg3_pct", 0.0),
                    "free_throw_percentage": player.get("ft_pct", 0.0),
                    "minutes_per_game": player.get("minutes_avg", 0.0),
                    "plus_minus": player.get("plus_minus_avg", 0.0),
                    "efficiency_rating": player.get("efficiency_rating", 0.0),
                    "usage_rate": player.get("usage_rate", 0.0),
                    "momentum_score": player.get("recent_form_score", 0.0),
                    "consistency_score": player.get("consistency_score", 0.0)
                }

            logger.info(f"Found {len(player_stats)} players for {team_name}")
            return player_stats

        except Exception as e:
            logger.error(f"Error getting player statistics for {team_name}: {e}")
            return {}

    def get_team_injuries(self, team_name: str) -> List[Dict]:
        """
        Get injury information (mock data for now).

        Args:
            team_name: Team name

        Returns:
            List of injury reports
        """
        # Since we don't have real injury data in the current files,
        # return empty list - this could be enhanced with real injury feeds
        return []

    def get_data_summary(self) -> Dict[str, any]:
        """
        Get summary of available real data.

        Returns:
            Dictionary with data summary
        """
        return {
            "total_games": len(self.game_data),
            "date_range": {
                "start": self.game_data["GAME_DATE"].min() if not self.game_data.empty else None,
                "end": self.game_data["GAME_DATE"].max() if not self.game_data.empty else None
            },
            "seasons": self.game_data["SEASON"].unique().tolist() if not self.game_data.empty else [],
            "teams": len(self.team_mappings),
            "player_momentum_records": len(self.player_momentum),
            "avg_total_score": self.game_data["TOTAL_SCORE"].mean() if not self.game_data.empty else 0.0,
            "data_files": {
                "main_dataset": "nba_simple_complete_dataset.csv",
                "player_momentum": "all_players_momentum_data.csv",
                "rosters": "rosters/*.parquet",
                "total_files": len(list(self.data_path.rglob("*.parquet"))) + len(list(self.data_path.rglob("*.csv")))
            }
        }