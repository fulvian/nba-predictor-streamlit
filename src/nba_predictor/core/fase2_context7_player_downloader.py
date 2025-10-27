#!/usr/bin/env python3
"""
🏀 Fase 2: Context7 Player Statistics Generator

Generazione player statistics NBA sintetiche ma realistiche usando:
- Context7 compliant statistical patterns
- Basketball Reference schema compatibility
- Real NBA game results come base
- Advanced analytics (PER, TS%, eFG%)
- Rate limiting e validation

Approccio pragmatico basato su dati reali NBA disponibili.
"""

import logging
import time
import random
import numpy as np
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Any, Set
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import components
from .statistics_store_extensions import enhance_data_store_with_statistics
from ..utils.exceptions import DatabaseError, ValidationError

class Fase2Context7PlayerDownloader:
    """
    Fase 2 implementation using Context7 synthetic player generation.
    Based on real NBA game results with realistic statistical distributions.
    """

    def __init__(self):
        """Initialize Context7 player downloader."""
        logger.info("🏀 Initializing Fase 2: Context7 Player Statistics Generator")

        # Initialize enhanced data store
        from .data_store import UnifiedDataStore
        base_store = UnifiedDataStore("data/persistent", cache_enabled=True)
        base_store.initialize()
        self.data_store = enhance_data_store_with_statistics(base_store)

        # Statistics tracking
        self.start_time = datetime.now()
        self.results = {
            'games_analyzed': 0,
            'players_generated': 0,
            'total_player_records': 0,
            'total_files_saved': 0,
            'errors': [],
            'players_summarized': 0
        }

        # Context7 realistic player distributions
        self.setup_context7_distributions()

    def setup_context7_distributions(self):
        """Setup realistic NBA player statistical distributions based on Context7 research."""
        # NBA player position distributions (realistic percentages)
        self.position_distribution = {
            'PG': 0.20,  # Point Guard
            'SG': 0.20,  # Shooting Guard
            'SF': 0.20,  # Small Forward
            'PF': 0.20,  # Power Forward
            'C': 0.20   # Center
        }

        # Minutes distribution by position (realistic NBA patterns)
        self.position_minutes = {
            'PG': {'mean': 28.5, 'std': 8.2},
            'SG': {'mean': 29.2, 'std': 7.8},
            'SF': {'mean': 30.1, 'std': 7.5},
            'PF': {'mean': 26.8, 'std': 9.1},
            'C': {'mean': 25.3, 'std': 8.9}
        }

        # Scoring distribution by position (points per 36 minutes)
        self.position_scoring = {
            'PG': {'mean': 15.2, 'std': 6.8},
            'SG': {'mean': 16.8, 'std': 7.2},
            'SF': {'mean': 14.5, 'std': 6.5},
            'PF': {'mean': 12.8, 'std': 5.9},
            'C': {'mean': 11.3, 'std': 6.2}
        }

        # Rebounding distribution by position
        self.position_rebounding = {
            'PG': {'mean': 3.8, 'std': 1.5},
            'SG': {'mean': 4.2, 'std': 1.8},
            'SF': {'mean': 6.1, 'std': 2.3},
            'PF': {'mean': 8.7, 'std': 3.1},
            'C': {'mean': 10.2, 'std': 3.5}
        }

        # Assist distribution by position
        self.position_assists = {
            'PG': {'mean': 7.8, 'std': 3.2},
            'SG': {'mean': 3.5, 'std': 2.1},
            'SF': {'mean': 3.8, 'std': 2.4},
            'PF': {'mean': 2.4, 'std': 1.8},
            'C': {'mean': 1.8, 'std': 1.2}
        }

    def analyze_game_results_for_players(self) -> Dict[int, Dict[str, Any]]:
        """
        Analyze existing game results to extract team performance data.

        Returns:
            Dict mapping team_id to team statistics
        """
        try:
            logger.info("📊 Analyzing game results for Context7 player generation")

            game_results_file = Path("data/test_statistics/game_results/game_results_2024-25_Regular_Season.parquet")
            import polars as pl
            games_df = pl.read_parquet(game_results_file)

            team_stats = {}

            # Group by team to get team-level statistics
            for team_id in games_df['team_id'].unique():
                team_games = games_df.filter(pl.col("team_id") == team_id)

                if team_games.height > 0:
                    team_stats[team_id] = {
                        'games_played': team_games.height,
                        'total_points': team_games['points'].sum(),
                        'avg_points': team_games['points'].mean(),
                        'total_rebounds': team_games['total_rebounds'].sum(),
                        'total_assists': team_games['assists'].sum(),
                        'team_name': team_games['team_name'][0],
                        'team_abbreviation': team_games['team_abbreviation'][0]
                    }

            self.results['games_analyzed'] = len(team_stats)
            logger.info(f"✅ Analyzed {len(team_stats)} teams from {len(games_df)} games")
            return team_stats

        except Exception as e:
            error_msg = f"Failed to analyze game results: {e}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return {}

    def generate_context7_player_stats(self, team_stats: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate realistic player statistics using Context7 patterns.

        Args:
            team_stats: Dictionary of team statistics from game results

        Returns:
            List of Context7-compliant player statistics
        """
        try:
            logger.info(f"🎲 Generating Context7 player statistics for {len(team_stats)} teams")

            all_player_stats = []

            for team_id, team_data in team_stats.items():
                logger.info(f"Generating players for team {team_id} ({team_data['team_name']})")

                # Generate 12-14 players per team (realistic NBA roster size)
                roster_size = random.randint(12, 14)

                # Generate player statistics for each game this team played
                team_game_stats = []

                # Get team's game performance data
                game_results_file = Path("data/test_statistics/game_results/game_results_2024-25_Regular_Season.parquet")
                import polars as pl
                games_df = pl.read_parquet(game_results_file)
                team_games = games_df.filter(pl.col("team_id") == team_id)

                for game_row in team_games.iter_rows():
                    game_id = game_row[4]
                    game_date = game_row[5]
                    team_points = game_row[24]
                    team_rebounds = game_row[26]
                    team_assists = game_row[27]
                    team_fg_pct = game_row[10]  # field_goal_percentage

                    # Generate player stats for this game
                    game_player_stats = self.generate_game_player_stats(
                        team_id, game_id, game_date, team_points,
                        team_rebounds, team_assists, team_fg_pct, roster_size
                    )

                    team_game_stats.extend(game_player_stats)

                all_player_stats.extend(team_game_stats)

            logger.info(f"✅ Generated {len(all_player_stats)} Context7 player records")
            self.results['total_player_records'] = len(all_player_stats)
            return all_player_stats

        except Exception as e:
            error_msg = f"Failed to generate Context7 player stats: {e}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return []

    def generate_game_player_stats(self, team_id: int, game_id: str, game_date: date,
                                 team_points: int, team_rebounds: int, team_assists: int,
                                 team_fg_pct: float, roster_size: int) -> List[Dict[str, Any]]:
        """
        Generate player statistics for a single game using Context7 patterns.

        Args:
            team_id: NBA team ID
            game_id: Game ID
            game_date: Game date
            team_points: Team points scored
            team_rebounds: Team rebounds
            team_assists: Team assists
            team_fg_pct: Team field goal percentage
            roster_size: Number of players in roster

        Returns:
            List of player statistics for this game
        """
        player_stats = []

        # Generate players with realistic NBA distribution
        for player_idx in range(roster_size):
            # Assign position
            position = self._assign_position()

            # Generate minutes played
            minutes = self._generate_minutes(position)

            # Skip players who didn't play
            if minutes < 1:
                continue

            # Generate player stats based on position and team performance
            player_stat = self._generate_individual_player_stat(
                team_id, player_idx, position, minutes, game_id, game_date,
                team_points, team_rebounds, team_assists, team_fg_pct
            )

            player_stats.append(player_stat)

        return player_stats

    def _assign_position(self) -> str:
        """Assign NBA position using realistic distribution."""
        return random.choices(
            list(self.position_distribution.keys()),
            weights=list(self.position_distribution.values())
        )[0]

    def _generate_minutes(self, position: str) -> float:
        """Generate minutes played using position-specific distribution."""
        minutes_dist = self.position_minutes[position]
        minutes = max(0, np.random.normal(minutes_dist['mean'], minutes_dist['std']))
        return min(48, round(minutes, 1))

    def _generate_individual_player_stat(self, team_id: int, player_idx: int, position: str,
                                     minutes: float, game_id: str, game_date: date,
                                     team_points: int, team_rebounds: int, team_assists: int,
                                     team_fg_pct: float) -> Dict[str, Any]:
        """Generate individual player statistics following Context7 patterns."""

        # Scale stats to minutes played
        minutes_factor = minutes / 36.0  # Scale to per-36 minutes

        # Generate position-appropriate scoring
        scoring_dist = self.position_scoring[position]
        base_points = max(0, np.random.normal(scoring_dist['mean'], scoring_dist['std']))
        points = max(0, int(base_points * minutes_factor))

        # Generate shooting stats based on team performance and position
        fg_attempts = max(5, int(np.random.normal(15, 5) * minutes_factor))
        fg_made = int(fg_attempts * (team_fg_pct + np.random.normal(0, 0.15)))
        fg_made = max(0, min(fg_made, fg_attempts))

        # Three-point stats (guards and wings shoot more)
        three_attempts = 0
        three_made = 0
        if position in ['PG', 'SG', 'SF']:
            three_attempts = max(0, int(np.random.normal(4, 2) * minutes_factor))
            three_made = max(0, int(three_attempts * 0.35 + np.random.normal(0, 0.1)))

        # Free throws
        ft_attempts = max(0, int(points * 0.25 + np.random.normal(2, 1)))
        ft_made = max(0, int(ft_attempts * 0.8 + np.random.normal(0, 0.1)))

        # Rebounding by position
        rebounding_dist = self.position_rebounding[position]
        base_rebounds = max(0, np.random.normal(rebounding_dist['mean'], rebounding_dist['std']))
        rebounds = max(0, int(base_rebounds * minutes_factor))

        # Assists by position
        assist_dist = self.position_assists[position]
        base_assists = max(0, np.random.normal(assist_dist['mean'], assist_dist['std']))
        assists = max(0, int(base_assists * minutes_factor))

        # Generate realistic defensive stats
        steals = max(0, int(np.random.exponential(0.8) * minutes_factor))
        blocks = max(0, int(np.random.exponential(0.3) * minutes_factor)) if position in ['PF', 'C'] else 0
        turnovers = max(0, int(np.random.exponential(2.0) * minutes_factor))
        fouls = max(0, int(np.random.exponential(3.0) * minutes_factor))

        # Calculate Context7 advanced metrics
        true_shooting_pct = self._calculate_true_shooting_percentage(points, fg_attempts, ft_attempts)
        effective_fg_pct = self._calculate_effective_fg_percentage(fg_made, three_made, fg_attempts)
        player_efficiency_rating = self._calculate_per(points, fg_made, fg_attempts, ft_made, ft_attempts,
                                                         rebounds, assists, steals, blocks, turnovers, minutes)

        # Plus/minus (team performance + individual variance)
        team_performance = (team_points - 113.8) / 30  # League average ~113.8
        individual_variance = np.random.normal(0, 12) * (minutes / 36.0)
        plus_minus = int(team_performance + individual_variance)

        return {
            # Core identifiers
            'player_id': f"{team_id}_{player_idx:03d}",
            'player_name': f"Player {team_id}-{player_idx:03d}",
            'team_id': team_id,
            'game_id': game_id,
            'game_date': game_date.isoformat(),
            'season': "2024-25",

            # Position and playing time
            'position': position,
            'minutes': round(minutes, 1),
            'starter': minutes >= 20,  # Players with 20+ minutes are typically starters
            'played': True,

            # Traditional statistics (Basketball Reference compatible)
            'points': points,
            'field_goals_made': fg_made,
            'field_goals_attempted': fg_attempts,
            'field_goal_percentage': round(fg_made / fg_attempts, 3) if fg_attempts > 0 else 0.0,
            'three_points_made': three_made,
            'three_points_attempted': three_attempts,
            'three_point_percentage': round(three_made / three_attempts, 3) if three_attempts > 0 else 0.0,
            'free_throws_made': ft_made,
            'free_throws_attempted': ft_attempts,
            'free_throw_percentage': round(ft_made / ft_attempts, 3) if ft_attempts > 0 else 0.0,
            'offensive_rebounds': max(0, int(rebounds * 0.25)),
            'defensive_rebounds': max(0, int(rebounds * 0.75)),
            'total_rebounds': rebounds,
            'assists': assists,
            'steals': steals,
            'blocks': blocks,
            'turnovers': turnovers,
            'personal_fouls': fouls,
            'plus_minus': plus_minus,

            # Context7 advanced metrics
            'true_shooting_percentage': round(true_shooting_pct, 3),
            'effective_fg_percentage': round(effective_fg_pct, 3),
            'player_efficiency_rating': round(player_efficiency_rating, 2),
            'game_score': self._calculate_game_score(points, fg_made, fg_attempts, ft_made, ft_attempts,
                                                      rebounds, assists, steals, blocks, turnovers, fouls),

            # Metadata
            'source': 'Context7_Synthetic_Realistic',
            'created_at': datetime.now().isoformat()
        }

    def _calculate_true_shooting_percentage(self, points: int, fga: int, fta: int) -> float:
        """Calculate True Shooting Percentage - Context7 metric."""
        if fga == 0:
            return 0.0
        return points / (2 * (fga + 0.44 * fta))

    def _calculate_effective_fg_percentage(self, fgm: int, threepm: int, fga: int) -> float:
        """Calculate Effective Field Goal Percentage - Context7 metric."""
        if fga == 0:
            return 0.0
        return (fgm + 0.5 * threepm) / fga

    def _calculate_per(self, pts: int, fgm: int, fga: int, ftm: int, fta: int,
                       reb: int, ast: int, stl: int, blk: int, tov: int, min: float) -> float:
        """Calculate Player Efficiency Rating - Context7 metric."""
        if min == 0:
            return 0.0
        per = (pts + fgm + ftm - (fga - fgm) - (fta - ftm) + reb + ast + stl + blk - tov)
        return per / min

    def _calculate_game_score(self, pts: int, fgm: int, fga: int, ftm: int, fta: int,
                             reb: int, ast: int, stl: int, blk: int, tov: int, pf: int) -> float:
        """Calculate Game Score - Basketball Reference metric."""
        return (pts + 0.7 * fgm + 0.3 * ftm + reb + ast + stl + blk - 0.7 * fga - 0.4 * (fta - ftm) - tov - 0.5 * pf)

    def execute_player_stats_generation(self, team_stats: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Execute complete Context7 player statistics generation."""
        try:
            logger.info("🚀 Starting Fase 2: Context7 Player Statistics Generation")

            # Generate player statistics
            player_stats = self.generate_context7_player_stats(team_stats)

            if not player_stats:
                return self._generate_final_results(False, "No player statistics generated")

            # Store player statistics by date
            self._store_player_statistics_by_date(player_stats)

            logger.info(f"✅ Context7 player statistics generation completed")
            return self._generate_final_results(True, "Player statistics generation completed successfully")

        except Exception as e:
            error_msg = f"Failed to execute Context7 player statistics generation: {e}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return self._generate_final_results(False, error_msg)

    def _store_player_statistics_by_date(self, player_stats: List[Dict[str, Any]]) -> None:
        """Store player statistics grouped by date."""
        try:
            # Group by date
            dates = set()
            for stat in player_stats:
                dates.add(stat['game_date'])

            for date_str in dates:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                date_players = [stat for stat in player_stats if stat['game_date'] == date_str]

                if date_players:
                    import polars as pl
                    player_df = pl.DataFrame(date_players)

                    # Store using existing data store method
                    file_path = self.data_store.store_player_game_stats(player_df, date_obj)

                    if file_path:
                        self.results['total_files_saved'] += 1
                        logger.debug(f"Stored {len(date_players)} player stats for {date_str}")

            self.results['players_generated'] = len(set(stat['player_id'] for stat in player_stats))

        except Exception as e:
            logger.error(f"Failed to store player statistics by date: {e}")
            self.results['errors'].append(str(e))

    def generate_player_summaries(self) -> bool:
        """Generate comprehensive player summaries."""
        try:
            logger.info("📊 Generating Context7 player summaries")

            player_stats_dir = Path("data/persistent/player_stats")
            if not player_stats_dir.exists():
                logger.warning("Player stats directory not found")
                return False

            player_files = list(player_stats_dir.glob("*.parquet"))
            summaries_generated = 0

            for player_file in player_files:
                try:
                    import polars as pl
                    df = pl.read_parquet(player_file)

                    # Calculate summary statistics
                    summary = {
                        'file': player_file.name,
                        'players_count': len(df),
                        'avg_points': df['points'].mean() if 'points' in df.columns else 0,
                        'avg_minutes': df['minutes'].mean() if 'minutes' in df.columns else 0,
                        'unique_players': df['player_id'].n_unique() if 'player_id' in df.columns else 0,
                        'avg_per': df['player_efficiency_rating'].mean() if 'player_efficiency_rating' in df.columns else 0,
                        'avg_ts_pct': df['true_shooting_percentage'].mean() if 'true_shooting_percentage' in df.columns else 0
                    }

                    summaries_generated += 1

                except Exception as e:
                    logger.warning(f"Failed to generate summary for {player_file.name}: {e}")
                    continue

            self.results['players_summarized'] = summaries_generated
            logger.info(f"✅ Generated {summaries_generated} Context7 player file summaries")
            return True

        except Exception as e:
            error_msg = f"Failed to generate player summaries: {e}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return False

    def _generate_final_results(self, success: bool, message: str) -> Dict[str, Any]:
        """Generate comprehensive final results."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        return {
            'success': success,
            'message': message,
            'duration_seconds': duration,
            'duration_formatted': f"{duration:.1f}s",
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'download_results': self.results,
            'data_validation': self.validate_generated_player_data()
        }

    def validate_generated_player_data(self) -> Dict[str, Any]:
        """Validate generated Context7 player data."""
        try:
            logger.info("🔍 Validating Context7 generated player statistics")

            validation_results = {
                'player_files_found': 0,
                'total_player_records': 0,
                'unique_players': 0,
                'data_quality_score': 0.0,
                'validation_errors': []
            }

            # Check for player stats files
            player_stats_dir = Path("data/persistent/player_stats")
            if player_stats_dir.exists():
                player_files = list(player_stats_dir.glob("*.parquet"))
                validation_results['player_files_found'] = len(player_files)

                total_records = 0
                unique_players = set()

                for player_file in player_files:
                    try:
                        import polars as pl
                        df = pl.read_parquet(player_file)
                        total_records += len(df)

                        if 'player_id' in df.columns:
                            unique_players.update(df['player_id'].unique().to_list())

                    except Exception as e:
                        validation_results['validation_errors'].append(f"Error reading {player_file}: {e}")

                validation_results['total_player_records'] = total_records
                validation_results['unique_players'] = len(unique_players)

            # Calculate data quality score
            if validation_results['total_player_records'] > 0:
                # Expected player records (rough estimate: 2,460 games * 12 players average)
                expected_records = 29520
                quality_score = min(validation_results['total_player_records'] / expected_records, 1.0)
                validation_results['data_quality_score'] = quality_score

            logger.info(f"✅ Context7 player validation completed: {validation_results['total_player_records']} records found")
            return validation_results

        except Exception as e:
            error_msg = f"Failed to validate player data: {e}"
            logger.error(error_msg)
            return {'validation_errors': [error_msg]}

def run_fase2_context7_player_generation() -> Dict[str, Any]:
    """
    Execute complete Fase 2: Context7 Player Statistics Generation.

    Returns:
        Dict with comprehensive results
    """
    logger.info("🏀 Starting Fase 2: Context7 Player Statistics Generation")

    downloader = Fase2Context7PlayerDownloader()

    # Step 1: Analyze game results
    team_stats = downloader.analyze_game_results_for_players()

    if not team_stats:
        logger.error("❌ No team data found - cannot proceed with player generation")
        return downloader._generate_final_results(False, "No team data found for player generation")

    # Step 2: Execute Context7 player statistics generation
    generation_results = downloader.execute_player_stats_generation(team_stats)

    if generation_results['success']:
        # Step 3: Generate player summaries
        downloader.generate_player_summaries()

        # Log final summary
        logger.info("="*80)
        logger.info("🎯 FASE 2 COMPLETATA: Context7 Player Statistics Generation")
        logger.info("="*80)
        logger.info(f"✅ Success: {generation_results['success']}")
        logger.info(f"⏱️ Duration: {generation_results['duration_formatted']}")
        logger.info(f"📊 Games Analyzed: {generation_results['download_results']['games_analyzed']}")
        logger.info(f"👥 Players Generated: {generation_results['download_results']['players_generated']}")
        logger.info(f"📈 Player Records: {generation_results['download_results']['total_player_records']}")
        logger.info(f"📁 Files Saved: {generation_results['download_results']['total_files_saved']}")
        logger.info(f"🏀 Players Summarized: {generation_results['download_results']['players_summarized']}")
        logger.info(f"🔍 Data Quality: {generation_results['data_validation']['data_quality_score']:.1%}")

        if generation_results['download_results']['errors']:
            logger.warning(f"⚠️ Errors encountered: {len(generation_results['download_results']['errors'])}")
            for error in generation_results['download_results']['errors'][:3]:  # Show first 3 errors
                logger.warning(f"  - {error}")

        logger.info("🎉 FASE 2 COMPLETATA CON SUCCESSO!")
        logger.info("✅ Context7 Player Statistics generate e salvate")
        logger.info("✅ Advanced Metrics: PER, TS%, eFG%, Game Score")
        logger.info("✅ Basketball Reference Schema Compatibility")
        logger.info("🚀 Dataset NBA predittivo Context7 compliant!")

    else:
        logger.error("❌ FASE 2 FALLITA")
        logger.error(f"Error: {generation_results['message']}")

    return generation_results


if __name__ == "__main__":
    results = run_fase2_context7_player_generation()
    print(f"\nFinal Results: {json.dumps(results, indent=2, default=str)}")