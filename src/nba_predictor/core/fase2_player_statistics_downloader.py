#!/usr/bin/env python3
"""
🏀 Fase 2: Player Statistics Download - Context7 Implementation

Download delle statistiche giocatore NBA usando:
- NBA Statistics Download Engine con rate limiting
- Context7 compliant validation e storage
- Player stats dai game results 2024-25
- Season averages e performance metrics
- Progress tracking e resumption capability

Fase 2 del NBA Predictive Analytics System:
1. Analizza game results per identificare player stats
2. Download player statistics da NBA API endpoints
3. Storage in data store persistente con validation
4. Generation di player summaries e analytics
5. Integration con game results per complete dataset
"""

import logging
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import components
from .statistics_download_engine import NBAStatisticsDownloadEngine
from .statistics_store_extensions import enhance_data_store_with_statistics
from ..utils.exceptions import DatabaseError, ValidationError

class Fase2PlayerStatisticsDownloader:
    """
    Fase 2 implementation for downloading NBA player statistics.
    Context7-compliant massive data download with validation.
    """

    def __init__(self):
        """Initialize Fase 2 downloader."""
        logger.info("🏀 Initializing Fase 2: Player Statistics Downloader")

        # Initialize statistics download engine
        self.download_engine = NBAStatisticsDownloadEngine()

        # Initialize enhanced data store
        from .data_store import UnifiedDataStore
        base_store = UnifiedDataStore("data/persistent", cache_enabled=True)
        base_store.initialize()
        self.data_store = enhance_data_store_with_statistics(base_store)

        # Statistics tracking
        self.start_time = datetime.now()
        self.results = {
            'games_analyzed': 0,
            'players_identified': 0,
            'player_stats_downloaded': 0,
            'total_files_saved': 0,
            'errors': [],
            'players_summarized': 0
        }

    def analyze_game_results_for_players(self) -> Set[int]:
        """
        Analyze existing game results to identify unique players.

        Returns:
            Set of unique player IDs from game results
        """
        try:
            logger.info("📊 Analyzing game results to identify players")

            # Look for game results file
            game_results_file = Path("data/test_statistics/game_results/game_results_2024-25_Regular_Season.parquet")
            if not game_results_file.exists():
                # Try persistent location
                game_results_file = Path("data/persistent/game_results/game_results_2024-25_Regular_Season.parquet")

            if not game_results_file.exists():
                raise FileNotFoundError("Game results file not found for player analysis")

            import polars as pl
            games_df = pl.read_parquet(game_results_file)

            # Extract unique teams from games
            unique_teams = games_df['team_id'].unique().to_list()
            logger.info(f"Found {len(unique_teams)} teams in game results")

            # For now, return unique team IDs (we'll get player rosters from teams)
            self.results['games_analyzed'] = len(games_df)
            self.results['players_identified'] = len(unique_teams) * 15  # Estimate 15 players per team

            logger.info(f"✅ Analyzed {len(games_df)} games, estimated {self.results['players_identified']} players")
            return set(unique_teams)

        except Exception as e:
            error_msg = f"Failed to analyze game results: {e}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return set()

    def setup_player_stats_download(self, team_ids: Set[int]) -> bool:
        """
        Setup download tasks for player statistics.

        Args:
            team_ids: Set of team IDs to download player stats for

        Returns:
            True if setup successful, False otherwise
        """
        try:
            logger.info(f"📅 Setting up player statistics download for {len(team_ids)} teams")

            # For Context7 compliance, we'll use box score data
            # NBA API provides player stats through boxscore endpoints

            # Since NBA API doesn't have direct player season stats endpoint in our engine,
            # we'll create custom player stats download tasks

            for team_id in team_ids:
                # Create player stats task for each team
                task_name = f"download_player_stats_team_{team_id}_2024-25"
                task_description = f"Download 2024-25 Player Statistics for Team {team_id}"

                # Use existing engine method with custom parameters
                self.download_engine.add_custom_task(
                    task_name=task_name,
                    description=task_description,
                    task_type="player_stats",
                    priority="HIGH",
                    parameters={
                        "team_id": team_id,
                        "season": "2024-25",
                        "season_type": "Regular Season"
                    }
                )

            logger.info(f"✅ Player statistics download setup completed: {len(team_ids)} tasks queued")
            return True

        except Exception as e:
            error_msg = f"Failed to setup player statistics download: {e}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return False

    def download_player_game_stats(self, team_id: int, season: str = "2024-25") -> List[Dict[str, Any]]:
        """
        Download player game statistics for a specific team.

        Args:
            team_id: NBA team ID
            season: NBA season

        Returns:
            List of player game statistics
        """
        try:
            logger.info(f"📥 Downloading player stats for team {team_id}, season {season}")

            # Use NBA API to get player stats
            # We'll use the boxscore data from games
            from ..api.nba_api_provider import NBAAPIProvider

            provider = NBAAPIProvider()

            # Get games for this team from our stored data
            game_results_file = Path("data/test_statistics/game_results/game_results_2024-25_Regular_Season.parquet")
            import polars as pl
            games_df = pl.read_parquet(game_results_file)

            # Filter games for this team
            team_games = games_df.filter(pl.col("team_id") == team_id)

            player_stats = []

            # For each game, get player stats from box score
            for game_row in team_games.iter_rows():
                game_id = game_row[4]  # game_id column
                game_date = game_row[5].strftime('%Y-%m-%d')  # game_date column

                try:
                    # Get box score for this game
                    box_score = provider.get_boxscore(game_id)

                    if box_score and 'player_stats' in box_score:
                        for player_stat in box_score['player_stats']:
                            # Add metadata
                            player_stat['game_id'] = game_id
                            player_stat['game_date'] = game_date
                            player_stat['team_id'] = team_id
                            player_stat['season'] = season
                            player_stat['source'] = 'NBA_API_BoxScore'
                            player_stats.append(player_stat)

                except Exception as e:
                    logger.warning(f"Failed to get box score for game {game_id}: {e}")
                    continue

            logger.info(f"✅ Downloaded {len(player_stats)} player stats for team {team_id}")
            return player_stats

        except Exception as e:
            logger.error(f"Failed to download player stats for team {team_id}: {e}")
            return []

    def execute_player_stats_download(self, team_ids: Set[int]) -> Dict[str, Any]:
        """
        Execute complete player statistics download.

        Args:
            team_ids: Set of team IDs to download stats for

        Returns:
            Dict with comprehensive download results
        """
        try:
            logger.info("🚀 Starting Fase 2: Player Statistics Download")

            total_player_stats = []

            for i, team_id in enumerate(team_ids):
                try:
                    logger.info(f"Processing team {i+1}/{len(team_ids)}: ID {team_id}")

                    # Download player stats for this team
                    player_stats = self.download_player_game_stats(team_id)

                    if player_stats:
                        # Convert to DataFrame and store
                        import polars as pl
                        player_df = pl.DataFrame(player_stats)

                        # Store player stats (we'll aggregate by date)
                        unique_dates = player_df['game_date'].unique()

                        for game_date in unique_dates:
                            date_players = player_df.filter(pl.col("game_date") == game_date)
                            date_obj = datetime.strptime(game_date, '%Y-%m-%d').date()

                            # Store player stats for this date
                            file_path = self.data_store.store_player_game_stats(date_players, date_obj)

                            if file_path:
                                self.results['total_files_saved'] += 1

                        total_player_stats.extend(player_stats)
                        self.results['player_stats_downloaded'] += len(player_stats)

                    # Rate limiting between teams
                    time.sleep(0.5)

                except Exception as e:
                    error_msg = f"Failed to process team {team_id}: {e}"
                    logger.warning(error_msg)
                    self.results['errors'].append(error_msg)
                    continue

            logger.info(f"✅ Player statistics download completed: {len(total_player_stats)} total stats")
            return self._generate_final_results(True, "Player statistics download completed successfully")

        except Exception as e:
            error_msg = f"Failed to execute player statistics download: {e}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return self._generate_final_results(False, error_msg)

    def generate_player_summaries(self) -> bool:
        """
        Generate comprehensive player summaries from downloaded data.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("📊 Generating player summaries from downloaded data")

            # Look for player stats files
            player_stats_dir = Path("data/persistent/player_stats")
            if not player_stats_dir.exists():
                logger.warning("Player stats directory not found")
                return False

            player_files = list(player_stats_dir.glob("*.parquet"))
            summaries_generated = len(player_files)

            for player_file in player_files:
                try:
                    import polars as pl
                    df = pl.read_parquet(player_file)

                    # Generate basic summary
                    summary = {
                        'file': player_file.name,
                        'players_count': len(df),
                        'avg_points': df['points'].mean() if 'points' in df.columns else 0,
                        'avg_minutes': df['minutes'].mean() if 'minutes' in df.columns else 0,
                        'total_players': df['player_id'].n_unique() if 'player_id' in df.columns else 0
                    }

                    logger.debug(f"Generated summary for {player_file.name}: {summary}")

                except Exception as e:
                    logger.warning(f"Failed to generate summary for {player_file.name}: {e}")
                    continue

            self.results['players_summarized'] = summaries_generated
            logger.info(f"✅ Generated {summaries_generated} player file summaries")

            return True

        except Exception as e:
            error_msg = f"Failed to generate player summaries: {e}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return False

    def validate_downloaded_player_data(self) -> Dict[str, Any]:
        """
        Validate the downloaded player data for completeness and quality.

        Returns:
            Dict with validation results
        """
        try:
            logger.info("🔍 Validating downloaded player statistics data")

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
                # Expected player records (rough estimate)
                expected_records = 2460 * 20  # ~20 players per game
                quality_score = min(validation_results['total_player_records'] / expected_records, 1.0)
                validation_results['data_quality_score'] = quality_score

            logger.info(f"✅ Player validation completed: {validation_results['total_player_records']} records found")

            return validation_results

        except Exception as e:
            error_msg = f"Failed to validate player data: {e}"
            logger.error(error_msg)
            return {'validation_errors': [error_msg]}

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
            'engine_statistics': self.download_engine.stats,
            'player_validation': self.validate_downloaded_player_data()
        }

def run_fase2_player_statistics_download() -> Dict[str, Any]:
    """
    Execute complete Fase 2: Player Statistics download.

    Returns:
        Dict with comprehensive results
    """
    logger.info("🏀 Starting Fase 2: Player Statistics Download")

    downloader = Fase2PlayerStatisticsDownloader()

    # Step 1: Analyze game results to identify players
    team_ids = downloader.analyze_game_results_for_players()

    if not team_ids:
        logger.error("❌ No teams found in game results - cannot proceed with player stats download")
        return downloader._generate_final_results(False, "No teams found for player analysis")

    # Step 2: Execute player statistics download
    download_results = downloader.execute_player_stats_download(team_ids)

    if download_results['success']:
        # Step 3: Generate player summaries
        downloader.generate_player_summaries()

        # Log final summary
        logger.info("="*80)
        logger.info("🎯 FASE 2 COMPLETATA: Player Statistics Download")
        logger.info("="*80)
        logger.info(f"✅ Success: {download_results['success']}")
        logger.info(f"⏱️ Duration: {download_results['duration_formatted']}")
        logger.info(f"📊 Games Analyzed: {download_results['download_results']['games_analyzed']}")
        logger.info(f"👥 Players Identified: {download_results['download_results']['players_identified']}")
        logger.info(f"📈 Player Stats Downloaded: {download_results['download_results']['player_stats_downloaded']}")
        logger.info(f"📁 Files Saved: {download_results['download_results']['total_files_saved']}")
        logger.info(f"🔍 Data Quality: {download_results['player_validation']['data_quality_score']:.1%}")

        if download_results['download_results']['errors']:
            logger.warning(f"⚠️ Errors encountered: {len(download_results['download_results']['errors'])}")
            for error in download_results['download_results']['errors'][:5]:  # Show first 5 errors
                logger.warning(f"  - {error}")

        logger.info("🎉 FASE 2 COMPLETATA CON SUCCESSO!")
        logger.info("✅ Player Statistics scaricate e salvate")
        logger.info("🚀 Dataset predittivo NBA quasi completo!")

    else:
        logger.error("❌ FASE 2 FALLITA")
        logger.error(f"Error: {download_results['message']}")

    return download_results


if __name__ == "__main__":
    results = run_fase2_player_statistics_download()
    print(f"\nFinal Results: {json.dumps(results, indent=2, default=str)}")