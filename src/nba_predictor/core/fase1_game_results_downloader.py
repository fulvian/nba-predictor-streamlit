#!/usr/bin/env python3
"""
🏀 Fase 1: Game Results & Box Scores Download - Context7 Implementation

Download completo dei risultati delle partite NBA usando:
- NBA Statistics Download Engine con rate limiting
- Context7 compliant validation e storage
- 2,460+ games dalla stagione 2024-25
- Progress tracking e resumption capability
- Advanced metrics calculation

Fase 1 del NBA Predictive Analytics System:
1. Setup engine con download task per stagione 2024-25
2. Download chunked con rate limiting NBA API
3. Storage in data store persistente con validation
4. Generation di team summaries e analytics
5. Progress reporting e error handling
"""

import logging
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Any, Optional
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

class Fase1GameResultsDownloader:
    """
    Fase 1 implementation for downloading complete NBA game results.
    Context7-compliant massive data download with validation.
    """

    def __init__(self):
        """Initialize Fase 1 downloader."""
        logger.info("🏀 Initializing Fase 1: Game Results & Box Scores Downloader")

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
            'seasons_processed': [],
            'total_games_downloaded': 0,
            'total_files_saved': 0,
            'errors': [],
            'teams_summarized': 0
        }

    def setup_2024_25_season_download(self) -> bool:
        """
        Setup download tasks for complete 2024-25 NBA season.

        Returns:
            True if setup successful, False otherwise
        """
        try:
            logger.info("📅 Setting up 2024-25 NBA season download")

            # Configure download for 2024-25 season
            seasons = ["2024-25"]
            self.download_engine.setup_complete_statistics_download(seasons)

            logger.info("✅ 2024-25 season download setup completed")
            return True

        except Exception as e:
            error_msg = f"Failed to setup 2024-25 season download: {e}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return False

    def execute_game_results_download(self) -> Dict[str, Any]:
        """
        Execute complete game results download for 2024-25 season.

        Returns:
            Dict with comprehensive download results
        """
        try:
            logger.info("🚀 Starting Fase 1: Game Results Download")

            # Setup download tasks
            if not self.setup_2024_25_season_download():
                return self._generate_final_results(False, "Setup failed")

            # Execute download with progress tracking
            logger.info("⬇️ Executing NBA API download with Context7 patterns")

            # Monitor progress
            last_stats = {}
            consecutive_no_progress = 0
            max_no_progress = 10  # Stop after 10 consecutive checks with no progress

            while True:
                # Get current statistics
                current_stats = self.download_engine.stats

                # Check for progress
                if current_stats.get('completed_tasks', 0) > last_stats.get('completed_tasks', 0):
                    consecutive_no_progress = 0
                    progress_pct = (current_stats['completed_tasks'] / current_stats['total_tasks']) * 100
                    logger.info(
                        f"📈 Progress: {current_stats['completed_tasks']}/{current_stats['total_tasks']} "
                        f"tasks ({progress_pct:.1f}%) - "
                        f"Success: {current_stats['completed_tasks']}, Failed: {current_stats['failed_tasks']}"
                    )
                else:
                    consecutive_no_progress += 1
                    logger.debug(f"No progress for {consecutive_no_progress} consecutive checks")

                # Run one download cycle
                cycle_results = self.download_engine.run_statistics_download_cycle(max_tasks=1)

                # Update our tracking results
                if cycle_results.get('processed', 0) > 0:
                    self.results['total_games_downloaded'] += cycle_results['processed']
                    self.results['total_files_saved'] += cycle_results['successful']

                last_stats = current_stats.copy()

                # Check completion or timeout
                if (current_stats['completed_tasks'] >= current_stats['total_tasks'] or
                    consecutive_no_progress >= max_no_progress):
                    break

                # Rate limiting between cycles
                time.sleep(1.0)

            # Store processed seasons
            self.results['seasons_processed'] = ["2024-25"]

            logger.info(f"✅ Download completed: {self.results['total_games_downloaded']} games processed")

            return self._generate_final_results(True, "Download completed successfully")

        except Exception as e:
            error_msg = f"Failed to execute game results download: {e}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return self._generate_final_results(False, error_msg)

    def generate_team_summaries(self) -> bool:
        """
        Generate comprehensive team summaries for downloaded data.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("📊 Generating team summaries for 2024-25 season")

            # Get all teams from stored data
            teams_file = Path("data/persistent/teams/teams_2025-10-27.parquet")
            if not teams_file.exists():
                logger.warning("Teams data file not found, skipping team summaries")
                return False

            import polars as pl
            teams_df = pl.read_parquet(teams_file)

            summaries_generated = 0

            for team_row in teams_df.iter_rows():
                team_id = team_row[0]  # team_id is first column
                team_name = team_row[1]  # team_name is second column

                try:
                    # Generate season summary
                    summary = self.data_store.get_team_season_summary("2024-25", team_id)

                    if 'error' not in summary:
                        summaries_generated += 1
                        logger.debug(f"✅ Generated summary for {team_name} (ID: {team_id})")
                    else:
                        logger.warning(f"❌ Failed to generate summary for {team_name}: {summary['error']}")

                except Exception as e:
                    logger.warning(f"❌ Error generating summary for {team_name}: {e}")
                    continue

            self.results['teams_summarized'] = summaries_generated
            logger.info(f"✅ Generated {summaries_generated} team summaries")

            return True

        except Exception as e:
            error_msg = f"Failed to generate team summaries: {e}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return False

    def validate_downloaded_data(self) -> Dict[str, Any]:
        """
        Validate the downloaded data for completeness and quality.

        Returns:
            Dict with validation results
        """
        try:
            logger.info("🔍 Validating downloaded game results data")

            validation_results = {
                'seasons_found': 0,
                'total_games': 0,
                'teams_with_data': 0,
                'data_quality_score': 0.0,
                'validation_errors': []
            }

            # Check for game results data
            game_results_dir = Path("data/persistent/game_results")
            if game_results_dir.exists():
                game_files = list(game_results_dir.glob("*.parquet"))
                validation_results['seasons_found'] = len(game_files)

                total_games = 0
                for game_file in game_files:
                    try:
                        import polars as pl
                        df = pl.read_parquet(game_file)
                        total_games += len(df)
                    except Exception as e:
                        validation_results['validation_errors'].append(f"Error reading {game_file}: {e}")

                validation_results['total_games'] = total_games

            # Calculate data quality score
            if validation_results['total_games'] > 0:
                # Expected games for 2024-25 season (approximate)
                expected_games = 2460
                quality_score = min(validation_results['total_games'] / expected_games, 1.0)
                validation_results['data_quality_score'] = quality_score

            logger.info(f"✅ Validation completed: {validation_results['total_games']} games found")

            return validation_results

        except Exception as e:
            error_msg = f"Failed to validate downloaded data: {e}"
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
            'data_validation': self.validate_downloaded_data()
        }

def run_fase1_game_results_download() -> Dict[str, Any]:
    """
    Execute complete Fase 1: Game Results & Box Scores download.

    Returns:
        Dict with comprehensive results
    """
    logger.info("🏀 Starting Fase 1: Game Results & Box Scores Download")

    downloader = Fase1GameResultsDownloader()

    # Execute download
    download_results = downloader.execute_game_results_download()

    if download_results['success']:
        # Generate team summaries
        downloader.generate_team_summaries()

        # Log final summary
        logger.info("="*80)
        logger.info("🎯 FASE 1 COMPLETATA: Game Results & Box Scores Download")
        logger.info("="*80)
        logger.info(f"✅ Success: {download_results['success']}")
        logger.info(f"⏱️ Duration: {download_results['duration_formatted']}")
        logger.info(f"📊 Games Downloaded: {download_results['download_results']['total_games_downloaded']}")
        logger.info(f"📁 Files Saved: {download_results['download_results']['total_files_saved']}")
        logger.info(f"📈 Teams Summarized: {download_results['download_results']['teams_summarized']}")
        logger.info(f"🔍 Data Quality: {download_results['data_validation']['data_quality_score']:.1%}")

        if download_results['download_results']['errors']:
            logger.warning(f"⚠️ Errors encountered: {len(download_results['download_results']['errors'])}")
            for error in download_results['download_results']['errors']:
                logger.warning(f"  - {error}")

        logger.info("🎉 FASE 1 COMPLETATA CON SUCCESSO!")
        logger.info("✅ Sistema pronto per Fase 2: Player Statistics Download")

    else:
        logger.error("❌ FASE 1 FALLITA")
        logger.error(f"Error: {download_results['message']}")

    return download_results


if __name__ == "__main__":
    results = run_fase1_game_results_download()
    print(f"\nFinal Results: {json.dumps(results, indent=2, default=str)}")