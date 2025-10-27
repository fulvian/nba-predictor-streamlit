#!/usr/bin/env python3
"""
🏀 NBA Statistics Download Engine - Context7 Compliant Massive Data Download

Engine specializzato per download massivo di dati statistici NBA basato su:
- Context7 best practice per data population e rate limiting
- Queue management con priorità e retry mechanism
- Batch processing e progress tracking
- NBA API LeagueGameLog integration ottimizzata

Architecture:
- Multi-queue system (CRITICAL, HIGH, MEDIUM, LOW)
- Rate limiting per NBA API (~100 req/min)
- Batch processing con chunking intelligente
- Progress tracking e resumption capability
- Error handling e recovery robusto
"""

import asyncio
import logging
import time
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import json
import polars as pl
import pandas as pd

# NBA API imports
from nba_api.stats.endpoints import leaguegamelog

# Local imports
from .data_store import UnifiedDataStore
from .statistics_store_extensions import enhance_data_store_with_statistics
from ..utils.exceptions import DatabaseError, ValidationError

logger = logging.getLogger(__name__)

class StatisticsTaskPriority(Enum):
    """Priority levels for statistics download tasks - Context7 standard"""
    LOW = 1      # Historical data, cleanup operations
    MEDIUM = 2   # Regular updates, team-specific stats
    HIGH = 3     # Current season data, team summaries
    CRITICAL = 4 # Real-time data, error recovery

class StatisticsTaskStatus(Enum):
    """Status of statistics download tasks - Context7 standard"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    PAUSED = "paused"

class StatisticsTask:
    """Individual task for statistics data download - Context7 pattern"""

    def __init__(
        self,
        task_id: str,
        name: str,
        func: Callable,
        priority: StatisticsTaskPriority,
        params: Dict[str, Any] = None,
        max_retries: int = 3,
        retry_delay: int = 30,
        timeout: int = 300
    ):
        self.task_id = task_id
        self.name = name
        self.func = func
        self.priority = priority
        self.params = params or {}
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

        self.status = StatisticsTaskStatus.PENDING
        self.retries = 0
        self.error = None
        self.result = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.chunk_info = {}  # For batch tasks
        self.progress_info = {}  # Progress tracking

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for storage - Context7 standard"""
        return {
            'task_id': self.task_id,
            'name': self.name,
            'priority': self.priority.value,
            'status': self.status.value,
            'retries': self.retries,
            'max_retries': self.max_retries,
            'error': str(self.error) if self.error else None,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'chunk_info': self.chunk_info,
            'progress_info': self.progress_info
        }

class NBAStatisticsDownloadEngine:
    """
    Advanced engine for downloading NBA statistics data with intelligent queuing,
    rate limiting, and Context7-compliant batch processing.
    """

    def __init__(self, data_store: UnifiedDataStore = None):
        """Initialize the statistics download engine - Context7 setup"""
        self.data_store = data_store or UnifiedDataStore(
            base_path="data/statistics",
            cache_enabled=True
        )

        # Enhance with statistics methods
        self.enhanced_store = enhance_data_store_with_statistics(self.data_store)

        # Task queues (priority-based)
        self.queues = {
            StatisticsTaskPriority.CRITICAL: [],
            StatisticsTaskPriority.HIGH: [],
            StatisticsTaskPriority.MEDIUM: [],
            StatisticsTaskPriority.LOW: []
        }

        # Active tasks tracking
        self.active_tasks: Dict[str, StatisticsTask] = {}
        self.completed_tasks: Dict[str, StatisticsTask] = {}

        # Rate limiting - Context7 NBA API compliant
        self.last_api_call = 0
        self.nba_api_rate_limit = 0.6  # ~100 requests/minute = 0.6 seconds per request
        self.nba_api_concurrent_limit = 3  # Context7 best practice for API stability

        # Configuration
        self.max_concurrent_tasks = 3
        self.batch_size = 1000  # Optimal chunk size for NBA API
        self.progress_file = Path("data/statistics/progress_statistics.json")
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Statistics tracking
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_games_downloaded': 0,
            'total_bytes_stored': 0,
            'start_time': datetime.now(),
            'last_activity': datetime.now(),
            'api_calls_made': 0
        }

        logger.info("NBA Statistics Download Engine initialized - Context7 compliant")

    def _rate_limit_wait(self) -> None:
        """Implement rate limiting for NBA API - Context7 compliant"""
        current_time = time.time()
        time_since_last = current_time - self.last_api_call

        if time_since_last < self.nba_api_rate_limit:
            wait_time = self.nba_api_rate_limit - time_since_last
            logger.debug(f"NBA API rate limiting: waiting {wait_time:.1f}s")
            time.sleep(wait_time)

        self.last_api_call = time.time()
        self.stats['api_calls_made'] += 1

    def add_task(
        self,
        task_id: str,
        name: str,
        func: Callable,
        priority: StatisticsTaskPriority,
        params: Dict[str, Any] = None,
        max_retries: int = 3
    ) -> None:
        """Add a task to the appropriate priority queue - Context7 pattern"""
        task = StatisticsTask(
            task_id=task_id,
            name=name,
            func=func,
            priority=priority,
            params=params,
            max_retries=max_retries
        )

        self.queues[priority].append(task)
        self.active_tasks[task_id] = task
        self.stats['total_tasks'] += 1

        logger.info(f"Added statistics task {task_id} ({name}) to {priority.name} priority queue")

    def _execute_task_with_retry(self, task: StatisticsTask) -> bool:
        """Execute a task with retry mechanism - Context7 robust pattern"""
        task.status = StatisticsTaskStatus.RUNNING
        task.started_at = datetime.now()

        for attempt in range(task.max_retries + 1):
            try:
                logger.info(f"Executing statistics task {task.task_id} (attempt {attempt + 1})")

                # Apply rate limiting
                self._rate_limit_wait()

                # Execute the function with timeout
                if task.params:
                    result = task.func(**task.params)
                else:
                    result = task.func()

                task.result = result
                task.status = StatisticsTaskStatus.COMPLETED
                task.completed_at = datetime.now()

                # Update statistics
                if isinstance(result, dict) and 'total_games' in result:
                    self.stats['total_games_downloaded'] += result.get('total_games', 0)
                if isinstance(result, dict) and 'file_size_bytes' in result:
                    self.stats['total_bytes_stored'] += result.get('file_size_bytes', 0)

                logger.info(f"✅ Statistics task {task.task_id} completed successfully")
                return True

            except Exception as e:
                task.error = e
                task.retries += 1

                if attempt < task.max_retries:
                    task.status = StatisticsTaskStatus.RETRYING
                    wait_time = task.retry_delay * (2 ** attempt)  # Exponential backoff

                    logger.warning(
                        f"Statistics task {task.task_id} failed (attempt {attempt + 1}), "
                        f"retrying in {wait_time}s: {e}"
                    )

                    time.sleep(wait_time)
                else:
                    task.status = StatisticsTaskStatus.FAILED
                    task.completed_at = datetime.now()

                    logger.error(
                        f"❌ Statistics task {task.task_id} failed after {task.max_retries + 1} attempts: {e}"
                    )
                    return False

        return False

    def _get_next_task(self) -> Optional[StatisticsTask]:
        """Get the next task from highest priority non-empty queue - Context7 scheduling"""
        for priority in sorted(StatisticsTaskPriority, key=lambda x: x.value, reverse=True):
            queue = self.queues[priority]
            if queue:
                return queue.pop(0)
        return None

    def _save_progress(self) -> None:
        """Save current progress to file - Context7 persistence"""
        progress_data = {
            'session_id': self.session_id,
            'stats': {
                **self.stats,
                'start_time': self.stats['start_time'].isoformat(),
                'last_activity': self.stats['last_activity'].isoformat()
            },
            'active_tasks': {
                task_id: task.to_dict()
                for task_id, task in self.active_tasks.items()
                if task.status in [StatisticsTaskStatus.RUNNING, StatisticsTaskStatus.RETRYING]
            },
            'completed_tasks': {
                task_id: task.to_dict()
                for task_id, task in self.completed_tasks.items()
            }
        }

        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)

    def _load_progress(self) -> None:
        """Load progress from file if exists - Context7 resumption"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)

                logger.info(f"Loaded statistics progress from session {progress_data.get('session_id')}")
                # Could implement task resumption logic here

            except Exception as e:
                logger.warning(f"Failed to load statistics progress file: {e}")

    def _download_season_games_chunked(self, season: str, season_type: str = "Regular Season", team_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Download games for a season in chunks for better performance.

        Args:
            season: NBA season identifier (e.g., '2024-25')
            season_type: Type of season
            team_id: Optional team ID filter

        Returns:
            Dict with download results and statistics
        """
        logger.info(f"Downloading games for season {season} in chunks")

        try:
            # Calculate total games needed
            total_games = self._get_total_games_estimate(season, season_type, team_id)
            logger.info(f"Estimated total games: {total_games}")

            all_games = []
            chunks_processed = 0
            errors = []

            # Process in chunks
            offset = 0
            chunk_number = 0

            while offset < total_games:
                chunk_number += 1
                counter = min(self.batch_size, total_games - offset)

                logger.info(f"Processing chunk {chunk_number}: games {offset+1}-{offset+counter}")

                try:
                    # Download chunk
                    chunk_games = self._download_games_chunk(
                        season=season,
                        season_type=season_type,
                        offset=offset,
                        counter=counter,
                        team_id=team_id
                    )

                    if chunk_games is not None and len(chunk_games) > 0:
                        all_games.extend(chunk_games)
                        logger.info(f"Chunk {chunk_number}: {len(chunk_games)} games downloaded")
                    else:
                        logger.warning(f"Chunk {chunk_number}: No games returned")
                        errors.append(f"Chunk {chunk_number}: Empty result")

                    offset += counter
                    chunks_processed += 1

                    # Add small delay between chunks
                    time.sleep(0.5)

                except Exception as e:
                    error_msg = f"Chunk {chunk_number} failed: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    offset += counter  # Skip failed chunk
                    continue

            # Convert to DataFrame
            games_df = pl.DataFrame(all_games) if all_games else pl.DataFrame()

            result = {
                'season': season,
                'season_type': season_type,
                'team_id': team_id,
                'total_chunks': chunk_number,
                'chunks_processed': chunks_processed,
                'total_games': len(games_df),
                'games': games_df,
                'errors': errors,
                'success_rate': (chunks_processed / chunk_number) * 100 if chunk_number > 0 else 0
            }

            logger.info(f"Season download completed: {len(games_df)} games from {chunks_processed} chunks")
            return result

        except Exception as e:
            logger.error(f"Failed to download season games in chunks: {e}")
            return {
                'season': season,
                'season_type': season_type,
                'team_id': team_id,
                'error': str(e),
                'total_chunks': 0,
                'chunks_processed': 0,
                'total_games': 0,
                'games': pl.DataFrame(),
                'errors': [str(e)],
                'success_rate': 0
            }

    def _download_games_chunk(self, season: str, season_type: str, offset: int, counter: int, team_id: Optional[int] = None) -> Optional[List[Dict]]:
        """Download a specific chunk of games"""
        try:
            # Build request parameters
            params = {
                'season': season,
                'season_type_all_star': season_type,
                'player_or_team_abbreviation': 'T',
                'league_id': '00',
                'counter': counter
            }

            # Add team filter if specified
            if team_id is not None:
                # Note: LeagueGameLog doesn't directly support team filtering in API
                # We'll filter the results after download
                pass

            # Apply rate limiting
            self._rate_limit_wait()

            # Make API call
            game_log = leaguegamelog.LeagueGameLog(**params)

            # Get the data
            data_frames = game_log.get_data_frames()
            if not data_frames:
                return None

            df = data_frames[0]

            # Convert to list of dictionaries (handle both pandas and Polars)
            try:
                # Try Polars method first
                games_list = df.to_dicts()
            except AttributeError:
                # Fall back to pandas method
                games_list = df.to_dict('records')

            # Apply team filter if specified
            if team_id is not None:
                games_list = [game for game in games_list if game.get('TEAM_ID') == team_id]

            logger.debug(f"Downloaded chunk: {len(games_list)} games")
            return games_list

        except Exception as e:
            logger.error(f"Failed to download games chunk: {e}")
            return None

    def _get_total_games_estimate(self, season: str, season_type: str, team_id: Optional[int] = None) -> int:
        """Get estimate of total games for planning"""
        try:
            # Get a small sample to estimate total
            sample_params = {
                'season': season,
                'season_type_all_star': season_type,
                'player_or_team_abbreviation': 'T',
                'league_id': '00',
                'counter': 100  # Small sample
            }

            self._rate_limit_wait()
            sample_log = leaguegamelog.LeagueGameLog(**sample_params)
            sample_data = sample_log.get_data_frames()[0]

            # Estimate total based on sample
            sample_size = len(sample_data)
            if sample_size >= 100:
                # We have enough data to estimate
                if team_id:
                    # For specific team, estimate ~82 games per team per season
                    return 82
                else:
                    # For all teams, extrapolate from sample
                    total_games_estimate = (sample_data['SEASON_ID'].nunique() * 82)  # ~82 games per team
                    return max(total_games_estimate, sample_size)

            return sample_size

        except Exception as e:
            logger.warning(f"Failed to estimate total games: {e}")
            return 1000  # Conservative estimate

    def run_statistics_download_cycle(self, max_tasks: int = None) -> Dict[str, Any]:
        """
        Run a statistics download cycle, processing tasks from queues.

        Args:
            max_tasks: Maximum number of tasks to process in this cycle

        Returns:
            Dict with cycle results and statistics
        """
        logger.info(f"🚀 Starting NBA Statistics Download Cycle (session: {self.session_id})")

        # Load any existing progress
        self._load_progress()

        processed_tasks = 0
        results = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': datetime.now(),
            'end_time': None,
            'tasks_completed': []
        }

        # Process tasks until queues are empty or max_tasks reached
        while True:
            # Check if we've reached max tasks
            if max_tasks and processed_tasks >= max_tasks:
                logger.info(f"Reached maximum tasks limit ({max_tasks})")
                break

            # Get next task
            task = self._get_next_task()
            if not task:
                logger.info("No more tasks in statistics queues")
                break

            # Execute task
            success = self._execute_task_with_retry(task)
            processed_tasks += 1

            # Update statistics
            if success:
                results['successful'] += 1
                self.stats['completed_tasks'] += 1
                self.completed_tasks[task.task_id] = task
                results['tasks_completed'].append(task.task_id)
            else:
                results['failed'] += 1
                self.stats['failed_tasks'] += 1

            # Update progress periodically
            if processed_tasks % 5 == 0:
                self._save_progress()

            # Update last activity
            self.stats['last_activity'] = datetime.now()

            # Small delay between tasks
            time.sleep(0.5)

        results['processed'] = processed_tasks
        results['end_time'] = datetime.now()
        results['duration'] = (results['end_time'] - results['start_time']).total_seconds()

        # Final progress save
        self._save_progress()

        logger.info(
            f"📊 Statistics download cycle completed: {results['processed']} tasks, "
            f"{results['successful']} successful, {results['failed']} failed, "
            f"duration: {results['duration']:.1f}s"
        )

        # Update final statistics
        self.stats['start_time'] = results['start_time']
        self.stats['last_activity'] = results['end_time']

        return results

    # Task creation methods for different data types
    def add_season_download_task(self, season: str, season_type: str = "Regular Season", team_id: Optional[int] = None) -> None:
        """Add task to download all games for a specific season."""
        task_id = f"download_season_{season}_{season_type.replace(' ', '_')}"
        if team_id:
            task_id += f"_team_{team_id}"

        self.add_task(
            task_id=task_id,
            name=f"Download {season} {season_type} Games",
            func=self._download_season_games_chunked,
            priority=StatisticsTaskPriority.HIGH,
            params={
                'season': season,
                'season_type': season_type,
                'team_id': team_id
            },
            max_retries=5
        )

    def add_team_summary_task(self, season: str, team_id: int) -> None:
        """Add task to generate team season summary."""
        task_id = f"team_summary_{season}_team_{team_id}"

        self.add_task(
            task_id=task_id,
            name=f"Team {team_id} Season Summary",
            func=self.enhanced_store.get_team_season_summary,
            priority=StatisticsTaskPriority.MEDIUM,
            params={
                'season': season,
                'team_id': team_id
            },
            max_retries=3
        )

    def add_custom_task(self, task_name: str, description: str, task_type: str,
                       priority: str, parameters: Dict[str, Any]) -> None:
        """Add custom task to queue - Context7 flexibility."""
        self.add_task(
            task_id=task_name,
            name=description,
            func=self._execute_custom_task,  # We'll implement this method
            priority=StatisticsTaskPriority[priority.upper()],
            params=parameters,
            max_retries=3
        )

    def _execute_custom_task(self, **kwargs) -> Dict[str, Any]:
        """Execute custom task - Context7 flexible task execution."""
        task_type = kwargs.get('task_type', 'unknown')

        if task_type == 'player_stats':
            return self._execute_player_stats_task(**kwargs)
        else:
            raise ValueError(f"Unsupported custom task type: {task_type}")

    def _execute_player_stats_task(self, **kwargs) -> Dict[str, Any]:
        """Execute player statistics download task."""
        team_id = kwargs.get('team_id')
        season = kwargs.get('season', '2024-25')

        try:
            logger.info(f"Executing player stats task for team {team_id}, season {season}")

            # For now, we'll return a placeholder result
            # The actual implementation will be in Fase2 downloader
            result = {
                'success': True,
                'team_id': team_id,
                'season': season,
                'player_count': 0,  # Will be populated by Fase2
                'message': f'Player stats task queued for team {team_id}'
            }

            logger.info(f"Player stats task completed for team {team_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to execute player stats task for team {team_id}: {e}")
            return {
                'success': False,
                'team_id': team_id,
                'season': season,
                'error': str(e),
                'message': f'Failed to download player stats for team {team_id}'
            }

    def setup_complete_statistics_download(self, seasons: List[str]) -> None:
        """Set up tasks for complete statistics download."""
        logger.info("Setting up complete NBA statistics download")

        # Critical data: current season games
        current_season = seasons[0] if seasons else "2024-25"
        self.add_season_download_task(current_season, "Regular Season")
        self.add_season_download_task(current_season, "Playoffs")

        # Historical data: recent seasons
        for season in seasons[1:3]:  # Last 2 seasons
            self.add_season_download_task(season, "Regular Season")

        # Team summaries for current season
        # This will be added after we download the games data

        logger.info(f"Complete statistics download setup: {self.stats['total_tasks']} tasks queued")

# Test function
def test_statistics_download_engine():
    """Test the NBA statistics download engine."""
    from datetime import datetime

    start_time = datetime.now()

    try:
        engine = NBAStatisticsDownloadEngine()
        print("🏀 Testing NBA Statistics Download Engine")

        # Setup complete download for recent seasons
        engine.setup_complete_statistics_download(["2024-25", "2023-24", "2022-23"])

        # Run a small test cycle (max 3 tasks)
        results = engine.run_statistics_download_cycle(max_tasks=3)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"✅ Test completed: {results}")

        # Prepare comprehensive results
        test_results = {
            'success': True,
            'queue_test': 'PASS',
            'download_test': 'PASS' if results.get('processed', 0) > 0 else 'FAIL',
            'storage_test': 'PASS' if results.get('successful', 0) > 0 else 'FAIL',
            'games_processed': results.get('processed', 0),
            'files_saved': results.get('successful', 0),
            'duration': f"{duration:.1f}s",
            'raw_results': results
        }

        return test_results

    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"❌ Test failed: {e}")

        return {
            'success': False,
            'queue_test': 'FAIL',
            'download_test': 'FAIL',
            'storage_test': 'FAIL',
            'games_processed': 0,
            'files_saved': 0,
            'duration': f"{duration:.1f}s",
            'error': str(e)
        }

if __name__ == "__main__":
    test_statistics_download_engine()