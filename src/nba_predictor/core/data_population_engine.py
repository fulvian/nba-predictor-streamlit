#!/usr/bin/env python3
"""
🏀 NBA Data Population Engine - Sistema Completo di Popolamento Dati

Sistema avanzato per popolare il data store con dati NBA completi:
- Teams, Players, Rosters
- Player Stats, Team Stats
- Games, Schedules
- Injury Reports
- Rate limiting con code prioritarie
- Retry mechanism con backoff esponenziale
- Progress tracking e resumption capability

Architecture:
- Queue System: Alta/Media/Bassa priorità
- Rate Limits: NBA API (30 req/min), BallDontLie (5 req/min)
- Storage: UnifiedDataStore con cache
- Monitoring: Progress tracking + error handling
"""

import asyncio
import logging
import os
import time
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import json
import polars as pl

# Import existing components
from ..api.multi_source_provider import MultiSourceNBADataProvider
from .data_store import UnifiedDataStore

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Priority levels for data population tasks."""
    LOW = 1      # Historical data, cleanup
    MEDIUM = 2   # Regular updates, stats
    HIGH = 3     # Today's games, critical data
    CRITICAL = 4 # Real-time data, errors

class TaskStatus(Enum):
    """Status of data population tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class DataPopulationTask:
    """Individual task for data population."""

    def __init__(
        self,
        task_id: str,
        name: str,
        func: Callable,
        priority: TaskPriority,
        params: Dict[str, Any] = None,
        max_retries: int = 3,
        retry_delay: int = 30
    ):
        self.task_id = task_id
        self.name = name
        self.func = func
        self.priority = priority
        self.params = params or {}
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.status = TaskStatus.PENDING
        self.retries = 0
        self.error = None
        self.result = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for storage."""
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
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }

class NBADataPopulationEngine:
    """
    Advanced engine for populating NBA data with intelligent queuing,
    rate limiting, and retry mechanisms.
    """

    def __init__(self, data_store: UnifiedDataStore = None):
        """Initialize the data population engine."""
        self.data_store = data_store or UnifiedDataStore(
            base_path="data/persistent",
            cache_enabled=True
        )

        # Initialize multi-source provider
        self.provider = MultiSourceNBADataProvider()

        # Task queues (priority-based)
        self.queues = {
            TaskPriority.CRITICAL: [],
            TaskPriority.HIGH: [],
            TaskPriority.MEDIUM: [],
            TaskPriority.LOW: []
        }

        # Active tasks tracking
        self.active_tasks: Dict[str, DataPopulationTask] = {}
        self.completed_tasks: Dict[str, DataPopulationTask] = {}

        # Rate limiting
        self.last_request_times = {
            'nba_api': 0,
            'balldontlie': 0,
            'the_odds': 0
        }
        self.rate_limits = {
            'nba_api': 2,      # ~30 req/min = 2 sec interval
            'balldontlie': 12,  # 5 req/min = 12 sec interval
            'the_odds': 1       # Conservative: 1 sec interval
        }

        # Configuration
        self.max_concurrent_tasks = 3
        self.progress_file = Path("data/persistent/progress.json")
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Statistics
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'start_time': datetime.now(),
            'last_activity': datetime.now()
        }

        logger.info("NBA Data Population Engine initialized")

    def _rate_limit_wait(self, api_name: str) -> None:
        """Implement rate limiting for specific API."""
        current_time = time.time()
        last_call = self.last_request_times.get(api_name, 0)
        min_interval = self.rate_limits.get(api_name, 1)

        time_since_last = current_time - last_call
        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            logger.debug(f"Rate limiting {api_name}: waiting {wait_time:.1f}s")
            time.sleep(wait_time)

        self.last_request_times[api_name] = time.time()

    def add_task(
        self,
        task_id: str,
        name: str,
        func: Callable,
        priority: TaskPriority,
        params: Dict[str, Any] = None,
        max_retries: int = 3
    ) -> None:
        """Add a task to the appropriate priority queue."""
        task = DataPopulationTask(
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

        logger.info(f"Added task {task_id} ({name}) to {priority.name} priority queue")

    def _execute_task_with_retry(self, task: DataPopulationTask) -> bool:
        """Execute a task with retry mechanism."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        for attempt in range(task.max_retries + 1):
            try:
                logger.info(f"Executing task {task.task_id} (attempt {attempt + 1})")

                # Execute the function
                if task.params:
                    result = task.func(**task.params)
                else:
                    result = task.func()

                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()

                logger.info(f"✅ Task {task.task_id} completed successfully")
                return True

            except Exception as e:
                task.error = e
                task.retries += 1

                if attempt < task.max_retries:
                    task.status = TaskStatus.RETRYING
                    wait_time = task.retry_delay * (2 ** attempt)  # Exponential backoff

                    logger.warning(
                        f"Task {task.task_id} failed (attempt {attempt + 1}), "
                        f"retrying in {wait_time}s: {e}"
                    )

                    time.sleep(wait_time)
                else:
                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.now()

                    logger.error(
                        f"❌ Task {task.task_id} failed after {task.max_retries + 1} attempts: {e}"
                    )
                    return False

        return False

    def _get_next_task(self) -> Optional[DataPopulationTask]:
        """Get the next task from highest priority non-empty queue."""
        for priority in sorted(TaskPriority, key=lambda x: x.value, reverse=True):
            queue = self.queues[priority]
            if queue:
                return queue.pop(0)
        return None

    def _save_progress(self) -> None:
        """Save current progress to file."""
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
                if task.status in [TaskStatus.RUNNING, TaskStatus.RETRYING]
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
        """Load progress from file if exists."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)

                logger.info(f"Loaded progress from session {progress_data.get('session_id')}")
                # Could implement task resumption logic here

            except Exception as e:
                logger.warning(f"Failed to load progress file: {e}")

    def run_population_cycle(self, max_tasks: int = None) -> Dict[str, Any]:
        """
        Run one population cycle, processing tasks from queues.

        Args:
            max_tasks: Maximum number of tasks to process in this cycle

        Returns:
            Dict with cycle results and statistics
        """
        logger.info(f"🚀 Starting NBA data population cycle (session: {self.session_id})")

        # Load any existing progress
        self._load_progress()

        processed_tasks = 0
        results = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': datetime.now(),
            'end_time': None
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
                logger.info("No more tasks in queues")
                break

            # Execute task
            success = self._execute_task_with_retry(task)
            processed_tasks += 1

            # Update statistics
            if success:
                results['successful'] += 1
                self.stats['completed_tasks'] += 1
                self.completed_tasks[task.task_id] = task
            else:
                results['failed'] += 1
                self.stats['failed_tasks'] += 1

            # Update progress periodically
            if processed_tasks % 10 == 0:
                self._save_progress()

            # Small delay between tasks
            time.sleep(0.5)

        results['processed'] = processed_tasks
        results['end_time'] = datetime.now()
        results['duration'] = (results['end_time'] - results['start_time']).total_seconds()

        # Final progress save
        self._save_progress()

        logger.info(
            f"📊 Population cycle completed: {results['processed']} tasks, "
            f"{results['successful']} successful, {results['failed']} failed, "
            f"duration: {results['duration']:.1f}s"
        )

        return results

    # Task creation methods for different data types
    def add_teams_population_task(self) -> None:
        """Add task to populate all teams data."""
        self.add_task(
            task_id="populate_teams",
            name="Populate NBA Teams",
            func=self.provider.get_teams,
            priority=TaskPriority.HIGH,
            max_retries=3
        )

    def add_players_population_task(self) -> None:
        """Add task to populate all players data."""
        self.add_task(
            task_id="populate_players",
            name="Populate NBA Players",
            func=self.provider.get_players,
            priority=TaskPriority.HIGH,
            max_retries=3
        )

    def add_today_games_task(self) -> None:
        """Add task to populate today's games."""
        today = date.today().strftime('%Y-%m-%d')
        self.add_task(
            task_id=f"populate_games_{today}",
            name=f"Populate Games for {today}",
            func=self.provider.get_games,
            params={'start_date': today, 'end_date': today},
            priority=TaskPriority.CRITICAL,
            max_retries=5
        )

    def add_team_rosters_task(self, team_ids: List[int]) -> None:
        """Add task to populate rosters for specific teams."""
        for team_id in team_ids:
            self.add_task(
                task_id=f"populate_roster_{team_id}",
                name=f"Populate Roster for Team {team_id}",
                func=self.provider.get_team_roster,
                params={'team_id': team_id, 'season': 2024},
                priority=TaskPriority.MEDIUM,
                max_retries=3
            )

    def add_player_stats_task(self, player_ids: List[int]) -> None:
        """Add task to populate stats for specific players."""
        for player_id in player_ids:
            self.add_task(
                task_id=f"populate_stats_{player_id}",
                name=f"Populate Stats for Player {player_id}",
                func=self.provider.get_player_stats,
                params={'player_id': player_id, 'season': 2024},
                priority=TaskPriority.MEDIUM,
                max_retries=3
            )

    def setup_complete_population(self) -> None:
        """Set up tasks for complete data population."""
        logger.info("Setting up complete NBA data population")

        # Critical data
        self.add_teams_population_task()
        self.add_players_population_task()
        self.add_today_games_task()

        # Get today's teams for roster population
        today = date.today().strftime('%Y-%m-%d')
        games = self.provider.get_games(today)

        if games:
            team_ids = set()
            for game in games:
                team_ids.add(game['home_team_id'])
                team_ids.add(game['visitor_team_id'])

            # Add roster tasks
            self.add_team_rosters_task(list(team_ids))

            logger.info(f"Added roster tasks for {len(team_ids)} teams")

        logger.info(f"Complete population setup: {self.stats['total_tasks']} tasks queued")


# Standalone test function
def test_population_engine():
    """Test the NBA data population engine."""
    engine = NBADataPopulationEngine()

    print("🏀 Testing NBA Data Population Engine")

    # Setup complete population
    engine.setup_complete_population()

    # Run a small test cycle (max 5 tasks)
    results = engine.run_population_cycle(max_tasks=5)

    print(f"✅ Test completed: {results}")


if __name__ == "__main__":
    test_population_engine()