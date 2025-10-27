#!/usr/bin/env python3
"""
🏀 Complete NBA Data Populator - Test Completo di Popolamento Dati

Test completo del sistema di popolamento dati NBA:
1. Setup e inizializzazione data store
2. Popolamento teams (30 NBA teams)
3. Popolamento players (5000+ players)
4. Popolamento games oggi (11 games)
5. Popolamento rosters squadre oggi
6. Salvataggio tutto nel data store persistente
7. Verifica dati salvati

Questo script testa l'intero flusso di lavoro per preparare
il sistema predittivo completo.
"""

import logging
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import polars as pl

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import components
from .data_store import UnifiedDataStore
from .data_store_extensions import enhance_data_store
from .data_population_engine import NBADataPopulationEngine, TaskPriority
from ..api.multi_source_provider import MultiSourceNBADataProvider

class CompleteDataPopulator:
    """
    Complete data population test for NBA system.
    """

    def __init__(self):
        """Initialize the complete data populator."""
        logger.info("🏀 Initializing Complete NBA Data Populator")

        # Initialize data store
        base_store = UnifiedDataStore(
            base_path="data/persistent",
            cache_enabled=True
        )
        base_store.initialize()
        # Enhance with NBA-specific methods
        self.data_store = enhance_data_store(base_store)

        # Initialize population engine
        self.population_engine = NBADataPopulationEngine(data_store=self.data_store)

        # Initialize provider
        self.provider = MultiSourceNBADataProvider()

        # Statistics tracking
        self.start_time = datetime.now()
        self.results = {
            'teams': 0,
            'players': 0,
            'games': 0,
            'rosters': 0,
            'files_saved': 0,
            'errors': []
        }

    def test_data_store_initialization(self) -> bool:
        """Test data store initialization and cleanup."""
        try:
            logger.info("📁 Testing data store initialization...")
            self.data_store.initialize()
            logger.info("✅ Data store initialized successfully")
            return True

        except Exception as e:
            error_msg = f"Failed to initialize data store: {e}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return False

    def populate_teams_data(self) -> bool:
        """Populate and save teams data."""
        try:
            logger.info("🏢 Populating teams data...")

            # Get teams from provider
            teams = self.provider.get_teams()
            self.results['teams'] = len(teams)

            if not teams:
                logger.warning("No teams data retrieved")
                return False

            # Convert to DataFrame
            teams_df = pl.DataFrame(teams)

            # Save to data store
            today_str = date.today().strftime('%Y-%m-%d')
            file_path = self.data_store.store_teams_data(teams_df, today_str)

            logger.info(f"✅ Teams data saved: {len(teams)} teams -> {file_path}")
            self.results['files_saved'] += 1
            return True

        except Exception as e:
            error_msg = f"Failed to populate teams data: {e}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return False

    def populate_players_data(self) -> bool:
        """Populate and save players data."""
        try:
            logger.info("👥 Populating players data...")

            # Get players from provider
            players = self.provider.get_players()
            self.results['players'] = len(players)

            if not players:
                logger.warning("No players data retrieved")
                return False

            # Convert to DataFrame
            players_df = pl.DataFrame(players)

            # Save to data store
            today_str = date.today().strftime('%Y-%m-%d')
            file_path = self.data_store.store_players_data(players_df, today_str)

            logger.info(f"✅ Players data saved: {len(players)} players -> {file_path}")
            self.results['files_saved'] += 1
            return True

        except Exception as e:
            error_msg = f"Failed to populate players data: {e}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return False

    def populate_games_data(self) -> bool:
        """Populate and save today's games data."""
        try:
            logger.info("📅 Populating today's games data...")

            today = date.today().strftime('%Y-%m-%d')

            # Get games from provider
            games = self.provider.get_games(today, today)
            self.results['games'] = len(games)

            if not games:
                logger.warning("No games data retrieved for today")
                return True  # Not an error if no games today

            # Convert to DataFrame
            games_df = pl.DataFrame(games)

            # Save to data store
            file_path = self.data_store.store_games_data(games_df, today)

            logger.info(f"✅ Games data saved: {len(games)} games -> {file_path}")
            self.results['files_saved'] += 1
            return True

        except Exception as e:
            error_msg = f"Failed to populate games data: {e}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return False

    def populate_rosters_data(self) -> bool:
        """Populate and save rosters for teams playing today."""
        try:
            logger.info("📋 Populating rosters data...")

            # Get today's games to find active teams
            today = date.today().strftime('%Y-%m-%d')
            games = self.provider.get_games(today, today)

            if not games:
                logger.info("No games today, skipping rosters")
                return True

            # Get unique team IDs
            team_ids = set()
            for game in games:
                team_ids.add(game['home_team_id'])
                team_ids.add(game['visitor_team_id'])

            logger.info(f"Found {len(team_ids)} teams with games today")

            rosters_total = 0
            today_str = date.today().strftime('%Y-%m-%d')

            for team_id in team_ids:
                try:
                    # Get roster from provider
                    roster = self.provider.get_team_roster(team_id, season=2024)

                    if roster:
                        # Convert to DataFrame
                        roster_df = pl.DataFrame(roster)

                        # Add team_id column if not present
                        if 'team_id' not in roster_df.columns:
                            roster_df = roster_df.with_columns([
                                pl.lit(team_id).alias('team_id')
                            ])

                        # Save to data store
                        file_path = self.data_store.store_roster_data(roster_df, team_id, today_str)

                        rosters_total += len(roster)
                        logger.debug(f"Saved roster for team {team_id}: {len(roster)} players")

                except Exception as e:
                    logger.warning(f"Failed to get roster for team {team_id}: {e}")
                    continue

            self.results['rosters'] = rosters_total
            if rosters_total > 0:
                self.results['files_saved'] += len(team_ids)

            logger.info(f"✅ Rosters data saved: {rosters_total} total players from {len(team_ids)} teams")
            return True

        except Exception as e:
            error_msg = f"Failed to populate rosters data: {e}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return False

    def verify_saved_data(self) -> bool:
        """Verify that data was saved correctly."""
        try:
            logger.info("🔍 Verifying saved data...")

            today_str = date.today().strftime('%Y-%m-%d')

            # Check metadata
            metadata = self.data_store.get_metadata()
            if metadata.height == 0:
                logger.warning("No metadata found in data store")
                return False

            logger.info(f"Found {metadata.height} tables in data store")

            # Check specific data types
            tables_found = 0
            for data_type in ['teams', 'players', 'games', 'rosters']:
                matching_tables = metadata.filter(
                    pl.col("table_name").str.contains(data_type)
                )
                if matching_tables.height > 0:
                    tables_found += 1
                    logger.info(f"✅ Found {data_type} data: {matching_tables.height} tables")

            if tables_found >= 2:  # At least teams and players
                logger.info("✅ Data verification passed")
                return True
            else:
                logger.warning(f"⚠️ Data verification incomplete: only {tables_found} data types found")
                return False

        except Exception as e:
            error_msg = f"Failed to verify saved data: {e}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return False

    def run_complete_population(self) -> Dict[str, Any]:
        """
        Run complete data population test.

        Returns:
            Dict with comprehensive results and statistics
        """
        logger.info("🚀 Starting Complete NBA Data Population Test")

        steps = [
            ("Initialize Data Store", self.test_data_store_initialization),
            ("Populate Teams Data", self.populate_teams_data),
            ("Populate Players Data", self.populate_players_data),
            ("Populate Games Data", self.populate_games_data),
            ("Populate Rosters Data", self.populate_rosters_data),
            ("Verify Saved Data", self.verify_saved_data)
        ]

        successful_steps = 0

        for step_name, step_func in steps:
            logger.info(f"\n{'='*60}")
            logger.info(f"STEP: {step_name}")
            logger.info(f"{'='*60}")

            start_step = time.time()

            try:
                success = step_func()
                step_duration = time.time() - start_step

                if success:
                    successful_steps += 1
                    logger.info(f"✅ {step_name} completed successfully ({step_duration:.1f}s)")
                else:
                    logger.error(f"❌ {step_name} failed ({step_duration:.1f}s)")

            except Exception as e:
                logger.error(f"❌ {step_name} crashed: {e}")
                self.results['errors'].append(f"{step_name} crashed: {e}")

        # Calculate final statistics
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()

        final_results = {
            'success_rate': successful_steps / len(steps),
            'successful_steps': successful_steps,
            'total_steps': len(steps),
            'total_duration': total_duration,
            'data_counts': self.results,
            'error_count': len(self.results['errors']),
            'errors': self.results['errors'],
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat()
        }

        # Final summary
        logger.info(f"\n{'='*80}")
        logger.info("🎯 COMPLETE DATA POPULATION TEST SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Success Rate: {final_results['success_rate']:.1%} ({successful_steps}/{len(steps)} steps)")
        logger.info(f"Duration: {total_duration:.1f} seconds")
        logger.info(f"Data Collected:")
        logger.info(f"  - Teams: {self.results['teams']}")
        logger.info(f"  - Players: {self.results['players']}")
        logger.info(f"  - Games Today: {self.results['games']}")
        logger.info(f"  - Roster Players: {self.results['rosters']}")
        logger.info(f"  - Files Saved: {self.results['files_saved']}")
        logger.info(f"Errors: {len(self.results['errors'])}")

        if self.results['errors']:
            logger.error("Errors encountered:")
            for error in self.results['errors']:
                logger.error(f"  - {error}")

        if final_results['success_rate'] >= 0.8:
            logger.info("🎉 COMPLETE POPULATION TEST: SUCCESS")
        else:
            logger.warning("⚠️ COMPLETE POPULATION TEST: PARTIAL SUCCESS")

        return final_results


def run_complete_population_test():
    """Run the complete data population test."""
    populator = CompleteDataPopulator()
    results = populator.run_complete_population()
    return results


if __name__ == "__main__":
    results = run_complete_population_test()
    print(f"\nFinal Results: {results}")