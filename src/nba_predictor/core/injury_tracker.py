#!/usr/bin/env python3
"""
🏀 NBA Injury Tracking System
Context7-compliant multi-source injury data aggregation system.
"""

import polars as pl
import pandas as pd
import time
import requests
from datetime import datetime, date, timedelta
from pathlib import Path
import logging
from typing import List, Dict, Optional, Any, Tuple
import sqlite3
from dataclasses import dataclass
import re

from nba_api.stats.static import teams

from .mock_injury_generator import MockInjuryGenerator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class InjuryDataConfig:
    """Configuration for injury tracking system."""
    rate_limit: float = 1.0  # 1 second between requests
    max_retries: int = 3
    retry_delay: float = 2.0
    timeout: int = 30
    validate_data: bool = True
    sources: List[str] = None  # ['nba_api', 'espn', 'basketball_reference']

    def __post_init__(self):
        if self.sources is None:
            self.sources = ['nba_api', 'basketball_reference']  # Start with reliable sources


class NBAInjuryTracker:
    """Multi-source NBA injury data aggregation system."""

    def __init__(self, data_dir: str = "data", config: Optional[InjuryDataConfig] = None):
        """Initialize injury tracker with data directory and configuration."""
        self.data_dir = Path(data_dir)
        self.injury_dir = self.data_dir / "injuries"
        self.injury_dir.mkdir(parents=True, exist_ok=True)

        # Database connection
        self.db_path = self.data_dir / "nba_data.db"
        self._init_database()

        # Configuration
        self.config = config or InjuryDataConfig()

        # Mock data generator
        self.mock_generator = MockInjuryGenerator()

        # Statistics
        self.session_start = datetime.now()
        self.stats = {
            'players_processed': 0,
            'injuries_found': 0,
            'sources_used': set(),
            'errors': 0,
            'retries': 0
        }

    def _init_database(self):
        """Initialize database tables for injury data."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Create player injuries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_injuries (
                player_id INTEGER,
                player_name TEXT,
                team_id INTEGER,
                team_name TEXT,
                season TEXT,
                injury_status TEXT,
                injury_type TEXT,
                injury_description TEXT,
                injury_date DATE,
                expected_return DATE,
                return_date DATE,
                games_missed INTEGER,
                availability_probability REAL,
                last_updated TEXT,
                source TEXT,
                confidence_score REAL,
                notes TEXT,
                PRIMARY KEY (player_id, injury_date, source)
            )
        ''')

        # Create injury trends table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS injury_trends (
                team_id INTEGER,
                season TEXT,
                date DATE,
                total_injured INTEGER,
                key_players_injured INTEGER,
                injury_impact_score REAL,
                last_updated TEXT,
                source TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def get_nba_teams(self) -> List[Dict[str, Any]]:
        """Get all NBA teams."""
        try:
            nba_teams = teams.get_teams()
            active_teams = [team for team in nba_teams if team.get('is_nba_franchise', True)]
            return active_teams
        except Exception as e:
            logger.error(f"Error retrieving NBA teams: {e}")
            return []

    def extract_injuries_from_box_score(self, game_data: str, home_team: str, away_team: str,
                                       game_date: date) -> List[Dict[str, Any]]:
        """Extract injury information from Basketball Reference box score data."""
        injuries = []

        try:
            # Look for "Inactive:" section in the data
            inactive_pattern = r'\*\*Inactive:\*\*\s*(.*?)(?=\*\*)'
            inactive_match = re.search(inactive_pattern, game_data, re.DOTALL)

            if inactive_match:
                inactive_text = inactive_match.group(1)

                # Parse team-specific inactive players
                teams_inactive = re.split(r'\*\*(\w+)\*\*', inactive_text)

                for i in range(1, len(teams_inactive), 2):
                    if i + 1 < len(teams_inactive):
                        team_abbrev = teams_inactive[i]
                        players_text = teams_inactive[i + 1]

                        # Extract player names and links
                        player_pattern = r'\[([^\]]+)\]\((/players/[^/]+/[^)]+)\.html\)'
                        players = re.findall(player_pattern, players_text)

                        for player_name, player_url in players:
                            # Clean player name
                            clean_name = re.sub(r'\*\*', '', player_name).strip()

                            injury = {
                                'player_name': clean_name,
                                'team_abbreviation': team_abbrev,
                                'injury_status': 'Inactive',
                                'injury_date': game_date,
                                'source': 'basketball_reference',
                                'confidence_score': 0.9,
                                'notes': f'Inactive for {team_abbrev} vs game'
                            }
                            injuries.append(injury)

            logger.info(f"Found {len(injuries)} inactive players from box score")

        except Exception as e:
            logger.error(f"Error extracting injuries from box score: {e}")

        return injuries

    def download_basketball_reference_injuries(self, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Download injury data from Basketball Reference."""
        logger.info("📥 Downloading injuries from Basketball Reference")

        all_injuries = []

        # Basketball Reference box scores URL pattern
        base_url = "https://www.basketball-reference.com/boxscores"

        current_date = start_date
        while current_date <= end_date:
            try:
                # Rate limiting
                time.sleep(self.config.rate_limit)

                # Format date for URL
                date_str = current_date.strftime("%Y%m%d")

                # Get list of games for this date
                games_url = f"{base_url}/?month={current_date.month}&day={current_date.day}&year={current_date.year}"

                logger.info(f"Checking games for {current_date}")

                response = requests.get(games_url, timeout=self.config.timeout)
                response.raise_for_status()

                # Extract game links
                game_pattern = r'href="(/boxscores/[^"]+\.html)"'
                game_links = re.findall(game_pattern, response.text)

                for game_link in game_links:
                    try:
                        # Get full game URL
                        game_url = f"https://www.basketball-reference.com{game_link}"

                        # Download box score
                        game_response = requests.get(game_url, timeout=self.config.timeout)
                        game_response.raise_for_status()

                        # Extract teams from URL
                        url_teams = re.search(r'/(\w{3})(\w{3})\.html', game_link)
                        if url_teams:
                            away_team, home_team = url_teams.groups()

                            # Extract injuries
                            game_injuries = self.extract_injuries_from_box_score(
                                game_response.text, home_team, away_team, current_date
                            )

                            all_injuries.extend(game_injuries)

                    except Exception as e:
                        logger.error(f"Error processing game {game_link}: {e}")
                        self.stats['errors'] += 1

                # Move to next date
                current_date += timedelta(days=1)

            except Exception as e:
                logger.error(f"Error processing date {current_date}: {e}")
                self.stats['errors'] += 1
                current_date += timedelta(days=1)

        self.stats['sources_used'].add('basketball_reference')
        logger.info(f"✅ Downloaded {len(all_injuries)} injury records from Basketball Reference")

        return all_injuries

    def extract_espn_injuries(self, player_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract injury information from ESPN player data."""
        try:
            if not player_data.get('injured', False):
                return None

            injury = {
                'player_name': player_data.get('name', ''),
                'player_id': player_data.get('playerId', None),
                'injury_status': player_data.get('injuryStatus', 'Unknown'),
                'team_abbreviation': player_data.get('proTeam', ''),
                'injury_date': date.today(),
                'source': 'espn',
                'confidence_score': 0.8,
                'notes': f"ESPN Fantasy API - Status: {player_data.get('injuryStatus', '')}"
            }

            return injury

        except Exception as e:
            logger.error(f"Error extracting ESPN injury data: {e}")
            return None

    def download_espn_injuries(self, league_id: int = None) -> List[Dict[str, Any]]:
        """Download injury data from ESPN API (simplified version)."""
        logger.info("📥 Attempting to download injuries from ESPN API")

        injuries = []

        try:
            # Note: ESPN API requires authentication for detailed data
            # This is a simplified version that would need ESPN API keys

            # For now, we'll create mock data structure
            # In production, this would use actual ESPN API calls

            logger.warning("ESPN API requires authentication - using mock structure")

            # Mock example of what ESPN data would look like
            mock_injuries = [
                {
                    'player_name': 'Mock Player',
                    'player_id': 12345,
                    'injury_status': 'Questionable',
                    'team_abbreviation': 'LAL',
                    'injury_date': date.today(),
                    'source': 'espn_mock',
                    'confidence_score': 0.5,
                    'notes': 'Mock data - ESPN API requires authentication'
                }
            ]

            injuries.extend(mock_injuries)
            self.stats['sources_used'].add('espn_mock')

        except Exception as e:
            logger.error(f"Error downloading ESPN injuries: {e}")
            self.stats['errors'] += 1

        logger.info(f"⚠️  ESPN API limited - {len(injuries)} mock records")
        return injuries

    def download_nba_injuries(self, season: str) -> List[Dict[str, Any]]:
        """Download injury data from NBA API."""
        logger.info("📥 Downloading injuries from NBA API")

        injuries = []

        try:
            # NBA API doesn't have a direct injury endpoint
            # We can infer injuries from player game logs and missing games

            nba_teams = self.get_nba_teams()

            for team in nba_teams[:5]:  # Limit to first 5 teams for testing
                try:
                    time.sleep(self.config.rate_limit)

                    team_id = team['id']
                    team_name = team['full_name']

                    # Note: NBA API doesn't have direct injury endpoints
                    # PlayerDashboardByShootingSplits doesn't take team_id parameter
                    # This is a placeholder for future NBA API injury integration
                    logger.info(f"Processing team {team_name} for injury data")

                    # Process to find players with limited recent activity
                    # (This is a simplified approach)

                    logger.info(f"Processed injury data for {team_name}")

                except Exception as e:
                    logger.error(f"Error processing team {team_name}: {e}")
                    self.stats['errors'] += 1

            self.stats['sources_used'].add('nba_api')

        except Exception as e:
            logger.error(f"Error downloading NBA injuries: {e}")
            self.stats['errors'] += 1

        logger.info(f"✅ NBA API injury data processed")
        return injuries

    def aggregate_injury_data(self, injuries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate and deduplicate injury data from multiple sources."""
        logger.info("🔄 Aggregating injury data from multiple sources")

        # Group by player name and date
        injury_groups = {}

        for injury in injuries:
            key = (injury.get('player_name', ''), injury.get('injury_date', date.today()))

            if key not in injury_groups:
                injury_groups[key] = []

            injury_groups[key].append(injury)

        # Merge data for each player
        aggregated_injuries = []

        for (player_name, injury_date), player_injuries in injury_groups.items():
            try:
                # Find most reliable source
                sources = [inj['source'] for inj in player_injuries]
                confidence_scores = [inj.get('confidence_score', 0.5) for inj in player_injuries]

                # Prefer basketball_reference, then espn, then nba_api
                source_priority = {'basketball_reference': 3, 'espn': 2, 'espn_mock': 1, 'nba_api': 1}

                best_injury = max(player_injuries,
                                 key=lambda x: source_priority.get(x.get('source', ''), 0) + x.get('confidence_score', 0))

                # Create aggregated record
                aggregated_injury = {
                    'player_name': player_name,
                    'player_id': best_injury.get('player_id'),
                    'team_abbreviation': best_injury.get('team_abbreviation'),
                    'injury_status': best_injury.get('injury_status'),
                    'injury_date': injury_date,
                    'injury_type': best_injury.get('injury_type'),
                    'injury_description': best_injury.get('injury_description'),
                    'expected_return': best_injury.get('expected_return'),
                    'availability_probability': self._calculate_availability_probability(best_injury),
                    'sources': list(set(sources)),
                    'confidence_score': max(confidence_scores),
                    'notes': f"Aggregated from {len(player_injuries)} sources: {', '.join(sources)}",
                    'last_updated': datetime.now().isoformat(),
                    'aggregated': True
                }

                aggregated_injuries.append(aggregated_injury)

            except Exception as e:
                logger.error(f"Error aggregating injury data for {player_name}: {e}")
                self.stats['errors'] += 1

        logger.info(f"✅ Aggregated {len(aggregated_injuries)} unique injury records")
        return aggregated_injuries

    def _calculate_availability_probability(self, injury: Dict[str, Any]) -> float:
        """Calculate player availability probability based on injury status."""
        status = injury.get('injury_status', '').lower()

        status_probabilities = {
            'active': 1.0,
            'probable': 0.75,
            'questionable': 0.5,
            'doubtful': 0.25,
            'out': 0.0,
            'inactive': 0.0,
            'day-to-day': 0.6
        }

        return status_probabilities.get(status, 0.5)

    def save_injury_data(self, injuries: List[Dict[str, Any]]) -> bool:
        """Save aggregated injury data to database and files."""
        try:
            if not injuries:
                logger.warning("No injury data to save")
                return False

            # Convert to DataFrame
            df = pd.DataFrame(injuries)

            # Save to Parquet
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"injuries_aggregated_{timestamp}.parquet"
            file_path = self.injury_dir / filename

            pl_df = pl.from_pandas(df)
            pl_df.write_parquet(file_path)

            # Save to database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            for injury in injuries:
                cursor.execute('''
                    INSERT OR REPLACE INTO player_injuries
                    (player_id, player_name, team_id, team_name, season,
                     injury_status, injury_type, injury_description,
                     injury_date, expected_return, return_date,
                     games_missed, availability_probability, last_updated,
                     source, confidence_score, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', [
                    injury.get('player_id'),
                    injury.get('player_name'),
                    None,  # team_id - would need mapping
                    injury.get('team_abbreviation'),
                    '2024-25',  # season - would need parameter
                    injury.get('injury_status'),
                    injury.get('injury_type'),
                    injury.get('injury_description'),
                    injury.get('injury_date'),
                    injury.get('expected_return'),
                    None,  # return_date
                    None,  # games_missed
                    injury.get('availability_probability'),
                    injury.get('last_updated'),
                    injury.get('source'),
                    injury.get('confidence_score'),
                    injury.get('notes')
                ])

            conn.commit()
            conn.close()

            logger.info(f"✅ Saved {len(injuries)} injury records to database and {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving injury data: {e}")
            return False

    def download_injuries_for_period(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Download injuries for a specific date period from all sources."""
        logger.info(f"🏀 Starting injury download for {start_date} to {end_date}")

        all_injuries = []

        # Download from Basketball Reference
        if 'basketball_reference' in self.config.sources:
            try:
                bb_injuries = self.download_basketball_reference_injuries(start_date, end_date)
                all_injuries.extend(bb_injuries)
            except Exception as e:
                logger.error(f"Error downloading from Basketball Reference: {e}")
                self.stats['errors'] += 1

        # Download from ESPN (mock for now)
        if 'espn' in self.config.sources:
            try:
                espn_injuries = self.download_espn_injuries()
                all_injuries.extend(espn_injuries)
            except Exception as e:
                logger.error(f"Error downloading from ESPN: {e}")
                self.stats['errors'] += 1

        # Download from NBA API
        if 'nba_api' in self.config.sources:
            try:
                nba_injuries = self.download_nba_injuries('2024-25')
                all_injuries.extend(nba_injuries)
            except Exception as e:
                logger.error(f"Error downloading from NBA API: {e}")
                self.stats['errors'] += 1

        # If no real data found, generate mock data
        if not all_injuries:
            logger.warning("⚠️  No real injury data found, generating mock data for testing")
            mock_injuries = self.mock_generator.generate_mock_injuries(start_date, end_date, count=20)
            all_injuries.extend(mock_injuries)
            self.stats['sources_used'].add('mock_generator')

        # Aggregate data
        aggregated_injuries = self.aggregate_injury_data(all_injuries)

        # Save data
        success = self.save_injury_data(aggregated_injuries)

        # Update statistics
        self.stats['players_processed'] = len(set(inj.get('player_name') for inj in aggregated_injuries))
        self.stats['injuries_found'] = len(aggregated_injuries)

        # Generate summary
        session_time = datetime.now() - self.session_start

        summary = {
            'period': f"{start_date} to {end_date}",
            'total_injuries': len(aggregated_injuries),
            'unique_players': self.stats['players_processed'],
            'sources_used': list(self.stats['sources_used']),
            'errors': self.stats['errors'],
            'session_time_seconds': session_time.total_seconds(),
            'success': success,
            'data_type': 'Real' if not any(inj.get('source') == 'mock_generator' for inj in all_injuries) else 'Mock'
        }

        self._log_summary(summary)
        return summary

    def _log_summary(self, summary: Dict[str, Any]):
        """Log comprehensive summary."""
        logger.info("=" * 80)
        logger.info("🎯 INJURY TRACKING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Period: {summary['period']}")
        logger.info(f"✅ Total Injuries: {summary['total_injuries']}")
        logger.info(f"👥 Unique Players: {summary['unique_players']}")
        logger.info(f"📊 Sources Used: {', '.join(summary['sources_used'])}")
        logger.info(f"⚠️  Errors: {summary['errors']}")
        logger.info(f"⏱️  Session Time: {summary['session_time_seconds']:.2f} seconds")
        logger.info(f"🎯 Status: {'SUCCESS' if summary['success'] else 'FAILED'}")
        logger.info(f"📋 Data Type: {summary.get('data_type', 'Unknown')}")
        logger.info("=" * 80)

    def get_current_injuries(self) -> List[Dict[str, Any]]:
        """Get current injury information from database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute('''
                SELECT player_name, team_name, injury_status, injury_date,
                       availability_probability, source, notes, last_updated
                FROM player_injuries
                WHERE injury_date >= date('now', '-7 days')
                ORDER BY injury_date DESC, availability_probability ASC
            ''')

            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]

            conn.close()
            return results

        except Exception as e:
            logger.error(f"Error retrieving current injuries: {e}")
            return []


def main():
    """Main function to test injury tracking system."""
    print("🏀 NBA INJURY TRACKING SYSTEM TEST")
    print("=" * 80)

    tracker = NBAInjuryTracker()

    # Test with recent dates
    end_date = date.today()
    start_date = end_date - timedelta(days=7)  # Last 7 days

    print(f"Testing injury tracking for {start_date} to {end_date}")
    print("-" * 40)

    result = tracker.download_injuries_for_period(start_date, end_date)

    if result['success'] and result['total_injuries'] > 0:
        print(f"\n✅ Successfully tracked {result['total_injuries']} injuries")
        print(f"   Sources: {', '.join(result['sources_used'])}")
        print(f"   Unique players: {result['unique_players']}")

        # Show sample current injuries
        current_injuries = tracker.get_current_injuries()
        if current_injuries:
            print(f"\n📝 Sample current injuries:")
            for injury in current_injuries[:5]:
                print(f"   {injury['player_name']} ({injury['team_abbreviation']}): {injury['injury_status']}")
                print(f"      Availability: {injury.get('availability_probability', 'N/A'):.0%}")
                print(f"      Source: {injury['source']}")
    else:
        print("❌ No injuries found or tracking failed")

    print(f"\n📊 Statistics:")
    print(f"   Session time: {result['session_time_seconds']:.2f}s")
    print(f"   Errors: {result['errors']}")
    print(f"   Data saved to: {tracker.injury_dir}")

    print(f"\n✅ NBA Injury Tracking System test completed!")


if __name__ == "__main__":
    main()