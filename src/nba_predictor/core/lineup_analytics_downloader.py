#!/usr/bin/env python3
"""
🏀 NBA Lineup Analytics Downloader
Context7-compliant lineup data extraction and analysis using LeagueDashLineups API.
"""

import time
import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import polars as pl
import pandas as pd
from nba_api.stats.endpoints.leaguedashlineups import LeagueDashLineups

from .data_store import UnifiedDataStore
from .roster_injury_schemas import LineupStats, LineupAnalysis

logger = logging.getLogger(__name__)

@dataclass
class LineupDownloadConfig:
    """Configuration for lineup data downloads."""
    season: str
    season_type: str = "Regular Season"
    measure_type: str = "Base"
    per_mode: str = "PerGame"
    group_quantity: int = 5
    pace_adjust: str = "N"
    plus_minus: str = "Y"
    rank: str = "N"
    min_games: int = 10
    timeout_seconds: int = 30
    retry_attempts: int = 3
    base_delay: float = 0.6

class NBALineupAnalyticsDownloader:
    """Context7-compliant NBA lineup analytics downloader."""

    def __init__(self, data_store: UnifiedDataStore):
        """Initialize lineup analytics downloader."""
        self.data_store = data_store
        self.api = LeagueDashLineups
        self.team_mappings = self._get_team_mappings()

    def _get_team_mappings(self) -> Dict[int, str]:
        """Get NBA team ID to name mappings."""
        return {
            1610612737: "Atlanta Hawks", 1610612738: "Boston Celtics",
            1610612740: "New Orleans Pelicans", 1610612741: "Chicago Bulls",
            1610612742: "Dallas Mavericks", 1610612743: "Denver Nuggets",
            1610612744: "Detroit Pistons", 1610612745: "Golden State Warriors",
            1610612746: "Houston Rockets", 1610612747: "Los Angeles Clippers",
            1610612748: "Los Angeles Lakers", 1610612749: "Miami Heat",
            1610612750: "Milwaukee Bucks", 1610612751: "Minnesota Timberwolves",
            1610612752: "New Orleans Pelicans", 1610612753: "New York Knicks",
            1610612754: "Oklahoma City Thunder", 1610612755: "Orlando Magic",
            1610612756: "Philadelphia 76ers", 1610612757: "Phoenix Suns",
            1610612758: "Portland Trail Blazers", 1610612759: "Sacramento Kings",
            1610612760: "San Antonio Spurs", 1610612761: "Seattle SuperSonics",
            1610612762: "Toronto Raptors", 1610612763: "Utah Jazz",
            1610612764: "Washington Wizards", 1610612765: "Detroit Pistons",
            1610612766: "Charlotte Hornets"
        }

    def download_team_lineups(self, team_id: int, config: LineupDownloadConfig) -> Optional[Dict[str, Any]]:
        """Download lineup data for a specific team."""
        logger.info(f"📊 Downloading lineup data for team {team_id} ({self.team_mappings.get(team_id, 'Unknown')})")

        try:
            # Make API call with retry logic
            for attempt in range(config.retry_attempts):
                try:
                    # LeagueDashLineups API call with correct parameters
                    response = self.api(
                        season=config.season,
                        season_type_all_star=config.season_type,
                        measure_type_detailed_defense=config.measure_type,
                        per_mode_detailed=config.per_mode,
                        group_quantity=config.group_quantity,
                        pace_adjust=config.pace_adjust,
                        plus_minus=config.plus_minus,
                        rank=config.rank,
                        date_from_nullable=f"{config.season.split('-')[0]}-10-01",  # Season start
                        date_to_nullable=f"{config.season.split('-')[1]}-04-30",     # Season end
                        last_n_games=0,  # All games
                        month=0,         # All months
                        opponent_team_id=0,
                        team_id_nullable=team_id  # This is optional and will filter results
                    )
                    data = response.get_data_frames()[0] if response.get_data_frames() else None

                    if data is not None and not data.empty:
                        logger.info(f"✅ Downloaded {len(data)} lineup records for team {team_id}")
                        return self._process_lineup_data(data, team_id, config)
                    else:
                        logger.warning(f"⚠️ No lineup data available for team {team_id}")
                        return None

                except Exception as api_error:
                    if attempt < config.retry_attempts - 1:
                        delay = config.base_delay * (2 ** attempt)
                        logger.warning(f"Retry {attempt + 1}/{config.retry_attempts} for team {team_id} after {delay:.1f}s: {api_error}")
                        time.sleep(delay)
                    else:
                        raise api_error

        except Exception as e:
            logger.error(f"❌ Failed to download lineup data for team {team_id}: {e}")
            return None

    def _process_lineup_data(self, data: pd.DataFrame, team_id: int, config: LineupDownloadConfig) -> Dict[str, Any]:
        """Process raw lineup data into Context7-compliant format."""
        logger.info(f"🔄 Processing lineup data for team {team_id}")

        # Filter out lineups with insufficient games
        filtered_data = data[data['GP'] >= config.min_games].copy()

        if filtered_data.empty:
            logger.warning(f"⚠️ No lineups with {config.min_games}+ games for team {team_id}")
            return {'success': False, 'lineups': [], 'team_id': team_id}

        # Convert to list of lineup stats
        lineups = []
        for _, row in filtered_data.iterrows():
            try:
                lineup_stats = LineupStats(
                    group_id=int(row['GROUP_ID']),
                    group_name=row['GROUP_NAME'],
                    team_id=team_id,
                    team_abbreviation=row['TEAM_ABBREVIATION'],
                    games_played=int(row['GP']),
                    wins=int(row['W']),
                    losses=int(row['L']),
                    win_percentage=float(row['W_PCT']),
                    minutes=float(row['MIN']),
                    field_goals_made=float(row['FGM']),
                    field_goals_attempted=float(row['FGA']),
                    field_goal_percentage=float(row['FG_PCT']) if pd.notna(row['FG_PCT']) else None,
                    three_points_made=float(row['FG3M']),
                    three_points_attempted=float(row['FG3A']),
                    three_point_percentage=float(row['FG3_PCT']) if pd.notna(row['FG3_PCT']) else None,
                    free_throws_made=float(row['FTM']),
                    free_throws_attempted=float(row['FTA']),
                    free_throw_percentage=float(row['FT_PCT']) if pd.notna(row['FT_PCT']) else None,
                    offensive_rebounds=float(row['OREB']),
                    defensive_rebounds=float(row['DREB']),
                    total_rebounds=float(row['REB']),
                    assists=float(row['AST']),
                    turnovers=float(row['TOV']),
                    steals=float(row['STL']),
                    blocks=float(row['BLK']),
                    blocked_attempts=float(row['BLKA']),
                    personal_fouls=float(row['PF']),
                    personal_fouls_drawn=float(row['PFD']),
                    points=float(row['PTS']),
                    plus_minus=float(row['PLUS_MINUS']) if pd.notna(row['PLUS_MINUS']) else 0.0,
                    season=config.season,
                    season_type=config.season_type,
                    last_updated=datetime.now().isoformat()
                )
                lineups.append(lineup_stats.dict())
            except Exception as e:
                logger.warning(f"⚠️ Failed to process lineup record: {e}")
                continue

        logger.info(f"✅ Processed {len(lineups)} valid lineups for team {team_id}")

        return {
            'success': True,
            'team_id': team_id,
            'team_name': self.team_mappings.get(team_id, 'Unknown'),
            'lineups': lineups,
            'total_lineups': len(lineups),
            'season': config.season,
            'season_type': config.season_type,
            'min_games_filter': config.min_games,
            'download_timestamp': datetime.now().isoformat()
        }

    def download_all_team_lineups(self, config: LineupDownloadConfig) -> Dict[str, Any]:
        """Download lineup data for all NBA teams."""
        logger.info(f"🏀 Starting lineup analytics download for {config.season} {config.season_type}")

        start_time = time.time()
        results = []
        successful_teams = 0
        total_lineups = 0

        # Process each team
        for team_id in sorted(self.team_mappings.keys()):
            logger.info(f"Processing team {successful_teams + 1}/{len(self.team_mappings)}: {self.team_mappings[team_id]}")

            team_result = self.download_team_lineups(team_id, config)

            if team_result and team_result.get('success', False):
                results.append(team_result)
                successful_teams += 1
                total_lineups += team_result.get('total_lineups', 0)

                # Store lineup data
                self._store_team_lineups(team_result)

            # Rate limiting
            time.sleep(config.base_delay)

        elapsed_time = time.time() - start_time

        summary = {
            'success': True,
            'season': config.season,
            'season_type': config.season_type,
            'teams_processed': successful_teams,
            'total_teams': len(self.team_mappings),
            'total_lineups': total_lineups,
            'processing_time_seconds': elapsed_time,
            'average_lineups_per_team': total_lineups / successful_teams if successful_teams > 0 else 0,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'measure_type': config.measure_type,
                'per_mode': config.per_mode,
                'min_games': config.min_games,
                'group_quantity': config.group_quantity
            }
        }

        logger.info(f"🎉 Lineup analytics download completed: {successful_teams}/{len(self.team_mappings)} teams, {total_lineups} lineups in {elapsed_time:.1f}s")
        return summary

    def _store_team_lineups(self, team_result: Dict[str, Any]) -> bool:
        """Store lineup data using the data store."""
        try:
            team_id = team_result['team_id']
            season = team_result['season']

            # Convert to Polars DataFrame
            df = pl.DataFrame(team_result['lineups'])

            # Store as Parquet
            filename = f"lineups_team_{team_id}_{season.replace('-', '_')}.parquet"
            # Use existing data store method pattern
            filepath = self.data_store.base_path / "lineups" / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(filepath)

            if filepath:
                logger.info(f"✅ Stored lineup data: {filepath}")
                return True
            else:
                logger.error(f"❌ Failed to store lineup data for team {team_id}")
                return False

        except Exception as e:
            logger.error(f"❌ Error storing lineup data: {e}")
            return False

    def analyze_lineup_effectiveness(self, team_id: int, season: str) -> Optional[Dict[str, Any]]:
        """Analyze lineup effectiveness for a specific team."""
        logger.info(f"📈 Analyzing lineup effectiveness for team {team_id} in {season}")

        try:
            # Get lineup data
            filename = f"lineups_team_{team_id}_{season.replace('-', '_')}.parquet"
            filepath = self.data_store.base_path / "lineups" / filename

            if not filepath.exists():
                logger.warning(f"⚠️ No lineup data found for team {team_id}")
                return None

            df = pl.read_parquet(filepath)

            # Convert to pandas for analysis
            pdf = df.to_pandas()

            # Calculate effectiveness metrics
            analysis = self._calculate_lineup_metrics(pdf, team_id, season)

            logger.info(f"✅ Lineup effectiveness analysis completed for team {team_id}")
            return analysis

        except Exception as e:
            logger.error(f"❌ Failed to analyze lineup effectiveness for team {team_id}: {e}")
            return None

    def _calculate_lineup_metrics(self, data: pd.DataFrame, team_id: int, season: str) -> Dict[str, Any]:
        """Calculate comprehensive lineup effectiveness metrics."""
        try:
            # Top lineups by various metrics
            top_by_win_pct = data.nlargest(5, 'win_percentage')
            top_by_plus_minus = data.nlargest(5, 'plus_minus')
            top_by_points = data.nlargest(5, 'points')

            # Overall team statistics
            total_lineups = len(data)
            avg_minutes = data['minutes'].mean()
            avg_plus_minus = data['plus_minus'].mean()
            avg_win_pct = data['win_percentage'].mean()

            # Effectiveness classification
            high_performers = data[data['win_percentage'] >= 0.600]
            effective_lineups = data[data['plus_minus'] > 0]

            return {
                'team_id': team_id,
                'team_name': self.team_mappings.get(team_id, 'Unknown'),
                'season': season,
                'analysis_timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_lineups': total_lineups,
                    'high_performance_lineups': len(high_performers),
                    'effective_lineups': len(effective_lineups),
                    'average_minutes_per_lineup': round(avg_minutes, 1),
                    'average_plus_minus': round(avg_plus_minus, 2),
                    'average_win_percentage': round(avg_win_pct, 3),
                    'high_performance_rate': round(len(high_performers) / total_lineups * 100, 1) if total_lineups > 0 else 0,
                    'effectiveness_rate': round(len(effective_lineups) / total_lineups * 100, 1) if total_lineups > 0 else 0
                },
                'top_lineups': {
                    'by_win_percentage': top_by_win_pct[['group_name', 'games_played', 'win_percentage', 'plus_minus', 'points']].to_dict('records'),
                    'by_plus_minus': top_by_plus_minus[['group_name', 'games_played', 'win_percentage', 'plus_minus', 'points']].to_dict('records'),
                    'by_points': top_by_points[['group_name', 'games_played', 'win_percentage', 'plus_minus', 'points']].to_dict('records')
                },
                'distribution': {
                    'minutes_distribution': {
                        'min': float(data['minutes'].min()),
                        'max': float(data['minutes'].max()),
                        'mean': float(avg_minutes),
                        'median': float(data['minutes'].median())
                    },
                    'performance_distribution': {
                        'win_pct_std': float(data['win_percentage'].std()),
                        'plus_minus_std': float(data['plus_minus'].std()),
                        'points_std': float(data['points'].std())
                    }
                }
            }

        except Exception as e:
            logger.error(f"❌ Error calculating lineup metrics: {e}")
            return {'error': str(e)}

    def generate_league_lineup_report(self, season: str) -> Dict[str, Any]:
        """Generate comprehensive league-wide lineup analysis."""
        logger.info(f"📊 Generating league lineup report for {season}")

        try:
            all_team_analyses = []
            total_lineups = 0

            # Analyze each team
            for team_id in sorted(self.team_mappings.keys()):
                analysis = self.analyze_lineup_effectiveness(team_id, season)
                if analysis and 'summary' in analysis:
                    all_team_analyses.append(analysis)
                    total_lineups += analysis['summary']['total_lineups']

            if not all_team_analyses:
                return {'error': 'No lineup data available for analysis'}

            # League-wide statistics
            avg_lineups_per_team = total_lineups / len(all_team_analyses)
            league_avg_plus_minus = sum(team['summary']['average_plus_minus'] for team in all_team_analyses) / len(all_team_analyses)
            league_avg_win_pct = sum(team['summary']['average_win_percentage'] for team in all_team_analyses) / len(all_team_analyses)

            # Top teams by lineup effectiveness
            top_teams_by_effectiveness = sorted(all_team_analyses, key=lambda x: x['summary']['effectiveness_rate'], reverse=True)[:5]
            top_teams_by_performance = sorted(all_team_analyses, key=lambda x: x['summary']['high_performance_rate'], reverse=True)[:5]

            report = {
                'season': season,
                'report_timestamp': datetime.now().isoformat(),
                'league_summary': {
                    'teams_analyzed': len(all_team_analyses),
                    'total_lineups': total_lineups,
                    'average_lineups_per_team': round(avg_lineups_per_team, 1),
                    'league_average_plus_minus': round(league_avg_plus_minus, 2),
                    'league_average_win_percentage': round(league_avg_win_pct, 3),
                    'league_effectiveness_rate': round(sum(team['summary']['effectiveness_rate'] for team in all_team_analyses) / len(all_team_analyses), 1),
                    'league_high_performance_rate': round(sum(team['summary']['high_performance_rate'] for team in all_team_analyses) / len(all_team_analyses), 1)
                },
                'top_performers': {
                    'by_effectiveness': [
                        {
                            'team_name': team['team_name'],
                            'effectiveness_rate': team['summary']['effectiveness_rate'],
                            'total_lineups': team['summary']['total_lineups'],
                            'average_plus_minus': team['summary']['average_plus_minus']
                        } for team in top_teams_by_effectiveness
                    ],
                    'by_high_performance': [
                        {
                            'team_name': team['team_name'],
                            'high_performance_rate': team['summary']['high_performance_rate'],
                            'total_lineups': team['summary']['total_lineups'],
                            'average_win_percentage': team['summary']['average_win_percentage']
                        } for team in top_teams_by_performance
                    ]
                },
                'detailed_analyses': all_team_analyses
            }

            logger.info(f"✅ League lineup report generated: {len(all_team_analyses)} teams, {total_lineups} total lineups")
            return report

        except Exception as e:
            logger.error(f"❌ Failed to generate league lineup report: {e}")
            return {'error': str(e)}