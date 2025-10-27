#!/usr/bin/env python3
"""
🏀 NBA Roster & Injury Data Store Extensions
Extended UnifiedDataStore methods for comprehensive roster, injury, and lineup data management.
"""

import polars as pl
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import logging

from .data_store import UnifiedDataStore
from .roster_injury_schemas import (
    PlayerInfo, RosterInfo, InjuryInfo, LineupStats, TeamRoster, LineupAnalysis,
    ROSTER_COLUMN_MAPPING, INJURY_COLUMN_MAPPING, LINEUP_COLUMN_MAPPING
)
from ..api.data_provider import NBADataProvider

logger = logging.getLogger(__name__)


class RosterInjuryStoreExtensions:
    """Extended data store methods for roster and injury data."""

    def __init__(self, data_store: 'UnifiedDataStore'):
        """Initialize with reference to main data store."""
        self.data_store = data_store
        self.provider = NBADataProvider()

    def store_team_roster(self, roster: TeamRoster, validate: bool = True) -> bool:
        """Store complete team roster data."""
        try:
            # Convert to DataFrame
            roster_data = []
            for player in roster.players:
                player_dict = player.dict()
                roster_data.append(player_dict)

            df = pl.DataFrame(roster_data)

            # Save to data store
            season = roster.season
            team_id = roster.team_id

            roster_file = Path(self.data_store.data_dir) / "rosters" / f"roster_team_{team_id}_{season.replace('-', '_')}.parquet"
            roster_file.parent.mkdir(parents=True, exist_ok=True)

            df.write_parquet(roster_file)

            # Also store roster summary in DuckDB
            summary_df = pl.DataFrame([{
                'team_id': roster.team_id,
                'team_name': roster.team_name,
                'team_abbreviation': roster.team_abbreviation,
                'season': roster.season,
                'season_type': roster.season_type,
                'total_players': roster.total_players,
                'active_players': roster.active_players,
                'injured_players': roster.injured_players,
                'total_salary': roster.total_salary,
                'salary_cap_space': roster.salary_cap_space,
                'last_updated': roster.last_updated,
                'source': roster.source,
                'file_path': str(roster_file)
            }])

            self.data_store.conn.execute("""
                INSERT OR REPLACE INTO team_rosters
                (team_id, team_name, team_abbreviation, season, season_type,
                 total_players, active_players, injured_players, total_salary,
                 salary_cap_space, last_updated, source, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                roster.team_id, roster.team_name, roster.team_abbreviation,
                roster.season, roster.season_type, roster.total_players,
                roster.active_players, roster.injured_players,
                roster.total_salary, roster.salary_cap_space,
                roster.last_updated, roster.source, str(roster_file)
            ])

            logger.info(f"Stored roster for {roster.team_name} ({len(roster.players)} players)")
            return True

        except Exception as e:
            logger.error(f"Error storing roster: {e}")
            return False

    def store_injury_info(self, injury: InjuryInfo, validate: bool = True) -> bool:
        """Store player injury information."""
        try:
            # Convert to DataFrame
            injury_df = pl.DataFrame([injury.dict()])

            # Save to data store
            season = injury.season
            player_id = injury.player_id
            team_id = injury.team_id

            injury_file = Path(self.data_store.data_dir) / "injuries" / f"injury_player_{player_id}_{season.replace('-', '_')}.parquet"
            injury_file.parent.mkdir(parents=True, exist_ok=True)

            injury_df.write_parquet(injury_file)

            # Store in DuckDB
            self.data_store.conn.execute("""
                INSERT OR REPLACE INTO player_injuries
                (player_id, team_id, season, injury_status, previous_status,
                 injury_type, injury_description, injury_date, expected_return,
                 return_date, severity, games_missed, consecutive_games_missed,
                 game_time_decision, availability_probability, practice_status,
                 last_updated, source, confidence_score, notes, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                injury.player_id, injury.team_id, injury.season, injury.injury_status.value,
                injury.previous_status.value if injury.previous_status else None,
                injury.injury_type, injury.injury_description,
                injury.injury_date, injury.expected_return, injury.return_date,
                injury.severity.value if injury.severity else None,
                injury.games_missed, injury.consecutive_games_missed,
                injury.game_time_decision, injury.availability_probability,
                injury.practice_status, injury.last_updated,
                injury.source, injury.confidence_score,
                injury.notes, str(injury_file)
            ])

            logger.info(f"Stored injury info for player {injury.player_id}: {injury.injury_status.value}")
            return True

        except Exception as e:
            logger.error(f"Error storing injury info: {e}")
            return False

    def store_lineup_stats(self, lineup: LineupStats, validate: bool = True) -> bool:
        """Store lineup performance statistics."""
        try:
            # Convert to DataFrame
            lineup_df = pl.DataFrame([lineup.dict()])

            # Save to data store
            season = lineup.season
            team_id = lineup.team_id
            group_id = lineup.group_id

            lineup_file = Path(self.data_store.data_dir) / "lineups" / f"lineup_team_{team_id}_group_{group_id}_{season.replace('-', '_')}.parquet"
            lineup_file.parent.mkdir(parents=True, exist_ok=True)

            lineup_df.write_parquet(lineup_file)

            # Store in DuckDB
            self.data_store.conn.execute("""
                INSERT OR REPLACE INTO lineup_stats
                (group_id, team_id, season, season_type, games_played, wins, losses,
                 win_percentage, minutes, offensive_rating, defensive_rating, net_rating,
                 plus_minus, effective_field_goal_percentage, true_shooting_percentage,
                 pace, field_goals_made, field_goals_attempted, field_goal_percentage,
                 three_pointers_made, three_pointers_attempted, three_point_percentage,
                 free_throws_made, free_throws_attempted, free_throw_percentage,
                 offensive_rebounds, defensive_rebounds, total_rebounds, assists, steals,
                 blocks, turnovers, personal_fouls, points, last_updated, source, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                lineup.group_id, lineup.team_id, lineup.season, lineup.season_type,
                lineup.games_played, lineup.wins, lineup.losses, lineup.win_percentage,
                lineup.minutes, lineup.offensive_rating, lineup.defensive_rating,
                lineup.net_rating, lineup.plus_minus,
                lineup.effective_field_goal_percentage, lineup.true_shooting_percentage,
                lineup.pace, lineup.field_goals_made, lineup.field_goals_attempted,
                lineup.field_goal_percentage, lineup.three_pointers_made,
                lineup.three_pointers_attempted, lineup.three_point_percentage,
                lineup.free_throws_made, lineup.free_throws_attempted,
                lineup.free_throw_percentage, lineup.offensive_rebounds,
                lineup.defensive_rebounds, lineup.total_rebounds,
                lineup.assists, lineup.steals, lineup.blocks, lineup.turnovers,
                lineup.personal_fouls, lineup.points, lineup.last_updated,
                lineup.source, str(lineup_file)
            ])

            logger.info(f"Stored lineup stats for group {group_id} (team {team_id})")
            return True

        except Exception as e:
            logger.error(f"Error storing lineup stats: {e}")
            return False

    def get_team_roster(self, team_id: int, season: str, season_type: str = "Regular Season") -> Optional[TeamRoster]:
        """Retrieve team roster from stored data."""
        try:
            # Get roster summary
            roster_summary = self.data_store.conn.execute("""
                SELECT * FROM team_rosters
                WHERE team_id = ? AND season = ? AND season_type = ?
            """, [team_id, season, season_type]).fetchdf()

            if len(roster_summary) == 0:
                return None

            roster_data = roster_summary[0]

            # Load detailed roster data
            roster_file = Path(roster_data['file_path'])
            if not roster_file.exists():
                return None

            df = pl.read_parquet(roster_file)

            # Convert to roster objects
            players = []
            for row in df.iter_rows(named=True):
                players.append(RosterInfo(**row))

            return TeamRoster(
                team_id=roster_data['team_id'],
                team_name=roster_data['team_name'],
                team_abbreviation=roster_data['team_abbreviation'],
                season=roster_data['season'],
                season_type=roster_data['season_type'],
                players=players,
                total_players=roster_data['total_players'],
                active_players=roster_data['active_players'],
                injured_players=roster_data['injured_players'],
                total_salary=roster_data['total_salary'],
                salary_cap_space=roster_data['salary_cap_space'],
                last_updated=roster_data['last_updated'],
                source=roster_data['source']
            )

        except Exception as e:
            logger.error(f"Error retrieving team roster: {e}")
            return None

    def get_player_injury_info(self, player_id: int, season: str) -> List[InjuryInfo]:
        """Retrieve all injury information for a player in a season."""
        try:
            injury_records = self.data_store.conn.execute("""
                SELECT * FROM player_injuries
                WHERE player_id = ? AND season = ?
                ORDER BY injury_date DESC, last_updated DESC
            """, [player_id, season]).fetchdf()

            injuries = []
            for record in injury_records:
                injuries.append(InjuryInfo(**record))

            return injuries

        except Exception as e:
            logger.error(f"Error retrieving injury info: {e}")
            return []

    def get_team_injuries(self, team_id: int, season: str, status_filter: Optional[str] = None) -> List[InjuryInfo]:
        """Get injury information for all players on a team."""
        try:
            query = """
                SELECT * FROM player_injuries
                WHERE team_id = ? AND season = ?
            """
            params = [team_id, season]

            if status_filter:
                query += " AND injury_status = ?"
                params.append(status_filter)

            query += " ORDER BY injury_date DESC, last_updated DESC"

            injury_records = self.data_store.conn.execute(query, params).fetchdf()

            injuries = []
            for record in injury_records:
                injuries.append(InjuryInfo(**record))

            return injuries

        except Exception as e:
            logger.error(f"Error retrieving team injuries: {e}")
            return []

    def get_lineup_stats(self, team_id: Optional[int] = None, season: str = "2024-25",
                      season_type: str = "Regular Season", min_games: int = 5) -> List[LineupStats]:
        """Get lineup statistics with filtering options."""
        try:
            query = """
                SELECT * FROM lineup_stats
                WHERE season = ? AND season_type = ? AND games_played >= ?
            """
            params = [season, season_type, min_games]

            if team_id:
                query += " AND team_id = ?"
                params.append(team_id)

            query += " ORDER BY net_rating DESC, games_played DESC"

            lineup_records = self.data_store.conn.execute(query, params).fetchdf()

            lineups = []
            for record in lineup_records:
                lineups.append(LineupStats(**record))

            return lineups

        except Exception as e:
            logger.error(f"Error retrieving lineup stats: {e}")
            return []

    def analyze_lineup_effectiveness(self, team_id: int, season: str,
                                 season_type: str = "Regular Season") -> Optional[LineupAnalysis]:
        """Analyze lineup effectiveness for a team."""
        try:
            lineups = self.get_lineup_stats(team_id, season, season_type, min_games=10)

            if not lineups:
                return None

            # Find most effective lineup (highest net rating)
            most_effective = max(lineups, key=lambda x: x.net_rating or 0)

            # Find most used lineup (most games played)
            most_used = max(lineups, key=lambda x: x.games_played)

            # Calculate statistics
            total_lineups = len(lineups)
            effective_lineups = len([l for l in lineups if (l.net_rating or 0) > 0])
            avg_lineup_size = sum(l.minutes for l in lineups) / sum(l.games_played for l in lineups) if lineups else 0
            avg_lineup_size = avg_lineup_size / 5 if avg_lineup_size > 0 else 0  # Normalize by 5 players

            return LineupAnalysis(
                team_id=team_id,
                season=season,
                season_type=season_type,
                most_effective_lineup=most_effective,
                most_used_lineup=most_used,
                total_lineups=total_lineups,
                effective_lineups=effective_lineups,
                average_lineup_size=avg_lineup_size,
                lineup_diversity_score=0.0,  # TODO: Calculate diversity metric
                last_updated=datetime.now(),
                source="NBA_Analysis"
            )

        except Exception as e:
            logger.error(f"Error analyzing lineup effectiveness: {e}")
            return None

    def get_all_injuries_by_date(self, season: str, from_date: Optional[date] = None,
                             to_date: Optional[date] = None) -> List[InjuryInfo]:
        """Get all injury records in a date range."""
        try:
            query = "SELECT * FROM player_injuries WHERE season = ?"
            params = [season]

            if from_date:
                query += " AND injury_date >= ?"
                params.append(from_date)

            if to_date:
                query += " AND injury_date <= ?"
                params.append(to_date)

            query += " ORDER BY injury_date DESC"

            injury_records = self.data_store.conn.execute(query, params).fetchdf()

            injuries = []
            for record in injury_records:
                injuries.append(InjuryInfo(**record))

            return injuries

        except Exception as e:
            logger.error(f"Error retrieving injuries by date: {e}")
            return []

    def get_injury_trends(self, team_id: int, season: str) -> Dict[str, Any]:
        """Analyze injury trends for a team."""
        try:
            injuries = self.get_team_injuries(team_id, season)

            if not injuries:
                return {}

            # Analyze injury types
            injury_types = {}
            injury_severity = {}
            monthly_injuries = {}

            for injury in injuries:
                # Count injury types
                if injury.injury_type:
                    injury_types[injury.injury_type] = injury_types.get(injury.injury_type, 0) + 1

                # Count severity levels
                if injury.severity:
                    severity_key = injury.severity.value
                    injury_severity[severity_key] = injury_severity.get(severity_key, 0) + 1

                # Count by month
                if injury.injury_date:
                    month_key = injury.injury_date.strftime("%Y-%m")
                    monthly_injuries[month_key] = monthly_injuries.get(month_key, 0) + 1

            return {
                'total_injuries': len(injuries),
                'injury_types': injury_types,
                'injury_severity': injury_severity,
                'monthly_trends': monthly_injuries,
                'currently_injured': len([i for i in injuries if i.injury_status in [
                    InjuryStatus.OUT, InjuryStatus.DAY_TO_DAY,
                    InjuryStatus.QUESTIONABLE, InjuryStatus.DOUBTFUL
                ]]),
                'games_missed_total': sum(i.games_missed or 0 for i in injuries)
            }

        except Exception as e:
            logger.error(f"Error analyzing injury trends: {e}")
            return {}

    def update_player_status(self, player_id: int, team_id: int, season: str,
                            status: str, notes: Optional[str] = None) -> bool:
        """Quickly update player status without full injury record."""
        try:
            # Create minimal injury record
            injury = InjuryInfo(
                player_id=player_id,
                team_id=team_id,
                season=season,
                injury_status=InjuryStatus(status) if status in [s.value for s in InjuryStatus] else InjuryStatus.ACTIVE,
                injury_type=None,
                injury_description=notes,
                injury_date=date.today() if status != "Active" else None,
                expected_return=None,
                return_date=None,
                severity=None,
                games_missed=0,
                consecutive_games_missed=0,
                game_time_decision=None,
                availability_probability=1.0 if status == "Active" else 0.0,
                practice_status=None,
                notes=notes,
                source="Manual_Update",
                confidence_score=0.9
            )

            return self.store_injury_info(injury)

        except Exception as e:
            logger.error(f"Error updating player status: {e}")
            return False