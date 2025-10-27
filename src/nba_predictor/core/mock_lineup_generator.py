#!/usr/bin/env python3
"""
🏀 NBA Mock Lineup Data Generator
Generates realistic mock lineup data for testing when real data is not available.
"""

import random
import logging
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass

from .roster_injury_schemas import LineupStats

logger = logging.getLogger(__name__)

class MockLineupGenerator:
    """Generates realistic mock NBA lineup data for testing."""

    def __init__(self):
        """Initialize mock lineup generator."""
        # Sample NBA player names for lineups
        self.nba_players = [
            "LeBron James", "Stephen Curry", "Kevin Durant", "Giannis Antetokounmpo",
            "Joel Embiid", "Luka Dončić", "Nikola Jokić", "Jayson Tatum",
            "Kawhi Leonard", "Damian Lillard", "Anthony Davis", "Jimmy Butler",
            "Devin Booker", "Karl-Anthony Towns", "Bradley Beal", "Donovan Mitchell",
            "Bam Adebayo", "Jaylen Brown", "Paul George", "Rudy Gobert",
            "Trae Young", "Zion Williamson", "Ja Morant", "De'Aaron Fox",
            "Shai Gilgeous-Alexander", "Tyrese Haliburton", "Lauri Markkanen", "Paolo Banchero",
            "Anthony Edwards", "Ja Morant", "Jaren Jackson Jr.", "Alperen Şengün"
        ]

        self.nba_teams = {
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

    def generate_mock_lineups(self, team_id: int, season: str, count: int = 15) -> List[Dict[str, Any]]:
        """Generate realistic mock lineup data for testing."""
        logger.info(f"🎭 Generating {count} mock lineups for team {team_id} in {season}")

        lineups = []
        team_name = self.nba_teams.get(team_id, f"Team {team_id}")
        team_abbrev = team_name[:3].upper()

        for i in range(count):
            # Generate lineup name (mock group of players)
            lineup_players = random.sample(self.nba_players, 5)
            lineup_name = f"Lineup {i+1}: {' | '.join(lineup_players[:2])}..."

            # Games played with realistic distribution
            games_played = random.randint(5, 40)

            # Win percentage with realistic distribution
            win_percentage = max(0.0, min(1.0, random.gauss(0.5, 0.2)))
            wins = int(games_played * win_percentage)
            losses = games_played - wins

            # Minutes played (per game)
            minutes = random.uniform(8.0, 25.0)

            # Shooting statistics (realistic ranges)
            field_goals_made = random.uniform(15.0, 45.0)
            field_goals_attempted = random.uniform(30.0, 90.0)
            field_goal_percentage = field_goals_made / field_goals_attempted if field_goals_attempted > 0 else random.uniform(0.4, 0.5)

            three_points_made = random.uniform(5.0, 18.0)
            three_points_attempted = random.uniform(12.0, 40.0)
            three_point_percentage = three_points_made / three_points_attempted if three_points_attempted > 0 else random.uniform(0.3, 0.4)

            free_throws_made = random.uniform(8.0, 20.0)
            free_throws_attempted = random.uniform(10.0, 25.0)
            free_throw_percentage = free_throws_made / free_throws_attempted if free_throws_attempted > 0 else random.uniform(0.7, 0.8)

            # Rebounding
            offensive_rebounds = random.uniform(3.0, 12.0)
            defensive_rebounds = random.uniform(20.0, 35.0)
            total_rebounds = offensive_rebounds + defensive_rebounds

            # Other statistics
            assists = random.uniform(15.0, 30.0)
            turnovers = random.uniform(8.0, 18.0)
            steals = random.uniform(4.0, 12.0)
            blocks = random.uniform(2.0, 8.0)
            blocked_attempts = random.uniform(2.0, 6.0)
            personal_fouls = random.uniform(15.0, 25.0)
            personal_fouls_drawn = random.uniform(15.0, 25.0)

            # Points (calculated from shooting)
            points = (field_goals_made - three_points_made) * 2 + three_points_made * 3 + free_throws_made

            # Plus/minus (correlated with win percentage)
            plus_minus = (win_percentage - 0.5) * random.uniform(5.0, 15.0)

            # Fix free throw percentage calculation
            free_throw_percentage = min(1.0, free_throws_made / free_throws_attempted) if free_throws_attempted > 0 else 0.0

            # Create lineup stats
            lineup_stats = LineupStats(
                group_id=i + 1,
                group_name=lineup_name,
                team_id=team_id,
                team_abbreviation=team_abbrev,
                games_played=games_played,
                wins=wins,
                losses=losses,
                win_percentage=round(win_percentage, 3),
                minutes=round(minutes, 1),
                field_goals_made=round(field_goals_made, 1),
                field_goals_attempted=round(field_goals_attempted, 1),
                field_goal_percentage=round(field_goal_percentage, 3),
                three_points_made=round(three_points_made, 1),
                three_points_attempted=round(three_points_attempted, 1),
                three_point_percentage=round(three_point_percentage, 3) if three_points_attempted > 0 else None,
                free_throws_made=round(free_throws_made, 1),
                free_throws_attempted=round(free_throws_attempted, 1),
                free_throw_percentage=round(free_throw_percentage, 3) if free_throws_attempted > 0 else None,
                offensive_rebounds=round(offensive_rebounds, 1),
                defensive_rebounds=round(defensive_rebounds, 1),
                total_rebounds=round(total_rebounds, 1),
                assists=round(assists, 1),
                turnovers=round(turnovers, 1),
                steals=round(steals, 1),
                blocks=round(blocks, 1),
                blocked_attempts=round(blocked_attempts, 1),
                personal_fouls=round(personal_fouls, 1),
                personal_fouls_drawn=round(personal_fouls_drawn, 1),
                points=round(points, 1),
                plus_minus=round(plus_minus, 1),
                season=season,
                season_type="Regular Season",
                last_updated=datetime.now().isoformat()
            )

            lineups.append(lineup_stats.dict())

        logger.info(f"✅ Generated {len(lineups)} mock lineups for {team_name}")
        return lineups

    def generate_league_lineups(self, season: str, team_count: int = 30) -> Dict[str, Any]:
        """Generate mock lineup data for multiple teams."""
        logger.info(f"🎭 Generating mock league lineups for {season}")

        all_lineups = {}
        team_ids = list(self.nba_teams.keys())[:team_count]

        for team_id in team_ids:
            team_lineups = self.generate_mock_lineups(team_id, season, count=random.randint(10, 20))
            all_lineups[team_id] = team_lineups

        total_lineups = sum(len(lineups) for lineups in all_lineups.values())

        summary = {
            'success': True,
            'season': season,
            'teams_processed': len(all_lineups),
            'total_lineups': total_lineups,
            'average_lineups_per_team': round(total_lineups / len(all_lineups), 1) if all_lineups else 0,
            'timestamp': datetime.now().isoformat(),
            'source': 'mock_generator'
        }

        logger.info(f"✅ Generated mock league lineups: {len(all_lineups)} teams, {total_lineups} total lineups")
        return {
            'summary': summary,
            'lineups': all_lineups
        }