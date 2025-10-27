#!/usr/bin/env python3
"""
🏀 NBA Mock Injury Data Generator
Generates realistic mock injury data for testing when real data is not available.
"""

import random
from datetime import date, datetime, timedelta
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class MockInjuryGenerator:
    """Generates realistic mock NBA injury data for testing."""

    def __init__(self):
        """Initialize mock injury generator."""
        self.injury_types = [
            "Ankle Sprain", "Knee Contusion", "Hamstring Strain", "Back Spasms",
            "Concussion", "Wrist Sprain", "Shoulder Strain", "Achilles Tendinitis",
            "Calf Strain", "Foot Injury", "Hip Flexor", "Neck Strain",
            "Groin Strain", "Elbow Sprain", "Finger Fracture", "Toe Injury"
        ]

        self.injury_descriptions = [
            "Suffered during practice", "Game-related injury", "Pre-existing condition",
            "Re-injury of previous issue", "Contact injury", "Non-contact injury",
            "Overuse injury", "Acute injury", "Chronic condition"
        ]

        # Realistic NBA player names for testing
        self.nba_players = [
            "LeBron James", "Stephen Curry", "Kevin Durant", "Giannis Antetokounmpo",
            "Joel Embiid", "Luka Dončić", "Nikola Jokić", "Jayson Tatum",
            "Kawhi Leonard", "Damian Lillard", "Anthony Davis", "Jimmy Butler",
            "Devin Booker", "Karl-Anthony Towns", "Bradley Beal", "Donovan Mitchell",
            "Bam Adebayo", "Jaylen Brown", "Paul George", "Rudy Gobert",
            "Trae Young", "Zion Williamson", "Ja Morant", "De'Aaron Fox",
            "Shai Gilgeous-Alexander", "Tyrese Haliburton", "Lauri Markkanen", "Paolo Banchero"
        ]

        self.nba_teams = {
            "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "CLE": "Cleveland Cavaliers",
            "NOP": "New Orleans Pelicans", "CHI": "Chicago Bulls", "DAL": "Dallas Mavericks",
            "DEN": "Denver Nuggets", "GSW": "Golden State Warriors", "HOU": "Houston Rockets",
            "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MIA": "Miami Heat",
            "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves", "BKN": "Brooklyn Nets",
            "NYK": "New York Knicks", "ORL": "Orlando Magic", "IND": "Indiana Pacers",
            "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns", "POR": "Portland Trail Blazers",
            "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs", "OKC": "Oklahoma City Thunder",
            "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "MEM": "Memphis Grizzlies",
            "WAS": "Washington Wizards", "DET": "Detroit Pistons", "CHA": "Charlotte Hornets"
        }

    def generate_mock_injuries(self, start_date: date, end_date: date, count: int = 20) -> List[Dict[str, Any]]:
        """Generate realistic mock injury data for testing."""
        logger.info(f"🎭 Generating {count} mock injuries for {start_date} to {end_date}")

        injuries = []

        for i in range(count):
            # Random player and team
            player = random.choice(self.nba_players)
            team_abbrev = random.choice(list(self.nba_teams.keys()))
            team_name = self.nba_teams[team_abbrev]

            # Random injury details
            injury_type = random.choice(self.injury_types)
            injury_description = random.choice(self.injury_descriptions)
            injury_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))

            # Injury status with realistic probabilities
            status_weights = ['Questionable', 'Day-to-Day', 'Doubtful', 'Out', 'Probable']
            injury_status = random.choices(status_weights, weights=[30, 25, 15, 15, 15])[0]

            # Calculate return date based on severity
            if injury_status == 'Out':
                days_out = random.randint(7, 45)  # 1-6 weeks
            elif injury_status == 'Doubtful':
                days_out = random.randint(3, 14)  # 3 days to 2 weeks
            elif injury_status == 'Questionable':
                days_out = random.randint(1, 7)   # 1-7 days
            else:
                days_out = random.randint(0, 3)   # 0-3 days

            expected_return = injury_date + timedelta(days=days_out)

            # Games missed (rough estimate)
            games_missed = max(0, days_out // 1)  # Assume roughly 1 game per day

            # Availability probability based on status
            availability_prob = {
                'Out': 0.0,
                'Doubtful': 0.25,
                'Questionable': 0.5,
                'Day-to-Day': 0.6,
                'Probable': 0.75
            }.get(injury_status, 0.5)

            # Create injury record
            injury = {
                'player_name': player,
                'player_id': random.randint(100000, 999999),  # Mock ID
                'team_id': random.randint(1610612700, 1610612766),  # Mock NBA team ID
                'team_name': team_name,
                'team_abbreviation': team_abbrev,
                'injury_status': injury_status,
                'injury_type': injury_type,
                'injury_description': f"{injury_type} - {injury_description}",
                'injury_date': injury_date,
                'expected_return': expected_return,
                'return_date': None,  # Not returned yet in most cases
                'games_missed': games_missed,
                'availability_probability': availability_prob,
                'source': 'mock_generator',
                'confidence_score': 0.7,  # Mock data confidence
                'notes': f"Mock injury data for testing - Status: {injury_status}",
                'last_updated': datetime.now().isoformat()
            }

            injuries.append(injury)

        logger.info(f"✅ Generated {len(injuries)} mock injuries")
        return injuries

    def generate_team_injury_summary(self, injuries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate team injury summary from mock data."""
        team_injuries = {}

        for injury in injuries:
            team = injury['team_abbreviation']
            if team not in team_injuries:
                team_injuries[team] = []

            team_injuries[team].append(injury)

        summaries = []
        for team, team_inj_list in team_injuries.items():
            # Calculate impact score
            total_injured = len(team_inj_list)
            key_players = len([inj for inj in team_inj_list if inj['availability_probability'] < 0.5])
            avg_impact = sum(1 - inj['availability_probability'] for inj in team_inj_list) / len(team_inj_list) if team_inj_list else 0

            impact_score = (total_injured * 0.3 + key_players * 0.5 + avg_impact * 0.2) * 100

            summary = {
                'team_id': next((inj['team_id'] for inj in team_inj_list), None),
                'team_name': next((inj['team_name'] for inj in team_inj_list), team),
                'team_abbreviation': team,
                'total_injured': total_injured,
                'key_players_injured': key_players,
                'injury_impact_score': min(100, impact_score),
                'injury_date': max(inj['injury_date'] for inj in team_inj_list),
                'last_updated': datetime.now().isoformat(),
                'source': 'mock_generator'
            }

            summaries.append(summary)

        return sorted(summaries, key=lambda x: x['injury_impact_score'], reverse=True)


def main():
    """Main function to test mock injury generator."""
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("🎭 NBA MOCK INJURY GENERATOR TEST")
    print("=" * 80)

    generator = MockInjuryGenerator()

    # Generate mock injuries
    end_date = date.today()
    start_date = end_date - timedelta(days=7)

    mock_injuries = generator.generate_mock_injuries(start_date, end_date, count=15)

    if mock_injuries:
        print(f"\n✅ Generated {len(mock_injuries)} mock injuries")
        print(f"\n📝 Sample injuries:")

        for i, injury in enumerate(mock_injuries[:5], 1):
            print(f"   {i}. {injury['player_name']} ({injury['team_abbreviation']})")
            print(f"      Status: {injury['injury_status']}")
            print(f"      Type: {injury['injury_type']}")
            print(f"      Games Missed: {injury['games_missed']}")
            print(f"      Availability: {injury['availability_probability']:.0%}")
            print(f"      Expected Return: {injury['expected_return']}")
            print()

        # Generate team summaries
        team_summaries = generator.generate_team_injury_summary(mock_injuries)

        print(f"📊 Team Injury Impact Summary:")
        for summary in team_summaries[:5]:
            print(f"   {summary['team_abbreviation']}: Impact Score {summary['injury_impact_score']:.1f}")
            print(f"      Total Injured: {summary['total_injured']}")
            print(f"      Key Players: {summary['key_players_injured']}")
            print()

    print(f"🎯 Mock injury generator test completed!")


if __name__ == "__main__":
    main()