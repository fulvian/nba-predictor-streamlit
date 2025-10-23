"""
🏀 Advanced NBA Schedule Provider - Context7 Compliant
Multi-source NBA schedule detection using official and reliable APIs
Based on comprehensive research of available NBA data sources
"""

import requests
import json
import time
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
import traceback

class AdvancedNBAScheduleProvider:
    """
    Advanced NBA Schedule Provider with multiple reliable data sources
    Context7 compliant implementation based on official NBA APIs
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; NBA-Predictor/1.0)',
            'Accept': 'application/json, text/plain',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache'
        })

        # Team name mapping for normalization
        self.team_name_mapping = {
            'Lakers': 'Los Angeles Lakers',
            'Warriors': 'Golden State Warriors',
            'Celtics': 'Boston Celtics',
            'Heat': 'Miami Heat',
            'Nets': 'Brooklyn Nets',
            'Bucks': 'Milwaukee Bucks',
            'Suns': 'Phoenix Suns',
            'Mavericks': 'Dallas Mavericks',
            '76ers': 'Philadelphia 76ers',
            'Nuggets': 'Denver Nuggets',
            'Clippers': 'Los Angeles Clippers',
            'Bulls': 'Chicago Bulls',
            'Raptors': 'Toronto Raptors',
            'Knicks': 'New York Knicks',
            'Cavaliers': 'Cleveland Cavaliers',
            'Pacers': 'Indiana Pacers',
            'Pistons': 'Detroit Pistons',
            'Hornets': 'Charlotte Hornets',
            'Wizards': 'Washington Wizards',
            'Magic': 'Orlando Magic',
            'Hawks': 'Atlanta Hawks',
            'Grizzlies': 'Memphis Grizzlies',
            'Pelicans': 'New Orleans Pelicans',
            'Spurs': 'San Antonio Spurs',
            'Kings': 'Sacramento Kings',
            'Timberwolves': 'Minnesota Timberwolves',
            'Thunder': 'Oklahoma City Thunder',
            'Trail Blazers': 'Portland Trail Blazers',
            'Jazz': 'Utah Jazz',
            'Rockets': 'Houston Rockets'
        }

    def get_scheduled_games(self, target_date: str) -> List[Dict[str, Any]]:
        """
        Get scheduled games for a specific date using multiple reliable sources

        Args:
            target_date: Date in 'YYYY-MM-DD' format

        Returns:
            List of game dictionaries with complete information
        """
        print(f"🏀 Advanced NBA Schedule Detection for {target_date}")
        print("=" * 60)

        all_games = []

        # Source 1: NBA Data API (suggested from research)
        games = self._try_nba_data_api(target_date)
        if games:
            all_games.extend(games)
            print(f"✅ NBA Data API: Found {len(games)} games")

        # Source 2: NBA.com JSON API (official)
        if not all_games:
            games = self._try_nba_json_api(target_date)
            if games:
                all_games.extend(games)
                print(f"✅ NBA.com JSON: Found {len(games)} games")

        # Source 3: ESPN API (reliable fallback)
        if not all_games:
            games = self._try_espn_api(target_date)
            if games:
                all_games.extend(games)
                print(f"✅ ESPN API: Found {len(games)} games")

        # Source 4: RapidAPI (if available)
        if not all_games:
            games = self._try_rapid_api(target_date)
            if games:
                all_games.extend(games)
                print(f"✅ RapidAPI: Found {len(games)} games")

        print(f"📊 FINAL RESULT: {len(all_games)} games found for {target_date}")

        if all_games:
            for i, game in enumerate(all_games, 1):
                home = game.get('home_team', 'Unknown')
                away = game.get('away_team', 'Unknown')
                time = game.get('time', 'TBD')
                source = game.get('source', 'unknown')
                print(f"   {i}. {away} @ {home} ({time}) [Source: {source}]")
        else:
            print(f"   ❌ NO GAMES FOUND for {target_date}")
            print(f"   🔍 This could mean:")
            print(f"      • No NBA games scheduled for this date")
            print(f"      • Season hasn't started yet")
            print(f"      • API limitations for future dates")

        return all_games

    def _try_nba_data_api(self, target_date: str) -> List[Dict[str, Any]]:
        """Try NBA Data API endpoint from research findings"""
        try:
            print(f"🔍 Trying NBA Data API for {target_date}...")

            # Convert date for API
            date_obj = datetime.strptime(target_date, '%Y-%m-%d')
            year = date_obj.year

            # Determine season (NBA seasons span two years)
            if date_obj.month >= 10:
                season = f"{year}-{str(year + 1)[-2:]}"
            else:
                season = f"{year - 1}-{str(year)[-2:]}"

            # NBA Data API endpoint (from research)
            url = f"https://api.nba.net/json/cms/{season}/league/schedule.json"

            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                games = []

                if 'sports_content' in data and 'schedule' in data['sports_content']:
                    schedule = data['sports_content']['schedule']
                    if 'games' in schedule:
                        for game_data in schedule['games']:
                            game_date = game_data.get('date', '')
                            if target_date in game_date:
                                home_team = self._normalize_team_name(game_data.get('home', {}).get('name', ''))
                                away_team = self._normalize_team_name(game_data.get('away', {}).get('name', ''))

                                games.append({
                                    'date': target_date,
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'time': game_data.get('time', 'TBD'),
                                    'game_id': game_data.get('id', f"NBA_{target_date}_{len(games)}"),
                                    'home_team_id': self._get_team_id(home_team),
                                    'away_team_id': self._get_team_id(away_team),
                                    'odds': [],
                                    'source': 'nba_data_api'
                                })

                print(f"   📊 NBA Data API response processed: {len(games)} games")
                return games
            else:
                print(f"   ⚠️ NBA Data API returned status {response.status_code}")
                return []

        except Exception as e:
            print(f"   ❌ NBA Data API error: {e}")
            return []

    def _try_nba_json_api(self, target_date: str) -> List[Dict[str, Any]]:
        """Try NBA.com official JSON API"""
        try:
            print(f"🔍 Trying NBA.com JSON API for {target_date}...")

            # NBA.com score endpoint
            date_obj = datetime.strptime(target_date, '%Y-%m-%d')
            formatted_date = date_obj.strftime('%Y%m%d')

            url = f"https://cdn.nba.com/static/json/liveData/scoreboard/{formatted_date}_scoreboard.json"

            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                games = []

                if 'scoreboard' in data and 'games' in data['scoreboard']:
                    for game_data in data['scoreboard']['games']:
                        home_team = self._normalize_team_name(game_data.get('homeTeam', {}).get('fullName', ''))
                        away_team = self._normalize_team_name(game_data.get('awayTeam', {}).get('fullName', ''))

                        games.append({
                            'date': target_date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'time': game_data.get('gameTimeUTC', 'TBD'),
                            'game_id': game_data.get('gameId', f"NBA_{target_date}_{len(games)}"),
                            'home_team_id': self._get_team_id(home_team),
                            'away_team_id': self._get_team_id(away_team),
                            'odds': [],
                            'source': 'nba_com_json'
                        })

                print(f"   📊 NBA.com JSON response processed: {len(games)} games")
                return games
            else:
                print(f"   ⚠️ NBA.com JSON returned status {response.status_code}")
                return []

        except Exception as e:
            print(f"   ❌ NBA.com JSON error: {e}")
            return []

    def _try_espn_api(self, target_date: str) -> List[Dict[str, Any]]:
        """Try ESPN API as reliable fallback"""
        try:
            print(f"🔍 Trying ESPN API for {target_date}...")

            date_obj = datetime.strptime(target_date, '%Y-%m-%d')
            year = date_obj.year

            # ESPN scoreboard API
            url = f"http://site.api.espn.com/apis/site/v2/scoreboard?sport=basketball&dates={target_date}"

            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                games = []

                if 'sports' in data:
                    for sport in data['sports']:
                        if sport['name'] == 'basketball' and 'leagues' in sport:
                            for league in sport['leagues']:
                                if league['name'] == 'NBA' and 'events' in league:
                                    for event in league['events']:
                                        if event.get('date', '')[:10] == target_date:
                                            home_team = self._normalize_team_name(event.get('competitions', [{}])[0].get('competitor', {}).get('name', ''))
                                            away_team = self._normalize_team_name(event.get('competitions', [{}])[1].get('competitor', {}).get('name', ''))

                                            games.append({
                                                'date': target_date,
                                                'home_team': home_team,
                                                'away_team': away_team,
                                                'time': event.get('time', 'TBD'),
                                                'game_id': event.get('id', f"ESPN_{target_date}_{len(games)}"),
                                                'home_team_id': self._get_team_id(home_team),
                                                'away_team_id': self._get_team_id(away_team),
                                                'odds': [],
                                                'source': 'espn_api'
                                            })

                print(f"   📊 ESPN API response processed: {len(games)} games")
                return games
            else:
                print(f"   ⚠️ ESPN API returned status {response.status_code}")
                return []

        except Exception as e:
            print(f"   ❌ ESPN API error: {e}")
            return []

    def _try_rapid_api(self, target_date: str) -> List[Dict[str, Any]]:
        """Try RapidAPI as additional source"""
        try:
            print(f"🔍 Trying RapidAPI for {target_date}...")

            # This is a placeholder for when RapidAPI or similar service is available
            # Implementation would depend on actual API documentation

            print(f"   ⚠️ RapidAPI not implemented yet")
            return []

        except Exception as e:
            print(f"   ❌ RapidAPI error: {e}")
            return []

    def _normalize_team_name(self, team_name: str) -> str:
        """Normalize team names using mapping"""
        if not team_name:
            return "Unknown"

        # Remove common variations and map to standard names
        clean_name = team_name.strip().replace('LA', 'Los Angeles').replace('NY', 'New York')

        # Check mapping
        for key, standard_name in self.team_name_mapping.items():
            if key.lower() in clean_name.lower():
                return standard_name

        return clean_name

    def _get_team_id(self, team_name: str) -> int:
        """Generate consistent team ID for normalized names"""
        # Simple hash-based ID generation for consistency
        team_hash = abs(hash(team_name.lower())) % 1000000
        return 1610000000 + team_hash

    def test_date(self, target_date: str):
        """Test the schedule detection for a specific date"""
        print(f"\n🧪 TESTING SCHEDULE DETECTION FOR {target_date}")
        print("=" * 50)

        games = self.get_scheduled_games(target_date)

        print(f"\n📊 SUMMARY FOR {target_date}:")
        print(f"   Total games found: {len(games)}")
        print(f"   Sources used: {list(set(g['source'] for g in games))}")

        return games

# Example usage and testing
if __name__ == "__main__":
    provider = AdvancedNBAScheduleProvider()

    # Test for today
    today = date.today().strftime('%Y-%m-%d')
    provider.test_date(today)

    # Test for October 25, 2025
    provider.test_date("2025-10-25")