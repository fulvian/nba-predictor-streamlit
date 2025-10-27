"""
🏀 NBA Schedule Fallback - Context7 Compliant Free API Solution

Sistema di fallback robusto che usa NBA API ufficiale gratuita per ottenere
il calendario NBA 2025-2026 quando The Odds API è fuori quota.

Basato su Context7 nba_api documentation:
- /stats/scheduleleaguev2 - Calendario completo stagionale
- /stats/scheduleleaguev2int - Versione internazionale
- Dati reali NBA.com senza API key
"""

import requests
from datetime import datetime, date
from typing import Dict, List, Optional
import json
import time
from nba_timezone_utils import NBATimezoneManager

class NBAScheduleFallback:
    """
    Client per il calendario NBA usando NBA API ufficiale gratuita.

    Fornisce accesso al calendario NBA 2025-2026 senza bisogno di API key.
    Basato su best practices Context7 per nba_api.
    """

    def __init__(self):
        """Initialize the NBA schedule fallback client."""
        self.base_url = "https://stats.nba.com/stats"
        self.headers = {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Host': 'stats.nba.com',
            'Origin': 'https://www.nba.com',
            'Referer': 'https://www.nba.com/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'x-nba-stats-origin': 'stats',
            'x-nba-stats-token': 'true'
        }
        self.timezone_manager = NBATimezoneManager()

    def get_nba_schedule(self, season: str = "2025-26", league_id: str = "00") -> List[Dict]:
        """
        Get the complete NBA schedule for a season using the official NBA API.

        Args:
            season: Season in format "YYYY-YY" (e.g., "2025-26")
            league_id: League ID ("00" for NBA)

        Returns:
            List of games with comprehensive information
        """
        try:
            print(f"🔄 Fetching NBA schedule from official API for season {season}...")

            # Use scheduleleaguev2 endpoint from Context7 documentation
            url = f"{self.base_url}/scheduleleaguev2"
            params = {
                'LeagueID': league_id,
                'Season': season
            }

            response = requests.get(url, params=params, headers=self.headers, timeout=30)

            if response.status_code == 200:
                data = response.json()
                return self._process_schedule_response(data, season)
            else:
                print(f"❌ NBA API Error: {response.status_code}")
                print(f"Response: {response.text[:200]}...")
                return []

        except Exception as e:
            print(f"❌ Exception fetching NBA schedule: {e}")
            return []

    def _process_schedule_response(self, data: Dict, season: str) -> List[Dict]:
        """
        Process the NBA API response into our game format.

        Args:
            data: Raw response from NBA API
            season: Season string for reference

        Returns:
            List of processed games
        """
        try:
            # Extract schedule data from response structure
            result_sets = data.get('resultSets', [])
            schedule_data = None

            for result_set in result_sets:
                if result_set.get('name') == 'LeagueSchedule' or 'Schedule' in result_set.get('name', ''):
                    schedule_data = result_set
                    break

            if not schedule_data:
                print("❌ No schedule data found in NBA API response")
                return []

            headers = schedule_data.get('headers', [])
            row_set = schedule_data.get('rowSet', [])

            # Find column indices
            col_indices = {}
            for i, header in enumerate(headers):
                col_indices[header] = i

            games = []
            for row in row_set:
                try:
                    game = self._extract_game_info(row, col_indices, season)
                    if game:
                        games.append(game)

                except Exception as e:
                    print(f"⚠️ Error processing game row: {e}")
                    continue

            print(f"✅ Processed {len(games)} games from NBA API")
            return games

        except Exception as e:
            print(f"❌ Error processing schedule response: {e}")
            return []

    def _extract_game_info(self, row: List, col_indices: Dict, season: str) -> Optional[Dict]:
        """
        Extract game information from a row of NBA API data.

        Args:
            row: Single row from NBA API response
            col_indices: Column name to index mapping
            season: Season string

        Returns:
            Game information dictionary or None if extraction fails
        """
        try:
            # Extract basic game info - handle different possible column names
            game_id = self._get_column_value(row, col_indices, ['GAME_ID', 'GameID'])
            game_date = self._get_column_value(row, col_indices, ['GAME_DATE', 'GameDate'])

            if not game_id or not game_date:
                return None

            # Extract team information
            home_team_name = self._get_column_value(row, col_indices, [
                'HOME_TEAM_NAME', 'HOME_TEAM', 'HomeTeam'
            ])
            away_team_name = self._get_column_value(row, col_indices, [
                'VISITOR_TEAM_NAME', 'AWAY_TEAM', 'AwayTeam'
            ])

            if not home_team_name or not away_team_name:
                # Try to extract from matchup
                matchup = self._get_column_value(row, col_indices, ['MATCHUP', 'Matchup'])
                if matchup:
                    home_team_name, away_team_name = self._parse_matchup(matchup)

            if not home_team_name or not away_team_name:
                return None

            # Parse game date and handle timezone
            utc_datetime = self._parse_game_date(game_date)
            if not utc_datetime:
                return None

            # Get timezone info for both teams
            home_local, home_tz = self.timezone_manager.convert_utc_to_local(utc_datetime, home_team_name)
            away_local, away_tz = self.timezone_manager.convert_utc_to_local(utc_datetime, away_team_name)

            # Extract scores if available
            home_score = self._get_column_value(row, col_indices, ['HOME_TEAM_SCORE', 'HOME_SCORE', 'HomeScore'])
            away_score = self._get_column_value(row, col_indices, ['VISITOR_TEAM_SCORE', 'AWAY_SCORE', 'AwayScore'])

            # Determine game status
            status = "Scheduled"
            if home_score and away_score:
                status = "Completed"

            # Create game object
            game = {
                'game_id': f"NBA_{game_id}",
                'date': home_local.strftime('%Y-%m-%d'),
                'time': home_local.strftime('%H:%M'),
                'time_utc': utc_datetime.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'away_team': away_team_name,
                'home_team': home_team_name,
                'status': status,
                'score': f"{away_score}-{home_score}" if home_score and away_score else '',
                'season': season,
                'league_id': self._get_column_value(row, col_indices, ['LEAGUE_ID', 'LeagueID'], '00'),
                'home_timezone': home_tz,
                'away_timezone': away_tz,
                'home_local_time': home_local.strftime('%Y-%m-%d %H:%M %Z'),
                'away_local_time': away_local.strftime('%Y-%m-%d %H:%M %Z'),
                'utc_datetime': utc_datetime.isoformat(),
                'source': 'NBA Official API (Free)',
                'api_endpoint': 'stats.nba.com/stats/scheduleleaguev2',
                'bookmakers_count': 0,  # NBA API doesn't provide odds
                'odds': {}  # No odds available from schedule API
            }

            return game

        except Exception as e:
            print(f"⚠️ Error extracting game info: {e}")
            return None

    def _get_column_value(self, row: List, col_indices: Dict, possible_names: List[str], default=None):
        """Get value from row using multiple possible column names."""
        for name in possible_names:
            if name in col_indices:
                idx = col_indices[name]
                if idx < len(row):
                    return row[idx]
        return default

    def _parse_matchup(self, matchup: str) -> tuple:
        """
        Parse matchup string to extract team names.

        Args:
            matchup: Matchup string (e.g., "LAL @ DEN" or "LAL vs DEN")

        Returns:
            Tuple of (away_team, home_team)
        """
        try:
            if ' @ ' in matchup:
                away_team, home_team = matchup.split(' @ ', 1)
            elif ' vs ' in matchup:
                home_team, away_team = matchup.split(' vs ', 1)
            else:
                # Fallback: try to parse other formats
                parts = matchup.split()
                if len(parts) >= 3:
                    away_team = ' '.join(parts[:len(parts)//2])
                    home_team = ' '.join(parts[len(parts)//2:])
                else:
                    return None, None

            return away_team.strip(), home_team.strip()

        except Exception as e:
            print(f"⚠️ Error parsing matchup '{matchup}': {e}")
            return None, None

    def _parse_game_date(self, game_date_str: str) -> Optional[datetime]:
        """
        Parse game date string to UTC datetime.

        Args:
            game_date_str: Date string from NBA API

        Returns:
            UTC datetime object or None
        """
        try:
            # NBA API dates are usually in UTC format
            if 'T' in game_date_str:
                # ISO format with time
                if game_date_str.endswith('Z'):
                    utc_datetime = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
                else:
                    utc_datetime = datetime.fromisoformat(game_date_str)
                    # Assume UTC if no timezone specified
                    if utc_datetime.tzinfo is None:
                        utc_datetime = utc_datetime.replace(tzinfo=self.timezone_manager.utc)
            else:
                # Date only, assume noon UTC as default time
                date_obj = datetime.fromisoformat(game_date_str)
                utc_datetime = date_obj.replace(hour=12, minute=0, tzinfo=self.timezone_manager.utc)

            return utc_datetime

        except Exception as e:
            print(f"⚠️ Error parsing game date '{game_date_str}': {e}")
            return None

    def get_games_for_date(self, target_date: date, season: str = "2025-26") -> List[Dict]:
        """
        Get NBA games for a specific date.

        Args:
            target_date: Date to get games for
            season: Season string

        Returns:
            List of games on the specified date
        """
        try:
            all_games = self.get_nba_schedule(season)
            target_date_str = target_date.strftime('%Y-%m-%d')

            # Filter games by date (checking both local and UTC dates)
            target_games = []
            for game in all_games:
                game_date = game.get('date', '')
                if game_date == target_date_str:
                    target_games.append(game)

            print(f"🏀 Found {len(target_games)} NBA games for {target_date_str}")
            return target_games

        except Exception as e:
            print(f"❌ Error getting games for date {target_date}: {e}")
            return []

def test_nba_schedule_fallback():
    """Test the NBA schedule fallback system."""
    print("🧪 Testing NBA Schedule Fallback System...")

    fallback = NBAScheduleFallback()

    # Test getting games for today
    today = date.today()
    games_today = fallback.get_games_for_date(today)

    print(f"\n📅 Games for {today}:")
    for game in games_today:
        print(f"  🏀 {game['away_team']} @ {game['home_team']}")
        print(f"     Time: {game['time']} ({game['home_timezone']})")
        print(f"     UTC: {game['time_utc']}")
        print(f"     Status: {game['status']}")
        print()

    # Test getting full schedule
    print(f"📊 Total games in season: {len(fallback.get_nba_schedule())}")

if __name__ == "__main__":
    test_nba_schedule_fallback()