#!/usr/bin/env python3
"""
🏀 Multi-Source NBA Data Provider - Sistema Ibrido Ottimizzato

Integra multiple API per fornire dati NBA completi:
- BallDontLie API: Games, Teams (5 req/min)
- NBA API (swar): Statistiche avanzate (~100 req/min)
- The Odds API: Quote scommesse
- NBA-injury-data (RapidAPI): Injury reports ($5-10/mese)

Strategia prioritaria: API gratuite → API a pagamento solo per dati critici
"""

import logging
import os
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import polars as pl

# Import API
from balldontlie import BalldontlieAPI
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    playercareerstats, leaguegamelog, commonteamroster,
    leaguestandings, playerdashboardbyyearoveryear
)
import requests

logger = logging.getLogger(__name__)

class MultiSourceNBADataProvider:
    """
    Provider dati NBA multi-sorgente con strategia di priorità ottimizzata.

    Priorità API:
    1. NBA API (swar) - FREE, statistiche complete
    2. BallDontLie API - FREE, games/teams
    3. The Odds API - FREE/low cost, quote
    4. NBA-injury-data (RapidAPI) - LOW COST, injury reports
    """

    def __init__(self):
        """Inizializza il provider multi-sorgente."""
        # BallDontLie API
        self.balldontlie_api_key = os.getenv('BALLDONTLIE_API_KEY', '0baa5751-350b-44b1-bb0b-7808683e4c96')
        self.bdl_client = BalldontlieAPI(api_key=self.balldontlie_api_key)

        # The Odds API
        self.the_odds_api_key = os.getenv('THE_ODDS_API_KEY', 'd01e24415744d440168e0a489f233aac')

        # Rate limiting management
        self.last_api_calls = {
            'balldontlie': 0,
            'the_odds': 0
        }
        self.rate_limits = {
            'balldontlie': 12,  # 5 req/min = 12 sec interval
            'the_odds': 1       # 1 sec per sicurezza
        }

        # Cache dati
        self._cache = {}
        self._cache_expiry = {}

        logger.info("Multi-Source NBA Data Provider initialized")

    def _rate_limit_wait(self, api_name: str) -> None:
        """Implementa rate limiting per API."""
        current_time = time.time()
        last_call = self.last_api_calls.get(api_name, 0)
        min_interval = self.rate_limits.get(api_name, 1)

        time_since_last = current_time - last_call
        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            logger.debug(f"Rate limiting {api_name}: waiting {wait_time:.1f}s")
            time.sleep(wait_time)

        self.last_api_calls[api_name] = time.time()

    def _get_cached_data(self, cache_key: str, expiry_minutes: int = 60) -> Optional[Any]:
        """Ottiene dati dalla cache se non scaduti."""
        if cache_key in self._cache:
            expiry_time = self._cache_expiry.get(cache_key, 0)
            if time.time() < expiry_time:
                return self._cache[cache_key]
            else:
                # Rimuovi dati scaduti
                del self._cache[cache_key]
                del self._cache_expiry[cache_key]
        return None

    def _cache_data(self, cache_key: str, data: Any, expiry_minutes: int = 60) -> None:
        """Salva dati in cache."""
        self._cache[cache_key] = data
        self._cache_expiry[cache_key] = time.time() + (expiry_minutes * 60)

    def get_teams(self) -> List[Dict[str, Any]]:
        """Ottieni tutte le squadre NBA (usa NBA API come prioritaria)."""
        cache_key = "teams"
        cached = self._get_cached_data(cache_key, expiry_minutes=1440)  # 24 ore cache
        if cached:
            return cached

        try:
            # Prova NBA API (swar) - prioritaria
            nba_teams = teams.get_teams()
            logger.info(f"✅ Got {len(nba_teams)} teams from NBA API")

            self._cache_data(cache_key, nba_teams, expiry_minutes=1440)
            return nba_teams

        except Exception as e:
            logger.warning(f"NBA API teams failed: {e}")

            # Fallback a BallDontLie API
            try:
                self._rate_limit_wait('balldontlie')
                teams_response = self.bdl_client.nba.teams.list(per_page=30)
                bdl_teams = [
                    {
                        'id': team.id,
                        'full_name': team.full_name,
                        'abbreviation': team.abbreviation,
                        'conference': team.conference,
                        'division': team.division
                    }
                    for team in teams_response.data
                ]
                logger.info(f"✅ Got {len(bdl_teams)} teams from BallDontLie API")

                self._cache_data(cache_key, bdl_teams, expiry_minutes=1440)
                return bdl_teams

            except Exception as e2:
                logger.error(f"Both teams APIs failed: {e2}")
                return []

    def get_players(self) -> List[Dict[str, Any]]:
        """Ottieni tutti i giocatori NBA (usa NBA API)."""
        cache_key = "players"
        cached = self._get_cached_data(cache_key, expiry_minutes=1440)  # 24 ore cache
        if cached:
            return cached

        try:
            all_players = players.get_players()
            logger.info(f"✅ Got {len(all_players)} players from NBA API")

            self._cache_data(cache_key, all_players, expiry_minutes=1440)
            return all_players

        except Exception as e:
            logger.error(f"Failed to get players: {e}")
            return []

    def get_player_stats(self, player_id: int, season: int = 2024) -> Optional[Dict[str, Any]]:
        """Ottieni statistiche complete giocatore (NBA API)."""
        cache_key = f"player_stats_{player_id}_{season}"
        cached = self._get_cached_data(cache_key, expiry_minutes=180)  # 3 ore cache
        if cached:
            return cached

        try:
            # Career stats
            career = playercareerstats.PlayerCareerStats(player_id=player_id)
            career_df = career.get_data_frames()[0]

            # Filtra per stagione
            season_stats = career_df[career_df['SEASON_ID'] == f'2{season-2000:02d}{season-1999:02d}']

            if season_stats.empty:
                logger.warning(f"No stats found for player {player_id} in season {season}")
                return None

            # Converti in dict
            stats_dict = season_stats.iloc[0].to_dict()
            logger.debug(f"Got stats for player {player_id}, season {season}")

            self._cache_data(cache_key, stats_dict, expiry_minutes=180)
            return stats_dict

        except Exception as e:
            logger.error(f"Failed to get player stats for {player_id}: {e}")
            return None

    def get_team_roster(self, team_id: int, season: int = 2024) -> List[Dict[str, Any]]:
        """Ottieni roster squadra (NBA API)."""
        cache_key = f"team_roster_{team_id}_{season}"
        cached = self._get_cached_data(cache_key, expiry_minutes=60)  # 1 ora cache
        if cached:
            return cached

        try:
            roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
            roster_df = roster.get_data_frames()[0]

            roster_list = roster_df.to_dict('records')
            logger.info(f"✅ Got {len(roster_list)} players for team {team_id}")

            self._cache_data(cache_key, roster_list, expiry_minutes=60)
            return roster_list

        except Exception as e:
            logger.error(f"Failed to get team roster for {team_id}: {e}")
            return []

    def get_games(self, start_date: str, end_date: str = None) -> List[Dict[str, Any]]:
        """Ottieni partite programma (BallDontLie API)."""
        if end_date is None:
            end_date = start_date

        cache_key = f"games_{start_date}_{end_date}"
        cached = self._get_cached_data(cache_key, expiry_minutes=30)  # 30 min cache
        if cached:
            return cached

        try:
            self._rate_limit_wait('balldontlie')

            # Parse dates
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')

            all_games = []
            current_date = start_dt

            while current_date <= end_dt:
                date_str = current_date.strftime('%Y-%m-%d')

                games_response = self.bdl_client.nba.games.list(
                    dates=[date_str],
                    per_page=100
                )

                day_games = [
                    {
                        'id': game.id,
                        'date': game.date,
                        'status': game.status,
                        'period': game.period,
                        'time': game.time,
                        'postseason': game.postseason,
                        'home_team': game.home_team.full_name,
                        'home_team_id': game.home_team.id,
                        'visitor_team': game.visitor_team.full_name,
                        'visitor_team_id': game.visitor_team.id,
                        'home_team_score': game.home_team_score,
                        'visitor_team_score': game.visitor_team_score
                    }
                    for game in games_response.data
                ]

                all_games.extend(day_games)
                current_date += timedelta(days=1)

            logger.info(f"✅ Got {len(all_games)} games from {start_date} to {end_date}")

            self._cache_data(cache_key, all_games, expiry_minutes=30)
            return all_games

        except Exception as e:
            logger.error(f"Failed to get games: {e}")
            return []

    def get_odds(self, game_date: str) -> List[Dict[str, Any]]:
        """Ottieni quote scommesse (The Odds API)."""
        cache_key = f"odds_{game_date}"
        cached = self._get_cached_data(cache_key, expiry_minutes=60)  # 1 ora cache
        if cached:
            return cached

        try:
            self._rate_limit_wait('the_odds')

            url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
            params = {
                'apiKey': self.the_odds_api_key,
                'regions': 'eu',
                'markets': 'h2h,spreads,totals',
                'oddsFormat': 'decimal',
                'dateFormat': 'iso',
                'commenceTimeFrom': f"{game_date}T00:00:00Z",
                'commenceTimeUntil': f"{game_date}T23:59:59Z"
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            odds_data = response.json()
            logger.info(f"✅ Got {len(odds_data)} odds for {game_date}")

            self._cache_data(cache_key, odds_data, expiry_minutes=60)
            return odds_data

        except Exception as e:
            logger.error(f"Failed to get odds: {e}")
            return []

    def get_comprehensive_game_data(self, game_date: str) -> Dict[str, Any]:
        """
        Ottieni dati completi per una data partite.

        Returns:
            Dict con:
            - games: partite programma
            - teams: informazioni squadre
            - rosters: roster completi
            - odds: quote scommesse
        """
        logger.info(f"🏀 Getting comprehensive data for {game_date}")

        # Get games
        games = self.get_games(game_date)

        # Get teams ( cached)
        all_teams = {team['id']: team for team in self.get_teams()}

        # Get rosters for teams playing today
        team_ids_today = set()
        for game in games:
            team_ids_today.add(game['home_team_id'])
            team_ids_today.add(game['visitor_team_id'])

        rosters = {}
        for team_id in team_ids_today:
            rosters[team_id] = self.get_team_roster(team_id)

        # Get odds
        odds = self.get_odds(game_date)

        return {
            'date': game_date,
            'games': games,
            'teams': all_teams,
            'rosters': rosters,
            'odds': odds,
            'timestamp': datetime.now().isoformat()
        }

    def get_enhanced_player_stats(self, team_id: int, season: int = 2024) -> pl.DataFrame:
        """
        Ottieni statistiche avanzate per tutti i giocatori di una squadra.

        Returns:
            Polars DataFrame con statistiche complete
        """
        try:
            # Get roster
            roster = self.get_team_roster(team_id, season)

            if not roster:
                return pl.DataFrame()

            # Get stats for each player
            all_stats = []
            for player in roster:
                player_id = player['PLAYER_ID']
                stats = self.get_player_stats(player_id, season)

                if stats:
                    # Add player info
                    stats['PLAYER_NAME'] = player['PLAYER']
                    stats['TEAM_ID'] = team_id
                    all_stats.append(stats)

            if all_stats:
                df = pl.DataFrame(all_stats)
                logger.info(f"✅ Got stats for {len(df)} players from team {team_id}")
                return df
            else:
                return pl.DataFrame()

        except Exception as e:
            logger.error(f"Failed to get enhanced player stats: {e}")
            return pl.DataFrame()


# Funzione di test
def test_multi_source_provider():
    """Testa il provider multi-sorgente."""
    provider = MultiSourceNBADataProvider()

    print("🏀 Test Multi-Source NBA Provider")

    # Test teams
    print("\n📊 Testing teams...")
    teams = provider.get_teams()
    print(f"✅ Teams: {len(teams)}")

    # Test players
    print("\n👥 Testing players...")
    players = provider.get_players()
    print(f"✅ Players: {len(players)}")

    # Test games today
    print("\n📅 Testing games...")
    today = date.today().strftime('%Y-%m-%d')
    games = provider.get_games(today)
    print(f"✅ Games today: {len(games)}")

    # Test comprehensive data
    print("\n📈 Testing comprehensive data...")
    comprehensive = provider.get_comprehensive_game_data(today)
    print(f"✅ Comprehensive data: {len(comprehensive['games'])} games, {len(comprehensive['teams'])} teams")

    # Test player stats
    if players:
        print("\n🎯 Testing player stats...")
        lebron = [p for p in players if p['full_name'] == 'LeBron James'][0]
        stats = provider.get_player_stats(lebron['id'])
        if stats:
            print(f"✅ LeBron stats: PPG {stats.get('PTS', 0):.1f}, APG {stats.get('AST', 0):.1f}")

    print("\n🎉 Multi-Source NBA Provider test completed!")


if __name__ == "__main__":
    test_multi_source_provider()