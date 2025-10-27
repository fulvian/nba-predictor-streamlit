"""Modern NBA API client with unified data store integration.

This module provides a modern async NBA API client that integrates with
the unified data store for efficient data persistence and retrieval.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
import polars as pl

from ..core.data_store import UnifiedDataStore
from ..utils.exceptions import APIError, ValidationError

logger = logging.getLogger(__name__)


class ModernNBAAPIClient:
    """Modern NBA API client with unified data store integration.

    This client provides async NBA API access with proper error handling,
    rate limiting, and integration with the unified data store for efficient
    data management.
    """

    def __init__(
        self,
        base_url: str = "https://stats.nba.com",
        timeout: float = 30.0,
        rate_limit_delay: float = 1.0,
        cache_ttl: int = 3600,
        headers: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Initialize the NBA API client.

        Args:
            base_url: Base URL for NBA API
            timeout: Request timeout in seconds
            rate_limit_delay: Delay between requests for rate limiting
            cache_ttl: Cache time-to-live in seconds
            headers: Optional custom headers to include

        Returns:
            None

        Raises:
            ValidationError: If base_url is invalid

        Example:
            >>> client = ModernNBAAPIClient()
            >>> games = await client.fetch_all_games(
            ...     date(2024,1,1), date(2024,1,31)
            ... )
            >>> print(f"Fetched {len(games)} games")
        """
        if not base_url or not base_url.strip():
            raise ValidationError("base_url is required")

        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.cache_ttl = cache_ttl

        # Cache for API responses
        self._cache: Dict[str, Any] = {}
        self._session: Optional[httpx.AsyncClient] = None

        # HTTP client configuration
        self._headers = {
            'Host': 'stats.nba.com',
            'User-Agent': 'NBA-Predictor/1.0.0',
            'Accept': 'application/json, text/plain, */*',
            'x-nba-stats-origin': 'stats',
            'Connection': 'keep-alive'
        }

        # Override headers with custom ones if provided
        if headers:
            self._headers.update(headers)

        logger.info(
            "ModernNBAAPIClient initialized",
            extra={
                "base_url": self.base_url,
                "timeout": timeout,
                "rate_limit_delay": rate_limit_delay
            }
        )

    @property
    def cache(self) -> Dict[str, Any]:
        """Get cache dictionary (for backward compatibility with tests)."""
        return self._cache

    @property
    def headers(self) -> Dict[str, str]:
        """Get HTTP headers."""
        return self._headers

    async def get_session(self) -> httpx.AsyncClient:
        """Get or create async HTTP session."""
        if self._session is None:
            self._session = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers=self._headers
            )
        return self._session

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._session is not None:
            await self._session.aclose()
            self._session = None

    async def __aenter__(self) -> "ModernNBAAPIClient":
        """Async context manager entry."""
        # Ensure session is created when entering context
        _ = await self.get_session()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.cleanup()

    def _get_cache_key(self, endpoint: str, **kwargs: Any) -> str:
        """Generate cache key for request."""
        key_parts = [endpoint]
        for k, v in sorted(kwargs.items()):
            if v is not None and not (k == "include_stats" and v is True):
                # Skip include_stats=True to keep cache keys simple for default case
                key_parts.append(f"{k}={v}")
        return ":".join(key_parts)

    def _store_in_cache(self, key: str, data: Any) -> None:
        """Store data in cache with timestamp."""
        self._cache[key] = {
            "data": data,
            "timestamp": datetime.now().timestamp()
        }

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache if not expired."""
        if key not in self._cache:
            return None

        cache_entry = self._cache[key]
        if self._is_cache_expired(cache_entry["timestamp"]):
            del self._cache[key]
            return None

        return cache_entry["data"]

    def _is_cache_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired."""
        return (datetime.now().timestamp() - timestamp) > self.cache_ttl

    async def fetch_all_games(
        self,
        start_date: date,
        end_date: date,
        include_stats: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch all NBA games in date range with optional statistics.

        Args:
            start_date: Start date for games
            end_date: End date for games
            include_stats: Include detailed statistics

        Returns:
            List of dictionaries with games data

        Raises:
            APIError: If API calls fail
            ValidationError: If response format is invalid

        Example:
            >>> client = ModernNBAAPIClient()
            >>> games = await client.fetch_all_games(
            ...     date(2024,1,1), date(2024,1,31)
            ... )
            >>> print(f"Fetched {len(games)} games")
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key("games", start_date=start_date.isoformat(), end_date=end_date.isoformat(), include_stats=include_stats)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.info(f"Returning cached games data for {start_date} to {end_date}")
                return cached_result

            # Construct API URL
            endpoint = "/games"
            params = {
                'GameDate': start_date.strftime('%Y%m%d'),
                'LeagueID': '00'  # NBA
            }

            # Add stats parameter if requested
            if include_stats:
                params.update({
                    'DayOffset': '0',
                    'GameSegment': '4'  # All quarters
                })

            # Make API request
            session = await self.get_session()
            await asyncio.sleep(self.rate_limit_delay)  # Rate limiting

            response = await session.get(
                f"{self.base_url}{endpoint}",
                params=params
            )
            response.raise_for_status()

            data = response.json()

            # Parse response
            games_list = self._parse_games_data(data, include_stats)

            # Return None if parsing failed
            if not games_list:
                return None

            # Cache the result
            self._store_in_cache(cache_key, games_list)

            logger.info(
                f"Successfully fetched {len(games_list)} games from {start_date} to {end_date}"
            )

            return games_list

        except Exception as e:
            logger.error(f"Failed to fetch games data: {e}")
            return None

    async def fetch_players(
        self,
        season_year: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch all NBA players for a season.

        Args:
            season_year: Optional season year (defaults to current season)

        Returns:
            List of dictionaries with player data
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key("players", season_year=season_year)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.info(f"Returning cached players data for season {season_year}")
                return cached_result

            # Construct API URL
            endpoint = "/commonallplayers"
            params = {
                'LeagueID': '00',
                'Season': season_year if season_year else '',
                'IsOnlyCurrentSeason': '1'
            }

            # Make API request
            session = await self.get_session()
            await asyncio.sleep(self.rate_limit_delay)  # Rate limiting

            response = await session.get(
                f"{self.base_url}{endpoint}",
                params=params
            )
            response.raise_for_status()

            data = response.json()

            # Parse response
            players_list = self._parse_players_data(data)

            # Cache the result
            self._store_in_cache(cache_key, players_list)

            logger.info(f"Successfully fetched {len(players_list)} players")
            return players_list

        except Exception as e:
            logger.error(f"Failed to fetch players data: {e}")
            return None

    async def fetch_player_stats(
        self,
        player_id: int,
        season_year: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch player statistics for a specific player.

        Args:
            player_id: NBA player ID
            season_year: Optional season year (defaults to current season)

        Returns:
            Dict with player statistics or None if not found

        Raises:
            APIError: If API call fails
            ValidationError: If player_id is invalid
        """
        if not player_id or player_id <= 0:
            raise ValidationError("player_id must be positive integer")

        try:
            # Determine season year
            if season_year is None:
                current_date = date.today()
                # NBA season typically starts in October
                if current_date.month >= 10:
                    season_year = current_date.year + 1
                else:
                    season_year = current_date.year

            # Check cache first
            cache_key = self._get_cache_key("player_stats", player_id=player_id, season_year=season_year)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.info(f"Returning cached player stats for player {player_id}")
                return cached_result[0] if cached_result else None

            # Construct API URL
            endpoint = f"/stats/player/{player_id}"
            params = {
                'LeagueID': '00',
                'Season': season_year,
                'SeasonType': 'Regular Season',
                'PerMode': 'Totals'
            }

            # Make API request
            session = await self.get_session()
            await asyncio.sleep(self.rate_limit_delay)  # Rate limiting

            response = await session.get(
                f"{self.base_url}{endpoint}",
                params=params
            )
            response.raise_for_status()

            data = response.json()

            # Parse response
            stats_list = self._parse_stats_data(data, "player")

            # Cache the result
            self._store_in_cache(cache_key, stats_list)

            logger.info(f"Successfully fetched stats for player {player_id}")
            return stats_list[0] if stats_list else None

        except Exception as e:
            logger.error(f"Failed to fetch player stats: {e}")
            return None

    async def fetch_team_stats(
        self,
        team_id: int,
        season_year: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch team statistics for a specific team.

        Args:
            team_id: NBA team ID
            season_year: Optional season year (defaults to current season)

        Returns:
            Dict with team statistics or None if not found

        Raises:
            APIError: If API call fails
            ValidationError: If team_id is invalid
        """
        if not team_id or team_id <= 0:
            raise ValidationError("team_id must be positive integer")

        try:
            # Determine season year
            if season_year is None:
                current_date = date.today()
                if current_date.month >= 10:
                    season_year = current_date.year + 1
                else:
                    season_year = current_date.year

            # Check cache first
            cache_key = self._get_cache_key("team_stats", team_id=team_id, season_year=season_year)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.info(f"Returning cached team stats for team {team_id}")
                return cached_result[0] if cached_result else None

            # Construct API URL
            endpoint = f"/stats/team/{team_id}"
            params = {
                'LeagueID': '00',
                'Season': season_year,
                'SeasonType': 'Regular Season',
                'PerMode': 'Totals'
            }

            # Make API request
            session = await self.get_session()
            await asyncio.sleep(self.rate_limit_delay)  # Rate limiting

            response = await session.get(
                f"{self.base_url}{endpoint}",
                params=params
            )
            response.raise_for_status()

            data = response.json()

            # Parse response
            stats_list = self._parse_stats_data(data, "team")

            # Cache the result
            self._store_in_cache(cache_key, stats_list)

            logger.info(f"Successfully fetched stats for team {team_id}")
            return stats_list[0] if stats_list else None

        except Exception as e:
            logger.error(f"Failed to fetch team stats: {e}")
            return None

    async def fetch_standings(
        self,
        season_year: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch NBA standings for a season.

        Args:
            season_year: Optional season year (defaults to current season)

        Returns:
            List of dictionaries with standings data
        """
        try:
            # Determine season year
            if season_year is None:
                current_date = date.today()
                if current_date.month >= 10:
                    season_year = str(current_date.year + 1)
                else:
                    season_year = str(current_date.year)

            # Check cache first
            cache_key = self._get_cache_key("standings", season_year=season_year)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.info(f"Returning cached standings data for season {season_year}")
                return cached_result

            # Construct API URL
            endpoint = "/standings"
            params = {
                'LeagueID': '00',
                'Season': season_year,
                'SeasonType': 'Regular Season'
            }

            # Make API request
            session = await self.get_session()
            await asyncio.sleep(self.rate_limit_delay)  # Rate limiting

            response = await session.get(
                f"{self.base_url}{endpoint}",
                params=params
            )
            response.raise_for_status()

            data = response.json()

            # Parse response
            standings_list = self._parse_standings_data(data)

            # Cache the result
            self._store_in_cache(cache_key, standings_list)

            logger.info(f"Successfully fetched standings for season {season_year}")
            return standings_list

        except Exception as e:
            logger.error(f"Failed to fetch standings: {e}")
            return None

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to NBA API with proper error handling.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            headers: Additional headers

        Returns:
            Parsed JSON response

        Raises:
            APIError: If request fails
        """
        try:
            # Apply rate limiting
            await asyncio.sleep(self.rate_limit_delay)

            # Prepare request
            url = f"{self.base_url}{endpoint}"
            request_headers = self._headers.copy()
            if headers:
                request_headers.update(headers)

            # Make async request
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    headers=request_headers
                )

                # Handle response
                response.raise_for_status()
                return response.json()

        except httpx.TimeoutError as e:
            logger.error(f"Request timeout: {e}")
            raise APIError(f"Request timeout after {self.timeout}s: {e}") from e
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e}")
            raise APIError(f"HTTP error {e.response.status_code}: {e}") from e
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise APIError(f"Request error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during API request: {e}")
            raise APIError(f"API request failed: {e}") from e

    def _parse_games_response(self, data: Dict[str, Any], include_stats: bool) -> pl.DataFrame:
        """Parse games API response into Polars DataFrame."""
        try:
            games = data.get('scoreboard', {}).get('games', [])

            if not games:
                logger.warning("No games found in API response")
                return pl.DataFrame()

            # Extract game data
            game_records = []
            for game in games:
                try:
                    # Basic game info
                    game_record = {
                        'game_id': game.get('gameId', ''),
                        'game_date': self._parse_nba_date(game.get('gameDate', '')),
                        'home_team_id': game.get('homeTeam', {}).get('teamId', 0),
                        'home_team': game.get('homeTeam', {}).get('teamName', ''),
                        'away_team_id': game.get('awayTeam', {}).get('teamId', 0),
                        'away_team': game.get('awayTeam', {}).get('teamName', ''),
                        'season': self._extract_season_from_date(game.get('gameDate', '')),
                        'game_status': game.get('gameStatus', ''),
                        'period': game.get('period', 0),
                        'time_remaining': game.get('timeRemaining', ''),
                        'arena': game.get('arena', '')
                    }

                    # Scores (if available)
                    if 'homeTeamScore' in game and 'awayTeamScore' in game:
                        game_record['home_score'] = game['homeTeamScore']
                        game_record['away_score'] = game['awayTeamScore']
                    elif 'homeTeam' in game and 'awayTeam' in game:
                        home_team_stats = game.get('homeTeam', {})
                        away_team_stats = game.get('awayTeam', {})
                        game_record['home_score'] = home_team_stats.get('score', 0)
                        game_record['away_score'] = away_team_stats.get('score', 0)

                    # Game time info
                    if 'gameTime' in game:
                        game_record['game_time'] = game['gameTime']
                    if 'startTimeUTC' in game:
                        game_record['start_time_utc'] = game['startTimeUTC']

                    # Stats (if requested)
                    if include_stats and 'gameLeaders' in game:
                        leaders = game.get('gameLeaders', [])
                        if leaders:
                            # Get leading player info
                            leader = leaders[0] if leaders else {}
                            game_record['leading_player_id'] = leader.get('playerId', 0)
                            game_record['leading_player_name'] = leader.get('playerName', '')
                            game_record['leading_points'] = leader.get('points', 0)
                            game_record['leading_team'] = leader.get('teamTricode', '')

                    game_records.append(game_record)

                except Exception as e:
                    logger.warning(f"Error parsing game record: {e}")
                    continue

            return pl.DataFrame(game_records)

        except Exception as e:
            logger.error(f"Failed to parse games response: {e}")
            raise ValidationError(f"Invalid games response format: {e}") from e

    def _parse_player_stats_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse player stats API response."""
        try:
            player = data.get('player', {})
            if not player:
                return {}

            # Basic player info
            stats = {
                'player_id': player.get('personId', 0),
                'first_name': player.get('firstName', ''),
                'last_name': player.get('lastName', ''),
                'position': player.get('pos', ''),
                'height_feet': player.get('heightFeet', 0.0),
                'height_inches': player.get('heightInches', 0),
                'weight_pounds': player.get('weightPounds', 0),
                'jersey_number': player.get('jersey', ''),
                'team_id': player.get('teamId', 0),
                'team_name': player.get('teamName', ''),
                'season': player.get('seasonYear', 0),
                'draft_year': player.get('draftYear', 0)
            }

            # Career stats
            career_stats = player.get('careerTotals', {})
            if career_stats:
                stats.update({
                    'career_points': career_stats.get('points', 0),
                    'career_assists': career_stats.get('assists', 0),
                    'career_rebounds': career_stats.get('rebounds', 0),
                    'career_steals': career_stats.get('steals', 0),
                    'career_blocks': career_stats.get('blocks', 0),
                    'career_turnovers': career_stats.get('turnovers', 0),
                    'career_games_played': career_stats.get('gamesPlayed', 0),
                    'career_minutes': career_stats.get('minutes', 0)
                })

            # Season stats
            season_stats = player.get('careerStats', [])
            if season_stats and len(season_stats) > 0:
                latest_season = season_stats[-1]  # Most recent season
                stats.update({
                    'season_points': latest_season.get('points', 0),
                    'season_assists': latest_season.get('assists', 0),
                    'season_rebounds': latest_season.get('rebounds', 0),
                    'season_steals': latest_season.get('steals', 0),
                    'season_blocks': latest_season.get('blocks', 0),
                    'season_turnovers': latest_season.get('turnovers', 0),
                    'season_games_played': latest_season.get('gamesPlayed', 0),
                    'season_minutes': latest_season.get('minutes', 0)
                })

            return stats

        except Exception as e:
            logger.error(f"Failed to parse player stats response: {e}")
            return {}

    def _parse_team_stats_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse team stats API response."""
        try:
            team = data.get('team', {})
            if not team:
                return {}

            stats = {
                'team_id': team.get('teamId', 0),
                'team_name': team.get('teamName', ''),
                'city': team.get('city', ''),
                'arena': team.get('arena', ''),
                'conference': team.get('conference', ''),
                'division': team.get('division', ''),
                'season': team.get('seasonYear', 0),
                'wins': team.get('wins', 0),
                'losses': team.get('losses', 0),
                'win_percentage': round(team.get('pct', 0), 1),
                'conference_rank': team.get('confRank', 0),
                'division_rank': team.get('divRank', 0)
            }

            # Additional stats
            team_stats = team.get('stats', {})
            if team_stats:
                stats.update({
                    'points_per_game': team_stats.get('ptsPerGame', 0),
                    'opponent_points_per_game': team_stats.get('oppPtsPerGame', 0),
                    'fast_break_points': team_stats.get('fastBreakPoints', 0),
                    'second_chance_points': team_stats.get('secondChancePoints', 0),
                    'points_off_turnovers': team_stats.get('ptsOffTurnovers', 0),
                    'turnovers': team_stats.get('tov', 0),
                    'personal_fouls': team_stats.get('pfouls', 0),
                    'team_fouls': team_stats.get('tfouls', 0)
                })

            return stats

        except Exception as e:
            logger.error(f"Failed to parse team stats response: {e}")
            return {}

    def _parse_standings_response(self, data: Dict[str, Any]) -> Optional[pl.DataFrame]:
        """Parse standings API response into Polars DataFrame."""
        try:
            league = data.get('league', {})
            standard = league.get('standard', {})
            teams = standard.get('teams', [])

            if not teams:
                logger.warning("No teams found in standings response")
                return None

            standings_records = []
            for team in teams:
                try:
                    record = {
                        'team_id': team.get('teamId', 0),
                        'team_name': team.get('teamName', ''),
                        'city': team.get('city', ''),
                        'conference': team.get('conference', ''),
                        'division': team.get('division', ''),
                        'wins': team.get('wins', 0),
                        'losses': team.get('losses', 0),
                        'percentage': round(team.get('pct', 0), 1),
                        'conference_rank': team.get('confRank', 0),
                        'division_rank': team.get('divRank', 0),
                        'games_back': team.get('gamesBack', 0),
                        'home_record': team.get('homeRecord', ''),
                        'away_record': team.get('awayRecord', ''),
                        'last_ten': team.get('lastTen', ''),
                        'streak': team.get('streak', 0),
                        'points': team.get('pts', 0)
                    }
                    standings_records.append(record)

                except Exception as e:
                    logger.warning(f"Error parsing team record: {e}")
                    continue

            return pl.DataFrame(standings_records)

        except Exception as e:
            logger.error(f"Failed to parse standings response: {e}")
            return None

    def _parse_nba_date(self, date_str: str) -> Optional[date]:
        """Parse NBA date string to date object."""
        try:
            # NBA API dates are typically in format "MM/DD/YYYY"
            if not date_str:
                return None

            # Try different date formats
            formats = [
                "%m/%d/%Y",
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%m-%d-%Y"
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt).date()
                except ValueError:
                    continue

            logger.warning(f"Unable to parse date: {date_str}")
            return None

        except Exception as e:
            logger.error(f"Error parsing date: {e}")
            return None

    def _extract_season_from_date(self, date_str: str) -> int:
        """Extract season year from date string."""
        try:
            date_obj = self._parse_nba_date(date_str)
            if not date_obj:
                return date.today().year

            # NBA season typically starts in October
            if date_obj.month >= 10:
                return date_obj.year + 1
            else:
                return date_obj.year

        except Exception:
            return date.today().year

    def clear_cache(self) -> None:
        """Clear all cached API responses."""
        self._cache.clear()
        logger.info("NBA API client cache cleared")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on NBA API connectivity."""
        try:
            # Make a simple request to test connectivity
            await self._make_request(
                method="GET",
                endpoint="/league/00",
                params={}
            )
            return {
                'status': 'healthy',
                'api_accessible': True,
                'cache_size': len(self._cache),
                'base_url': self.base_url
            }

        except Exception as e:
            logger.error(f"NBA API health check failed: {e}")
            return {
                'status': 'unhealthy',
                'api_accessible': False,
                'error': str(e)
            }

    def _parse_games_data(self, data: Dict[str, Any], include_stats: bool = True) -> List[Dict[str, Any]]:
        """Parse games data from NBA API response."""
        try:
            if not data or 'league' not in data or 'standard' not in data['league']:
                return []

            games = data['league']['standard']
            parsed_games = []

            for game in games:
                try:
                    # Basic game info
                    game_record = {
                        'game_id': game.get('gameId', ''),
                        'season_id': game.get('seasonId', ''),
                        'home_team_id': int(game.get('hTeam', {}).get('teamId', 0)),
                        'home_score': int(game.get('hTeam', {}).get('score', 0)),
                        'away_team_id': int(game.get('vTeam', {}).get('teamId', 0)),
                        'away_score': int(game.get('vTeam', {}).get('score', 0)),
                    }

                    # Add stats if requested
                    if include_stats and 'stats' in game:
                        stats = game['stats']
                        game_record.update({
                            'attendance': stats.get('attendance', 0),
                            'duration': stats.get('duration', ''),
                        })

                    parsed_games.append(game_record)

                except Exception as e:
                    logger.warning(f"Failed to parse game record: {e}")
                    continue

            return parsed_games

        except Exception as e:
            logger.error(f"Failed to parse games data: {e}")
            return []

    def _parse_players_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse players data from NBA API response."""
        try:
            if not data or 'league' not in data or 'standard' not in data['league']:
                return []

            players = data['league']['standard']
            parsed_players = []

            for player in players:
                try:
                    player_record = {
                        'player_id': player.get('personId', 0),
                        'first_name': player.get('firstName', ''),
                        'last_name': player.get('lastName', ''),
                        'position': player.get('pos') if player.get('pos') else None,
                        'height_feet': player.get('heightFeet', 0.0),
                        'height_inches': player.get('heightInches', 0),
                        'weight_pounds': player.get('weightPounds', 0),
                        'jersey_number': player.get('jersey', ''),
                        'team_id': int(player.get('teamId', 0)) if player.get('teamId') else None,
                        'team_name': player.get('teamName', ''),
                    }

                    parsed_players.append(player_record)

                except Exception as e:
                    logger.warning(f"Failed to parse player record: {e}")
                    continue

            return parsed_players

        except Exception as e:
            logger.error(f"Failed to parse players data: {e}")
            return []

    def _parse_stats_data(self, data: Dict[str, Any], stats_type: str) -> List[Dict[str, Any]]:
        """Parse stats data (player or team) from NBA API response."""
        try:
            if not data or 'league' not in data or 'standard' not in data['league']:
                return []

            stats_list = data['league']['standard']
            parsed_stats = []

            for stats in stats_list:
                try:
                    if 'stat' not in stats:
                        continue

                    stat_data = stats['stat']

                    if stats_type == "player":
                        stats_record = {
                            'player_id': int(stats.get('playerId', 0)),
                            'team_id': int(stats.get('teamId', 0)),
                            'season_id': stats.get('seasonId', ''),
                            'points_per_game': stat_data.get('pointsPerGame', 0.0),
                            'rebounds_per_game': stat_data.get('reboundsPerGame', 0.0),
                            'assists_per_game': stat_data.get('assistsPerGame', 0.0),
                            'steals_per_game': stat_data.get('stealsPerGame', 0.0),
                            'blocks_per_game': stat_data.get('blocksPerGame', 0.0),
                            'field_goal_percentage': stat_data.get('fieldGoalPercentage', 0.0),
                            'three_point_field_goal_percentage': stat_data.get('threePointFieldGoalPercentage', 0.0),
                            'free_throw_percentage': stat_data.get('freeThrowPercentage', 0.0),
                        }
                    else:  # team stats
                        stats_record = {
                            'team_id': int(stats.get('teamId', 0)),
                            'season_id': stats.get('seasonId', ''),
                            'points_per_game': stat_data.get('pointsPerGame', 0.0),
                            'rebounds_per_game': stat_data.get('reboundsPerGame', 0.0),
                            'assists_per_game': stat_data.get('assistsPerGame', 0.0),
                            'steals_per_game': stat_data.get('stealsPerGame', 0.0),
                            'blocks_per_game': stat_data.get('blocksPerGame', 0.0),
                            'field_goal_percentage': stat_data.get('fieldGoalPercentage', 0.0),
                            'three_point_field_goal_percentage': stat_data.get('threePointFieldGoalPercentage', 0.0),
                            'free_throw_percentage': stat_data.get('freeThrowPercentage', 0.0),
                        }

                    parsed_stats.append(stats_record)

                except Exception as e:
                    logger.warning(f"Failed to parse stats record: {e}")
                    continue

            return parsed_stats

        except Exception as e:
            logger.error(f"Failed to parse stats data: {e}")
            return []

    def _parse_standings_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse standings data from NBA API response."""
        try:
            if not data or 'league' not in data or 'standard' not in data['league']:
                return []

            conference_data = data['league']['standard'].get('conference', {})
            parsed_standings = []

            # Process Eastern Conference
            if 'east' in conference_data:
                for team in conference_data['east']:
                    try:
                        team_record = {
                            'team_id': int(team.get('teamId', 0)),
                            'team_name': team.get('teamName', ''),
                            'conference': 'east',
                            'wins': team.get('wins', 0),
                            'losses': team.get('losses', 0),
                            'win_percentage': team.get('winPercentage', 0.0),
                            'loss_percentage': team.get('lossPercentage', 0.0),
                        }
                        parsed_standings.append(team_record)
                    except Exception as e:
                        logger.warning(f"Failed to parse East team record: {e}")
                        continue

            # Process Western Conference
            if 'west' in conference_data:
                for team in conference_data['west']:
                    try:
                        team_record = {
                            'team_id': int(team.get('teamId', 0)),
                            'team_name': team.get('teamName', ''),
                            'conference': 'west',
                            'wins': team.get('wins', 0),
                            'losses': team.get('losses', 0),
                            'win_percentage': team.get('winPercentage', 0.0),
                            'loss_percentage': team.get('lossPercentage', 0.0),
                        }
                        parsed_standings.append(team_record)
                    except Exception as e:
                        logger.warning(f"Failed to parse West team record: {e}")
                        continue

            return parsed_standings

        except Exception as e:
            logger.error(f"Failed to parse standings data: {e}")
            return []