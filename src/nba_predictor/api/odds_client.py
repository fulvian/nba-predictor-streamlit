"""Modern Odds API client with unified data store integration.

This module provides a modern async client for The Odds API that integrates with
the unified data store for efficient data persistence and retrieval of betting odds.
"""

import asyncio
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import httpx

from ..utils.exceptions import APIError, ValidationError

logger = logging.getLogger(__name__)


class ModernOddsAPIClient:
    """Modern Odds API client with unified data store integration.

    This client provides async access to The Odds API with proper error handling,
    rate limiting, and integration with the unified data store for efficient
    data management of betting odds and bookmaker information.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.the-odds-api.com/v4",
        timeout: float = 30.0,
        rate_limit_delay: float = 1.0,
        cache_ttl: int = 1800,  # 30 minutes for odds data
        headers: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Initialize the Odds API client.

        Args:
            api_key: The Odds API key for authentication
            base_url: Base URL for The Odds API
            timeout: Request timeout in seconds
            rate_limit_delay: Delay between requests for rate limiting
            cache_ttl: Cache time-to-live in seconds (shorter for odds data)
            headers: Optional custom headers to include

        Returns:
            None

        Raises:
            ValidationError: If api_key is invalid

        Example:
            >>> client = ModernOddsAPIClient(api_key="your-api-key")
            >>> odds = await client.fetch_nba_odds(
            ...     regions="us",
            ...     markets="h2h,spreads,totals"
            ... )
            >>> print(f"Fetched odds for {len(odds)} games")
        """
        if not api_key or not api_key.strip():
            raise ValidationError("api_key is required and cannot be empty")

        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.cache_ttl = cache_ttl

        # Cache for API responses
        self._cache: Dict[str, Any] = {}
        self._session: Optional[httpx.AsyncClient] = None

        # HTTP client configuration
        self._headers = {
            'Accept': 'application/json',
            'User-Agent': 'NBA-Predictor/1.0.0',
            'Connection': 'keep-alive'
        }

        # Override headers with custom ones if provided
        if headers:
            self._headers.update(headers)

        logger.info(
            "ModernOddsAPIClient initialized",
            extra={
                "base_url": self.base_url,
                "timeout": timeout,
                "rate_limit_delay": rate_limit_delay,
                "cache_ttl": cache_ttl
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

    async def __aenter__(self) -> "ModernOddsAPIClient":
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
            if v is not None:
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

    async def fetch_nba_odds(
        self,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        odds_format: str = "american",
        date_from: Optional[date] = None,
        date_to: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch NBA betting odds from The Odds API.

        Args:
            regions: Betting regions (us, uk, eu, etc.)
            markets: Betting markets (h2h, spreads, totals, etc.)
            odds_format: Format for odds (american, decimal, etc.)
            date_from: Optional start date for filtering
            date_to: Optional end date for filtering

        Returns:
            List of dictionaries with odds data

        Raises:
            APIError: If API calls fail
            ValidationError: If parameters are invalid
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key(
                "nba_odds",
                regions=regions,
                markets=markets,
                odds_format=odds_format,
                date_from=date_from.isoformat() if date_from else None,
                date_to=date_to.isoformat() if date_to else None
            )
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.info(f"Returning cached NBA odds data")
                return cached_result

            # Construct API URL
            endpoint = "/sports/basketball_nba/odds"
            params = {
                'apiKey': self.api_key,
                'regions': regions,
                'markets': markets,
                'oddsFormat': odds_format,
                'dateFormat': 'iso'
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
            odds_list = self._parse_odds_data(data)

            # Return None if parsing failed
            if not odds_list:
                return None

            # Filter by date range if provided
            if date_from or date_to:
                odds_list = self._filter_by_date_range(
                    odds_list, date_from, date_to
                )

                # Return None if filtering resulted in empty list
                if not odds_list:
                    return None

            # Cache the result
            self._store_in_cache(cache_key, odds_list)

            logger.info(f"Successfully fetched odds for {len(odds_list)} games")
            return odds_list

        except Exception as e:
            logger.error(f"Failed to fetch NBA odds: {e}")
            return None

    async def fetch_nba_scores(
        self,
        days_from: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Fetch NBA scores and game information.

        Args:
            days_from: Number of days from today to fetch games

        Returns:
            List of dictionaries with game scores

        Raises:
            APIError: If API calls fail
            ValidationError: If parameters are invalid
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key("nba_scores", days_from=days_from)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.info(f"Returning cached NBA scores data")
                return cached_result

            # Construct API URL
            endpoint = "/sports/basketball_nba/scores"
            params = {
                'apiKey': self.api_key,
                'daysFrom': str(days_from),
                'dateFormat': 'iso'
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
            scores_list = self._parse_scores_data(data)

            # Cache the result
            self._store_in_cache(cache_key, scores_list)

            logger.info(f"Successfully fetched scores for {len(scores_list)} games")
            return scores_list

        except Exception as e:
            logger.error(f"Failed to fetch NBA scores: {e}")
            return None

    async def fetch_event_odds(
        self,
        event_id: str,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        odds_format: str = "american"
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch odds for a specific event.

        Args:
            event_id: The Odds API event ID
            regions: Betting regions
            markets: Betting markets
            odds_format: Format for odds

        Returns:
            Dictionary with event odds or None if not found

        Raises:
            APIError: If API call fails
            ValidationError: If parameters are invalid
        """
        if not event_id or not event_id.strip():
            raise ValidationError("event_id is required")

        try:
            # Check cache first
            cache_key = self._get_cache_key(
                "event_odds",
                event_id=event_id,
                regions=regions,
                markets=markets,
                odds_format=odds_format
            )
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.info(f"Returning cached event odds for {event_id}")
                return cached_result[0] if cached_result else None

            # Construct API URL
            endpoint = f"/sports/basketball_nba/events/{event_id}/odds"
            params = {
                'apiKey': self.api_key,
                'regions': regions,
                'markets': markets,
                'oddsFormat': odds_format,
                'dateFormat': 'iso'
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
            odds_list = self._parse_odds_data(data)

            # Cache the result
            self._store_in_cache(cache_key, odds_list)

            logger.info(f"Successfully fetched odds for event {event_id}")
            return odds_list[0] if odds_list else None

        except Exception as e:
            logger.error(f"Failed to fetch event odds: {e}")
            return None

    async def fetch_historical_odds(
        self,
        event_id: str,
        snapshot_date: datetime,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        odds_format: str = "american"
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch historical odds for a specific event.

        Args:
            event_id: The Odds API event ID
            snapshot_date: Date/time for historical snapshot
            regions: Betting regions
            markets: Betting markets
            odds_format: Format for odds

        Returns:
            Dictionary with historical odds or None if not found

        Raises:
            APIError: If API call fails
            ValidationError: If parameters are invalid
        """
        if not event_id or not event_id.strip():
            raise ValidationError("event_id is required")

        try:
            # Check cache first
            cache_key = self._get_cache_key(
                "historical_odds",
                event_id=event_id,
                snapshot_date=snapshot_date.isoformat(),
                regions=regions,
                markets=markets,
                odds_format=odds_format
            )
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                logger.info(f"Returning cached historical odds for {event_id}")
                return cached_result[0] if cached_result else None

            # Construct API URL
            endpoint = f"/historical/sports/basketball_nba/events/{event_id}/odds"
            params = {
                'apiKey': self.api_key,
                'date': snapshot_date.isoformat(),
                'regions': regions,
                'markets': markets,
                'oddsFormat': odds_format,
                'dateFormat': 'iso'
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
            odds_list = self._parse_historical_odds_data(data)

            # Cache the result
            self._store_in_cache(cache_key, odds_list)

            logger.info(f"Successfully fetched historical odds for event {event_id}")
            return odds_list[0] if odds_list else None

        except Exception as e:
            logger.error(f"Failed to fetch historical odds: {e}")
            return None

    def _parse_odds_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse odds data from The Odds API response."""
        try:
            if not data or not isinstance(data, list):
                return []

            parsed_odds = []

            for game in data:
                try:
                    # Basic game info
                    game_record = {
                        'id': game.get('id', ''),
                        'sport_key': game.get('sport_key', ''),
                        'sport_title': game.get('sport_title', ''),
                        'commence_time': game.get('commence_time', ''),
                        'home_team': game.get('home_team', ''),
                        'away_team': game.get('away_team', ''),
                        'bookmakers': self._parse_bookmakers(game.get('bookmakers', [])),
                        'bookmakers_count': len(game.get('bookmakers', []))
                    }

                    parsed_odds.append(game_record)

                except Exception as e:
                    logger.warning(f"Failed to parse odds record: {e}")
                    continue

            return parsed_odds

        except Exception as e:
            logger.error(f"Failed to parse odds data: {e}")
            return []

    def _parse_scores_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse scores data from The Odds API response."""
        try:
            if not data or not isinstance(data, list):
                return []

            parsed_scores = []

            for game in data:
                try:
                    # Basic game info
                    game_record = {
                        'id': game.get('id', ''),
                        'sport_key': game.get('sport_key', ''),
                        'sport_title': game.get('sport_title', ''),
                        'commence_time': game.get('commence_time', ''),
                        'home_team': game.get('home_team', ''),
                        'away_team': game.get('away_team', ''),
                        'completed': game.get('completed', False),
                        'scores': game.get('scores', []),
                        'last_update': game.get('last_update', '')
                    }

                    parsed_scores.append(game_record)

                except Exception as e:
                    logger.warning(f"Failed to parse scores record: {e}")
                    continue

            return parsed_scores

        except Exception as e:
            logger.error(f"Failed to parse scores data: {e}")
            return []

    def _parse_historical_odds_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse historical odds data from The Odds API response."""
        try:
            if not data or 'data' not in data:
                return []

            # Extract timestamp information
            historical_data = {
                'timestamp': data.get('timestamp', ''),
                'previous_timestamp': data.get('previous_timestamp', ''),
                'next_timestamp': data.get('next_timestamp', ''),
                'games': []
            }

            # Parse games data
            for game in data.get('data', []):
                try:
                    game_record = {
                        'id': game.get('id', ''),
                        'sport_key': game.get('sport_key', ''),
                        'sport_title': game.get('sport_title', ''),
                        'commence_time': game.get('commence_time', ''),
                        'home_team': game.get('home_team', ''),
                        'away_team': game.get('away_team', ''),
                        'bookmakers': self._parse_bookmakers(game.get('bookmakers', []))
                    }

                    historical_data['games'].append(game_record)

                except Exception as e:
                    logger.warning(f"Failed to parse historical odds record: {e}")
                    continue

            return [historical_data]

        except Exception as e:
            logger.error(f"Failed to parse historical odds data: {e}")
            return []

    def _parse_bookmakers(self, bookmakers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse bookmakers data from odds response."""
        try:
            if not bookmakers or not isinstance(bookmakers, list):
                return []

            parsed_bookmakers = []

            for bookmaker in bookmakers:
                try:
                    bookmaker_record = {
                        'key': bookmaker.get('key', ''),
                        'title': bookmaker.get('title', ''),
                        'last_update': bookmaker.get('last_update', ''),
                        'markets': self._parse_markets(bookmaker.get('markets', []))
                    }

                    parsed_bookmakers.append(bookmaker_record)

                except Exception as e:
                    logger.warning(f"Failed to parse bookmaker record: {e}")
                    continue

            return parsed_bookmakers

        except Exception as e:
            logger.error(f"Failed to parse bookmakers data: {e}")
            return []

    def _parse_markets(self, markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse betting markets data."""
        try:
            if not markets or not isinstance(markets, list):
                return []

            parsed_markets = []

            for market in markets:
                try:
                    market_record = {
                        'key': market.get('key', ''),
                        'outcomes': self._parse_outcomes(market.get('outcomes', []))
                    }

                    # Add spread information if available
                    if market.get('outcomes') and len(market['outcomes']) > 0:
                        first_outcome = market['outcomes'][0]
                        if 'point' in first_outcome:
                            market_record['point'] = first_outcome.get('point')

                    parsed_markets.append(market_record)

                except Exception as e:
                    logger.warning(f"Failed to parse market record: {e}")
                    continue

            return parsed_markets

        except Exception as e:
            logger.error(f"Failed to parse markets data: {e}")
            return []

    def _parse_outcomes(self, outcomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse outcomes data from betting markets."""
        try:
            if not outcomes or not isinstance(outcomes, list):
                return []

            parsed_outcomes = []

            for outcome in outcomes:
                try:
                    outcome_record = {
                        'name': outcome.get('name', ''),
                        'price': outcome.get('price'),
                        'point': outcome.get('point')
                    }

                    parsed_outcomes.append(outcome_record)

                except Exception as e:
                    logger.warning(f"Failed to parse outcome record: {e}")
                    continue

            return parsed_outcomes

        except Exception as e:
            logger.error(f"Failed to parse outcomes data: {e}")
            return []

    def _filter_by_date_range(
        self,
        odds_list: List[Dict[str, Any]],
        date_from: Optional[date],
        date_to: Optional[date]
    ) -> List[Dict[str, Any]]:
        """Filter odds list by date range."""
        try:
            if not date_from and not date_to:
                return odds_list

            filtered_odds = []

            for odds in odds_list:
                try:
                    commence_time_str = odds.get('commence_time', '')
                    if not commence_time_str:
                        continue

                    commence_time = datetime.fromisoformat(
                        commence_time_str.replace('Z', '+00:00')
                    ).date()

                    # Check date range
                    if date_from and commence_time < date_from:
                        continue
                    if date_to and commence_time > date_to:
                        continue

                    filtered_odds.append(odds)

                except Exception as e:
                    logger.warning(f"Failed to filter odds by date: {e}")
                    continue

            return filtered_odds

        except Exception as e:
            logger.error(f"Failed to filter odds by date range: {e}")
            return odds_list

    async def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get API usage statistics.

        Returns:
            Dictionary with usage statistics

        Raises:
            APIError: If API call fails
        """
        try:
            # Make a simple API call to get usage headers
            session = await self.get_session()
            await asyncio.sleep(self.rate_limit_delay)

            response = await session.get(
                f"{self.base_url}/sports",
                params={'apiKey': self.api_key}
            )

            # Extract usage information from headers
            usage_stats = {
                'requests_remaining': response.headers.get('x-requests-remaining', 'unknown'),
                'requests_used': response.headers.get('x-requests-used', 'unknown'),
                'requests_last': response.headers.get('x-requests-last', 'unknown'),
                'status': 'success',
                'cache_size': len(self._cache),
                'base_url': self.base_url
            }

            return usage_stats

        except Exception as e:
            logger.error(f"Failed to get usage stats: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'cache_size': len(self._cache),
                'base_url': self.base_url
            }