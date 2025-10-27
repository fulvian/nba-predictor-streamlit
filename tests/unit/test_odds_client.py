"""Unit tests for ModernOddsAPIClient.

This module provides comprehensive test coverage for the modern async Odds API client,
including HTTP client mocking, error handling, caching, and data parsing validation.
"""

import json
from datetime import date, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
import httpx
from httpx import Response

from nba_predictor.api.odds_client import ModernOddsAPIClient
from nba_predictor.utils.exceptions import ValidationError


class TestModernOddsAPIClient:
    """Test cases for ModernOddsAPIClient."""

    @pytest.fixture
    def api_key(self):
        """Test API key."""
        return "test-api-key-12345"

    @pytest.fixture
    def client(self, api_key):
        """Create test client instance."""
        return ModernOddsAPIClient(api_key=api_key)

    @pytest.fixture
    def mock_httpx_client(self):
        """Mock httpx.AsyncClient for testing."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def sample_odds_response(self):
        """Sample Odds API odds response."""
        return [
            {
                "id": "game123",
                "sport_key": "basketball_nba",
                "sport_title": "NBA",
                "commence_time": "2024-02-01T19:00:00Z",
                "home_team": "Los Angeles Lakers",
                "away_team": "Boston Celtics",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "title": "DraftKings",
                        "last_update": "2024-02-01T18:00:00Z",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {
                                        "name": "Los Angeles Lakers",
                                        "price": -110
                                    },
                                    {
                                        "name": "Boston Celtics",
                                        "price": -110
                                    }
                                ]
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {
                                        "name": "Los Angeles Lakers",
                                        "price": -110,
                                        "point": -2.5
                                    },
                                    {
                                        "name": "Boston Celtics",
                                        "price": -110,
                                        "point": 2.5
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ]

    @pytest.fixture
    def sample_scores_response(self):
        """Sample Odds API scores response."""
        return [
            {
                "id": "game123",
                "sport_key": "basketball_nba",
                "sport_title": "NBA",
                "commence_time": "2024-02-01T19:00:00Z",
                "home_team": "Los Angeles Lakers",
                "away_team": "Boston Celtics",
                "completed": True,
                "scores": [
                    {
                        "name": "Los Angeles Lakers",
                        "score": 110
                    },
                    {
                        "name": "Boston Celtics",
                        "score": 105
                    }
                ],
                "last_update": "2024-02-01T21:30:00Z"
            }
        ]

    @pytest.fixture
    def sample_historical_response(self):
        """Sample Odds API historical response."""
        return {
            "timestamp": "2024-02-01T18:00:00Z",
            "previous_timestamp": "2024-02-01T17:55:00Z",
            "next_timestamp": "2024-02-01T18:05:00Z",
            "data": [
                {
                    "id": "game123",
                    "sport_key": "basketball_nba",
                    "sport_title": "NBA",
                    "commence_time": "2024-02-01T19:00:00Z",
                    "home_team": "Los Angeles Lakers",
                    "away_team": "Boston Celtics",
                    "bookmakers": [
                        {
                            "key": "draftkings",
                            "title": "DraftKings",
                            "last_update": "2024-02-01T18:00:00Z",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {
                                            "name": "Los Angeles Lakers",
                                            "price": -110
                                        },
                                        {
                                            "name": "Boston Celtics",
                                            "price": -110
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_initialization(self, api_key):
        """Test client initialization."""
        client = ModernOddsAPIClient(
            api_key=api_key,
            timeout=30.0,
            rate_limit_delay=0.2
        )

        assert client.api_key == api_key
        assert client.timeout == 30.0
        assert client.rate_limit_delay == 0.2
        assert client.base_url == "https://api.the-odds-api.com/v4"
        assert len(client.cache) == 0
        assert client._session is None

    @pytest.mark.asyncio
    async def test_initialization_invalid_api_key(self):
        """Test client initialization with invalid API key."""
        with pytest.raises(ValidationError):
            ModernOddsAPIClient(api_key="")

        with pytest.raises(ValidationError):
            ModernOddsAPIClient(api_key=None)

    @pytest.mark.asyncio
    async def test_session_management(self, client):
        """Test async session management."""
        # Session should be created lazily
        assert client._session is None

        # First call should create session
        session = await client.get_session()
        assert session is not None
        assert client._session is session

        # Second call should return existing session
        session2 = await client.get_session()
        assert session is session2

        # Cleanup should close session
        await client.cleanup()
        assert client._session is None

    @pytest.mark.asyncio
    async def test_cache_key_generation(self, client):
        """Test cache key generation."""
        key1 = client._get_cache_key("odds", regions="us", markets="h2h")
        key2 = client._get_cache_key("odds", regions="uk", markets="h2h")
        key3 = client._get_cache_key("odds", regions="us", markets="h2h")

        assert key1 != key2  # Different regions
        assert key1 == key3  # Same parameters
        assert "odds" in key1
        assert "regions=us" in key1
        assert "markets=h2h" in key1

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, client):
        """Test cache TTL expiration."""
        # Add expired item to cache
        expired_key = "test_expired"
        client.cache[expired_key] = {
            "data": {"test": "data"},
            "timestamp": datetime.now().timestamp() - 2000  # > 30 minutes ago
        }

        # Should not return expired data
        result = client._get_from_cache(expired_key)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_storage_and_retrieval(self, client):
        """Test cache storage and retrieval."""
        test_key = "test_key"
        test_data = {"test": "data", "count": 123}

        # Store data
        client._store_in_cache(test_key, test_data)

        # Retrieve data
        result = client._get_from_cache(test_key)
        assert result == test_data

        # Verify cache structure
        cache_entry = client.cache[test_key]
        assert "data" in cache_entry
        assert "timestamp" in cache_entry
        assert cache_entry["data"] == test_data

    @pytest.mark.asyncio
    async def test_fetch_nba_odds_success(
        self,
        client,
        mock_httpx_client,
        sample_odds_response
    ):
        """Test successful NBA odds fetch."""
        # Mock successful response
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_odds_response
        mock_httpx_client.get.return_value = mock_response

        # Mock session
        with patch.object(client, 'get_session', return_value=mock_httpx_client):
            result = await client.fetch_nba_odds(
                regions="us",
                markets="h2h,spreads,totals"
            )

        # Verify response structure
        assert result is not None
        assert len(result) == 1

        odds = result[0]
        assert odds["id"] == "game123"
        assert odds["sport_key"] == "basketball_nba"
        assert odds["home_team"] == "Los Angeles Lakers"
        assert odds["away_team"] == "Boston Celtics"
        assert odds["bookmakers_count"] == 1

        # Verify bookmakers data
        bookmakers = odds["bookmakers"]
        assert len(bookmakers) == 1
        assert bookmakers[0]["key"] == "draftkings"
        assert bookmakers[0]["title"] == "DraftKings"

        # Verify API call was made correctly
        mock_httpx_client.get.assert_called_once()
        call_args = mock_httpx_client.get.call_args
        assert "sports/basketball_nba/odds" in call_args[0][0]
        assert call_args[1]["params"]["regions"] == "us"
        assert call_args[1]["params"]["markets"] == "h2h,spreads,totals"

    @pytest.mark.asyncio
    async def test_fetch_nba_odds_caching(
        self,
        client,
        mock_httpx_client,
        sample_odds_response
    ):
        """Test that NBA odds data is cached."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_odds_response
        mock_httpx_client.get.return_value = mock_response

        with patch.object(client, 'get_session', return_value=mock_httpx_client):
            # First call should make API request
            result1 = await client.fetch_nba_odds(regions="us", markets="h2h")

            # Second call should use cache
            result2 = await client.fetch_nba_odds(regions="us", markets="h2h")

        # Verify API was called only once
        mock_httpx_client.get.assert_called_once()

        # Verify results are identical
        assert result1 == result2

        # Verify cache contains data
        cache_key = client._get_cache_key("nba_odds", regions="us", markets="h2h", odds_format="american")
        assert cache_key in client.cache

    @pytest.mark.asyncio
    async def test_fetch_nba_scores_success(
        self,
        client,
        mock_httpx_client,
        sample_scores_response
    ):
        """Test successful NBA scores fetch."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_scores_response
        mock_httpx_client.get.return_value = mock_response

        with patch.object(client, 'get_session', return_value=mock_httpx_client):
            result = await client.fetch_nba_scores(days_from=1)

        # Verify response structure
        assert result is not None
        assert len(result) == 1

        scores = result[0]
        assert scores["id"] == "game123"
        assert scores["sport_key"] == "basketball_nba"
        assert scores["home_team"] == "Los Angeles Lakers"
        assert scores["away_team"] == "Boston Celtics"
        assert scores["completed"] is True

        # Verify scores data
        scores_list = scores["scores"]
        assert len(scores_list) == 2
        assert scores_list[0]["score"] == 110
        assert scores_list[1]["score"] == 105

    @pytest.mark.asyncio
    async def test_fetch_event_odds_success(
        self,
        client,
        mock_httpx_client,
        sample_odds_response
    ):
        """Test successful event odds fetch."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_odds_response
        mock_httpx_client.get.return_value = mock_response

        with patch.object(client, 'get_session', return_value=mock_httpx_client):
            result = await client.fetch_event_odds(
                event_id="game123",
                regions="us",
                markets="h2h"
            )

        # Verify response structure
        assert result is not None
        assert result["id"] == "game123"
        assert result["home_team"] == "Los Angeles Lakers"
        assert result["away_team"] == "Boston Celtics"

    @pytest.mark.asyncio
    async def test_fetch_historical_odds_success(
        self,
        client,
        mock_httpx_client,
        sample_historical_response
    ):
        """Test successful historical odds fetch."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_historical_response
        mock_httpx_client.get.return_value = mock_response

        snapshot_date = datetime(2024, 2, 1, 18, 0, 0)

        with patch.object(client, 'get_session', return_value=mock_httpx_client):
            result = await client.fetch_historical_odds(
                event_id="game123",
                snapshot_date=snapshot_date,
                regions="us",
                markets="h2h"
            )

        # Verify response structure
        assert result is not None
        assert "timestamp" in result
        assert result["timestamp"] == "2024-02-01T18:00:00Z"
        assert "games" in result
        assert len(result["games"]) == 1

        game = result["games"][0]
        assert game["id"] == "game123"
        assert game["home_team"] == "Los Angeles Lakers"

    @pytest.mark.asyncio
    async def test_fetch_nba_odds_with_date_filtering(
        self,
        client,
        mock_httpx_client,
        sample_odds_response
    ):
        """Test NBA odds fetch with date filtering."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_odds_response
        mock_httpx_client.get.return_value = mock_response

        with patch.object(client, 'get_session', return_value=mock_httpx_client):
            result = await client.fetch_nba_odds(
                regions="us",
                markets="h2h",
                date_from=date(2024, 2, 1),
                date_to=date(2024, 2, 2)
            )

        # Should return the game (Feb 1 is within the range)
        assert result is not None
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_rate_limiting(self, client, mock_httpx_client):
        """Test rate limiting delay."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_httpx_client.get.return_value = mock_response

        with patch.object(client, 'get_session', return_value=mock_httpx_client):
            with patch('asyncio.sleep') as mock_sleep:
                await client.fetch_nba_odds(regions="us", markets="h2h")

                # Verify sleep was called for rate limiting
                mock_sleep.assert_called_once_with(client.rate_limit_delay)

    @pytest.mark.asyncio
    async def test_http_error_handling(self, client, mock_httpx_client):
        """Test HTTP error handling."""
        # Mock 404 response
        mock_response = Mock(spec=Response)
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_httpx_client.get.return_value = mock_response

        with patch.object(client, 'get_session', return_value=mock_httpx_client):
            result = await client.fetch_nba_odds(regions="us", markets="h2h")

        # Should return None on error
        assert result is None

    @pytest.mark.asyncio
    async def test_network_error_handling(self, client, mock_httpx_client):
        """Test network error handling."""
        # Mock network timeout
        mock_httpx_client.get.side_effect = httpx.TimeoutException("Request timeout")

        with patch.object(client, 'get_session', return_value=mock_httpx_client):
            result = await client.fetch_nba_odds(regions="us", markets="h2h")

        # Should return None on error
        assert result is None

    @pytest.mark.asyncio
    async def test_json_parsing_error_handling(self, client, mock_httpx_client):
        """Test JSON parsing error handling."""
        # Mock response with invalid JSON
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_httpx_client.get.return_value = mock_response

        with patch.object(client, 'get_session', return_value=mock_httpx_client):
            result = await client.fetch_nba_odds(regions="us", markets="h2h")

        # Should return None on JSON parsing error
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_event_id(self, client):
        """Test handling of invalid event ID."""
        with pytest.raises(ValidationError):
            await client.fetch_event_odds(event_id="")

        with pytest.raises(ValidationError):
            await client.fetch_event_odds(event_id=None)

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, mock_httpx_client, api_key):
        """Test using client as async context manager."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_httpx_client.get.return_value = mock_response

        async with ModernOddsAPIClient(api_key=api_key) as client:
            with patch.object(client, 'get_session', return_value=mock_httpx_client):
                await client.fetch_nba_odds(regions="us", markets="h2h")

            # Session should be created
            assert client._session is not None

        # Session should be closed after context exit
        assert client._session is None

    @pytest.mark.asyncio
    async def test_custom_headers_and_timeout(self, api_key):
        """Test client with custom headers and timeout."""
        custom_headers = {"User-Agent": "Custom-Agent/1.0"}
        client = ModernOddsAPIClient(
            api_key=api_key,
            timeout=60.0,
            headers=custom_headers,
            rate_limit_delay=0.5
        )

        assert client.timeout == 60.0
        assert client.headers["User-Agent"] == "Custom-Agent/1.0"
        assert client.rate_limit_delay == 0.5

        await client.cleanup()  # Proper cleanup

    @pytest.mark.asyncio
    async def test_parse_bookmakers_with_missing_fields(self, client):
        """Test parsing bookmakers data with missing optional fields."""
        bookmakers_data = [
            {
                "key": "draftkings",
                # Missing title and other optional fields
                "markets": []
            }
        ]

        result = client._parse_bookmakers(bookmakers_data)

        assert len(result) == 1
        bookmaker = result[0]
        assert bookmaker["key"] == "draftkings"
        assert bookmaker["title"] == ""  # Default empty string
        assert bookmaker["markets"] == []

    @pytest.mark.asyncio
    async def test_parse_markets_with_spreads(self, client):
        """Test parsing markets data with spread information."""
        markets_data = [
            {
                "key": "spreads",
                "outcomes": [
                    {
                        "name": "Team A",
                        "price": -110,
                        "point": -2.5
                    },
                    {
                        "name": "Team B",
                        "price": -110,
                        "point": 2.5
                    }
                ]
            }
        ]

        result = client._parse_markets(markets_data)

        assert len(result) == 1
        market = result[0]
        assert market["key"] == "spreads"
        assert market["point"] == -2.5  # First outcome's point
        assert len(market["outcomes"]) == 2

    @pytest.mark.asyncio
    async def test_get_usage_stats(self, client, mock_httpx_client):
        """Test getting API usage statistics."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.headers = {
            'x-requests-remaining': '1000',
            'x-requests-used': '5',
            'x-requests-last': '1'
        }
        mock_httpx_client.get.return_value = mock_response

        with patch.object(client, 'get_session', return_value=mock_httpx_client):
            result = await client.get_usage_stats()

        assert result is not None
        assert result["status"] == "success"
        assert result["requests_remaining"] == "1000"
        assert result["requests_used"] == "5"
        assert result["requests_last"] == "1"

    @pytest.mark.asyncio
    async def test_filter_by_date_range(self, client):
        """Test filtering odds by date range."""
        odds_data = [
            {
                "id": "game1",
                "commence_time": "2024-02-01T19:00:00Z",
                "home_team": "Team A",
                "away_team": "Team B"
            },
            {
                "id": "game2",
                "commence_time": "2024-02-05T19:00:00Z",
                "home_team": "Team C",
                "away_team": "Team D"
            }
        ]

        # Filter by date range (Feb 1-3)
        filtered = client._filter_by_date_range(
            odds_data,
            date_from=date(2024, 2, 1),
            date_to=date(2024, 2, 3)
        )

        assert len(filtered) == 1
        assert filtered[0]["id"] == "game1"

    @pytest.mark.asyncio
    async def test_parse_outcomes_with_missing_point(self, client):
        """Test parsing outcomes data without point spread."""
        outcomes_data = [
            {
                "name": "Team A",
                "price": -110
                # Missing point field
            }
        ]

        result = client._parse_outcomes(outcomes_data)

        assert len(result) == 1
        outcome = result[0]
        assert outcome["name"] == "Team A"
        assert outcome["price"] == -110
        assert outcome["point"] is None

    def test_is_cache_expired(self, client):
        """Test cache expiration check."""
        # Fresh cache entry
        fresh_timestamp = datetime.now().timestamp() - 900  # 15 minutes ago
        assert not client._is_cache_expired(fresh_timestamp)

        # Expired cache entry (31 minutes ago, > 30 minute TTL)
        expired_timestamp = datetime.now().timestamp() - 1860
        assert client._is_cache_expired(expired_timestamp)