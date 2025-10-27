"""Unit tests for ModernNBAAPIClient.

This module provides comprehensive test coverage for the modern async NBA API client,
including HTTP client mocking, error handling, caching, and data parsing validation.
"""

import json
from datetime import date, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
import httpx
from httpx import Response

from nba_predictor.api.nba_client import ModernNBAAPIClient


class TestModernNBAAPIClient:
    """Test cases for ModernNBAAPIClient."""

    @pytest.fixture
    def client(self):
        """Create test client instance."""
        return ModernNBAAPIClient()

    @pytest.fixture
    def mock_httpx_client(self):
        """Mock httpx.AsyncClient for testing."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def sample_games_response(self):
        """Sample NBA API games response."""
        return {
            "league": {
                "standard": [
                    {
                        "gameId": "0022400001",
                        "gameUrlCode": "20240201/NYKGSW",
                        "gameStatusText": "Final",
                        "seasonId": "22023",
                        "gameDate": "2024-02-01T00:00:00Z",
                        "hTeam": {
                            "teamId": "1610612747",
                            "teamName": "Los Angeles Lakers",
                            "score": 120,
                            "linescore": [
                                {"period": 1, "score": 30},
                                {"period": 2, "score": 35}
                            ]
                        },
                        "vTeam": {
                            "teamId": "1610612740",
                            "teamName": "New Orleans Pelicans",
                            "score": 115,
                            "linescore": [
                                {"period": 1, "score": 28},
                                {"period": 2, "score": 32}
                            ]
                        },
                        "stats": {
                            "attendance": 18997,
                            "duration": "2:23"
                        }
                    }
                ]
            }
        }

    @pytest.fixture
    def sample_players_response(self):
        """Sample NBA API players response."""
        return {
            "league": {
                "standard": [
                    {
                        "personId": 2544,
                        "firstName": "LeBron",
                        "lastName": "James",
                        "teamId": "1610612747",
                        "pos": "F",
                        "heightFeet": 6,
                        "heightInches": 9,
                        "weightPounds": 250,
                        "dateOfBirthUTC": "1984-12-30T00:00:00Z",
                        "teams": [
                            {
                                "teamId": "1610612747",
                                "seasonStart": "2018",
                                "seasonEnd": "2023"
                            }
                        ]
                    }
                ]
            }
        }

    @pytest.fixture
    def sample_player_stats_response(self):
        """Sample NBA API player stats response."""
        return {
            "league": {
                "standard": [
                    {
                        "playerId": 2544,
                        "teamId": "1610612747",
                        "seasonId": "22023",
                        "stat": {
                            "pointsPerGame": 25.7,
                            "reboundsPerGame": 7.3,
                            "assistsPerGame": 8.3,
                            "stealsPerGame": 1.3,
                            "blocksPerGame": 0.5,
                            "fieldGoalsMade": 10.4,
                            "fieldGoalsAttempted": 21.1,
                            "fieldGoalPercentage": 0.492,
                            "threePointFieldGoalsMade": 2.3,
                            "threePointFieldGoalsAttempted": 6.4,
                            "threePointFieldGoalPercentage": 0.359,
                            "freeThrowsMade": 2.7,
                            "freeThrowsAttempted": 3.5,
                            "freeThrowPercentage": 0.775,
                            "gamesPlayed": 71,
                            "minutesPerGame": 35.5
                        }
                    }
                ]
            }
        }

    @pytest.fixture
    def sample_team_stats_response(self):
        """Sample NBA API team stats response."""
        return {
            "league": {
                "standard": [
                    {
                        "teamId": "1610612747",
                        "seasonId": "22023",
                        "stat": {
                            "pointsPerGame": 118.9,
                            "reboundsPerGame": 44.2,
                            "assistsPerGame": 27.4,
                            "stealsPerGame": 7.8,
                            "blocksPerGame": 4.9,
                            "fieldGoalsMade": 43.9,
                            "fieldGoalsAttempted": 90.7,
                            "fieldGoalPercentage": 0.484,
                            "threePointFieldGoalsMade": 13.2,
                            "threePointFieldGoalsAttempted": 35.8,
                            "threePointFieldGoalPercentage": 0.369,
                            "freeThrowsMade": 17.9,
                            "freeThrowsAttempted": 23.2,
                            "freeThrowPercentage": 0.771,
                            "turnoversPerGame": 14.2,
                            "pointsOffTurnoversPerGame": 17.8,
                            "fastBreakPointsPerGame": 13.5,
                            "pointsInPaintPerGame": 46.7
                        }
                    }
                ]
            }
        }

    @pytest.fixture
    def sample_standings_response(self):
        """Sample NBA API standings response."""
        return {
            "league": {
                "standard": {
                    "conference": {
                        "east": [
                            {
                                "teamId": "1610612748",
                                "teamName": "Miami Heat",
                                "conference": "east",
                                "playoffRank": 1,
                                "wins": 53,
                                "losses": 29,
                                "winPercentage": 0.646,
                                "lossPercentage": 0.354,
                                "conferenceGames": {
                                    "wins": 32,
                                    "losses": 20
                                },
                                "divisionGames": {
                                    "wins": 10,
                                    "losses": 8
                                },
                                "home": {
                                    "wins": 28,
                                    "losses": 13
                                },
                                "away": {
                                    "wins": 25,
                                    "losses": 16
                                },
                                "lastTenGames": {
                                    "wins": 7,
                                    "losses": 3
                                },
                                "currentStreak": 2,
                                "isStreakWinning": True,
                                "pointsPerGame": 110.1,
                                "opponentPointsPerGame": 108.9,
                                "differencePointsPerGame": 1.2
                            }
                        ],
                        "west": [
                            {
                                "teamId": "1610612747",
                                "teamName": "Los Angeles Lakers",
                                "conference": "west",
                                "playoffRank": 1,
                                "wins": 57,
                                "losses": 25,
                                "winPercentage": 0.695,
                                "lossPercentage": 0.305,
                                "conferenceGames": {
                                    "wins": 35,
                                    "losses": 17
                                },
                                "divisionGames": {
                                    "wins": 12,
                                    "losses": 6
                                },
                                "home": {
                                    "wins": 30,
                                    "losses": 11
                                },
                                "away": {
                                    "wins": 27,
                                    "losses": 14
                                },
                                "lastTenGames": {
                                    "wins": 8,
                                    "losses": 2
                                },
                                "currentStreak": 3,
                                "isStreakWinning": True,
                                "pointsPerGame": 118.9,
                                "opponentPointsPerGame": 108.9,
                                "differencePointsPerGame": 10.0
                            }
                        ]
                    }
                }
            }
        }

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test client initialization."""
        client = ModernNBAAPIClient(
            timeout=30.0,
            rate_limit_delay=0.2
        )

        assert client.timeout == 30.0
        assert client.rate_limit_delay == 0.2
        assert client.base_url == "https://stats.nba.com"
        assert len(client.cache) == 0
        assert client._session is None

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
        key1 = client._get_cache_key("players", season_year=2023)
        key2 = client._get_cache_key("players", season_year=2024)
        key3 = client._get_cache_key("players", season_year=2023)

        assert key1 != key2  # Different season
        assert key1 == key3  # Same parameters
        assert "players" in key1
        assert "2023" in key1

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, client):
        """Test cache TTL expiration."""
        # Add expired item to cache
        expired_key = "test_expired"
        client.cache[expired_key] = {
            "data": {"test": "data"},
            "timestamp": datetime.now().timestamp() - 4000  # > 1 hour ago
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
    async def test_fetch_all_games_success(
        self,
        client,
        mock_httpx_client,
        sample_games_response
    ):
        """Test successful games fetch."""
        # Mock successful response
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_games_response
        mock_httpx_client.get.return_value = mock_response

        # Mock session
        with patch.object(client, 'get_session', return_value=mock_httpx_client):
            result = await client.fetch_all_games(
                start_date=date(2024, 2, 1),
                end_date=date(2024, 2, 1),
                include_stats=True
            )

        # Verify response structure
        assert result is not None
        assert len(result) == 1

        game = result[0]
        assert game["game_id"] == "0022400001"
        assert game["home_team_id"] == 1610612747
        assert game["away_team_id"] == 1610612740
        assert game["home_score"] == 120
        assert game["away_score"] == 115
        assert game["season_id"] == "22023"

        # Verify API call was made correctly
        mock_httpx_client.get.assert_called_once()
        call_args = mock_httpx_client.get.call_args
        assert "games" in call_args[0][0]
        assert "20240201" in call_args[1]["params"]["GameDate"]

    @pytest.mark.asyncio
    async def test_fetch_all_games_with_stats(
        self,
        client,
        mock_httpx_client,
        sample_games_response
    ):
        """Test games fetch with stats included."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_games_response
        mock_httpx_client.get.return_value = mock_response

        with patch.object(client, 'get_session', return_value=mock_httpx_client):
            result = await client.fetch_all_games(
                start_date=date(2024, 2, 1),
                end_date=date(2024, 2, 1),
                include_stats=True
            )

        game = result[0]
        assert "attendance" in game
        assert "duration" in game
        assert game["attendance"] == 18997

    @pytest.mark.asyncio
    async def test_fetch_all_games_without_stats(
        self,
        client,
        mock_httpx_client,
        sample_games_response
    ):
        """Test games fetch without stats."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_games_response
        mock_httpx_client.get.return_value = mock_response

        with patch.object(client, 'get_session', return_value=mock_httpx_client):
            result = await client.fetch_all_games(
                start_date=date(2024, 2, 1),
                end_date=date(2024, 2, 1),
                include_stats=False
            )

        game = result[0]
        assert "attendance" not in game
        assert "duration" not in game

    @pytest.mark.asyncio
    async def test_fetch_players_success(
        self,
        client,
        mock_httpx_client,
        sample_players_response
    ):
        """Test successful players fetch."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_players_response
        mock_httpx_client.get.return_value = mock_response

        with patch.object(client, 'get_session', return_value=mock_httpx_client):
            result = await client.fetch_players(season_year=2023)

        # Verify response structure
        assert result is not None
        assert len(result) == 1

        player = result[0]
        assert player["player_id"] == 2544
        assert player["first_name"] == "LeBron"
        assert player["last_name"] == "James"
        assert player["team_id"] == 1610612747
        assert player["position"] == "F"
        assert player["height_feet"] == 6
        assert player["height_inches"] == 9
        assert player["weight_pounds"] == 250

    @pytest.mark.asyncio
    async def test_fetch_player_stats_success(
        self,
        client,
        mock_httpx_client,
        sample_player_stats_response
    ):
        """Test successful player stats fetch."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_player_stats_response
        mock_httpx_client.get.return_value = mock_response

        with patch.object(client, 'get_session', return_value=mock_httpx_client):
            result = await client.fetch_player_stats(player_id=2544, season_year=2023)

        # Verify response structure
        assert result is not None
        assert result["player_id"] == 2544
        assert result["team_id"] == 1610612747
        assert result["season_id"] == "22023"
        assert result["points_per_game"] == 25.7
        assert result["rebounds_per_game"] == 7.3
        assert result["assists_per_game"] == 8.3
        assert result["field_goal_percentage"] == 0.492

    @pytest.mark.asyncio
    async def test_fetch_team_stats_success(
        self,
        client,
        mock_httpx_client,
        sample_team_stats_response
    ):
        """Test successful team stats fetch."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_team_stats_response
        mock_httpx_client.get.return_value = mock_response

        with patch.object(client, 'get_session', return_value=mock_httpx_client):
            result = await client.fetch_team_stats(team_id=1610612747, season_year=2023)

        # Verify response structure
        assert result is not None
        assert result["team_id"] == 1610612747
        assert result["season_id"] == "22023"
        assert result["points_per_game"] == 118.9
        assert result["rebounds_per_game"] == 44.2
        assert result["assists_per_game"] == 27.4
        assert result["field_goal_percentage"] == 0.484

    @pytest.mark.asyncio
    async def test_fetch_standings_success(
        self,
        client,
        mock_httpx_client,
        sample_standings_response
    ):
        """Test successful standings fetch."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_standings_response
        mock_httpx_client.get.return_value = mock_response

        with patch.object(client, 'get_session', return_value=mock_httpx_client):
            result = await client.fetch_standings(season_year="2023")

        # Verify response structure
        assert result is not None
        assert len(result) == 2  # East and West

        # Check Eastern Conference
        east = result[0]
        assert east["team_id"] == 1610612748
        assert east["team_name"] == "Miami Heat"
        assert east["conference"] == "east"
        assert east["wins"] == 53
        assert east["losses"] == 29
        assert east["win_percentage"] == 0.646

        # Check Western Conference
        west = result[1]
        assert west["team_id"] == 1610612747
        assert west["team_name"] == "Los Angeles Lakers"
        assert west["conference"] == "west"
        assert west["wins"] == 57
        assert west["losses"] == 25
        assert west["win_percentage"] == 0.695

    @pytest.mark.asyncio
    async def test_fetch_all_games_caching(
        self,
        client,
        mock_httpx_client,
        sample_games_response
    ):
        """Test that games data is cached."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = sample_games_response
        mock_httpx_client.get.return_value = mock_response

        with patch.object(client, 'get_session', return_value=mock_httpx_client):
            # First call should make API request
            result1 = await client.fetch_all_games(
                start_date=date(2024, 2, 1),
                end_date=date(2024, 2, 1)
            )

            # Second call should use cache
            result2 = await client.fetch_all_games(
                start_date=date(2024, 2, 1),
                end_date=date(2024, 2, 1)
            )

        # Verify API was called only once
        mock_httpx_client.get.assert_called_once()

        # Verify results are identical
        assert result1 == result2

        # Verify cache contains data
        cache_key = client._get_cache_key("games", start_date="2024-02-01", end_date="2024-02-01")
        assert cache_key in client.cache

    @pytest.mark.asyncio
    async def test_rate_limiting(self, client, mock_httpx_client):
        """Test rate limiting delay."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"league": {"standard": []}}
        mock_httpx_client.get.return_value = mock_response

        with patch.object(client, 'get_session', return_value=mock_httpx_client):
            with patch('asyncio.sleep') as mock_sleep:
                await client.fetch_all_games(
                    start_date=date(2024, 2, 1),
                    end_date=date(2024, 2, 1)
                )

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
            result = await client.fetch_all_games(
                start_date=date(2024, 2, 1),
                end_date=date(2024, 2, 1)
            )

        # Should return None on error
        assert result is None

    @pytest.mark.asyncio
    async def test_network_error_handling(self, client, mock_httpx_client):
        """Test network error handling."""
        # Mock network timeout
        mock_httpx_client.get.side_effect = httpx.TimeoutException("Request timeout")

        with patch.object(client, 'get_session', return_value=mock_httpx_client):
            result = await client.fetch_all_games(
                start_date=date(2024, 2, 1),
                end_date=date(2024, 2, 1)
            )

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
            result = await client.fetch_all_games(
                start_date=date(2024, 2, 1),
                end_date=date(2024, 2, 1)
            )

        # Should return None on JSON parsing error
        assert result is None

    @pytest.mark.asyncio
    async def test_response_data_validation(self, client, mock_httpx_client):
        """Test response data validation."""
        # Mock response with missing expected data structure
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"invalid": "structure"}
        mock_httpx_client.get.return_value = mock_response

        with patch.object(client, 'get_session', return_value=mock_httpx_client):
            result = await client.fetch_all_games(
                start_date=date(2024, 2, 1),
                end_date=date(2024, 2, 1)
            )

        # Should return None for invalid data structure
        assert result is None

    @pytest.mark.asyncio
    async def test_parse_games_data_with_missing_optional_fields(
        self,
        client,
        sample_games_response
    ):
        """Test parsing games data with missing optional fields."""
        # Remove optional fields from test data
        games_data = sample_games_response.copy()
        games_data["league"]["standard"][0].pop("stats", None)

        result = client._parse_games_data(games_data, include_stats=True)

        assert len(result) == 1
        game = result[0]
        assert "attendance" not in game
        assert "duration" not in game
        # Required fields should still be present
        assert game["game_id"] == "0022400001"

    @pytest.mark.asyncio
    async def test_parse_players_data_with_missing_fields(self, client):
        """Test parsing players data with missing optional fields."""
        players_data = {
            "league": {
                "standard": [
                    {
                        "personId": 2544,
                        "firstName": "LeBron",
                        "lastName": "James",
                        # Missing optional fields
                    }
                ]
            }
        }

        result = client._parse_players_data(players_data)

        assert len(result) == 1
        player = result[0]
        assert player["player_id"] == 2544
        assert player["first_name"] == "LeBron"
        assert player["last_name"] == "James"
        # Optional fields should have default values
        assert player["team_id"] is None
        assert player["position"] is None

    @pytest.mark.asyncio
    async def test_parse_stats_data_with_missing_fields(self, client):
        """Test parsing stats data with missing optional fields."""
        stats_data = {
            "league": {
                "standard": [
                    {
                        "playerId": 2544,
                        "teamId": 1610612747,
                        "seasonId": "22023",
                        "stat": {
                            "pointsPerGame": 25.7,
                            # Missing other stats
                        }
                    }
                ]
            }
        }

        result = client._parse_stats_data(stats_data, "player")

        assert len(result) == 1
        stats = result[0]
        assert stats["player_id"] == 2544
        assert stats["points_per_game"] == 25.7
        # Optional fields should have default values
        assert stats["rebounds_per_game"] == 0.0

    def test_is_cache_expired(self, client):
        """Test cache expiration check."""
        # Fresh cache entry
        fresh_timestamp = datetime.now().timestamp() - 1000  # 16 minutes ago
        assert not client._is_cache_expired(fresh_timestamp)

        # Expired cache entry
        expired_timestamp = datetime.now().timestamp() - 4000  # Over 1 hour ago
        assert client._is_cache_expired(expired_timestamp)

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, mock_httpx_client):
        """Test using client as async context manager."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"league": {"standard": []}}
        mock_httpx_client.get.return_value = mock_response

        async with ModernNBAAPIClient() as client:
            with patch.object(client, 'get_session', return_value=mock_httpx_client):
                await client.fetch_all_games(
                    start_date=date(2024, 2, 1),
                    end_date=date(2024, 2, 1)
                )

            # Session should be created
            assert client._session is not None

        # Session should be closed after context exit
        assert client._session is None

    @pytest.mark.asyncio
    async def test_custom_headers_and_timeout(self):
        """Test client with custom headers and timeout."""
        custom_headers = {"User-Agent": "Custom-Agent/1.0"}
        client = ModernNBAAPIClient(
            timeout=60.0,
            headers=custom_headers,
            rate_limit_delay=0.5
        )

        assert client.timeout == 60.0
        assert client.headers["User-Agent"] == "Custom-Agent/1.0"
        assert client.rate_limit_delay == 0.5

        await client.cleanup()  # Proper cleanup

    @pytest.mark.asyncio
    async def test_parse_standings_data_with_missing_fields(self, client):
        """Test parsing standings data with missing optional fields."""
        standings_data = {
            "league": {
                "standard": {
                    "conference": {
                        "east": [
                            {
                                "teamId": 1610612748,
                                "teamName": "Miami Heat",
                                "conference": "east",
                                # Missing optional fields
                            }
                        ]
                    }
                }
            }
        }

        result = client._parse_standings_data(standings_data)

        assert len(result) == 1
        standings = result[0]
        assert standings["team_id"] == 1610612748
        assert standings["team_name"] == "Miami Heat"
        assert standings["conference"] == "east"
        # Optional fields should have default values
        assert standings["wins"] == 0
        assert standings["losses"] == 0
        assert standings["win_percentage"] == 0.0