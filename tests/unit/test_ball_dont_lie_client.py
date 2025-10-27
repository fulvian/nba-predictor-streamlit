#!/usr/bin/env python3
"""
Unit tests for BallDontLie API client.
"""

import pytest
import os
from datetime import date, timedelta
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv

# Load environment variables before importing the client
load_dotenv()

from ball_dont_lie_client import NBABallDontLieClient, RateLimitException, APIException


class TestNBABallDontLieClient:
    """Test suite for NBABallDontLieClient."""

    @pytest.fixture
    def api_key(self):
        """Get API key from environment or use test key."""
        return os.getenv('BALLDONTLIE_API_KEY', 'test-api-key')

    @pytest.fixture
    def client(self, api_key):
        """Create client instance for testing."""
        with patch('ball_dont_lie_client.BalldontlieAPI'), \
             patch('ball_dont_lie_client.Limiter'):
            return NBABallDontLieClient(api_key)

    def test_init_with_valid_key(self, api_key):
        """Test client initialization with valid API key."""
        with patch('ball_dont_lie_client.BalldontlieAPI') as mock_api, \
             patch('ball_dont_lie_client.Limiter') as mock_limiter:

            client = NBABallDontLieClient(api_key)

            assert client is not None
            mock_api.assert_called_once_with(api_key=api_key)
            mock_limiter.assert_called_once()

    def test_init_with_empty_key(self):
        """Test client initialization fails with empty API key."""
        with pytest.raises(ValueError, match="BallDontLie API key is required"):
            NBABallDontLieClient("")

    def test_init_with_none_key(self):
        """Test client initialization fails with None API key."""
        with pytest.raises(ValueError, match="BallDontLie API key is required"):
            NBABallDontLieClient(None)

    def test_date_range_validation(self, client):
        """Test date range validation."""
        start_date = date(2025, 10, 27)
        end_date = date(2025, 10, 25)  # Before start date

        with pytest.raises(ValueError, match="Start date cannot be after end date"):
            client.get_games_for_date_range(start_date, end_date)

    def test_single_date_request(self, client):
        """Test getting games for a single date."""
        test_date = date(2025, 10, 27)

        # Mock API response
        mock_game = Mock()
        mock_game.id = 18446861
        mock_game.date = "2025-10-27T23:00:00Z"
        mock_game.season = 2025
        mock_game.status = "Scheduled"
        mock_game.home_team_score = 0
        mock_game.visitor_team_score = 0
        mock_game.period = 0
        mock_game.home_team = Mock()
        mock_game.home_team.id = 9
        mock_game.home_team.full_name = "Detroit Pistons"
        mock_game.visitor_team = Mock()
        mock_game.visitor_team.id = 6
        mock_game.visitor_team.full_name = "Cleveland Cavaliers"

        mock_response = Mock()
        mock_response.data = [mock_game]

        with patch.object(client.api.nba.games, 'list', return_value=mock_response), \
             patch.object(client.limiter, 'try_acquire'):

            games = client.get_games_for_date_range(test_date)

            assert len(games) == 1
            assert games[0]['away_team'] == "Cleveland Cavaliers"
            assert games[0]['home_team'] == "Detroit Pistons"
            assert games[0]['date'] == "2025-10-27"
            assert games[0]['status'] == "Scheduled"

    def test_date_range_request(self, client):
        """Test getting games for a date range."""
        start_date = date(2025, 10, 27)
        end_date = date(2025, 10, 28)

        # Mock API response with multiple games
        mock_games = []
        for i in range(3):
            mock_game = Mock()
            mock_game.id = 18446861 + i
            mock_game.date = f"2025-10-{27 + i}T23:00:00Z"
            mock_game.season = 2025
            mock_game.status = "Scheduled"
            mock_game.home_team_score = 0
            mock_game.visitor_team_score = 0
            mock_game.period = 0
            mock_game.home_team = Mock()
            mock_game.home_team.id = 9 + i
            mock_game.home_team.full_name = f"Home Team {i}"
            mock_game.visitor_team = Mock()
            mock_game.visitor_team.id = 6 + i
            mock_game.visitor_team.full_name = f"Away Team {i}"
            mock_games.append(mock_game)

        mock_response = Mock()
        mock_response.data = mock_games

        with patch.object(client.api.nba.games, 'list', return_value=mock_response), \
             patch.object(client.limiter, 'try_acquire'):

            games = client.get_games_for_date_range(start_date, end_date)

            assert len(games) == 3
            # Verify API was called with correct dates
            client.api.nba.games.list.assert_called_once()
            call_args = client.api.nba.games.list.call_args
            dates = call_args[1]['dates']
            assert '2025-10-27' in dates
            assert '2025-10-28' in dates

    def test_rate_limiting(self, client):
        """Test rate limiting behavior."""
        from pyrate_limiter import BucketFullException

        with patch.object(client.limiter, 'try_acquire', side_effect=BucketFullException("test_item", None)):
            with pytest.raises(RateLimitException, match="API rate limit exceeded"):
                client.get_games_for_date_range(date(2025, 10, 27))

    def test_api_exception_handling(self, client):
        """Test API exception handling."""
        with patch.object(client.limiter, 'try_acquire'), \
             patch.object(client.api.nba.games, 'list', side_effect=Exception("API Error")):

            with pytest.raises(APIException, match="Failed to fetch games"):
                client.get_games_for_date_range(date(2025, 10, 27))

    def test_empty_response_handling(self, client):
        """Test handling of empty API response."""
        mock_response = Mock()
        mock_response.data = []

        with patch.object(client.api.nba.games, 'list', return_value=mock_response), \
             patch.object(client.limiter, 'try_acquire'):

            games = client.get_games_for_date_range(date(2025, 10, 27))

            assert games == []

    def test_game_format_conversion(self, client):
        """Test conversion of BallDontLie game format to NBA format."""
        mock_game = Mock()
        mock_game.id = 18446861
        mock_game.date = "2025-10-27T23:00:00Z"
        mock_game.season = 2025
        mock_game.status = "Final"
        mock_game.home_team_score = 110
        mock_game.visitor_team_score = 105
        mock_game.period = 4
        mock_game.home_team = Mock()
        mock_game.home_team.id = 9
        mock_game.home_team.full_name = "Detroit Pistons"
        mock_game.visitor_team = Mock()
        mock_game.visitor_team.id = 6
        mock_game.visitor_team.full_name = "Cleveland Cavaliers"

        mock_response = Mock()
        mock_response.data = [mock_game]

        with patch.object(client.api.nba.games, 'list', return_value=mock_response), \
             patch.object(client.limiter, 'try_acquire'):

            games = client.get_games_for_date_range(date(2025, 10, 27))

            assert len(games) == 1
            game = games[0]
            assert game['away_team'] == "Cleveland Cavaliers"
            assert game['home_team'] == "Detroit Pistons"
            assert game['away_team_id'] == 6
            assert game['home_team_id'] == 9
            assert game['game_id'] == "BDL_18446861"
            assert game['date'] == "2025-10-27"
            assert game['status'] == "Final"
            assert game['score'] == "105-110"
            assert game['away_score'] == 105
            assert game['home_score'] == 110
            assert game['period'] == 4
            assert game['season'] == 2025
            assert game['source'] == "BallDontLie API (Official NBA Schedule)"

    def test_date_range_warning(self, client):
        """Test warning for date range exceeding 5 days."""
        start_date = date(2025, 10, 27)
        end_date = date(2025, 11, 5)  # 9 days later

        mock_response = Mock()
        mock_response.data = []

        with patch.object(client.api.nba.games, 'list', return_value=mock_response), \
             patch.object(client.limiter, 'try_acquire'), \
             patch.object(client.logger, 'warning') as mock_warning:

            client.get_games_for_date_range(start_date, end_date)

            mock_warning.assert_called_once()
            assert "exceeds recommended 5-day limit" in mock_warning.call_args[0][0]

    def test_connection_test_success(self, client):
        """Test successful API connection test."""
        mock_response = Mock()

        with patch.object(client.limiter, 'try_acquire'), \
             patch.object(client.api.nba.games, 'list', return_value=mock_response):

            result = client.test_connection()

            assert result is True

    def test_connection_test_failure(self, client):
        """Test failed API connection test."""
        with patch.object(client.limiter, 'try_acquire'), \
             patch.object(client.api.nba.games, 'list', side_effect=Exception("Connection failed")):

            result = client.test_connection()

            assert result is False

    def test_connection_test_rate_limit(self, client):
        """Test connection test with rate limit (should still return True)."""
        from pyrate_limiter import BucketFullException

        with patch.object(client.limiter, 'try_acquire', side_effect=BucketFullException("test_item", None)):
            result = client.test_connection()

            # Rate limit during connection test means API is working
            assert result is True


# Integration test (only runs if API key is available and not in CI)
@pytest.mark.integration
class TestBallDontLieIntegration:
    """Integration tests for BallDontLie client with real API."""

    @pytest.fixture(autouse=True)
    def skip_if_no_api_key(self):
        """Skip integration tests if no API key is available."""
        if not os.getenv('BALLDONTLIE_API_KEY'):
            pytest.skip("BALLDONTLIE_API_KEY not available for integration tests")

    def test_real_api_connection(self):
        """Test real API connection."""
        client = NBABallDontLieClient(os.getenv('BALLDONTLIE_API_KEY'))

        result = client.test_connection()
        assert result is True

    def test_real_today_games(self):
        """Test getting today's games from real API."""
        client = NBABallDontLieClient(os.getenv('BALLDONTLIE_API_KEY'))

        # Get today's games
        games = client.get_games_for_date_range(date.today())

        # Verify structure of returned games
        for game in games:
            assert 'away_team' in game
            assert 'home_team' in game
            assert 'date' in game
            assert 'status' in game
            assert 'source' in game
            assert game['source'] == "BallDontLie API (Official NBA Schedule)"
            assert isinstance(game['away_team_id'], int)
            assert isinstance(game['home_team_id'], int)
            assert game['game_id'].startswith('BDL_')


if __name__ == "__main__":
    # Run basic test if executed directly
    pytest.main([__file__, "-v"])