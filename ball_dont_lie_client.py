#!/usr/bin/env python3
"""
🏀 NBA BallDontLie API Client with Rate Limiting

This module provides a robust client for the BallDontLie API that delivers
official NBA schedule data with built-in rate limiting to respect API limits.

Key Features:
- Real NBA games schedule (not betting odds)
- Rate limiting (5 requests/minute for free tier)
- Comprehensive error handling
- Data format conversion to internal NBA format
- Single date and date range support
"""

import os
import time
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from dotenv import load_dotenv

# BallDontLie API imports
from balldontlie import BalldontlieAPI  # type: ignore
from pyrate_limiter import Duration, Rate, Limiter, BucketFullException

# Configure logging
logger = logging.getLogger(__name__)

# Custom exceptions
class RateLimitException(Exception):
    """Raised when API rate limit is exceeded."""
    pass

class APIException(Exception):
    """Raised when API call fails."""
    pass

@dataclass
class NBAGame:
    """Standard NBA game data structure."""
    away_team: str
    home_team: str
    away_team_id: int
    home_team_id: int
    game_id: str
    date: str
    time: str
    time_utc: str
    status: str
    score: str
    away_score: int = 0
    home_score: int = 0
    period: int = 0
    season: int = 2025

class NBABallDontLieClient:
    """
    NBA BallDontLie API client with rate limiting for real NBA games data.

    Provides access to official NBA schedule and game data with built-in
    rate limiting to respect API limits (5 requests/minute on free tier).

    Attributes:
        api: BallDontLie API client instance
        limiter: PyrateLimiter instance for rate control
        logger: Logger instance for debugging
    """

    def __init__(self, api_key: str) -> None:
        """
        Initialize BallDontLie API client with rate limiting.

        Args:
            api_key: BallDontLie API key from environment

        Raises:
            ValueError: If api_key is None or empty
            Exception: If API client initialization fails
        """
        if not api_key:
            raise ValueError("BallDontLie API key is required")

        try:
            # Initialize BallDontLie API client
            self.api = BalldontlieAPI(api_key=api_key)

            # Initialize rate limiter (5 requests per minute for free tier)
            rate = Rate(5, Duration.MINUTE)
            self.limiter = Limiter(rate)

            # Setup logging
            self.logger = logging.getLogger(__name__)

            self.logger.info("✅ NBABallDontLieClient initialized successfully")
            self.logger.info(f"   🏀 BallDontLie API: Connected")
            self.logger.info(f"   🚦 Rate Limiting: 5 requests/minute")

        except Exception as e:
            logger.error(f"❌ Failed to initialize BallDontLie client: {e}")
            raise APIException(f"Failed to initialize BallDontLie API client: {e}") from e

    def get_games_for_date_range(
        self,
        start_date: date,
        end_date: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """
        Get NBA games for specified date range with rate limiting.

        Args:
            start_date: Start date for games search
            end_date: End date for games search (default: start_date)

        Returns:
            List of NBA games with complete information

        Raises:
            RateLimitException: When API rate limit is exceeded
            APIException: When API call fails
            ValueError: If date range is invalid

        Example:
            >>> client = NBABallDontLieClient("api_key")
            >>> games = client.get_games_for_date_range(date(2025, 10, 27))
            >>> print(f"Found {len(games)} games")
        """
        if end_date is None:
            end_date = start_date

        if start_date > end_date:
            raise ValueError("Start date cannot be after end date")

        # Calculate date difference to validate range
        date_diff = (end_date - start_date).days
        if date_diff > 5:
            self.logger.warning(f"Date range of {date_diff + 1} days exceeds recommended 5-day limit")

        try:
            # Apply rate limiting before API call
            self.limiter.try_acquire("api_call")

            # Convert dates to BallDontLie format (YYYY-MM-DD)
            date_list = []
            current_date = start_date
            while current_date <= end_date:
                date_list.append(current_date.strftime('%Y-%m-%d'))
                current_date += timedelta(days=1)

            self.logger.info(f"🏀 BallDontLie API: Requesting games for {len(date_list)} date(s)")
            self.logger.info(f"   📅 Date range: {date_list[0]} to {date_list[-1]}")

            # Make API call
            games_response = self.api.nba.games.list(dates=date_list)

            if games_response and hasattr(games_response, 'data'):
                games_data = games_response.data
                self.logger.info(f"✅ BallDontLie API: {len(games_data)} games retrieved")

                # Convert to internal format
                processed_games = []
                for game in games_data:
                    try:
                        nba_game = self._convert_ball_dont_lie_game_to_nba_format(game)
                        processed_games.append(nba_game)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error processing game: {e}")
                        continue

                self.logger.info(f"   🔄 Processed {len(processed_games)} games to internal format")
                return processed_games
            else:
                self.logger.warning("⚠️ BallDontLie API: No games data returned")
                return []

        except BucketFullException as e:
            reset_time = e.meta_info.get("reset_time", 60)
            self.logger.error(
                f"🚦 Rate limit exceeded",
                extra={"error": str(e), "retry_after": f"{reset_time} seconds"}
            )
            raise RateLimitException(f"API rate limit exceeded. Please wait {reset_time} seconds.") from e

        except Exception as e:
            self.logger.error(
                f"❌ BallDontLie API call failed",
                extra={"start_date": start_date.isoformat(), "end_date": end_date.isoformat(), "error": str(e)}
            )
            raise APIException(f"Failed to fetch games from BallDontLie API: {e}") from e

    def _convert_ball_dont_lie_game_to_nba_format(
        self,
        bdl_game: Any
    ) -> Dict[str, Any]:
        """
        Convert BallDontLie game format to internal NBA format.

        Args:
            bdl_game: BallDontLie game object

        Returns:
            Standardized NBA game dictionary

        Raises:
            ValueError: If required game data is missing
        """
        try:
            # Extract game information
            home_team = bdl_game.home_team
            away_team = bdl_game.visitor_team

            # Handle game status and time
            status = "Scheduled"
            score = ""
            time_str = ""

            if hasattr(bdl_game, 'status') and bdl_game.status:
                if bdl_game.status.lower() == 'final':
                    status = "Final"
                    if hasattr(bdl_game, 'home_team_score') and hasattr(bdl_game, 'visitor_team_score'):
                        score = f"{bdl_game.visitor_team_score}-{bdl_game.home_team_score}"
                elif bdl_game.status.lower() in ['1st q', '2nd q', '3rd q', '4th q']:
                    status = "In Progress"
                    if hasattr(bdl_game, 'home_team_score') and hasattr(bdl_game, 'visitor_team_score'):
                        score = f"{bdl_game.visitor_team_score}-{bdl_game.home_team_score}"

            # Parse date and time
            game_date = ""
            game_time = ""
            game_time_utc = ""

            # Use the status field for accurate UTC timestamps (contains datetime from API)
            if hasattr(bdl_game, 'status') and bdl_game.status and 'T' in bdl_game.status:
                # BallDontLie provides UTC timestamp in status field like "2025-10-27T23:00:00Z"
                datetime_str = bdl_game.status
                try:
                    # Parse the UTC datetime
                    dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                    game_date = dt.date().strftime('%Y-%m-%d')
                    game_time = dt.strftime('%H:%M')
                    game_time_utc = dt.isoformat()
                    self.logger.debug(f"✅ Parsed UTC datetime: {datetime_str} -> {game_date} {game_time}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Error parsing status datetime {datetime_str}: {e}")
                    # Fallback to date field if datetime parsing fails
                    if hasattr(bdl_game, 'date'):
                        game_date = bdl_game.date[:10]
            elif hasattr(bdl_game, 'date'):
                # Fallback to date field if status field doesn't contain datetime
                date_str = bdl_game.date
                self.logger.warning(f"⚠️ Using fallback date field: {date_str}")
                game_date = date_str[:10]  # Just the date part
                game_time = "00:00"  # Default time
                game_time_utc = f"{date_str}T00:00:00Z"

            # Build standardized game object
            nba_game: Dict[str, Any] = {
                'away_team': away_team.full_name if hasattr(away_team, 'full_name') else str(away_team),
                'home_team': home_team.full_name if hasattr(home_team, 'full_name') else str(home_team),
                'away_team_id': away_team.id if hasattr(away_team, 'id') else 0,
                'home_team_id': home_team.id if hasattr(home_team, 'id') else 0,
                'game_id': f"BDL_{bdl_game.id}" if hasattr(bdl_game, 'id') else f"BDL_unknown",
                'date': game_date,  # UTC date from datetime field
                'time': game_time,  # UTC time from datetime field
                'time_utc': game_time_utc,
                'utc_datetime': game_time_utc,  # For main app timezone processing
                'status': status,
                'score': score,
                'away_score': bdl_game.visitor_team_score if hasattr(bdl_game, 'visitor_team_score') else 0,
                'home_score': bdl_game.home_team_score if hasattr(bdl_game, 'home_team_score') else 0,
                'period': bdl_game.period if hasattr(bdl_game, 'period') else 0,
                'season': bdl_game.season if hasattr(bdl_game, 'season') else 2025,
                'odds': {},  # BallDontLie doesn't provide odds
                'bookmakers_count': 0,
                'source': 'BallDontLie API (Official NBA Schedule)',
                'api_endpoint': 'api.balldontlie.io/v1/games',
                'commence_time_utc': game_time_utc
            }

            return nba_game

        except Exception as e:
            logger.error(f"❌ Error converting BallDontLie game format: {e}")
            raise ValueError(f"Failed to convert BallDontLie game data: {e}") from e

    def test_connection(self) -> bool:
        """
        Test API connection with a minimal request.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.logger.info("🔍 Testing BallDontLie API connection...")

            # Apply rate limiting
            self.limiter.try_acquire("test_call")

            # Make a minimal API call (today's games)
            today = date.today()
            today_str = today.strftime('%Y-%m-%d')

            response = self.api.nba.games.list(dates=[today_str], per_page=1)

            if response is not None:
                self.logger.info("✅ BallDontLie API connection test successful")
                return True
            else:
                self.logger.error("❌ BallDontLie API connection test failed: No response")
                return False

        except BucketFullException as e:
            self.logger.warning(f"⚠️ Rate limit during connection test: {e}")
            return True  # Rate limit means API is working

        except Exception as e:
            self.logger.error(f"❌ BallDontLie API connection test failed: {e}")
            return False


def main() -> bool:
    """Test the BallDontLie client implementation."""
    print("🏀 TEST NBA BALLDONTLIE CLIENT")
    print("=" * 50)

    # Load environment variables
    load_dotenv()

    # Load API key from environment
    api_key = os.getenv('BALLDONTLIE_API_KEY')
    if not api_key:
        print("❌ BALLDONTLIE_API_KEY not found in environment")
        return False

    try:
        # Initialize client
        print("🔧 Initializing BallDontLie client...")
        client = NBABallDontLieClient(api_key)

        # Test connection
        print("\n🔍 Testing API connection...")
        connection_ok = client.test_connection()
        print(f"   Connection: {'✅ OK' if connection_ok else '❌ FAILED'}")

        # Test today's games
        print(f"\n📅 Testing today's games ({date.today()})...")
        today_games = client.get_games_for_date_range(date.today())
        print(f"   Games found: {len(today_games)}")

        if today_games:
            print("\n🏀 Sample games:")
            for i, game in enumerate(today_games[:3], 1):
                print(f"   {i}. {game['away_team']} @ {game['home_team']}")
                print(f"      Date: {game['date']} Time: {game['time']}")
                print(f"      Status: {game['status']} Score: {game['score']}")
                print(f"      Source: {game['source']}")

        # Test date range
        print(f"\n📅 Testing 3-day range...")
        start_date = date.today()
        end_date = date.today() + timedelta(days=2)
        range_games = client.get_games_for_date_range(start_date, end_date)
        print(f"   Games in range: {len(range_games)}")

        print("\n✅ BallDontLie client test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ BallDontLie client test failed: {e}")
        return False


if __name__ == "__main__":
    success: bool = main()
    exit(0 if success else 1)