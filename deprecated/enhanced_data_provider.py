#!/usr/bin/env python3
"""
Enhanced NBA Data Provider with Live Data API as primary endpoint
Context7-compliant solution to NBA API connectivity issues
"""

import requests
import json
import traceback
from datetime import datetime, date, timedelta
from dateutil import parser

class EnhancedNBADataProvider:
    """Enhanced NBA Data Provider using Live Data API as primary endpoint"""

    def __init__(self):
        """Initialize the enhanced data provider"""
        self.timeout = 30
        self.headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate, br',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        print("✅ EnhancedNBADataProvider initialized with Live Data API support")

    def _try_live_data_api(self, target_date=None):
        """Try NBA Live Data API (primary endpoint)"""
        try:
            print(f"   🔄 NBA Live Data API - Primary endpoint")

            # Import Live Data endpoint
            from nba_api.live.nba.endpoints import scoreboard

            # Create ScoreBoard instance
            board = scoreboard.ScoreBoard()

            # Get games
            games_dict = board.games.get_dict()

            if games_dict:
                scheduled_games = []
                board_date = board.score_board_date

                # Convert target_date to string for comparison
                if target_date:
                    target_date_str = target_date.strftime('%Y-%m-%d')
                    board_date_str = datetime.strptime(board_date, '%Y-%m-%d').strftime('%Y-%m-%d')

                    # Only return games for the target date
                    if target_date_str != board_date_str:
                        print(f"   ℹ️  Live API has games for {board_date_str}, not {target_date_str}")
                        return scheduled_games

                # Process games
                for game in games_dict:
                    try:
                        # Extract game information
                        away_team = game.get('awayTeam', {}).get('teamName', 'Unknown')
                        home_team = game.get('homeTeam', {}).get('teamName', 'Unknown')
                        game_id = game.get('gameId', 'N/A')
                        game_status = game.get('gameStatusText', 'Unknown')
                        game_time_utc = game.get('gameTimeUTC', '')

                        # Parse game time
                        game_date = board_date
                        if game_time_utc:
                            try:
                                game_dt = parser.parse(game_time_utc).replace(tzinfo=datetime.timezone.utc)
                                game_date = game_dt.strftime('%Y-%m-%d')
                            except:
                                pass

                        # Add score if game is final or in progress
                        score_info = ""
                        away_score = game.get('awayTeam', {}).get('score', 0)
                        home_score = game.get('homeTeam', {}).get('score', 0)

                        if away_score and home_score and game_status in ['Final', 'In Progress', 'Q1', 'Q2', 'Q3', 'Q4']:
                            score_info = f" ({away_score}-{home_score})"

                        scheduled_games.append({
                            'away_team': away_team,
                            'home_team': home_team,
                            'away_team_id': game.get('awayTeam', {}).get('teamId', 0),
                            'home_team_id': game.get('homeTeam', {}).get('teamId', 0),
                            'game_id': game_id,
                            'date': game_date,
                            'time_utc': game_time_utc,
                            'status': game_status,
                            'score': f"{away_score}-{home_score}" if away_score and home_score else "",
                            'source': 'NBA Live Data API (CDN)',
                            'api_endpoint': 'cdn.nba.com/static/json/liveData/scoreboard'
                        })

                    except Exception as e:
                        print(f"   ⚠️  Error processing game {game.get('gameId', 'unknown')}: {e}")
                        continue

                print(f"   ✅ Live Data API: Found {len(scheduled_games)} games")
                return scheduled_games
            else:
                print(f"   ❌ Live Data API: No games found")
                return []

        except Exception as e:
            print(f"   ❌ Live Data API Error: {str(e)}")
            return []

    def _try_stats_api_fallback(self, specific_date):
        """Try original Stats API as fallback"""
        try:
            print(f"   🔄 Stats API Fallback - {specific_date}")

            # Import the original data provider
            from data_provider import NBADataProvider
            dp = NBADataProvider()

            # Get games using original method
            games = dp.get_scheduled_games(specific_date=specific_date)

            if games:
                print(f"   ✅ Stats API Fallback: Found {len(games)} games")
                return games
            else:
                print(f"   ❌ Stats API Fallback: No games found")
                return []

        except Exception as e:
            print(f"   ❌ Stats API Fallback Error: {str(e)}")
            return []

    def get_scheduled_games(self, specific_date=None):
        """
        Get NBA scheduled games using enhanced multi-endpoint approach
        Primary: Live Data API, Fallback: Stats API
        """
        print(f"🏀 Enhanced NBA Game Detection - Multi-Endpoint Approach")

        if specific_date:
            target_date = datetime.strptime(specific_date, '%Y-%m-%d').date()
        else:
            target_date = date.today()

        print(f"📅 Target Date: {target_date}")

        scheduled_games = []

        # Try Live Data API first (fast and reliable)
        try:
            live_games = self._try_live_data_api(target_date)
            if live_games:
                scheduled_games.extend(live_games)
        except Exception as e:
            print(f"   ⚠️  Live Data API failed: {e}")

        # If no games from Live API, try Stats API as fallback
        if not scheduled_games:
            try:
                stats_games = self._try_stats_api_fallback(target_date.strftime('%Y-%m-%d'))
                if stats_games:
                    scheduled_games.extend(stats_games)
            except Exception as e:
                print(f"   ⚠️  Stats API fallback failed: {e}")

        # Remove duplicates based on game_id
        seen_game_ids = set()
        unique_games = []
        for game in scheduled_games:
            if game.get('game_id') not in seen_game_ids:
                seen_game_ids.add(game.get('game_id'))
                unique_games.append(game)

        print(f"📊 Final Result: {len(unique_games)} unique games found")

        if unique_games:
            print("🏀 GAMES DETECTED:")
            for i, game in enumerate(unique_games, 1):
                score_text = f" [{game.get('score', '')}]" if game.get('score') else ""
                source = game.get('source', 'Unknown')
                print(f"   {i}. {game['away_team']} @ {game['home_team']}{score_text} - {source}")

        return unique_games

    def test_api_performance(self):
        """Test performance of both APIs"""
        print("🔧 API Performance Test")
        print("=" * 50)

        results = {}

        # Test Live Data API
        print("\n📡 Testing Live Data API...")
        try:
            import time
            start_time = time.time()
            live_games = self._try_live_data_api()
            live_time = time.time() - start_time

            results['live_api'] = {
                'success': True,
                'games_count': len(live_games),
                'response_time': live_time
            }
            print(f"✅ Live API: {len(live_games)} games in {live_time:.2f}s")

        except Exception as e:
            results['live_api'] = {
                'success': False,
                'error': str(e),
                'response_time': 0
            }
            print(f"❌ Live API failed: {e}")

        # Test Stats API
        print("\n📊 Testing Stats API...")
        try:
            import time
            start_time = time.time()
            today = date.today().strftime('%Y-%m-%d')
            stats_games = self._try_stats_api_fallback(today)
            stats_time = time.time() - start_time

            results['stats_api'] = {
                'success': True,
                'games_count': len(stats_games),
                'response_time': stats_time
            }
            print(f"✅ Stats API: {len(stats_games)} games in {stats_time:.2f}s")

        except Exception as e:
            results['stats_api'] = {
                'success': False,
                'error': str(e),
                'response_time': 0
            }
            print(f"❌ Stats API failed: {e}")

        # Summary
        print("\n📊 PERFORMANCE SUMMARY:")
        print("-" * 40)

        for api_name, result in results.items():
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            print(f"{api_name.upper()}: {status}")
            if result['success']:
                print(f"   Games: {result['games_count']}")
                print(f"   Time: {result['response_time']:.2f}s")
            print()

        return results


def main():
    """Test the enhanced data provider"""
    print("🚀 Enhanced NBA Data Provider Test")
    print("Context7-Compliant Multi-Endpoint Solution")
    print("=" * 60)

    # Initialize enhanced provider
    provider = EnhancedNBADataProvider()

    # Test today's games
    print("\n📅 Testing Today's Games:")
    today_games = provider.get_scheduled_games()

    # Test specific date
    print("\n📅 Testing Oct 25, 2025:")
    future_games = provider.get_scheduled_games(specific_date='2025-10-25')

    # Performance test
    provider.test_api_performance()

    # Summary
    total_games = len(today_games) + len(future_games)
    print(f"\n🎯 SUMMARY:")
    print(f"   Today's games: {len(today_games)}")
    print(f"   Future games: {len(future_games)}")
    print(f"   Total detected: {total_games}")

    if total_games > 0:
        print("🎉 SUCCESS! Enhanced NBA Data Provider is working!")
    else:
        print("⚠️  No games detected - check NBA season schedule")

    return total_games > 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)