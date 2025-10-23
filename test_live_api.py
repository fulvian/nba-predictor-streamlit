#!/usr/bin/env python3
"""
Test the NBA Live Data API endpoint discovered via Context7
This uses the alternative CDN endpoint that might work when stats.nba.com is down
"""

import sys
import os
from datetime import datetime, timezone
from dateutil import parser

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_nba_live_api():
    """Test NBA Live Data API endpoint"""
    print("🏀 NBA Live Data API Test - Context7 Discovery")
    print("=" * 60)

    try:
        # Import the Live Data endpoint discovered via Context7
        from nba_api.live.nba.endpoints import scoreboard
        print("✅ Successfully imported nba_api.live.nba.endpoints.scoreboard")

        print("\n🔄 Testing NBA Live Data API...")
        print("📡 Endpoint: https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json")

        # Create ScoreBoard instance
        board = scoreboard.ScoreBoard()
        print("✅ ScoreBoard object created successfully")

        # Get scoreboard date
        print(f"📅 ScoreBoard Date: {board.score_board_date}")

        # Get games as dictionary
        games_dict = board.games.get_dict()
        print(f"🎯 Games retrieved: {len(games_dict)}")

        if games_dict:
            print("\n🏀 TODAY'S NBA GAMES (Live Data API):")
            print("-" * 60)

            for i, game in enumerate(games_dict, 1):
                try:
                    # Parse game time
                    gameTimeLTZ = parser.parse(game["gameTimeUTC"]).replace(tzinfo=timezone.utc).astimezone(tz=None)

                    # Extract game info
                    awayTeam = game.get('awayTeam', {}).get('teamName', 'Unknown')
                    homeTeam = game.get('homeTeam', {}).get('teamName', 'Unknown')
                    gameId = game.get('gameId', 'N/A')
                    gameStatus = game.get('gameStatusText', 'Unknown')

                    print(f"   {i}. {awayTeam} @ {homeTeam}")
                    print(f"      🆔 Game ID: {gameId}")
                    print(f"      ⏰ Time: {gameTimeLTZ.strftime('%H:%M %Z')}")
                    print(f"      📊 Status: {gameStatus}")

                    # Add scores if available
                    awayScore = game.get('awayTeam', {}).get('score', 0)
                    homeScore = game.get('homeTeam', {}).get('score', 0)
                    if awayScore and homeScore:
                        print(f"      🏆 Score: {awayScore} - {homeScore}")

                    print()

                except Exception as e:
                    print(f"   ❌ Error parsing game {i}: {e}")

            return True

        else:
            print("❌ No games found in Live Data API")
            return False

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 This means the nba_api library might not be properly installed")
        return False

    except Exception as e:
        print(f"❌ API Error: {e}")
        print("💡 This might be a connectivity issue or the API might be temporarily down")
        return False

def test_live_vs_stats_comparison():
    """Compare Live API vs Stats API performance"""
    print("\n" + "=" * 60)
    print("🔄 COMPARISON: Live API vs Stats API")
    print("=" * 60)

    results = {}

    # Test Live API
    print("📡 Testing Live Data API...")
    try:
        from nba_api.live.nba.endpoints import scoreboard
        import time

        start_time = time.time()
        board = scoreboard.ScoreBoard()
        games = board.games.get_dict()
        live_time = time.time() - start_time

        results['live_api'] = {
            'success': True,
            'games_count': len(games),
            'response_time': live_time,
            'endpoint': 'cdn.nba.com/static/json/liveData/scoreboard'
        }

        print(f"✅ Live API: {len(games)} games in {live_time:.2f}s")

    except Exception as e:
        results['live_api'] = {
            'success': False,
            'error': str(e),
            'endpoint': 'cdn.nba.com/static/json/liveData/scoreboard'
        }
        print(f"❌ Live API failed: {e}")

    # Test Stats API (our current failing one)
    print("\n📊 Testing Stats API...")
    try:
        from data_provider import NBADataProvider
        import time

        dp = NBADataProvider()
        today = datetime.now().strftime('%Y-%m-%d')

        start_time = time.time()
        games = dp.get_scheduled_games(specific_date=today)
        stats_time = time.time() - start_time

        results['stats_api'] = {
            'success': True,
            'games_count': len(games),
            'response_time': stats_time,
            'endpoint': 'stats.nba.com/stats/scoreboardv2'
        }

        print(f"✅ Stats API: {len(games)} games in {stats_time:.2f}s")

    except Exception as e:
        results['stats_api'] = {
            'success': False,
            'error': str(e),
            'endpoint': 'stats.nba.com/stats/scoreboardv2'
        }
        print(f"❌ Stats API failed: {e}")

    # Summary
    print("\n📊 COMPARISON SUMMARY:")
    print("-" * 40)

    for api_name, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"{api_name.upper()}: {status}")
        if result['success']:
            print(f"   Games: {result['games_count']}")
            print(f"   Time: {result['response_time']:.2f}s")
            print(f"   Endpoint: {result['endpoint']}")
        else:
            print(f"   Error: {result['error']}")
        print()

    # Recommendation
    if results.get('live_api', {}).get('success') and not results.get('stats_api', {}).get('success'):
        print("🎯 RECOMMENDATION: Use Live Data API as primary endpoint!")
        return True
    elif results.get('stats_api', {}).get('success') and not results.get('live_api', {}).get('success'):
        print("🎯 RECOMMENDATION: Keep Stats API as primary endpoint")
        return True
    elif results.get('live_api', {}).get('success') and results.get('stats_api', {}).get('success'):
        print("🎯 RECOMMENDATION: Use both APIs with Live Data as primary")
        return True
    else:
        print("⚠️  RECOMMENDATION: Both APIs failing - check connectivity")
        return False

def main():
    """Run all tests"""
    print("🚀 NBA API Discovery Test - Context7 Research Results")
    print("=" * 70)

    # Test Live API
    live_success = test_nba_live_api()

    # Compare APIs
    comparison_success = test_live_vs_stats_comparison()

    print("\n" + "=" * 70)
    print("🎯 FINAL RESULTS:")
    print(f"   Live API Test: {'✅ PASSED' if live_success else '❌ FAILED'}")
    print(f"   Comparison Test: {'✅ USEFUL' if comparison_success else '❌ INCONCLUSIVE'}")

    if live_success:
        print("\n🎉 SUCCESS! We found a working NBA API alternative!")
        print("💡 Next step: Implement Live Data API as primary endpoint")
    else:
        print("\n⚠️  Live API also has issues - may need different approach")

    return live_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)