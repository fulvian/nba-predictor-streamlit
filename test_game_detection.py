#!/usr/bin/env python3
"""
🧪 TEST SISTEMA DETECTION GAMES NBA
Test completo del sistema di detection multi-source
"""

import sys
import traceback
from datetime import datetime

def test_game_detection():
    """Test completo del sistema di detection"""

    print("🏀 NBA GAMES DETECTION TEST")
    print("=" * 50)

    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"📅 Testing per oggi: {today_str}")
    print()

    # TEST 1: Data Provider
    print("📊 TEST 1: Data Provider (con tutti i fallback)")
    try:
        from data_provider import NBADataProvider
        provider = NBADataProvider()

        games = provider.get_scheduled_games(days_ahead=1, specific_date=today_str)

        if games:
            print(f"✅ Data Provider trovato {len(games)} partite:")
            sources = {}
            for game in games:
                source = game.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1

                home_team = game.get('home_team', 'Unknown')
                away_team = game.get('away_team', 'Unknown')
                time = game.get('time', 'TBD')
                print(f"   • {away_team} @ {home_team} ({time}) [Source: {source}]")

            print(f"   Sources: {sources}")
        else:
            print("❌ Data Provider non ha trovato partite")

    except Exception as e:
        print(f"❌ Data Provider ERROR: {e}")
        traceback.print_exc()

    print()

    # TEST 2: Schedule Scraper
    print("🌐 TEST 2: Schedule Scraper standalone")
    try:
        from nba_schedule_scraper import NBAScheduleScraper
        scraper = NBAScheduleScraper()

        games_df = scraper.get_todays_games(today_str)

        if not games_df.empty:
            print(f"✅ Schedule Scraper trovato {len(games_df)} partite:")
            for _, game in games_df.iterrows():
                home = game.get('home_team', 'Unknown')
                away = game.get('away_team', 'Unknown')
                time = game.get('time', 'TBD')
                print(f"   • {away} @ {home} ({time})")
        else:
            print("❌ Schedule Scraper non ha trovato partite")

    except Exception as e:
        print(f"❌ Schedule Scraper ERROR: {e}")
        traceback.print_exc()

    print()

    # TEST 3: NBA API Direct
    print("📡 TEST 3: NBA API Direct")
    try:
        from nba_api.stats.endpoints import scoreboardv2

        headers = {
            'Host': 'stats.nba.com',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'x-nba-stats-origin': 'stats',
            'x-nba-stats-token': 'true',
            'Connection': 'keep-alive',
            'Referer': 'https://stats.nba.com/',
            'Origin': 'https://stats.nba.com'
        }

        scoreboard = scoreboardv2.ScoreboardV2(
            game_date=today_str,
            league_id='00',
            headers=headers
        )

        games = scoreboard.game_header.get_data_frame()

        if not games.empty:
            print(f"✅ NBA API diretta trovato {len(games)} partite:")
            for _, game in games.iterrows():
                print(f"   • Game ID: {game['GAME_ID']}")
        else:
            print("❌ NBA API diretta non ha trovato partite (DataFrame vuoto)")

    except Exception as e:
        print(f"❌ NBA API Direct ERROR: {e}")
        traceback.print_exc()

    print()
    print("🎉 TEST COMPLETATO")

if __name__ == "__main__":
    test_game_detection()