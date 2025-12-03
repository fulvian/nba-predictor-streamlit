#!/usr/bin/env python3
"""
Test NBA API per recuperare punteggi finali
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from nba_predictor.utils.robust_bet_settlement import NBABoxscoreAPI

def test_nba_api():
    print("🏀 Test NBA API per Final Scores...")

    api = NBABoxscoreAPI()

    # Game ID reali trovati nel database
    test_games = ['0042400407', '0042400406']

    for game_id in test_games:
        print(f"\n📋 Testando Game ID: {game_id}")

        try:
            # Prova a ottenere il punteggio finale
            final_score = api.get_game_boxscore(game_id)

            if final_score:
                home_score, away_score = final_score
                print(f"   ✅ SUCCESS: {away_score}-{home_score} (Away-Home)")
            else:
                print(f"   ❌ No final score found")

        except Exception as e:
            print(f"   ❌ ERROR: {e}")

    print(f"\n🎯 Test completato!")

if __name__ == "__main__":
    test_nba_api()