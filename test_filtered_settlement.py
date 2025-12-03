#!/usr/bin/env python3
"""
Test del settlement system con il filtro per le scommesse di test
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime
from nba_predictor.utils.betting_database_manager import BettingDatabaseManager
from nba_predictor.utils.robust_bet_settlement import RobustBetSettlement

def test_filtered_settlement():
    print("🧪 Test del settlement system con filtro anti-test bets...")

    # 1. Database manager per ottenere scommesse filtrate
    db_manager = BettingDatabaseManager()

    # 2. Robust settlement system
    settlement_system = RobustBetSettlement(db_manager)

    print("\n1. 📋 Scommesse pending filtrate (senza test bets):")
    pending_bets = db_manager.get_pending_bets()
    print(f"   Trovate {len(pending_bets)} scommesse reali NBA")

    for bet in pending_bets:
        print(f"   🎯 Scommessa: {bet.bet_id}")
        print(f"      Game ID: {bet.game_id}")
        print(f"      Team: {bet.home_team} vs {bet.away_team}")
        print(f"      Bet Type: {bet.bet_type} Line: {bet.line}")

        # Test NBA API per questa partita
        try:
            final_score = settlement_system.nba_api.get_game_boxscore(bet.game_id)
            if final_score:
                home_score, away_score = final_score
                print(f"      ✅ Punteggio disponibile: {away_score}-{home_score}")
                print(f"      🎯 Questa scommessa può essere processata!")
            else:
                print(f"      ❌ Nessun punteggio trovato per {bet.game_id}")
        except Exception as e:
            print(f"      ❌ Errore nel recupero punteggio: {e}")

    print(f"\n2. 🚀 Test complete settlement process:")

    try:
        result = settlement_system.execute_robust_settlement()
        print(f"   ✅ Settlement completato!")
        print(f"   Result: {result}")

        # Analisi dei risultati
        if result.get('success', False):
            settled = result.get('settled_bets', 0)
            total = result.get('total_pending', 0)
            print(f"\n   📊 Report Settlement:")
            print(f"      - Scommesse processate: {total}")
            print(f"      - Scommesse settle: {settled}")
            print(f"      - Success rate: {result.get('success_rate', 0):.1f}%")

            if settled > 0:
                print(f"\n   🎉 Scommesse settlate con successo:")
                for detail in result.get('details', []):
                    if detail.get('result') == 'settled':
                        print(f"      - {detail.get('bet_id')}: {detail.get('message', '')}")
        else:
            print(f"   ❌ Settlement fallito: {result.get('message', 'Unknown error')}")

    except Exception as e:
        print(f"   ❌ Errore nel settlement: {e}")
        import traceback
        traceback.print_exc()

    db_manager.conn.close()
    print("\n🎉 Test completato!")

if __name__ == "__main__":
    test_filtered_settlement()