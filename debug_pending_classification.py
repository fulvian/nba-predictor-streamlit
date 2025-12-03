#!/usr/bin/env python3
"""
Debug script per identificare perché le partite concluse sono classificate come "pending"
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime, timedelta
from nba_predictor.utils.betting_database_manager import BettingDatabaseManager
from nba_predictor.utils.robust_bet_settlement import RobustBetSettlement

def debug_pending_games():
    print("🔍 DEBUG: Analisi delle partite classificate come 'pending'")

    # 1. Database manager per ottenere scommesse
    db_manager = BettingDatabaseManager()

    # 2. Robust settlement system
    settlement_system = RobustBetSettlement(db_manager)

    print("\n1. 📋 Scommesse con status = 'pending' nel database:")
    pending_bets = db_manager.get_pending_bets()
    print(f"   Trovate {len(pending_bets)} scommesse con status 'pending'")

    for bet in pending_bets[:3]:  # Prime 3 per analisi
        print(f"\n   🎯 Scommessa: {bet.bet_id}")
        print(f"      Game ID: {bet.game_id}")
        print(f"      Team: {bet.home_team} vs {bet.away_team}")
        print(f"      Data: {bet.game_date}")
        print(f"      Status DB: {bet.status}")
        print(f"      Bet Type: {bet.bet_type}")
        print(f"      Line: {bet.line}")

        # 3. Controlla se la partita è considerata "completata" dal settlement system
        print(f"\n   🔍 Analisi completion per game_id {bet.game_id}:")

        # Prova a ottenere punteggio finale
        try:
            final_score = settlement_system.nba_api.get_game_boxscore(bet.game_id)
            if final_score:
                home_score, away_score = final_score
                print(f"      ✅ Punteggio disponibile: {away_score}-{home_score}")
                print(f"      ❌ MA LA SCOMMESSA È ANCORA 'pending'!")

                # Controlla se ci sono altre scommesse per questa partita
                other_bets = db_manager.get_bets_by_game_id(bet.game_id)
                pending_count = sum(1 for b in other_bets if b.status == 'pending')
                settled_count = sum(1 for b in other_bets if b.status == 'settled')

                print(f"      📊 Stats partita {bet.game_id}:")
                print(f"         Total bets: {len(other_bets)}")
                print(f"         Pending: {pending_count}")
                print(f"         Settled: {settled_count}")

            else:
                print(f"      ❌ Nessun punteggio trovato per {bet.game_id}")
                print(f"      ⚠️  Questo potrebbe essere il problema!")

        except Exception as e:
            print(f"      ❌ Errore nel recupero punteggio: {e}")

    print(f"\n4. 🎯 Test completo settlement system:")

    # Esegui analisi pending bets
    try:
        game_matches = settlement_system.analyze_pending_bets()
        print(f"   Game matches trovati: {len(game_matches)}")

        for match in game_matches:
            print(f"\n   🏀 Match: {match.bet_id}")
            print(f"      Game ID: {match.game_id}")
            print(f"      Team: {match.home_team} vs {match.away_team}")
            print(f"      Mapped: {match.home_team_mapped} vs {match.away_team_mapped}")
            print(f"      Data: {match.game_date}")
            print(f"      Bet Type: {match.bet_type}")
            print(f"      Line: {match.line}")

            # Prova a ottenere punteggio
            final_score = settlement_system.get_final_scores([match])
            if final_score:
                print(f"      ✅ Punteggio recuperato: {final_score[0].final_score}")
            else:
                print(f"      ❌ Nessun punteggio recuperato!")

    except Exception as e:
        print(f"   ❌ Errore nell'analisi: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n5. 🎯 Esecuzione complete settlement process:")
    try:
        result = settlement_system.execute_robust_settlement()
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   ❌ Errore nel settlement: {e}")
        import traceback
        traceback.print_exc()

    db_manager.conn.close()
    print("\n🎉 DEBUG completato!")

if __name__ == "__main__":
    debug_pending_games()