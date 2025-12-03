#!/usr/bin/env python3
"""
Script per disabilitare le scommesse di test aggiornando lo stato invece di cancellarle
"""

import duckdb
from pathlib import Path

def disable_test_bets():
    print("🔧 Disabilitazione scommesse di test dal database...")

    # Connettersi al database
    db_path = Path(__file__).parent / "data" / "nba_betting.duckdb"
    conn = duckdb.connect(str(db_path), read_only=False)

    try:
        # 1. Visualizza le scommesse attuali tramite vista active_bets
        print("\n📋 Scommesse attuali con status 'pending' (vista active_bets):")
        result = conn.execute("""
            SELECT bet_id, game_id, home_team, away_team, status, game_date, placed_at
            FROM active_bets
            WHERE status = 'pending'
            ORDER BY placed_at DESC
        """).fetchall()

        print(f"   Trovate {len(result)} scommesse 'pending':")
        for row in result:
            print(f"   - {row[0]}: {row[2]} vs {row[3]} (Game ID: {row[1]}, Data: {row[5]})")

        # 2. Identifica scommesse di test da disabilitare
        test_bets = []
        nba_bets = []

        for row in result:
            bet_id, game_id, home_team, away_team = row[0], row[1], row[2], row[3]
            if (game_id.startswith('TEST') or
                game_id.startswith('EXAMPLE') or
                game_id.startswith('CUSTOM') or
                'Test' in home_team or
                'Test' in away_team or
                'Unknown' in home_team or
                'Unknown' in away_team):
                test_bets.append(bet_id)
            elif len(game_id) == 10 and game_id.isdigit():
                nba_bets.append((bet_id, game_id, home_team, away_team))

        if test_bets:
            print(f"\n🎯 Identificate {len(test_bets)} scommesse di test da disabilitare:")
            for bet_id in test_bets:
                print(f"   - {bet_id}")

            # 3. Aggiorna lo stato delle scommesse di test a 'cancelled'
            placeholders = ','.join(['?' for _ in test_bets])

            update_result = conn.execute(f"""
                UPDATE bets
                SET status = 'cancelled', updated_at = CURRENT_TIMESTAMP
                WHERE bet_id IN ({placeholders})
            """, test_bets)

            print(f"   ✅ Disabilitate {update_result.rowcount} scommesse di test (status -> 'cancelled')")

        else:
            print("\n✅ Nessuna scommessa di test trovata")

        # 4. Visualizza le scommesse rimanenti
        print("\n📊 Scommesse rimanenti con status 'pending':")
        remaining = conn.execute("""
            SELECT bet_id, game_id, home_team, away_team, status, game_date, placed_at
            FROM active_bets
            WHERE status = 'pending'
            ORDER BY placed_at DESC
        """).fetchall()

        print(f"   Rimangono {len(remaining)} scommesse 'pending':")
        for row in remaining:
            print(f"   - {row[0]}: {row[2]} vs {row[3]} (Game ID: {row[1]}, Data: {row[5]})")

        # 5. Mostra le scommesse NBA reali
        if nba_bets:
            print(f"\n🏀 Scommesse NBA reali identificate:")
            for bet_id, game_id, home_team, away_team in nba_bets:
                print(f"   - {bet_id}: {home_team} vs {away_team} (Game ID: {game_id})")

            # 6. Test del sistema di settlement sulla prima scommessa NBA reale
            print(f"\n🧪 Test recupero punteggio NBA per game_id {nba_bets[0][1]}:")
            test_game_id = nba_bets[0][1]

            try:
                from nba_predictor.utils.robust_bet_settlement import NBABoxscoreAPI
                api = NBABoxscoreAPI()
                final_score = api.get_game_boxscore(test_game_id)

                if final_score:
                    home_score, away_score = final_score
                    print(f"   ✅ Punteggio trovato: {away_score}-{home_score}")
                    print(f"   🎯 La scommessa NBA può essere processata!")

                    # 7. Test completo del settlement system
                    print(f"\n🚀 Test complete settlement system:")
                    from nba_predictor.utils.betting_database_manager import BettingDatabaseManager
                    from nba_predictor.utils.robust_bet_settlement import RobustBetSettlement

                    db_manager = BettingDatabaseManager()
                    settlement_system = RobustBetSettlement(db_manager)

                    result = settlement_system.execute_robust_settlement()
                    print(f"   Result: {result}")

                else:
                    print(f"   ❌ Nessun punteggio trovato per {test_game_id}")

            except Exception as e:
                print(f"   ❌ Errore nel test API: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n⚠️ Nessuna scommessa NBA reale trovata")

        conn.commit()
        print("\n🎉 Disabilitazione completata con successo!")

    except Exception as e:
        print(f"❌ Errore durante la disabilitazione: {e}")
        import traceback
        traceback.print_exc()

    finally:
        conn.close()

if __name__ == "__main__":
    disable_test_bets()