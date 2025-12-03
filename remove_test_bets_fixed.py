#!/usr/bin/env python3
"""
Script per rimuovere le scommesse di test utilizzando la vista active_bets
"""

import duckdb
from pathlib import Path

def remove_test_bets():
    print("🧹 Rimozione scommesse di test dal database...")

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

        # 2. Identifica scommesse di test da rimuovere
        test_bets = []
        for row in result:
            bet_id, game_id, home_team, away_team = row[0], row[1], row[2], row[3]
            if (game_id.startswith('TEST') or
                game_id.startswith('EXAMPLE') or
                'Test' in home_team or
                'Test' in away_team or
                'Unknown' in home_team or
                'Unknown' in away_team):
                test_bets.append(bet_id)

        if test_bets:
            print(f"\n🎯 Identificate {len(test_bets)} scommesse di test da rimuovere:")
            for bet_id in test_bets:
                print(f"   - {bet_id}")

            # 3. Rimuovi le scommesse di test dalla tabella bets
            placeholders = ','.join(['?' for _ in test_bets])

            delete_result = conn.execute(f"""
                DELETE FROM bets
                WHERE bet_id IN ({placeholders})
            """, test_bets)

            print(f"   ✅ Rimosse {delete_result.rowcount} scommesse di test")

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

        # 5. Controlla se ci sono scommesse reali NBA (game ID di 10 cifre)
        nba_bets = []
        for row in remaining:
            game_id = row[1]
            if len(game_id) == 10 and game_id.isdigit():
                nba_bets.append(row)

        if nba_bets:
            print(f"\n🏀 Trovate {len(nba_bets)} scommesse NBA reali:")
            for row in nba_bets:
                print(f"   - {row[0]}: {row[2]} vs {row[3]} (Game ID: {row[1]})")

            # 6. Test del sistema di settlement sulla scommessa NBA reale
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
                else:
                    print(f"   ❌ Nessun punteggio trovato per {test_game_id}")

            except Exception as e:
                print(f"   ❌ Errore nel test API: {e}")
        else:
            print("\n⚠️ Nessuna scommessa NBA reale trovata")

        conn.commit()
        print("\n🎉 Pulizia completata con successo!")

    except Exception as e:
        print(f"❌ Errore durante la pulizia: {e}")
        import traceback
        traceback.print_exc()

    finally:
        conn.close()

if __name__ == "__main__":
    remove_test_bets()