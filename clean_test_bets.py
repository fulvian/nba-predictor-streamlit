#!/usr/bin/env python3
"""
Script per rimuovere le scommesse di test che bloccano il sistema
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import duckdb
from pathlib import Path

def clean_test_bets():
    print("🧹 Pulizia scommesse di test dal database...")

    # Connettersi al database
    db_path = Path(__file__).parent / "data" / "nba_betting.duckdb"
    conn = duckdb.connect(str(db_path), read_only=False)

    try:
        # 1. Visualizza le scommesse attuali
        print("\n📋 Scommesse attuali con status 'pending':")
        result = conn.execute("""
            SELECT bet_id, game_id, home_team, away_team, status, placed_at
            FROM bets
            WHERE status = 'pending'
            ORDER BY placed_at DESC
        """).fetchall()

        print(f"   Trovate {len(result)} scommesse 'pending':")
        for row in result:
            print(f"   - {row[0]}: {row[2]} vs {row[3]} (Game ID: {row[1]})")

        # 2. Identifica scommesse di test (game_id che iniziano con 'TEST')
        test_bets = conn.execute("""
            SELECT bet_id, game_id, home_team, away_team
            FROM bets
            WHERE status = 'pending'
            AND (game_id LIKE 'TEST%' OR home_team LIKE '%Test%' OR away_team LIKE '%Test%')
        """).fetchall()

        if test_bets:
            print(f"\n🎯 Identificate {len(test_bets)} scommesse di test da rimuovere:")
            for row in test_bets:
                print(f"   - {row[0]}: {row[2]} vs {row[3]} (Game ID: {row[1]})")

            # 3. Rimuovi le scommesse di test
            test_bet_ids = [row[0] for row in test_bets]
            placeholders = ','.join(['?' for _ in test_bet_ids])

            delete_result = conn.execute(f"""
                DELETE FROM bets
                WHERE bet_id IN ({placeholders})
            """, test_bet_ids)

            print(f"   ✅ Rimosse {delete_result.rowcount} scommesse di test")

        else:
            print("\n✅ Nessuna scommessa di test trovata")

        # 4. Visualizza le scommesse rimanenti
        print("\n📊 Scommesse rimanenti con status 'pending':")
        remaining = conn.execute("""
            SELECT bet_id, game_id, home_team, away_team, status, placed_at
            FROM bets
            WHERE status = 'pending'
            ORDER BY placed_at DESC
        """).fetchall()

        print(f"   Rimangono {len(remaining)} scommesse 'pending':")
        for row in remaining:
            print(f"   - {row[0]}: {row[2]} vs {row[3]} (Game ID: {row[1]})")

        # 5. Controlla se ci sono scommesse reali NBA
        nba_bets = conn.execute("""
            SELECT bet_id, game_id, home_team, away_team
            FROM bets
            WHERE status = 'pending'
            AND game_id REGEXP '^[0-9]{10}$'  -- NBA game IDs have 10 digits
        """).fetchall()

        if nba_bets:
            print(f"\n🏀 Trovate {len(nba_bets)} scommesse NBA reali:")
            for row in nba_bets:
                print(f"   - {row[0]}: {row[2]} vs {row[3]} (Game ID: {row[1]})")
        else:
            print("\n⚠️ Nessuna scommessa NBA reale trovata tra le pending")

        conn.commit()
        print("\n🎉 Pulizia completata con successo!")

    except Exception as e:
        print(f"❌ Errore durante la pulizia: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()

    finally:
        conn.close()

if __name__ == "__main__":
    clean_test_bets()