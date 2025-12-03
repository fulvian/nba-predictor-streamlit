#!/usr/bin/env python3
"""
Test semplice per verificare i dati nel database
"""

import duckdb
from pathlib import Path

def test_simple():
    print("🧪 Test Semplice del Database...")

    # Connettersi al database delle scommesse
    db_path = Path(__file__).parent / "data" / "nba_betting.duckdb"
    print(f"1. Connessione a: {db_path}")

    try:
        conn = duckdb.connect(str(db_path), read_only=False)
        print("   ✅ Connesso")
    except Exception as e:
        print(f"   ❌ Errore: {e}")
        return False

    # 2. Vedere tutte le tabelle
    print("2. Tabelle disponibili:")
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        for table in tables:
            print(f"   - {table[0]}")
    except Exception as e:
        print(f"   ❌ Errore: {e}")
        return False

    # 3. Controllare la tabella active_bets
    print("3. Contenuto di active_bets:")
    try:
        result = conn.execute("SELECT * FROM active_bets LIMIT 5").fetchall()
        print(f"   Trovati {len(result)} record:")
        for row in result:
            print(f"   {row}")
    except Exception as e:
        print(f"   ❌ Errore: {e}")
        return False

    # 4. Controllare le scommesse pendenti
    print("4. Scommesse pendenti:")
    try:
        pending = conn.execute("SELECT COUNT(*) FROM active_bets WHERE status = 'pending'").fetchone()
        print(f"   Scommesse pendenti: {pending[0]}")

        if pending[0] > 0:
            details = conn.execute("""
                SELECT bet_id, game_id, home_team, away_team, bet_type, line, placed_at
                FROM active_bets WHERE status = 'pending'
                ORDER BY placed_at DESC LIMIT 3
            """).fetchall()

            print("   Esempi di scommesse pendenti:")
            for bet in details:
                print(f"   - {bet[0]}: {bet[2]} vs {bet[3]} ({bet[4]} {bet[5]}) - {bet[6]}")
    except Exception as e:
        print(f"   ❌ Errore: {e}")
        return False

    conn.close()
    print("🎉 Test completato!")
    return True

if __name__ == "__main__":
    test_simple()