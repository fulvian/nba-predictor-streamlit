#!/usr/bin/env python3
"""
Verifica struttura database per identificare le colonne corrette
"""

import duckdb
from pathlib import Path

def check_db_structure():
    print("🔍 Analisi struttura database...")

    # Connettersi al database
    db_path = Path(__file__).parent / "data" / "nba_betting.duckdb"
    conn = duckdb.connect(str(db_path), read_only=False)

    try:
        # 1. Visualizza tutte le tabelle
        print("\n📋 Tabelle nel database:")
        tables = conn.execute("SHOW TABLES").fetchall()
        for table in tables:
            print(f"   - {table[0]}")

        # 2. Struttura tabella bets
        print("\n🏗️ Struttura tabella bets:")
        columns = conn.execute("DESCRIBE bets").fetchall()
        for col in columns:
            print(f"   - {col[0]}: {col[1]}")

        # 3. Controlla se esiste la vista active_bets
        try:
            print("\n👁️ Struttura vista active_bets:")
            view_columns = conn.execute("DESCRIBE active_bets").fetchall()
            for col in view_columns:
                print(f"   - {col[0]}: {col[1]}")
        except Exception as e:
            print(f"   ❌ Vista active_bets non trovata: {e}")

        # 4. Mostra le scommesse pending con le colonne disponibili
        print("\n📊 Scommesse con status = 'pending':")
        try:
            # Prima prova con la vista active_bets
            result = conn.execute("""
                SELECT * FROM active_bets WHERE status = 'pending' LIMIT 3
            """).fetchall()

            print(f"   Trovate {len(result)} scommesse tramite vista active_bets:")
            for i, row in enumerate(result):
                print(f"   Row {i+1}: {row}")

        except Exception as e:
            print(f"   ❌ Errore con vista active_bets: {e}")

            # Prova direttamente con la tabella bets
            try:
                result = conn.execute("""
                    SELECT * FROM bets WHERE status = 'pending' LIMIT 3
                """).fetchall()

                print(f"   Trovate {len(result)} scommesse tramite tabella bets:")
                for i, row in enumerate(result):
                    print(f"   Row {i+1}: {row}")

            except Exception as e2:
                print(f"   ❌ Errore con tabella bets: {e2}")

        # 5. Controlla colonne specifiche che potrebbero contenere team names
        print("\n🔍 Ricerca colonne con team names:")
        all_columns = [col[0] for col in columns]
        team_related = [col for col in all_columns if 'team' in col.lower() or 'home' in col.lower() or 'away' in col.lower()]
        print(f"   Colonne potenzialmente correlate ai team: {team_related}")

    finally:
        conn.close()

if __name__ == "__main__":
    check_db_structure()