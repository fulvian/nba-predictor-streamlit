#!/usr/bin/env python3
"""
Analizza i vincoli del database per identificare il problema
"""

import duckdb
from pathlib import Path

def check_constraints():
    print("🔍 Analisi vincoli del database...")

    # Connettersi al database
    db_path = Path(__file__).parent / "data" / "nba_betting.duckdb"
    conn = duckdb.connect(str(db_path), read_only=False)

    try:
        # 1. Visualizza tutti i constraints
        print("\n📋 Vincoli del database:")
        try:
            constraints = conn.execute("SELECT * FROM information_schema.table_constraints").fetchall()
            print(f"   Trovati {len(constraints)} constraints:")
            for constraint in constraints:
                print(f"   - {constraint}")
        except Exception as e:
            print(f"   Errore: {e}")

        # 2. Visualizza chiavi esterne specifiche
        print("\n🔗 Chiavi esterne:")
        try:
            foreign_keys = conn.execute("""
                SELECT
                    tc.table_name,
                    tc.constraint_name,
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
            """).fetchall()

            print(f"   Trovate {len(foreign_keys)} foreign keys:")
            for fk in foreign_keys:
                print(f"   - {fk[0]}.{fk[2]} -> {fk[3]}.{fk[4]} (constraint: {fk[1]})")

        except Exception as e:
            print(f"   Errore: {e}")

        # 3. Controlla riferimenti alla tabella bets
        print("\n🎯 Riferimenti alla tabella bets:")
        try:
            references = conn.execute("""
                SELECT
                    tc.table_name,
                    kcu.column_name,
                    tc.constraint_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                AND ccu.table_name = 'bets'
            """).fetchall()

            print(f"   Trovate {len(references)} tabelle che referenziano 'bets':")
            for ref in references:
                print(f"   - {ref[0]}.{ref[1]} (constraint: {ref[2]})")

        except Exception as e:
            print(f"   Errore: {e}")

        # 4. Controlla se la scommessa problematica esiste nelle tabelle correlate
        problem_bet_id = "CUSTOM_Thunder_Pacers_OVER_202.0"
        print(f"\n🔍 Ricerca della scommessa problematica: {problem_bet_id}")

        tables = ['bets', 'betting_analysis', 'betting_performance', 'game_results']
        for table in tables:
            try:
                result = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE bet_id = ?", [problem_bet_id]).fetchone()
                if result[0] > 0:
                    print(f"   ✅ Trovata in {table}: {result[0]} records")

                    # Mostra i dettagli
                    details = conn.execute(f"SELECT * FROM {table} WHERE bet_id = ? LIMIT 3", [problem_bet_id]).fetchall()
                    for detail in details:
                        print(f"      {detail}")
                else:
                    print(f"   ❌ Non trovata in {table}")
            except Exception as e:
                print(f"   ❌ Errore cercando in {table}: {e}")

        # 5. Prova a disabilitare i constraints temporaneamente
        print(f"\n🔧 Tentativo di disabilitare constraints:")
        try:
            conn.execute("PRAGMA disable_foreign_keys")
            print("   ✅ Foreign keys disabilitati")

            # Prova l'update
            test_bets = ["TEST_BET_001"]
            placeholders = ','.join(['?' for _ in test_bets])

            update_result = conn.execute(f"""
                UPDATE bets
                SET status = 'cancelled', updated_at = CURRENT_TIMESTAMP
                WHERE bet_id IN ({placeholders})
            """, test_bets)

            print(f"   ✅ Update test riuscito: {update_result.rowcount} righe modificate")
            conn.rollback()  # Annulla le modifiche di test

        except Exception as e:
            print(f"   ❌ Errore: {e}")
            conn.rollback()

    finally:
        conn.close()

if __name__ == "__main__":
    check_constraints()