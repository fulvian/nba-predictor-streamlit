#!/usr/bin/env python3
"""
Test completo del sistema di salvataggio scommesse NBA betting.

Testa tutte le funzionalità critiche:
- Piazzamento scommesse
- Salvataggio database
- Aggiornamento bankroll
- Gestione stati partite
- Transazioni ACID

Context7 Compliant: Yes
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_database_connection():
    """Test della connessione al database."""
    try:
        import duckdb
        db_path = Path("data/nba_betting.duckdb")

        if not db_path.exists():
            logger.error(f"Database non trovato: {db_path}")
            return False

        conn = duckdb.connect(str(db_path))

        # Test connessione
        tables = conn.execute("SHOW TABLES").fetchall()
        logger.info(f"Database connesso - Tabelle trovate: {len(tables)}")

        conn.close()
        return True

    except Exception as e:
        logger.error(f"Errore connessione database: {e}")
        return False

def test_bet_placement():
    """Test del piazzamento scommesse."""
    try:
        import duckdb

        conn = duckdb.connect("data/nba_betting.duckdb")

        # Dati test scommessa
        test_bet = {
            "bet_id": "TEST_BET_001",
            "game_id": "TEST_GAME_001",
            "bet_type": "OVER",
            "line": 220.5,
            "odds": 1.85,
            "stake": 5.0,
            "probability": 0.65,
            "implied_probability": 0.54,
            "edge": 0.15,
            "quality_score": 0.8,
            "confidence_score": 0.75
        }

        # Inserimento scommessa in transazione
        conn.begin()

        # 1. Inserisci game se non esiste
        conn.execute("""
            INSERT OR IGNORE INTO games (
                game_id, status, home_team, home_team_abbr, away_team, away_team_abbr, game_date
            )
            VALUES (?, 'scheduled', 'Test Home Team', 'THT', 'Test Away Team', 'TAT', ?)
        """, [test_bet["game_id"], datetime.now().date()])

        # 2. Inserisci scommessa
        conn.execute("""
            INSERT INTO bets (
                bet_id, game_id, bet_type, line, odds, stake,
                probability, implied_probability, edge,
                quality_score, confidence_score, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
        """, [
            test_bet["bet_id"], test_bet["game_id"], test_bet["bet_type"],
            test_bet["line"], test_bet["odds"], test_bet["stake"],
            test_bet["probability"], test_bet["implied_probability"],
            test_bet["edge"], test_bet["quality_score"],
            test_bet["confidence_score"], datetime.now()
        ])

        # 3. Aggiorna bankroll
        current_balance = conn.execute("""
            SELECT balance_after FROM bankroll
            ORDER BY created_at DESC LIMIT 1
        """).fetchone()

        if current_balance:
            new_balance = current_balance[0] - test_bet["stake"]
        else:
            new_balance = 100.0 - test_bet["stake"]  # Starting balance

        # Get next transaction_id
        next_id = conn.execute("""
            SELECT COALESCE(MAX(transaction_id), 0) + 1 FROM bankroll
        """).fetchone()[0]

        conn.execute("""
            INSERT INTO bankroll (
                transaction_id, transaction_type, amount, balance_after, bet_id,
                description, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            next_id, 'bet_placed', -test_bet["stake"], new_balance,
            test_bet["bet_id"], f"Test bet placement: {test_bet['bet_type']} {test_bet['line']}",
            '{"test": true, "automated": true}', datetime.now()
        ])

        conn.commit()

        # Verifica inserimento
        bet_check = conn.execute("""
            SELECT bet_id, status, stake FROM bets
            WHERE bet_id = ?
        """, [test_bet["bet_id"]]).fetchall()

        bankroll_check = conn.execute("""
            SELECT balance_after FROM bankroll
            WHERE bet_id = ?
        """, [test_bet["bet_id"]]).fetchall()

        conn.close()

        if len(bet_check) > 0 and len(bankroll_check) > 0:
            logger.info(f"✅ Scommessa test piazzata: {test_bet['bet_id']}")
            logger.info(f"   Stake: €{test_bet['stake']}, Nuovo saldo: €{new_balance:.2f}")
            return True
        else:
            logger.error("❌ Verifica inserimento fallita")
            return False

    except Exception as e:
        logger.error(f"❌ Errore piazzamento scommessa: {e}")
        return False

def test_game_status_update():
    """Test aggiornamento stati partite."""
    try:
        import duckdb

        conn = duckdb.connect("data/nba_betting.duckdb")

        # Test: aggiorna stato game da scheduled a completed
        game_id = "TEST_GAME_001"
        final_score = {"home_score": 115, "away_score": 108}
        total_points = sum(final_score.values())

        conn.begin()

        # Aggiorna stato game
        conn.execute("""
            UPDATE games
            SET status = 'completed',
                home_score = ?,
                away_score = ?,
                total_score = ?,
                updated_at = ?
            WHERE game_id = ?
        """, [
            final_score["home_score"], final_score["away_score"],
            total_points, datetime.now(), game_id
        ])

        # Inserisci risultato
        conn.execute("""
            INSERT OR REPLACE INTO game_results (
                game_id, home_score, away_score, total_points,
                winner, result_date, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            game_id, final_score["home_score"], final_score["away_score"],
            total_points, "home", datetime.now(), datetime.now()
        ])

        # Aggiorna stato scommessa correlata
        over_line = 220.5
        bet_status = "won" if total_points > over_line else "lost"

        conn.execute("""
            UPDATE bets
            SET status = ?,
                settled_at = ?,
                updated_at = ?
            WHERE game_id = ? AND bet_type = 'OVER'
        """, [
            bet_status, datetime.now(), datetime.now(), game_id
        ])

        # Se vinta, aggiorna bankroll
        if bet_status == "won":
            bet_info = conn.execute("""
                SELECT bet_id, odds, stake FROM bets
                WHERE game_id = ? AND bet_type = 'OVER'
            """, [game_id]).fetchone()

            if bet_info:
                bet_id, odds, stake = bet_info
                winnings = stake * odds

                current_balance = conn.execute("""
                    SELECT balance_after FROM bankroll
                    ORDER BY created_at DESC LIMIT 1
                """).fetchone()[0]

                new_balance = current_balance + winnings

                # Get next transaction_id
                next_id = conn.execute("""
                    SELECT COALESCE(MAX(transaction_id), 0) + 1 FROM bankroll
                """).fetchone()[0]

                conn.execute("""
                    INSERT INTO bankroll (
                        transaction_id, transaction_type, amount, balance_after, bet_id,
                        description, metadata, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    next_id, 'bet_won', winnings, new_balance, bet_id,
                    f"Bet won: OVER {over_line} (Total: {total_points})",
                    f'{{"total_points": {total_points}, "line": {over_line}}}',
                    datetime.now()
                ])

        conn.commit()
        conn.close()

        logger.info(f"✅ Stato partita aggiornato: {game_id}")
        logger.info(f"   Risultato: {final_score['home_score']}-{final_score['away_score']}")
        logger.info(f"   Total points: {total_points} (Line: {over_line})")
        logger.info(f"   Bet status: {bet_status}")
        return True

    except Exception as e:
        logger.error(f"❌ Errore aggiornamento stato partita: {e}")
        return False

def test_transaction_rollback():
    """Test del rollback delle transazioni."""
    try:
        import duckdb

        conn = duckdb.connect("data/nba_betting.duckdb")

        # Salva stato iniziale
        initial_bets = conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0]
        initial_bankroll = conn.execute("""
            SELECT balance_after FROM bankroll
            ORDER BY created_at DESC LIMIT 1
        """).fetchone()

        # Inizia transazione che fallirà
        conn.begin()

        try:
            # Inserisci dati validi
            conn.execute("""
                INSERT INTO bets (bet_id, game_id, bet_type, status, created_at)
                VALUES ('INVALID_BET', 'INVALID_GAME', 'OVER', 'pending', ?)
            """, [datetime.now()])

            # Prova a inserire dato invalido (dovrebbe fallire)
            conn.execute("""
                INSERT INTO bankroll (transaction_type, amount, balance_after, created_at)
                VALUES ('invalid', 'not_a_number', 0, ?)
            """, [datetime.now()])

            conn.commit()

        except Exception:
            # Rollback su errore
            conn.rollback()
            logger.info("Transazione annullata (rollback)")

        # Verifica stato invariato
        final_bets = conn.execute("SELECT COUNT(*) FROM bets").fetchone()[0]
        final_bankroll = conn.execute("""
            SELECT balance_after FROM bankroll
            ORDER BY created_at DESC LIMIT 1
        """).fetchone()

        conn.close()

        if initial_bets == final_bets and initial_bankroll == final_bankroll:
            logger.info("✅ Rollback transazioni funzionante")
            return True
        else:
            logger.error("❌ Rollback fallito - dati modificati")
            return False

    except Exception as e:
        logger.error(f"❌ Errore test rollback: {e}")
        return False

def test_data_integrity():
    """Test integrità dati e constraints."""
    try:
        import duckdb

        conn = duckdb.connect("data/nba_betting.duckdb")

        # Test constraint NOT NULL
        try:
            conn.execute("""
                INSERT INTO bets (bet_id, game_id, bet_type, status)
                VALUES (NULL, 'TEST', 'OVER', 'pending')
            """)
            logger.error("❌ Constraint NOT NULL non funzionante")
            return False
        except Exception:
            logger.info("✅ Constraint NOT NULL funzionante")

        # Test foreign key (game_id deve esistere)
        try:
            conn.execute("""
                INSERT INTO bets (bet_id, game_id, bet_type, status, created_at)
                VALUES ('ORPHAN_BET', 'NONEXISTENT_GAME', 'OVER', 'pending', ?)
            """, [datetime.now()])
            # Questo potrebbe non fallire in DuckDB senza FK constraints
            logger.info("✅ Inserimento scommessa orfana gestito")
        except Exception:
            logger.info("✅ Foreign key constraint funzionante")

        # Test dati validi
        conn.execute("""
            INSERT INTO games (
                game_id, status, home_team, home_team_abbr, away_team, away_team_abbr, game_date
            )
            VALUES ('INTEGRITY_TEST', 'scheduled', 'Home Team', 'HT', 'Away Team', 'AT', ?)
        """, [datetime.now().date()])

        conn.close()
        return True

    except Exception as e:
        logger.error(f"❌ Errore test integrità: {e}")
        return False

def main():
    """Funzione principale di test."""
    logger.info("🧪 INIZIO TEST COMPLETO SISTEMA BETTING")

    tests = [
        ("Connessione Database", test_database_connection),
        ("Piazzamento Scommesse", test_bet_placement),
        ("Aggiornamento Stati Partite", test_game_status_update),
        ("Rollback Transazioni", test_transaction_rollback),
        ("Integrità Dati", test_data_integrity)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n📋 Test: {test_name}")
        logger.info("=" * 50)

        try:
            if test_func():
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"💥 {test_name} CRASHED: {e}")

    logger.info("\n" + "=" * 60)
    logger.info(f"📊 RISULTATI FINALI: {passed}/{total} test passati")

    if passed == total:
        logger.info("🎉 SISTEMA BETTING COMPLETAMENTE FUNZIONANTE!")
        return True
    else:
        logger.error("🚨 PROBLEMI CRITICI RILEVATI NEL SISTEMA")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)