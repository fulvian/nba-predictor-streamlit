#!/usr/bin/env python3
"""
EMERGENCY FIX - Ricostruisce completamente il database per risolvere il problema di cancellazione.
"""

import sys
import logging
from pathlib import Path
import duckdb
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def emergency_database_rebuild():
    """
    Ricostruisce completamente il database da zero per risolvere la corruzione.
    """
    logger.info("🚨 EMERGENCY DATABASE REBUILD")
    logger.info("=" * 50)

    project_root = Path(__file__).parent
    db_path = project_root / "data" / "nba_data.duckdb"
    backup_path = project_root / "data" / "nba_data_emergency_backup.duckdb"

    try:
        # Step 1: Ferma tutte le connessioni e crea backup di emergenza
        logger.info("Step 1: Creazione backup di emergenza...")
        if db_path.exists():
            if backup_path.exists():
                backup_path.unlink()
            shutil.copy2(db_path, backup_path)
            logger.info("✅ Backup di emergenza creato")

        # Step 2: Rimuovi il database corrotto
        logger.info("Step 2: Rimozione database corrotto...")
        if db_path.exists():
            db_path.unlink()
            logger.info("✅ Database corrotto rimosso")

        # Step 3: Crea nuovo database pulito
        logger.info("Step 3: Creazione nuovo database pulito...")
        with duckdb.connect(str(db_path)) as conn:
            logger.info("✅ Nuovo database creato")

            # Step 4: Ricrea le tabelle essenziali
            logger.info("Step 4: Ricreazione tabelle...")

            # Tabella betting_settings
            conn.execute("""
                CREATE TABLE betting_settings (
                    setting_key VARCHAR PRIMARY KEY,
                    setting_value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Inserisci impostazioni di base
            conn.execute("""
                INSERT INTO betting_settings (setting_key, setting_value) VALUES
                ('current_bankroll', '1000.0'),
                ('initial_bankroll', '1000.0'),
                ('max_bet_percentage', '5.0'),
                ('default_bet_amount', '25.0'),
                ('auto_settlement_enabled', 'true'),
                ('risk_tolerance', 'medium'),
                ('min_odds_threshold', '1.5'),
                ('max_odds_threshold', '3.0'),
                ('max_stake_percentage', '5.0'),
                ('min_edge_threshold', '2.0'),
                ('max_daily_bets', '10'),
                ('auto_stake_calculation', 'true')
            """)
            logger.info("✅ betting_settings creata")

            # Tabella placed_bets
            conn.execute("""
                CREATE TABLE placed_bets (
                    bet_id VARCHAR PRIMARY KEY,
                    game_id VARCHAR NOT NULL,
                    bet_type VARCHAR NOT NULL,
                    line FLOAT,
                    odds FLOAT NOT NULL,
                    stake FLOAT NOT NULL,
                    potential_return FLOAT,
                    edge FLOAT,
                    probability FLOAT,
                    quality_score FLOAT,
                    risk_level VARCHAR,
                    status VARCHAR NOT NULL DEFAULT 'pending',
                    placed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    settled_at TIMESTAMP,
                    result_amount FLOAT,
                    profit_loss FLOAT,
                    bookmaker VARCHAR DEFAULT 'Internal',
                    notes TEXT,
                    home_team VARCHAR,
                    away_team VARCHAR,
                    analysis_id VARCHAR
                )
            """)
            logger.info("✅ placed_bets creata")

            # Tabella bankroll_history
            conn.execute("""
                CREATE TABLE bankroll_history (
                    history_id INTEGER PRIMARY KEY,
                    bet_id VARCHAR,
                    transaction_type VARCHAR,
                    amount FLOAT,
                    balance_before FLOAT,
                    balance_after FLOAT,
                    notes TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logger.info("✅ bankroll_history creata")

            # Tabella betting_analysis
            conn.execute("""
                CREATE TABLE betting_analysis (
                    analysis_id VARCHAR PRIMARY KEY,
                    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_prediction TEXT,
                    confidence_score FLOAT,
                    market_efficiency FLOAT,
                    expected_value FLOAT,
                    kelly_fraction FLOAT,
                    recommended_stake FLOAT,
                    risk_assessment TEXT,
                    notes TEXT,
                    game_id VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    bet_type VARCHAR,
                    line FLOAT,
                    odds FLOAT,
                    edge FLOAT,
                    probability FLOAT,
                    quality_score FLOAT,
                    risk_level VARCHAR,
                    recommendation TEXT,
                    confidence_level VARCHAR,
                    model_predictions TEXT,
                    home_team VARCHAR,
                    away_team VARCHAR,
                    implied_probability FLOAT,
                    true_probability FLOAT,
                    edge_score FLOAT,
                    risk_score FLOAT,
                    consistency_score FLOAT,
                    stake FLOAT,
                    roi FLOAT,
                    is_value BOOLEAN,
                    central_line FLOAT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logger.info("✅ betting_analysis creata")

            # Tabella bankroll
            conn.execute("""
                CREATE TABLE bankroll (
                    id INTEGER PRIMARY KEY,
                    balance FLOAT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Inserisci bankroll iniziale
            conn.execute("""
                INSERT INTO bankroll (id, balance) VALUES (1, 1000.0)
            """)
            logger.info("✅ bankroll creata")

            # Altre tabelle di supporto
            conn.execute("CREATE TABLE betting_logs (log_id INTEGER PRIMARY KEY, message TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            conn.execute("CREATE TABLE data_metadata (key VARCHAR PRIMARY KEY, value TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")

            logger.info("✅ Tutte le tabelle ricreate")

            # Step 5: Verifica integrità
            logger.info("Step 5: Verifica integrità database...")
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = [table[0] for table in tables]
            logger.info(f"✅ Tabelle create: {table_names}")

            # Test operazioni di base
            test_result = conn.execute("SELECT COUNT(*) FROM betting_settings").fetchone()[0]
            logger.info(f"✅ betting_settings: {test_result} impostazioni")

            logger.info("🎉 DATABASE REBUILD COMPLETATO CON SUCCESSO!")
            return True

    except Exception as e:
        logger.error(f"❌ Emergency rebuild failed: {e}")
        return False

def create_emergency_cancel_function():
    """
    Crea una funzione di cancellazione di emergenza che bypassa il sistema corrente.
    """
    logger.info("🔧 Creando funzione di cancellazione di emergenza...")

    emergency_code = '''
import duckdb
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def emergency_cancel_bet(bet_id: str) -> dict:
    """
    Funzione di emergenza per cancellare scommesse che bypassa il sistema normale.
    """
    try:
        project_root = Path(__file__).parent
        db_path = project_root / "data" / "nba_data.duckdb"

        with duckdb.connect(str(db_path)) as conn:
            # Ottieni dettagli scommessa
            bet_info = conn.execute("""
                SELECT bet_id, stake, status FROM placed_bets WHERE bet_id = ?
            """, [bet_id]).fetchone()

            if not bet_info:
                return {"success": False, "error": "Scommessa non trovata"}

            if bet_info[2] != 'pending':
                return {"success": False, "error": f"Scommessa già {bet_info[2]}"}

            stake = float(bet_info[1])

            # Transazione di cancellazione
            conn.execute("BEGIN TRANSACTION")
            try:
                # Aggiorna stato scommessa
                conn.execute("""
                    UPDATE placed_bets
                    SET status = 'cancelled',
                        settled_at = CURRENT_TIMESTAMP,
                        result_amount = ?,
                        profit_loss = 0
                    WHERE bet_id = ?
                """, [stake, bet_id])

                # Aggiorna bankroll
                current_bankroll = float(conn.execute(
                    "SELECT setting_value FROM betting_settings WHERE setting_key = 'current_bankroll'"
                ).fetchone()[0])

                new_bankroll = current_bankroll + stake

                conn.execute("""
                    UPDATE betting_settings
                    SET setting_value = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE setting_key = 'current_bankroll'
                """, [str(new_bankroll)])

                # Registra nella history
                next_id = conn.execute(
                    "SELECT COALESCE(MAX(history_id), 0) + 1 FROM bankroll_history"
                ).fetchone()[0]

                conn.execute("""
                    INSERT INTO bankroll_history
                    (history_id, bet_id, transaction_type, amount, balance_before, balance_after, notes)
                    VALUES (?, ?, 'bet_cancelled', ?, ?, ?, ?)
                """, [next_id, bet_id, stake, current_bankroll, new_bankroll, "Emergency cancellation"])

                conn.execute("COMMIT")

                logger.info(f"✅ Scommessa {bet_id} cancellata con successo")
                logger.info(f"   Bankroll: {current_bankroll} → {new_bankroll}")

                return {
                    "success": True,
                    "message": "Scommessa cancellata con successo",
                    "old_bankroll": current_bankroll,
                    "new_bankroll": new_bankroll,
                    "stake_returned": stake
                }

            except Exception as e:
                conn.execute("ROLLBACK")
                logger.error(f"Errore durante cancellazione scommessa {bet_id}: {e}")
                return {"success": False, "error": f"Errore database: {str(e)}"}

    except Exception as e:
        logger.error(f"Errore critico durante cancellazione scommessa {bet_id}: {e}")
        return {"success": False, "error": f"Errore critico: {str(e)}"}

# Funzione per testare tutte le scommesse pending
def test_pending_bets():
    """
    Testa tutte le scommesse pending e restituisce i dettagli.
    """
    try:
        project_root = Path(__file__).parent
        db_path = project_root / "data" / "nba_data.duckdb"

        with duckdb.connect(str(db_path)) as conn:
            pending_bets = conn.execute("""
                SELECT bet_id, stake, status, home_team, away_team, bet_type, line, odds
                FROM placed_bets
                WHERE status = 'pending'
                ORDER BY placed_at DESC
            """).fetchall()

            return {
                "success": True,
                "bets": pending_bets,
                "count": len(pending_bets)
            }

    except Exception as e:
        return {"success": False, "error": str(e)}

logger.info("✅ Funzioni di emergenza caricate")
'''

    # Scrivi il codice di emergenza
    emergency_file = Path(__file__).parent / "emergency_cancel_functions.py"
    with open(emergency_file, 'w') as f:
        f.write(emergency_code)

    logger.info(f"✅ Funzioni di emergenza create: {emergency_file}")
    return True

def main():
    """Main emergency fix process."""
    logger.info("🚨 EMERGENCY FIX - Cancellazione Scommesse")
    logger.info("=" * 60)

    # Step 1: Ricostruisci database
    rebuild_success = emergency_database_rebuild()

    if not rebuild_success:
        logger.error("❌ Database rebuild fallito")
        return False

    # Step 2: Crea funzioni di emergenza
    create_emergency_cancel_function()

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("🎉 EMERGENCY FIX COMPLETATO!")
    logger.info("")
    logger.info("✅ Database completamente ricostruito")
    logger.info("✅ Tabelle essenziali create")
    logger.info("✅ Impostazioni default ripristinate")
    logger.info("✅ Funzioni di emergenza disponibili")
    logger.info("")
    logger.info("📋 Prossimi passi:")
    logger.info("1. Riavvia il betting workflow dashboard")
    logger.info("2. La cancellazione delle scommesse dovrebbe ora funzionare")
    logger.info("3. In caso di problemi, usa le funzioni di emergenza")
    logger.info("")
    logger.info("🔧 File creati:")
    logger.info("- emergency_cancel_functions.py: Funzioni di emergenza")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)