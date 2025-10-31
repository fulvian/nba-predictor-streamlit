#!/usr/bin/env python3
"""
Database Reset Utility - DuckDB Corruption Fix

Utility per ripristinare il database DuckDB e risolvere i problemi di corruption.
Crea un nuovo database pulito con schema corretto.
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path

import duckdb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_duckdb_database():
    """
    Reset del database DuckDB per risolvere problemi di corruption.

    Process:
    1. Backup del database corrente
    2. Creazione nuovo database pulito
    3. Setup schema betting tables
    4. Verifica integrità
    """

    # Paths
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    db_path = data_dir / "nba_data.duckdb"

    # Timestamp per backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = data_dir / f"nba_data_backup_reset_{timestamp}.duckdb"

    logger.info("🔄 Iniziando reset database DuckDB...")

    try:
        # 1. Backup database esistente
        if db_path.exists():
            logger.info(f"📦 Backup database esistente: {backup_path}")
            shutil.copy2(db_path, backup_path)
        else:
            logger.info("📝 Database esistente non trovato, creazione da zero")

        # 2. Rimuovi database corrotto
        if db_path.exists():
            db_path.unlink()
            logger.info("🗑️ Database corrotto rimosso")

        # 3. Crea nuovo database pulito
        logger.info("🆕 Creazione nuovo database pulito...")
        conn = duckdb.connect(str(db_path))

        # 4. Setup schema base
        logger.info("📋 Creazione schema base...")

        # Create betting_analysis table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS betting_analysis (
                analysis_id VARCHAR PRIMARY KEY,
                game_id VARCHAR NOT NULL,
                bet_type VARCHAR NOT NULL,
                line DOUBLE,
                odds DOUBLE NOT NULL,
                edge DOUBLE,
                probability DOUBLE,
                implied_probability DOUBLE,
                true_probability DOUBLE,
                quality_score DOUBLE,
                edge_score DOUBLE,
                confidence_score DOUBLE,
                risk_score DOUBLE,
                consistency_score DOUBLE,
                kelly_fraction DOUBLE,
                stake DOUBLE,
                roi DOUBLE,
                is_value BOOLEAN,
                risk_level VARCHAR,
                central_line DOUBLE,
                timestamp TIMESTAMP,
                home_team VARCHAR,
                away_team VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create placed_bets table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS placed_bets (
                bet_id VARCHAR PRIMARY KEY,
                analysis_id VARCHAR,
                game_id VARCHAR NOT NULL,
                bet_type VARCHAR NOT NULL,
                line DOUBLE,
                odds DOUBLE NOT NULL,
                stake DOUBLE NOT NULL,
                quality_score DOUBLE,
                risk_level VARCHAR,
                status VARCHAR DEFAULT 'pending',
                placed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                settled_at TIMESTAMP,
                result_amount DOUBLE,
                profit_loss DOUBLE,
                bookmaker VARCHAR DEFAULT 'Internal',
                notes TEXT,
                home_team VARCHAR,
                away_team VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create bankroll_history table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bankroll_history (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                change_type VARCHAR NOT NULL,
                amount DOUBLE NOT NULL,
                balance_after DOUBLE,
                description TEXT,
                bet_id VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 5. Setup initial bankroll
        logger.info("💰 Setup bankroll iniziale...")
        conn.execute("""
            INSERT INTO bankroll_history (id, change_type, amount, balance_after, description)
            VALUES (1, 'initial_deposit', 1000.0, 1000.0, 'Bankroll iniziale dopo reset database')
        """)

        # 6. Verifica integrità
        logger.info("✅ Verifica integrità database...")

        # Test query
        tables = conn.execute("SHOW TABLES").fetchall()
        logger.info(f"📊 Tabelle create: {[t[0] for t in tables]}")

        # Test inserimento
        conn.execute("""
            INSERT INTO betting_analysis (
                analysis_id, game_id, bet_type, odds, edge, probability,
                quality_score, risk_level, timestamp
            ) VALUES (
                'test_analysis_001', 'test_game', 'moneyline', 2.0, 0.1, 0.5,
                0.8, 'low', CURRENT_TIMESTAMP
            )
        """)

        # Verifica inserimento
        count = conn.execute("SELECT COUNT(*) FROM betting_analysis").fetchone()[0]
        logger.info(f"🧪 Test inserimento: {count} record in betting_analysis")

        # Cleanup test data
        conn.execute("DELETE FROM betting_analysis WHERE analysis_id = 'test_analysis_001'")

        conn.close()

        logger.info("🎉 Database reset completato con successo!")
        logger.info(f"📍 Database location: {db_path}")
        logger.info(f"💾 Backup location: {backup_path}")

        return True

    except Exception as e:
        logger.error(f"❌ Errore durante reset database: {e}")
        return False

if __name__ == "__main__":
    success = reset_duckdb_database()
    if success:
        print("✅ Database reset completato!")
        print("🚀 Puoi riavviare la dashboard ora")
    else:
        print("❌ Reset fallito - controlla i log")