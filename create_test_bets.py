#!/usr/bin/env python3
"""
Script per creare scommesse di test per verificare il sistema.
"""

import logging
from pathlib import Path
import duckdb
from datetime import datetime, timedelta
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_bets():
    """
    Crea scommesse di test per verificare il sistema di betting.
    """
    logger.info("🎯 CREAZIONE SCOMMESSE DI TEST")
    logger.info("=" * 50)

    project_root = Path(__file__).parent
    db_path = project_root / "data" / "nba_data.duckdb"

    # Dati di test per scommesse realistiche
    test_bets = [
        {
            "game_id": "TEST001",
            "home_team": "Boston Celtics",
            "away_team": "Philadelphia 76ers",
            "bet_type": "OVER",
            "line": 225.5,
            "odds": 1.85,
            "stake": 50.0,
            "status": "pending"
        },
        {
            "game_id": "TEST002",
            "home_team": "Los Angeles Lakers",
            "away_team": "Phoenix Suns",
            "bet_type": "UNDER",
            "line": 230.0,
            "odds": 1.90,
            "stake": 30.0,
            "status": "pending"
        },
        {
            "game_id": "TEST003",
            "home_team": "Milwaukee Bucks",
            "away_team": "Golden State Warriors",
            "bet_type": "OVER",
            "line": 235.5,
            "odds": 1.82,
            "stake": 40.0,
            "status": "pending"
        }
    ]

    try:
        with duckdb.connect(str(db_path)) as conn:
            logger.info("📝 Inserimento scommesse di test...")

            for i, bet_data in enumerate(test_bets, 1):
                # Calcola potenziale ritorno
                potential_return = bet_data["stake"] * bet_data["odds"]

                # Genera ID univoco
                bet_id = f"bet_TEST_{i:03d}_{uuid.uuid4().hex[:8]}"

                # Inserisci scommessa
                conn.execute("""
                    INSERT INTO placed_bets (
                        bet_id, game_id, bet_type, line, odds, stake, potential_return,
                        edge, probability, quality_score, risk_level, status, placed_at,
                        home_team, away_team, bookmaker, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    bet_id,
                    bet_data["game_id"],
                    bet_data["bet_type"],
                    bet_data["line"],
                    bet_data["odds"],
                    bet_data["stake"],
                    potential_return,
                    0.05,  # edge
                    0.54,  # probability
                    0.75,  # quality_score
                    "medium",  # risk_level
                    bet_data["status"],
                    datetime.now() - timedelta(hours=i),  # placed_at
                    bet_data["home_team"],
                    bet_data["away_team"],
                    "TestBookmaker",
                    "Scommessa di test creata automaticamente"
                ])

                logger.info(f"✅ Scommessa {i} creata: {bet_id}")
                logger.info(f"   {bet_data['home_team']} vs {bet_data['away_team']}")
                logger.info(f"   {bet_data['bet_type']} {bet_data['line']} @ {bet_data['odds']}")
                logger.info(f"   Stake: €{bet_data['stake']} → Potenziale: €{potential_return:.2f}")

            # Verifica inserimento
            total_bets = conn.execute("SELECT COUNT(*) FROM placed_bets").fetchone()[0]
            pending_bets = conn.execute("SELECT COUNT(*) FROM placed_bets WHERE status = 'pending'").fetchone()[0]

            logger.info("=" * 50)
            logger.info("🎉 SCOMMESSE DI TEST CREATE!")
            logger.info(f"✅ Totale scommesse: {total_bets}")
            logger.info(f"✅ Scommesse pending: {pending_bets}")
            logger.info(f"💰 Bankroll attuale: €1000.00")

            # Mostra riepilogo
            bets_summary = conn.execute("""
                SELECT bet_id, home_team, away_team, bet_type, line, odds, stake, status
                FROM placed_bets
                ORDER BY placed_at DESC
            """).fetchall()

            logger.info("📋 Riepilogo scommesse:")
            for bet in bets_summary:
                logger.info(f"   {bet[0]}: {bet[1]} vs {bet[2]} - {bet[3]} {bet[4]} @ {bet[5]} (€{bet[6]}, {bet[7]})")

            return True

    except Exception as e:
        logger.error(f"❌ Errore creazione scommesse test: {e}")
        return False

def main():
    """Esecuzione principale."""
    logger.info("🚀 AVVIO CREAZIONE SCOMMESSE DI TEST")

    if create_test_bets():
        logger.info("🎯 Scommesse di test create con successo!")
        logger.info("Ora puoi verificare nella dashboard che appaiano correttamente.")
        return True
    else:
        logger.error("❌ Creazione scommesse test fallita!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)