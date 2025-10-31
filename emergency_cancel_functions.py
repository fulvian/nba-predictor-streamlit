
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
