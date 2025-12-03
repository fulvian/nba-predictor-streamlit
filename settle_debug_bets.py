#!/usr/bin/env python3
"""
Script specifico per aggiornare le 3 scommesse debug pending del 30 ottobre.
Assegna team realistici e determina i risultati.
"""

import sys
import logging
from pathlib import Path
import duckdb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def settle_debug_bets():
    """
    Aggiorna le 3 scommesse debug pending con team e risultati realistici.
    """
    logger.info("🎯 AGGIORNAMENTO SCOMMESSE DEBUG PENDING")

    project_root = Path(__file__).parent
    db_path = project_root / "data" / "nba_data.duckdb"

    # Dati realistici per partite NBA del 30 ottobre
    game_results = {
        "Phoenix Suns vs Los Angeles Lakers": {
            "home_score": 125, "away_score": 119, "total_points": 244
        },
        "Boston Celtics vs Philadelphia 76ers": {
            "home_score": 112, "away_score": 105, "total_points": 217
        },
        "Miami Heat vs San Antonio Spurs": {
            "home_score": 109, "away_score": 102, "total_points": 211
        }
    }

    try:
        with duckdb.connect(str(db_path)) as conn:
            # Trova le 3 scommesse debug pending
            debug_bets = conn.execute("""
                SELECT bet_id, stake, odds, potential_return, bet_type, line
                FROM placed_bets
                WHERE status = 'pending'
                AND bet_id LIKE 'bet_DEBUG%'
                ORDER BY placed_at
            """).fetchall()

            logger.info(f"Trovate {len(debug_bets)} scommesse debug pending")

            if len(debug_bets) != 3:
                logger.error(f"Atteso 3 scommesse debug, trovate {len(debug_bets)}")
                return False

            # Mappa delle partite per le scommesse debug
            games = list(game_results.keys())

            updated_count = 0
            total_profit_loss = 0.0

            for i, bet in enumerate(debug_bets):
                bet_id, stake, odds, potential_return, bet_type, line = bet

                # Assegna partita specifica
                game_matchup = games[i]
                home_team, away_team = game_matchup.split(" vs ")
                result_data = game_results[game_matchup]

                logger.info(f"Elaborazione {bet_id}: {home_team} vs {away_team}")

                # Determina risultato basato sul totale punti
                total_points = result_data['total_points']
                line_value = float(line) if line else 225.5

                if bet_type.upper() == 'OVER':
                    result = 'won' if total_points > line_value else 'lost'
                elif bet_type.upper() == 'UNDER':
                    result = 'won' if total_points < line_value else 'lost'
                else:
                    result = 'lost'

                # Calcola importi
                stake_float = float(stake)
                potential_return_float = float(potential_return)

                if result == 'won':
                    result_amount = potential_return_float
                    profit_loss = result_amount - stake_float
                elif result == 'lost':
                    result_amount = 0.0
                    profit_loss = -stake_float
                else:  # void
                    result_amount = stake_float
                    profit_loss = 0.0

                # Esegui aggiornamento
                try:
                    conn.execute("BEGIN TRANSACTION")

                    # Aggiorna scommessa
                    conn.execute("""
                        UPDATE placed_bets
                        SET status = ?, settled_at = CURRENT_TIMESTAMP,
                            result_amount = ?, profit_loss = ?,
                            home_team = ?, away_team = ?
                        WHERE bet_id = ?
                    """, [result, result_amount, profit_loss, home_team, away_team, bet_id])

                    # Aggiorna bankroll
                    current_bankroll_result = conn.execute("""
                        SELECT setting_value FROM betting_settings
                        WHERE setting_key = 'current_bankroll'
                    """).fetchone()

                    if current_bankroll_result:
                        current_bankroll = float(current_bankroll_result[0])
                        new_bankroll = current_bankroll + result_amount

                        conn.execute("""
                            UPDATE betting_settings
                            SET setting_value = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE setting_key = 'current_bankroll'
                        """, [str(new_bankroll)])

                        # Registra history
                        next_id_result = conn.execute(
                            "SELECT COALESCE(MAX(history_id), 0) + 1 FROM bankroll_history"
                        ).fetchone()
                        next_id = next_id_result[0] if next_id_result else 1

                        conn.execute("""
                            INSERT INTO bankroll_history
                            (history_id, bet_id, transaction_type, amount, balance_before, balance_after, notes)
                            VALUES (?, ?, 'bet_settled', ?, ?, ?, ?)
                        """, [next_id, bet_id, result_amount, current_bankroll, new_bankroll,
                              f"Debug bet settled: {result} ({game_matchup})"])

                        conn.execute("COMMIT")

                        updated_count += 1
                        total_profit_loss += profit_loss

                        logger.info(f"✅ {bet_id}: {result} - P&L: {profit_loss:+.2f}€")
                        logger.info(f"   Partita: {home_team} vs {away_team} (Totale: {total_points}, Line: {line_value})")

                    else:
                        conn.execute("ROLLBACK")
                        logger.error(f"Bankroll non trovata per {bet_id}")

                except Exception as e:
                    conn.execute("ROLLBACK")
                    logger.error(f"Errore aggiornamento {bet_id}: {e}")

            logger.info("=" * 50)
            logger.info(f"🎉 AGGIORNAMENTO COMPLETATO!")
            logger.info(f"✅ Scommesse aggiornate: {updated_count}")
            logger.info(f"💰 Profit/Loss totale: {total_profit_loss:+.2f}€")

            # Verifica bankroll finale
            final_bankroll = conn.execute("""
                SELECT setting_value FROM betting_settings
                WHERE setting_key = 'current_bankroll'
            """).fetchone()[0]

            logger.info(f"💳 Bankroll finale: {final_bankroll}€")

            return updated_count == 3

    except Exception as e:
        logger.error(f"❌ Errore aggiornamento scommesse debug: {e}")
        return False

def verify_no_pending_debug():
    """Verifica che non ci siano più scommesse debug pending."""
    project_root = Path(__file__).parent
    db_path = project_root / "data" / "nba_data.duckdb"

    try:
        with duckdb.connect(str(db_path)) as conn:
            pending_debug = conn.execute("""
                SELECT COUNT(*) FROM placed_bets
                WHERE status = 'pending'
                AND bet_id LIKE 'bet_DEBUG%'
            """).fetchone()[0]

            if pending_debug == 0:
                logger.info("✅ Nessuna scommessa debug pending rimasta")
                return True
            else:
                logger.warning(f"⚠️ {pending_debug} scommesse debug pending ancora rimaste")
                return False

    except Exception as e:
        logger.error(f"❌ Errore verifica: {e}")
        return False

def main():
    """Processo principale di aggiornamento scommesse debug."""
    logger.info("🚀 AGGIORNAMENTO SCOMMESSE DEBUG PENDING")
    logger.info("=" * 50)

    # Step 1: Aggiorna scommesse debug
    if not settle_debug_bets():
        logger.error("❌ Aggiornamento fallito")
        return False

    # Step 2: Verifica risultato
    if not verify_no_pending_debug():
        logger.error("❌ Verifica fallita")
        return False

    logger.info("")
    logger.info("🎉 PROCESSO COMPLETATO CON SUCCESSO!")
    logger.info("Tutte le scommesse debug sono state aggiornate correttamente.")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)