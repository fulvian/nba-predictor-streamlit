#!/usr/bin/env python3
"""
Module import to fix bet cancellation issues.
Import this module in betting_workflow_dashboard.py to override the problematic settle_bet method.
"""

import duckdb
import time
import logging

logger = logging.getLogger(__name__)

def improved_settle_bet(self, bet_id: str, result: str, final_score: float = None) -> bool:
    """
    Improved settle_bet method with robust error handling and retry logic.
    """
    max_retries = 3
    retry_delay = 0.5

    for attempt in range(max_retries):
        conn = None
        try:
            # Create fresh connection
            conn = duckdb.connect(str(self.db_path) if hasattr(self, 'db_path') else 'data/nba_data.duckdb')

            # Get bet details
            bet_info = conn.execute("""
                SELECT bet_id, stake, odds, potential_return, status, game_id
                FROM placed_bets WHERE bet_id = ?
            """, [bet_id]).fetchone()

            if not bet_info:
                logger.error(f"Bet not found: {bet_id}")
                return False

            if bet_info[4] != 'pending':
                logger.warning(f"Bet {bet_id} already settled with status: {bet_info[4]}")
                return False

            stake, odds, potential_return = float(bet_info[1]), float(bet_info[2]), float(bet_info[3])

            # Calculate result
            if result == 'won':
                result_amount = potential_return
                profit_loss = result_amount - stake
            elif result == 'lost':
                result_amount = 0.0
                profit_loss = -stake
            elif result == 'void':
                result_amount = stake
                profit_loss = 0.0
            else:  # cancelled
                result_amount = stake
                profit_loss = 0.0

            # Transaction
            conn.execute("BEGIN TRANSACTION")
            try:
                # Update bet
                conn.execute("""
                    UPDATE placed_bets
                    SET status = ?, settled_at = CURRENT_TIMESTAMP, result_amount = ?, profit_loss = ?
                    WHERE bet_id = ?
                """, [result, result_amount, profit_loss, bet_id])

                # Update bankroll
                current_bankroll = float(conn.execute(
                    "SELECT setting_value FROM betting_settings WHERE setting_key = 'current_bankroll'"
                ).fetchone()[0])

                new_bankroll = current_bankroll + result_amount
                conn.execute("""
                    UPDATE betting_settings
                    SET setting_value = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE setting_key = 'current_bankroll'
                """, [str(new_bankroll)])

                # History record
                next_id = conn.execute(
                    "SELECT COALESCE(MAX(history_id), 0) + 1 FROM bankroll_history"
                ).fetchone()[0]

                conn.execute("""
                    INSERT INTO bankroll_history (history_id, bet_id, transaction_type, amount, balance_before, balance_after, notes)
                    VALUES (?, ?, 'bet_settled', ?, ?, ?, ?)
                """, [next_id, bet_id, result_amount, current_bankroll, new_bankroll, f"Bet settled: {result}"])

                conn.execute("COMMIT")
                logger.info(f"✅ Bet {bet_id} settled successfully as {result}")
                return True

            except Exception as e:
                conn.execute("ROLLBACK")
                raise e

        except Exception as e:
            if "invalidated" in str(e).lower() and attempt < max_retries - 1:
                logger.warning(f"Database invalidated, retrying... ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"Failed to settle bet {bet_id}: {e}")
                return False
        finally:
            if conn:
                conn.close()

    return False

# Monkey patch the method
from nba_predictor.utils.betting_database_manager import BettingDatabaseManager
BettingDatabaseManager.settle_bet = improved_settle_bet

logger.info("✅ Improved settle_bet method patched")
