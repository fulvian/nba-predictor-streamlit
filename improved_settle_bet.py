
    def settle_bet(self, bet_id: str, result: str, final_score: float = None) -> bool:
        """
        Settle a bet with result - Improved version with better error handling.

        Args:
            bet_id: Bet ID to settle
            result: 'won', 'lost', 'void', or 'cancelled'
            final_score: Final score for line bets (optional)

        Returns:
            True if successful
        """
        max_retries = 3
        retry_delay = 0.5  # seconds

        for attempt in range(max_retries):
            conn = None
            try:
                # Fresh connection for each attempt
                conn = duckdb.connect(self.db_path)

                # Get bet details with explicit column names
                bet_info = conn.execute("""
                    SELECT
                        bet_id, stake, odds, potential_return, status, game_id
                    FROM placed_bets
                    WHERE bet_id = ?
                """, [bet_id]).fetchone()

                if not bet_info:
                    logger.error(f"Bet not found: {bet_id}")
                    return False

                if bet_info[4] != 'pending':
                    logger.warning(f"Bet {bet_id} already settled with status: {bet_info[4]}")
                    return False

                bet_id_db, stake, odds, potential_return, status, game_id = bet_info

                # Validate stake value
                try:
                    stake = float(stake)
                    if stake <= 0:
                        logger.error(f"Invalid stake amount for bet {bet_id}: {stake}")
                        return False
                except (ValueError, TypeError):
                    logger.error(f"Could not convert stake to float for bet {bet_id}: {stake}")
                    return False

                # Calculate result based on outcome
                if result == 'won':
                    result_amount = float(potential_return)
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

                # Start transaction with retry logic
                conn.execute("BEGIN TRANSACTION")

                try:
                    # Update bet status with explicit values
                    conn.execute("""
                        UPDATE placed_bets
                        SET
                            status = ?,
                            settled_at = CURRENT_TIMESTAMP,
                            result_amount = ?,
                            profit_loss = ?
                        WHERE bet_id = ?
                    """, [result, result_amount, profit_loss, bet_id])

                    # Update bankroll with validation
                    current_bankroll_result = conn.execute("""
                        SELECT setting_value
                        FROM betting_settings
                        WHERE setting_key = 'current_bankroll'
                    """).fetchone()

                    if not current_bankroll_result:
                        logger.error(f"Current bankroll setting not found for bet {bet_id}")
                        raise Exception("Bankroll setting missing")

                    try:
                        current_bankroll = float(current_bankroll_result[0])
                        new_bankroll = current_bankroll + result_amount

                        # Validate new bankroll
                        if new_bankroll < 0:
                            logger.warning(f"Bankroll would become negative: {new_bankroll}")

                        # Update bankroll setting
                        conn.execute("""
                            UPDATE betting_settings
                            SET setting_value = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE setting_key = 'current_bankroll'
                        """, [str(new_bankroll)])

                    except (ValueError, TypeError) as e:
                        logger.error(f"Bankroll calculation error for bet {bet_id}: {e}")
                        raise Exception(f"Bankroll calculation failed: {e}")

                    # Get next history_id with proper error handling
                    try:
                        next_id_result = conn.execute(
                            "SELECT COALESCE(MAX(history_id), 0) + 1 FROM bankroll_history"
                        ).fetchone()
                        next_history_id = next_id_result[0] if next_id_result else 1
                    except Exception as e:
                        logger.warning(f"Could not get next history_id, using 1: {e}")
                        next_history_id = 1

                    # Record bankroll change
                    conn.execute("""
                        INSERT INTO bankroll_history (
                            history_id, bet_id, transaction_type, amount,
                            balance_before, balance_after, notes
                        )
                        VALUES (?, ?, 'bet_settled', ?, ?, ?, ?)
                    """, [
                        next_history_id, bet_id, result_amount,
                        current_bankroll, new_bankroll, f"Bet settled: {result}"
                    ])

                    # Commit transaction
                    conn.execute("COMMIT")

                    logger.info(
                        f"Bet settled successfully: {bet_id} - {result}, "
                        f"P&L: €{profit_loss:.2f}, Bankroll: {current_bankroll} → {new_bankroll}"
                    )
                    return True

                except Exception as e:
                    conn.execute("ROLLBACK")
                    raise e

            except duckdb.InvalidatedException as e:
                logger.warning(f"Database invalidated on attempt {attempt + 1}: {e}")
                if conn:
                    conn.close()
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Max retries exceeded for bet {bet_id}")
                    return False

            except Exception as e:
                logger.error(f"Failed to settle bet {bet_id} on attempt {attempt + 1}: {e}")
                if conn:
                    conn.close()

                if "invalidated" in str(e).lower() and attempt < max_retries - 1:
                    logger.info(f"Database error detected, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return False

            finally:
                if conn:
                    conn.close()

        return False
    