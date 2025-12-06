import duckdb
import uuid
import logging
from datetime import timedelta

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DB_PATH = "data/nba_betting.duckdb"
INITIAL_DEPOSIT = 84.75  # Verified with User
USER_ID = "test_user_001"


def migrate():
    logger.info("🚀 Starting Ledger Migration (Corrected)...")
    conn = duckdb.connect(DB_PATH)

    # 1. Reset Transactions Table
    logger.info("🧹 Clearing existing transactions...")
    try:
        conn.execute("DELETE FROM transactions")
    except Exception as e:
        logger.warning(f"Could not clear table (might not exist): {e}")

    # 2. Determine Start Time (Backdate Deposit)
    try:
        min_time_res = conn.execute("SELECT MIN(created_at) FROM bets").fetchone()
        if min_time_res and min_time_res[0]:
            first_bet_time = min_time_res[0]
            deposit_time = first_bet_time - timedelta(seconds=1)
            logger.info(
                f"🕒 First bet was at {first_bet_time}. Backdating deposit to {deposit_time}"
            )
        else:
            deposit_time = "CURRENT_TIMESTAMP"
            logger.info("🕒 No bets found. Deposit will be at current time.")
    except Exception as e:
        logger.warning(f"Could not fetch min time: {e}")
        deposit_time = "CURRENT_TIMESTAMP"

    # 3. Create Initial Deposit Transaction
    deposit_id = str(uuid.uuid4())
    logger.info(f"💰 creating Initial Deposit: €{INITIAL_DEPOSIT}")

    # Handle timestamp format
    if isinstance(deposit_time, str):
        conn.execute(
            f"""
            INSERT INTO transactions (transaction_id, user_id, amount, type, description, created_at)
            VALUES (?, ?, ?, ?, ?, {deposit_time})
        """,
            (
                deposit_id,
                USER_ID,
                INITIAL_DEPOSIT,
                "DEPOSIT",
                "Initial Bankroll Deposit",
            ),
        )
    else:
        conn.execute(
            """
            INSERT INTO transactions (transaction_id, user_id, amount, type, description, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                deposit_id,
                USER_ID,
                INITIAL_DEPOSIT,
                "DEPOSIT",
                "Initial Bankroll Deposit",
                deposit_time,
            ),
        )

    # 4. Process All Bets
    # Use .df() as verified available
    bets = conn.execute("SELECT * FROM bets ORDER BY created_at ASC").df()
    logger.info(f"📊 Processing {len(bets)} bets...")

    processed_count = 0

    for _, bet in bets.iterrows():
        bet_id = bet["bet_id"]
        amount = float(bet["amount"])
        status = bet["status"]
        odds = float(bet["odds"])
        game_id = bet["game_id"]
        created_at = bet["created_at"]

        # A. Debit for Bet Placement
        conn.execute(
            """
            INSERT INTO transactions (transaction_id, user_id, amount, type, description, reference_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                str(uuid.uuid4()),
                USER_ID,
                -amount,
                "BET_PLACEMENT",
                f"Bet {bet_id} on {game_id}",
                bet_id,
                created_at,
            ),
        )

        # B. Credit for Wins/Refunds
        if status in ["WON", "PUSH", "void", "VOID"]:
            payout = 0.0
            desc = ""
            type_str = ""

            if status == "WON":
                payout = amount * odds
                desc = "Bet Won Payout"
                type_str = "BET_PAYOUT"
            else:
                payout = amount
                desc = "Bet Refund"
                type_str = "BET_REFUND"

            # Settlement time assumption: 2.5 hours after creation if settled_at missing
            settle_time = created_at + timedelta(hours=2.5)
            if bet["settled_at"]:
                settle_time = bet["settled_at"]

            conn.execute(
                """
                INSERT INTO transactions (transaction_id, user_id, amount, type, description, reference_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    str(uuid.uuid4()),
                    USER_ID,
                    payout,
                    type_str,
                    desc,
                    bet_id,
                    settle_time,
                ),
            )

        processed_count += 1

    logger.info(f"✅ Processed {processed_count} bets.")

    # 5. Verify Final Balance
    res = conn.execute(
        "SELECT SUM(amount) FROM transactions WHERE user_id = ?", (USER_ID,)
    ).fetchone()
    free_bankroll = res[0] if res and res[0] else 0.0

    # Get Locked
    res_locked = conn.execute(
        "SELECT SUM(amount) FROM bets WHERE status='PENDING' AND user_id = ?",
        (USER_ID,),
    ).fetchone()
    locked = res_locked[0] if res_locked and res_locked[0] else 0.0

    logger.info("------------------------------------------------")
    logger.info(f"💵 Calculated Free Bankroll (Ledger): €{free_bankroll:.2f}")
    logger.info(f"🔒 Locked Stakes (Bets):           €{locked:.2f}")
    logger.info(f"📈 Total Equity:                   €{free_bankroll + locked:.2f}")
    logger.info("------------------------------------------------")

    conn.close()


if __name__ == "__main__":
    migrate()
