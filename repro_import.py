import sys
import os
import duckdb
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal

# Add src to pythonpath
sys.path.append(os.path.abspath("src"))

from nba_predictor.bankroll.models import BetRecord, BetResult
from nba_predictor.bankroll.engine import TransactionEngine

# Setup logging
logging.basicConfig(level=logging.DEBUG)


def run_repro():
    print("--- Starting Repro Import ---")

    # Setup DB manually to inspect
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE bet_records (
            bet_id VARCHAR PRIMARY KEY,
            game_id VARCHAR NOT NULL,
            bet_type VARCHAR NOT NULL,
            selection VARCHAR NOT NULL,
            odds DECIMAL(10,4) NOT NULL,
            stake DECIMAL(15,2) NOT NULL,
            result VARCHAR NOT NULL,
            payout DECIMAL(15,2) NOT NULL,
            profit_loss DECIMAL(15,2) NOT NULL,
            placed_timestamp TIMESTAMP NOT NULL,
            settled_timestamp TIMESTAMP,
            metadata JSON,
            checksum VARCHAR NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create BetRecord using actual class
    original_bet = BetRecord(
        bet_id="test_bet_1",
        game_id="GAME_1",
        bet_type="moneyline",
        selection="home",
        odds=Decimal("2.5"),
        stake=Decimal("100.00"),
        result=BetResult.PENDING,
        payout=Decimal("0.00"),
        profit_loss=Decimal("0.00"),
        placed_timestamp=datetime.now(timezone.utc),
        settled_timestamp=None,
        metadata={"tag": "test"},
    )

    print(f"Original Checksum: {original_bet.checksum}")
    print(f"Original TS: {original_bet.placed_timestamp}")
    print(
        f"Formatted TS in checksum: {original_bet.placed_timestamp.astimezone(timezone.utc).replace(tzinfo=None).isoformat()}"
    )

    # Simulate Engine Save info
    # Engine does:
    # placed_timestamp.astimezone(timezone.utc).replace(tzinfo=None)
    # str(metadata)

    placed_ts_naive = original_bet.placed_timestamp.astimezone(timezone.utc).replace(
        tzinfo=None
    )
    metadata_json = json.dumps(original_bet.metadata)

    conn.execute(
        "INSERT INTO bet_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        [
            original_bet.bet_id,
            original_bet.game_id,
            original_bet.bet_type,
            original_bet.selection,
            float(original_bet.odds),
            float(original_bet.stake),
            original_bet.result.value,
            float(original_bet.payout),
            float(original_bet.profit_loss),
            placed_ts_naive,
            None,
            metadata_json,
            original_bet.checksum,
        ],
    )

    # Simulate Engine Retrieve
    row = conn.execute("SELECT * FROM bet_records").fetchone()
    print(f"\nRow Fetch: {row}")
    # map row to record logic
    # placed_ts = row[9] (if naive, attach UTC) -> BUT wait. Models.py uses replace(tzinfo=None) for checksum.

    # Engine logic for retrieval:
    placed_ts = row[9]
    if isinstance(placed_ts, str):
        placed_ts = datetime.fromisoformat(placed_ts)
    if placed_ts.tzinfo is None:
        placed_ts = placed_ts.replace(tzinfo=timezone.utc)

    # Metadata logic
    metadata_raw = row[11]
    # engine.py logic:
    if isinstance(metadata_raw, str):
        try:
            metadata = json.loads(metadata_raw)
        except:
            metadata = eval(metadata_raw) if metadata_raw else {}
    else:
        metadata = {}

    print(f"\nReconstructing with:")
    print(
        f"TS Original: {original_bet.placed_timestamp.astimezone(timezone.utc).replace(tzinfo=None).isoformat()}"
    )
    print(
        f"TS Fetched:  {placed_ts.astimezone(timezone.utc).replace(tzinfo=None).isoformat()}"
    )
    print(f"Metadata Org: {original_bet.metadata}")
    print(f"Metadata New: {metadata}")
    print(f"Odds Org: {original_bet.odds} ({type(original_bet.odds)})")
    print(f"Odds New: {Decimal(str(row[4]))} ({type(Decimal(str(row[4])))})")

    try:
        reconstructed = BetRecord(
            bet_id=row[0],
            game_id=row[1],
            bet_type=row[2],
            selection=row[3],
            odds=Decimal(str(row[4])),
            stake=Decimal(str(row[5])),
            result=BetResult(row[6]),
            payout=Decimal(str(row[7])),
            profit_loss=Decimal(str(row[8])),
            placed_timestamp=placed_ts,
            settled_timestamp=None,
            metadata=metadata,
        )

        print(f"\nReconstructed Checksum: {reconstructed.checksum}")
        print(f"Stored Checksum:        {row[12]}")

        if reconstructed.checksum != row[12]:
            print("❌ FAIL")
        else:
            print("✅ PASS")
    except Exception as e:
        print(f"Reconstruction failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_repro()
