import duckdb
import json
import hashlib
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from enum import Enum


# --- Mock Models ---
class BetResult(Enum):
    PENDING = "PENDING"
    WON = "WON"
    LOST = "LOST"
    VOID = "VOID"
    CASHOUT = "CASHOUT"


@dataclass(frozen=True)
class BetRecord:
    bet_id: str
    game_id: str
    bet_type: str
    selection: str
    odds: Decimal
    stake: Decimal
    result: BetResult
    payout: Decimal
    profit_loss: Decimal
    placed_timestamp: datetime
    settled_timestamp: Optional[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = field(init=False)

    def __post_init__(self):
        # Timezone handling
        if self.placed_timestamp.tzinfo is None:
            object.__setattr__(
                self,
                "placed_timestamp",
                self.placed_timestamp.replace(tzinfo=timezone.utc),
            )
        if self.settled_timestamp and self.settled_timestamp.tzinfo is None:
            object.__setattr__(
                self,
                "settled_timestamp",
                self.settled_timestamp.replace(tzinfo=timezone.utc),
            )

        # Checksum calculation logic from models.py
        checksum_data = {
            "bet_id": self.bet_id,
            "game_id": self.game_id,
            "bet_type": self.bet_type,
            "selection": self.selection,
            "odds": float(self.odds),
            "stake": float(self.stake),
            "result": self.result.value,
            "payout": float(self.payout),
            "profit_loss": float(self.profit_loss),
            "placed_timestamp": self.placed_timestamp.isoformat(),
            "settled_timestamp": self.settled_timestamp.isoformat()
            if self.settled_timestamp
            else None,
            "metadata": self.metadata,
        }
        checksum_str = json.dumps(checksum_data, sort_keys=True)
        checksum = hashlib.sha256(checksum_str.encode()).hexdigest()
        object.__setattr__(self, "checksum", checksum)
        print(f"DEBUG: Calculated checksum: {checksum}")
        print(f"DEBUG: Checksum data string: {checksum_str}")


# --- Reproduction ---
def run_reproduction():
    print("--- Starting Reproduction ---")

    # 1. Setup DB
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

    # 2. Create BetRecord
    original_bet = BetRecord(
        bet_id=f"bet_{uuid.uuid4().hex}",
        game_id="GAME_123",
        bet_type="moneyline",
        selection="home",
        odds=Decimal("2.50"),  # Note: logic used Decimal(str(row[4]))
        stake=Decimal("100.00"),
        result=BetResult.PENDING,
        payout=Decimal("0.00"),
        profit_loss=Decimal("0.00"),
        placed_timestamp=datetime.now(timezone.utc),
        settled_timestamp=None,
        metadata={"user_id": 123, "tags": ["nba", "test"]},
    )

    print(f"\nOriginal Checksum: {original_bet.checksum}")

    # 3. Insert into DB
    conn.execute(
        """
        INSERT INTO bet_records (
            bet_id, game_id, bet_type, selection, odds, stake, result, 
            payout, profit_loss, placed_timestamp, settled_timestamp, 
            metadata, checksum
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        [
            original_bet.bet_id,
            original_bet.game_id,
            original_bet.bet_type,
            original_bet.selection,
            original_bet.odds,  # Decimal
            original_bet.stake,  # Decimal
            original_bet.result.value,
            original_bet.payout,
            original_bet.profit_loss,
            original_bet.placed_timestamp,
            original_bet.settled_timestamp,
            json.dumps(original_bet.metadata),
            original_bet.checksum,
        ],
    )

    # 4. Read back
    row = conn.execute("SELECT * FROM bet_records").fetchone()
    print(f"\nRow fetched: {row}")
    # Row indices:
    # 0:bet_id, 1:game_id, 2:bet_type, 3:selection, 4:odds, 5:stake, 6:result,
    # 7:payout, 8:profit_loss, 9:placed_ts, 10:settled_ts, 11:metadata, 12:checksum

    # 5. Reconstruct logic (from engine.py _row_to_bet_record)
    placed_ts = row[9]
    if isinstance(placed_ts, str):
        placed_ts = datetime.fromisoformat(placed_ts)
    if placed_ts.tzinfo is None:
        placed_ts = placed_ts.replace(tzinfo=timezone.utc)

    settled_ts = row[10]
    if settled_ts:
        if isinstance(settled_ts, str):
            settled_ts = datetime.fromisoformat(settled_ts)
        if settled_ts.tzinfo is None:
            settled_ts = settled_ts.replace(tzinfo=timezone.utc)

    metadata_raw = row[11]
    # Check what DuckDB actually returns for JSON column? String or object?
    # In engine.py logic:
    if isinstance(metadata_raw, str):
        try:
            metadata = json.loads(metadata_raw)
        except:
            metadata = {}
    elif metadata_raw is None:
        metadata = {}
    else:
        # Maybe it returns a dict already? (unlikely for lightweight driver unless configured)
        metadata = metadata_raw

    # Reconstruction
    reconstructed_bet = BetRecord(
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
        settled_timestamp=settled_ts,
        metadata=metadata,
    )

    print(f"\nTimezone Check:")
    print(f"Original TS: {original_bet.placed_timestamp.isoformat()}")
    print(f"Fetched TS:  {placed_ts.isoformat()}")

    print(f"\nReconstructed Checksum: {reconstructed_bet.checksum}")
    print(f"Stored Checksum:        {row[12]}")

    if reconstructed_bet.checksum != row[12]:
        print("\n❌ MISMATCH DETECTED!")
    else:
        print("\n✅ MATCH!")


if __name__ == "__main__":
    run_reproduction()
