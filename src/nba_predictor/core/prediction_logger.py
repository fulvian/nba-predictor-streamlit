import duckdb
import json
import hashlib
import logging
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class PredictionLogger:
    """
    Handles robust logging of prediction data to DuckDB for MLOps and meta-model training.

    Adheres to Point-in-Time (PIT) principles to prevent data leakage.
    Schema includes snapshots of market lines, model versions, and feature sets at prediction time.
    """

    def __init__(
        self, db_path: str = "data/nba_data.duckdb", schema_version: str = "v1.0"
    ):
        """
        Initialize the PredictionLogger.

        Args:
            db_path: Path to the DuckDB database file. Defaults to shared nba_data.duckdb.
            schema_version: Version of the logging schema (for evolution).
        """
        self.db_path = db_path
        self.schema_version = schema_version
        self.conn = None
        self._ensure_db_directory()
        self._init_db()

    def _ensure_db_directory(self):
        """Ensure the directory for the DB exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self):
        """Establish a connection to DuckDB."""
        # read_only=False is required for inserts
        # config for concurrency handles multi-process access better
        try:
            self.conn = duckdb.connect(database=self.db_path, read_only=False)
            # Basic optimization settings
            self.conn.execute("SET memory_limit = '1GB'")
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB at {self.db_path}: {e}")
            raise

    def _hash_string(self, input_string: str) -> str:
        """Create a reproducible hash for string content (prompts, rationale)."""
        if not input_string:
            return ""
        return hashlib.sha256(input_string.encode()).hexdigest()

    def _hash_context(self, context: dict) -> str:
        """Create a reproducible hash for dictionary content (features)."""
        if not context:
            return ""
        # Sort keys to ensure consistent hashing
        return self._hash_string(json.dumps(context, sort_keys=True))

    def _retry_operation(self, func, max_retries=3, delay=0.5):
        """Retry DB operations to handle locking (common in file-based DBs)."""
        last_error = None
        for attempt in range(max_retries):
            try:
                # Re-connect on each attempt to be safe with file locks
                if not self.conn:
                    self._connect()
                return func()
            except Exception as e:
                last_error = e
                if "database is locked" in str(e).lower():
                    time.sleep(delay * (2**attempt))
                    if self.conn:
                        try:
                            self.conn.close()
                        except:
                            pass
                        self.conn = None
                else:
                    raise e
        raise last_error

    def _init_db(self):
        """Initialize the predictions table if it doesn't exist."""

        def _create():
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS predictions (
                    -- PIT PRIMARY KEY: Uniquely identifies decision moment
                    game_id VARCHAR NOT NULL,
                    prediction_timestamp TIMESTAMP NOT NULL,
                    quant_model_version VARCHAR NOT NULL,
                    
                    -- Key Timestamps
                    market_line_captured_at TIMESTAMP,
                    game_start_timestamp TIMESTAMP,
                    feature_timestamp TIMESTAMP,  -- When was feature data frozen?
                    
                    -- Game Context
                    game_date DATE NOT NULL,
                    home_team_id VARCHAR NOT NULL,
                    away_team_id VARCHAR NOT NULL,
                    
                    -- Raw Components (for Meta-Model features)
                    quant_model_prediction DOUBLE NOT NULL,
                    quant_model_uncertainty DOUBLE,
                    quant_features_hash VARCHAR,  -- Hash of feature config
                    
                    llm_model_version VARCHAR,
                    llm_prompt_hash VARCHAR,      -- Hash of prompt template
                    llm_raw_adjustment DOUBLE,
                    llm_rationale_hash VARCHAR, -- Hash of rationale text
                    llm_uncertainty DOUBLE,
                    llm_risk_level VARCHAR,

                    -- Market Data
                    market_line DOUBLE,
                    market_line_timestamp TIMESTAMP,

                    -- Final Prediction & Weights
                    unified_prediction DOUBLE NOT NULL,
                    weight_quant DOUBLE NOT NULL,
                    weight_llm DOUBLE NOT NULL,
                    weight_market DOUBLE NOT NULL,

                    -- Outcome (to be filled later)
                    outcome DOUBLE,
                    result_status VARCHAR, -- e.g., PENDING, WIN, LOSS, PUSH

                    -- Schema Versioning
                    schema_version VARCHAR NOT NULL,
                    
                    PRIMARY KEY (game_id, prediction_timestamp, quant_model_version)
                );
            """)

        try:
            self._retry_operation(_create)
            logger.info(
                f"PredictionLogger initialized at {self.db_path} (schema {self.schema_version})"
            )
        finally:
            if self.conn:
                self.conn.close()
                self.conn = None

    def log_prediction(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        game_date: date,
        quant_pred: float,
        final_pred: float,
        weights: Dict[str, float],
        quant_version: str,
        market_line: Optional[float] = None,
        llm_adjustment: Optional[float] = None,
        llm_rationale: Optional[str] = None,
        llm_version: str = "default_llm",
        llm_uncertainty: Optional[float] = None,
        llm_risk_level: Optional[str] = None,
        team_stats: Optional[dict] = None,
        game_start_ts: Optional[datetime] = None,
    ):
        """
        Log a complete prediction event.
        """
        now = datetime.now()

        # Hashes
        quant_features_hash = self._hash_context(team_stats) if team_stats else None
        llm_rationale_hash = self._hash_string(llm_rationale) if llm_rationale else None
        # Placeholder for prompt hash - in future pass actual prompt or ID
        llm_prompt_hash = self._hash_string("default_prompt")

        # Extract weights
        w_quant = weights.get("quant", 0.0)
        w_llm = weights.get("consensus", 0.0)  # Using 'consensus' key as per pipeline
        w_market = weights.get("market", 0.0)

        def _insert():
            self.conn.execute(
                """
                INSERT INTO predictions VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                    ?, ?, ?, ?, ?, ?, ?
                )
            """,
                [
                    game_id,
                    now,
                    quant_version,
                    now if market_line else None,  # market_line_captured_at
                    game_start_ts,
                    now,  # feature_timestamp (approximate)
                    game_date,
                    home_team,
                    away_team,
                    quant_pred,
                    None,  # quant_model_uncertainty
                    quant_features_hash,
                    llm_version,
                    llm_prompt_hash,
                    llm_adjustment if llm_adjustment is not None else 0.0,
                    llm_rationale_hash,
                    llm_uncertainty,
                    llm_risk_level,
                    market_line,
                    now if market_line else None,  # market_line_timestamp
                    final_pred,
                    w_quant,
                    w_llm,
                    w_market,
                    None,  # outcome
                    "PENDING",  # result_status
                    self.schema_version,
                ],
            )

        try:
            self._retry_operation(_insert)
            logger.info(f"Logged prediction for {game_id}")
        except Exception as e:
            logger.error(f"Failed to log prediction for {game_id}: {e}")
            raise
        finally:
            if self.conn:
                self.conn.close()
                self.conn = None

    def add_outcome(self, game_id: str, result_total: float):
        """Update actual outcome for a game."""

        def _update():
            # Determine status if possible
            # This logic is simple; real evaluation usually needs line comparison
            self.conn.execute(
                f"""
                UPDATE predictions
                SET outcome = ?,
                    result_status = 'COMPLETED'
                WHERE game_id = ?
            """,
                [result_total, game_id],
            )

        try:
            self._retry_operation(_update)
            logger.info(f"Updated outcome for {game_id}: {result_total}")
        except Exception as e:
            logger.error(f"Failed to update outcome for {game_id}: {e}")
        finally:
            if self.conn:
                self.conn.close()
                self.conn = None

    def get_logs(self, limit: int = 100) -> list:
        """Fetch recent logs for verification."""

        def _fetch():
            return self.conn.execute(f"""
                SELECT * FROM predictions 
                ORDER BY prediction_timestamp DESC 
                LIMIT {limit}
            """).fetchall()

        try:
            res = self._retry_operation(_fetch)
            return res
        finally:
            if self.conn:
                self.conn.close()
                self.conn = None
