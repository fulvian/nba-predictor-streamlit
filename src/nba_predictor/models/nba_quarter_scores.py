import polars as pl
import duckdb


class NbaQuarterScoresRepository:
    def __init__(self, db_path: str = "data/nba_betting.duckdb"):
        self.db_path = db_path
        self._conn = None

    @property
    def conn(self):
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path)
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def initialize_schema(self):
        """Create the nba_quarter_scores table if not exists."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS nba_quarter_scores (
                game_id VARCHAR PRIMARY KEY,
                q1_home INTEGER,
                q2_home INTEGER,
                q3_home INTEGER,
                q4_home INTEGER,
                ot_home INTEGER,
                q1_away INTEGER,
                q2_away INTEGER,
                q3_away INTEGER,
                q4_away INTEGER,
                ot_away INTEGER,
                half_home INTEGER, -- Calculated
                half_away INTEGER, -- Calculated
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

    def insert_scores(self, df: pl.DataFrame) -> int:
        """Insert scores from Polars DataFrame."""
        # Ensure correct column order/existence
        columns = [
            "game_id",
            "q1_home",
            "q2_home",
            "q3_home",
            "q4_home",
            "ot_home",
            "q1_away",
            "q2_away",
            "q3_away",
            "q4_away",
            "ot_away",
            "half_home",
            "half_away",
        ]

        # Insert using DuckDB Native Polars integration
        # Create temp view
        self.conn.register("df_view", df)

        self.conn.execute(f"""
            INSERT OR REPLACE INTO nba_quarter_scores ({", ".join(columns)})
            SELECT {", ".join(columns)} FROM df_view
        """)

        count = self.conn.execute("SELECT count(*) FROM df_view").fetchone()[0]
        self.conn.unregister("df_view")
        return count
