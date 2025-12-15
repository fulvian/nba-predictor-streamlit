import json
import logging
import duckdb
from pathlib import Path

# Setup Logger
logger = logging.getLogger(__name__)


class FeedbackLoop:
    """
    Component for the Meta-Learning Feedback Loop.
    Analyzes past errors to provide context for future Consensus predictions.
    """

    def __init__(self, db_path: str = "data/nba_betting.duckdb"):
        self.db_path = db_path
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        if not Path(self.db_path).exists():
            logger.warning(
                f"Database not found at {self.db_path}. Feedback loop will be skipped."
            )

    def fetch_team_history(self, team_name: str, limit: int = 10) -> list[dict]:
        """
        Retrieves the last N settled/completed bets for a specific team.
        Returns a list of dictionaries with Prediction, Result, and calculated Error.
        """
        if not Path(self.db_path).exists():
            return []

        try:
            # FIX: Use duckdb instead of sqlite3 for .duckdb files
            conn = duckdb.connect(self.db_path, read_only=True)

            # Query DuckDB
            # Updated: Use 'created_at' instead of 'date' and include WON/LOST statuses.
            query = """
                SELECT 
                    created_at,
                    home_team,
                    away_team,
                    prediction, 
                    result,
                    home_score,
                    away_score,
                    bet_id
                FROM bets 
                WHERE (home_team = ? OR away_team = ?)
                  AND status IN ('SETTLED', 'WON', 'LOST')
                ORDER BY created_at DESC 
                LIMIT ?
            """
            rows = conn.execute(query, [team_name, team_name, limit]).fetchall()

            history = []
            for row in rows:
                try:
                    # DuckDB returns tuples, indexing is same
                    # Parse JSON fields if they are stored as strings
                    prediction_data = (
                        json.loads(row[3]) if isinstance(row[3], str) else row[3]
                    )

                    # Simplify: We just need total score prediction vs actual total
                    # Assuming prediction_data has 'total' and home/away score gives actual.
                    pred_total = float(prediction_data.get("total", 0.0))

                    if row[5] is None or row[6] is None:
                        continue  # Skip if no scores

                    actual_total = float(row[5]) + float(row[6])

                    # Error > 0 means Overestimation (Predicted > Actual)
                    # Error < 0 means Underestimation (Predicted < Actual)
                    error = pred_total - actual_total

                    history.append(
                        {
                            "bet_id": row[7],
                            "date": row[0],
                            "prediction": pred_total,
                            "actual": actual_total,
                            "error": error,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to parse row for {team_name}: {e}")
                    continue

            conn.close()
            return history

        except Exception as e:
            logger.error(f"Error fetching history for {team_name}: {e}")
            return []

    def calculate_weighted_bias(
        self, history: list[dict], alpha: float = 0.25
    ) -> float:
        """
        Calculates Exponential Moving Average (EMA) of forecast errors.
        Alpha 0.25 gives adequate weight to the last ~4 games without chasing noise.
        """
        if not history:
            return 0.0

        # Sort chronological: Oldest -> Newest
        ordered_history = history[::-1]

        ema = 0.0
        first = True

        for game in ordered_history:
            error = game.get("error", 0.0)
            if first:
                ema = error
                first = False
            else:
                ema = (error * alpha) + (ema * (1 - alpha))

        return ema

    def generate_correction_prompt(self, team1: str, team2: str) -> str:
        """
        Generates the prompt to inject into Consensus if bias is detected.
        """
        t1_hist = self.fetch_team_history(team1)
        t2_hist = self.fetch_team_history(team2)

        t1_bias = self.calculate_weighted_bias(t1_hist)
        t2_bias = self.calculate_weighted_bias(t2_hist)

        # TRIGGER THRESHOLD:
        # 3.0 was too low (noise). 5.0 is approx 0.5 SD in NBA Totals.
        # We only want to correct persistent, structural errors.
        THRESHOLD = 5.0

        prompt_parts = []

        if abs(t1_bias) > THRESHOLD:
            direction = "OVERESTIMATING" if t1_bias > 0 else "UNDERESTIMATING"
            prompt_parts.append(
                f"**{team1} Analysis (Last {len(t1_hist)} Games):**\n"
                f"- Bias: {direction} TOTAL SCORE by {abs(t1_bias):.1f} pts (EMA). \n"
                f"- Instruction: Apply a {('negative' if t1_bias > 0 else 'positive')} correction to your total score prediction."
            )

        if abs(t2_bias) > THRESHOLD:
            direction = "OVERESTIMATING" if t2_bias > 0 else "UNDERESTIMATING"
            prompt_parts.append(
                f"**{team2} Analysis (Last {len(t2_hist)} Games):**\n"
                f"- Bias: {direction} TOTAL SCORE by {abs(t2_bias):.1f} pts (EMA). \n"
                f"- Instruction: Apply a {('negative' if t2_bias > 0 else 'positive')} correction to your total score prediction."
            )

        if not prompt_parts:
            return ""

        return (
            "\n\n[SYSTEM NOTIFICATION: DETECTED PREDICTION BIAS]\n"
            "The following meta-learning insights are based on your recent error patterns:\n\n"
            + "\n\n".join(prompt_parts)
            + "\n\n> **MANDATORY**: Adjust your final score prediction to account for these detected biases."
        )


if __name__ == "__main__":
    # Test stub
    fl = FeedbackLoop()
    print("Feedback Loop Module Initialized.")
