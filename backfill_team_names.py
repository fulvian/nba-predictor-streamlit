import duckdb
import logging
from datetime import datetime
from nba_api.stats.endpoints import boxscoresummaryv2
import json
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DB_PATH = "data/nba_betting.duckdb"


def backfill_teams():
    logger.info("Starting Backfill of Team Names...")

    conn = duckdb.connect(DB_PATH)

    # Check if columns exist (Force Migration)
    try:
        conn.execute("SELECT home_team FROM bets LIMIT 1")
    except:
        logger.info("Columns missing. Adding them now...")
        try:
            conn.execute("ALTER TABLE bets ADD COLUMN home_team VARCHAR(100)")
            conn.execute("ALTER TABLE bets ADD COLUMN away_team VARCHAR(100)")
            logger.info("✅ Columns added successfully.")
        except Exception as e:
            logger.error(f"Failed to add columns: {e}")
            return

    # Get bets with missing names OR "Unknown Team" placeholders (re-run to fix them)
    # searching for "Unknown Team" to fix the BDL ones
    bets = conn.execute(
        "SELECT bet_id, game_id, created_at FROM bets WHERE home_team IS NULL OR away_team IS NULL OR home_team = 'Unknown Team'"
    ).fetchall()
    logger.info(f"Found {len(bets)} bets to backfill.")

    updated_count = 0

    from nba_api.stats.static import teams

    nba_teams = teams.get_teams()
    id_to_name = {t["id"]: t["full_name"] for t in nba_teams}

    for bet in bets:
        bet_id, game_id, created_at = bet
        home_team, away_team = None, None

        # logger.info(f"Processing Bet {bet_id} (Game {game_id})...")

        if str(game_id).startswith("00"):
            # Official NBA ID
            try:
                box = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id)
                summary = box.game_summary.get_dict()["data"]
                if summary:
                    headers = box.game_summary.get_dict()["headers"]
                    row = summary[0]
                    h_idx = headers.index("HOME_TEAM_ID")
                    v_idx = headers.index("VISITOR_TEAM_ID")
                    home_id = row[h_idx]
                    visit_id = row[v_idx]
                    home_team = id_to_name.get(home_id, f"Team {home_id}")
                    away_team = id_to_name.get(visit_id, f"Team {visit_id}")
            except Exception as e:
                pass  # logger.error(f"Error for {game_id}: {e}")

        # ---------------------------
        # Improved Fallback: Parse JSON from prediction
        # ---------------------------
        if not home_team or not away_team:
            try:
                pred_row = conn.execute(
                    f"SELECT prediction FROM bets WHERE bet_id='{bet_id}'"
                ).fetchone()
                if pred_row and pred_row[0]:
                    raw_text = str(pred_row[0])
                    found_json = False

                    try:
                        # Try parsing as JSON
                        data = json.loads(raw_text)

                        # Check for team_metrics
                        # Structure seen: "team_metrics": {"home": {"team_name": "Orlando Magic"}, ...}
                        metrics = data.get("team_metrics", {})
                        if metrics:
                            h = metrics.get("home", {}).get("team_name")
                            a = metrics.get("away", {}).get("team_name")

                            if h and a:
                                home_team = h
                                away_team = a
                                found_json = True
                    except:
                        pass

                    if not found_json:
                        # Regex Fallback if JSON fails or incomplete
                        match = re.search(
                            r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:vs\.?|@)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
                            raw_text,
                        )
                        if match:
                            away_team = match.group(1).strip()
                            home_team = match.group(2).strip()
            except Exception as e:
                pass

        # Final Fallback
        if not home_team:
            home_team = "Unknown Team"
        if not away_team:
            away_team = "Unknown Team"

        if home_team and away_team:
            conn.execute(
                "UPDATE bets SET home_team = ?, away_team = ? WHERE bet_id = ?",
                (home_team, away_team, bet_id),
            )
            updated_count += 1
            if home_team != "Unknown Team":
                logger.info(f"✅ Updated {bet_id}: {away_team} @ {home_team}")

    logger.info(f"Backfill Complete. Updated {updated_count}/{len(bets)} bets.")

    # ---------------------------
    # BANKROLL VERIFICATION
    # ---------------------------
    logger.info("\n💰 Verifying Bankroll Calculation...")
    try:
        from nba_predictor.utils.betting_database_manager import BettingDatabaseManager

        mgr = BettingDatabaseManager()
        free, locked = mgr.calculate_bankroll_from_db("test_user_001")
        logger.info(f"   ✅ Calculated Free Bankroll: €{free:.2f}")
        logger.info(f"   ✅ Calculated Locked: €{locked:.2f}")
        logger.info(f"   ✅ Total Equity: €{free + locked:.2f}")
    except Exception as e:
        logger.error(f"Bankroll check failed: {e}")

    conn.close()


if __name__ == "__main__":
    backfill_teams()
