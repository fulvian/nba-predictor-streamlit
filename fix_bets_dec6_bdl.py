import duckdb
import os
import logging
from datetime import date
from dotenv import load_dotenv
from nba_predictor.api.ball_dont_lie_client import NBABallDontLieClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fix_bets_dec6")


def fix_bets():
    load_dotenv()
    api_key = os.getenv("BALLDONTLIE_API_KEY")
    if not api_key:
        logger.error("BALLDONTLIE_API_KEY not found!")
        return

    try:
        # 1. Fetch metadata for Dec 6
        logger.info("Fetching game metadata from BallDontLie for 2025-12-06...")
        client = NBABallDontLieClient(api_key)
        games = client.get_games_for_date_range(date(2025, 12, 6))

        game_map = {}
        for g in games:
            game_map[g["game_id"]] = (g["home_team"], g["away_team"])

        logger.info(f"Found {len(game_map)} games in metadata.")

        # 2. Update DB (DuckDB)
        db_path = "data/nba_betting.duckdb"
        logger.info(f"Connecting to {db_path}...")
        conn = duckdb.connect(db_path)

        # Find broken bets (Note: DuckDB syntax)
        # Check if columns exist first? SecureBettingDatabaseManager code shows they are added via ALTER if missing.
        # Assuming they exist but are null.

        query_check = "SELECT bet_id, game_id FROM bets WHERE game_id LIKE 'BDL_%' AND (home_team IS NULL OR home_team = '')"
        broken_bets = conn.execute(query_check).fetchall()
        logger.info(f"Found {len(broken_bets)} bets with missing matching info.")

        updated_count = 0
        for bet_id, game_id in broken_bets:
            # Map BDL_ID matching. BDL client returns BDL_xxxxx.
            # In DB, format is likely BDL_xxxxx.
            if game_id in game_map:
                home, away = game_map[game_id]
                # Escape single quotes in names just in case (e.g. 76ers? no, names are usually safe but...)
                home_safe = home.replace("'", "''")
                away_safe = away.replace("'", "''")

                update_sql = f"UPDATE bets SET home_team = '{home_safe}', away_team = '{away_safe}' WHERE bet_id = '{bet_id}'"
                conn.execute(update_sql)
                updated_count += 1
            else:
                logger.warning(f"Game ID {game_id} not found in metadata map!")

        # DuckDB auto-commits usually, but explicit check doesn't hurt.
        conn.checkpoint()  # Force write to disk?
        conn.close()
        logger.info(f"Successfully updated {updated_count} bets.")

    except Exception as e:
        logger.error(f"Error fixing bets: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    fix_bets()
