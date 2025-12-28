"""
Feasibility Spike: Test nba_api Live Feed Latency.

Goal:
1. Connect to NBA API Scoreboard endpoint.
2. Check for Live Games.
3. If Live, poll every 5s to measure update frequency.
4. If Not Live, print schedule and exit.

Usage:
    python src/nba_predictor/spikes/test_nba_live_feed.py
"""

import sys
import os
import time
import logging
from datetime import datetime
import json

# Try importing nba_api
try:
    from nba_api.live.nba.endpoints import scoreboard
except ImportError:
    print("Error: nba_api not installed. Run 'pip install nba_api'")
    sys.exit(1)

# Add project root
sys.path.append(os.getcwd())

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("NBALiveSpike")


def measure_latency():
    logger.info("Connecting to NBA Live Scoreboard...")

    try:
        board = scoreboard.ScoreBoard()
        games = board.games.get_dict()
    except Exception as e:
        logger.error(f"Failed to fetch scoreboard: {e}")
        return

    live_games = [
        g for g in games if g["gameStatus"] == 2
    ]  # 2 = Live, 1 = Scheduled, 3 = Final
    scheduled_games = [g for g in games if g["gameStatus"] == 1]

    logger.info(f"Total Games Today: {len(games)}")
    logger.info(f"Live Games: {len(live_games)}")

    if not live_games:
        logger.info("No games are currently LIVE.")
        if scheduled_games:
            next_start = scheduled_games[0].get("gameTimeUTC")
            logger.info(f"Next game starts at: {next_start}")
        return

    # If Live, Poll for 60 seconds
    target_game = live_games[0]
    game_id = target_game["gameId"]
    home_team = target_game["homeTeam"]["teamName"]
    away_team = target_game["awayTeam"]["teamName"]

    logger.info(f"Monitoring Game: {away_team} @ {home_team} (ID: {game_id})")
    logger.info("Polling for 60 seconds to check update frequency...")

    last_score_str = ""
    updates_count = 0
    start_time = time.time()

    try:
        while time.time() - start_time < 60:
            # Re-fetch
            board = scoreboard.ScoreBoard()
            current_games = board.games.get_dict()
            game_data = next((g for g in current_games if g["gameId"] == game_id), None)

            if not game_data:
                logger.warning("Game data likely ended or disappeared.")
                break

            home_score = game_data["homeTeam"]["score"]
            away_score = game_data["awayTeam"]["score"]
            clock = game_data["gameClock"]
            period = game_data["period"]

            score_str = f"Q{period} {clock} | {away_team} {away_score} - {home_score} {home_team}"

            if score_str != last_score_str:
                logger.info(f"UPDATE [Lat: Unknown]: {score_str}")
                last_score_str = score_str
                updates_count += 1

            time.sleep(5)  # Poll every 5s

    except KeyboardInterrupt:
        logger.info("Stopping...")

    logger.info(f"Monitoring complete. Updates detected: {updates_count}")
    logger.info(
        "Conclusion: nba_api works. Latency check requires comparison with TV feed."
    )


if __name__ == "__main__":
    measure_latency()
