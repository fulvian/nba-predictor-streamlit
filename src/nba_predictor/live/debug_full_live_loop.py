import sys
import os

sys.path.append(os.getcwd())

from unittest.mock import MagicMock, patch
from src.nba_predictor.live.monitor import LiveMonitor
from src.nba_predictor.live.engine import StrategyEngine
import polars as pl
from datetime import datetime


def test_full_loop():
    print("Testing Full Live Loop with Mock Data...")

    # 1. Mock Monitor to return a specific game scenario
    monitor = LiveMonitor()

    # inject mock context directly to avoid DB dependency for this test
    # Denver (Home, Alt=1) vs Lakers (Away, Rest=1 B2B)
    monitor.cached_context = pl.DataFrame(
        [
            {
                "game_id": "MOCK_GAME",
                "team_id": 100,
                "is_home": 1,
                "is_high_altitude": 1,
                "density_4d": 0,
                "rest_days": 2,
            },  # DEN
            {
                "game_id": "MOCK_GAME",
                "team_id": 200,
                "is_home": 0,
                "is_high_altitude": 0,
                "density_4d": 0,
                "rest_days": 1,
            },  # LAL (B2B)
        ]
    )
    monitor.last_context_update = datetime.now()

    # Mock API Response (At Halftime)
    mock_game_api = {
        "gameId": "MOCK_GAME",
        "gameStatus": 2,  # Live
        "period": 2,
        "gameClock": "0:00",  # Halftime Trigger
        "homeTeam": {"teamId": 100, "teamName": "Denver", "score": 50},
        "awayTeam": {"teamId": 200, "teamName": "Lakers", "score": 45},
    }

    # Patch the scoreboard call
    with patch("nba_api.live.nba.endpoints.scoreboard.ScoreBoard") as mock_board:
        mock_instance = mock_board.return_value
        mock_instance.games.get_dict.return_value = [mock_game_api]

        # Run Monitor Fetch
        print("Fetching State...")
        current_state = monitor.fetch_current_state()
        print(f"Enriched Games: {len(current_state)}")
        # print(current_state)

        # Run Strategy Engine
        print("Running Strategy Engine...")
        engine = StrategyEngine()
        alerts = engine.evaluate(current_state)

        print("\n--- ALERTS GENERATED ---")
        for a in alerts:
            print(f"[{a.severity}] {a.strategy_name}: {a.message}")

        # Assertions
        assert len(alerts) == 1
        assert "The Denver Lung" in alerts[0].strategy_name
        print("\n✅ Verification SUCCESS: Denver Lung Alert Triggered!")


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.getcwd())
    test_full_loop()
