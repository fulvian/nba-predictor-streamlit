from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Alert:
    game_id: str
    timestamp: datetime
    strategy_name: str
    message: str
    severity: str  # 'INFO', 'WARNING', 'BET'
    matchup: str


class StrategyEngine:
    """
    Evaluates Live Game State against known Alpha Strategies.
    Strategies:
    1. 'Denver Lung': High Altitude Home vs B2B Away @ Halftime.
    2. 'Tired Frontrunner': Tired Home (3in4) Leading >5 @ Start Q4.
    """

    def evaluate(self, games: List[Dict[str, Any]]) -> List[Alert]:
        alerts = []

        for game in games:
            # Skip if no context
            if "home_context" not in game or "away_context" not in game:
                continue

            home_ctx = game["home_context"]
            away_ctx = game["away_context"]

            status = game["status"]  # 2 = Live
            period = game["period"]
            clock = game["clock"]  # String usually "12:00" or "PT05M"

            # --- STRATEGY 1: DENVER LUNG ---
            # Trigger: Halftime (Period 2 End or Period 3 Start)
            # Using Period >= 2 check? Halftime is usually Period 3 with high clock 12:00 or Period 2 with 0:00.
            # Let's say we check "Is Late Q2 or Early Q3"?
            # Simplest: Check if Game is at Halftime.
            # But nba_api status might just say "2" (Live).
            # Trigger window: Period 2 End (Clock 0:00) to Period 3 Start (Clock 11:00).

            is_high_altitude = home_ctx.get("is_high_altitude", 0) == 1
            is_b2b_opponent = (
                away_ctx.get("rest_days", 99) <= 1
            )  # B2B is 1 day rest in our logic (played yesterday)

            if is_high_altitude and is_b2b_opponent:
                # Check Timing (Halftime Window)
                if period == 2 and game["home_score"] > 0:  # Approaching half
                    # Just an INFO alert that setup is active
                    pass

                # BET TRIGGER: Halftime
                is_halftime = (
                    (period == 2 and clock == "0:00")
                    or (period == 3 and clock.startswith("12:"))
                    or (period == 3 and clock == "PT12M")
                )

                if is_halftime:
                    alerts.append(
                        Alert(
                            game_id=game["game_id"],
                            timestamp=datetime.now(),
                            strategy_name="The Denver Lung 🏔️",
                            message=f"BET HOME 2H/Live. Altitude Advantage vs B2B Opponent ({game['home_team']} vs {game['away_team']}).",
                            severity="BET",
                            matchup=f"{game['home_team']} vs {game['away_team']}",
                        )
                    )

            # --- STRATEGY 2: TIRED FRONTRUNNER ---
            # Trigger: Start of Q4 (Period 3 End / Period 4 Start)
            # Condition: Home Density >= 2 (3in4) AND Leading > 5

            home_density = home_ctx.get("density_4d", 0)

            if home_density >= 2:
                # Check Timing (Q3 End / Q4 Start)
                is_q4_start = (
                    (period == 3 and clock == "0:00")
                    or (period == 4 and clock.startswith("12:"))
                    or (period == 4 and clock == "PT12M")
                )

                if is_q4_start:
                    margin = game["home_score"] - game["away_score"]
                    if margin > 5:
                        alerts.append(
                            Alert(
                                game_id=game["game_id"],
                                timestamp=datetime.now(),
                                strategy_name="Tired Frontrunner 🥱",
                                message=f"FADE HOME Q4. Tired Home Team Leading by {margin}. Bet Away Q4.",
                                severity="BET",
                                matchup=f"{game['home_team']} vs {game['away_team']}",
                            )
                        )

        return alerts
