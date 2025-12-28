#!/usr/bin/env python3
"""
🌊 Dynamic Bias Manager for NBA Predictions
Implements Exponential Moving Average (EMA) bias correction to track and correct
systematic model errors per team (momentum).

Perplexity/User Best Practices 2025:
- Alpha (λ): 0.15 (Low reactivity to filter noise)
- Logic: Bias_t = λ * Bias_{t-1} + (1-λ) * (Actual - Predicted)
  Note: The formula in the plan was Bias_t = λ * Bias_{t-1} + (1-λ) * Error
  This implies NewBias is a weighted average of OldBias and LatestError.
  Wait, standard EMA formula is: EMA_t = alpha * Value_t + (1-alpha) * EMA_{t-1}
  Here "Value" is the Error.
  So: Bias_t = alpha * Error_t + (1-alpha) * Bias_{t-1}

  Let's stick to the User's formula concept which essentially means:
  "Update the bias by shifting it slightly towards the recent error."

"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TeamBiasState:
    team_id: str
    current_bias: float = 0.0
    last_update: str = ""  # ISO Date
    match_count: int = 0
    history: list = field(default_factory=list)


class DynamicBiasManager:
    """
    Manages dynamic bias correction for NBA teams using EMA.
    Tracks if the model consistently over/under estimates a team.
    """

    # Low alpha to filter noise (Perplexity 2025 Recommendation)
    EMA_ALPHA = 0.15

    # Storage file
    STORAGE_FILE = "data/intelligence/dynamic_bias_v1.json"

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or self.STORAGE_FILE
        self.states: Dict[str, TeamBiasState] = {}
        self._load_state()
        self._ensure_storage_dir()

    def _ensure_storage_dir(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

    def _load_state(self):
        """Load bias states from JSON."""
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
                for team_id, state_data in data.items():
                    self.states[team_id] = TeamBiasState(**state_data)
            logger.info(f"🌊 Loaded Dynamic Bias state for {len(self.states)} teams.")
        except Exception as e:
            logger.error(f"Failed to load dynamic bias state: {e}")

    def save_state(self):
        """Save bias states to JSON."""
        try:
            data = {
                team_id: {
                    "team_id": s.team_id,
                    "current_bias": s.current_bias,
                    "last_update": s.last_update,
                    "match_count": s.match_count,
                    "history": s.history[-50:],  # Keep last 50 only
                }
                for team_id, s in self.states.items()
            }
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save dynamic bias state: {e}")

    def get_game_bias(self, home_team_id: str, away_team_id: str) -> float:
        """
        Calculate total bias correction for a game.
        Total Bias = (HomeBias + AwayBias) / 2

        If we historically UNDERestimate Home (Bias > 0) and OVERestimate Away (Bias < 0),
        we sum them to get the net adjustment.
        """
        home_bias = self.states.get(
            home_team_id, TeamBiasState(home_team_id)
        ).current_bias
        away_bias = self.states.get(
            away_team_id, TeamBiasState(away_team_id)
        ).current_bias

        # We assume the bias represents "How much points the TEAM adds/subtracts vs expectation"
        # Since Totals involves two teams, we average their specific model-errors.
        net_bias = (home_bias + away_bias) / 2.0

        logger.info(
            f"🌊 Game Bias for {home_team_id} vs {away_team_id}: "
            f"Home={home_bias:.2f}, Away={away_bias:.2f} -> Net={net_bias:.2f}"
        )

        return net_bias

    def update_bias(
        self, team_id: str, predicted_score: float, actual_score: float, game_date: str
    ):
        """
        Update the bias state for a team based on a game result.
        Error = Actual - Predicted
        Bias_t = Alpha * Error + (1-Alpha) * Bias_{t-1}

        Note: This effectively shifts the bias towards the recent error.
        If Error is +10 (Actual > Pred), Bias increases.
        Next prediction will be Pred + Bias (resulting in higher prediction).
        """
        state = self.states.get(team_id, TeamBiasState(team_id=team_id))

        error = actual_score - predicted_score

        # EMA Update
        old_bias = state.current_bias
        new_bias = (self.EMA_ALPHA * error) + ((1.0 - self.EMA_ALPHA) * old_bias)

        # Update State
        state.current_bias = new_bias
        state.last_update = game_date
        state.match_count += 1
        state.history.append(
            {
                "date": game_date,
                "pred": predicted_score,
                "actual": actual_score,
                "error": error,
                "pre_bias": old_bias,
                "post_bias": new_bias,
            }
        )

        self.states[team_id] = state
        self.save_state()

        logger.info(
            f"🌊 Updated Bias for {team_id}: {old_bias:.2f} -> {new_bias:.2f} (Err={error:.1f})"
        )


# Singleton
_bias_manager = None


def get_dynamic_bias_manager() -> DynamicBiasManager:
    global _bias_manager
    if _bias_manager is None:
        _bias_manager = DynamicBiasManager()
    return _bias_manager
