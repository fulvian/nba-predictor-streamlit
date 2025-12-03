"""
BayesianUpdater Module
----------------------
Dynamically updates prediction probabilities based on new information (e.g., injuries)
using Bayesian inference and Monte Carlo simulations.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class BayesianUpdateResult:
    """Result of a Bayesian update."""

    original_prob: float
    updated_prob: float
    original_score_dist: Tuple[float, float]  # (mean, std)
    updated_score_dist: Tuple[float, float]  # (mean, std)
    confidence_interval: Tuple[float, float]  # (low, high)
    simulation_runs: int


class BayesianUpdater:
    """
    Updates predictions using Bayesian inference and Monte Carlo simulations.

    Implements:
    - Likelihood mapping from injury impact scores
    - Posterior distribution calculation
    - Monte Carlo simulation for robust confidence intervals
    """

    def __init__(self, simulation_runs: int = 5000):
        """
        Initialize the Bayesian Updater.

        Args:
            simulation_runs: Number of Monte Carlo simulation runs (default 5000).
        """
        self.simulation_runs = simulation_runs

    def map_impact_to_likelihood(self, impact_score: float) -> float:
        """
        Map an injury impact score to a likelihood adjustment factor.

        Args:
            impact_score: Impact score from InjuryImpactAnalyzer (e.g., 1.5 for starter).

        Returns:
            Likelihood adjustment factor (negative impact reduces score/prob).
        """
        # Simple linear mapping for now:
        # Impact 1.0 (Bench) -> -1.0 point adjustment
        # Impact 2.0 (Star) -> -3.0 point adjustment
        # This is a simplified heuristic that should be refined with historical data.
        return -1.5 * impact_score

    def update_prediction(
        self, baseline_mean: float, baseline_std: float, injury_impacts: List[float]
    ) -> BayesianUpdateResult:
        """
        Update a prediction based on injury impacts using Monte Carlo simulation.

        Args:
            baseline_mean: Original predicted score (mean).
            baseline_std: Original prediction standard deviation (uncertainty).
            injury_impacts: List of impact scores for new injuries.

        Returns:
            BayesianUpdateResult containing updated metrics.
        """
        # 1. Calculate Total Impact Adjustment
        total_adjustment = sum(
            self.map_impact_to_likelihood(impact) for impact in injury_impacts
        )

        # 2. Define Posterior Distribution Parameters
        # The mean shifts by the total adjustment
        # The uncertainty (std) increases because news introduces volatility
        posterior_mean = baseline_mean + total_adjustment
        posterior_std = (
            baseline_std * 1.15
        )  # Increase uncertainty by 15% due to late news

        # 3. Run Monte Carlo Simulation
        # Generate 5000 samples from the posterior distribution
        simulated_scores = np.random.normal(
            posterior_mean, posterior_std, self.simulation_runs
        )

        # 4. Calculate Updated Metrics from Simulation
        sim_mean = float(np.mean(simulated_scores))
        sim_std = float(np.std(simulated_scores))

        # Calculate 95% Confidence Interval
        ci_low = float(np.percentile(simulated_scores, 2.5))
        ci_high = float(np.percentile(simulated_scores, 97.5))

        # Calculate Win Probability (assuming target is > opponent_score,
        # but here we just return the score distribution update for now)
        # For a binary probability update, we would need the opponent's score distribution.
        # Here we assume this is a Total Score prediction or Team Score prediction.

        return BayesianUpdateResult(
            original_prob=0.0,  # Placeholder, depends on context (win vs total)
            updated_prob=0.0,  # Placeholder
            original_score_dist=(baseline_mean, baseline_std),
            updated_score_dist=(sim_mean, sim_std),
            confidence_interval=(ci_low, ci_high),
            simulation_runs=self.simulation_runs,
        )

    def update_win_probability(
        self,
        team_mean: float,
        team_std: float,
        opponent_mean: float,
        opponent_std: float,
        team_injuries: List[float],
    ) -> BayesianUpdateResult:
        """
        Update win probability for a specific matchup.
        """
        # Update Team Score Distribution
        team_update = self.update_prediction(team_mean, team_std, team_injuries)
        new_team_mean, new_team_std = team_update.updated_score_dist

        # Calculate original win prob
        # Z = (Mean_Team - Mean_Opponent) / sqrt(Std_Team^2 + Std_Opponent^2)
        z_orig = (team_mean - opponent_mean) / np.sqrt(team_std**2 + opponent_std**2)
        orig_prob = norm.cdf(z_orig)

        # Calculate new win prob
        z_new = (new_team_mean - opponent_mean) / np.sqrt(
            new_team_std**2 + opponent_std**2
        )
        new_prob = norm.cdf(z_new)

        return BayesianUpdateResult(
            original_prob=float(orig_prob),
            updated_prob=float(new_prob),
            original_score_dist=(team_mean, team_std),
            updated_score_dist=(new_team_mean, new_team_std),
            confidence_interval=team_update.confidence_interval,
            simulation_runs=self.simulation_runs,
        )
