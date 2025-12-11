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
        """
        self.simulation_runs = simulation_runs
        # Top 50 NBA Players (2024-25) for Dynamic Tiering
        # This acts as a 'Tier 1/2' lookup. Others are assumed Starters/Bench.
        self.STAR_PLAYERS = {
            "Nikola Jokic",
            "Giannis Antetokounmpo",
            "Luka Doncic",
            "Joel Embiid",
            "Shai Gilgeous-Alexander",
            "Jayson Tatum",
            "Stephen Curry",
            "Kevin Durant",
            "LeBron James",
            "Anthony Davis",
            "Devin Booker",
            "Anthony Edwards",
            "Jalen Brunson",
            "Tyrese Haliburton",
            "Kawhi Leonard",
            "Donovan Mitchell",
            "Victor Wembanyama",
            "Bam Adebayo",
            "De'Aaron Fox",
            "Domantas Sabonis",
            "Ja Morant",
            "Kyrie Irving",
            "Paul George",
            "Damian Lillard",
            "Jimmy Butler",
            "Zion Williamson",
            "Trae Young",
            "Lauri Markkanen",
            "Karl-Anthony Towns",
            "Jaylen Brown",
            "Tyrese Maxey",
            "Pascal Siakam",
            "Jamal Murray",
            "DeMar DeRozan",
            "Julius Randle",
            "Chet Holmgren",
            "Paolo Banchero",
            "Scottie Barnes",
            "Alperen Sengun",
            "LaMelo Ball",
        }

    def get_player_tier_impact(self, player_name: str) -> float:
        """
        Determine impact score based on player tier.

        Returns:
            float: Impact score (e.g., 2.5 for Star, 1.0 for Starter)
        """
        if not player_name:
            return 0.5

        # Normalize comparison
        player_norm = player_name.strip().title()

        # Check explicit star list
        if player_norm in self.STAR_PLAYERS:
            return 3.0  # Star Tier

        # Heuristic for other distinct names or logic could go here
        # For now, default to Starter/Rotation level
        return 1.2

    def map_impact_to_likelihood(self, impact_score: float) -> float:
        """
        Map an injury impact score to a likelihood adjustment factor.
        """
        # Linear mapping: Impact * -1.5 (e.g., Star 3.0 * -1.5 = -4.5 pts)
        return -1.5 * impact_score

    def update_prediction(
        self, baseline_mean: float, baseline_std: float, injury_impacts: List[float]
    ) -> BayesianUpdateResult:
        """
        Update a prediction based on injury impacts using Monte Carlo simulation.
        """
        # 1. Calculate Total Impact Adjustment
        total_adjustment = sum(
            self.map_impact_to_likelihood(impact) for impact in injury_impacts
        )

        # 2. Define Posterior Distribution Parameters
        # The mean shifts by the total adjustment
        # The uncertainty (std) increases because news introduces volatility
        posterior_mean = baseline_mean + total_adjustment

        # Dynamic Uncertainty Increase: More impact = more uncertainty
        volatility_factor = 1.0 + (
            sum(injury_impacts) * 0.1
        )  # +10% std per impact unit
        posterior_std = baseline_std * volatility_factor

        # 3. Run Monte Carlo Simulation
        simulated_scores = np.random.normal(
            posterior_mean, posterior_std, self.simulation_runs
        )

        # 4. Calculate Updated Metrics from Simulation
        sim_mean = float(np.mean(simulated_scores))
        sim_std = float(np.std(simulated_scores))

        # Calculate 95% Confidence Interval
        ci_low = float(np.percentile(simulated_scores, 2.5))
        ci_high = float(np.percentile(simulated_scores, 97.5))

        return BayesianUpdateResult(
            original_prob=0.0,
            updated_prob=0.0,
            original_score_dist=(baseline_mean, baseline_std),
            updated_score_dist=(sim_mean, sim_std),
            confidence_interval=(ci_low, ci_high),
            simulation_runs=self.simulation_runs,
        )

    def update_prediction_with_items(
        self, baseline_mean: float, baseline_std: float, news_items: List[Dict]
    ) -> BayesianUpdateResult:
        """
        Wrapper to handle raw news items and calculate dynamic impact.
        """
        impact_scores = []
        for item in news_items:
            if isinstance(item, dict) and item.get("type") == "injury":
                status = str(item.get("status", "")).lower()
                player = item.get("player", "")

                # Base Impact from Tier
                base_impact = self.get_player_tier_impact(player)

                # Status Multiplier
                if "out" in status:
                    mult = 1.0
                elif "doubtful" in status:
                    mult = 0.75
                elif "questionable" in status:
                    mult = 0.5
                else:
                    mult = 0.2  # Probable/Available

                impact_scores.append(base_impact * mult)

        return self.update_prediction(baseline_mean, baseline_std, impact_scores)

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
