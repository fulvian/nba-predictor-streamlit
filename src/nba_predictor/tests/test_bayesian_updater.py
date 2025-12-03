"""
Unit tests for BayesianUpdater
"""

import pytest
import numpy as np
from src.nba_predictor.intelligence.bayesian_updater import BayesianUpdater


class TestBayesianUpdater:
    def setup_method(self):
        self.updater = BayesianUpdater(simulation_runs=5000)

    def test_impact_mapping(self):
        # Impact 1.0 -> -1.5
        assert self.updater.map_impact_to_likelihood(1.0) == -1.5
        # Impact 2.0 -> -3.0
        assert self.updater.map_impact_to_likelihood(2.0) == -3.0

    def test_update_prediction_score_drop(self):
        # Scenario: Team projected 110 pts, Star player (Impact 2.0) out
        baseline_mean = 110.0
        baseline_std = 10.0
        injuries = [2.0]  # Star player out

        result = self.updater.update_prediction(baseline_mean, baseline_std, injuries)

        # Mean should drop by approx 3.0 points
        assert result.updated_score_dist[0] < baseline_mean
        assert abs(result.updated_score_dist[0] - 107.0) < 0.5  # Allow for MC noise

        # Uncertainty should increase
        assert result.updated_score_dist[1] > baseline_std

    def test_update_win_probability(self):
        # Scenario: Team A (110, 10) vs Team B (108, 10)
        # Original: Team A favored
        # News: Team A Star (Impact 2.0) OUT

        team_mean = 110.0
        team_std = 10.0
        opp_mean = 108.0
        opp_std = 10.0
        injuries = [2.0]

        result = self.updater.update_win_probability(
            team_mean, team_std, opp_mean, opp_std, injuries
        )

        # Original prob should be > 50%
        assert result.original_prob > 0.5

        # Updated prob should be lower than original
        assert result.updated_prob < result.original_prob

        # With -3.0 adjustment, Team A becomes 107 vs 108, so prob should be < 50%
        assert result.updated_prob < 0.5

    def test_simulation_stability(self):
        # Run twice with same inputs, results should be very close (stable)
        baseline_mean = 110.0
        baseline_std = 10.0
        injuries = [1.0]

        res1 = self.updater.update_prediction(baseline_mean, baseline_std, injuries)
        res2 = self.updater.update_prediction(baseline_mean, baseline_std, injuries)

        # Means should be within small margin
        # Relaxed threshold to 0.5 to account for Monte Carlo variance
        assert abs(res1.updated_score_dist[0] - res2.updated_score_dist[0]) < 0.5
