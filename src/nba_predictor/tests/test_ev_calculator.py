"""
Unit tests for EVCalculator
"""

import pytest
from src.nba_predictor.analytics.ev_calculator import EVCalculator


class TestEVCalculator:
    def setup_method(self):
        self.calculator = EVCalculator(
            bankroll=1000.0,
            kelly_fraction=0.25,  # Quarter Kelly
            min_edge=0.025,  # 2.5% edge
            min_model_prob=0.60,  # 60% min confidence
        )

    def test_implied_probability(self):
        # -110 odds -> 52.38%
        assert (
            abs(self.calculator.calculate_implied_probability(-110) - 0.5238) < 0.0001
        )
        # +100 odds -> 50.00%
        assert abs(self.calculator.calculate_implied_probability(100) - 0.5000) < 0.0001
        # +200 odds -> 33.33%
        assert abs(self.calculator.calculate_implied_probability(200) - 0.3333) < 0.0001

    def test_decimal_odds(self):
        # -110 -> 1.909
        assert abs(self.calculator.calculate_decimal_odds(-110) - 1.9090) < 0.0001
        # +100 -> 2.00
        assert abs(self.calculator.calculate_decimal_odds(100) - 2.0000) < 0.0001

    def test_value_bet_identification(self):
        # Scenario: Model says 65% win, Odds are +100 (50% implied)
        # Edge = 15% (well above 2.5%)
        # Model Prob = 65% (above 60%)
        result = self.calculator.calculate_ev(model_prob=0.65, american_odds=100)

        assert result.is_value_bet is True
        assert result.edge > 10.0
        assert result.recommended_stake_amount > 0

    def test_safety_filter_low_edge(self):
        # Scenario: Model says 53% win, Odds are -110 (52.38% implied)
        # Edge = ~0.6% (below 2.5%)
        result = self.calculator.calculate_ev(model_prob=0.53, american_odds=-110)

        assert result.is_value_bet is False
        assert "Edge" in result.reason
        assert result.recommended_stake_amount == 0

    def test_safety_filter_low_confidence(self):
        # Scenario: Model says 55% win, Odds are +150 (40% implied)
        # Edge = 15% (High edge)
        # BUT Model Prob = 55% (below 60% threshold)
        result = self.calculator.calculate_ev(model_prob=0.55, american_odds=150)

        assert result.is_value_bet is False
        assert "Model confidence" in result.reason
        assert result.recommended_stake_amount == 0

    def test_kelly_stake_calculation(self):
        # Scenario:
        # Bankroll = 1000
        # Model Prob (p) = 0.65
        # Odds = +100 -> b = 1.0
        # q = 0.35
        # Full Kelly = (1.0 * 0.65 - 0.35) / 1.0 = 0.30 (30%)
        # Quarter Kelly = 0.30 * 0.25 = 0.075 (7.5%)
        # Stake = 1000 * 0.075 = 75.0

        result = self.calculator.calculate_ev(model_prob=0.65, american_odds=100)

        assert abs(result.kelly_stake_percentage - 7.5) < 0.1
        assert abs(result.recommended_stake_amount - 75.0) < 1.0
