"""
Unit tests for NewsIntelligence
"""

import pytest
from src.nba_predictor.intelligence.news_aggregator import NewsAggregator


class TestNewsAggregator:
    def setup_method(self):
        self.aggregator = NewsAggregator()

    def test_parse_injury_report_out(self):
        text = "LeBron James is OUT tonight vs Warriors"
        result = self.aggregator.parse_injury_report(text)

        assert result is not None
        assert result["status"] == "OUT"
        assert result["type"] == "injury"

    def test_parse_no_injury(self):
        text = "Lakers are looking to bounce back tonight"
        result = self.aggregator.parse_injury_report(text)

        assert result is None
