import logging
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from nba_predictor.utils.legacy_risk_manager import LegacyRiskManager

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_risk_enrichment():
    risk_manager = LegacyRiskManager()

    # Baseline Scenario
    edge = 0.10  # 10% Edge
    prob = 0.55  # 55% Win Prob
    odds = 1.90  # -110 American

    logger.info("--- Testing Enriched Quality Score ---")

    # 1. Standard Score (No LLM info)
    standard_res = risk_manager.calculate_quality_score(edge, prob, odds)
    qs_std = standard_res["quality_score"]
    logger.info(f"Standard QS: {qs_std:.4f}")

    # 2. High Risk Penalty
    high_risk_res = risk_manager.calculate_quality_score(
        edge, prob, odds, llm_risk_level="HIGH", llm_confidence=0.6
    )
    qs_high = high_risk_res["quality_score"]
    logger.info(f"High Risk QS: {qs_high:.4f} (Penalty applied)")

    # 3. Low Risk Boost
    low_risk_res = risk_manager.calculate_quality_score(
        edge, prob, odds, llm_risk_level="LOW", llm_confidence=0.9
    )
    qs_low = low_risk_res["quality_score"]
    logger.info(f"Low Risk QS:  {qs_low:.4f} (Boost applied)")

    # Assertions
    # High Risk should lower the score
    assert qs_high < qs_std, f"High Risk ({qs_high}) should be < Standard ({qs_std})"

    # Low Risk + High Conf should boost the score
    assert qs_low > qs_std, f"Low Risk+Conf ({qs_low}) should be > Standard ({qs_std})"

    # Calculate Impact %
    penalty_pct = (qs_high - qs_std) / qs_std * 100
    boost_pct = (qs_low - qs_std) / qs_std * 100

    logger.info(f"Penalty Impact: {penalty_pct:.1f}%")
    logger.info(f"Boost Impact:   +{boost_pct:.1f}%")

    logger.info("✅ Risk Enrichment Logic Verified Successfully")


if __name__ == "__main__":
    test_risk_enrichment()
