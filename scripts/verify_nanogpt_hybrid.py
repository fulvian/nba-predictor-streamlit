import sys
import os
import json
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from nba_predictor.intelligence.nanogpt_client import NanoGPTClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_parsing():
    client = NanoGPTClient()

    # Mock Response with Hybrid Data
    mock_response = {
        "results": [
            {
                "model": "model_a",
                "content": '```json\n{\n  "point_adjustment": -3.5,\n  "uncertainty_factor": 0.2,\n  "risk_level": "LOW",\n  "reasoning": "Star out."\n}\n```',
            },
            {
                "model": "model_b",  # Fallback case (no uncertainty)
                "content": '```json\n{\n  "point_adjustment": -2.5,\n  "confidence": 60,\n  "risk_level": "MED",\n  "reasoning": "Uncertain usage."\n}\n```',
            },
        ]
    }

    logger.info("Testing with Hybrid Data...")
    result = client._aggregate_consensus_results(mock_response)

    logger.info(f"Result: {json.dumps(result, indent=2)}")

    # Validation logic
    # Model A: Adj -3.5, Unc 0.2
    # Model B: Adj -2.5, Conf 60 -> Unc = 1 - 0.6 = 0.4
    # Avg Adj: -3.0
    # Avg Unc: (0.2 + 0.4) / 2 = 0.3

    expected_adj = -3.0
    expected_unc = 0.3

    assert abs(result["point_adjustment"] - expected_adj) < 0.01, (
        f"Expected adj {expected_adj}, got {result['point_adjustment']}"
    )
    assert abs(result["uncertainty_factor"] - expected_unc) < 0.01, (
        f"Expected unc {expected_unc}, got {result['uncertainty_factor']}"
    )
    assert result["risk_level"] == "MED", (
        f"Expected High Risk Conservative aggregation to be MED (since one was MED), got {result['risk_level']}"
    )

    logger.info("✅ Hybrid Parsing Test Passed!")


def test_prompt_construction():
    client = NanoGPTClient()
    context = {
        "team1": "Lakers",
        "team2": "Warriors",
        "predicted_total": 230.5,
        "market_line": 233.5,
        "deviation_from_market": -3.0,
        "confidence": "55%",
        "news": [{"type": "injury", "text": "LeBron James is OUT"}],
    }

    prompt = client._construct_prompt(context)
    logger.info("Generated Prompt:")
    print(prompt)

    assert "Uncertainty Factor (Variance)" in prompt, (
        "Prompt missing Uncertainty instruction"
    )
    assert "Market Line: 233.5" in prompt, "Prompt missing Market Line"

    logger.info("✅ Prompt Construction Test Passed!")


if __name__ == "__main__":
    test_parsing()
    test_prompt_construction()
