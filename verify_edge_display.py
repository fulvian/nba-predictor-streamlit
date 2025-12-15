import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

# Add src to path
sys.path.append("/Users/fulvioventura/nba-predictor-streamlit/src")


# Mock UnifiedPredictionResult
@dataclass
class UnifiedPredictionResult:
    predicted_total: float
    confidence: float
    confidence_interval: Tuple[float, float]
    prediction_type: str = "total"

    # Add other required fields with dummy values
    home_score: float = 100.0
    away_score: float = 95.0
    home_win_prob: float = 0.6
    model_agreement: float = 1.0
    feature_importance: Dict[str, float] = None
    risk_assessment: Dict[str, Any] = None
    system_metrics: Dict[str, Any] = None
    shap_plot_data: Dict[str, Any] = None


# Import the class to test
from nba_predictor.streamlit.components.enhanced_prediction_bridge_professional import (
    EnhancedPredictionBridgeProfessional,
)


def test_edge_display():
    bridge = EnhancedPredictionBridgeProfessional()

    # excessive UNDER scenario (Predicted << Line)
    # Predicted: 195.0, Line: 225.0
    # Difference: -30.0
    # Old Edge: -13.3%
    # New Expected: 13.3% (Under)

    prediction = UnifiedPredictionResult(
        predicted_total=195.0, confidence=80.0, confidence_interval=(185.0, 205.0)
    )

    betting_line = 225.0

    result = bridge._analyze_betting_edge(prediction, betting_line)

    print("--- Test Result ---")
    print(f"Predicted Total: {prediction.predicted_total}")
    print(f"Betting Line: {betting_line}")
    print(f"Raw Edge (Points): {result['edge_points']}")
    print(f"Edge Percentage: {result['edge_percentage']}%")
    print(f"Direction: {result['direction']}")
    print(f"Assessment: {result['assessment']}")

    # Assertions
    assert result["edge_points"] == -30.0, "Raw edge points should be -30.0"
    assert result["edge_percentage"] > 0, "Edge percentage should be positive"
    assert abs(result["edge_percentage"] - 13.33) < 0.1, (
        "Edge percentage should be approx 13.33%"
    )
    assert result["direction"] == "Under", "Direction should be Under"
    assert "Under" in result["assessment"], "Assessment should mention Under"
    assert "-" not in f"{result['edge_percentage']:.1f}", (
        "Displayed percentage should not be negative"
    )

    print(
        "\n✅ Verification PASSED: Edge is correctly displayed as absolute % with direction."
    )


if __name__ == "__main__":
    try:
        test_edge_display()
    except AssertionError as e:
        print(f"\n❌ Verification FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ formatting error: {e}")
        sys.exit(1)
