import sys
from pathlib import Path
from typing import Any, Dict
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import streamlit as st
from unittest.mock import MagicMock

# Mock streamlit functions
st.error = MagicMock()
st.warning = MagicMock()
st.info = MagicMock()
st.markdown = MagicMock()


def mock_columns(spec):
    if isinstance(spec, int):
        return [MagicMock() for _ in range(spec)]
    return [MagicMock() for _ in spec]


st.columns = MagicMock(side_effect=mock_columns)
st.expander = MagicMock()
st.rerun = MagicMock()


# Mock WICState
class MockWICState:
    @staticmethod
    def get_selected_game():
        return {"home_team": "Boston", "away_team": "Lakers"}


# Import the dashboard module
# We need to mock assets and components imports inside the module if they cause issues,
# but let's try importing first.
try:
    from nba_predictor.streamlit import new_wic_dashboard_v2 as dashboard

    dashboard.WICState = MockWICState

    print("✅ Dashboard module imported successfully")
except Exception as e:
    print(f"❌ Failed to import dashboard: {e}")
    sys.exit(1)


def test_render_with_kill_switch():
    print("\n🧪 Testing Render with Kill Switch...")
    prediction = {
        "home_team": "Boston",
        "away_team": "Lakers",
        "unified_prediction": 225.5,
        "raw_quant_prediction": 228.0,
        "confidence": 65.0,
        "calibrated_confidence": 55.0,
        "kill_switch_active": True,
        "kill_switch_reason": "Insufficient samples",
        "consensus_adjustment": -2.5,
        "recommendation": "SKIP",
    }

    try:
        dashboard.render_prediction_summary_v2(prediction)

        # Verify calls
        st.error.assert_called_with(
            "⛔ **BET VETOED BY BAYESIAN KILL-SWITCH**: Insufficient samples"
        )
        print("✅ Kill Switch Banner verified")
        print("✅ Render completed without error")
    except Exception as e:
        print(f"❌ Render failed: {e}")


def test_render_normal_with_calibration():
    print("\n🧪 Testing Render Normal with Calibration...")
    prediction = {
        "home_team": "Boston",
        "away_team": "Lakers",
        "unified_prediction": 225.5,
        "raw_quant_prediction": 225.0,
        "confidence": 75.0,
        "calibrated_confidence": 68.5,
        "kill_switch_active": False,
        "consensus_adjustment": 0.5,
        "recommendation": "BET OVER",
    }

    try:
        dashboard.render_prediction_summary_v2(prediction)
        # We can't easily check the exact HTML string in mock, but ensuring it runs is good step
        print("✅ Render completed without error")
    except Exception as e:
        print(f"❌ Render failed: {e}")


if __name__ == "__main__":
    test_render_with_kill_switch()
    test_render_normal_with_calibration()
