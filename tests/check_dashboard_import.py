import sys
import os

# Add the project root to the python path
sys.path.append(os.getcwd())
# Add the module directory to python path (simulating Streamlit)
sys.path.append(os.path.join(os.getcwd(), "nba_predictive_system"))

try:
    import predictive_analytics_dashboard

    print("Import successful!")

    # Verify src import
    from src.nba_predictor.utils.manual_odds_calculator import ManualOddsCalculator

    print("ManualOddsCalculator import successful!")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
