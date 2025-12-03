import sys
from pathlib import Path
import traceback

# Setup path as in the bridge
project_root = Path("/Users/fulvioventura/nba-predictor-streamlit")
sys.path.append(str(project_root))
sys.path.append(str(project_root / "nba_predictive_system"))

print(f"Python path: {sys.path}")

try:
    print("Attempting to import advanced_nba_prediction_engine...")
    import advanced_nba_prediction_engine

    print("✅ Import SUCCESS!")
    print(f"Module: {advanced_nba_prediction_engine}")
except Exception:
    print("❌ Import FAILED")
    traceback.print_exc()
