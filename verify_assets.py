import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from nba_predictor.streamlit import assets

print("--- ICON_CALENDAR ---")
print(assets.ICON_CALENDAR)
print("---------------------")
