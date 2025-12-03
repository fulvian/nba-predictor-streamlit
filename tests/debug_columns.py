import sys
import os
from datetime import date
import pandas as pd

# Add the project root to the python path
sys.path.append(os.getcwd())
# Add the module directory to python path
sys.path.append(os.path.join(os.getcwd(), "nba_predictive_system"))

from nba_predictive_system.unified_nba_data_pipeline import UnifiedNBADataPipeline


def debug_pipeline_output():
    pipeline = UnifiedNBADataPipeline()
    today = date.today()

    try:
        data = pipeline.fetch_all_data(
            date_range=(today, today), include_boxscores=False
        )
        games_df = data.get("games")

        if games_df is not None and not games_df.empty:
            print("FULL COLUMNS LIST:")
            print(list(games_df.columns))
        else:
            print("No games found for today.")

    except Exception as e:
        print(f"Error fetching data: {e}")


if __name__ == "__main__":
    debug_pipeline_output()
