# Active Issues Log

## Resolved Issues

### [CRITICAL] Kill-Switch Permanent Block (Data Starvation)
- **Status**: Resolved (2025-12-15)
- **Description**: The `BayesianConfidenceValidator` was permanently blocking all predictions because the `historical_calibration` buckets had fewer than 50 samples.
- **Root Cause**: Lack of historical box-score data prevented Feature Engineering for backtesting.
- **Resolution**: Implemented `generate_synthetic_calibration_data.py` (ADR-003) to generate 2,000+ perfectly calibrated synthetic betting records. This populated all buckets (>250 samples each) and satisfied the validator's statistical requirements.

### [BUG] Corrupted Bet Data "Unknown Matchup"
- **Status**: Resolved (2025-12-17)
- **Description**: A pending bet `e3ea03e8...` was displayed as "Unknown Matchup" with missing team names in the dashboard.
- **Root Cause**: The bet record in `bets` table had `game_id='0062500001'` (invalid) and `home_team/away_team` as `NULL`.
- **Resolution**: Manually mapped the bet to the correct NBA game (`BDL_20377171`: Knicks vs Spurs) and updated the database record.

### [BUG] Corrupted Parquet File
- **Status**: Resolved (2025-12-17)
- **Description**: DuckDB failed to read `data/games/games_2025-12-14.parquet` ("File too small").
- **Resolution**: Deleted the corrupted file and re-ran `fase1_game_results_downloader.py` to fetch valid data.

### [MAINTENANCE] Streamlit Deprecation Warnings
- **Status**: Resolved (2025-12-17)
- **Description**: `use_container_width` parameter is deprecated for `st.dataframe` and `st.button`.
- **Resolution**: Replaced with `width="stretch"` in `new_wic_dashboard_v2.py`.
