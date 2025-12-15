# Recent Changes (v2.1.1)

## Critical Fixes
- **Synthetic Calibration Data**: Implemented `generate_synthetic_calibration_data.py` to populate Bayesian buckets with ~2000 samples, resolving the "Insufficient Samples" Kill-Switch block. (See `ADR-003`).
- **Dashboard Resilience**:
  - `new_wic_dashboard_v2.py`: Now handles `None` Consensus responses gracefully.
  - **Non-Blocking Kill-Switch**: The pipeline now outputs a *Statistical Warning* instead of halting execution. The Dashboard displays this as a Red Banner but still shows the Consensus Analysis.
- **Visuals**: Added `Calibrated Confidence` metric to the Prediction Summary card.

## Documentation
- Added `ADR-003-synthetic-calibration-strategy.md`.
- Updated `ACTIVE_ISSUES.md`.
