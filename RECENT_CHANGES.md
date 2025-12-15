# Recent Changes

## [2.1.0] - 2025-12-15
### Feature: Probabilistic Pipeline Redesign & Dashboard Integration
- **Advanced Feature Engineering**: Implemented `AdvancedFeatureEngine` calculating Pace, Rest Days, Recent Form (L10 weighted), and Defensive Rating.
- **Probability Calibration**: Integrated `PlattCalibrator` (C=0.01) to transform raw model confidence into realistic probabilities (Test ECE: 0.050).
- **Safety Mechanism**: Implemented `BayesianConfidenceValidator` ("Kill-Switch") to veto bets with insufficient data (N<50) or high uncertainty, protecting bankroll.
- **Bias Correction**: Integrated `FeedbackLoop` (EMA, alpha=0.4) to auto-correct team-specific biases.
- **Dashboard V2**: Updated `new_wic_dashboard_v2.py` to display:
  - **Calibrated Confidence**: Showing users the "real" probability.
  - **Kill-Switch Alerts**: Red banner preventing bets when safety checks fail.
- **Deployment**: Updated `startup.py` with atomic process termination and clearer logs.
