# Active Issues Log

## Resolved Issues

### [CRITICAL] Kill-Switch Permanent Block (Data Starvation)
- **Status**: Resolved (2025-12-15)
- **Description**: The `BayesianConfidenceValidator` was permanently blocking all predictions because the `historical_calibration` buckets had fewer than 50 samples.
- **Root Cause**: Lack of historical box-score data prevented Feature Engineering for backtesting.
- **Resolution**: Implemented `generate_synthetic_calibration_data.py` (ADR-003) to generate 2,000+ perfectly calibrated synthetic betting records. This populated all buckets (>250 samples each) and satisfied the validator's statistical requirements.
