# ADR-002: Probabilistic Pipeline Redesign & Advanced Feature Integration

## Status
Accepted

## Context
The previous prediction pipeline relied on raw model confidence scores which were often miscalibrated (overconfident). Furthermore, the system lacked team-specific bias correction and a safety mechanism to prevent betting when statistical evidence was insufficient. The feature set was also limited, missing dynamic factors like Pace, Rest Days, and Recent Form.

## Decision
We redesigned the `UnifiedHybridPipeline` to incorporate a "Safety-First" Probabilistic Architecture.

### 1. Advanced Feature Engineering (Pillar 1)
- **Component**: `AdvancedFeatureEngine`
- **Logic**: Introduces dynamic features computed at runtime:
  - `Pace`: Game speed estimation to correlate with Total Points.
  - `Rest Days`: Fatigue factor based on schedule.
  - `Recent Form (L5)`: Rolling 5-game performance metrics (Offensive/Defensive Rating).
  - `Defensive Rating`: Team defensive efficiency.
- **Integration**: Injected into both Training and Inference data pipelines.

### 2. Probability Calibration (Pillar 2)
- **Component**: `PlattCalibrator`
- **Method**: Platt Scikit-learn Logistic Regression.
- **Configuration**:
  - `regularization_strength (C)`: 0.01 (Aggressive regularization for small N=40 dataset).
  - `class_weight`: 'balanced' to handle outcome imbalance.
- **Outcome**: Converts raw score (0-1) into a true probability (ECE reduced by 36% in tests).

### 3. Team-Specific Bias Correction (Pillar 3)
- **Component**: `FeedbackLoop`
- **Logic**: Tracks historical prediction errors per team using Exponential Moving Average (EMA).
  - `alpha`: 0.4 (High responsiveness to recent changes).
- **Usage**: Injected into the Consensus LLM prompt to adjust qualitative analysis (e.g., "Team X consistently underperforms model expectations by 3.5 points").

### 4. Bayesian Kill-Switch (Pillar 4)
- **Component**: `BayesianConfidenceValidator`
- **Logic**: A "Circuit Breaker" that vetoes bets if the calibrated probability bin has insufficient historical samples.
  - `min_samples`: 50 (Strict threshold).
  - `max_bucket_ece`: 0.15 (Maximum allowed calibration error).
- **Impact**: Automatically sets `bet_recommendation` to "PASS" if validation fails, protecting the bankroll from "hallucinated" confidence.

## Consequences

### Positive
- **Safety**: System now defaults to "NO BET" when uncertain, preventing losses on low-confidence/low-data predictions.
- **Accuracy**: Probability scores now reflect real-world win rates (Test ECE 0.05).
- **Adaptability**: Team bias loop allows the system to "learn" from recent streaks faster than retraining the core model.

### Negative
- **Complexity**: Pipeline inference flow is more complex (Data -> Features -> Model -> Calibration -> Validator -> Consensus).
- **Data Hunger**: The Kill-Switch requires a substantial history of settled bets (>50 per bin) to "unlock" trading, potentially reducing volume initially.

## Verification
- **Backtest**: `src/nba_predictor/evaluation/backtest_redesign.py` passed (Kill-Switch active).
- **Unit Tests**: `verify_pillar1_3.py` passed.
