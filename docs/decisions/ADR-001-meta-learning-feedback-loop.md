# ADR-001: Meta-Learning Feedback Loop for Consensus Bias Correction

**Status**: Accepted
**Date**: 2025-12-15
**Authors**: Antigravity Agent, User

## Context
The Consensus Engine (LLM) previously operated statelessly regarding its own performance. It did not "know" if it had been consistently overestimating or underestimating totals for specific teams, potentially repeating the same errors (e.g., failing to account for a team's pace change).

We needed a way to introduce "System 2" thinking where the system analyzes its own past errors and provides corrective feedback to the "System 1" (Consensus Engine) before it makes a new prediction.

## Decision
We decided to implement a **Meta-Learning Feedback Loop** that injects a context-aware "System Notification" into the prompt sent to the Consensus Engine.

### Detailed Design
1.  **Mechanism**: A `FeedbackLoop` component fetches the last 10 settled bets for each team involved in a matchup.
2.  **Bias Metric**: We calculate the prediction error (`Predicted - Actual`) and smooth it using an **Exponential Moving Average (EMA)**.
3.  **Prompt Injection**: If the bias is significant, we prepend/append a mandatory instruction to the prompt: *"You have been Overestimating Team X by Y points. Adjust accordingly."*

### Validated Parameters
Following validation with the Consensus Engine (Moonshot Kimi k2, DeepSeek), we refined the initial parameters to avoid "chasing noise" (overfitting to random variance):
*   **Alpha (EMA)**: `0.25` (Initially proposed 0.4). This provides a smoother trend line, effectively weighing the last ~4 games.
*   **Trigger Threshold**: `5.0 points` (Initially proposed 3.0). Since NBA standard deviation is ~8-10 points, a 3-point bias is often noise. 5.0 points represents a structural deviation worth correcting.

## Consequences
### Positive
*   **Self-Correction**: The system can now adapt to regime changes (e.g., a team playing faster due to a lineup change) faster than the raw stats might indicate.
*   **Context Awareness**: The LLM is made aware of its own blind spots.
*   **Automation**: No manual intervention is required to "tune" the model for specific teams.

### Negative
*   **Complexity**: Adds a rigid dependency on the local database state (`bets` table) and the settlement process (`auto_bet_settlement.py`).
*   **Risk of Oscillation**: If the model over-corrects (e.g., adjusting -10 when the bias is +6), it could create a negative feedback loop. The conservative parameters (Alpha 0.25, Threshold 5.0) mitigate this but do not eliminate it.

## Compliance
This implementation follows the **User Global Rules** for:
*   **NanoGPT Consensus Protocol**: Used `ask_consensus` (Complexity: `hard`) to validate parameters.
*   **Documentation**: architecture doc updated, ADR created.
