# FeedbackLoop Component

## Overview
The `FeedbackLoop` component (`src/nba_predictor/intelligence/feedback_loop.py`) is responsible for implementing the "Meta-Learning" capability of the system. It enables the Consensus Engine to "learn" from its past prediction errors by injecting context-aware bias corrections into the LLM's prompt.

## Responsibilities
1.  **Historical Analysis**: Query the local database (`nba_betting.duckdb`) for the recent betting history of involved teams.
2.  **Bias Calculation**: Compute the prediction bias (Predicted vs. Actual) using an Exponential Moving Average (EMA).
3.  **Prompt Generation**: enhanced "System Notification" prompt if the bias exceeds a defined threshold, instructing the LLM to apply corrections.

## Key Algorithms

### Exponential Moving Average (EMA)
To calculate the structural bias for a team, we use an EMA to give more weight to recent performances while maintaining memory of trends.

-   **Formula**: `EMA_t = (Error_t * Alpha) + (EMA_{t-1} * (1 - Alpha))`
-   **Alpha**: `0.25` (Standard) - Tuned to balance responsiveness and stability (approx. last 4 games weight).
-   **Error**: `Predicted Total - Actual Total`
    *   Positive (+): Overestimation
    *   Negative (-): Underestimation

### Threshold Trigger
Corrections are only generated if the absolute EMA bias exceeds the **Threshold of 5.0 points**. This prevents the system from "chasing noise" or random variance typical in NBA scores.

## Interaction Flow
1.  **Trigger**: `UnifiedHybridPipeline.predict_unified_with_consensus` calls `feedback_loop.generate_correction_prompt(team1, team2)`.
2.  **Data Fetch**: `FeedbackLoop` queries `bets` table for last 10 settled/won/lost games.
3.  **Calculation**: Computes EMA bias for both teams.
4.  **Decision**: Checks if `abs(bias) > 5.0`.
5.  **Output**: Returns a string (the correction prompt) or empty string.
6.  **Injection**: `NanoGPTClient` appends this string to the final Consensus Prompt.

## Dependencies
-   **Database**: Requires `nba_betting.duckdb` with `bets` table containing `prediction` (JSON) and `result` fields.
-   **Statuses**: Relies on bets being marked as `SETTLED`, `WON`, or `LOST`.

## Configuration
Parameters are defined as class constants or default arguments:
-   `Alpha`: 0.25
-   `Threshold`: 5.0
