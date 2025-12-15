# Unified Hybrid Pipeline Architecture

## Overview
The `UnifiedHybridPipeline` is the core orchestration engine of the NBA Predictor, merging quantitative analysis from the Random Forest model with qualitative "Consensus" from the LLM agent constellation.

## Key Update (v2.1.0)
Version 2.1.0 introduces a **Probabilistic Layer** to the pipeline, transforming it from a raw prediction engine into a calibrated decision support system.

### Components
1.  **Quantitative Model (Random Forest)**: Provide baseline point predictions and raw confidence.
2.  **Advanced Feature Engine**: Injects context (Pace, Rest, Form) into the data flow.
3.  **Probability Calibrator (Platt Scaling)**:
    - **Purpose**: Transforms raw RF confidence (often overconfident) into true probabilities.
    - **Logic**: Logistic Regression (`C=0.01`, class_weight='balanced') mapping score -> probability.
    - **Metric**: Achieves Test ECE of ~0.050.
4.  **Bayesian Confidence Validator (Kill-Switch)**:
    - **Purpose**: Bankroll protection.
    - **Logic**: Calculates Credible Intervals for the calibrated probability.
    - **Action**: Vetoes ("SKIP") any bet where N < 50 or uncertainty is too high.
5.  **Feedback Loop (Bias Correction)**:
    - **Purpose**: Auto-corrects team-specific errors.
    - **Logic**: EMA (alpha=0.4) on past errors injects prompts into the LLM Consensus.
6.  **Consensus Engine (LLM)**:
    - **Purpose**: Strategic reasoning and final adjustment.
    - **Input**: Unified features + Bias Correction Context + Calibrated Confidence.

## Data Flow
```mermaid
graph TD
    A[Raw Data] --> B[Feature Engineering]
    B --> C[Quant Model (RF)]
    C --> D[Platt Calibrator]
    D --> E{Bayesian Validator}
    E -- Veto (N<50) --> F[Block Bet (KILL-SWITCH)]
    E -- Pass --> G[Unified Prediction]
    G --> H[Feedback Loop]
    H --> I[Consensus Engine]
    I --> J[Final Dashboard Output]
```

## Usage
The pipeline is initialized via `pipeline = UnifiedHybridPipeline()` which automatically loads the `probability_calibrator.pkl` model and initializes the sub-components.
