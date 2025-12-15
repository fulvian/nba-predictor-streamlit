### [2025-12-15] Meta-Learning Feedback Loop Integration
- **Implemented `FeedbackLoop`**: New component in `src/nba_predictor/intelligence/feedback_loop.py` to calculate valid EMA bias from `bets` history.
- **Updated `NanoGPTClient`**: Now accepts `meta_learning_context` to inject bias correction prompts into the Consensus Request.
- **Integrated Pipeline**: `UnifiedHybridPipeline` now automatically checks for bias before querying Consensus.
- **Consensus Validation**: Tuned parameters based on reasoning engine feedback:
    - **Alpha**: 0.25 (was 0.4)
    - **Threshold**: 5.0 points (was 3.0)
- **Docs**: Added `docs/architecture/components/feedback_loop.md` and `docs/decisions/ADR-001-meta-learning-feedback-loop.md`.
