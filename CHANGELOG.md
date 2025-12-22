# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.2.1] - 2025-12-22

### Fixed

- **Schema Mismatch Error in Dashboard**: Resolved `DATABASE_ERROR: could not append value "7:30 pm ET"` by implementing explicit type casting in DuckDB queries using `::VARCHAR` syntax. This handles Parquet files with heterogeneous schemas (mixed Null/String types for `game_time` column).
- **DuckDB-Polars Conversion**: Replaced `fetchall()` + `pl.DataFrame()` with native `.pl()` method for direct Arrow-based conversion, preventing schema inference errors.
- **Streamlit Button API**: Fixed deprecated `width="stretch"` parameter in `st.button()`, replaced with `use_container_width=True`.

### Technical Details

- Root cause: File `games_2025-12-22.parquet` had `game_time` as String type while other files had Null type
- Solution: Explicit column selection with `::VARCHAR` cast during `read_parquet` query
- Files modified: `data_store.py`, `new_wic_dashboard_v2.py`

## [2.2.0] - 2025-12-15

### Added

- **Multi-Persona Consensus System (v2.0)**: Complete reimplementation of NBA betting prompt with 4 specialized agents:
  - **The Quant**: Pure statistical analysis, implied probability calculation, statistical edge identification
  - **The Scout**: Narrative analysis, injury impact beyond box score, coaching strategy, motivation factors
  - **The Bookmaker**: Market efficiency analysis, uncertainty decomposition (aleatoric vs epistemic), EV calculation
  - **The Moderator**: Bayesian fusion with weighted synthesis (Quant 40%, Scout 30%, Bookmaker 30%)
- **Chain-of-Thought Reasoning**: Each persona explicitly reasons step-by-step before concluding
- **New Output Fields**:
  - `persona_votes`: Individual adjustments from each persona (quant, scout, bookmaker)
  - `ev_assessment`: Expected Value classification (POSITIVE/NEGATIVE/NEUTRAL)
- **Verification Scripts**: Added `verify_real_consensus_fix.py` for testing consensus integration

### Changed

- **Timeout Configuration**: Increased from 180s → 600s (10 min) for thinking models compatibility
- **Token Limits**: Increased max_tokens from 2048 → 16384 in NanoGPT-Consensus-MCP

### Fixed

- **402 Payment Required**: Resolved by removing incompatible memory headers for Flat Plan
- **Timeout Race Condition**: Fixed hardcoded timeout in `unified_hybrid_pipeline.py` that was overriding client default

## [2.1.1] - 2025-12-15

### Fixed

- **Kill-Switch Deadlock**: Resolved persistent "Insufficient Samples" block by implementing Synthetic Calibration Data generation (ADR-003).
- **Dashboard Crash**: Fixed `NoneType` error in dashboard when Consensus Analysis is missing.
- **UI UX**: Added "Calibrated Confidence" display and non-blocking Warning Banner in Dashboard V2.

## [2.1.0] - 2025-12-15

### Added

- **UnifiedHybridPipeline**: Complete probabilistic redesign (ADR-002).
  - **Advanced Feature Engineering**: Pace, Rest Days, Recent Form, Defensive Rating.
  - **Probability Calibration**: `PlattCalibrator` (Logistic Regression, C=0.01) significantly reduces overconfidence (Test ECE: 0.050).
  - **Bayesian Kill-Switch**: `BayesianConfidenceValidator` protects bankroll by vetoing safe bets with insufficient sample size (N<50) or high uncertainty.
  - **Team-Specific Bias Correction**: `FeedbackLoop` (EMA, alpha=0.4) automatically corrects system bias for specific teams.
- **Dashboard V2 Updates**:
  - **Visual Calibration**: Display of "Calibrated Confidence" vs "Raw Confidence".
  - **Safety Banners**: "BET VETOED" alerts when the Kill-Switch is active.
- **Documentation**: Added ADR-002, updated Walkthrough.

### Changed

- **Startup Script**: Updated `startup.py` to use safe, 2-stage process termination (SIGTERM -> SIGKILL).
- **Pipeline Logic**: Moved from static confidence thresholds to dynamic, calibrated probabilities.

## [2.0.0] - 2024-10-27

### 🚀 Major Features

- **Complete System Refactoring**: Migrated to modern Python `src/` layout structure
- **Unified Data Store**: Implemented Polars + DuckDB + Parquet integration for high-performance analytics
- **Modern API Clients**: Created async HTTP clients for NBA and Odds APIs with HTTPX
- **Automatic Sync Engine**: Added background data synchronization with configurable intervals
- **Modular Streamlit Components**: Redesigned UI with modular navigation and reusable components