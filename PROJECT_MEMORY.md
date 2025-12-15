# Project Memory: nba-predictor-streamlit

## 1. Migration Status
- **Docs Coverage**: ~5% (Bootstrapped via Hybrid Migration)
- **Status**: Active Development. Components are documented upon modification ("Touch-it, Document-it").

## 2. Architecture Overview
- **Tech Stack**:
    - **Frontend**: Streamlit
    - **Data Store**: DuckDB (OLAP), SQLite (Cache), JSON (Bankroll)
    - **AI/ML**: NanoGPT Consensus Engine (Hybrid LLM Reasoning)
    - **Language**: Python 3.9+
- **Key Patterns**:
    - **Unified Hybrid Pipeline**: Combines Quantitative Models + LLM Qualitative Analysis.
    - **Meta-Learning Feedback Loop**: Self-correcting mechanism using EMA of past errors to inject prompts into the Consensus Engine.
    - **Consensus Reasoning**: Aggregates insights from multiple LLM personas (e.g., Kimi, GLM, DeepSeek).

## 3. Critical Gaps
- Most legacy components (scrapers, older models) are undocumented.
- Refer to `docs/architecture/` for newly documented components (e.g., Feedback Loop).
