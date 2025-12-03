# 📘 NBA Predictor - System Development Documentation

> **Version:** 2.0.0
> **Last Updated:** 2025-12-02
> **Status:** Production Ready
> **Author:** Antigravity (Google Deepmind) & User

---

## 1. Executive Summary

The **NBA Predictor** is a professional-grade predictive analytics system designed to forecast NBA game outcomes with high precision. It leverages a hybrid architecture that combines:
1.  **Real-Time Data Integration:** Fetches live game data, player stats, and odds from multiple external APIs.
2.  **Advanced Machine Learning:** Utilizes an ensemble of XGBoost and Neural Networks, enhanced by a "Unified Hybrid Pipeline" that merges rule-based expert systems with data-driven ML models.
3.  **Risk Management:** Implements a sophisticated betting strategy based on the Kelly Criterion and dynamic bankroll management.
4.  **Interactive Dashboard:** A Streamlit-based UI ("WIC Dashboard") that guides the user through a structured workflow: *Update -> Schedule -> Predict -> Analyze -> Trade -> Portfolio*.

The system is built with a "No Compromise" philosophy, ensuring that every prediction is grounded in real, up-to-date data and statistically validated models.

---

## 2. System Architecture

The system follows a modular **Layered Architecture**, separating concerns between data acquisition, processing, prediction, and presentation.

```mermaid
graph TD
    subgraph "External Data Layer"
        API1[NBA API]
        API2[BallDontLie]
        API3[The Odds API]
    end

    subgraph "Data Ingestion & Storage"
        DP[NBADataProvider] -->|Fetch & Normalize| UDS[UnifiedDataStore]
        UDS -->|Store| DB[(DuckDB / Parquet)]
        UDS -->|Cache| Redis[Redis Cache]
    end

    subgraph "Core Logic & ML"
        UHP[Unified Hybrid Pipeline]
        ANPE[Advanced Prediction Engine]
        NEP[NBA Ensemble Predictor]
        
        UDS --> UHP
        UHP --> ANPE
        UHP --> NEP
        ANPE -->|Rule-Based| Pred[Prediction Result]
        NEP -->|ML-Based| Pred
    end

    subgraph "Business Logic"
        LRM[Legacy Risk Manager]
        SBDM[Secure Betting DB Manager]
        Bankroll[Bankroll Manager]
        
        Pred --> LRM
        LRM -->|Stake Calc| Trade[Trade Execution]
        Trade --> SBDM
        Trade --> Bankroll
    end

    subgraph "Presentation Layer"
        WIC[WIC Dashboard (Streamlit)]
        WIC -->|User Input| UHP
        WIC -->|Display| Pred
        WIC -->|Manage| Trade
    end

    API1 --> DP
    API2 --> DP
    API3 --> DP
```

### Key Components

*   **UnifiedDataStore (`src/nba_predictor/core/data_store.py`)**: The central data access layer. It abstracts the underlying storage (Parquet files queried via DuckDB) and provides a unified API for retrieving games, stats, and odds.
*   **NBADataProvider (`src/nba_predictor/api/data_provider.py`)**: Responsible for fetching external data. It handles API rate limits, data normalization, and fallback strategies (e.g., checking local cache before calling APIs).
*   **AdvancedNBAPredictionEngine (`nba_predictive_system/advanced_nba_prediction_engine.py`)**: The core expert system. It calculates team ratings (Offensive/Defensive), pace, and applies deterministic situational adjustments (Back-to-Back, Rest Days, Travel) based on historical averages.
*   **LegacyRiskManager (`src/nba_predictor/utils/legacy_risk_manager.py`)**: Implements the financial logic. It calculates the optimal stake using a modified Kelly Criterion, considering the "Quality Score" of a bet (Edge, Confidence, Odds).
*   **WIC Dashboard (`src/nba_predictor/streamlit/new_wic_dashboard.py`)**: The user interface. It implements the "Workflow Intelligent Control" pattern, guiding the user step-by-step.

---

## 3. Technology Stack

### Core Runtime
*   **Language:** Python 3.10+
*   **Environment Management:** `venv`

### Data & Storage
*   **DuckDB:** High-performance analytical SQL database for querying Parquet files.
*   **Parquet:** Columnar storage format for efficient data compression and retrieval.
*   **Pandas / Polars:** Data manipulation and analysis. Polars is used for high-performance data processing.
*   **JSON:** Lightweight storage for configuration and state (e.g., `bankroll.json`).

### Machine Learning
*   **Scikit-Learn:** Base ML algorithms and preprocessing (StandardScaler, etc.).
*   **XGBoost:** Gradient boosting framework for high-performance classification/regression.
*   **TensorFlow / Keras:** Neural Networks for deep learning components (optional/hybrid).
*   **SciPy:** Statistical functions (Normal distribution, Z-scores).

### Web & UI
*   **Streamlit:** Rapid application development framework for the dashboard.
*   **Plotly:** Interactive charting (if used in future expansions).

### External APIs
*   **nba_api:** Official NBA stats endpoints.
*   **BallDontLie API:** Free NBA data (players, teams, games).
*   **The Odds API:** Real-time betting odds.

---

## 4. Detailed ML Logic & Data Flow

### 4.1. Data Fetching & "The Fletcher"
The system employs a robust data fetching strategy ("The Fletcher"):
1.  **Check Local:** `UnifiedDataStore` checks if data for the requested date exists in local Parquet files.
2.  **Fetch Remote:** If missing, `NBADataProvider` queries external APIs.
3.  **Normalize:** Data is standardized (Team IDs, Date formats) to match the internal schema.
4.  **Persist:** New data is saved to Parquet for future use.

### 4.2. Feature Engineering
The `AdvancedNBAPredictionEngine` generates features dynamically:
*   **Team Metrics:** Offensive Rating, Defensive Rating, Pace, Net Rating.
*   **Situational Factors:**
    *   *Rest Days:* Deterministic penalty/boost based on 0, 1, 2, 3+ days rest.
    *   *Travel:* Fixed penalty for away games involving travel.
    *   *Back-to-Back:* Fixed penalty for playing on consecutive nights.
*   **Matchup Analysis:** Head-to-head history and style contrast (e.g., Fast Pace vs. Slow Pace).

### 4.3. Prediction Generation
The prediction process is deterministic to ensure stability:
1.  **Base Prediction:** Calculated from Team Metrics (Home Offense vs. Away Defense, adjusted for Pace).
2.  **Adjustments:** Situational factors are added/subtracted.
3.  **Confidence Interval:** A 95% confidence interval is generated based on historical standard error.
4.  **Probability Distribution:** Z-scores are calculated against the betting line to determine the probability of Over/Under.

### 4.4. Model Precision
*   **Calibration:** The system uses deterministic adjustments derived from historical averages (e.g., -2.5 points for B2B) rather than random sampling, ensuring consistent outputs.
*   **Validation:** The `UnifiedHybridPipeline` ensures predictions fall within realistic NBA ranges (e.g., Total Score 180-280).

---

## 5. Project Structure

```text
nba-predictor-streamlit/
├── data/                       # Data storage (Parquet, JSON, DuckDB)
│   ├── bankroll.json           # Current bankroll state
│   ├── games/                  # Historical game data
│   └── ...
├── docs/                       # Documentation
├── nba_predictive_system/      # Core ML & Prediction Logic
│   ├── advanced_nba_prediction_engine.py  # MAIN ENGINE
│   └── ...
├── src/
│   └── nba_predictor/
│       ├── api/                # External API handlers
│       ├── core/               # Core data logic (UnifiedDataStore)
│       ├── streamlit/          # Dashboard code
│       │   └── new_wic_dashboard.py  # MAIN ENTRY POINT
│       └── utils/              # Utilities (Risk Manager, DB Manager)
├── tests/                      # Unit and Integration tests
├── requirements.txt            # Python dependencies
└── README.md                   # Quick start guide
```

---

## 6. Setup & Deployment

### Prerequisites
*   Python 3.10+
*   Git

### Installation
1.  **Clone Repository:**
    ```bash
    git clone https://github.com/fulvian/nba-predictor-streamlit.git
    cd nba-predictor-streamlit
    ```
2.  **Create Virtual Environment:**
    ```bash
    python -m venv .venv_new
    source .venv_new/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the System
To launch the WIC Dashboard:
```bash
streamlit run src/nba_predictor/streamlit/new_wic_dashboard.py
```

---

## 7. Best Practices Implemented

*   **Modular Design:** Clear separation of concerns allows for easy maintenance and testing.
*   **Type Hinting:** Extensive use of Python type hints (`typing`) for code clarity and error prevention.
*   **Error Handling:** Robust `try-except` blocks with logging to handle API failures and data issues gracefully.
*   **Configuration Management:** Key parameters (bankroll, API keys) are separated from code.
*   **Documentation:** Comprehensive docstrings and this architecture document ensure knowledge transfer.
*   **Testing:** Integration tests (e.g., `test_data_provider.py`) verify core functionality.
*   **Security:** `SecureBettingDatabaseManager` prevents SQL injection and ensures data integrity.

---

*Generated by Antigravity for User Fulvio Ventura.*
