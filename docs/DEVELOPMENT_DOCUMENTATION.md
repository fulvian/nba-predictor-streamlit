# 📘 NBA Predictor - System Development Documentation

> **Version:** 2.1.0
> **Last Updated:** 2025-12-03
> **Status:** Production Ready
> **Author:** Antigravity (Google Deepmind) & User

---

## 1. Executive Summary

The **NBA Predictor** is a professional-grade predictive analytics system designed to forecast NBA game outcomes with high precision. It leverages a hybrid architecture that combines:
1.  **Real-Time Data Integration:** Fetches live game data, player stats, and odds from multiple external APIs.
2.  **Advanced Machine Learning:** Utilizes an ensemble of XGBoost and Neural Networks, enhanced by a "Unified Hybrid Pipeline" that merges rule-based expert systems with data-driven ML models.
3.  **Value Betting (EV+):** Identifies profitable betting opportunities using **Fractional Kelly Criterion** and strict safety filters (Edge > 2.5%, Confidence > 60%).
4.  **Dynamic Bayesian Updates:** Adjusts predictions in real-time based on breaking news (e.g., injuries) using **Monte Carlo simulations** and Bayesian inference.
5.  **Interactive Dashboard:** A Streamlit-based UI ("WIC Dashboard") that guides the user through a structured workflow: *Update -> Schedule -> Predict -> Analyze -> Trade -> Portfolio*.

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
        API4[Twitter/X API (News)]
    end

    subgraph "Data Ingestion & Storage"
        DP[NBADataProvider] -->|Fetch & Normalize| UDS[UnifiedDataStore]
        NA[NewsAggregator] -->|Fetch News| UDS
        UDS -->|Store| DB[(DuckDB / Parquet)]
        UDS -->|Cache| Redis[Redis Cache]
    end

    subgraph "Core Logic & ML"
        UHP[Unified Hybrid Pipeline]
        ANPE[Advanced Prediction Engine]
        NEP[NBA Ensemble Predictor]
        EVC[EV Calculator]
        BU[Bayesian Updater]
        
        UDS --> UHP
        UHP --> ANPE
        UHP --> NEP
        UHP --> EVC
        UHP --> BU
        ANPE -->|Rule-Based| Pred[Unified Prediction Result]
        NEP -->|ML-Based| Pred
        EVC -->|Value Analysis| Pred
        BU -->|Live Updates| Pred
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
    API4 --> NA
```

### Key Components

*   **UnifiedDataStore (`src/nba_predictor/core/data_store.py`)**: The central data access layer. It abstracts the underlying storage (Parquet files queried via DuckDB) and provides a unified API for retrieving games, stats, and odds.
*   **UnifiedHybridPipeline (`src/nba_predictor/core/unified_hybrid_pipeline.py`)**: The orchestrator that combines data, ML models, and analytical components. It integrates `EVCalculator` and `BayesianUpdater` into the prediction flow.
*   **EVCalculator (`src/nba_predictor/analytics/ev_calculator.py`)**: Calculates Expected Value (EV) and recommends stake sizes using a **Fractional Kelly (0.25x)** strategy. It applies safety filters to ensure only high-quality bets are recommended.
*   **BayesianUpdater (`src/nba_predictor/intelligence/bayesian_updater.py`)**: Dynamically adjusts predictions based on real-time news. It uses **Monte Carlo simulations (5000 runs)** to model the impact of injuries and other factors on the score distribution.
*   **NewsAggregator (`src/nba_predictor/intelligence/news_aggregator.py`)**: Aggregates real-time news from sources like Twitter/X and The Odds API to feed the Bayesian Updater.
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

### Machine Learning & Analytics
*   **Scikit-Learn:** Base ML algorithms and preprocessing (StandardScaler, etc.).
*   **XGBoost:** Gradient boosting framework for high-performance classification/regression.
*   **SciPy / NumPy:** Statistical functions (Normal distribution, Z-scores, Monte Carlo simulations).
*   **Bayesian Inference:** Custom logic for probabilistic updates.

### Web & UI
*   **Streamlit:** Rapid application development framework for the dashboard.
*   **Plotly:** Interactive charting (if used in future expansions).

### External APIs
*   **nba_api:** Official NBA stats endpoints.
*   **BallDontLie API:** Free NBA data (players, teams, games).
*   **The Odds API:** Real-time betting odds.
*   **Twitter/X API:** Real-time news (simulated/integrated).

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

### 4.3. Prediction Generation & Analysis
The prediction process is a multi-stage pipeline:
1.  **Base Prediction:** Calculated from Team Metrics and ML models.
2.  **Adjustments:** Situational factors are applied.
3.  **Bayesian Update (New):** If new information (e.g., injury news) is available, a Bayesian update is performed using Monte Carlo simulations to adjust the mean and standard deviation of the prediction.
4.  **EV Analysis (New):** The final probability is compared against the bookmaker's implied probability.
    *   *Edge Calculation:* `(Model Prob - Implied Prob) / Implied Prob`
    *   *Kelly Staking:* `(Edge / Odds - 1)` * Fraction (0.25)
5.  **Validation:** The `UnifiedHybridPipeline` ensures predictions fall within realistic NBA ranges (e.g., Total Score 180-280).

### 4.4. Model Precision
*   **Calibration:** The system uses deterministic adjustments derived from historical averages rather than random sampling, ensuring consistent outputs.
*   **Uncertainty Modeling:** Bayesian updates explicitly model the increase in uncertainty (wider confidence intervals) when late-breaking news occurs.

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
