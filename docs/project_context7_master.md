# Context7 Project Master: NBA Live Betting & Predictor

## 1. System Overview
**Project Name**: NBA Predictor / Live Betting Monitor
**Architecture**: Hybrid Python Monorepo
- **Backend (Core)**: Pure Python Service (`src/nba_predictor`)
  - `BetfairService`: Threaded singleton managing Betfair API streams.
  - `DataStore`: DuckDB/Polars for historical data.
- **Frontend (Legacy)**: Streamlit (`04_Live_Betting_Monitor.py`) - Deprecated.
- **Frontend (Project NEON)**: Reflex (`ui_reflex/`) - Production UI.
  - **Stack**: Reflex (Python -> React/FastAPI).
  - **Style**: Cyberpunk/Bloomberg aesthetics via Tailwind/CSS.
  - **State**: `ui_reflex/neon_dashboard/state.py` proxies the backend singleton.

## 2. Key Components

### A. Betfair Service (`src/nba_predictor/live/betfair_service.py`)
- **Role**: The heart of the live system.
- **Mechanism**: Connects to Betfair Stream API.
- **Data Flow**: Stream -> `market_cache` -> `get_live_dashboard_data()`.
- **Concurrency**: Thread-safe with internal locks. Runs on a separate thread via `threading.Thread`.

### B. Reflex Frontend (`ui_reflex/`)
- **Config**: `rxconfig.py`.
- **State (`state.py`)**:
    - Proxies `BetfairService` via `get_service()`.
    - Polling Loop: `update_tick` (Background Task) runs every 1s to fetch data from the service Singleton.
    - **Why?**: keeps the UI reactive without blocking the API stream.
- **UI (`neon_dashboard.py`)**:
    - **Grid System**: Responsive layout (HUD, Left Panel, Right Panel).
    - **Styling (`styles.py`)**: Centralized palette (Neon Cyan `#66fcf1`, Deep Carbon `#0b0c10`).
    - **Components**: Custom `metric_card`, `market_row`, `odds_row`.

## 3. Deployment & Startup
- **Launcher**: `startup.py` (Root).
- **Process**:
    1. Checks dependencies (`reflex`, `duckdb`).
    2. Kills zombies (`pkill -f reflex`).
    3. Changes CWD to `ui_reflex/`.
    4. Runs `reflex run`.
- **Ports**:
    - Frontend: 3000
    - Backend: 8000

## 4. Architectural Decisions
- **ADR 001**: Hybrid Bitemporal Architecture (ETL).
- **ADR 002**: Frontend Migration to Reflex (UI Overhaul).
    - **Reasoning**: Streamlit lacked styling control for "Pro" dark themes and had high latency. Reflex offers native React performance with Python DX.

## 5. Development Rules
- **Formatting**: Black/Ruff.
- **Singleton Pattern**: The `BetfairService` is the Source of Truth. The UI is just a view.
- **Reflex**: Use `rx.background` for ANY blocking IO. Never block the main thread.
