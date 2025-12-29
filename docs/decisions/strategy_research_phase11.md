
# 🧠 Phase 11: Competitive Strategy Research & Validation

**Date**: Dec 29, 2025
**Source**: Perplexity Deep Research & Context7 (Betfair API)

## 1. Validation of Current Strategies
### A. "The Denver Lung" 🏔️
*   **Verdict**: **VALIDATED (High Potential)**.
*   **Why**: Historical data confirms Denver/Utah Home Advantage is statistically significant (+3-4 pts vs spread), especially against B2B opponents.
*   **Optimization Required**:
    *   **Health Check**: Ensure key stars (Jokic/Murray) are active.
    *   **Market Confirmation**: Filter for **Home ML Handle > 59%** (Smart money backing the altitude edge).
    *   **Timing**: Best entry is **Halftime** or **Q3 Break**, not pre-game (better CLV).

### B. "Tired Frontrunner" 🥱
*   **Verdict**: **VALIDATED (Medium Potential)**.
*   **Optimization Required**:
    *   **Pace Filter**: Only fade if **Game Pace > 110 possessions** (High pace exacerbates fatigue).
    *   **Odds Drift**: Wait for live odds to drift **> 5%** towards the underdog before entering (CONFIRMATION).

---

## 2. New Proposed Strategies (Data-Driven)
### C. "Pace Surge Fade" ⚡
*   **Concept**: Fatigue kills defense first. High pace + Fatigue = Collapse.
*   **Trigger**:
    *   Home Team Density >= 2 (3-in-4).
    *   **Live Pace > 110** (Running game).
    *   Status: Leading by > 8 pts in Q3.
*   **Action**: **Bet Under (Total)** or **Fade Home Spread** in Q4.
*   **Why**: Tired teams stop scoring efficiently late in fast games.

### D. "Micro-Panic Reversal" 🔴 (The "Live Panic")
*   **Concept**: Algorithmic overreaction to short-term variance (e.g., 0-8 run).
*   **Trigger**:
    *   **Odds Drift**: Price moves > **10 ticks within 3 seconds**.
    *   **Volume Spike**: Traded Volume > **$1k within 5 seconds**.
    *   **Score State**: Favorite is still winning or tied.
*   **Action**: **Back the Drifter** (The team whose odds worsened too fast).
*   **Tech Stack**: Requires **WebSocket Streaming** (Polling is too slow).

---

## 3. Technical Constraints (Betfair API)
*   **Polling (API-NG)**: Limited to 5 requests/second (200ms spacing). **Too slow for "Micro-Panic".**
*   **Streaming (Exchange Stream)**: **NO Request Limits**. Push-based. Latency < 100ms.
    *   **Decision**: Must use `MarketStreamer` (WebSocket) for Strategy D (Panic).
    *   Strategies A, B, C can use Polling (60s is fine).

---

## 4. Recommendation for Implementation
1.  **Hybrid Architecture**:
    *   **Streamer Service**: Dedicato solo al monitoraggio "Micro-Panic" (Strategy D).
    *   **Poller Service**: Continua a gestire le strategie "Macro" (A, B, C) ogni 60s.
2.  **Dashboard Update**:
    *   Aggiungere indicatore "Live Pace" (per Strategy C).
    *   Aggiungere indicatore "Volume Spike" (per Strategy D).
