
# 💎 Betfair Streamlit Integration Plan (Best Practices)

To integrate the stateful, real-time `MarketStreamer` + `PanicDetector` into the stateless Streamlit app, we will use the **Singleton Service Pattern** with `st.cache_resource`.

## 1. Architecture: The "BetfairService" Singleton
We need a persistent python object that survives Streamlit re-runs.
*   **Component**: `src/nba_predictor/live/betfair_service.py`
*   **Mechanism**: `@st.cache_resource`
*   **Responsibility**:
    1.  Initialize `BetfairClient` (Auth).
    2.  Manage background thread for `MarketStreamer`.
    3.  Hold a thread-safe `Queue` of Alerts/Signals.
    4.  Provide a `get_alerts()` method for the UI to consume non-blockingly.

## 2. UI Integration: The "Live Monitor" Page
The Streamlit page will check this service for updates.
*   **Page**: `04_Live_Betting_Monitor.py`
*   **Logic**:
    *   On load -> `service = get_betfair_service()`
    *   On button "Start Monitoring" -> `service.start_streaming()`
    *   **Loop**: Use `st.empty()` or `st.fragment` (Streamlit >1.37) for a micro-poll loop (every 1s) to fetch new alerts without reloading the whole page.

## 3. Data Flow
1.  **WebSocket Thread**: Pushes data to `PanicDetector`.
2.  **PanicDetector**: Pushes `PanicAlert` objects to `service.alert_queue`.
3.  **Streamlit Main Thread**: Calls `service.get_latest_alerts()` every second.
4.  **UI**: Appends new alerts to `st.session_state.alerts` and renders them.

## 4. Implementation Steps
1.  [ ] Create `BetfairService` class (Thread-Safe Wrapper).
2.  [ ] Update `04_Live_Betting_Monitor.py` to use `st.cache_resource`.
3.  [ ] Add "Connect/Disconnect" controls and "Live Alerts" feed.

## Why this is Robust?
*   **No Freezing**: `MarketStreamer` runs in its own thread (already implemented).
*   **Persistence**: `st.cache_resource` prevents reconnection on every UI interaction.
*   **Efficiency**: Queue-based consumption means we process only new events.
