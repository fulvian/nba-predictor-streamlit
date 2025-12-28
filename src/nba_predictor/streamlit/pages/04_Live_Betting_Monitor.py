import streamlit as st
import time
import pandas as pd
import sys
import os

# Add project root
sys.path.append(os.getcwd())

from src.nba_predictor.live.monitor import LiveMonitor
from src.nba_predictor.live.engine import StrategyEngine

st.set_page_config(page_title="NBA Live Alpha Monitor", page_icon="📡", layout="wide")

st.title("📡 Live Betting Alpha Monitor")
st.markdown("""
**Macro-Live System (Track B)**: Detects "Denver Lung" (Altitude) and "Tired Legs" (Fatigue) opportunities at Quarter Breaks.
""")

# Initialize Session State
if "monitor" not in st.session_state:
    st.session_state["monitor"] = LiveMonitor()
if "engine" not in st.session_state:
    st.session_state["engine"] = StrategyEngine()
if "live_alerts" not in st.session_state:
    st.session_state["live_alerts"] = []

# Sidebar Controls
polling_active = st.sidebar.checkbox("Active Polling (60s)", value=False)
manual_refresh = st.sidebar.button("Manual Refresh")


# Helper to Tag Context
def get_game_tags(g):
    tags = []
    # Home Tags
    h_ctx = g.get("home_context", {})
    if h_ctx.get("is_high_altitude", 0) == 1:
        tags.append("🏔️ Altitude Home")
    if h_ctx.get("density_4d", 0) >= 2:
        tags.append("🥱 Tired Home")

    # Away Tags
    a_ctx = g.get("away_context", {})
    if a_ctx.get("rest_days", 99) <= 1:
        tags.append("⚠️ B2B Away")

    return " ".join(tags)


# Main Loop
if polling_active or manual_refresh:
    with st.spinner("Fetching Live Data & Context..."):
        monitor = st.session_state["monitor"]
        engine = st.session_state["engine"]

        # 1. Fetch
        games = monitor.fetch_current_state()

        # 2. Evaluate
        new_alerts = engine.evaluate(games)

        # Merge Alerts (Simple append for now, better dedup needed later)
        if new_alerts:
            # Simple dedup by message + timestamp roughly?
            # Ideally we keep a persistent log.
            for a in new_alerts:
                st.toast(f"🚨 {a.message}", icon="💰")

            st.session_state["live_alerts"].extend(new_alerts)

        # 3. Display Main Table
        if not games:
            st.info(
                "No games currently live or scheduled found via API (or API Error)."
            )
        else:
            # Transform for Display
            display_rows = []
            for g in games:
                status_map = {1: "Scheduled", 2: "🔴 LIVE", 3: "Final"}
                row = {
                    "Status": status_map.get(g["status"], "Unknown"),
                    "Clock": f"Q{g['period']} {g['clock']}",
                    "Matchup": f"{g['away_team']} @ {g['home_team']}",
                    "Score": f"{g['away_score']} - {g['home_score']}",
                    "Alpha Tags": get_game_tags(g),
                    "Home Context": f"Rest: {g.get('home_context', {}).get('rest_days', '?')} | Dens: {g.get('home_context', {}).get('density_4d', '?')}",
                    "Away Context": f"Rest: {g.get('away_context', {}).get('rest_days', '?')} | Dens: {g.get('away_context', {}).get('density_4d', '?')}",
                }
                display_rows.append(row)

            st.dataframe(pd.DataFrame(display_rows), use_container_width=True)

        # 4. Display Alerts Log
        st.subheader("🚨 Generated Alerts Log")
        if st.session_state["live_alerts"]:
            alert_df = pd.DataFrame([vars(a) for a in st.session_state["live_alerts"]])
            # Sort by timestamp desc
            alert_df = alert_df.sort_values("timestamp", ascending=False)
            st.dataframe(alert_df, use_container_width=True)
        else:
            st.write("No active alerts generated yet.")

    if polling_active:
        time.sleep(60)
        st.rerun()

else:
    st.info("Polling Paused. Enable in Sidebar or Click Refresh.")
