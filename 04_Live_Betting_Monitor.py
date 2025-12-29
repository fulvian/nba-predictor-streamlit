import streamlit as st
import time
import pandas as pd
import numpy as np
import os
from datetime import datetime

from src.nba_predictor.live.betfair_service import BetfairService
from src.nba_predictor.betfair.client import BetfairClient

# === PAGE CONFIGURATION ===
st.set_page_config(
    page_title="Antigravity Live Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === CUSTOM CSS (PRO TRADING TERMINAL THEME) ===
st.markdown(
    """
    <style>
        /* [GLOBAL] Main Background & Text */
        .stApp {
            background-color: #0b0c10; /* Deep Carbon */
            color: #c5c6c7; /* Pale Grey Text */
        }
        
        /* [SIDEBAR] Darker & Cleaner */
        section[data-testid="stSidebar"] {
            background-color: #1f2833;
            border-right: 1px solid #45a29e;
        }
        
        /* [CONTRAST FIX] Force Widget Labels to White */
        .stSelectbox label, .stRadio label, .stNumberInput label, .stCheckbox label, .stSlider label {
            color: #66fcf1 !important; /* Neon Blue/Cyan for headers */
            font-weight: 600;
        }
        
        /* [CONTRAST FIX] Radio/Checkbox Option Text */
        div[data-testid="stMarkdownContainer"] p {
            color: #c5c6c7 !important;
        }
        
        /* [METRICS] Professional Cards */
        div[data-testid="stMetric"] {
            background-color: #1f2833;
            border: 1px solid #45a29e; /* Neon Cyan Border */
            padding: 10px;
            border-radius: 4px; /* Sharper corners */
            box-shadow: 0 0 10px rgba(69, 162, 158, 0.2);
        }
        
        /* Metric Label & Value Overrides */
        div[data-testid="stMetricLabel"] {
            color: #45a29e !important; /* Muted Cyan */
        }
        div[data-testid="stMetricValue"] {
            color: #ffffff !important;
        }
        
        /* [DATAFRAME] Dark Mode Hacks for st.dataframe */
        div[data-testid="stDataFrame"] {
            background-color: #1f2833;
            border: 1px solid #333;
        }
        
        /* [BUTTONS] High Visibility Action Buttons */
        div.stButton > button {
            background-color: #45a29e;
            color: #0b0c10;
            font-weight: bold;
            border: none;
            border-radius: 4px;
        }
        div.stButton > button:hover {
            background-color: #66fcf1;
            color: #000;
        }
        
        /* [ALERTS] Console Style */
        .stCode {
            background-color: #000000 !important;
            border: 1px solid #45a29e;
            color: #00ff00 !important; /* Matrix Green */
        }
        
        /* [HEADERS] */
        h1, h2, h3 {
            color: #66fcf1 !important;
            font-family: 'Roboto Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# === SINGLETON SERVICE CACHE ===
@st.cache_resource
def get_service_v3():
    return BetfairService()


service = get_service_v3()


# === HELPER FUNCTIONS ===
def safe_float_format(val):
    if val is None or pd.isna(val):
        return "-"
    return f"{val:.2f}"


def safe_currency_format(val):
    if val is None or pd.isna(val):
        return "-"
    return f"€{val:,.0f}"


def odds_styler(val, type="back"):
    if pd.isna(val):
        return ""
    color = "#007bff" if type == "back" else "#ffb6c1"  # Blue for Back, Pink for Lay
    text_color = "white" if type == "back" else "black"
    return f"background-color: {color}; color: {text_color}; font-weight: bold;"


# === UI LAYOUT ===

# 1. SIDEBAR CONTROLS
with st.sidebar:
    st.markdown("### 🎛️ Control Center")

    # Initialize Client if needed (for finding markets)
    if "bf_client" not in st.session_state:
        st.session_state.bf_client = BetfairClient(
            app_key=os.getenv("BETFAIR_APP_KEY", "QkYxxm82m7tiUQrI"),
            username=os.getenv("BETFAIR_USERNAME", "fulviold@gmail.com"),
            password=os.getenv("BETFAIR_PASSWORD", "9#!Vq-!45ukvu&6"),
            certs_path=os.getenv(
                "BETFAIR_CERTS_PATH",
                "/Users/fulvioventura/nba-predictor-streamlit/certs",
            ),
            locale="italy",
        )
        try:
            st.session_state.bf_client.login()
            st.success("✔ API Connected")
        except Exception as e:
            st.error(f"Login Failed: {e}")

    # Mode Selector
    mode = st.radio("Operation Mode", ["🔭 Single Scout", "🛰️ League Monitor"], index=1)

    st.markdown("---")

    # LEAGUE MONITOR CONTROLS
    if mode == "🛰️ League Monitor":
        st.caption("Auto-discover and stream all live events.")
        if st.button("🔄 Scan Live Basketball"):
            try:
                events = st.session_state.bf_client.list_nba_events()
                if not events:
                    st.warning("No events found.")
                else:
                    event_ids = [e.event.id for e in events][:10]  # Limit for demo
                    cats = st.session_state.bf_client.get_market_catalogue(
                        event_ids, max_results=len(event_ids)
                    )

                    st.session_state.multi_market_ids = [c.market_id for c in cats]
                    st.session_state.multi_market_names = {
                        c.market_id: c.event.name for c in cats
                    }
                    st.success(f"Loaded {len(cats)} Markets")
            except Exception as e:
                st.error(f"Scan Error: {e}")

        if "multi_market_ids" in st.session_state and st.session_state.multi_market_ids:
            if not service.is_running:
                if st.button("▶️ Start League Stream"):
                    try:
                        service.start_monitoring(
                            st.session_state.multi_market_ids,
                            st.session_state.multi_market_names,
                        )
                        st.rerun()
                    except Exception as e:
                        err_msg = str(e)
                        if "MAX_CONNECTION_LIMIT_EXCEEDED" in err_msg:
                            st.error("🚨 CONNECTION LIMIT EXCEEDED!")
                            st.warning(
                                "You have too many active connections. Please stop the Streamlit app (Ctrl+C) and restart the terminal."
                            )
                        else:
                            st.error(f"Stream Error: {err_msg}")
            else:
                if st.button("⏹️ Stop Stream"):
                    service.stop()
                    st.rerun()

    # SINGLE SCOUT CONTROLS
    else:
        st.caption("Focus on a single high-priority game.")
        # (Simplified for now to focus on the request fix)
        st.info("Switch to League Monitor for demo.")

    # AUTO-TRADING SETTINGS
    st.markdown("---")
    st.markdown("### 🤖 Auto-Trader")

    paper_trade = st.toggle("📝 Paper Mode", value=True)
    stake_size = st.slider("Stake Size (€)", 2.0, 50.0, 5.0)

    auto_active = st.toggle("⚡ Enable Bot", value=service.auto_trade_enabled)

    if auto_active != service.auto_trade_enabled:
        if auto_active:
            service.enable_auto_trading(stake_size, paper_trade)
        else:
            service.disable_auto_trading()

    if service.is_running:
        st.markdown("---")
        st.caption(f"Status: 🟢 Running\nPID: {os.getpid()}")


# 2. MAIN DASHBOARD AREA

if service.is_running:
    # --- HEADS UP DISPLAY (HUD) ---
    st.markdown("## 📡 Live War Room")

    stats = service.get_trading_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Markets Live", len(getattr(service, "monitored_market_ids", [])))
    c2.metric("Open Positions", stats["open_positions"])
    c3.metric("Daily P&L", f"€{stats['daily_pnl']:.2f}", delta=stats["daily_pnl"])
    c4.metric("Pending Alerts", len(st.session_state.get("alert_history", [])))

    # --- GLOBAL GRID ---
    st.markdown("### 🌍 Global Monitor")

    # We use a container to update without flashing the whole page if possible
    main_placeholder = st.empty()

    # --- FOCUS SELECTOR (Static Position) ---
    focus_market_id = None
    if getattr(service, "monitored_market_ids", []):
        focus_market_id = st.selectbox(
            "🔍 Tactical View - Select Market",
            options=service.monitored_market_ids,
            format_func=lambda x: service.market_names.get(x, x),
            key="focus_selector_v2",  # Unique key
        )

    # --- CONTROL LOOP ---
    stop_btn = st.button("⏸ Pause Feed")

    if not stop_btn:
        while service.is_running:
            with main_placeholder.container():
                # 1. LIVE GRID RENDER
                grid_data = service.get_live_dashboard_data()
                if grid_data:
                    df_grid = pd.DataFrame(grid_data)
                    st.dataframe(
                        df_grid,
                        column_config={
                            "Status": st.column_config.TextColumn(
                                "Status", width="small"
                            ),
                            "Volume (€)": st.column_config.ProgressColumn(
                                "Volume", format="€%d", min_value=0, max_value=50000
                            ),
                            "In Play": st.column_config.CheckboxColumn(
                                "Live", disabled=True
                            ),
                        },
                        hide_index=True,
                        use_container_width=True,
                    )

                # 2. TACTICAL DETAIL VIEW
                if focus_market_id:
                    st.markdown("---")
                    st.markdown(
                        f"### 🎯 Tactical Detail: {service.market_names.get(focus_market_id)}"
                    )

                    data_odds = service.get_live_odds(focus_market_id)

                    if data_odds:
                        df_odds = pd.DataFrame(data_odds)

                        # Prepare Display DataFrame with specific ordering
                        # Using exact column names to match the Styler logic
                        cols_map = {
                            "selection_id": "Runner",
                            "back_price": "Back",
                            "lay_price": "Lay",
                            "spread": "Spread",
                            "total_matched": "Vol (€)",
                            "back_size": "Back Vol",
                            "lay_size": "Lay Vol",
                        }

                        # Filter and Rename
                        df_display = df_odds[
                            [c for c in cols_map.keys() if c in df_odds.columns]
                        ].rename(columns=cols_map)

                        # Apply Styling (The FIX for TypeError is safe formatting + handling NaNs)
                        styler = df_display.style.format(
                            {
                                "Back": safe_float_format,
                                "Lay": safe_float_format,
                                "Spread": safe_float_format,
                                "Vol (€)": safe_currency_format,
                                "Back Vol": safe_currency_format,
                                "Lay Vol": safe_currency_format,
                            }
                        )

                        # Apply Betfair Colors (Back=Blue, Lay=Pink)
                        # We use applymap for specific columns
                        styler.map(
                            lambda x: "background-color: #007bff; color: white",
                            subset=["Back"],
                        )
                        styler.map(
                            lambda x: "background-color: #fca5a5; color: black",
                            subset=["Lay"],
                        )

                        st.dataframe(styler, use_container_width=True, hide_index=True)

                    else:
                        st.info("Waiting for odds data stream...")

                # 3. ALERTS TERMINAL
                st.markdown("### 📟 System Log")
                hist = st.session_state.get("alert_history", [])
                if hist:
                    # Show last 3 messages as code block for "terminal" feel
                    log_lines = []
                    for alert in reversed(hist[-5:]):
                        t = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S")
                        lvl = alert.severity
                        msg = f"[{t}] [{lvl}] {service.market_names.get(alert.market_id, 'UNK')} - {alert.details}"
                        log_lines.append(msg)
                    st.code("\n".join(log_lines), language="text")
                else:
                    st.code(
                        "System Idle. Listening for anomalous order flow...",
                        language="text",
                    )

            time.sleep(1)  # Live update rate 1Hz

            # Background Alert Fetch
            new_alerts = service.get_new_alerts()
            if new_alerts:
                if "alert_history" not in st.session_state:
                    st.session_state.alert_history = []
                st.session_state.alert_history.extend(new_alerts)

else:
    # LANDING STATE (Service Stopped)
    st.markdown(
        """
        <div style='text-align: center; padding: 50px;'>
            <h1>🏀 NBA Predictor System</h1>
            <h3>Live Market Structure Analysis</h3>
            <p style='color: #9ca3af;'>Connect to Betfair Exchange to begin real-time monitoring.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
