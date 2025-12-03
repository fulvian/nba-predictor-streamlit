"""
WIC Dashboard Components
Reusable UI components for the Workflow Intelligent Control Dashboard.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, Callable
from datetime import datetime


def render_wic_header(title: str, current_step: int, total_steps: int = 5):
    """
    Renders the standard WIC header with breadcrumb progress.
    """
    st.title(f"🏀 {title}")

    # Progress Bar
    progress = current_step / total_steps
    st.progress(progress)

    # Breadcrumbs
    steps = ["Update", "Schedule", "Predict", "Analyze", "Trade", "Portfolio"]
    cols = st.columns(len(steps))
    for i, step_name in enumerate(steps):
        if i == current_step:
            cols[i].markdown(f"**🔵 {step_name}**")
        elif i < current_step:
            cols[i].markdown(f"✅ {step_name}")
        else:
            cols[i].markdown(f"⚪ {step_name}")

    st.markdown("---")


def render_game_card(game: Dict[str, Any], on_analyze: Callable[[str], None]):
    """
    Renders a card for a single game in the scheduler list.
    """
    with st.container():
        # Styling for the card
        st.markdown(
            """
        <style>
        .game-card {
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #e0e0e0;
            margin-bottom: 1rem;
            background-color: #f9f9f9;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

        with col1:
            st.subheader(f"{game.get('away_team')} @ {game.get('home_team')}")
            st.caption(f"Game ID: {game.get('game_id')}")

        with col2:
            game_date = game.get("game_date")
            if isinstance(game_date, (str, datetime)):
                st.text(f"📅 {str(game_date)}")
            st.text(f"⏰ {game.get('game_time', 'TBD')}")

        with col3:
            status = game.get("status", "Scheduled")
            st.info(status)

        with col4:
            if st.button("Analyze 🔍", key=f"btn_analyze_{game.get('game_id')}"):
                on_analyze(game.get("game_id"))


def render_prediction_summary(prediction: Dict[str, Any]):
    """
    Render a modern, card-based prediction summary.
    """
    if not prediction:
        return

    # Extract data with correct keys
    home_team = prediction.get("home_team", "Home Team")
    away_team = prediction.get("away_team", "Away Team")
    predicted_total = prediction.get("predicted_total", 0)

    # Confidence: model_confidence is 0-1, so multiply by 100
    confidence = prediction.get("model_confidence", 0) * 100

    # Expected Value / Edge
    # Extract from professional_analysis if available
    prof_analysis = prediction.get("professional_analysis", {})
    edge_analysis = prof_analysis.get("edge_analysis", {})
    ev = edge_analysis.get("edge_percentage", 0)

    recommendation = prediction.get("recommendation", "No Bet")

    # Determine styles based on EV
    ev_class = "ev-positive" if ev > 0 else "ev-negative"
    ev_display = f"+{ev:.1f}%" if ev > 0 else f"{ev:.1f}%"

    # Create HTML Card (Light Minimalist v4)
    # SVG Icons
    icon_basketball = '<svg class="icon-svg" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><path d="M5.6 5.6l12.8 12.8"></path><path d="M18.4 5.6l-12.8 12.8"></path></svg>'
    icon_chart = '<svg class="icon-svg" viewBox="0 0 24 24"><path d="M18 20V10"></path><path d="M12 20V4"></path><path d="M6 20v-6"></path></svg>'
    icon_money = '<svg class="icon-svg" viewBox="0 0 24 24"><rect x="2" y="5" width="20" height="14" rx="2"></rect><line x1="2" y1="10" x2="22" y2="10"></line></svg>'
    icon_check = '<svg class="icon-svg" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"></polyline></svg>'

    card_html = f"""
<div class="game-card">
<div class="card-header">
<div class="team-names">{away_team} <span style="color: var(--text-secondary); font-weight: 300;">@</span> {home_team}</div>
<div class="game-status">{icon_basketball} Live Analysis</div>
</div>
<div class="stats-grid">
<div class="stat-box">
<div class="stat-label">{icon_chart} Predicted Total</div>
<div class="stat-value">{predicted_total:.1f}</div>
</div>
<div class="stat-box">
<div class="stat-label">{icon_check} Confidence</div>
<div class="stat-value">{confidence:.1f}%</div>
</div>
<div class="stat-box">
<div class="stat-label">{icon_money} Edge</div>
<div class="edge-badge {ev_class}">{ev_display}</div>
</div>
</div>
<div class="rec-box">
<div class="rec-label">Recommendation</div>
<div class="rec-value">{recommendation}</div>
</div>
</div>
"""

    st.markdown(card_html, unsafe_allow_html=True)

    # Detailed Stats Expanders
    with st.expander("📊 Advanced Analytics & Factors", expanded=False):
        cols = st.columns(2)
        with cols[0]:
            st.markdown("#### 📈 Momentum Factors")
            # Check for situational factors
            sit_factors = prediction.get("situational_factors", {})
            if sit_factors:
                for factor, value in sit_factors.items():
                    # Normalize value for progress bar if needed, or just display text
                    st.text(f"{factor}: {value}")

            # Also check 'factors' if it exists (legacy)
            if "factors" in prediction:
                for factor, value in prediction["factors"].items():
                    st.progress(min(max(value, 0.0), 1.0), text=f"{factor}")

        with cols[1]:
            st.markdown("#### 🧠 Model Confidence")
            st.metric("Model Certainty", f"{confidence:.1f}%")

            # Show insights
            if "insights" in prof_analysis:
                for insight in prof_analysis["insights"]:
                    st.info(insight)


def render_betting_card(
    bookmaker_odds: Dict[str, float],
    system_probs: Dict[str, float],
    manual_line_key: str,
) -> Optional[float]:
    """
    Renders the betting analysis card with manual input.
    Returns the manually entered line/odds.
    """
    st.subheader("📊 Betting Analysis")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Bookmaker Odds")
        if bookmaker_odds:
            for bookie, odds in bookmaker_odds.items():
                st.write(f"**{bookie}**: {odds}")
        else:
            st.warning("No live odds available.")

    with c2:
        st.markdown("### Manual Analysis")
        manual_val = st.number_input(
            "Enter Central Line / Manual Odds",
            min_value=1.01,
            max_value=100.0,
            value=2.0,
            step=0.01,
            key=manual_line_key,
        )
        return manual_val


def render_kpi_card(label: str, value: Any, delta: Any = None, color: str = "normal"):
    """
    Renders a KPI metric card.
    """
    st.metric(label=label, value=value, delta=delta)


def render_toast(message: str, type: str = "success"):
    """
    Wrapper for st.toast
    """
    icon = "✅" if type == "success" else "⚠️" if type == "warning" else "❌"
    st.toast(f"{icon} {message}")
