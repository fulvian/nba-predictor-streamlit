"""
WIC Dashboard Components
Reusable UI components for the Workflow Intelligent Control Dashboard.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Any

import streamlit as st

from nba_predictor.streamlit import assets


def render_wic_header(title: str, current_step: int, total_steps: int = 5):
    """
    Renders the standard WIC header with breadcrumb progress.
    """
    st.title(f"{title}")

    # Progress Bar
    progress = current_step / total_steps
    st.progress(progress)

    # Breadcrumbs (Editorial style with Icons)
    steps = ["Update", "Schedule", "Predict", "Analyze", "Trade", "Portfolio"]
    cols = st.columns(len(steps))
    for i, step_name in enumerate(steps):
        if i == current_step:
            cols[i].markdown(
                f"<div style='display:flex;align-items:center;gap:4px;font-weight:700;border-bottom:2px solid var(--color-primary);padding-bottom:4px;'>{assets.ICON_TARGET} {step_name}</div>",
                unsafe_allow_html=True,
            )
        elif i < current_step:
            cols[i].markdown(
                f"<div style='display:flex;align-items:center;gap:4px;color:var(--color-text-secondary);'>{assets.ICON_CHECK_CIRCLE} {step_name}</div>",
                unsafe_allow_html=True,
            )
        else:
            cols[i].markdown(
                f"<div style='display:flex;align-items:center;gap:4px;color:var(--color-border);'>{assets.ICON_Target if False else ''} {step_name}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")


def render_game_card(game: dict[str, Any], on_analyze: Callable[[str], None]):
    """
    Renders a card for a single game in the scheduler list.
    """
    with st.container():
        # Styling for the card
        # Styling for the card is now handled in style_light.css
        # .game-card class is used

        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

        with col1:
            st.subheader(f"{game.get('away_team')} @ {game.get('home_team')}")
            st.caption(f"Game ID: {game.get('game_id')}")

        with col2:
            game_date = game.get("game_date")
            if isinstance(game_date, (str, datetime)):
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:4px'>{assets.ICON_CALENDAR} {str(game_date)}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:4px'>{assets.ICON_CLOCK} {game.get('game_time', 'TBD')}</div>",
                unsafe_allow_html=True,
            )

        with col3:
            status = game.get("status", "Scheduled")
            st.info(status)

        with col4:
            # Use custom SVG icon next to button
            game_id = game.get("game_id")
            c_icon, c_btn = st.columns([1, 3])
            with c_icon:
                st.markdown(
                    assets.ICON_ANALYZE.replace('width="24"', 'width="20"').replace(
                        'height="24"', 'height="20"'
                    ),
                    unsafe_allow_html=True,
                )
            with c_btn:
                if st.button("Analyze", key=f"btn_analyze_{game_id}", type="primary"):
                    on_analyze(game_id)


def render_prediction_summary(prediction: dict[str, Any]):
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

    # Create HTML Card (Editorial Style v7 - with Assets)
    card_html = f"""
<div class="game-card">
    <div class="card-header">
        <div class="team-names">{away_team} <span style="font-family: var(--font-body); font-weight: 300; font-size: 1.2rem; color: var(--color-text-secondary);">at</span> {home_team}</div>
        <div class="game-status" style="display:flex;align-items:center;gap:6px;">{assets.ICON_BASKETBALL.replace('width="24"', 'width="20"').replace('height="24"', 'height="20"')} Live Analysis</div>
    </div>
    <div class="stats-grid">
        <div class="stat-box">
            <div class="stat-label">Away Team</div>
            <div class="stat-value">{away_team}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Home Team</div>
            <div class="stat-value">{home_team}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">{assets.ICON_ANALYTICS} Predicted Total</div>
            <div class="stat-value">{predicted_total:.1f}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">{assets.ICON_BRAIN} Confidence</div>
            <div class="stat-value">{confidence:.1f}%</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">{assets.ICON_BETTING} Edge</div>
            <div class="edge-badge {ev_class}">{ev_display}</div>
        </div>
    </div>
    <div class="rec-box">
        <div class="rec-label" style="display:flex;align-items:center;gap:6px;">{assets.ICON_LIGHTBULB} Recommendation</div>
        <div class="rec-value">{recommendation}</div>
    </div>
</div>
"""

    st.markdown(card_html, unsafe_allow_html=True)

    # Detailed Stats Expanders
    with st.expander("Advanced Analytics & Factors", expanded=False):
        cols = st.columns(2)
        with cols[0]:
            st.markdown(
                f"#### {assets.ICON_NAV_CHART} Momentum Factors", unsafe_allow_html=True
            )
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
            st.markdown(
                f"#### {assets.ICON_BRAIN} Model Confidence", unsafe_allow_html=True
            )
            st.metric("Model Certainty", f"{confidence:.1f}%")

            # Show insights
            if "insights" in prof_analysis:
                for insight in prof_analysis["insights"]:
                    st.info(insight)


def render_betting_card(
    bookmaker_odds: dict[str, float],
    system_probs: dict[str, float],
    manual_line_key: str,
) -> float | None:
    """
    Renders the betting analysis card with manual input.
    Returns the manually entered line/odds.
    """
    st.markdown(f"### {assets.ICON_NAV_CHART} Betting Analysis", unsafe_allow_html=True)

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
    """
    Wrapper for st.toast with custom SVG icons
    """
    # Streamlit toast doesn't support HTML/SVG directly in the icon parameter easily without hacky CSS.
    # We will use the 'icon' parameter with standard emojis for now as fallback,
    # OR we can try to use the new material icons if supported.
    # However, to be strictly "Anthropic", we might want to use a custom notification component.
    # For now, let's stick to clean emojis that match the color scheme or just text.

    # Actually, let's try to use the 'icon' parameter with a valid emoji that fits the style better,
    # or just omit it and use the message.

    if type == "success":
        st.toast(message)
    elif type == "warning":
        st.toast(message)
    else:
        st.toast(message)
