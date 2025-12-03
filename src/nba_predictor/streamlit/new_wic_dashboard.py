"""
WIC Dashboard: Workflow Intelligent Control
The central control unit for the NBA prediction and betting system.
"""

import streamlit as st
import logging
import pandas as pd
from datetime import datetime, timedelta, date
import time
import json
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Fix relative import for direct execution
sys.path.append(str(Path(__file__).resolve().parents[2]))


# Import Components & Utils
from nba_predictor.streamlit.components.wic_components import (
    render_wic_header,
    render_game_card,
    render_prediction_summary,
    render_kpi_card,
    render_toast,
)
from nba_predictor.streamlit.utils.wic_state_manager import WICState
from nba_predictor.utils.betting_database_manager import get_secure_database_manager
from nba_predictor.core.data_store import UnifiedDataStore
from nba_predictor.streamlit.components.enhanced_prediction_bridge_professional import (
    get_enhanced_prediction_bridge_professional,
)
from nba_predictor.utils.legacy_risk_manager import LegacyRiskManager
from nba_predictor.api.data_provider import NBADataProvider
from nba_predictor.utils.team_normalizer import TeamNameNormalizer
import polars as pl

# Initialize Managers
db_manager = get_secure_database_manager()
data_store = UnifiedDataStore(base_path="data")
ml_bridge = get_enhanced_prediction_bridge_professional()
risk_manager = LegacyRiskManager(data_path="data")


def auto_update_and_settle():
    """
    Automatically updates game data and settles pending bets on dashboard load.
    """
    try:
        logger.info("🚀 Starting Auto-Update & Settlement process...")
        # Get Pending Bets
        current_user = st.session_state.get("user_id", "test_user_001")
        pending_bets = db_manager.safe_get_user_bets(
            user_id=current_user,
            status="PENDING",
            limit=100,
        )

        logger.info(
            f"🔍 Found {len(pending_bets)} pending bets for user '{current_user}'"
        )

        if not pending_bets:
            logger.info("❌ No pending bets found. Exiting.")
            return

        # 2. Fetch Latest Game Results (Force Refresh)
        try:
            # Determine date range from pending bets to minimize API calls
            bet_dates = []
            for b in pending_bets:
                try:
                    d_str = str(b.get("created_at")).split()[0]
                    d_obj = datetime.strptime(d_str, "%Y-%m-%d").date()
                    bet_dates.append(d_obj)
                except Exception as e:
                    logger.warning(
                        f"Failed to parse date for bet {b.get('bet_id')}: {e}"
                    )

            logger.info(f"📅 Bet dates to check: {bet_dates}")

            if bet_dates:
                # Use a set to avoid duplicate calls
                unique_dates = sorted(list(set(bet_dates)))
                provider = NBADataProvider()

                for d in unique_dates:
                    d_str = d.strftime("%Y-%m-%d")
                    logger.info(f"🔄 Refreshing data for {d_str}")
                    fetched_games = provider.get_scheduled_games(specific_date=d_str)

                    if fetched_games:
                        # Convert to Polars DataFrame for storage
                        games_data = []
                        for g in fetched_games:
                            games_data.append(
                                {
                                    "game_id": g.get("game_id"),
                                    "game_date": g.get("date"),
                                    "home_team": g.get("home_team"),
                                    "away_team": g.get("away_team"),
                                    "season": "2025-26",
                                    "game_time": g.get("time", "TBD"),
                                    "status": g.get("status", "Scheduled"),
                                    "home_score": g.get("home_score", 0),
                                    "away_score": g.get("away_score", 0),
                                    "match_id": g.get(
                                        "match_id"
                                    ),  # Important: Persist Canonical ID
                                }
                            )

                        if games_data:
                            new_games_df = pl.DataFrame(games_data)
                            logger.info(
                                f"   💾 Saving {len(games_data)} games to store for {d_str}"
                            )
                            db_manager.data_store.store_games_data(new_games_df, d_str)

                logger.info("✅ Data refresh cycle completed")

            # Now query the Data Store (which reads the updated parquet files)
            all_games = db_manager.data_store.get_games_data()
            games_list = all_games.to_dicts()
            games_dict = {str(g["game_id"]): g for g in games_list}

            # Create Canonical ID Index
            games_by_match_id = {}
            for g in games_list:
                if g.get("match_id"):
                    games_by_match_id[g.get("match_id")] = g
                else:
                    # Try to generate on the fly if missing (e.g. old data)
                    try:
                        g_date_str = str(g.get("date", ""))
                        try:
                            g_date = datetime.strptime(g_date_str, "%Y-%m-%d").date()
                        except:
                            g_date = date.today()

                        mid = TeamNameNormalizer.generate_match_id(
                            g_date, g.get("home_team", ""), g.get("away_team", "")
                        )
                        games_by_match_id[mid] = g
                    except:
                        pass

            logger.info(f"📚 Loaded {len(games_by_match_id)} games with Canonical IDs")

        except Exception as e:
            logger.error(f"Failed to refresh game data: {e}")
            games_dict = {}
            games_by_match_id = {}

        settled_count = 0

        for bet in pending_bets:
            bet_game_id = str(bet.get("game_id"))
            logger.debug(f"🧐 Processing bet {bet_game_id}")
            game = games_dict.get(bet_game_id)

            # Fallback: Match by Canonical ID
            if not game:
                try:
                    # 1. Extract details from prediction JSON
                    raw_pred = bet.get("prediction", "{}")
                    prediction_data = json.loads(raw_pred)

                    # Try to find team names
                    home_team = prediction_data.get("home_team")
                    away_team = prediction_data.get("away_team")

                    if not home_team or not away_team:
                        metrics = prediction_data.get("team_metrics", {})
                        if metrics:
                            home_team = metrics.get("home", {}).get("team_name")
                            away_team = metrics.get("away", {}).get("team_name")

                    logger.debug(
                        f"   Extracted teams: Home='{home_team}', Away='{away_team}'"
                    )

                    # 2. Extract Date
                    bet_date_str = str(bet.get("created_at")).split()[0]
                    try:
                        bet_date = datetime.strptime(bet_date_str, "%Y-%m-%d").date()
                    except:
                        bet_date = date.today()

                    # 3. Generate Canonical ID
                    if home_team and away_team:
                        canonical_id = TeamNameNormalizer.generate_match_id(
                            bet_date, home_team, away_team
                        )

                        logger.debug(f"   🔍 Looking for Canonical ID '{canonical_id}'")

                        # 4. Lookup
                        matched_game = games_by_match_id.get(canonical_id)

                        if matched_game:
                            logger.info(
                                f"   ✅ MATCH FOUND: {matched_game.get('game_id')}"
                            )
                            game = matched_game  # Assign to game variable!
                        else:
                            logger.warning(f"   ❌ No match found for '{canonical_id}'")

                except Exception as e:
                    logger.error(f"Canonical ID matching failed: {e}")

            # Check if game is finished
            if game:
                status = game.get("status")
                logger.debug(f"   Game found. Status: '{status}'")

                if status in ["Final", "Completed", "Finished"]:
                    logger.info("   ✅ Game is Final. Attempting settlement...")

                    try:
                        # Determine Outcome
                        outcome = "PENDING"
                        payout = 0.0

                        bet_type = bet.get("bet_type", "")
                        home_score = float(game.get("home_score", 0))
                        away_score = float(game.get("away_score", 0))
                        total_score = home_score + away_score

                        # Simple parsing of bet type (e.g., "OVER 220.5", "Lakers -5.5")
                        if "OVER" in bet_type.upper():
                            line = float(bet_type.split()[-1])
                            if total_score > line:
                                outcome = "WON"
                            else:
                                outcome = "LOST"
                        elif "UNDER" in bet_type.upper():
                            line = float(bet_type.split()[-1])
                            if total_score < line:
                                outcome = "WON"
                            else:
                                outcome = "LOST"
                        # Add more bet types (Moneyline, Spread) here as needed

                        logger.info(
                            f"   Outcome: {outcome} (Score: {home_score}-{away_score})"
                        )

                        if outcome != "PENDING":
                            # Update DB
                            profit_loss = 0.0
                            if outcome == "WON":
                                payout = float(bet.get("amount")) * float(
                                    bet.get("odds")
                                )
                                profit_loss = payout - float(bet.get("amount"))
                                # Update Bankroll
                                update_bankroll(payout)
                            else:
                                profit_loss = -float(bet.get("amount"))

                            db_manager.safe_update_bet_status(
                                bet_id=bet.get("bet_id"),
                                status=outcome,
                                result=outcome,
                                profit_loss=profit_loss,
                                audit_user="WIC_DASHBOARD",
                            )
                            settled_count += 1
                            logger.info(f"   ✅ Bet settled as {outcome}")

                    except Exception as e:
                        logger.error(f"   ❌ Settlement Failed: {e}")
                else:
                    logger.debug("   ⏳ Game not final. Skipping.")
            else:
                logger.warning("   ❌ No game found for bet.")

        if settled_count > 0:
            render_toast(
                f"Settled {settled_count} bets based on latest results!", "success"
            )

    except Exception as e:
        st.error(f"Error during auto-settlement: {str(e)}")


def get_bankroll() -> float:
    """Get current bankroll from file."""
    try:
        with open("data/bankroll.json", "r") as f:
            data = json.load(f)
            return float(data.get("current_bankroll", 0.0))
    except Exception:
        return 0.0


def update_bankroll(amount_change: float) -> bool:
    """Update bankroll by adding amount_change (negative for deductions)."""
    try:
        current = get_bankroll()
        new_amount = current + amount_change

        with open("data/bankroll.json", "w") as f:
            json.dump({"current_bankroll": new_amount}, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to update bankroll: {e}")
        return False


def render_step_1_scheduler():
    """
    Step 1: Game Selection (Scheduler)
    """
    st.subheader("📅 Games Schedule")

    # Date Filter
    col1, col2 = st.columns([2, 1])
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date.today(),  # Default to today
            min_value=date(2023, 1, 1),
            max_value=date(2026, 12, 31),
        )
    with col2:
        days = st.number_input("Days to show", min_value=1, value=3)

    end_date = start_date + timedelta(days=days)

    # Fetch Games
    try:
        # Initialize Data Store
        data_store.initialize()

        # Convert dates to string format YYYY-MM-DD
        s_str = start_date.strftime("%Y-%m-%d")
        e_str = end_date.strftime("%Y-%m-%d")

        # Fetch games from store
        games_df = data_store.get_games_data(date_range=(s_str, e_str))

        if games_df.is_empty():
            st.info(
                f"Local store empty. Fetching fresh data from API for {s_str} to {e_str}..."
            )

            try:
                # Initialize provider
                provider = NBADataProvider()

                # Fetch games (days_ahead needs to cover the range)
                # Calculate days difference
                delta = end_date - start_date
                days_to_fetch = max(delta.days, 1)

                fetched_games = provider.get_scheduled_games(
                    days_ahead=days_to_fetch,
                    specific_date=s_str if days_to_fetch == 1 else None,
                )

                if fetched_games:
                    # Convert to Polars DataFrame for storage
                    # Map keys to match store requirements
                    games_data = []
                    for g in fetched_games:
                        games_data.append(
                            {
                                "game_id": g.get("game_id"),
                                "game_date": g.get("date"),
                                "home_team": g.get("home_team"),
                                "away_team": g.get("away_team"),
                                "season": "2025-26",  # Default for now
                                "game_time": g.get("time", "TBD"),
                                "status": g.get("status", "Scheduled"),
                                "home_score": g.get("home_score", 0),
                                "away_score": g.get("away_score", 0),
                            }
                        )

                    new_games_df = pl.DataFrame(games_data)

                    # Store in data_store
                    # We store by date, so we might need to split if multiple dates
                    # But store_games_data takes a date_str for partitioning.
                    # Let's group by date.
                    dates = new_games_df["game_date"].unique()
                    for d in dates:
                        day_games = new_games_df.filter(pl.col("game_date") == d)
                        data_store.store_games_data(day_games, d)

                    # Re-fetch from store
                    games_df = data_store.get_games_data(date_range=(s_str, e_str))

                    if games_df.is_empty():
                        st.warning(
                            "Fetched games but failed to retrieve from store. Showing fetched data directly."
                        )
                        games_df = new_games_df
                else:
                    st.warning(f"No games found from API between {s_str} and {e_str}.")
                    return

            except Exception as api_error:
                st.error(f"Failed to fetch from API: {str(api_error)}")
                return

        # Convert to list of dicts for rendering
        games = games_df.to_dicts()

        for game in games:
            # Ensure required keys exist for the card
            game_display = {
                "game_id": game.get("game_id"),
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "game_date": game.get("game_date"),
                "game_time": game.get("game_time", "TBD"),
                "status": game.get("status", "Scheduled"),
            }
            render_game_card(
                game_display, on_analyze=lambda gid, g=game_display: select_game(gid, g)
            )

    except Exception as e:
        st.error(f"Error fetching games: {str(e)}")


def select_game(game_id: str, game_data: dict):
    """Callback to select a game and move to Step 2."""
    WICState.set_selected_game(game_id, game_data)
    WICState.set_step(2)
    st.rerun()


def render_step_2_predictor():
    """
    Step 2: Game Analysis (Predictor)
    """
    game = WICState.get_selected_game()
    if not game:
        st.error("No game selected. Please go back to Schedule.")
        if st.button("Back to Schedule"):
            WICState.set_step(1)
            st.rerun()
        return

    st.subheader(f"🔮 Prediction: {game.get('away_team')} @ {game.get('home_team')}")

    # Check if prediction already exists in state to avoid re-running ML on every rerun
    prediction = WICState.get_prediction()

    if not prediction:
        with st.spinner("Running Advanced ML Analysis..."):
            try:
                # Parse date
                g_date = game.get("game_date")
                if isinstance(g_date, str):
                    g_date = datetime.strptime(g_date, "%Y-%m-%d").date()

                # Call ML Bridge
                prediction = ml_bridge.get_professional_prediction(
                    home_team=game.get("home_team"),
                    away_team=game.get("away_team"),
                    game_date=g_date,
                    betting_line=game.get("total_line", 220.0),  # Add explicit line
                    include_detailed_analysis=True,
                    force_refresh=True,  # Force refresh to prevent stale predictions
                )

                # Store in State
                WICState.set_prediction(prediction)

            except Exception as e:
                st.error(f"ML Analysis Failed: {str(e)}")
                return

    # Display Prediction
    render_prediction_summary(prediction)

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("⬅ Back"):
            WICState.prev_step()
            st.rerun()
    with col2:
        if st.button("Next: Analyze Betting ➡", type="primary"):
            WICState.next_step()
            st.rerun()


def render_step_3_analyst():
    """
    Step 3: Betting Analysis (Analyst)
    """
    game = WICState.get_selected_game()
    prediction = WICState.get_prediction()

    if not game or not prediction:
        st.error("Missing game or prediction data. Please go back.")
        if st.button("Back to Predictor"):
            WICState.set_step(2)
            st.rerun()
        return

    st.subheader(f"📊 Analysis: {game.get('away_team')} @ {game.get('home_team')}")

    # 1. Get Manual Input (Central Line)
    st.markdown("""
    ### 🎯 Market Input
    Enter the **Central Line** from the bookmaker (the point total where Over/Under odds are ~2.00).
    """)

    col_input, col_info = st.columns([1, 2])
    with col_input:
        central_line = st.number_input(
            "Central Line (Points)",
            min_value=150.0,
            max_value=300.0,
            value=220.0,
            step=0.5,
            help="The total points line where Over and Under are priced equally (50% probability).",
        )

    if central_line:
        # 2. Perform Analysis using LegacyRiskManager

        # Map prediction to distribution format expected by Risk Manager
        distribution = {
            "predicted_mu": prediction.get("predicted_total", 0),
            "predicted_sigma": prediction.get("standard_error", 15.0),
        }

        # Generate and analyze opportunities
        opportunities = risk_manager.analyze_betting_opportunities(
            distribution=distribution,
            central_line=central_line,
            bankroll=risk_manager.current_bankroll,
        )

        # Get Optimal Bet
        optimal_bet = risk_manager.calculate_optimal_bet(opportunities)

        if optimal_bet:
            st.markdown("### 💡 System Recommendation (Optimal Bet)")

            # Display Optimal Bet Card
            opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)

            with opt_col1:
                st.metric(
                    "Recommended Bet",
                    f"{optimal_bet['type']} {optimal_bet['line']}",
                    help="The specific line and side to bet on.",
                )
            with opt_col2:
                st.metric(
                    "Stake",
                    f"€{optimal_bet['stake']:.2f}",
                    help=f"Calculated using Kelly Fraction: {optimal_bet.get('kelly_fraction', 0):.2%}",
                )
            with opt_col3:
                edge = optimal_bet["edge"]
                st.metric(
                    "Edge",
                    f"{edge:.1%}",
                    delta="Positive" if edge > 0 else "Negative",
                    help="Mathematical advantage over the bookmaker.",
                )
            with opt_col4:
                q_score = optimal_bet["quality_score"] * 100
                st.metric(
                    "Quality Score",
                    f"{q_score:.1f}/100",
                    help="Composite score of Edge, Confidence, Risk, and Consistency.",
                )

            # Detailed Breakdown
            with st.expander("🔍 Detailed Analysis Metrics", expanded=True):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Model Probability", f"{optimal_bet['probability']:.1%}")
                m2.metric(
                    "Implied Probability", f"{optimal_bet['implied_probability']:.1%}"
                )
                m3.metric("Odds", f"{optimal_bet['odds']:.2f}")
                m4.metric(
                    "Expected Value",
                    f"€{optimal_bet['stake'] * optimal_bet['edge']:.2f}",
                )

            # Store Recommendation in State
            st.session_state[WICState.KEY_RECOMMENDED_BET] = optimal_bet

        else:
            st.warning(
                "No value bets found for this line. The market is efficient or the edge is too small."
            )
            st.session_state[WICState.KEY_RECOMMENDED_BET] = None

        # Show other opportunities table
        with st.expander("📋 All Opportunities (Ranked by Quality)"):
            if opportunities:
                df_opps = pd.DataFrame(opportunities)
                # Format for display
                display_cols = [
                    "type",
                    "line",
                    "odds",
                    "probability",
                    "edge",
                    "quality_score",
                    "stake",
                ]
                df_display = df_opps[display_cols].copy()
                df_display["probability"] = df_display["probability"].map(
                    "{:.1%}".format
                )
                df_display["edge"] = df_display["edge"].map("{:.1%}".format)
                df_display["quality_score"] = (df_display["quality_score"] * 100).map(
                    "{:.1f}".format
                )
                df_display["stake"] = df_display["stake"].map("€{:.2f}".format)

                st.dataframe(df_display, use_container_width=True)

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("⬅ Back"):
            WICState.prev_step()
            st.rerun()
    with col2:
        # Only enable Next if a valid recommendation exists
        if st.session_state.get(WICState.KEY_RECOMMENDED_BET):
            if st.button("Next: Place Bet ➡", type="primary"):
                WICState.next_step()
                st.rerun()
        else:
            st.warning("A valid recommendation is required to proceed.")


def render_step_4_trader():
    """
    Step 4: Bet Placement (Trader)
    """
    game = WICState.get_selected_game()
    rec = st.session_state.get(WICState.KEY_RECOMMENDED_BET)

    if not game or not rec:
        st.error("Missing bet recommendation. Please go back.")
        if st.button("Back to Analyst"):
            WICState.set_step(3)
            st.rerun()
        return

    st.subheader(f"💰 Place Bet: {game.get('away_team')} @ {game.get('home_team')}")

    # Bet Placement Form
    with st.form("place_bet_form"):
        st.markdown("### Confirm Bet Details")

        c1, c2, c3 = st.columns(3)
        with c1:
            # Construct readable bet type
            bet_desc = f"{rec.get('type')} {rec.get('line')}"
            st.text_input("Bet Selection", value=bet_desc, disabled=True)
        with c2:
            odds = st.number_input(
                "Odds", value=float(rec.get("odds")), min_value=1.01, step=0.01
            )
        with c3:
            amount = st.number_input(
                "Stake (€)", value=float(rec.get("stake")), min_value=0.0, step=1.0
            )

        st.caption(
            f"Potential Return: €{amount * odds:.2f} (Profit: €{amount * (odds - 1):.2f})"
        )

        st.info(
            f"Reasoning: Quality Score {rec.get('quality_score') * 100:.1f}/100 | Edge {rec.get('edge'):.1%}"
        )

        submitted = st.form_submit_button("✅ Confirm & Place Bet", type="primary")

        if submitted:
            if amount <= 0:
                st.error("Stake must be greater than 0.")
            else:
                try:
                    # Insert into Database
                    bet_id = db_manager.safe_insert_bet(
                        user_id="test_user_001",  # Fixed User ID for testing
                        game_id=game.get("game_id"),
                        bet_type=bet_desc,
                        amount=amount,
                        odds=odds,
                        prediction=json.dumps(WICState.get_prediction(), default=str),
                        confidence_interval=None,
                        audit_user="WIC_DASHBOARD",
                    )

                    if bet_id:
                        # Deduct stake from bankroll
                        if update_bankroll(-amount):
                            render_toast(
                                f"Bet Placed Successfully! ID: {bet_id}", "success"
                            )
                            time.sleep(1)  # Give time for toast
                            WICState.next_step()
                            st.rerun()
                        else:
                            st.error("Bet placed but failed to update bankroll.")
                    else:
                        st.error("Failed to place bet. Database error.")

                except Exception as e:
                    st.error(f"Error placing bet: {str(e)}")

    # Navigation
    if st.button("⬅ Back"):
        WICState.prev_step()
        st.rerun()


def render_step_5_portfolio():
    """
    Step 5: Portfolio Management (Portfolio)
    """
    st.subheader("📈 Betting Portfolio")

    # 1. KPIs
    summary = db_manager.safe_get_user_summary(user_id="test_user_001")

    # Load real bankroll
    current_bankroll = get_bankroll()  # Updated to use the new helper function

    kpi0, kpi1, kpi2, kpi3, kpi4 = st.columns(5)
    with kpi0:
        render_kpi_card("Bankroll", f"€{current_bankroll:.2f}")
    with kpi1:
        render_kpi_card("Total Bets", summary.get("total_bets", 0))
    with kpi2:
        win_rate = summary.get("win_rate", 0)
        render_kpi_card("Win Rate", f"{win_rate:.1f}%")
    with kpi3:
        roi = summary.get("roi", 0)
        render_kpi_card("ROI", f"{roi:.1f}%", delta=f"{roi:.1f}%", color="normal")
    with kpi4:
        pl = summary.get("net_profit_loss", 0) or 0
        render_kpi_card("Net P&L", f"€{pl:.2f}", delta=f"€{pl:.2f}")

    st.markdown("---")

    # 2. Bet History Tabs
    tab1, tab2 = st.tabs(["⏳ Pending Bets", "📜 Bet History"])

    with tab1:
        pending_bets = db_manager.safe_get_user_bets(
            user_id="test_user_001", status="PENDING"
        )

        # Pre-fetch game info for mapping
        try:
            all_games = db_manager.data_store.get_games_data()
            # Create mapping: game_id -> "Away @ Home"
            # Handle potential type differences by converting to string
            game_map = {
                str(row["game_id"]): f"{row['away_team']} @ {row['home_team']}"
                for row in all_games.to_dicts()
            }
        except Exception as e:
            st.error(f"Error loading game data: {e}")
            game_map = {}

        if pending_bets:
            for bet in pending_bets:
                with st.container():
                    c1, c2, c3, c4, c5 = st.columns([3, 2, 1, 1, 1])
                    with c1:
                        game_id = str(bet.get("game_id"))
                        matchup = game_map.get(game_id, "Unknown Matchup")
                        st.write(f"**{matchup}**")
                        st.caption(f"ID: {game_id} • 📅 {bet.get('created_at')}")
                    with c2:
                        st.write(f"**{bet.get('bet_type')}**")
                    with c3:
                        st.write(f"@{bet.get('odds')}")
                    with c4:
                        st.write(f"€{bet.get('amount')}")
                    with c5:
                        if st.button(
                            "🗑️", key=f"del_{bet.get('bet_id')}", help="Delete Bet"
                        ):
                            if db_manager.safe_delete_bet(
                                bet.get("bet_id"),
                                "test_user_001",
                                audit_user="WIC_DASHBOARD",
                            ):
                                # Refund stake to bankroll
                                refund_amount = float(bet.get("amount", 0.0))
                                if update_bankroll(refund_amount):
                                    st.toast("Bet deleted and refunded!", icon="🗑️")
                                else:
                                    st.warning(
                                        "Bet deleted but bankroll update failed."
                                    )

                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Could not delete bet.")
                st.divider()
        else:
            st.info("No pending bets.")

    with tab2:
        history_bets = db_manager.safe_get_user_bets(user_id="test_user_001", limit=50)
        # Filter out pending
        settled_bets = [b for b in history_bets if b["status"] != "PENDING"]

        if settled_bets:
            # Display as a list with delete buttons (like Pending tab) for control
            for bet in settled_bets:
                with st.container():
                    c1, c2, c3, c4, c5, c6 = st.columns([3, 1, 1, 1, 1, 1])
                    with c1:
                        game_id = str(bet.get("game_id"))
                        # Try to find matchup in game_map if available, else use ID
                        matchup = game_map.get(game_id, f"Game {game_id}")
                        st.write(f"**{matchup}**")
                        st.caption(f"📅 {bet.get('created_at')}")
                    with c2:
                        st.write(f"**{bet.get('bet_type')}**")
                    with c3:
                        st.write(f"@{bet.get('odds')}")
                    with c4:
                        st.write(f"€{bet.get('amount')}")
                    with c5:
                        res = bet.get("result")
                        color = "green" if res == "WON" else "red"
                        st.markdown(f":{color}[{res}]")
                        if res == "WON":
                            st.caption(f"+€{bet.get('profit_loss'):.2f}")
                    with c6:
                        if st.button(
                            "🗑️",
                            key=f"del_hist_{bet.get('bet_id')}",
                            help="Delete Record",
                        ):
                            # Bankroll Adjustment Logic
                            refund_amount = 0.0
                            stake = float(bet.get("amount", 0.0))

                            if bet.get("result") == "WON":
                                # Revert Win: Deduct Payout, Add back Stake (effectively remove Profit)
                                # Payout = Stake + Profit
                                # Current Bankroll has (Stake + Profit) added (relative to post-bet state)
                                # We want to go back to Pre-Bet state.
                                # Pre-Bet = Current - Profit - Stake + Stake? No.
                                # Let's trace:
                                # Start: 100
                                # Bet 10: 90
                                # Win 20 (Profit 10): 110
                                # Delete: Want 100.
                                # Adjustment: -10. (Which is -Profit).
                                # Profit = Payout - Stake.
                                # So Adjustment = -(Payout - Stake).
                                # Wait, update_bankroll adds the argument.
                                # So we pass -(Payout - Stake).
                                payout = stake * float(bet.get("odds", 1.0))
                                profit = payout - stake
                                refund_amount = -profit

                            elif bet.get("result") == "LOST":
                                # Revert Loss: Add back Stake
                                # Start: 100
                                # Bet 10: 90
                                # Loss: 90
                                # Delete: Want 100.
                                # Adjustment: +10.
                                refund_amount = stake

                            if db_manager.safe_delete_bet(
                                bet.get("bet_id"),
                                "test_user_001",
                                audit_user="WIC_DASHBOARD",
                            ):
                                if update_bankroll(refund_amount):
                                    st.toast(
                                        f"Bet deleted. Bankroll adjusted by €{refund_amount:.2f}",
                                        icon="🗑️",
                                    )
                                else:
                                    st.warning(
                                        "Bet deleted but bankroll update failed."
                                    )

                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Could not delete bet.")
                    st.divider()
        else:
            st.info("No settled bets history.")

    # Navigation
    st.markdown("---")
    if st.button("🔄 Start New Analysis", type="primary"):
        WICState.reset()
        st.rerun()


# --- UI Configuration ---
from pathlib import Path


def load_css():
    """Load custom CSS for modern UI (v2)."""
    css_file = Path(__file__).parent / "style_v2.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("style_v2.css not found. Using default theme.")


def main():
    """
    Main entry point for the Streamlit dashboard.
    """
    # Initialize Page Config
    st.set_page_config(
        page_title="NBA Predictor Professional",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load Custom CSS
    load_css()

    # Initialize State
    WICState.initialize()

    # --- Sidebar ---
    with st.sidebar:
        st.title("🏀 NBA Pro")
        st.markdown("---")
        if st.button("🏠 New Analysis", use_container_width=True):
            WICState.reset()
            st.rerun()

        if st.button("📈 Portfolio", use_container_width=True):
            WICState.set_step(5)
            st.rerun()

    current_step = WICState.get_current_step()

    # Render Header
    render_wic_header("Workflow Intelligent Control", current_step)

    # Step 0: Auto-Run (Force Run for Debugging)
    if "auto_settled" not in st.session_state:
        print("\n\n!!! 🚀 AUTO-SETTLEMENT TRIGGERED !!!\n\n")
        auto_update_and_settle()
        st.session_state["auto_settled"] = True

    # Workflow Router
    if current_step == 1:
        render_step_1_scheduler()
    elif current_step == 2:
        render_step_2_predictor()
    elif current_step == 3:
        render_step_3_analyst()
    elif current_step == 4:
        render_step_4_trader()
    elif current_step == 5:
        render_step_5_portfolio()


if __name__ == "__main__":
    main()
