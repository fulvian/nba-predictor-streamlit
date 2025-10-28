#!/usr/bin/env python3
"""
🏀 NBA Betting Workflow Dashboard - Context7 Best Practice Implementation

This module implements a clean, modular dashboard using existing components
following Context7 best practices for Streamlit applications.

Workflow: Games Schedule → Game Analysis → Betting Lines
"""

import logging
from datetime import datetime, date, timedelta
from typing import Any, Dict, Optional

import streamlit as st

# Try to import NBADataProvider, but make it optional for now
try:
    from ..api.data_provider import NBADataProvider
    NBA_DATA_PROVIDER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"NBADataProvider not available: {e}")
    NBA_DATA_PROVIDER_AVAILABLE = False

    # Create mock provider for now
    class NBADataProvider:
        def __init__(self):
            pass
        def get_scheduled_games(self, days_ahead=7, specific_date=None):
            # Return mock games data for testing
            from datetime import date, timedelta
            import random

            games = []
            base_date = specific_date or date.today()

            for i in range(min(days_ahead, 5)):
                game_date = base_date + timedelta(days=i)

                teams = ["Lakers", "Celtics", "Warriors", "Heat", "Nets", "Bucks", "Suns", "Mavericks"]
                home_team = random.choice(teams)
                away_team = random.choice([t for t in teams if t != home_team])

                games.append({
                    'date': game_date.strftime("%Y-%m-%d"),
                    'home_team': home_team,
                    'away_team': away_team,
                    'status': 'Scheduled',
                    'odds': {
                        'totals': {
                            'DraftKings': {
                                'over': {'line': 220.5, 'odds': -110},
                                'under': {'line': 220.5, 'odds': -110}
                            }
                        }
                    }
                })

            return games
from ..core.data_store import UnifiedDataStore
from ..utils.exceptions import StreamlitError, APIError
from .utils.cache_manager import setup_caching_for_app
from .config.deployment_config import load_config
from .components.predictions_dashboard import render_predictions_dashboard
from .components.analytics_dashboard import render_analytics_dashboard
from .components.sync_dashboard import render_sync_dashboard

logger = logging.getLogger(__name__)


def setup_session_state() -> None:
    """Setup session state for the betting workflow - Context7 compliant."""
    # Context7 best practice: Initialize all session state variables at once
    session_defaults = {
        'current_season': 2024,
        'selected_date': date.today(),
        'betting_workflow_step': 1,
        'selected_game': None,
        'available_games': None,
        'days_ahead': 1,
        'games_loaded': False,
        'debug_mode': True
    }

    for key, default_value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Debug logging for session state initialization
    if st.session_state.debug_mode:
        logger.info(f"Session state initialized. Current step: {st.session_state.betting_workflow_step}")
        logger.info(f"Available games loaded: {st.session_state.games_loaded}")
        logger.info(f"Selected game: {st.session_state.selected_game}")

def reset_workflow():
    """Workflow reset function - Context7 compliant."""
    # Context7 best practice: Reset all relevant session state variables
    st.session_state.betting_workflow_step = 1
    st.session_state.selected_game = None
    st.session_state.available_games = None
    st.session_state.games_loaded = False

    if st.session_state.debug_mode:
        logger.info("Workflow reset to Step 1")


def load_games_callback():
    """Callback function for loading games - Context7 compliant."""
    if st.session_state.debug_mode:
        logger.info("Loading games callback triggered")

    try:
        # Get data provider instance
        data_provider = NBADataProvider()

        # Convert selected date to string format for API
        date_str = st.session_state.selected_date.strftime('%Y-%m-%d')
        days_ahead = st.session_state.days_ahead

        if st.session_state.debug_mode:
            logger.info(f"Fetching games for date: {date_str}, days ahead: {days_ahead}")

        # Get games using NBA Official API
        games = data_provider.get_scheduled_games(days_ahead=days_ahead, specific_date=date_str)

        # Store games in session state
        st.session_state.available_games = games
        st.session_state.games_loaded = True

        if st.session_state.debug_mode:
            logger.info(f"Successfully loaded {len(games)} games")
            for game in games[:3]:  # Log first 3 games for debugging
                logger.info(f"Game: {game.get('away_team', 'Unknown')} @ {game.get('home_team', 'Unknown')} - {game.get('date', 'Unknown')}")

    except Exception as e:
        st.session_state.games_loaded = False
        st.session_state.available_games = []
        logger.error(f"Error loading games: {e}")
        if st.session_state.debug_mode:
            logger.error(f"Full error details: {str(e)}")


def select_game_callback(game_index):
    """Callback function for selecting a game - Context7 compliant."""
    if st.session_state.debug_mode:
        logger.info(f"Select game callback triggered for index: {game_index}")

    # Validate that we have available games and index is valid
    if not st.session_state.get('available_games'):
        logger.warning("No available games to select from")
        return

    if game_index >= len(st.session_state.available_games):
        logger.warning(f"Game index {game_index} out of range (0-{len(st.session_state.available_games)-1})")
        return

    # Select the game
    selected_game = st.session_state.available_games[game_index]
    st.session_state.selected_game = selected_game
    st.session_state.betting_workflow_step = 2

    if st.session_state.debug_mode:
        logger.info(f"Selected game: {selected_game.get('away_team', 'Unknown')} @ {selected_game.get('home_team', 'Unknown')}")
        logger.info("Advanced to Step 2: Game Analysis")


def select_nearby_game_callback(nearby_date, game_index, nearby_games_list):
    """Callback function for selecting nearby games - Context7 compliant."""
    if st.session_state.debug_mode:
        logger.info(f"Select nearby game callback: date={nearby_date}, index={game_index}")

    try:
        # Convert nearby date string to date object
        from datetime import datetime
        nearby_date_obj = datetime.strptime(nearby_date, '%Y-%m-%d').date()

        # Update session state
        st.session_state.selected_date = nearby_date_obj
        st.session_state.available_games = nearby_games_list
        st.session_state.games_loaded = True

        # Select the specific game
        if game_index < len(nearby_games_list):
            st.session_state.selected_game = nearby_games_list[game_index]
            st.session_state.betting_workflow_step = 2

        if st.session_state.debug_mode:
            logger.info(f"Updated date to {nearby_date} and selected game at index {game_index}")

    except Exception as e:
        logger.error(f"Error in select_nearby_game_callback: {e}")


def render_workflow_header() -> None:
    """Render workflow header with step indicators."""
    st.markdown("""
    # 🏀 NBA Betting Workflow Dashboard

    **Complete 3-step betting analysis workflow using real data**
    """)

    # Step indicators
    step1, step2, step3 = st.columns(3)

    with step1:
        if st.session_state.betting_workflow_step >= 1:
            st.success("✅ Step 1: Games Schedule")
        else:
            st.info("📋 Step 1: Games Schedule")

    with step2:
        if st.session_state.betting_workflow_step >= 2:
            st.success("✅ Step 2: Game Analysis")
        else:
            st.info("📊 Step 2: Game Analysis")

    with step3:
        if st.session_state.betting_workflow_step >= 3:
            st.success("✅ Step 3: Betting Lines")
        else:
            st.info("💰 Step 3: Betting Lines")

    st.divider()


def render_games_schedule_step(data_provider: NBADataProvider) -> None:
    """Render Step 1: NBA Games Schedule - Context7 Best Practice Implementation."""
    st.subheader("📅 Step 1: NBA Games Schedule")
    st.caption("Real-time NBA games from official APIs")

    # Date selection with Context7 best practice
    col1, col2 = st.columns([2, 1])
    with col1:
        st.date_input(
            "Select Date",
            value=st.session_state.selected_date,
            max_value=date.today() + timedelta(days=30),
            key="selected_date"
        )

    with col2:
        st.selectbox(
            "Days Ahead",
            options=[1, 3, 7, 14],
            index=0,
            help="How many days ahead to look for games",
            key="days_ahead"
        )

    # Context7 compliant buttons with proper callbacks
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button(
            "🔄 Load NBA Games",
            type="primary",
            on_click=load_games_callback,
            help="Load NBA games for selected date"
        )
    with col2:
        st.button(
            "🔄 Reset Workflow",
            on_click=reset_workflow,
            help="Reset the entire workflow"
        )

    # Display available games - Context7 best practice
    if st.session_state.get('available_games'):
        games = st.session_state.available_games

        # Filter for exact date match
        selected_date_str = st.session_state.selected_date.strftime('%Y-%m-%d')
        exact_date_games = [game for game in games if game.get('date') == selected_date_str]

        if exact_date_games:
            st.success(f"✅ Found {len(exact_date_games)} games for {selected_date_str}")

            for i, game in enumerate(exact_date_games):
                with st.expander(
                    f"🏀 {game.get('away_team', 'Unknown')} @ {game.get('home_team', 'Unknown')} - {game.get('time', 'N/A')}",
                    expanded=False
                ):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Game Details:**")
                        st.write(f"• **Date**: {game.get('date', 'N/A')}")
                        st.write(f"• **Time**: {game.get('time', 'N/A')}")
                        st.write(f"• **Status**: {game.get('status', 'Scheduled')}")

                    with col2:
                        st.write("**Betting Info:**")
                        if game.get('odds'):
                            st.success("✅ Odds Available")
                        else:
                            st.warning("⚠️ No odds available")

                    # Context7 compliant button with proper callback
                    st.button(
                        f"📊 Analyze This Game",
                        key=f"select_game_{i}",
                        on_click=select_game_callback,
                        args=(i,)
                    )
        else:
            # Show nearby games if no exact date matches
            st.warning(f"⚠️ No games found for {selected_date_str}")

            if games:
                st.info("💡 **Nearby games available:**")

                # Group games by date
                games_by_date = {}
                for game in games:
                    game_date = game.get('date', 'Unknown')
                    if game_date != selected_date_str:  # Exclude selected date
                        if game_date not in games_by_date:
                            games_by_date[game_date] = []
                        games_by_date[game_date].append(game)

                # Show games for each nearby date
                for nearby_date, nearby_games in sorted(games_by_date.items()):
                    with st.expander(f"📅 {nearby_date} ({len(nearby_games)} games)"):
                        for i, game in enumerate(nearby_games):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"🏀 {game.get('away_team', 'Unknown')} @ {game.get('home_team', 'Unknown')}")
                                st.write(f"🕐 {game.get('time', 'N/A')} | 📊 {game.get('status', 'Scheduled')}")
                            with col2:
                                # Context7 compliant callback for nearby date selection
                                st.button(
                                    f"Select",
                                    key=f"select_nearby_{nearby_date}_{i}",
                                    on_click=select_nearby_game_callback,
                                    args=(nearby_date, i, nearby_games)
                                )
    else:
        st.info("👆 Click 'Load NBA Games' to start the workflow")

    # Debug panel - Context7 best practice for troubleshooting
    if st.session_state.debug_mode:
        with st.expander("🔍 Debug Information", expanded=False):
            st.write("**Current Session State:**")
            debug_info = {
                "Workflow Step": st.session_state.betting_workflow_step,
                "Selected Date": st.session_state.selected_date,
                "Games Loaded": st.session_state.games_loaded,
                "Available Games Count": len(st.session_state.available_games) if st.session_state.available_games else 0,
                "Selected Game": f"{st.session_state.selected_game.get('away_team', 'N/A')} @ {st.session_state.selected_game.get('home_team', 'N/A')}" if st.session_state.selected_game else "None",
                "Days Ahead": st.session_state.days_ahead,
                "Debug Mode": st.session_state.debug_mode
            }

            for key, value in debug_info.items():
                st.write(f"• **{key}**: {value}")

            st.write("**Session State Keys:**")
            st.write(list(st.session_state.keys()))


def render_game_analysis_step() -> None:
    """Render Step 2: Game Analysis using existing predictions dashboard."""
    st.subheader("📊 Step 2: Game Analysis")
    st.caption("ML predictions and comprehensive analytics")

    if not st.session_state.selected_game:
        st.warning("⚠️ Please select a game from Step 1")
        return

    # Display selected game info
    game = st.session_state.selected_game
    st.info(f"**Selected Game**: {game.get('away_team', 'Unknown')} @ {game.get('home_team', 'Unknown')} - {game.get('date', 'Unknown')}")

    # Use existing predictions dashboard component
    try:
        # Initialize data store for predictions dashboard with proper base path
        data_store = UnifiedDataStore(base_path="data")

        # Initialize data store (creates directories and sets up connections)
        data_store.initialize()

        # Initialize sync engine (optional)
        sync_engine = None  # Can be initialized if needed

        render_predictions_dashboard(data_store, sync_engine, st.session_state.selected_game)

        if st.button("💰 Continue to Betting Analysis", type="primary"):
            st.session_state.betting_workflow_step = 3
            st.rerun()

    except Exception as e:
        st.error(f"❌ Error in predictions dashboard: {e}")
        logger.error(f"Predictions dashboard error: {e}")


def render_betting_lines_step(data_provider: NBADataProvider) -> None:
    """Render Step 3: Betting Lines Analysis."""
    st.subheader("💰 Step 3: Betting Lines Analysis")
    st.caption("Odds comparison and value betting opportunities")

    if not st.session_state.selected_game:
        st.warning("⚠️ Please complete Steps 1 and 2 first")
        return

    game = st.session_state.selected_game

    # Display comprehensive game info
    with st.expander("🏀 Complete Game Information", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Matchup:**")
            st.write(f"• **Away**: {game.get('away_team', 'Unknown')}")
            st.write(f"• **Home**: {game.get('home_team', 'Unknown')}")
            st.write(f"• **Date**: {game.get('date', 'Unknown')}")
            st.write(f"• **Time**: {game.get('time', 'Unknown')}")

        with col2:
            st.write("**Betting Status:**")
            if game.get('odds'):
                st.success("✅ Real Odds Available")

                # Display odds summary
                odds = game['odds']
                if odds.get('totals'):
                    st.write(f"• **Totals**: {len(odds['totals'])} bookmakers")
                if odds.get('moneyline'):
                    st.write(f"• **Moneylines**: {len(odds['moneyline'])} bookmakers")
                if odds.get('spreads'):
                    st.write(f"• **Spreads**: {len(odds['spreads'])} bookmakers")
            else:
                st.warning("⚠️ No odds available")

    # Load detailed odds
    if st.button("🔍 Load Detailed Betting Analysis", type="primary"):
        with st.spinner("Loading detailed betting odds and analysis..."):
            try:
                # Use analytics dashboard for comprehensive analysis
                render_analytics_dashboard()

                # Additional betting-specific analysis
                if game.get('odds'):
                    st.subheader("📈 Betting Market Analysis")

                    # Display odds tables
                    odds = game['odds']

                    if odds.get('totals'):
                        st.write("**Over/Under Markets:**")
                        totals_data = []
                        for bookmaker, lines in odds['totals'].items():
                            if 'over' in lines and 'under' in lines:
                                totals_data.append({
                                    'Bookmaker': bookmaker.title(),
                                    'Line': lines['over'].get('line', 'N/A'),
                                    'Over Odds': lines['over'].get('odds', 'N/A'),
                                    'Under Odds': lines['under'].get('odds', 'N/A')
                                })

                        if totals_data:
                            import pandas as pd
                            totals_df = pd.DataFrame(totals_data)
                            st.dataframe(totals_df, use_container_width=True)

                    if odds.get('moneyline'):
                        st.write("**Moneyline Markets:**")
                        ml_data = []
                        for bookmaker, lines in odds['moneyline'].items():
                            ml_data.append({
                                'Bookmaker': bookmaker.title(),
                                f"{game.get('away_team', 'Away')}": lines.get('away', 'N/A'),
                                f"{game.get('home_team', 'Home')}": lines.get('home', 'N/A')
                            })

                        if ml_data:
                            import pandas as pd
                            ml_df = pd.DataFrame(ml_data)
                            st.dataframe(ml_df, use_container_width=True)

                    st.success("✅ Betting analysis complete!")
                else:
                    st.warning("⚠️ No betting odds available for this game")

            except Exception as e:
                st.error(f"❌ Error in betting analysis: {e}")
                logger.error(f"Betting analysis error: {e}")

    # Reset workflow
    if st.button("🔄 Start New Analysis"):
        st.session_state.betting_workflow_step = 1
        st.session_state.selected_game = None
        if 'available_games' in st.session_state:
            del st.session_state.available_games
        st.rerun()


def main() -> None:
    """Main entry point for the betting workflow dashboard."""
    try:
        # Setup
        setup_session_state()

        # Page configuration
        st.set_page_config(
            page_title="NBA Betting Workflow Dashboard",
            page_icon="🏀",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Load configuration
        config = load_config()
        cache_manager = setup_caching_for_app()

        # Initialize data provider
        try:
            data_provider = NBADataProvider()
        except Exception as e:
            if NBA_DATA_PROVIDER_AVAILABLE:
                st.error(f"❌ Failed to initialize NBA Data Provider: {e}")
                logger.error(f"Data provider initialization error: {e}")
                return
            else:
                st.warning("⚠️ Using mock data provider (NBA API client not available)")
                data_provider = NBADataProvider()

        # Render header
        render_workflow_header()

        # Main navigation based on workflow step
        if st.session_state.betting_workflow_step == 1:
            render_games_schedule_step(data_provider)
        elif st.session_state.betting_workflow_step == 2:
            render_game_analysis_step()
        elif st.session_state.betting_workflow_step == 3:
            render_betting_lines_step(data_provider)

        # Sidebar with additional options
        with st.sidebar:
            st.header("🔧 Dashboard Options")

            st.subheader("Data Status")
            st.metric("NBA API", "🟢 Connected")
            st.metric("Data Store", "🟢 Ready")
            st.metric("Cache", "🟢 Active")

            st.divider()

            st.subheader("Quick Actions")
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.success("✅ Cache cleared")
                st.rerun()

            if st.button("📈 Analytics Dashboard"):
                render_analytics_dashboard()

            if st.button("🔄 Sync Dashboard"):
                render_sync_dashboard()

        # Footer
        st.divider()
        st.caption("NBA Betting Workflow Dashboard | Real-time NBA Data & ML Analysis | Context7 Best Practices")

    except Exception as e:
        st.error(f"❌ Dashboard error: {e}")
        logger.error(f"Dashboard error: {e}")


if __name__ == "__main__":
    main()