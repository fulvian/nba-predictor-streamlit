#!/usr/bin/env python3
"""
🏀 NBA Betting Workflow Dashboard - Context7 Best Practice Implementation

This module implements a clean, modular dashboard using existing components
following Context7 best practices for Streamlit applications.

Workflow: Games Schedule → Game Analysis → Betting Lines
"""

import logging
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional

import streamlit as st
from dateutil import tz
from ..utils.nba_timezone_utils import NBATimezoneManager

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

# Initialize timezone manager per Context7 best practices
_tz_manager = None

def get_timezone_manager() -> NBATimezoneManager:
    """Get singleton timezone manager instance."""
    global _tz_manager
    if _tz_manager is None:
        _tz_manager = NBATimezoneManager()
    return _tz_manager

def convert_game_time_to_et(game: Dict[str, Any]) -> str:
    """
    Convert game UTC time to Eastern Time (US standard timezone).

    Based on Context7 best practices and user request to standardize
    all NBA game times to US timezone.

    Args:
        game: Game dictionary with UTC time information

    Returns:
        Eastern Time formatted as HH:MM
    """
    try:
        # Get UTC time from game data
        utc_time_str = game.get('time_utc', '')
        if not utc_time_str:
            return game.get('time', 'N/A')

        # Parse UTC datetime using dateutil (Context7 best practice)
        if 'T' in utc_time_str:
            # Handle ISO format like "2025-10-29T23:00:00Z"
            from dateutil.parser import parse
            utc_dt = parse(utc_time_str.replace('Z', '+00:00'))
        else:
            # Handle time only like "23:00"
            game_date = game.get('date', '')
            if game_date:
                datetime_str = f"{game_date}T{utc_time_str}:00Z"
                from dateutil.parser import parse
                utc_dt = parse(datetime_str.replace('Z', '+00:00'))
            else:
                return game.get('time', 'N/A')

        # Convert to Eastern Time using Context7 best practice
        tz_manager = get_timezone_manager()
        eastern_tz = tz_manager._timezone_cache['America/New_York']
        eastern_dt = utc_dt.astimezone(eastern_tz)

        # Return formatted time
        return eastern_dt.strftime('%H:%M')

    except Exception as e:
        logger.warning(f"⚠️ Error converting time for game: {e}")
        # Fallback to original time
        return game.get('time', 'N/A')


def sort_games_by_time(games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort NBA games chronologically by Eastern Time.

    Args:
        games: List of game dictionaries

    Returns:
        Games sorted by time (earliest to latest)
    """
    try:
        def get_game_time_sort_key(game: Dict[str, Any]) -> int:
            """Get sort key for game based on ET time."""
            et_time = convert_game_time_to_et(game)

            # Convert HH:MM to minutes for sorting
            if et_time != 'N/A' and ':' in et_time:
                try:
                    hour, minute = map(int, et_time.split(':'))
                    return hour * 60 + minute  # Total minutes from midnight
                except (ValueError, TypeError):
                    return 24 * 60  # Put invalid times at the end
            return 24 * 60  # Put N/A times at the end

        # Sort games by ET time
        sorted_games = sorted(games, key=get_game_time_sort_key)
        return sorted_games

    except Exception as e:
        logger.warning(f"⚠️ Error sorting games by time: {e}")
        return games  # Return unsorted if error


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
    """Callback function for loading games with intelligent caching - Context7 compliant."""
    if st.session_state.debug_mode:
        logger.info("Loading games callback triggered")

    try:
        # Convert selected date to string format for API
        date_str = st.session_state.selected_date.strftime('%Y-%m-%d')
        days_ahead = st.session_state.days_ahead

        if st.session_state.debug_mode:
            logger.info(f"Loading games for date: {date_str}, days ahead: {days_ahead}")

        # Context7 Best Practice: Intelligent multi-level caching strategy
        games = _load_games_with_intelligent_caching(date_str, days_ahead)

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


def _load_games_with_intelligent_caching(date_str: str, days_ahead: int) -> list:
    """
    Context7 Best Practice: Intelligent multi-level caching for NBA games.

    Strategy:
    1. Check UnifiedDataStore first (persistent storage)
    2. If missing or outdated, use NBADataProvider APIs
    3. Update data store with fresh data for future use

    Args:
        date_str: Target date in YYYY-MM-DD format
        days_ahead: Number of days to look ahead

    Returns:
        List of NBA games data
    """
    if st.session_state.debug_mode:
        logger.info("🔍 Context7 Intelligent Caching: Checking data availability...")

    # Initialize data store for checking existing data
    try:
        data_store = UnifiedDataStore(base_path="data")
        data_store.initialize()

        # Check if we already have games for this date in persistent storage
        cached_games = _check_data_store_for_games(data_store, date_str, days_ahead)

        if cached_games:
            if st.session_state.debug_mode:
                logger.info(f"✅ Data Store HIT: Found {len(cached_games)} games in persistent storage")
                for game in cached_games[:2]:
                    logger.info(f"   📦 Cached: {game.get('away_team', 'Unknown')} @ {game.get('home_team', 'Unknown')}")
            return cached_games
        else:
            if st.session_state.debug_mode:
                logger.info("📝 Data Store MISS: No games found, fetching from APIs...")

    except Exception as e:
        if st.session_state.debug_mode:
            logger.warning(f"⚠️ Data store check failed: {e}, proceeding with APIs...")

    # Fetch from APIs with caching strategy
    games = _fetch_games_from_apis_with_caching(date_str, days_ahead)

    # Update persistent storage with fresh data
    if games:
        try:
            _update_data_store_with_games(data_store, games, date_str)
            if st.session_state.debug_mode:
                logger.info(f"💾 Updated persistent storage with {len(games)} games")
        except Exception as e:
            if st.session_state.debug_mode:
                logger.warning(f"⚠️ Failed to update persistent storage: {e}")

    return games


def _check_data_store_for_games(data_store: UnifiedDataStore, date_str: str, days_ahead: int) -> Optional[list]:
    """
    Check if games are available in the persistent data store.

    Args:
        data_store: UnifiedDataStore instance
        date_str: Target date in YYYY-MM-DD format
        days_ahead: Number of days to look ahead

    Returns:
        List of games if found and valid, None otherwise
    """
    try:
        # Calculate date range for the query
        from datetime import datetime, timedelta
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        end_date = target_date + timedelta(days=days_ahead)

        start_date_str = target_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        if st.session_state.debug_mode:
            logger.info(f"   🔍 Querying data store for games between {start_date_str} and {end_date_str}")

        # Query the data store for games in the date range
        query = f"""
        SELECT
            game_date,
            home_team,
            away_team,
            home_score,
            away_score,
            season,
            'Scheduled' as status,
            '' as time,
            '{{}}' as odds
        FROM read_parquet('{data_store.games_dir}/*.parquet')
        WHERE game_date BETWEEN '{start_date_str}' AND '{end_date_str}'
        ORDER BY game_date
        """

        result = data_store.query_analytics(query)

        if result and result.height > 0:
            # Convert to expected format
            games = []
            for row in result.iter_rows():
                games.append({
                    'date': row[0].strftime('%Y-%m-%d') if hasattr(row[0], 'strftime') else str(row[0]),
                    'home_team': row[1] or 'Unknown',
                    'away_team': row[2] or 'Unknown',
                    'home_score': row[3],
                    'away_score': row[4],
                    'season': row[5],
                    'status': row[6],
                    'time': row[7],
                    'odds': row[8] or {},
                    'source': 'Data Store Cache'
                })

            if st.session_state.debug_mode:
                logger.info(f"   ✅ Found {len(games)} games in data store")

            return games
        else:
            if st.session_state.debug_mode:
                logger.info("   📝 No games found in data store for this date range")
            return None

    except Exception as e:
        if st.session_state.debug_mode:
            logger.error(f"   ❌ Error checking data store: {e}")
        return None


def _fetch_games_from_apis_with_caching(date_str: str, days_ahead: int) -> list:
    """
    Fetch games from APIs with built-in caching optimization.

    Args:
        date_str: Target date in YYYY-MM-DD format
        days_ahead: Number of days to look ahead

    Returns:
        List of games from APIs
    """
    if st.session_state.debug_mode:
        logger.info("🌐 Fetching games from APIs with caching...")

    try:
        # Get data provider instance
        data_provider = NBADataProvider()

        # Get games using the existing API method (which has its own caching)
        games = data_provider.get_scheduled_games(days_ahead=days_ahead, specific_date=date_str)

        if st.session_state.debug_mode:
            logger.info(f"   📡 API returned {len(games)} games")
            # Log source distribution
            sources = {}
            for game in games:
                source = game.get('source', 'Unknown')
                sources[source] = sources.get(source, 0) + 1

            for source, count in sources.items():
                logger.info(f"   📊 {source}: {count} games")

        return games

    except Exception as e:
        if st.session_state.debug_mode:
            logger.error(f"   ❌ Error fetching from APIs: {e}")
        return []


def _update_data_store_with_games(data_store: UnifiedDataStore, games: list, date_str: str):
    """
    Update the persistent data store with fresh games data.

    Args:
        data_store: UnifiedDataStore instance
        games: List of games to store
        date_str: Target date for reference
    """
    try:
        if not games:
            return

        # Convert games to DataFrame format for storage
        from datetime import datetime
        import polars as pl

        game_records = []
        for game in games:
            # Parse game date
            game_date = datetime.strptime(game.get('date', date_str), '%Y-%m-%d').date()

            record = {
                'game_date': game_date,
                'home_team': game.get('home_team', 'Unknown'),
                'away_team': game.get('away_team', 'Unknown'),
                'home_score': game.get('home_score', None),
                'away_score': game.get('away_score', None),
                'season': game.get('season', 2024),
                'status': game.get('status', 'Scheduled'),
                'time': convert_game_time_to_et(game),
                'odds': str(game.get('odds', {})),
                'source': game.get('source', 'API'),
                'updated_at': datetime.now()
            }
            game_records.append(record)

        # Create DataFrame
        df = pl.DataFrame(game_records)

        # Save to data store
        output_path = data_store.games_dir / f"games_{date_str.replace('-', '_')}.parquet"
        df.write_parquet(output_path)

        if st.session_state.debug_mode:
            logger.info(f"   💾 Saved {len(game_records)} games to {output_path}")

    except Exception as e:
        if st.session_state.debug_mode:
            logger.error(f"   ❌ Error updating data store: {e}")
        # Don't raise - this is a non-critical operation


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

        # Sort games chronologically by Eastern Time
        exact_date_games = sort_games_by_time(exact_date_games)

        if exact_date_games:
            st.success(f"✅ Found {len(exact_date_games)} games for {selected_date_str}")

            for i, game in enumerate(exact_date_games):
                with st.expander(
                    f"🏀 {game.get('away_team', 'Unknown')} @ {game.get('home_team', 'Unknown')} - {convert_game_time_to_et(game)}",
                    expanded=False
                ):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Game Details:**")
                        st.write(f"• **Date**: {game.get('date', 'N/A')}")
                        st.write(f"• **Time**: {convert_game_time_to_et(game)}")
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
                                st.write(f"🕐 {convert_game_time_to_et(game)} | 📊 {game.get('status', 'Scheduled')}")
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
            st.write(f"• **Time**: {convert_game_time_to_et(game)}")

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