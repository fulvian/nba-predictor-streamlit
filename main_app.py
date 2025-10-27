"""Main entry point for NBA Predictor Streamlit application.

This module provides a proper entry point that can be executed directly
with `streamlit run main_app.py` to avoid relative import issues.
"""

import sys
from pathlib import Path

# Add src directory to Python path for absolute imports
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import Streamlit and the data provider
import streamlit as st
from datetime import date, datetime
try:
    from data_provider import NBADataProvider
except ImportError:
    NBADataProvider = None
try:
    from nba_timezone_utils import NBATimezoneManager, generate_nba_schedule_fallback
except ImportError:
    NBATimezoneManager = None
    generate_nba_schedule_fallback = None
# from mock_data_provider import create_nba_data_provider  # Disabilitato - usiamo solo API reali

def create_modern_dashboard():
    """Create a modern dashboard that uses the backend data provider."""

    # Configure page
    st.set_page_config(
        page_title="NBA Predictor Analytics - Modern Dashboard",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize data provider - SOLO API REALI
    try:
        data_provider = NBADataProvider()
        st.success("✅ Connected to BallDontLie API for real NBA games")
    except Exception as e:
        st.error(f"❌ Failed to initialize NBA data provider: {e}")
        st.stop()

    # Sidebar with navigation
    st.sidebar.title("🏀 NBA Predictor Analytics")
    st.sidebar.caption("Modern Dashboard with Real-time Data")

    # Main navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏀 Games Schedule",
        "📊 Analytics",
        "💰 Betting Odds",
        "🔧 System Status"
    ])

    with tab1:
        render_games_schedule_with_date_range(data_provider)

    with tab2:
        render_analytics(data_provider)

    with tab3:
        render_betting_odds(data_provider)

    with tab4:
        render_system_status(data_provider)

def render_games_schedule_with_date_range(data_provider):
    """
    Render enhanced games schedule with date range selection.

    Features:
    - Single date selection (existing)
    - Date range selection (1-5 days)
    - Real NBA games data from BallDontLie API
    - Rate limiting status indicator
    """
    st.header("📅 NBA Games Schedule")
    st.caption("Real NBA games from official schedule with BallDontLie API")

    # Initialize timezone manager with fallback
    if NBATimezoneManager:
        tz_manager = NBATimezoneManager()
    else:
        # Simple timezone fallback
        from dateutil import tz
        class SimpleTimezoneManager:
            def convert_utc_to_local(self, utc_dt, team_name):
                eastern = tz.gettz('America/New_York')
                local_dt = utc_dt.astimezone(eastern)
                return local_dt, 'Eastern (Fallback)'

            def get_game_times_by_timezone(self, utc_dt):
                return {
                    'UTC': utc_dt.strftime('%H:%M UTC'),
                    'Eastern': utc_dt.astimezone(tz.gettz('America/New_York')).strftime('%H:%M ET'),
                    'Pacific': utc_dt.astimezone(tz.gettz('America/Los_Angeles')).strftime('%H:%M PT')
                }
        tz_manager = SimpleTimezoneManager()

    # Date range selection options
    st.subheader("📆 Date Selection")

    col1, col2 = st.columns([1, 2])

    with col1:
        date_range_option = st.selectbox(
            "Select Date Range:",
            ["Single Date", "Next 3 Days", "Next 5 Days"],
            help="Choose how many days of games to display"
        )

    with col2:
        if date_range_option == "Single Date":
            selected_date = st.date_input("Select Date", value=date.today())
            days_ahead = 1
            specific_date = selected_date.strftime('%Y-%m-%d')
        else:
            days_ahead = 3 if date_range_option == "Next 3 Days" else 5
            specific_date = None
            st.info(f"Showing games for next {days_ahead} days from today")

    # Additional options
    show_timezone_info = st.checkbox("🌍 Show timezone details", value=True)
    show_rate_limit_status = st.checkbox("🚦 Show API status", value=True)

    # Load games button
    if st.button("🔄 Load NBA Games", type="primary"):
        with st.spinner(f"Loading NBA games..."):
            try:
                # Use the new enhanced data provider
                st.write("🏀 Connecting to BallDontLie API for real NBA schedule...")
                games = data_provider.get_scheduled_games(days_ahead=days_ahead, specific_date=specific_date)

                if games:
                    # Determine data source
                    bdl_games = [g for g in games if 'BallDontLie' in g.get('source', '')]
                    odds_games = [g for g in games if 'The Odds' in g.get('source', '')]
                    nba_games = [g for g in games if 'NBA Live' in g.get('source', '')]

                    if bdl_games:
                        data_source = "BallDontLie API (Official NBA Schedule)"
                        st.success(f"✅ Found {len(bdl_games)} real NBA games from official schedule")
                    elif odds_games:
                        data_source = "The Odds API (Betting Odds - Fallback)"
                        st.info(f"📊 Found {len(odds_games)} games with betting odds")
                    else:
                        data_source = "NBA Live API (Completed Games - Fallback)"
                        st.info(f"📊 Found {len(nba_games)} completed games")
                else:
                    st.warning("⚠️ No games found from any source")
                    data_source = "No Data Available"

                # Enhanced timezone processing
                processed_games = []
                for game in games:
                    try:
                        # Parse UTC time and convert to local timezones
                        if 'utc_datetime' in game:
                            utc_dt = datetime.fromisoformat(game['utc_datetime'].replace('Z', '+00:00'))
                        else:
                            # Fallback parsing for older format
                            from dateutil import parser
                            utc_dt = parser.parse(game['time_utc'])

                        # Get local times for both teams
                        home_local, home_tz = tz_manager.convert_utc_to_local(utc_dt, game['home_team'])
                        away_local, away_tz = tz_manager.convert_utc_to_local(utc_dt, game['away_team'])

                        # Use UTC date for primary filtering (matches user expectation)
                        utc_date = utc_dt.date().strftime('%Y-%m-%d')
                        local_date = home_local.strftime('%Y-%m-%d')

                        # Enhance game data with timezone info
                        enhanced_game = game.copy()
                        enhanced_game.update({
                            'utc_date': utc_date,  # Use for primary filtering
                            'local_date': local_date,  # Display local time info
                            'home_local_time': home_local.strftime('%H:%M'),
                            'away_local_time': away_local.strftime('%H:%M'),
                            'home_timezone': home_tz,
                            'away_timezone': away_tz,
                            'all_timezones': tz_manager.get_game_times_by_timezone(utc_dt),
                            'utc_datetime_iso': utc_dt.isoformat()
                        })

                        processed_games.append(enhanced_game)

                    except Exception as e:
                        print(f"⚠️ Error processing game {game.get('game_id', 'unknown')}: {e}")
                        # Still add original game if timezone processing fails
                        enhanced_game = game.copy()
                        enhanced_game['utc_date'] = game.get('date', '')  # Use original date
                        enhanced_game['local_date'] = game.get('date', '')
                        enhanced_game['home_local_time'] = game.get('time', 'Unknown')
                        processed_games.append(enhanced_game)

                # Filter games by selected date (NBA schedule date filtering)
                selected_date_str = selected_date.strftime('%Y-%m-%d')

                # Primary filtering: use original NBA schedule date from 'date' field
                # This ensures we show all NBA games scheduled for that calendar day
                nba_schedule_games = [
                    game for game in games
                    if game.get('date') == selected_date_str
                ]

                # Secondary filtering: also check processed games for timezone matches
                selected_games = []
                for game in processed_games:
                    game_date = game.get('date')  # Original NBA date
                    game_utc_date = game.get('utc_date')
                    game_local_date = game.get('local_date')

                    # Include if matches NBA schedule date OR timezone-processed dates
                    if (game_date == selected_date_str or
                        game_utc_date == selected_date_str or
                        game_local_date == selected_date_str):
                        selected_games.append(game)

                # Remove duplicates (some games might match multiple criteria)
                unique_games = []
                seen_ids = set()
                for game in selected_games:
                    game_id = game.get('game_id')
                    if game_id and game_id not in seen_ids:
                        unique_games.append(game)
                        seen_ids.add(game_id)
                selected_games = unique_games

                # Debug info
                matching_schedule = len(nba_schedule_games)
                matching_processed = len(selected_games)
                st.write(f"🏀 **NBA schedule date**: {matching_schedule} games")
                st.write(f"🎯 **Total unique games**: {matching_processed} games")

                # Show comprehensive debug info
                st.info(f"📊 **Data Source**: {data_source}")
                st.write(f"🎯 **Looking for games on**: {selected_date_str}")

                # Show available dates (both UTC and local)
                if processed_games:
                    utc_dates = sorted(set(game.get('utc_date') for game in processed_games if game.get('utc_date')))
                    local_dates = sorted(set(game.get('local_date') for game in processed_games if game.get('local_date')))
                    st.write(f"📅 **Available UTC dates**: {', '.join(utc_dates)}")
                    if local_dates != utc_dates:
                        st.write(f"🌍 **Available local dates**: {', '.join(local_dates)}")

                if selected_games:
                    st.success(f"🏀 Found {len(selected_games)} games for {selected_date} (UTC)")

                    for game in selected_games:
                        # Display both UTC time and local time
                        display_time = game.get('home_local_time', game.get('time', 'Unknown'))
                        utc_time_info = game.get('time_utc', 'Unknown')

                        with st.expander(f"🏀 {game['away_team']} @ {game['home_team']} - {display_time} (Local)", expanded=True):
                            col1, col2 = st.columns(2)

                            with col1:
                                st.write("**📅 Game Details:**")
                                st.write(f"• **UTC Date**: {game.get('utc_date', game.get('date', 'Unknown'))}")
                                st.write(f"• **Local Date**: {game.get('local_date', game.get('date', 'Unknown'))}")
                                st.write(f"• **UTC Time**: {utc_time_info}")
                                st.write(f"• **Home Time**: {game.get('home_local_time', game.get('time', 'Unknown'))} ({game.get('home_timezone', 'Unknown')})")
                                st.write(f"• **Away Time**: {game.get('away_local_time', game.get('time', 'Unknown'))} ({game.get('away_timezone', 'Unknown')})")
                                st.write(f"• **Status**: {game.get('status', 'Unknown')}")
                                st.write(f"• **Source**: {game.get('source', 'Unknown')}")

                                # Show timezone conversion details if requested
                                if show_timezone_info and 'all_timezones' in game:
                                    st.write("**🌍 All Timezones:**")
                                    for tz_label, tz_time in game['all_timezones'].items():
                                        st.write(f"• {tz_label}: {tz_time}")

                            with col2:
                                st.write("**💰 Betting Information:**")
                                if game.get('odds') and game['odds'].get('moneyline'):
                                    st.write("**Moneyline Odds:**")
                                    for team, odd in game['odds']['moneyline'].items():
                                        st.write(f"• {team}: **{odd['price']}** ({odd['bookmaker']})")
                                else:
                                    st.write("No odds available")

                                st.metric("Bookmakers", game.get('bookmakers_count', 0))

                else:
                    st.info(f"ℹ️ No games found for {selected_date} (UTC)")

                    # Show nearby games with timezone info
                    if processed_games:
                        nearby_games = []
                        for game in processed_games:
                            try:
                                game_date = date.fromisoformat(game.get('utc_date', game.get('date', '2025-01-01')))
                                if abs(game_date - selected_date).days <= 3:
                                    nearby_games.append(game)
                            except:
                                continue

                        if nearby_games:
                            st.write("**📍 Nearby Games (±3 days):**")
                            for game in nearby_games[:10]:
                                time_info = f"{game.get('home_local_time', game.get('time', 'Unknown'))} ({game.get('home_timezone', 'Unknown')})"
                                utc_date = game.get('utc_date', game.get('date', 'Unknown'))
                                st.write(f"• {utc_date}: {game['away_team']} @ {game['home_team']} - {time_info}")

            except Exception as e:
                st.error(f"❌ Error loading games: {e}")
                st.exception(e)

def render_analytics(data_provider):
    """Render analytics dashboard."""
    st.header("📊 NBA Analytics")
    st.caption("Advanced analytics and insights")

    tabs = st.tabs(["📈 Trends", "🏆 Teams", "👥 Players"])

    with tabs[0]:
        st.subheader("League Trends")
        st.info("📊 Analytics features coming soon - powered by DuckDB + Polars")

    with tabs[1]:
        st.subheader("Team Performance")
        st.info("📊 Team analytics coming soon")

    with tabs[2]:
        st.subheader("Player Statistics")
        st.info("📊 Player analytics coming soon")

def render_betting_odds(data_provider):
    """Render betting odds information."""
    st.header("💰 Betting Odds Analysis")
    st.caption("Real-time odds from multiple bookmakers")

    try:
        # Get current odds
        games = data_provider._get_odds_api_games(days_ahead=7)

        if games:
            st.success(f"✅ Current odds available for {len(games)} upcoming games")

            # Show odds summary
            total_bookmakers = sum(g.get('bookmakers_count', 0) for g in games)
            st.metric("Total Games", len(games))
            st.metric("Total Bookmaker Markets", total_bookmakers)

            # Detailed odds for first few games
            for game in games[:3]:
                with st.expander(f"🎰 {game['away_team']} @ {game['home_team']}"):
                    if game.get('odds') and game['odds'].get('moneyline'):
                        st.write("**Moneyline Odds:**")
                        for team, odd in game['odds']['moneyline'].items():
                                    st.write(f"• {team}: **{odd['price']}** ({odd['bookmaker']})")
        else:
            st.warning("⚠️ No odds data available")

    except Exception as e:
        st.error(f"❌ Error loading odds: {e}")

def render_system_status(data_provider):
    """Render system status information."""
    st.header("🔧 System Status")
    st.caption("Backend system information")

    st.subheader("📡 Data Sources")

    # Test The Odds API
    try:
        response = data_provider.odds_session.get(
            f"{data_provider.odds_base_url}/sports",
            headers=data_provider.odds_headers,
            timeout=10
        )
        if response.status_code == 200:
            st.success("✅ The Odds API: Connected")
        else:
            st.error(f"❌ The Odds API: Error {response.status_code}")
    except Exception as e:
        st.error(f"❌ The Odds API: {e}")

    # Show API configuration
    with st.expander("🔧 API Configuration"):
        st.write("**The Odds API Configuration:**")
        st.write(f"• Base URL: {data_provider.odds_base_url}")
        st.write(f"• API Key: {'*' * 20}{data_provider.odds_api_key[-4:]}")
        st.write(f"• NBA Teams Loaded: {len(data_provider.nba_teams_info)}")
        st.write(f"• Session Active: {data_provider.odds_session is not None}")

if __name__ == "__main__":
    create_modern_dashboard()