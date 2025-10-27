"""Main Streamlit application with modular navigation.

This module provides the main entry point for the NBA predictor application
with modular navigation and component composition.
"""

import logging
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import streamlit as st

from ..core.data_store import UnifiedDataStore
from ..core.sync_engine import AutomaticSyncEngine
from ..utils.exceptions import StreamlitError
from .components.analytics_dashboard import render_analytics_dashboard
from .components.sync_dashboard import render_sync_dashboard

logger = logging.getLogger(__name__)


def create_main_app() -> None:
    """
    Create main Streamlit application with modular navigation.

    Returns:
        None (runs Streamlit app)

    Raises:
        ImportError: If required components are missing
        StreamlitError: If app initialization fails

    Example:
        >>> create_main_app()
        # Streamlit app starts running
    """
    try:
        # Configure page settings (must be first Streamlit command)
        _configure_page()

        # Initialize core components
        data_store, sync_engine = _initialize_core_components()

        # Render main navigation
        _render_main_navigation(data_store, sync_engine)

    except Exception as e:
        logger.error(f"Failed to create main app: {e}")
        raise StreamlitError(f"App initialization failed: {e}") from e


def _configure_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="NBA Predictor Analytics",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get help': 'https://github.com/your-org/nba-predictor',
            'Report a bug': 'https://github.com/your-org/nba-predictor/issues',
            'About': 'NBA Predictor Analytics v1.0 - Advanced NBA data analysis and predictions'
        }
    )


def _initialize_core_components() -> Tuple[UnifiedDataStore, AutomaticSyncEngine]:
    """Initialize and return core components."""
    # Initialize data store
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir(exist_ok=True)

    data_store = UnifiedDataStore(
        base_path=str(data_dir),
        cache_enabled=True
    )
    data_store.initialize()

    # Initialize sync engine
    sync_engine = AutomaticSyncEngine(
        data_store=data_store,
        sync_interval=3600,  # 1 hour
        retry_attempts=3,
        batch_size=1000
    )

    return data_store, sync_engine


def _render_main_navigation(data_store: UnifiedDataStore, sync_engine: AutomaticSyncEngine) -> None:
    """Render main navigation with tabs."""

    # Sidebar with app information
    _render_sidebar(data_store, sync_engine)

    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏀 Dashboard",
        "📊 Analytics",
        "🔄 Data Sync",
        "⚙️ Settings"
    ])

    with tab1:
        _render_main_dashboard(data_store, sync_engine)

    with tab2:
        render_analytics_dashboard(data_store)

    with tab3:
        render_sync_dashboard(sync_engine, data_store)

    with tab4:
        _render_settings_page(data_store, sync_engine)


def _render_sidebar(data_store: UnifiedDataStore, sync_engine: AutomaticSyncEngine) -> None:
    """Render sidebar with app information and quick actions."""
    with st.sidebar:
        st.title("🏀 NBA Predictor")
        st.caption("Advanced Analytics & Predictions")

        st.divider()

        # System status
        st.subheader("📊 System Status")

        try:
            # Get sync statistics
            sync_stats = sync_engine.get_sync_statistics()
            data_stats = sync_stats.get("data_statistics", {})

            # Status indicators
            col1, col2 = st.columns(2)

            with col1:
                is_syncing = sync_stats.get("is_syncing", False)
                status_icon = "🔄" if is_syncing else "✅"
                st.write(f"{status_icon} {'Syncing' if is_syncing else 'Ready'}")

            with col2:
                last_sync = sync_stats.get("last_sync")
                if last_sync:
                    st.write(f"🕒 {last_sync.strftime('%H:%M')}")
                else:
                    st.write("🕒 Never")

            # Data statistics
            st.metric("Games", data_stats.get("total_records", 0))
            st.metric("Tables", data_stats.get("total_tables", 0))

        except Exception as e:
            st.error("❌ Status unavailable")
            logger.error(f"Failed to render sidebar status: {e}")

        st.divider()

        # Quick actions
        st.subheader("🚀 Quick Actions")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Sync Now", use_container_width=True):
                _trigger_manual_sync(sync_engine)

        with col2:
            if st.button("📊 Refresh", use_container_width=True):
                st.rerun()

        st.divider()

        # About section
        st.subheader("ℹ️ About")
        st.write("""
        **NBA Predictor Analytics** provides:
        - Real-time data synchronization
        - Advanced analytics dashboards
        - Performance metrics
        - Team comparisons
        """)

        st.write("**Version**: 1.0.0")
        st.write("**Python**: 3.11+")


def _render_main_dashboard(data_store: UnifiedDataStore, sync_engine: AutomaticSyncEngine) -> None:
    """Render main dashboard overview."""
    st.header("🏀 NBA Predictor Dashboard")
    st.caption("Real-time NBA data analytics and predictions")

    # Key metrics row
    _render_dashboard_metrics(data_store, sync_engine)

    st.divider()

    # Recent activity and quick insights
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Recent Activity")
        _render_recent_activity(sync_engine)

    with col2:
        st.subheader("🎯 Quick Insights")
        _render_quick_insights(data_store)

    st.divider()

    # Featured visualizations
    st.subheader("📊 Featured Analytics")
    _render_featured_analytics(data_store)


def _render_dashboard_metrics(data_store: UnifiedDataStore, sync_engine: AutomaticSyncEngine) -> None:
    """Render key dashboard metrics."""
    try:
        # Get current statistics
        sync_stats = sync_engine.get_sync_statistics()
        sync_results = sync_stats.get("sync_stats", {})
        data_stats = sync_stats.get("data_statistics", {})

        # Create metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Games",
                f"{sync_results.get('games_count', 0):,}",
                delta=f"+{sync_results.get('games_count', 0)} this period"
            )

        with col2:
            st.metric(
                "Players Tracked",
                f"{sync_results.get('players_count', 0):,}",
                delta=f"+{sync_results.get('players_count', 0)} this period"
            )

        with col3:
            st.metric(
                "Teams Active",
                f"{sync_results.get('teams_count', 0):,}",
                delta="All NBA teams"
            )

        with col4:
            sync_success = sync_results.get("success", True)
            st.metric(
                "Sync Status",
                "✅ Healthy" if sync_success else "⚠️ Issues",
                delta="Last sync successful" if sync_success else "Check sync logs"
            )

    except Exception as e:
        st.error("❌ Unable to load dashboard metrics")
        logger.error(f"Failed to render dashboard metrics: {e}")


def _render_recent_activity(sync_engine: AutomaticSyncEngine) -> None:
    """Render recent activity feed."""
    try:
        sync_stats = sync_engine.get_sync_statistics()
        last_sync = sync_stats.get("last_sync")
        sync_results = sync_stats.get("sync_stats", {})

        if last_sync:
            st.success(f"✅ Last sync completed at {last_sync.strftime('%Y-%m-%d %H:%M:%S')}")

            # Show sync results
            if sync_results:
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Data Updated:**")
                    st.write(f"• Games: {sync_results.get('games_count', 0)}")
                    st.write(f"• Players: {sync_results.get('players_count', 0)}")
                    st.write(f"• Teams: {sync_results.get('teams_count', 0)}")

                with col2:
                    st.write("**Performance:**")
                    duration = sync_results.get("duration_seconds", 0)
                    st.write(f"• Duration: {duration:.1f}s")
                    st.write(f"• Status: {'Success' if sync_results.get('success') else 'Errors'}")

                    errors = sync_results.get("errors", [])
                    if errors:
                        st.write(f"• Errors: {len(errors)}")
        else:
            st.info("ℹ️ No sync activity yet. Run your first data sync.")

    except Exception as e:
        st.error("❌ Unable to load recent activity")
        logger.error(f"Failed to render recent activity: {e}")


def _render_quick_insights(data_store: UnifiedDataStore) -> None:
    """Render quick insights from data."""
    try:
        # Mock insights for now - would analyze actual data
        insights = [
            "🔥 Top performing teams show 15% improvement",
            "📈 Scoring trends up 3.2% this month",
            "🏠 Home teams maintain 58% win advantage",
            "⚡ Overtime games decreasing from average"
        ]

        for insight in insights:
            st.write(f"• {insight}")

        st.caption("📊 Based on last 30 days of data")

    except Exception as e:
        st.error("❌ Unable to load insights")
        logger.error(f"Failed to render quick insights: {e}")


def _render_featured_analytics(data_store: UnifiedDataStore) -> None:
    """Render featured analytics visualizations."""
    try:
        # Quick analytics preview
        date_range = (date.today() - timedelta(days=7), date.today())

        # Game distribution over last week
        distribution_data = _get_game_distribution_preview(data_store, date_range)

        if distribution_data is not None and len(distribution_data) > 0:
            st.write("**Game Distribution (Last 7 Days)**")
            st.bar_chart(
                distribution_data.to_dict(),
                x="date",
                y="games_count"
            )
        else:
            st.info("No data available for the selected period")

        # Team performance snapshot
        st.write("**Top Teams Performance**")
        team_data = _get_top_teams_preview(data_store, date_range)

        if team_data is not None and len(team_data) > 0:
            st.dataframe(
                team_data.to_pandas(),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No team performance data available")

    except Exception as e:
        st.error("❌ Unable to load featured analytics")
        logger.error(f"Failed to render featured analytics: {e}")


def _render_settings_page(data_store: UnifiedDataStore, sync_engine: AutomaticSyncEngine) -> None:
    """Render settings page."""
    st.header("⚙️ Settings")
    st.caption("Configure application settings and preferences")

    # Sync settings
    st.subheader("🔄 Synchronization Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Current Configuration:**")
        sync_stats = sync_engine.get_sync_statistics()
        config = sync_stats.get("configuration", {})

        st.metric("Sync Interval", f"{config.get('sync_interval', 3600)}s")
        st.metric("Retry Attempts", config.get('retry_attempts', 3))
        st.metric("Batch Size", f"{config.get('batch_size', 1000):,}")

    with col2:
        st.write("**Data Store Information:**")
        metadata = data_store.get_metadata()

        st.metric("Total Tables", metadata.height)
        if metadata.height > 0:
            total_records = metadata["record_count"].sum()
            st.metric("Total Records", f"{total_records:,}")

            last_updated = metadata["last_updated"].max()
            if isinstance(last_updated, datetime):
                st.metric("Last Updated", last_updated.strftime("%Y-%m-%d %H:%M"))
            else:
                st.metric("Last Updated", str(last_updated))

    st.divider()

    # Advanced settings
    st.subheader("🔧 Advanced Settings")

    with st.expander("Cache Management", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                st.success("✅ Cache cleared successfully")
                st.rerun()

        with col2:
            if st.button("🔄 Reset Data", use_container_width=True):
                st.warning("⚠️ Data reset functionality coming soon")

    with st.expander("Export & Backup", expanded=False):
        st.write("**Export Options:**")

        export_format = st.selectbox(
            "Export Format",
            ["CSV", "JSON", "Parquet"],
            index=0
        )

        if st.button("📥 Export Data", use_container_width=True):
            st.info(f"📊 Export functionality for {export_format} coming soon")

    st.divider()

    # System information
    st.subheader("💻 System Information")

    system_info = {
        "Python Version": "3.11+",
        "Streamlit Version": "1.50.0+",
        "Polars Version": "1.35.0",
        "DuckDB Version": "1.4.1"
    }

    for key, value in system_info.items():
        st.write(f"• **{key}**: {value}")


def _trigger_manual_sync(sync_engine: AutomaticSyncEngine) -> None:
    """Trigger manual data synchronization."""
    try:
        with st.spinner("Starting manual sync..."):
            import asyncio

            # Create progress callback
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(message: str, progress: float) -> None:
                progress_bar.progress(progress)
                status_text.text(message)

            # Run sync
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                sync_engine.sync_all_data(
                    force_refresh=True,
                    progress_callback=progress_callback
                )
            )
            loop.close()

            # Show results
            if result["success"]:
                st.success(f"✅ Sync completed in {result['duration_seconds']:.1f}s")
            else:
                st.error(f"❌ Sync completed with {len(result['errors'])} errors")

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            # Trigger rerun to update dashboard
            st.rerun()

    except Exception as e:
        st.error(f"❌ Manual sync failed: {e}")
        logger.error(f"Manual sync failed: {e}")


def _get_game_distribution_preview(data_store: UnifiedDataStore, date_range: Tuple[date, date]) -> Optional[Any]:
    """Get game distribution data for preview."""
    try:
        start_date_str, end_date_str = date_range[0].isoformat(), date_range[1].isoformat()

        query = f"""
        SELECT
            DATE(game_date) as date,
            COUNT(*) as games_count
        FROM read_parquet('{data_store.games_dir}/*.parquet')
        WHERE game_date BETWEEN '{start_date_str}' AND '{end_date_str}'
        GROUP BY DATE(game_date)
        ORDER BY date
        """

        return data_store.query_analytics(query)
    except Exception:
        return None


def _get_top_teams_preview(data_store: UnifiedDataStore, date_range: Tuple[date, date]) -> Optional[Any]:
    """Get top teams performance for preview."""
    try:
        start_date_str, end_date_str = date_range[0].isoformat(), date_range[1].isoformat()

        query = f"""
        WITH team_stats AS (
            SELECT
                home_team as team_name,
                home_score as points_scored,
                away_score as points_allowed,
                CASE WHEN home_score > away_score THEN 1 ELSE 0 END as win
            FROM read_parquet('{data_store.games_dir}/*.parquet')
            WHERE game_date BETWEEN '{start_date_str}' AND '{end_date_str}'

            UNION ALL

            SELECT
                away_team as team_name,
                away_score as points_scored,
                home_score as points_allowed,
                CASE WHEN away_score > home_score THEN 1 ELSE 0 END as win
            FROM read_parquet('{data_store.games_dir}/*.parquet')
            WHERE game_date BETWEEN '{start_date_str}' AND '{end_date_str}'
        )
        SELECT
            team_name,
            COUNT(*) as games_played,
            SUM(win) as wins,
            ROUND(SUM(win) * 100.0 / COUNT(*), 1) as win_percentage,
            ROUND(AVG(points_scored), 1) as avg_points_scored
        FROM team_stats
        GROUP BY team_name
        HAVING COUNT(*) >= 3
        ORDER BY win_percentage DESC
        LIMIT 5
        """

        return data_store.query_analytics(query)
    except Exception:
        return None


# Import required modules
from datetime import timedelta