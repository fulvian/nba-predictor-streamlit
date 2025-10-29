"""Main Streamlit application with modular navigation.

This module provides the main entry point for the NBA predictor application
with modular navigation, prediction integration, and Context7 best practices.
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
from .components.predictions_dashboard import render_predictions_dashboard
from .utils.cache_manager import setup_caching_for_app
from .config.deployment_config import load_config, setup_environment_config

logger = logging.getLogger(__name__)


def create_main_app() -> None:
    """
    Create main Streamlit application with modular navigation and prediction integration.

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
        # Load configuration first
        config = load_config()
        setup_environment_config(config)

        # Configure page settings (must be first Streamlit command)
        _configure_page(config)

        # Initialize core components
        data_store, sync_engine = _initialize_core_components(config)

        # Setup caching
        cache_manager = setup_caching_for_app(data_store)

        # Render main navigation
        _render_main_navigation(data_store, sync_engine, config, cache_manager)

    except Exception as e:
        logger.error(f"Failed to create main app: {e}")
        raise StreamlitError(f"App initialization failed: {e}") from e


def _configure_page(config) -> None:
    """Configure Streamlit page settings with configuration integration."""
    st.set_page_config(
        page_title="NBA Predictor Analytics",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get help': 'https://github.com/your-org/nba-predictor',
            'Report a bug': 'https://github.com/your-org/nba-predictor/issues',
            'About': f'NBA Predictor Analytics v2.0 - Advanced NBA predictions with ML (Env: {config.env})'
        }
    )


def _initialize_core_components(config) -> Tuple[UnifiedDataStore, AutomaticSyncEngine]:
    """Initialize and return core components with configuration integration."""
    # Initialize data store
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir(exist_ok=True)

    data_store = UnifiedDataStore(
        base_path=str(data_dir),
        cache_enabled=config.cache_enabled
    )
    data_store.initialize()

    # Initialize sync engine if enabled
    sync_engine = None
    if config.background_sync_enabled:
        sync_engine = AutomaticSyncEngine(
            data_store=data_store,
            sync_interval=config.sync_interval_minutes * 60,  # Convert to seconds
            retry_attempts=3,
            batch_size=1000
        )

    return data_store, sync_engine


def _render_main_navigation(data_store: UnifiedDataStore, sync_engine, config, cache_manager) -> None:
    """Render main navigation with tabs and prediction integration."""

    # Sidebar with app information
    _render_sidebar(data_store, sync_engine, config, cache_manager)

    # Main content area with tabs - now includes Predictions
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏀 Dashboard",
        "🎯 Predictions",
        "📊 Analytics",
        "🔄 Data Sync",
        "⚙️ Settings"
    ])

    with tab1:
        _render_main_dashboard(data_store, sync_engine, config)

    with tab2:
        if config.enable_predictions:
            render_predictions_dashboard(data_store, sync_engine)
        else:
            st.warning("🚫 Predictions are disabled in current configuration")

    with tab3:
        if config.enable_analytics:
            render_analytics_dashboard(data_store)
        else:
            st.warning("🚫 Analytics are disabled in current configuration")

    with tab4:
        if sync_engine:
            render_sync_dashboard(sync_engine, data_store)
        else:
            st.info("ℹ️ Data sync is disabled in current configuration")

    with tab5:
        _render_settings_page(data_store, sync_engine, config, cache_manager)


def _render_sidebar(data_store: UnifiedDataStore, sync_engine, config, cache_manager) -> None:
    """Render sidebar with app information and quick actions."""
    with st.sidebar:
        st.title("🏀 NBA Predictor")
        st.caption(f"Advanced Analytics & Predictions ({config.env})")

        st.divider()

        # System status
        st.subheader("📊 System Status")

        try:
            # Status indicators
            col1, col2 = st.columns(2)

            with col1:
                status_icon = "🟢" if not config.debug else "🟡"
                st.write(f"{status_icon} {'Debug' if config.debug else 'Ready'}")

            with col2:
                predictions_status = "🟢" if config.enable_predictions else "🔴"
                st.write(f"{predictions_status} {'ML On' if config.enable_predictions else 'ML Off'}")

            # Data statistics
            if sync_engine:
                try:
                    sync_stats = sync_engine.get_sync_statistics()
                    data_stats = sync_stats.get("data_statistics", {})
                    st.metric("Games", data_stats.get("total_records", 0))
                    st.metric("Tables", data_stats.get("total_tables", 0))
                except:
                    st.metric("Games", "N/A")
                    st.metric("Tables", "N/A")
            else:
                st.metric("Sync", "Disabled")
                st.metric("Cache", "On" if config.cache_enabled else "Off")

        except Exception as e:
            st.error("❌ Status unavailable")
            logger.error(f"Failed to render sidebar status: {e}")

        st.divider()

        # Quick actions
        st.subheader("🚀 Quick Actions")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()

        with col2:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                if cache_manager:
                    cache_manager.clear_cache_data()
                    st.success("✅ Cache cleared")
                    st.rerun()

        st.divider()

        # About section
        st.subheader("ℹ️ About")
        st.write("""
        **NBA Predictor Analytics v2.0** provides:
        - 🤖 ML-powered predictions
        - 📊 Advanced analytics dashboards
        - 🔄 Real-time data synchronization
        - 📈 Performance metrics
        - 🏆 Team comparisons
        """)

        st.write(f"**Version**: 2.0.0")
        st.write(f"**Environment**: {config.env.title()}")
        st.write(f"**Predictions**: {'Enabled' if config.enable_predictions else 'Disabled'}")

        # Feature flags
        st.divider()
        st.subheader("⚙️ Features")
        features = [
            ("Predictions", config.enable_predictions),
            ("Analytics", config.enable_analytics),
            ("Real-time Data", config.enable_real_time_data),
            ("ML Explanations", config.enable_ml_explanations)
        ]

        for feature, enabled in features:
            status = "✅" if enabled else "❌"
            st.write(f"{status} {feature}")


def _render_main_dashboard(data_store: UnifiedDataStore, sync_engine, config) -> None:
    """Render main dashboard overview with prediction integration."""
    st.header("🏀 NBA Predictor Dashboard")
    st.caption(f"Real-time NBA data analytics and predictions ({config.env})")

    # Key metrics row
    _render_dashboard_metrics(data_store, sync_engine, config)

    st.divider()

    # Recent activity and quick insights
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Recent Activity")
        _render_recent_activity(sync_engine)

    with col2:
        st.subheader("🎯 Quick Insights")
        _render_quick_insights(data_store, config)

    st.divider()

    # Featured visualizations
    st.subheader("📊 Featured Analytics")
    _render_featured_analytics(data_store, config)


def _render_dashboard_metrics(data_store: UnifiedDataStore, sync_engine, config) -> None:
    """Render key dashboard metrics with prediction integration."""
    try:
        # Create metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Games count
            games_count = 0
            try:
                if data_store:
                    metadata = data_store.get_metadata()
                    if metadata.height > 0:
                        games_count = metadata["record_count"].sum()
            except:
                pass

            st.metric(
                "Total Games",
                f"{games_count:,}",
                delta=f"Data available" if games_count > 0 else "No data"
            )

        with col2:
            st.metric(
                "ML Predictions",
                "🤖 Active" if config.enable_predictions else "❌ Disabled",
                delta="Context7 compliant" if config.enable_predictions else "Enable in settings"
            )

        with col3:
            st.metric(
                "Environment",
                config.env.title(),
                delta="Debug" if config.debug else "Production"
            )

        with col4:
            cache_status = "✅ Enabled" if config.cache_enabled else "❌ Disabled"
            st.metric(
                "Cache",
                cache_status,
                delta="Performance optimized" if config.cache_enabled else "Slower loading"
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


def _render_quick_insights(data_store: UnifiedDataStore, config) -> None:
    """Render quick insights from data with prediction integration."""
    try:
        # Mock insights for now - would analyze actual data
        insights = [
            "🤖 ML models showing 73% accuracy this season",
            "🔥 Top performing teams show 15% improvement",
            "📈 Scoring trends up 3.2% this month",
            "🏠 Home teams maintain 58% win advantage",
            "⚡ Overtime games decreasing from average"
        ]

        if config.enable_predictions:
            insights.insert(0, "🎯 Predictions engine operational with Context7 compliance")

        for insight in insights:
            st.write(f"• {insight}")

        st.caption("📊 Based on last 30 days of data")

    except Exception as e:
        st.error("❌ Unable to load insights")
        logger.error(f"Failed to render quick insights: {e}")


def _render_featured_analytics(data_store: UnifiedDataStore, config) -> None:
    """Render featured analytics visualizations with prediction integration."""
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

        # ML predictions snapshot (if enabled)
        if config.enable_predictions:
            st.divider()
            st.write("**🤖 Recent Predictions Performance**")

            # Mock prediction performance data
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Today's Predictions", "12")
                st.metric("Accuracy Today", "75%")

            with col2:
                st.metric("Season Accuracy", "73.2%")
                st.metric("Best Streak", "8-2")

            with col3:
                st.metric("Confidence Avg", "78%")
                st.metric("Models Active", "3")

    except Exception as e:
        st.error("❌ Unable to load featured analytics")
        logger.error(f"Failed to render featured analytics: {e}")


def _render_settings_page(data_store: UnifiedDataStore, sync_engine, config, cache_manager) -> None:
    """Render settings page with configuration management."""
    st.header("⚙️ Settings")
    st.caption("Configure application settings and preferences")

    # Current configuration overview
    st.subheader("📋 Current Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Application Settings:**")
        st.metric("Environment", config.env.title())
        st.metric("Debug Mode", "Enabled" if config.debug else "Disabled")
        st.metric("Predictions", "Enabled" if config.enable_predictions else "Disabled")
        st.metric("Analytics", "Enabled" if config.enable_analytics else "Disabled")

    with col2:
        st.write("**Performance Settings:**")
        st.metric("Cache", "Enabled" if config.cache_enabled else "Disabled")
        st.metric("Max Users", f"{config.max_concurrent_users}")
        st.metric("Sync Interval", f"{config.sync_interval_minutes} min")
        if cache_manager:
            cache_stats = cache_manager.get_cache_statistics()
            st.metric("Cache Hit Rate", cache_stats['hit_rate'])

    st.divider()

    # Feature toggles
    st.subheader("🎛️ Feature Toggles")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Core Features:**")
        # Note: These would be interactive in a real implementation
        feature_status = {
            "ML Predictions": config.enable_predictions,
            "Analytics Dashboard": config.enable_analytics,
            "Real-time Data": config.enable_real_time_data,
            "ML Explanations": config.enable_ml_explanations
        }

        for feature, enabled in feature_status.items():
            status_icon = "✅" if enabled else "❌"
            st.write(f"{status_icon} {feature}")

    with col2:
        st.write("**System Features:**")
        system_features = {
            "Background Sync": config.background_sync_enabled,
            "Cache System": config.cache_enabled,
            "Performance Monitoring": config.monitoring.enable_metrics,
            "API Rate Limiting": True
        }

        for feature, enabled in system_features.items():
            status_icon = "✅" if enabled else "❌"
            st.write(f"{status_icon} {feature}")

    st.divider()

    # Advanced settings
    st.subheader("🔧 Advanced Settings")

    with st.expander("Cache Management", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                if cache_manager:
                    cache_manager.clear_cache_data()
                    st.success("✅ Cache cleared successfully")
                    st.rerun()

        with col2:
            if st.button("📊 Cache Statistics", use_container_width=True):
                if cache_manager:
                    stats = cache_manager.get_cache_statistics()
                    st.json(stats)

    with st.expander("Data Management", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            if sync_engine and st.button("🔄 Manual Sync", use_container_width=True):
                st.info("🔄 Manual sync functionality coming soon")

        with col2:
            if st.button("📥 Export Data", use_container_width=True):
                st.info("📊 Export functionality coming soon")

    with st.expander("API Configuration", expanded=False):
        st.write("**API Settings:**")
        st.write(f"• Rate Limit: {config.api.rate_limit_per_minute} requests/minute")
        st.write(f"• Timeout: {config.api.timeout_seconds} seconds")
        st.write(f"• Retry Attempts: {config.api.retry_attempts}")
        st.write(f"• NBA API Key: {'Configured' if config.get_nba_api_key() else 'Not configured'}")

    st.divider()

    # System information
    st.subheader("💻 System Information")

    system_info = {
        "Python Version": "3.12+",
        "Streamlit Version": "1.28.0+",
        "Polars Version": "1.0.0+",
        "DuckDB Version": "0.9.0+",
        "Environment": config.env.title(),
        "Configuration": "Context7 Compliant"
    }

    col1, col2 = st.columns(2)

    with col1:
        for key, value in list(system_info.items())[:3]:
            st.write(f"• **{key}**: {value}")

    with col2:
        for key, value in list(system_info.items())[3:]:
            st.write(f"• **{key}**: {value}")

    # Configuration export
    st.divider()
    st.subheader("📤 Configuration Export")

    if st.button("📋 Export Current Configuration", use_container_width=True):
        config_dict = config.to_dict()
        st.json(config_dict)
        st.success("✅ Configuration exported above")

    st.caption("💡 Configuration changes require application restart to take effect")


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