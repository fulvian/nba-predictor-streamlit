"""Streamlit component for data synchronization visualization.

This module provides a real-time dashboard for monitoring and controlling
NBA data synchronization operations using Streamlit fragments.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import polars as pl
import streamlit as st

from ...core.data_store import UnifiedDataStore
from ...core.sync_engine import AutomaticSyncEngine
from ...utils.exceptions import StreamlitError

logger = logging.getLogger(__name__)


@st.fragment(run_every=30)
def render_sync_dashboard(
    sync_engine: AutomaticSyncEngine,
    data_store: UnifiedDataStore
) -> None:
    """
    Render data synchronization dashboard with real-time updates.

    Args:
        sync_engine: Automatic sync engine instance
        data_store: Unified data store instance

    Returns:
        None (renders to Streamlit)

    Raises:
        StreamlitError: If rendering fails

    Example:
        >>> render_sync_dashboard(engine, store)
        # Dashboard appears in Streamlit app
    """
    try:
        st.title("🔄 Data Synchronization Dashboard")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Get current sync status
        sync_stats = sync_engine.get_sync_statistics()

        # Render sync status overview
        _render_sync_status_overview(sync_stats, sync_engine)

        # Render data statistics
        _render_data_statistics(sync_stats.get("data_statistics", {}))

        # Render sync controls
        _render_sync_controls(sync_engine)

        # Render recent activity log
        _render_activity_log(sync_stats.get("sync_stats", {}))

    except Exception as e:
        logger.error(f"Failed to render sync dashboard: {e}")
        raise StreamlitError(f"Dashboard rendering failed: {e}", component="sync_dashboard") from e


def _render_sync_status_overview(sync_stats: Dict[str, Any], sync_engine: AutomaticSyncEngine) -> None:
    """Render sync status overview section."""
    st.subheader("📊 Sync Status Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        is_syncing = sync_stats.get("is_syncing", False)
        status_color = "🟢" if not is_syncing else "🔄"
        st.metric("Current Status", f"{status_color} {'Syncing' if is_syncing else 'Idle'}")

    with col2:
        last_sync = sync_stats.get("last_sync")
        last_sync_str = last_sync.strftime("%H:%M:%S") if last_sync else "Never"
        st.metric("Last Sync", last_sync_str)

    with col3:
        sync_interval = sync_stats.get("configuration", {}).get("sync_interval", 3600)
        interval_str = f"{sync_interval // 60}m" if sync_interval >= 60 else f"{sync_interval}s"
        st.metric("Sync Interval", interval_str)

    with col4:
        should_sync = sync_engine.should_sync()
        next_sync_color = "🔴" if should_sync else "🟢"
        st.metric("Next Sync", f"{next_sync_color} {'Due' if should_sync else 'Scheduled'}")

    # Progress bar for next sync
    if sync_stats.get("last_sync"):
        time_since_last = (datetime.now() - sync_stats["last_sync"]).total_seconds()
        progress = min(time_since_last / sync_interval, 1.0)
        st.progress(progress, "Time until next sync")


def _render_data_statistics(data_stats: Dict[str, Any]) -> None:
    """Render data statistics section."""
    st.subheader("📈 Data Statistics")

    if not data_stats:
        st.info("No data statistics available")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Tables", data_stats.get("total_tables", 0))

    with col2:
        total_records = data_stats.get("total_records", 0)
        st.metric("Total Records", f"{total_records:,}")

    with col3:
        last_updated = data_stats.get("last_updated")
        if last_updated:
            last_updated_str = last_updated.strftime("%Y-%m-%d %H:%M")
        else:
            last_updated_str = "Never"
        st.metric("Last Updated", last_updated_str)

    # Show detailed breakdown if available
    try:
        if data_store is not None:
            metadata = data_store.get_metadata()
            if metadata.height > 0:
                st.write("**Data Breakdown:**")

                # Create a summary table
                summary_data = []
                for row in metadata.iter_rows():
                    table_name = row[0].split("_")[0].capitalize()  # Extract data type
                    record_count = row[2]
                    last_updated = row[1].strftime("%m-%d %H:%M") if row[1] else "Never"

                    summary_data.append({
                        "Data Type": table_name,
                        "Records": f"{record_count:,}",
                        "Last Updated": last_updated
                    })

                summary_df = pl.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

    except Exception as e:
        logger.warning(f"Failed to render data breakdown: {e}")


def _render_sync_controls(sync_engine: AutomaticSyncEngine) -> None:
    """Render sync control buttons."""
    st.subheader("🎛️ Sync Controls")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🚀 Start Manual Sync", disabled=sync_engine._is_syncing, use_container_width=True):
            with st.spinner("Starting manual sync..."):
                try:
                    # Create progress callback
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def progress_callback(message: str, progress: float) -> None:
                        progress_bar.progress(progress)
                        status_text.text(message)

                    # Run sync in event loop
                    if "sync_task" not in st.session_state:
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
                            st.success(f"✅ Sync completed! {result['duration_seconds']:.1f}s")
                        else:
                            st.error(f"❌ Sync completed with {len(result['errors'])} errors")

                        # Clear progress
                        progress_bar.empty()
                        status_text.empty()

                        # Trigger rerun to update dashboard
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Sync failed: {e}")
                    logger.error(f"Manual sync failed: {e}")

    with col2:
        if st.button("⚙️ Configuration", use_container_width=True):
            with st.expander("Sync Configuration", expanded=True):
                config = sync_engine.get_sync_statistics()["configuration"]

                col_config1, col_config2 = st.columns(2)

                with col_config1:
                    st.metric("Sync Interval", f"{config['sync_interval']}s")
                    st.metric("Batch Size", f"{config['batch_size']:,}")

                with col_config2:
                    st.metric("Retry Attempts", config['retry_attempts'])

                if st.button("🔄 Reset Configuration", key="reset_config"):
                    st.info("Configuration reset functionality coming soon!")

    with col3:
        if st.button("📊 View Logs", use_container_width=True):
            with st.expander("Recent Sync Activity", expanded=True):
                _render_detailed_activity_log(sync_engine)


def _render_activity_log(sync_stats: Dict[str, Any]) -> None:
    """Render recent activity log section."""
    st.subheader("📋 Recent Activity")

    if not sync_stats:
        st.info("No recent sync activity")
        return

    # Show latest sync results
    if sync_stats.get("start_time"):
        start_time = sync_stats["start_time"]
        duration = sync_stats.get("duration_seconds", 0)
        success = sync_stats.get("success", False)

        col1, col2, col3 = st.columns(3)

        with col1:
            status_icon = "✅" if success else "❌"
            st.metric("Status", f"{status_icon} {'Success' if success else 'Errors'}")

        with col2:
            st.metric("Duration", f"{duration:.1f}s")

        with col3:
            start_time_str = start_time.strftime("%H:%M:%S")
            st.metric("Started", start_time_str)

        # Show data counts
        data_counts = [
            ("Games", sync_stats.get("games_count", 0)),
            ("Players", sync_stats.get("players_count", 0)),
            ("Teams", sync_stats.get("teams_count", 0)),
            ("Odds", sync_stats.get("odds_count", 0))
        ]

        count_cols = st.columns(4)
        for i, (data_type, count) in enumerate(data_counts):
            with count_cols[i]:
                st.metric(data_type, f"{count:,}")

        # Show errors if any
        errors = sync_stats.get("errors", [])
        if errors:
            st.error("❌ Sync Errors:")
            for error in errors:
                st.code(error, language="text")


def _render_detailed_activity_log(sync_engine: AutomaticSyncEngine) -> None:
    """Render detailed activity log with history."""
    # This would typically load from a persistent log storage
    # For now, show current session information

    st.info("📝 Detailed sync log (current session)")

    # Mock log data for demonstration
    log_entries = [
        {"timestamp": datetime.now(), "level": "INFO", "message": "Sync engine initialized"},
        {"timestamp": datetime.now(), "level": "INFO", "message": "Data store connection established"},
    ]

    if sync_engine._last_sync:
        log_entries.append({
            "timestamp": sync_engine._last_sync,
            "level": "INFO",
            "message": f"Last sync completed successfully"
        })

    if not log_entries:
        st.write("No log entries available")
        return

    # Display log entries
    for entry in log_entries:
        timestamp = entry.get("timestamp")
        if isinstance(timestamp, datetime):
            timestamp_str = timestamp.strftime("%H:%M:%S")
        else:
            timestamp_str = str(timestamp)

        level = str(entry.get("level", "INFO"))
        level_icon = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "📝")
        message = str(entry.get("message", ""))

        st.text(f"{level_icon} {timestamp_str} [{level}] {message}")


# Global reference for data store (used in _render_data_statistics)
data_store: Optional[UnifiedDataStore] = None


def set_data_store(store: UnifiedDataStore) -> None:
    """Set the global data store reference for the dashboard."""
    global data_store
    data_store = store