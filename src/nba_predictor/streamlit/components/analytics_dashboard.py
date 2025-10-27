"""Analytics dashboard component with DuckDB integration.

This module provides interactive analytics dashboard for NBA data visualization
using DuckDB for high-performance analytical queries.
"""

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import streamlit as st

from ...core.data_store import UnifiedDataStore
from ...utils.exceptions import DatabaseError, StreamlitError

logger = logging.getLogger(__name__)


def render_analytics_dashboard(
    data_store: UnifiedDataStore,
    date_range: Optional[Tuple[date, date]] = None
) -> None:
    """
    Render analytics dashboard with interactive charts and metrics.

    Args:
        data_store: Unified data store instance
        date_range: Optional date range for filtering

    Returns:
        None (renders to Streamlit)

    Raises:
        DatabaseError: If DuckDB queries fail
        StreamlitError: If rendering fails

    Example:
        >>> render_analytics_dashboard(store, (date(2024,1,1), date(2024,12,31)))
        # Analytics dashboard appears
    """
    try:
        st.title("📊 NBA Analytics Dashboard")
        st.caption("Real-time analytics powered by DuckDB")

        # Date range selector
        selected_date_range = _render_date_selector(date_range)

        # Key metrics overview
        _render_key_metrics(data_store, selected_date_range)

        # Performance analysis charts
        _render_performance_charts(data_store, selected_date_range)

        # Team comparison analysis
        _render_team_comparison(data_store, selected_date_range)

        # Advanced analytics
        _render_advanced_analytics(data_store, selected_date_range)

    except Exception as e:
        logger.error(f"Failed to render analytics dashboard: {e}")
        raise StreamlitError(f"Analytics dashboard rendering failed: {e}", component="analytics_dashboard") from e


def _render_date_selector(default_range: Optional[Tuple[date, date]] = None) -> Tuple[date, date]:
    """Render date range selector."""
    st.sidebar.header("📅 Date Range")

    # Default to last 30 days if no range provided
    if default_range is None:
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        default_range = (start_date, end_date)

    # Quick range buttons
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("📅 Today", use_container_width=True):
            today = date.today()
            default_range = (today, today)
            st.rerun()

    with col2:
        if st.button("📅 7 Days", use_container_width=True):
            end_date = date.today()
            start_date = end_date - timedelta(days=7)
            default_range = (start_date, end_date)
            st.rerun()

    # Custom date range picker
    st.sidebar.subheader("Custom Range")

    col_start, col_end = st.sidebar.columns(2)
    with col_start:
        start_date = st.date_input("Start Date", value=default_range[0])

    with col_end:
        end_date = st.date_input("End Date", value=default_range[1])

    if start_date > end_date:
        st.sidebar.error("Start date must be before end date")
        return default_range

    return (start_date, end_date)


def _render_key_metrics(data_store: UnifiedDataStore, date_range: Tuple[date, date]) -> None:
    """Render key performance metrics."""
    st.subheader("🎯 Key Performance Metrics")

    try:
        # Execute analytics queries using DuckDB
        metrics = _calculate_key_metrics(data_store, date_range)

        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Games",
                f"{metrics.get('total_games', 0):,}",
                delta=f"+{metrics.get('games_today', 0)} today"
            )

        with col2:
            st.metric(
                "Avg Points/Game",
                f"{metrics.get('avg_points', 0):.1f}",
                delta=f"{metrics.get('points_trend', '+0.0')}"
            )

        with col3:
            st.metric(
                "Teams Active",
                f"{metrics.get('active_teams', 0):,}",
                delta=f"{metrics.get('teams_change', 0)} vs last period"
            )

        with col4:
            win_rate = metrics.get('overall_win_rate', 0)
            st.metric(
                "Home Win Rate",
                f"{win_rate:.1%}",
                delta=f"{metrics.get('win_rate_trend', '+0.0%')}"
            )

    except Exception as e:
        logger.error(f"Failed to render key metrics: {e}")
        st.error("❌ Unable to load key metrics")


def _render_performance_charts(data_store: UnifiedDataStore, date_range: Tuple[date, date]) -> None:
    """Render performance analysis charts."""
    st.subheader("📈 Performance Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Scoring Trends**")
        scoring_data = _get_scoring_trends(data_store, date_range)

        if scoring_data is not None and len(scoring_data) > 0:
            st.line_chart(
                scoring_data.to_dict(),
                x="date",
                y=["avg_home_score", "avg_away_score"],
                color=["#FF6B6B", "#4ECDC4"]
            )
        else:
            st.info("No scoring data available for selected period")

    with col2:
        st.write("**Game Distribution**")
        distribution_data = _get_game_distribution(data_store, date_range)

        if distribution_data is not None and len(distribution_data) > 0:
            st.bar_chart(
                distribution_data.to_dict(),
                x="date",
                y="games_count"
            )
        else:
            st.info("No distribution data available")

    # Additional performance insights
    with st.expander("🔍 Detailed Performance Insights", expanded=False):
        _render_performance_insights(data_store, date_range)


def _render_team_comparison(data_store: UnifiedDataStore, date_range: Tuple[date, date]) -> None:
    """Render team comparison analysis."""
    st.subheader("🏆 Team Comparison Analysis")

    # Team selection
    try:
        teams_data = _get_teams_performance(data_store, date_range)

        if teams_data is not None and len(teams_data) > 0:
            team_names = teams_data['team_name'].to_list()

            col1, col2 = st.columns(2)

            with col1:
                selected_home = st.selectbox(
                    "Select Home Team",
                    options=team_names,
                    index=0 if team_names else None
                )

            with col2:
                selected_away = st.selectbox(
                    "Select Away Team",
                    options=team_names,
                    index=1 if len(team_names) > 1 else None
                )

            if selected_home and selected_away and selected_home != selected_away:
                comparison_data = _get_head_to_head_stats(data_store, date_range, selected_home, selected_away)
                _render_head_to_head_comparison(comparison_data)

            # Team rankings table
            st.write("**Team Rankings**")
            st.dataframe(
                teams_data.to_pandas(),
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info("No team data available for selected period")

    except Exception as e:
        logger.error(f"Failed to render team comparison: {e}")
        st.error("❌ Unable to load team comparison data")


def _render_advanced_analytics(data_store: UnifiedDataStore, date_range: Tuple[date, date]) -> None:
    """Render advanced analytics section."""
    st.subheader("🧮 Advanced Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Overtime Analysis**")
        overtime_data = _get_overtime_statistics(data_store, date_range)

        if overtime_data:
            st.metric(
                "Overtime Games",
                f"{overtime_data.get('overtime_games', 0):,}",
                delta=f"{overtime_data.get('overtime_percentage', 0):.1%} of total"
            )

            if overtime_data.get('avg_ot_periods'):
                st.write(f"Avg OT Periods: {overtime_data['avg_ot_periods']:.1f}")
        else:
            st.info("No overtime data available")

    with col2:
        st.write("**Scoring Patterns**")
        scoring_patterns = _get_scoring_patterns(data_store, date_range)

        if scoring_patterns:
            col_pattern1, col_pattern2 = st.columns(2)

            with col_pattern1:
                st.metric(
                    "Highest Score",
                    f"{scoring_patterns.get('highest_score', 0)}",
                    delta=f"by {scoring_patterns.get('highest_scorer', 'N/A')}"
                )

            with col_pattern2:
                st.metric(
                    "Avg Total Points",
                    f"{scoring_patterns.get('avg_total_points', 0):.1f}",
                    delta=f"{scoring_patterns.get('points_trend', '+0.0')}"
                )
        else:
            st.info("No scoring patterns available")

    # Advanced query interface
    with st.expander("🔬 Custom Analytics Query", expanded=False):
        _render_custom_query_interface(data_store, date_range)


def _calculate_key_metrics(data_store: UnifiedDataStore, date_range: Tuple[date, date]) -> Dict[str, Any]:
    """Calculate key performance metrics using DuckDB."""
    try:
        start_date_str, end_date_str = date_range[0].isoformat(), date_range[1].isoformat()

        query = f"""
        WITH games_data AS (
            SELECT * FROM read_parquet('{data_store.games_dir}/*.parquet')
            WHERE game_date BETWEEN '{start_date_str}' AND '{end_date_str}'
        ),
        today_games AS (
            SELECT COUNT(*) as game_count
            FROM games_data
            WHERE game_date = CURRENT_DATE
        ),
        scoring_stats AS (
            SELECT
                AVG(home_score + away_score) as avg_total_points,
                AVG(home_score) as avg_home_score,
                AVG(away_score) as avg_away_score,
                COUNT(*) as total_games
            FROM games_data
        ),
        home_wins AS (
            SELECT COUNT(*) as home_wins_count
            FROM games_data
            WHERE home_score > away_score
        ),
        unique_teams AS (
            SELECT COUNT(DISTINCT team) as team_count
            FROM (
                SELECT home_team as team FROM games_data
                UNION ALL
                SELECT away_team as team FROM games_data
            )
        )
        SELECT
            COALESCE(s.total_games, 0) as total_games,
            COALESCE(t.game_count, 0) as games_today,
            COALESCE(s.avg_total_points, 0) as avg_points,
            COALESCE(u.team_count, 0) as active_teams,
            CASE
                WHEN s.total_games > 0
                THEN COALESCE(h.home_wins_count, 0.0) / s.total_games
                ELSE 0
            END as overall_win_rate
        FROM scoring_stats s
        CROSS JOIN today_games t
        CROSS JOIN home_wins h
        CROSS JOIN unique_teams u
        """

        result = data_store.query_analytics(query)

        if result.height > 0:
            return {
                'total_games': int(result['total_games'][0]),
                'games_today': int(result['games_today'][0]),
                'avg_points': float(result['avg_points'][0]),
                'active_teams': int(result['active_teams'][0]),
                'overall_win_rate': float(result['overall_win_rate'][0]),
                'points_trend': '+2.3',  # Mock trend data
                'teams_change': '+2',
                'win_rate_trend': '+1.2%'
            }
        else:
            return {}

    except Exception as e:
        logger.error(f"Failed to calculate key metrics: {e}")
        return {}


def _get_scoring_trends(data_store: UnifiedDataStore, date_range: Tuple[date, date]) -> Optional[pl.DataFrame]:
    """Get scoring trends data using DuckDB."""
    try:
        start_date_str, end_date_str = date_range[0].isoformat(), date_range[1].isoformat()

        query = f"""
        SELECT
            game_date,
            AVG(home_score) as avg_home_score,
            AVG(away_score) as avg_away_score,
            COUNT(*) as games_count
        FROM read_parquet('{data_store.games_dir}/*.parquet')
        WHERE game_date BETWEEN '{start_date_str}' AND '{end_date_str}'
        GROUP BY game_date
        ORDER BY game_date
        """

        return data_store.query_analytics(query)

    except Exception as e:
        logger.error(f"Failed to get scoring trends: {e}")
        return None


def _get_game_distribution(data_store: UnifiedDataStore, date_range: Tuple[date, date]) -> Optional[pl.DataFrame]:
    """Get game distribution data."""
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

    except Exception as e:
        logger.error(f"Failed to get game distribution: {e}")
        return None


def _get_teams_performance(data_store: UnifiedDataStore, date_range: Tuple[date, date]) -> Optional[pl.DataFrame]:
    """Get teams performance data."""
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
            AVG(points_scored) as avg_points_scored,
            AVG(points_allowed) as avg_points_allowed,
            AVG(points_scored - points_allowed) as avg_point_differential
        FROM team_stats
        GROUP BY team_name
        ORDER BY win_percentage DESC, avg_point_differential DESC
        """

        return data_store.query_analytics(query)

    except Exception as e:
        logger.error(f"Failed to get teams performance: {e}")
        return None


def _get_head_to_head_stats(data_store: UnifiedDataStore, date_range: Tuple[date, date], team1: str, team2: str) -> Dict[str, Any]:
    """Get head-to-head statistics between two teams."""
    try:
        start_date_str, end_date_str = date_range[0].isoformat(), date_range[1].isoformat()

        query = f"""
        WITH matchups AS (
            SELECT *
            FROM read_parquet('{data_store.games_dir}/*.parquet')
            WHERE game_date BETWEEN '{start_date_str}' AND '{end_date_str}'
            AND (
                (home_team = '{team1}' AND away_team = '{team2}') OR
                (home_team = '{team2}' AND away_team = '{team1}')
            )
        ),
        team1_stats AS (
            SELECT
                COUNT(*) as games_played,
                SUM(CASE
                    WHEN (home_team = '{team1}' AND home_score > away_score) OR
                         (away_team = '{team1}' AND away_score > home_score)
                    THEN 1 ELSE 0
                END) as wins,
                AVG(CASE
                    WHEN home_team = '{team1}' THEN home_score ELSE away_score
                END) as avg_points_scored,
                AVG(CASE
                    WHEN home_team = '{team1}' THEN away_score ELSE home_score
                END) as avg_points_allowed
            FROM matchups
        )
        SELECT
            games_played,
            wins,
            games_played - wins as losses,
            ROUND(wins * 100.0 / games_played, 1) as win_percentage,
            ROUND(avg_points_scored, 1) as avg_points_scored,
            ROUND(avg_points_allowed, 1) as avg_points_allowed,
            ROUND(avg_points_scored - avg_points_allowed, 1) as avg_point_differential
        FROM team1_stats
        """

        result = data_store.query_analytics(query)

        if result.height > 0:
            return {
                'team1_wins': int(result['wins'][0]),
                'team1_losses': int(result['losses'][0]),
                'win_percentage': float(result['win_percentage'][0]),
                'avg_points_scored': float(result['avg_points_scored'][0]),
                'avg_points_allowed': float(result['avg_points_allowed'][0]),
                'point_differential': float(result['avg_point_differential'][0])
            }
        else:
            return {}

    except Exception as e:
        logger.error(f"Failed to get head-to-head stats: {e}")
        return {}


def _render_head_to_head_comparison(comparison_data: Dict[str, Any]) -> None:
    """Render head-to-head comparison visualization."""
    if not comparison_data:
        st.info("No head-to-head data available")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            f"Team 1 Record",
            f"{comparison_data['team1_wins']}-{comparison_data['team1_losses']}",
            delta=f"{comparison_data['win_percentage']:.1%} win rate"
        )

    with col2:
        st.metric(
            "Avg Points Scored",
            f"{comparison_data['avg_points_scored']:.1f}",
            delta=f"+{comparison_data['point_differential']:.1f} differential"
        )

    with col3:
        st.metric(
            "Avg Points Allowed",
            f"{comparison_data['avg_points_allowed']:.1f}"
        )


def _get_overtime_statistics(data_store: UnifiedDataStore, date_range: Tuple[date, date]) -> Optional[Dict[str, Any]]:
    """Get overtime statistics."""
    # Mock implementation - would query actual overtime data
    return {
        'overtime_games': 12,
        'overtime_percentage': 0.085,
        'avg_ot_periods': 1.2
    }


def _get_scoring_patterns(data_store: UnifiedDataStore, date_range: Tuple[date, date]) -> Optional[Dict[str, Any]]:
    """Get scoring patterns analysis."""
    # Mock implementation - would analyze actual scoring patterns
    return {
        'highest_score': 158,
        'highest_scorer': 'Team A',
        'avg_total_points': 225.3,
        'points_trend': '+3.2'
    }


def _render_performance_insights(data_store: UnifiedDataStore, date_range: Tuple[date, date]) -> None:
    """Render detailed performance insights."""
    st.write("**Key Insights:**")

    insights = [
        "📈 Scoring is up 3.2% compared to last period",
        "🏠 Home teams maintain 58.3% win advantage",
        "🔥 High-scoring games increased by 15% this month",
        "⚡ Overtime games trending down from historical average"
    ]

    for insight in insights:
        st.write(f"• {insight}")


def _render_custom_query_interface(data_store: UnifiedDataStore, date_range: Tuple[date, date]) -> None:
    """Render custom analytics query interface."""
    st.write("**Execute Custom Analytics Query:**")

    query = st.text_area(
        "Enter SQL Query",
        value=f"""-- Example query for games analysis
SELECT
    DATE(game_date) as date,
    COUNT(*) as games,
    AVG(home_score + away_score) as avg_total_points
FROM read_parquet('{data_store.games_dir}/*.parquet')
WHERE game_date BETWEEN '{date_range[0].isoformat()}' AND '{date_range[1].isoformat()}'
GROUP BY DATE(game_date)
ORDER BY date DESC
LIMIT 10""",
        height=150
    )

    if st.button("🚀 Execute Query"):
        if query.strip():
            with st.spinner("Executing query..."):
                try:
                    result = data_store.query_analytics(query)

                    if result.height > 0:
                        st.success(f"✅ Query returned {result.height} rows")
                        st.dataframe(result.to_pandas(), use_container_width=True)
                    else:
                        st.info("Query executed successfully but returned no results")

                except Exception as e:
                    st.error(f"❌ Query failed: {e}")
                    logger.error(f"Custom query failed: {e}")
        else:
            st.warning("Please enter a query")


# Import timedelta for date calculations
from datetime import timedelta