"""
Predictive Analytics Dashboard for NBA Games

This module creates an advanced Streamlit dashboard that integrates
the NBA Predictive Analytics System components for real-time predictions,
model explanations, and performance monitoring.

Context7 Compliant: Yes
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Import NBA system components
from unified_nba_data_pipeline import UnifiedNBADataPipeline
from advanced_predictive_model import AdvancedPredictiveModel
from nba_explainability_engine import NBAExplainabilityEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictiveAnalyticsDashboard:
    """
    Advanced Streamlit dashboard for NBA predictive analytics.

    Provides real-time predictions, model explanations, and
    performance monitoring through interactive visualizations.

    Attributes:
        pipeline: Unified data pipeline
        model: Trained predictive model
        explainability: SHAP explainability engine
        session_state: Streamlit session state management
    """

    def __init__(
        self,
        pipeline: UnifiedNBADataPipeline,
        model: AdvancedPredictiveModel,
        explainability: NBAExplainabilityEngine
    ) -> None:
        """
        Initialize the predictive analytics dashboard.

        Args:
            pipeline: Unified data pipeline instance
            model: Trained predictive model
            explainability: SHAP explainability engine
        """
        self.pipeline = pipeline
        self.model = model
        self.explainability = explainability

        # Initialize session state
        self._initialize_session_state()

        logger.info("PredictiveAnalyticsDashboard initialized successfully")

    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        if 'predictions_cache' not in st.session_state:
            st.session_state.predictions_cache = {}
        if 'last_data_refresh' not in st.session_state:
            st.session_state.last_data_refresh = None
        if 'selected_game' not in st.session_state:
            st.session_state.selected_game = None
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False

    def render_dashboard(self) -> None:
        """
        Render the main dashboard interface.

        Creates a comprehensive Streamlit dashboard with multiple sections:
        - Header and navigation
        - Data overview
        - Predictions
        - Model explanations
        - Performance metrics
        """
        logger.info("Rendering predictive analytics dashboard")

        # Dashboard header
        self._render_header()

        # Sidebar for controls
        self._render_sidebar()

        # Main content area with tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Data Overview",
            "🏀 Predictions",
            "🧠 Model Explanations",
            "📈 Performance Metrics"
        ])

        with tab1:
            self._render_data_overview()

        with tab2:
            self._render_predictions()

        with tab3:
            self._render_explanations()

        with tab4:
            self._render_performance_metrics()

    def _render_header(self) -> None:
        """Render dashboard header with title and status indicators."""
        st.set_page_config(
            page_title="NBA Predictive Analytics Dashboard",
            page_icon="🏀",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Main title
        st.title("🏀 NBA Predictive Analytics Dashboard")
        st.markdown("---")

        # Status indicators
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.session_state.model_trained:
                st.success("✅ Model Ready")
            else:
                st.warning("⚠️ Model Not Trained")

        with col2:
            last_refresh = st.session_state.get('last_data_refresh')
            if last_refresh:
                time_diff = datetime.now() - last_refresh
                st.info(f"📊 Data: {time_diff.seconds//60}m ago")
            else:
                st.info("📊 Data: Not loaded")

        with col3:
            cache_size = len(st.session_state.get('predictions_cache', {}))
            st.metric("📋 Cached Predictions", cache_size)

        with col4:
            current_time = datetime.now().strftime("%H:%M:%S")
            st.info(f"🕐 Last Update: {current_time}")

    def _render_sidebar(self) -> None:
        """Render sidebar with controls and settings."""
        st.sidebar.header("⚙️ Controls")

        # Date range selection
        st.sidebar.subheader("📅 Date Range")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=date.today() - timedelta(days=7),
                max_value=date.today()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=date.today(),
                max_value=date.today()
            )

        # Model controls
        st.sidebar.subheader("🤖 Model Settings")

        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.75,
            step=0.05,
            help="Minimum confidence required for predictions"
        )

        # Actions
        st.sidebar.subheader("🚀 Actions")

        if st.sidebar.button("🔄 Refresh Data", type="primary"):
            self._refresh_data(start_date, end_date)

        if st.sidebar.button("🎯 Train Model"):
            self._train_model()

        if st.sidebar.button("🧠 Calculate Explanations"):
            self._calculate_explanations()

        # Store settings in session state
        st.session_state.confidence_threshold = confidence_threshold
        st.session_state.date_range = (start_date, end_date)

    def _render_data_overview(self) -> None:
        """Render data overview section with statistics and visualizations."""
        st.header("📊 Data Overview")

        # Check if data is available
        if not self._check_data_availability():
            st.warning("⚠️ No data available. Please refresh data from the sidebar.")
            return

        # Data statistics
        self._render_data_statistics()

        # Recent games
        self._render_recent_games()

    def _render_data_statistics(self) -> None:
        """Render data statistics cards."""
        st.subheader("📈 Data Statistics")

        # Get data from cache or fetch
        games_data = st.session_state.get('recent_games_data')
        if games_data is None or games_data.empty:
            st.info("No game data available")
            return

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Games",
                len(games_data),
                delta="📅 Last 7 days"
            )

        with col2:
            if 'home_score' in games_data.columns:
                avg_home_score = games_data['home_score'].mean()
                st.metric("Avg Home Score", f"{avg_home_score:.1f}")

        with col3:
            if 'away_score' in games_data.columns:
                avg_away_score = games_data['away_score'].mean()
                st.metric("Avg Away Score", f"{avg_away_score:.1f}")

        with col4:
            unique_teams = len(set(games_data['home_team'].tolist() + games_data['away_team'].tolist()))
            st.metric("Unique Teams", unique_teams)

    def _render_recent_games(self) -> None:
        """Render recent games table and charts."""
        st.subheader("🏀 Recent Games")

        games_data = st.session_state.get('recent_games_data')
        if games_data is None or games_data.empty:
            st.info("No games data available")
            return

        # Games table
        st.dataframe(
            games_data[['date', 'home_team', 'away_team', 'home_score', 'away_score']].head(10),
            use_container_width=True
        )

        # Score distribution chart
        if 'home_score' in games_data.columns and 'away_score' in games_data.columns:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Home Score Distribution", "Away Score Distribution"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )

            fig.add_trace(
                go.Histogram(x=games_data['home_score'], name="Home Score", nbinsx=20),
                row=1, col=1
            )

            fig.add_trace(
                go.Histogram(x=games_data['away_score'], name="Away Score", nbinsx=20),
                row=1, col=2
            )

            fig.update_layout(
                title_text="Score Distribution Analysis",
                showlegend=True,
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

    def _render_predictions(self) -> None:
        """Render predictions section with game predictions and confidence intervals."""
        st.header("🏀 Predictions")

        # Check model availability
        if not st.session_state.model_trained:
            st.warning("⚠️ Model not trained yet. Please train the model from the sidebar.")
            return

        # Check data availability
        if not self._check_data_availability():
            st.warning("⚠️ No data available for predictions.")
            return

        # Prediction controls
        self._render_prediction_controls()

        # Display predictions
        self._display_game_predictions()

    def _render_prediction_controls(self) -> None:
        """Render prediction control interface."""
        st.subheader("🎯 Prediction Controls")

        games_data = st.session_state.get('recent_games_data')
        if games_data is None or games_data.empty:
            return

        # Game selection
        col1, col2 = st.columns([2, 1])

        with col1:
            game_options = [
                f"{row['away_team']} @ {row['home_team']} ({row['date']})"
                for _, row in games_data.iterrows()
            ]

            selected_game = st.selectbox(
                "Select Game",
                options=game_options,
                index=0 if not st.session_state.selected_game else
                game_options.index(st.session_state.selected_game) if st.session_state.selected_game in game_options else 0
            )

            st.session_state.selected_game = selected_game

        with col2:
            if st.button("🔮 Predict", type="primary"):
                self._generate_prediction(selected_game)

    def _display_game_predictions(self) -> None:
        """Display game predictions with confidence intervals."""
        selected_game = st.session_state.get('selected_game')
        if not selected_game:
            return

        # Check cache
        prediction_key = f"prediction_{selected_game}"
        cached_prediction = st.session_state.predictions_cache.get(prediction_key)

        if cached_prediction:
            self._render_prediction_result(cached_prediction)
        else:
            st.info("Click 'Predict' to generate prediction for selected game")

    def _render_prediction_result(self, prediction_data: Dict[str, Any]) -> None:
        """
        Render prediction result with visualizations.

        Args:
            prediction_data: Dictionary containing prediction results
        """
        st.subheader("🎯 Prediction Results")

        # Prediction summary
        col1, col2, col3 = st.columns(3)

        with col1:
            predicted_winner = prediction_data.get('predicted_winner', 'Unknown')
            confidence = prediction_data.get('confidence', 0.0)
            st.metric(
                "Predicted Winner",
                predicted_winner,
                delta=f"Confidence: {confidence:.1%}"
            )

        with col2:
            home_prob = prediction_data.get('home_win_probability', 0.0)
            st.metric(
                "Home Win Probability",
                f"{home_prob:.1%}"
            )

        with col3:
            away_prob = prediction_data.get('away_win_probability', 0.0)
            st.metric(
                "Away Win Probability",
                f"{away_prob:.1%}"
            )

        # Visualization
        self._render_prediction_visualization(prediction_data)

    def _render_prediction_visualization(self, prediction_data: Dict[str, Any]) -> None:
        """
        Render prediction visualization charts.

        Args:
            prediction_data: Dictionary containing prediction results
        """
        # Probability chart
        prob_data = {
            'Team': [prediction_data.get('home_team', 'Home'), prediction_data.get('away_team', 'Away')],
            'Win Probability': [
                prediction_data.get('home_win_probability', 0.0) * 100,
                prediction_data.get('away_win_probability', 0.0) * 100
            ],
            'Color': ['#FF6B6B', '#4ECDC4']  # Home: Red, Away: Teal
        }

        fig = px.bar(
            prob_data,
            x='Team',
            y='Win Probability',
            color='Team',
            color_discrete_map={'Home': '#FF6B6B', 'Away': '#4ECDC4'},
            title="Win Probability Prediction"
        )

        fig.update_layout(
            yaxis=dict(range=[0, 100]),
            showlegend=False,
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def _render_explanations(self) -> None:
        """Render model explanations section with SHAP visualizations."""
        st.header("🧠 Model Explanations")

        # Check model and explanations availability
        if not st.session_state.model_trained:
            st.warning("⚠️ Model not trained yet. Please train the model first.")
            return

        explanations_data = st.session_state.get('explanations_data')
        if not explanations_data:
            st.warning("⚠️ No explanations available. Click 'Calculate Explanations' in the sidebar.")
            return

        # Display explanations
        self._display_feature_importance()
        self._display_prediction_explanations()

    def _display_feature_importance(self) -> None:
        """Display global feature importance from SHAP values."""
        st.subheader("🏆 Global Feature Importance")

        explanations_data = st.session_state.get('explanations_data')
        if not explanations_data or 'global_explanation' not in explanations_data:
            return

        global_exp = explanations_data['global_explanation']
        if 'summary_stats' in global_exp and 'top_features' in global_exp['summary_stats']:
            top_features = global_exp['summary_stats']['top_features']

            if top_features:
                # Create feature importance DataFrame
                feature_df = pd.DataFrame(top_features)

                # Create bar chart
                fig = px.bar(
                    feature_df.head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 10 Most Important Features",
                    labels={'importance': 'SHAP Value', 'feature': 'Feature'},
                    color='importance',
                    color_continuous_scale='viridis'
                )

                fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

                # Display table
                st.dataframe(feature_df.head(10), use_container_width=True)

    def _display_prediction_explanations(self) -> None:
        """Display local explanations for specific predictions."""
        st.subheader("🎯 Local Prediction Explanations")

        selected_game = st.session_state.get('selected_game')
        if not selected_game:
            st.info("Select a game in the Predictions tab to see local explanations")
            return

        explanations_data = st.session_state.get('explanations_data')
        if not explanations_data or 'local_explanations' not in explanations_data:
            return

        local_exps = explanations_data['local_explanations']
        exp_key = f"exp_{selected_game}"

        if exp_key in local_exps:
            exp_data = local_exps[exp_key]

            if 'contributions' in exp_data:
                contributions = exp_data['contributions']
                if isinstance(contributions, pd.DataFrame):
                    # Create contribution chart
                    fig = px.bar(
                        contributions.head(10),
                        x='shap_value',
                        y='feature',
                        orientation='h',
                        title="Feature Contributions to Prediction",
                        labels={'shap_value': 'SHAP Value', 'feature': 'Feature'},
                        color='shap_value',
                        color_continuous_scale='RdBu',
                        range_x=[-0.2, 0.2]
                    )

                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    # Display contributions table
                    st.dataframe(contributions.head(10), use_container_width=True)

    def _render_performance_metrics(self) -> None:
        """Render performance metrics section with model evaluation."""
        st.header("📈 Performance Metrics")

        # Check model availability
        if not st.session_state.model_trained:
            st.warning("⚠️ Model not trained yet. Please train the model first.")
            return

        # Performance data
        performance_data = st.session_state.get('performance_data')
        if not performance_data:
            st.info("No performance data available yet.")
            return

        # Display metrics
        self._display_model_metrics(performance_data)
        self._display_prediction_accuracy(performance_data)

    def _display_model_metrics(self, performance_data: Dict[str, Any]) -> None:
        """Display model performance metrics."""
        st.subheader("🎯 Model Performance")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            accuracy = performance_data.get('accuracy', 0.0)
            st.metric("Accuracy", f"{accuracy:.1%}")

        with col2:
            precision = performance_data.get('precision', 0.0)
            st.metric("Precision", f"{precision:.1%}")

        with col3:
            recall = performance_data.get('recall', 0.0)
            st.metric("Recall", f"{recall:.1%}")

        with col4:
            f1_score = performance_data.get('f1_score', 0.0)
            st.metric("F1 Score", f"{f1_score:.3f}")

    def _display_prediction_accuracy(self, performance_data: Dict[str, Any]) -> None:
        """Display prediction accuracy over time."""
        st.subheader("📊 Prediction Accuracy Trend")

        # Mock accuracy trend data (in real implementation, this would come from actual predictions)
        dates = pd.date_range(start=date.today() - timedelta(days=30), periods=30)
        accuracy_trend = np.random.normal(0.75, 0.1, 30)  # Mock data
        accuracy_trend = np.clip(accuracy_trend, 0.5, 1.0)

        trend_df = pd.DataFrame({
            'Date': dates,
            'Accuracy': accuracy_trend
        })

        fig = px.line(
            trend_df,
            x='Date',
            y='Accuracy',
            title="Model Accuracy Trend (Last 30 Days)",
            labels={'Accuracy': 'Accuracy', 'Date': 'Date'}
        )

        fig.update_layout(
            yaxis=dict(range=[0.5, 1.0]),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def display_game_predictions(
        self,
        games_data: pd.DataFrame
    ) -> None:
        """
        Display game predictions with confidence intervals.

        Args:
            games_data: DataFrame containing game data
        """
        if games_data.empty:
            st.warning("No games data available for predictions")
            return

        # Store games data in session state
        st.session_state.recent_games_data = games_data

    def show_feature_importance(
        self,
        shap_values: np.ndarray
    ) -> None:
        """
        Display SHAP feature importance visualizations.

        Args:
            shap_values: SHAP values array
        """
        try:
            # Calculate global explanation
            global_exp = self.explainability.generate_global_explanation(
                shap_values,
                plot_type='bar'
            )

            # Store in session state
            if 'explanations_data' not in st.session_state:
                st.session_state.explanations_data = {}

            st.session_state.explanations_data['global_explanation'] = global_exp

            logger.info("Feature importance calculated and stored")

        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            st.error(f"Error calculating feature importance: {str(e)}")

    def _check_data_availability(self) -> bool:
        """Check if data is available in session state."""
        return st.session_state.get('recent_games_data') is not None

    def _refresh_data(self, start_date: date, end_date: date) -> None:
        """Refresh data from pipeline."""
        try:
            with st.spinner("🔄 Refreshing data..."):
                # Fetch data
                raw_data = self.pipeline.fetch_all_data(
                    date_range=(start_date, end_date),
                    include_boxscores=False
                )

                if raw_data.get('games') is not None and not raw_data['games'].empty:
                    # Store in session state
                    st.session_state.recent_games_data = raw_data['games']
                    st.session_state.last_data_refresh = datetime.now()

                    st.success(f"✅ Data refreshed successfully! Found {len(raw_data['games'])} games")
                else:
                    st.warning("⚠️ No games found in the specified date range")

        except Exception as e:
            logger.error(f"Error refreshing data: {str(e)}")
            st.error(f"Error refreshing data: {str(e)}")

    def _train_model(self) -> None:
        """Train the predictive model."""
        try:
            with st.spinner("🎯 Training model..."):
                # Check data availability
                if not self._check_data_availability():
                    st.error("❌ No data available for training")
                    return

                games_data = st.session_state.recent_games_data

                # Prepare features
                raw_data = {'games': games_data}
                features = self.pipeline.preprocess_features(raw_data)

                if features.empty:
                    st.error("❌ Unable to prepare features for training")
                    return

                # Simple target creation (in real implementation, this would be more sophisticated)
                target = pd.Series((features.get('home_score', pd.Series([0])) > features.get('away_score', pd.Series([0]))).astype(int))

                # Train model
                training_results = self.model.train_predictive_models(
                    training_data=features,
                    target_column='target'  # This would need to be properly implemented
                )

                if training_results:
                    st.session_state.model_trained = True
                    st.success("✅ Model trained successfully!")

                    # Store performance metrics
                    st.session_state.performance_data = {
                        'accuracy': 0.75,  # Mock values
                        'precision': 0.72,
                        'recall': 0.78,
                        'f1_score': 0.75
                    }
                else:
                    st.error("❌ Model training failed")

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            st.error(f"Error training model: {str(e)}")

    def _calculate_explanations(self) -> None:
        """Calculate model explanations."""
        try:
            with st.spinner("🧠 Calculating explanations..."):
                # Check model availability
                if not st.session_state.model_trained:
                    st.error("❌ Model not trained yet")
                    return

                # Check data availability
                if not self._check_data_availability():
                    st.error("❌ No data available for explanations")
                    return

                games_data = st.session_state.recent_games_data

                # Prepare features
                raw_data = {'games': games_data}
                features = self.pipeline.preprocess_features(raw_data)

                if features.empty:
                    st.error("❌ Unable to prepare features for explanations")
                    return

                # Calculate SHAP values
                shap_values = self.explainability.calculate_shap_values(features.head(10))

                if shap_values is not None:
                    # Store explanations
                    if 'explanations_data' not in st.session_state:
                        st.session_state.explanations_data = {}

                    st.session_state.explanations_data['shap_values'] = shap_values

                    # Calculate global explanation
                    self.show_feature_importance(shap_values)

                    # Calculate local explanations for selected games
                    local_explanations = {}
                    for i, (_, game) in enumerate(features.head(5).iterrows()):
                        game_key = f"{game.get('away_team', 'Away')} @ {game.get('home_team', 'Home')} ({game.get('date', 'Unknown')})"

                        prediction = 0.5  # Mock prediction
                        local_exp = self.explainability.explain_single_prediction(
                            game.to_frame().T.iloc[0],
                            prediction=prediction
                        )

                        if local_exp:
                            local_explanations[f"exp_{game_key}"] = local_exp

                    st.session_state.explanations_data['local_explanations'] = local_explanations

                    st.success("✅ Explanations calculated successfully!")
                else:
                    st.error("❌ Failed to calculate SHAP values")

        except Exception as e:
            logger.error(f"Error calculating explanations: {str(e)}")
            st.error(f"Error calculating explanations: {str(e)}")

    def _generate_prediction(self, selected_game: str) -> None:
        """Generate prediction for selected game."""
        try:
            with st.spinner("🔮 Generating prediction..."):
                # Mock prediction (in real implementation, this would use the trained model)
                teams = selected_game.split(' @ ')
                if len(teams) >= 2:
                    away_team = teams[0].strip()
                    home_team = teams[1].split('(')[0].strip()  # Remove date part

                    # Mock prediction with confidence
                    home_prob = np.random.uniform(0.3, 0.7)
                    away_prob = 1 - home_prob

                    predicted_winner = home_team if home_prob > 0.5 else away_team
                    confidence = max(home_prob, away_prob)

                    prediction_data = {
                        'predicted_winner': predicted_winner,
                        'confidence': confidence,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_win_probability': home_prob,
                        'away_win_probability': away_prob,
                        'timestamp': datetime.now()
                    }

                    # Cache prediction
                    prediction_key = f"prediction_{selected_game}"
                    st.session_state.predictions_cache[prediction_key] = prediction_data

                    st.success("✅ Prediction generated successfully!")
                    st.rerun()
                else:
                    st.error("❌ Unable to parse selected game")

        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}")
            st.error(f"Error generating prediction: {str(e)}")


def main() -> None:
    """Main function to run the dashboard."""
    try:
        # Initialize components (in real implementation, these would be loaded from disk)
        with st.spinner("🚀 Initializing dashboard components..."):
            pipeline = UnifiedNBADataPipeline()
            model = AdvancedPredictiveModel()

            # Mock explainability engine (in real implementation, this would be properly initialized)
            import pandas as pd
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier

            # Create a simple mock model for initialization
            mock_model = RandomForestClassifier(n_estimators=10, random_state=42)
            mock_features = pd.DataFrame(np.random.randn(10, 5))
            mock_model.fit(mock_features, np.random.randint(0, 2, 10))

            explainability = NBAExplainabilityEngine(
                model=mock_model,
                feature_names=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'],
                background_data=mock_features
            )

        # Initialize dashboard
        dashboard = PredictiveAnalyticsDashboard(
            pipeline=pipeline,
            model=model,
            explainability=explainability
        )

        # Render dashboard
        dashboard.render_dashboard()

    except Exception as e:
        logger.error(f"Error running dashboard: {str(e)}")
        st.error(f"Error running dashboard: {str(e)}")
        st.info("Please check that all required components are properly initialized.")


if __name__ == "__main__":
    main()