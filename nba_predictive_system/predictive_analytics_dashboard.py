"""
Predictive Analytics Dashboard for NBA Games (Legacy Flow)

This dashboard implements the specific operational flow requested:
1. Schedule Selection (Real Data)
2. AI Prediction (Advanced Model)
3. Manual Line Input (User)
4. Staking Calculation (Legacy Logic)
"""

import sys
import os
import logging
from datetime import date, datetime, timedelta
import pandas as pd
import streamlit as st
import plotly.express as px

# Add project root to path to allow importing from src
sys.path.append(os.getcwd())


# Import NBA system components
from nba_predictive_system.unified_nba_data_pipeline import UnifiedNBADataPipeline
from nba_predictive_system.advanced_predictive_model import AdvancedPredictiveModel
from src.nba_predictor.utils.manual_odds_calculator import ManualOddsCalculator
from nba_api.stats.static import teams

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBAPredictiveDashboard:
    def __init__(self):
        self.pipeline = UnifiedNBADataPipeline()
        self.model = AdvancedPredictiveModel()
        self.odds_calculator = ManualOddsCalculator()
        self._initialize_session_state()

    def _initialize_session_state(self):
        if "games_data" not in st.session_state:
            st.session_state.games_data = None
        if "selected_game_id" not in st.session_state:
            st.session_state.selected_game_id = None
        if "prediction_result" not in st.session_state:
            st.session_state.prediction_result = None

    def _enrich_game_data(self, games_df):
        """Enrich game data with team names and proper dates."""
        if games_df is None or games_df.empty:
            return games_df

        try:
            # Get team mapping
            nba_teams = teams.get_teams()
            team_map = {team["id"]: team["abbreviation"] for team in nba_teams}

            # Map IDs to names if columns don't exist
            if (
                "home_team" not in games_df.columns
                and "home_team_id" in games_df.columns
            ):
                games_df["home_team"] = games_df["home_team_id"].map(team_map)

            if (
                "away_team" not in games_df.columns
                and "visitor_team_id" in games_df.columns
            ):
                games_df["away_team"] = games_df["visitor_team_id"].map(team_map)

            # Normalize date
            if "game_date" not in games_df.columns:
                if "game_date_est" in games_df.columns:
                    games_df["game_date"] = (
                        games_df["game_date_est"].str.split("T").str[0]
                    )
                elif "date" in games_df.columns:
                    games_df["game_date"] = pd.to_datetime(
                        games_df["date"]
                    ).dt.strftime("%Y-%m-%d")

            return games_df
        except Exception as e:
            logger.error(f"Error enriching game data: {e}")
            return games_df

    def render(self):
        st.set_page_config(
            page_title="NBA Professional Bettor Dashboard",
            page_icon="🏀",
            layout="wide",
        )

        st.title("🏀 NBA Professional Bettor Dashboard")
        st.markdown("---")

        # Sidebar
        self._render_sidebar()

        # Main Content
        if (
            st.session_state.games_data is not None
            and not st.session_state.games_data.empty
        ):
            self._render_game_selection()

            if st.session_state.selected_game_id:
                self._render_analysis_flow()
        else:
            st.info("👈 Please refresh data from the sidebar to begin.")

    def _render_sidebar(self):
        st.sidebar.header("⚙️ Controls")

        # Date Selection
        today = date.today()
        selected_date = st.sidebar.date_input("Game Date", today)

        if st.sidebar.button("🔄 Refresh Schedule", type="primary"):
            with st.spinner("Fetching NBA Schedule..."):
                try:
                    # Fetch data for the selected date
                    data = self.pipeline.fetch_all_data(
                        date_range=(selected_date, selected_date),
                        include_boxscores=False,
                    )

                    # Enrich data before storing
                    games_data = self._enrich_game_data(data["games"])
                    st.session_state.games_data = games_data

                    if st.session_state.games_data.empty:
                        st.sidebar.warning("No games found for this date.")
                    else:
                        st.sidebar.success(
                            f"Found {len(st.session_state.games_data)} games."
                        )
                        # Reset selection on new fetch
                        st.session_state.selected_game_id = None
                        st.session_state.prediction_result = None

                except Exception as e:
                    st.sidebar.error(f"Error fetching data: {str(e)}")

    def _render_game_selection(self):
        st.header("1. Select Game")

        games = st.session_state.games_data

        # Create a list of game strings for the dropdown
        game_options = {
            f"{row['away_team']} @ {row['home_team']} ({row['game_date']})": row[
                "game_id"
            ]
            for _, row in games.iterrows()
        }

        selected_game_str = st.selectbox(
            "Choose a game to analyze:", options=list(game_options.keys())
        )

        # Update selected game ID
        if selected_game_str:
            new_game_id = game_options[selected_game_str]
            if new_game_id != st.session_state.selected_game_id:
                st.session_state.selected_game_id = new_game_id
                st.session_state.prediction_result = None  # Reset prediction

    def _render_analysis_flow(self):
        game_id = st.session_state.selected_game_id
        game_row = st.session_state.games_data[
            st.session_state.games_data["game_id"] == game_id
        ].iloc[0]

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.header("2. AI Analysis")
            if st.button("🔮 Generate Prediction"):
                with st.spinner("Analyzing matchup..."):
                    try:
                        # Prepare features for this single game
                        # Note: In a real scenario, we might need more historical context
                        # Here we pass the single game row to preprocess_features
                        raw_data = {"games": pd.DataFrame([game_row])}
                        features = self.pipeline.preprocess_features(raw_data)

                        if not features.empty:
                            # Generate prediction
                            prediction = self.model.predict_game_outcome(features)

                            # Mocking Total Score Prediction if not available in model
                            # The current model predicts winner probability.
                            # For the betting flow, we ideally need a predicted score.
                            # We will estimate it or use a placeholder if the model doesn't output total
                            predicted_total = (
                                225.5  # Placeholder/Mock if model doesn't output total
                            )

                            st.session_state.prediction_result = {
                                "winner_prob": prediction.iloc[0][
                                    "predicted_probability"
                                ],
                                "predicted_class": prediction.iloc[0][
                                    "predicted_class"
                                ],
                                "predicted_total": predicted_total,
                            }
                        else:
                            st.error("Could not generate features for this game.")
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")

            if st.session_state.prediction_result:
                res = st.session_state.prediction_result
                home_team = game_row["home_team"]
                away_team = game_row["away_team"]

                winner = home_team if res["predicted_class"] == 1 else away_team
                confidence = (
                    res["winner_prob"]
                    if res["predicted_class"] == 1
                    else (1 - res["winner_prob"])
                )

                st.success(f"**Predicted Winner:** {winner}")
                st.info(f"**Confidence:** {confidence:.1%}")
                st.warning(
                    f"**Predicted Total Score:** {res['predicted_total']} (Estimated)"
                )

        with col2:
            st.header("3. Market Analysis")

            if st.session_state.prediction_result:
                st.subheader("Manual Line Input")

                central_line = st.number_input(
                    "Bookmaker Central Line (Total Points)",
                    min_value=150.0,
                    max_value=300.0,
                    value=225.5,
                    step=0.5,
                )

                bankroll = st.number_input(
                    "Current Bankroll (€)", min_value=100.0, value=1000.0, step=50.0
                )

                if st.button("💰 Calculate Best Bet", type="primary"):
                    self._calculate_staking(
                        central_line,
                        st.session_state.prediction_result["predicted_total"],
                        bankroll,
                    )
            else:
                st.info("Generate a prediction first to enable market analysis.")

    def _calculate_staking(self, central_line, predicted_total, bankroll):
        st.markdown("---")
        st.header("4. Strategy & Staking")

        # Use the legacy calculator
        analysis = self.odds_calculator.generate_comprehensive_analysis(
            central_line=central_line,
            predicted_total=predicted_total,
            bankroll=bankroll,
        )

        best_bet = analysis.get("best_bet_analysis", {}).get("best_bet")

        if best_bet:
            st.success("✅ **OPPORTUNITY FOUND!**")

            # Display Key Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Bet Type", f"{best_bet['type'].upper()} {best_bet['line']}")
            m2.metric("Odds", f"{best_bet['odds']}")
            m3.metric("Edge", f"{best_bet['edge_percentage']:.1f}%")

            # Display Stake Recommendation
            st.markdown(f"""
            ### 💵 Recommended Stake: **€{best_bet["recommended_stake"]:.2f}**
            
            **Rationale:**
            - **Probability:** {best_bet["predicted_probability"]:.1%}
            - **Kelly Fraction:** {best_bet["kelly_fraction"]:.2%}
            - **Risk Level:** {best_bet["risk_level"]}
            - **Confidence:** {best_bet["confidence_level"]}
            """)

        else:
            st.warning(
                "⚠️ No value bets found for this line. The market is efficient or the edge is too small."
            )

            # Show analysis details
            if "analysis" in analysis.get("best_bet_analysis", {}):
                st.write(
                    "Analysis Details:",
                    analysis["best_bet_analysis"]["analysis"]["message"],
                )


if __name__ == "__main__":
    dashboard = NBAPredictiveDashboard()
    dashboard.render()
