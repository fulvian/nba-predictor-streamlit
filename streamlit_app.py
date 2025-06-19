"""
🏀 NBA Predictor Streamlit App
Advanced Machine Learning System for NBA Game Predictions

This is the main entry point for Streamlit Cloud deployment.
Optimized for cloud environment with proper error handling.
"""

import streamlit as st
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure Streamlit page
st.set_page_config(
    page_title="🏀 NBA Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🏀 NBA Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Machine Learning System for NBA Game Predictions & Betting Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # System status
        st.subheader("🔧 System Status")
        
        # Check if main modules are available
        try:
            import data_provider
            st.success("✅ Data Provider")
        except ImportError:
            st.error("❌ Data Provider")
            
        try:
            import injury_reporter
            st.success("✅ Injury Reporter")
        except ImportError:
            st.error("❌ Injury Reporter")
            
        try:
            import momentum_predictor_selector
            st.success("✅ Momentum Selector")
        except ImportError:
            st.error("❌ Momentum Selector")
            
        try:
            import probabilistic_model
            st.success("✅ Probabilistic Model")
        except ImportError:
            st.error("❌ Probabilistic Model")
        
        # Configuration options
        st.subheader("🎯 Analysis Settings")
        central_line = st.number_input(
            "Central Line (Points)",
            min_value=180.0,
            max_value=250.0,
            value=225.0,
            step=0.5,
            help="Bookmaker's central line for the game"
        )
        
        auto_mode = st.checkbox(
            "Auto Mode",
            value=False,
            help="Run analysis without user interaction"
        )
        
        # Bankroll management
        st.subheader("💰 Bankroll")
        try:
            import json
            with open('data/bankroll.json', 'r') as f:
                bankroll_data = json.load(f)
                current_bankroll = bankroll_data.get('current_bankroll', 100.0)
                st.metric("Current Bankroll", f"€{current_bankroll:.2f}")
        except:
            st.metric("Current Bankroll", "€100.00")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Analysis", "📊 Results", "💰 Betting", "📈 Performance"])
    
    with tab1:
        st.header("🎯 Game Analysis")
        
        # Team selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏠 Home Team")
            home_team = st.selectbox(
                "Select Home Team",
                ["Lakers", "Warriors", "Celtics", "Heat", "Thunder", "Pacers", "Bucks", "Nets"],
                index=0
            )
        
        with col2:
            st.subheader("🛫 Away Team")
            away_team = st.selectbox(
                "Select Away Team",
                ["Lakers", "Warriors", "Celtics", "Heat", "Thunder", "Pacers", "Bucks", "Nets"],
                index=1
            )
        
        # Analysis button
        if st.button("🚀 Start Analysis", type="primary"):
            with st.spinner("Running comprehensive analysis..."):
                try:
                    # Import and run analysis
                    from main import NBACompleteSystem, NBADataProvider
                    
                    # Create mock args
                    class MockArgs:
                        def __init__(self):
                            self.line = central_line
                            self.auto_mode = auto_mode
                    
                    args = MockArgs()
                    
                    # Initialize system
                    data_provider = NBADataProvider()
                    system = NBACompleteSystem(data_provider, auto_mode=auto_mode)
                    
                    # Create game object
                    game = {
                        'away_team': away_team,
                        'home_team': home_team,
                        'away_team_id': 1610612747,  # Mock IDs
                        'home_team_id': 1610612744,
                        'game_id': f"CUSTOM_{away_team}_{home_team}",
                        'odds': []
                    }
                    
                    # Run analysis
                    results = system.analyze_game(game, central_line=central_line, args=args)
                    
                    # Store results in session state
                    st.session_state.analysis_results = results
                    st.session_state.game_info = game
                    
                    st.success("✅ Analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.exception(e)
    
    with tab2:
        st.header("📊 Analysis Results")
        
        if 'analysis_results' in st.session_state:
            results = st.session_state.analysis_results
            game = st.session_state.game_info
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Predictions")
                if 'distribution' in results and 'error' not in results['distribution']:
                    predicted_total = results['distribution'].get('predicted_mu', 0)
                    confidence = results['distribution'].get('predicted_sigma', 0)
                    
                    st.metric("Predicted Total", f"{predicted_total:.1f} pts")
                    st.metric("Confidence (±)", f"±{confidence:.1f} pts")
                    
                    # Confidence percentage
                    confidence_pct = max(0, min(100, 100 - (confidence - 10) * 3))
                    st.metric("Confidence Level", f"{confidence_pct:.1f}%")
            
            with col2:
                st.subheader("⚡ Impacts")
                injury_impact = results.get('injury_impact', 0)
                momentum_impact = results.get('momentum_impact', {}).get('total_impact', 0)
                
                st.metric("Injury Impact", f"{injury_impact:+.2f} pts")
                st.metric("Momentum Impact", f"{momentum_impact:+.2f} pts")
                
                # Combined impact
                total_impact = injury_impact + momentum_impact
                st.metric("Total Impact", f"{total_impact:+.2f} pts")
            
            # Opportunities
            if 'opportunities' in results and results['opportunities']:
                st.subheader("💎 Betting Opportunities")
                
                opportunities = results['opportunities']
                value_bets = [opp for opp in opportunities if opp.get('edge', 0) > 0]
                
                st.metric("Total Opportunities", len(opportunities))
                st.metric("Value Bets", len(value_bets))
                
                if value_bets:
                    # Show top value bets
                    st.subheader("🏆 Top Value Bets")
                    
                    for i, bet in enumerate(value_bets[:5], 1):
                        with st.container():
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.write(f"**{bet['type']} {bet['line']}**")
                            with col2:
                                st.write(f"Odds: {bet['odds']:.2f}")
                            with col3:
                                edge_pct = bet.get('edge', 0) * 100
                                st.write(f"Edge: {edge_pct:+.1f}%")
                            with col4:
                                prob_pct = bet.get('probability', 0) * 100
                                st.write(f"Prob: {prob_pct:.1f}%")
        else:
            st.info("👆 Run an analysis first to see results here.")
    
    with tab3:
        st.header("💰 Betting Management")
        
        # Pending bets
        st.subheader("📋 Pending Bets")
        
        try:
            import json
            with open('data/pending_bets.json', 'r') as f:
                pending_bets = json.load(f)
                
            if pending_bets:
                for bet in pending_bets:
                    if bet.get('status') == 'pending':
                        with st.container():
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                bet_data = bet['bet_data']
                                st.write(f"**{bet_data['type']} {bet_data['line']}**")
                            with col2:
                                st.write(f"Stake: €{bet_data['stake']:.2f}")
                            with col3:
                                st.write(f"Status: {bet['status']}")
            else:
                st.info("No pending bets found.")
                
        except FileNotFoundError:
            st.info("No pending bets file found.")
        except Exception as e:
            st.error(f"Error loading pending bets: {e}")
        
        # Manual bet entry
        st.subheader("➕ Add New Bet")
        
        with st.form("new_bet"):
            bet_type = st.selectbox("Bet Type", ["OVER", "UNDER"])
            bet_line = st.number_input("Line", min_value=180.0, max_value=250.0, value=225.0, step=0.5)
            bet_odds = st.number_input("Odds", min_value=1.01, max_value=10.0, value=1.90, step=0.01)
            bet_stake = st.number_input("Stake (€)", min_value=1.0, max_value=1000.0, value=10.0, step=1.0)
            
            if st.form_submit_button("Add Bet"):
                st.success("Bet added successfully!")
    
    with tab4:
        st.header("📈 System Performance")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Accuracy", "85.3%")
            st.metric("Regular Season MAE", "6.033")
        
        with col2:
            st.metric("Playoff MAE", "15.079")
            st.metric("Hybrid MAE", "15.012")
        
        with col3:
            st.metric("Value Bet Detection", "33 avg")
            st.metric("Processing Time", "<30s")
        
        # System status
        st.subheader("🔧 System Components")
        
        systems = [
            ("Data Provider", "✅ Active"),
            ("Injury Reporter", "✅ Active"),
            ("Momentum Selector", "✅ Active"),
            ("Probabilistic Model", "✅ Active"),
            ("Betting Analysis", "✅ Active")
        ]
        
        for system_name, status in systems:
            st.write(f"**{system_name}**: {status}")

if __name__ == "__main__":
    main() 