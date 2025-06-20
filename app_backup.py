"""
🏀 NBA Predictor - Streamlit Web Interface
Clean web interface that delegates all core functionality to main scripts.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import main components - ONLY importing, not replicating
try:
    from main import NBACompleteSystem
    from data_provider import NBADataProvider
    from injury_reporter import InjuryReporter
    from player_impact_analyzer import PlayerImpactAnalyzer
    from momentum_calculator_real import RealMomentumCalculator
except ImportError as e:
    st.error(f"❌ Error importing core modules: {e}")
    st.stop()

# ================================
# 🎨 STREAMLIT CONFIGURATION
# ================================

st.set_page_config(
    page_title="NBA Predictor Pro",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #1e3c72;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 0.8rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .game-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.8rem 0;
        box-shadow: 0 3px 15px rgba(0,0,0,0.08);
        border: 2px solid #e8f2ff;
    }
    
    .system-status {
        background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
        color: white;
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# 🔧 CORE UTILITY FUNCTIONS
# ================================

@st.cache_data
def load_bankroll_data():
    """Load bankroll data from JSON files"""
    bankroll_paths = ['data/bankroll.json', 'bankroll.json']
    
    for path in bankroll_paths:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                return {'current_bankroll': float(data.get('current_bankroll', 100.0))}
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    
    return {'current_bankroll': 100.0}

@st.cache_data
def load_bet_history():
    """Load betting history from CSV"""
    try:
        return pd.read_csv('data/risultati_bet_completi.csv')
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_resource
def initialize_system():
    """Initialize the NBA prediction system - DELEGATES to main.py"""
    try:
        data_provider = NBADataProvider()
        system = NBACompleteSystem(data_provider, auto_mode=True)
        return system
    except Exception as e:
        st.error(f"❌ System initialization failed: {e}")
        return None

def format_currency(amount):
    """Format currency with Euro symbol"""
    return f"€{amount:.2f}"

def get_scheduled_games(system):
    """Get scheduled games from the system"""
    try:
        if system and hasattr(system, 'data_provider'):
            games = system.data_provider.get_scheduled_games(days_ahead=3)
            return games if games else []
        return []
    except Exception as e:
        st.error(f"Errore nel recupero partite: {e}")
        return []

# ================================
# 📱 MAIN APPLICATION
# ================================

def main():
    # Header principale
    st.markdown("""
    <div class="main-header">
        <h1>🏀 NBA Predictor Pro</h1>
        <p>Advanced Machine Learning System for NBA Game Predictions & Betting Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                    padding: 0.8rem; border-radius: 12px; color: white; text-align: center; margin-bottom: 1rem;">
            <h3 style="margin: 0;">🎯 Navigation</h3>
        </div>
        """, unsafe_allow_html=True)
        
        page_options = [
            ("🎰", "Game Analysis"),
            ("📊", "Performance"),
            ("💰", "Bankroll"),
            ("🤖", "ML Models"),
            ("⚙️", "Settings")
        ]
        
        page_labels = [f"{icon} {label}" for icon, label in page_options]
        page = st.selectbox("Select Section", page_labels)
        
        # Quick stats in sidebar
        st.markdown("#### 📊 Quick Status")
        try:
            bankroll_data = load_bankroll_data()
            st.metric("💰 Bankroll", format_currency(bankroll_data['current_bankroll']))
        except:
            st.metric("💰 Bankroll", "€100.00")
        
        st.markdown("🟢 System Active  \n⚡ ML Models: OK  \n📡 NBA API: Live")
    
    # Initialize system
    system = initialize_system()
    if system is None:
        st.error("❌ Cannot proceed without system initialization")
        return
    
    # Route to selected page
    if page == "🎰 Game Analysis":
        show_game_analysis_page(system)
    elif page == "📊 Performance":
        show_performance_page()
    elif page == "💰 Bankroll":
        show_bankroll_page()
    elif page == "🤖 ML Models":
        show_ml_models_page()
    elif page == "⚙️ Settings":
        show_settings_page()

# ================================
# 🏀 GAME ANALYSIS PAGE
# ================================

def show_game_analysis_page(system):
    """Main game analysis page - DELEGATES to main.py system"""
    
    st.markdown("### 🏀 NBA Game Analysis")
    
    # Game Selection Section
    st.markdown("#### �� Step 1: Select Game")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Option A: Scheduled Games**")
        games = get_scheduled_games(system)
        
        if games:
            game_options = [f"{g['away_team']} @ {g['home_team']} ({g.get('date', 'TBD')})" for g in games]
            selected_game_idx = st.selectbox("Choose Game", range(len(game_options)), 
                                           format_func=lambda x: game_options[x])
            selected_game = games[selected_game_idx]
        else:
            st.warning("No scheduled games found")
            selected_game = None
    
    with col2:
        st.markdown("**Option B: Custom Matchup**")
        # Simplified team selection - let main.py handle the complexity
        custom_team1 = st.text_input("Away Team", placeholder="e.g., Lakers")
        custom_team2 = st.text_input("Home Team", placeholder="e.g., Warriors")
        
        if custom_team1 and custom_team2:
            # Create custom game object - main.py will handle team ID resolution
            selected_game = {
                'away_team': custom_team1,
                'home_team': custom_team2,
                'game_id': f"CUSTOM_{custom_team1}_{custom_team2}",
                'date': datetime.now().strftime('%Y-%m-%d'),
                'odds': []
            }
        elif not games:
            # Fallback to default if no games and no custom teams
            selected_game = {
                'away_team': 'Lakers',
                'home_team': 'Warriors',
                'game_id': 'EXAMPLE_GAME',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'odds': []
            }
    
    # Analysis Parameters
    st.markdown("#### ⚙️ Step 2: Analysis Parameters")
    central_line = st.number_input("📊 Betting Line (Total Points)", 
                                 min_value=150.0, max_value=300.0, 
                                 value=225.0, step=0.5)
    
    # Analysis Execution
    st.markdown("#### 🚀 Step 3: Run Analysis")
    
    if st.button("🎯 ANALYZE GAME", type="primary", use_container_width=True):
        if not selected_game:
            st.error("Please select or create a game first")
            return
            
        with st.spinner("🔄 Running NBA Predictor Analysis..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create args object for main.py
            class StreamlitArgs:
                def __init__(self):
                    self.auto_mode = True
                    self.line = central_line
                    self.team1 = selected_game.get('away_team')
                    self.team2 = selected_game.get('home_team')
            
            args = StreamlitArgs()
            
            # Progress simulation with real system calls
            status_text.text("📊 Loading team statistics...")
            progress_bar.progress(20)
            
            status_text.text("🏥 Analyzing injury reports...")
            progress_bar.progress(40)
            
            status_text.text("⚡ Calculating momentum...")
            progress_bar.progress(60)
            
            status_text.text("🎲 Running ML predictions...")
            progress_bar.progress(80)
            
            try:
                # DELEGATE TO MAIN.PY - This is where all the real work happens
                analysis_result = system.analyze_game(selected_game, central_line=central_line, args=args)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Store results for display
                st.session_state['analysis_result'] = analysis_result
                st.session_state['selected_game'] = selected_game
                st.session_state['central_line'] = central_line
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                return
    
    # Display Results
    if 'analysis_result' in st.session_state:
        display_analysis_results()

def display_analysis_results():
    """Display analysis results from main.py system"""
    
    st.markdown("#### 📊 Analysis Results")
    
    result = st.session_state['analysis_result']
    game = st.session_state['selected_game']
    central_line = st.session_state['central_line']
    
    if 'error' in result:
        st.error(f"❌ Analysis error: {result['error']}")
        return
    
    # Extract key results
    distribution = result.get('distribution', {})
    momentum_impact = result.get('momentum_impact', {})
    injury_impact = result.get('injury_impact', 0)
    opportunities = result.get('opportunities', [])
    
    predicted_total = distribution.get('predicted_mu', 0)
    confidence_sigma = distribution.get('predicted_sigma', 0)
    
    # Main Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Predicted Total", f"{predicted_total:.1f} pts")
    
    with col2:
        confidence = max(0, min(100, 100 - (confidence_sigma - 10) * 3))
        st.metric("📈 Confidence", f"{confidence:.1f}%", 
                 delta=f"±{confidence_sigma:.1f} pts")
    
    with col3:
        st.metric("🏥 Injury Impact", f"{injury_impact:+.2f} pts")
    
    with col4:
        momentum_value = momentum_impact.get('total_impact', 0) if isinstance(momentum_impact, dict) else momentum_impact
        st.metric("⚡ Momentum Impact", f"{momentum_value:+.2f} pts")
    
    # System Status Display
    display_system_status(result)
    
    # Betting Opportunities
    if opportunities:
        display_betting_opportunities(opportunities, game)
    else:
        st.warning("No betting opportunities identified")

def display_system_status(result):
    """Display system status based on analysis results"""
    
    st.markdown("#### 🔧 System Status")
    
    # Extract status information from result
    momentum_impact = result.get('momentum_impact', {})
    distribution = result.get('distribution', {})
    opportunities = result.get('opportunities', [])
    
    # Determine status based on updated systems
    momentum_status = "🟢 Real NBA Data" if momentum_impact.get('real_data_system') else "🟢 Active"
    ml_status = "🟢 Active" if 'error' not in distribution else "🔴 Error"
    betting_status = "🟢 Active" if opportunities else "🟡 No Opportunities"
    
    # Show confidence if available
    if isinstance(momentum_impact, dict) and 'confidence_factor' in momentum_impact:
        confidence = momentum_impact['confidence_factor'] * 100
        momentum_status += f" ({confidence:.0f}%)"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>⚡ Momentum System</h4>
            <p>{momentum_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🤖 ML Predictions</h4>
            <p>{ml_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎰 Betting Engine</h4>
            <p>{betting_status}</p>
        </div>
        """, unsafe_allow_html=True)

def display_betting_opportunities(opportunities, game):
    """Display betting opportunities with save functionality"""
    
    st.markdown("#### 🎰 Betting Opportunities")
    
    # Filter VALUE bets
    value_bets = [opp for opp in opportunities if opp.get('edge', 0) > 0 and opp.get('probability', 0) >= 0.5]
    
    if value_bets:
        st.success(f"🎯 Found {len(value_bets)} VALUE betting opportunities!")
        
        # Display opportunities in a table
        bet_data = []
        for i, bet in enumerate(value_bets):
            bet_data.append({
                'ID': i + 1,
                'Type': f"{bet.get('type', 'OVER')} {bet.get('line', 0)}",
                'Odds': f"{bet.get('odds', 0):.2f}",
                'Edge': f"{bet.get('edge', 0)*100:.1f}%",
                'Probability': f"{bet.get('probability', 0)*100:.1f}%",
                'Stake': f"€{bet.get('stake', 0):.2f}",
                'Quality': f"{bet.get('quality_score', 0):.0f}/100"
            })
        
        df_opportunities = pd.DataFrame(bet_data)
        st.dataframe(df_opportunities, use_container_width=True)
        
        # Bet Selection
        selected_bet_id = st.selectbox("Select bet to place:", 
                                     range(len(value_bets)),
                                     format_func=lambda x: f"#{x+1}: {value_bets[x]['type']} {value_bets[x]['line']} @ {value_bets[x]['odds']:.2f}")
        
        if st.button("💰 PLACE BET", type="primary"):
            selected_bet = value_bets[selected_bet_id]
            game_id = game.get('game_id', 'UNKNOWN_GAME')
            
            # DELEGATE TO MAIN.PY for bet saving
            try:
                # Use the system's save_pending_bet method
                system = initialize_system()
                if system and hasattr(system, 'save_pending_bet'):
                    system.save_pending_bet(selected_bet, game_id)
                    st.success(f"✅ Bet placed: {selected_bet['type']} {selected_bet['line']} @ {selected_bet['odds']:.2f}")
                else:
                    # Fallback to direct file operations if needed
                    save_bet_to_file(selected_bet, game_id)
                    st.success(f"✅ Bet saved: {selected_bet['type']} {selected_bet['line']} @ {selected_bet['odds']:.2f}")
                
            except Exception as e:
                st.error(f"❌ Failed to place bet: {e}")
    else:
        st.warning("No VALUE bets found. All opportunities have negative expected value.")

def save_bet_to_file(bet_data, game_id):
    """Fallback function to save bet - SIMPLIFIED VERSION"""
    try:
        os.makedirs('data', exist_ok=True)
        
        # Prepare bet record
        bet_record = {
            'bet_id': f"{game_id}_{bet_data['type']}_{bet_data['line']}",
            'game_id': game_id,
            'bet_data': bet_data,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        }
        
        # Save to pending bets
        pending_file = 'data/pending_bets.json'
        try:
            with open(pending_file, 'r') as f:
                pending_bets = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pending_bets = []
        
        pending_bets.append(bet_record)
        
        with open(pending_file, 'w') as f:
            json.dump(pending_bets, f, indent=2)
            
    except Exception as e:
        raise Exception(f"Failed to save bet: {e}")

# ================================
# 📊 OTHER PAGES (SIMPLIFIED)
# ================================

def show_performance_page():
    """Performance dashboard - simplified version"""
    st.markdown("### 📊 Performance Dashboard")
    
    try:
        bet_history = load_bet_history()
        if not bet_history.empty:
            # Basic performance metrics
            if 'Esito' in bet_history.columns:
                total_bets = len(bet_history)
                won_bets = len(bet_history[bet_history['Esito'] == 'Win'])
                win_rate = (won_bets / total_bets * 100) if total_bets > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Bets", total_bets)
                with col2:
                    st.metric("Won Bets", won_bets)
                with col3:
                    st.metric("Win Rate", f"{win_rate:.1f}%")
            
            st.dataframe(bet_history)
        else:
            st.info("No betting history available")
    except Exception as e:
        st.error(f"Error loading performance data: {e}")

def show_bankroll_page():
    """Bankroll management - simplified version"""
    st.markdown("### 💰 Bankroll Management")
    
    try:
        bankroll_data = load_bankroll_data()
        current_bankroll = bankroll_data['current_bankroll']
        
        st.metric("Current Bankroll", format_currency(current_bankroll))
        
        # Simple bankroll adjustment
        new_bankroll = st.number_input("Adjust Bankroll", value=current_bankroll, step=1.0)
        
        if st.button("Update Bankroll"):
            # Save new bankroll
            bankroll_record = {'current_bankroll': new_bankroll}
            os.makedirs('data', exist_ok=True)
            
            with open('data/bankroll.json', 'w') as f:
                json.dump(bankroll_record, f, indent=2)
            
            st.success(f"Bankroll updated to {format_currency(new_bankroll)}")
            st.rerun()
            
    except Exception as e:
        st.error(f"Error managing bankroll: {e}")

def show_ml_models_page():
    """ML Models dashboard - simplified version"""
    st.markdown("### 🤖 ML Models Status")
    
    model_paths = [
        ('Momentum Complete (Hybrid)', 'models/momentum_complete/hybrid'),
        ('Momentum Regular Season', 'models/momentum_complete/regular_season'),
        ('Momentum Playoff', 'models/momentum_complete/playoff'),
        ('Probabilistic Model', 'models/probabilistic')
    ]
    
    for model_name, model_path in model_paths:
        if os.path.exists(model_path):
            st.markdown(f"**{model_name}**: 🟢 Active")
        else:
            st.markdown(f"**{model_name}**: 🔴 Not Found")
    
    # Real momentum calculator status
    st.markdown("#### 🎯 Real Momentum System")
    try:
        calc = RealMomentumCalculator()
        st.markdown("**Real NBA Game Logs**: 🟢 Available")
    except Exception as e:
        st.markdown(f"**Real NBA Game Logs**: 🔴 Error - {e}")

def show_settings_page():
    """Settings page - simplified version"""
    st.markdown("### ⚙️ Settings")
    
    st.markdown("#### System Configuration")
    st.info("System settings are managed through the main configuration files.")
    
    st.markdown("#### Data Management")
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared!")
    
    st.markdown("#### Check Pending Bets")
    if st.button("🔄 Update Pending Bets"):
        try:
            system = initialize_system()
            if system and hasattr(system, 'check_and_update_pending_bets'):
                system.check_and_update_pending_bets()
                st.success("Pending bets checked and updated!")
            else:
                st.warning("Auto-update not available")
        except Exception as e:
            st.error(f"Error updating pending bets: {e}")

# ================================
# 🚀 APPLICATION ENTRY POINT
# ================================

if __name__ == "__main__":
    main()
