"""
🏀 NBA Predictor Pro - Streamlit Interface
Complete replication of main.py functionality with organized tabs.
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
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import ALL main components - EXACT REPLICATION of main.py
try:
    from main import NBACompleteSystem
    from data_provider import NBADataProvider
    from injury_reporter import InjuryReporter
    from player_impact_analyzer import PlayerImpactAnalyzer
    from momentum_calculator_real import RealMomentumCalculator
    from probabilistic_model import ProbabilisticModel
    # Try to import momentum systems with same fallback logic as main.py
    try:
        from momentum_predictor_selector import MomentumPredictorSelector
        MOMENTUM_SELECTOR_AVAILABLE = True
    except ImportError:
        try:
            from advanced_player_momentum_predictor import AdvancedPlayerMomentumPredictor
            MOMENTUM_SELECTOR_AVAILABLE = False
            ADVANCED_MOMENTUM_AVAILABLE = True
        except ImportError:
            try:
                from player_momentum_predictor import PlayerMomentumPredictor
                MOMENTUM_SELECTOR_AVAILABLE = False
                ADVANCED_MOMENTUM_AVAILABLE = False
            except ImportError:
                MOMENTUM_SELECTOR_AVAILABLE = False
                ADVANCED_MOMENTUM_AVAILABLE = None
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

# Enhanced CSS
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
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #1e3c72;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 0.8rem 0;
    }
    .injury-card {
        background: #fff5f5;
        border: 1px solid #feb2b2;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .momentum-card {
        background: #f0fff4;
        border: 1px solid #9ae6b4;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .betting-opportunity {
        background: #f7fafc;
        border: 2px solid #4299e1;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# 🗂️ SESSION STATE MANAGEMENT
# ================================

if 'system' not in st.session_state:
    st.session_state['system'] = None
if 'analysis_result' not in st.session_state:
    st.session_state['analysis_result'] = None
if 'selected_game' not in st.session_state:
    st.session_state['selected_game'] = None
if 'bankroll' not in st.session_state:
    st.session_state['bankroll'] = 89.48

# ================================
# 🔧 CORE FUNCTIONS (EXACT REPLICA)
# ================================

@st.cache_resource
def initialize_complete_system():
    """Initialize COMPLETE NBA system - EXACT replica of main.py"""
    with st.spinner("🚀 Initializing NBA Complete System..."):
        try:
            data_provider = NBADataProvider()
            system = NBACompleteSystem(data_provider, auto_mode=True)
            return system
        except Exception as e:
            st.error(f"❌ System initialization failed: {e}")
            return None

@st.cache_data
def load_all_data():
    """Load all data sources like main.py"""
    data = {}
    
    # Bankroll
    bankroll_paths = ['data/bankroll.json', 'bankroll.json']
    for path in bankroll_paths:
        try:
            with open(path, 'r') as f:
                bankroll_data = json.load(f)
                data['bankroll'] = float(bankroll_data.get('current_bankroll', 89.48))
                break
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    else:
        data['bankroll'] = 89.48
    
    # Bet history
    try:
        data['bet_history'] = pd.read_csv('data/risultati_bet_completi.csv')
    except FileNotFoundError:
        data['bet_history'] = pd.DataFrame()
    
    # Pending bets
    try:
        with open('data/pending_bets.json', 'r') as f:
            data['pending_bets'] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data['pending_bets'] = []
    
    return data

def format_currency(amount):
    """Format currency with Euro symbol"""
    return f"€{amount:.2f}"

# ================================
# 📱 MAIN APPLICATION
# ================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏀 NBA Predictor Pro</h1>
        <p>Complete Machine Learning System - Exact Main.py Functionality</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    if st.session_state['system'] is None:
        st.session_state['system'] = initialize_complete_system()
        if st.session_state['system']:
            st.success("✅ NBA Complete System Initialized Successfully!")
    
    system = st.session_state['system']
    if not system:
        st.error("❌ Cannot proceed without system initialization")
        return
    
    # Load all data
    all_data = load_all_data()
    st.session_state['bankroll'] = all_data['bankroll']
    
    # Sidebar - System Status (EXACT replica)
    with st.sidebar:
        st.markdown("### 🎯 NBA Predictor Status")
        st.metric("💰 Current Bankroll", format_currency(all_data['bankroll']))
        
        # System components status
        st.markdown("#### 🔧 System Components")
        st.markdown("🟢 **Data Provider**: Active")
        st.markdown("🟢 **Injury Reporter**: Active") 
        st.markdown("🟢 **Impact Analyzer**: VORP v7.0")
        
        # Momentum system status
        if hasattr(system, 'use_real_momentum') and system.use_real_momentum:
            st.markdown("🟢 **Momentum System**: Real NBA Data")
        elif hasattr(system, 'use_momentum_selector') and system.use_momentum_selector:
            st.markdown("🟢 **Momentum System**: ML Selector")
        elif hasattr(system, 'use_advanced_momentum') and system.use_advanced_momentum:
            st.markdown("🟢 **Momentum System**: Advanced ML")
        else:
            st.markdown("🟢 **Momentum System**: Base")
        
        st.markdown("🟢 **Probabilistic Model**: Active")
        st.markdown("🟢 **Betting Engine**: Active")
        
        # Quick actions
        st.markdown("#### ⚡ Quick Actions")
        if st.button("🔄 Check Pending Bets", use_container_width=True):
            with st.spinner("Checking pending bets..."):
                system.check_and_update_pending_bets()
                st.rerun()
        
        if st.button("💾 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # MAIN TABS - Organized workflow
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏀 Game Analysis", 
        "📊 Results Display", 
        "💰 Betting Center", 
        "📈 Performance", 
        "⚙️ Management"
    ])
    
    with tab1:
        show_game_analysis_tab(system)
    
    with tab2:
        show_results_display_tab()
    
    with tab3:
        show_betting_center_tab(system, all_data)
    
    with tab4:
        show_performance_tab(all_data)
    
    with tab5:
        show_management_tab(system, all_data)

# ================================
# 🏀 TAB 1: GAME ANALYSIS (EXACT MAIN.PY FLOW)
# ================================

def show_game_analysis_tab(system):
    st.markdown("### 🏀 NBA Game Analysis")
    st.markdown("Complete replication of main.py analyze_game functionality")
    
    # Step 1: Game Selection (EXACT replica)
    st.markdown("#### 📅 Step 1: Game Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Option A: Scheduled Games**")
        try:
            scheduled_games = system.data_provider.get_scheduled_games(days_ahead=3)
            if scheduled_games:
                game_options = [f"{g['away_team']} @ {g['home_team']} ({g.get('date', 'TBD')})" for g in scheduled_games]
                selected_idx = st.selectbox("Choose scheduled game:", range(len(game_options)), 
                                          format_func=lambda x: game_options[x])
                selected_game = scheduled_games[selected_idx]
            else:
                st.warning("No scheduled games found")
                selected_game = None
        except Exception as e:
            st.error(f"Error loading scheduled games: {e}")
            selected_game = None
    
    with col2:
        st.markdown("**Option B: Custom Analysis**")
        team1 = st.text_input("Away Team", placeholder="e.g., Thunder")
        team2 = st.text_input("Home Team", placeholder="e.g., Pacers")
        
        if team1 and team2:
            # Get team IDs like main.py
            try:
                away_info = system.data_provider._find_team_by_name(team1)
                home_info = system.data_provider._find_team_by_name(team2)
                
                if away_info and home_info:
                    selected_game = {
                        'away_team': team1,
                        'home_team': team2,
                        'away_team_id': away_info['id'],
                        'home_team_id': home_info['id'],
                        'game_id': f"CUSTOM_{team1}_{team2}",
                        'odds': []
                    }
                else:
                    st.error(f"Team not found: {team1 if not away_info else team2}")
                    selected_game = None
            except Exception as e:
                st.error(f"Error finding teams: {e}")
                selected_game = None
        elif not scheduled_games:
            # Fallback like main.py
            selected_game = {
                'away_team': 'Lakers', 'home_team': 'Warriors',
                'away_team_id': 1610612747, 'home_team_id': 1610612744,
                'game_id': 'EXAMPLE_GAME', 'odds': []
            }
    
    # Step 2: Analysis Parameters (EXACT replica)
    st.markdown("#### ⚙️ Step 2: Analysis Parameters")
    central_line = st.number_input("📊 Central Line (Total Points)", 
                                 min_value=150.0, max_value=300.0, 
                                 value=225.0, step=0.5,
                                 help="This will be used for betting opportunities generation")
    
    auto_mode = st.checkbox("🤖 Auto Mode (No user interaction)", value=True)
    
    # Step 3: Run Analysis (EXACT main.py flow)
    st.markdown("#### 🚀 Step 3: Execute Analysis")
    
    if st.button("🎯 RUN COMPLETE NBA ANALYSIS", key="run_analysis_main", type="primary", use_container_width=True):
        if not selected_game:
            st.error("Please select a game first")
            return
        
        # Create args object EXACTLY like main.py
        class StreamlitArgs:
            def __init__(self):
                self.auto_mode = auto_mode
                self.line = central_line
                self.team1 = selected_game.get('away_team')
                self.team2 = selected_game.get('home_team')
        
        args = StreamlitArgs()
        
        # Progress tracking EXACTLY like main.py console output
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_container = st.container()
            
            # STEP 1: Team Statistics
            with log_container:
                st.info("🚀 Starting NBA Complete Analysis...")
                status_text.text("1. Acquiring team statistics...")
                progress_bar.progress(15)
            
            # STEP 2: Roster Data
            status_text.text("2. Acquiring player rosters...")
            progress_bar.progress(25)
            
            # STEP 3: Injury Analysis  
            status_text.text("3. Analyzing injury impact...")
            progress_bar.progress(40)
            
            # STEP 4: Momentum Calculation
            status_text.text("4. Calculating momentum...")
            progress_bar.progress(60)
            
            # STEP 5: Probabilistic Model
            status_text.text("5. Running probabilistic model...")
            progress_bar.progress(80)
            
            # STEP 6: Betting Analysis
            status_text.text("6. Analyzing betting opportunities...")
            progress_bar.progress(95)
            
            try:
                # EXACT CALL TO MAIN.PY SYSTEM
                with st.expander("🔍 System Analysis Log", expanded=True):
                    log_placeholder = st.empty()
                    
                    # Capture the analysis (this mirrors main.py output)
                    analysis_result = system.analyze_game(selected_game, central_line=central_line, args=args)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis completed successfully!")
                    
                    # Store results in session state
                    st.session_state['analysis_result'] = analysis_result
                    st.session_state['selected_game'] = selected_game
                    st.session_state['analysis_args'] = args
                    
                    st.success("🎉 Analysis completed! Check the Results Display tab for detailed output.")
                    
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                st.exception(e)

# ================================
# 📊 TAB 2: RESULTS DISPLAY (EXACT MAIN.PY OUTPUT)
# ================================

def show_results_display_tab():
    st.markdown("### 📊 Analysis Results Display")
    st.markdown("Exact replication of main.py console output in web format")
    
    # Safety checks per variabili session_state
    if 'analysis_result' not in st.session_state or st.session_state['analysis_result'] is None:
        st.info("👆 Run an analysis in the Game Analysis tab first")
        return
    
    result = st.session_state['analysis_result']
    game = st.session_state.get('selected_game', {})  # Aggiunta protezione
    
    if 'error' in result:
        st.error(f"❌ Analysis error: {result['error']}")
        return
    
    # Extract all results EXACTLY like main.py
    team_stats = result.get('team_stats', {})
    injury_impact = result.get('injury_impact', 0)
    momentum_impact = result.get('momentum_impact', {})
    distribution = result.get('distribution', {})
    opportunities = result.get('opportunities', [])
    
    # EXACT REPLICA OF MAIN.PY FINAL SUMMARY
    st.markdown("## 🎯 FINAL SUMMARY")
    st.markdown("*Exact replication of main.py display_final_summary output*")
    
    # Header information
    home_team = game.get('home_team', 'Home')
    away_team = game.get('away_team', 'Away')
    predicted_total = distribution.get('predicted_mu', 0)
    confidence_sigma = distribution.get('predicted_sigma', 0)
    
    # Score predictions (EXACT formula from main.py)
    momentum_total_impact = momentum_impact.get('total_impact', 0) if isinstance(momentum_impact, dict) else momentum_impact
    home_predicted = predicted_total / 2 + (momentum_total_impact / 2) + (injury_impact / 2)
    away_predicted = predicted_total / 2 - (momentum_total_impact / 2) - (injury_impact / 2)
    confidence_percentage = max(0, min(100, 100 - (confidence_sigma - 10) * 3))
    
    # Game Information Box - LAYOUT ADATTIVO PER NOMI LUNGHI
    with st.container():
        st.markdown("#### 🏀 Game Information")
        
        # Funzione per abbreviare nomi squadre lunghi
        def abbreviate_team_name(team_name):
            """Abbrevia nomi squadre troppo lunghi mantenendo leggibilità"""
            if len(team_name) <= 12:
                return team_name
            # Logica di abbreviazione intelligente
            team_mappings = {
                "Oklahoma City Thunder": "OKC Thunder",
                "Indiana Pacers": "Pacers", 
                "Los Angeles Lakers": "Lakers",
                "Los Angeles Clippers": "Clippers",
                "Golden State Warriors": "Warriors",
                "San Antonio Spurs": "Spurs",
                "Portland Trail Blazers": "Trail Blazers",
                "Minnesota Timberwolves": "Timberwolves",
                "Philadelphia 76ers": "76ers",
                "Charlotte Hornets": "Hornets"
            }
            return team_mappings.get(team_name, team_name[:12] + "...")
        
        # Abbrevia nomi se necessario
        away_short = abbreviate_team_name(away_team)
        home_short = abbreviate_team_name(home_team)
        
        # Layout a 2 righe per evitare sovrapposizioni
        # Riga 1: Predicted Score con layout migliorato e font più grandi
        st.markdown("**🎯 Predicted Score**")
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.markdown(f"""
            <div style='text-align: center;'>
                <div style='font-size: 1.3em; font-weight: bold; color: #1f77b4;'>{away_short}</div>
                <div style='font-size: 2.2em; font-weight: bold; color: #ff7f0e;'>{away_predicted:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style='text-align: center; padding-top: 20px;'>
                <div style='font-size: 1.8em; font-weight: bold; color: #2ca02c;'>VS</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div style='text-align: center;'>
                <div style='font-size: 1.3em; font-weight: bold; color: #1f77b4;'>{home_short}</div>
                <div style='font-size: 2.2em; font-weight: bold; color: #ff7f0e;'>{home_predicted:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Riga 2: Altre metriche su 2 colonne più larghe
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("📊 Total Points", f"{predicted_total:.1f}")
        with col2:
            st.metric("📈 Confidence", f"{confidence_percentage:.1f}%", 
                     delta=f"σ: {confidence_sigma:.1f}")
    
    # Impact Analysis (EXACT replica)
    st.markdown("#### ⚡ Impact Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🏥 Injury Impact", f"{injury_impact:+.2f} pts")
    with col2:
        st.metric("⚡ Momentum Impact", f"{momentum_total_impact:+.2f} pts")
    with col3:
        st.metric("💰 Current Bankroll", format_currency(st.session_state.get('bankroll', 89.48)))
    
    # EXACT INJURY DETAILS TABLE (replica of main.py)
    show_injury_details_table(result, game)
    
    # EXACT SYSTEM STATUS (replica of main.py)
    show_system_status_display(result)
    
    # EXACT BETTING ANALYSIS (replica of main.py)
    show_betting_analysis_display(result, opportunities)

def show_injury_details_table(result, game):
    """EXACT replica of main.py injury details table"""
    st.markdown("#### 🏥 INJURY DETAILS")
    
    # Get injury details from system (EXACT extraction logic) - con protezioni
    system = st.session_state.get('system')
    if not system:
        st.warning("System not available for injury details")
        return
        
    home_impact_result = getattr(system, '_last_home_impact_result', {'injured_players_details': []})
    away_impact_result = getattr(system, '_last_away_impact_result', {'injured_players_details': []})
    
    # Parse injury details EXACTLY like main.py
    home_injuries = []
    away_injuries = []
    
    for detail in home_impact_result.get('injured_players_details', []):
        try:
            if ' - Impatto: ' in detail:
                player_part, impact_part = detail.split(' - Impatto: ')
                player_name = player_part.split(' (')[0]
                status = player_part.split(' (')[1].split(')')[0].upper()
                impact_clean = impact_part.replace(' pts', '').split('[')[0].strip()
                impact = abs(float(impact_clean))
                home_injuries.append({"player": player_name, "status": status, "impact": impact})
        except:
            continue
    
    for detail in away_impact_result.get('injured_players_details', []):
        try:
            if ' - Impatto: ' in detail:
                player_part, impact_part = detail.split(' - Impatto: ')
                player_name = player_part.split(' (')[0]
                status = player_part.split(' (')[1].split(')')[0].upper()
                impact_clean = impact_part.replace(' pts', '').split('[')[0].strip()
                impact = abs(float(impact_clean))
                away_injuries.append({"player": player_name, "status": status, "impact": impact})
        except:
            continue
    
    # Display EXACTLY like main.py table
    if not home_injuries and not away_injuries:
        st.info("✅ No significant injuries detected for either team")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**🏠 {game.get('home_team', 'Home')} Team**")
            if home_injuries:
                for inj in home_injuries:
                    st.markdown(f"🚨 **{inj['player']}** ({inj['status']}) - Impact: +{inj['impact']:.2f} pts")
            else:
                st.markdown("✅ No injuries with impact")
        
        with col2:
            st.markdown(f"**🛫 {game.get('away_team', 'Away')} Team**")
            if away_injuries:
                for inj in away_injuries:
                    st.markdown(f"🚨 **{inj['player']}** ({inj['status']}) - Impact: +{inj['impact']:.2f} pts")
            else:
                st.markdown("✅ No injuries with impact")

def show_system_status_display(result):
    """EXACT replica of main.py system status summary"""
    st.markdown("#### 🔧 SYSTEM STATUS SUMMARY")
    
    momentum_impact = result.get('momentum_impact', {})
    distribution = result.get('distribution', {})
    opportunities = result.get('opportunities', [])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Momentum System Status
        if momentum_impact.get('real_data_system'):
            confidence = momentum_impact.get('confidence_factor', 0) * 100
            st.success(f"⚡ **Momentum System**: 🎯 NBA Real Data ({confidence:.0f}%)")
        elif momentum_impact.get('selector_system'):
            confidence = momentum_impact.get('confidence_factor', 0) * 100
            model = momentum_impact.get('model_used', 'unknown')
            st.success(f"⚡ **Momentum System**: 🚀 ML Selector ({model.upper()}, {confidence:.0f}%)")
        else:
            st.success("⚡ **Momentum System**: ✅ Active")
    
    with col2:
        # ML Prediction Status
        if 'error' not in distribution:
            st.success("🤖 **ML Predictions**: ✅ Active")
        else:
            st.error("🤖 **ML Predictions**: ❌ Error")
    
    with col3:
        # Betting Analysis Status
        value_bets = [opp for opp in opportunities if opp.get('edge', 0) > 0 and opp.get('probability', 0) >= 0.5]
        if value_bets:
            st.success(f"🎰 **Betting Engine**: 🎯 {len(value_bets)} VALUE bets")
        elif opportunities:
            st.warning("🎰 **Betting Engine**: ⚠️ Active - No VALUE")
        else:
            st.error("🎰 **Betting Engine**: ❌ No opportunities")

def _calculate_optimal_bet_app(opportunities):
    """
    EXACT COPY of main.py _calculate_optimal_bet method for app.py
    Calcola la scommessa ottimale usando un algoritmo razionale.
    Ora considera solo scommesse con probabilità >= 50%.
    """
    try:
        # SOGLIA EDGE PIÙ ALTA: Minimo 1% per essere considerato VALUE
        value_bets = [opp for opp in opportunities if opp.get('edge', 0) > 0.01 and opp.get('probability', 0) >= 0.50]
        if not value_bets:
            return None
        
        scored_bets = []
        
        for bet in value_bets:
            edge = bet.get('edge', 0)
            probability = bet.get('probability', 0)
            odds = bet.get('odds', 1.0)
            
            if edge <= 0 or probability <= 0 or odds <= 1.0:
                continue
            
            # CORREZIONE DRASTICA: Edge deve essere significativo per ottenere punti alti
            if edge >= 0.15:  # 15%+
                edge_score = 30
            elif edge >= 0.10:  # 10-15%
                edge_score = 25 + (edge - 0.10) * 100  # Scala 25-30
            elif edge >= 0.05:  # 5-10%
                edge_score = 15 + (edge - 0.05) * 200  # Scala 15-25
            elif edge >= 0.02:  # 2-5%
                edge_score = 5 + (edge - 0.02) * 333   # Scala 5-15
            else:  # <2%
                edge_score = edge * 250  # Max 5 punti per edge molto bassi
            
            # SISTEMA AMPLIFICATO - Ogni % conta nella fascia critica 50-55%
            if probability > 0.65:
                prob_score = 35              # Bonus per probabilità molto alte
            elif 0.60 <= probability <= 0.65:
                prob_score = 25 + (probability - 0.60) * 200  # Scala 25-35
            elif 0.55 <= probability < 0.60:
                prob_score = 15 + (probability - 0.55) * 200  # Scala 15-25
            elif 0.52 <= probability < 0.55:
                prob_score = 5 + (probability - 0.52) * 333   # Scala 5-15 (AMPLIFICATA)
            else:  # 50-52% - FASCIA CRITICA
                prob_score = (probability - 0.50) * 250       # 0-5 punti (MOLTO RIPIDA) 
            
            # SISTEMA QUOTE POTENZIATO - Range ottimale privilegiato (20% peso)
            if 1.70 <= odds <= 1.95:
                odds_score = 30              # POTENZIATO: Range ottimale massimo premio
            elif 1.60 <= odds < 1.70:
                odds_score = 18              # Buono ma margine basso
            elif 1.95 < odds <= 2.10:
                odds_score = 20              # Ancora accettabile
            elif 2.10 < odds <= 2.30:
                odds_score = 12              # Rischio moderato
            elif 2.30 < odds <= 2.60:
                odds_score = 8               # Rischio alto
            else:
                odds_score = max(3, 15 - abs(odds - 1.8) * 8)  # Penalizzazione severa
            
            # SISTEMA PULITO - Solo 3 componenti indipendenti
            total_score = (
                edge_score * 0.30 +      # Edge 
                prob_score * 0.50 +      # Probabilità dominante
                odds_score * 0.20        # Quote potenziate (+5% dal Kelly eliminato)
            )

            bet_copy = bet.copy()
            # Normalizzazione corretta: max possibile = 35+30+25+10 = 100
            normalized_score = total_score  # Già su scala 0-100
            
            bet_copy.update({
                'optimization_score': normalized_score,
                'edge_score': edge_score,
                'prob_score': prob_score,
                'odds_score': odds_score,
                'total_raw_score': total_score
            })
            scored_bets.append(bet_copy)

        if not scored_bets:
            return None

        best_bet = max(scored_bets, key=lambda x: x['optimization_score'])
        return best_bet
        
    except Exception as e:
        st.error(f"⚠️ Errore nel calcolo scommessa ottimale: {e}")
        return None

def show_betting_analysis_display(result, opportunities):
    """EXACT replica of main.py betting analysis display - GRAPHICALLY ENHANCED"""
    st.markdown("#### 💎 BETTING ANALYSIS")
    
    if not opportunities:
        st.warning("❌ No betting opportunities generated")
        return
    
    # Filter VALUE bets EXACTLY like main.py
    all_opportunities = sorted(opportunities, key=lambda x: x.get('edge', 0), reverse=True)
    value_bets = [opp for opp in all_opportunities if opp.get('edge', 0) > 0 and opp.get('probability', 0) >= 0.5]
    
    if value_bets:
        st.success(f"🎯 Trovate **{len(value_bets)}** opportunità VALUE su **{len(all_opportunities)}** linee analizzate")
        
        # CALCOLA RACCOMANDAZIONI ESATTE COME MAIN.PY usando la funzione copiata
        optimal_bet = _calculate_optimal_bet_app(all_opportunities)
        highest_prob_bet = max(value_bets, key=lambda x: x.get('probability', 0))
        highest_edge_bet = max(value_bets, key=lambda x: x.get('edge', 0))
        highest_odds_bet = max(value_bets, key=lambda x: x.get('odds', 0))
        
        # Lista delle raccomandazioni principali ESATTA COME MAIN.PY
        recommendations = []
        
        # 1. SCELTA DEL SISTEMA (Ottimale)
        if optimal_bet:
            recommendations.append({
                'bet': optimal_bet,
                'category': '🏆 SCELTA DEL SISTEMA',
                'color': 'gold'
            })
        
        # 2. PIÙ PROBABILE  
        recommendations.append({
            'bet': highest_prob_bet,
            'category': '📊 MASSIMA PROBABILITÀ',
            'color': 'lightgreen'
        })
        
        # 3. MASSIMO EDGE
        recommendations.append({
            'bet': highest_edge_bet,
            'category': '🔥 MASSIMO EDGE',
            'color': 'salmon'
        })
        
        # 4. QUOTA MAGGIORE
        recommendations.append({
            'bet': highest_odds_bet,
            'category': '💰 QUOTA MASSIMA',
            'color': 'plum'
        })
        
        # Rimuovi duplicati mantenendo l'ordine ESATTO MAIN.PY
        seen_bets = set()
        unique_recommendations = []
        for rec in recommendations:
            bet_key = f"{rec['bet']['type']}_{rec['bet']['line']}"
            if bet_key not in seen_bets:
                seen_bets.add(bet_key)
                unique_recommendations.append(rec)
        
        # DISPLAY RACCOMANDAZIONI PRINCIPALI - ENHANCED TABLE
        st.markdown("##### 🏆 Raccomandazioni Categorizzate")
        
        # Mostra le raccomandazioni principali in cards eleganti
        for i, rec in enumerate(unique_recommendations, 1):
            bet = rec['bet']
            edge = bet.get('edge', 0) * 100
            prob = bet.get('probability', 0) * 100
            quality = bet.get('quality_score', 0)
            
            # Card per ogni raccomandazione con icone e metriche
            col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([0.5, 3, 1.5, 1, 1, 1, 1, 1])
            
            with col1:
                st.markdown(f"**#{i}**")
            with col2:
                st.markdown(f"**{rec['category']}**")
            with col3:
                st.markdown(f"`{bet['type']} {bet['line']}`")
            with col4:
                st.markdown(f"**{bet['odds']:.2f}**")
            with col5:
                # Color coding per edge
                if edge >= 30:
                    st.markdown(f"🔥 **{edge:.1f}%**")
                elif edge >= 20:
                    st.markdown(f"⚡ **{edge:.1f}%**")
                else:
                    st.markdown(f"📊 **{edge:.1f}%**")
            with col6:
                # Color coding per probabilità
                if prob >= 70:
                    st.markdown(f"🟢 **{prob:.1f}%**")
                elif prob >= 60:
                    st.markdown(f"🟡 **{prob:.1f}%**")
                else:
                    st.markdown(f"🔴 **{prob:.1f}%**")
            with col7:
                st.markdown(f"**{quality:.1f}**")
            with col8:
                st.markdown(f"**€{bet['stake']:.2f}**")
            
            # Separator line
            if i < len(unique_recommendations):
                st.markdown("---")
        
        # ALTRE VALUE BETS - ESATTO MAIN.PY
        other_bets = []
        for bet in value_bets:
            bet_key = f"{bet['type']}_{bet['line']}"
            if bet_key not in seen_bets:
                other_bets.append(bet)
                seen_bets.add(bet_key)
        
        # Ordina le altre per STAKE decrescente ESATTO MAIN.PY
        other_bets = sorted(other_bets, key=lambda x: x.get('stake', 0), reverse=True)
        
        if other_bets:
            st.markdown("##### 💎 Altre Opportunità VALUE")
            
            # Header per la tabella
            col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([0.5, 3, 1.5, 1, 1, 1, 1, 1])
            with col1:
                st.markdown("**#**")
            with col2:
                st.markdown("**CATEGORIA**")
            with col3:
                st.markdown("**LINE**")
            with col4:
                st.markdown("**ODDS**")
            with col5:
                st.markdown("**EDGE**")
            with col6:
                st.markdown("**PROB**")
            with col7:
                st.markdown("**QUAL**")
            with col8:
                st.markdown("**STAKE**")
            
            st.markdown("---")
            
            # Display altre VALUE bets in formato compatto
            for i, bet in enumerate(other_bets, len(unique_recommendations) + 1):
                edge = bet.get('edge', 0) * 100
                prob = bet.get('probability', 0) * 100
                quality = bet.get('quality_score', 0)
                
                col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([0.5, 3, 1.5, 1, 1, 1, 1, 1])
                
                with col1:
                    st.markdown(f"{i}")
                with col2:
                    st.markdown("💎 **ALTRE VALUE BETS**")
                with col3:
                    st.markdown(f"`{bet['type']} {bet['line']}`")
                with col4:
                    st.markdown(f"{bet['odds']:.2f}")
                with col5:
                    st.markdown(f"{edge:.1f}%")
                with col6:
                    st.markdown(f"{prob:.1f}%")
                with col7:
                    st.markdown(f"{quality:.1f}")
                with col8:
                    st.markdown(f"€{bet['stake']:.2f}")
        
        # FOOTER INFORMATIVO COME MAIN.PY
        st.info("💡 **VALUE** = Edge > 0% AND Probabilità ≥ 50%")

    else:
        st.warning("❌ Nessuna opportunità VALUE trovata - prime 5 opzioni migliori")
        top_5 = all_opportunities[:5]
        
        bet_data = []
        for i, bet in enumerate(top_5):
            edge = bet.get('edge', 0) * 100
            status = "📊 MARGINALE" if edge > -2.0 else "📉 SCARSA" if edge > -5.0 else "❌ PESSIMA"
            
            bet_data.append({
                '#': i + 1,
                'Tipo': status,
                'Line': f"{bet.get('type', 'OVER')} {bet.get('line', 0)}",
                'Odds': f"{bet.get('odds', 0):.2f}",
                'Edge': f"{edge:.1f}%",
                'Prob': f"{bet.get('probability', 0)*100:.1f}%",
                'Qual': f"{bet.get('quality_score', 0):.0f}",
                'Stake': f"€{bet.get('stake', 0):.2f}"
            })
        
        df_top = pd.DataFrame(bet_data)
        st.dataframe(df_top, use_container_width=True, hide_index=True)

# ================================
# 💰 TAB 3: BETTING CENTER
# ================================

def show_betting_center_tab(system, all_data):
    st.markdown("### 💰 Betting Center")
    st.markdown("Place bets and manage betting operations")
    
    # Safety checks per variabili session_state
    if 'analysis_result' not in st.session_state or st.session_state['analysis_result'] is None:
        st.info("👆 Run an analysis first to see betting opportunities")
        return
    
    if 'selected_game' not in st.session_state:
        st.warning("No game selected - run analysis first")
        return
    
    result = st.session_state['analysis_result']
    opportunities = result.get('opportunities', [])
    
    if not opportunities:
        st.warning("No betting opportunities available from last analysis")
        return
    
    # VALUE bets section
    all_opportunities = sorted(opportunities, key=lambda x: x.get('edge', 0), reverse=True)
    value_bets = [opp for opp in all_opportunities if opp.get('edge', 0) > 0 and opp.get('probability', 0) >= 0.5]
    
    if value_bets:
        st.success(f"🎯 {len(value_bets)} VALUE betting opportunities available")
        
        # CALCOLA OPTIMAL BET COME MAIN.PY - REPLICA ESATTA usando la funzione copiata
        optimal_bet = _calculate_optimal_bet_app(all_opportunities)
        highest_prob_bet = max(value_bets, key=lambda x: x.get('probability', 0))
        highest_edge_bet = max(value_bets, key=lambda x: x.get('edge', 0))
        highest_odds_bet = max(value_bets, key=lambda x: x.get('odds', 0))
        
        # CREA RACCOMANDAZIONI CATEGORIZZATE COME MAIN.PY
        recommendations = []
        
        # 1. SCELTA DEL SISTEMA (Ottimale)
        if optimal_bet:
            recommendations.append({
                'bet': optimal_bet,
                'category': '🏆 SCELTA DEL SISTEMA',
                'description': 'Algoritmo di ottimizzazione completo'
            })
        
        # 2. PIÙ PROBABILE
        recommendations.append({
            'bet': highest_prob_bet,
            'category': '📊 MASSIMA PROBABILITÀ',
            'description': 'Massima probabilità di successo'
        })
        
        # 3. MASSIMO EDGE
        recommendations.append({
            'bet': highest_edge_bet,
            'category': '🔥 MASSIMO EDGE',
            'description': 'Massimo vantaggio matematico'
        })
        
        # 4. QUOTA MAGGIORE
        recommendations.append({
            'bet': highest_odds_bet,
            'category': '💰 QUOTA MASSIMA',
            'description': 'Massimo payout potenziale'
        })
        
        # Rimuovi duplicati mantenendo l'ordine
        seen_bets = set()
        unique_recommendations = []
        for rec in recommendations:
            bet_key = f"{rec['bet']['type']}_{rec['bet']['line']}"
            if bet_key not in seen_bets:
                seen_bets.add(bet_key)
                unique_recommendations.append(rec)
        
        # DISPLAY RACCOMANDAZIONI PRINCIPALI
        st.markdown("#### 🏆 Raccomandazioni Categorizzate")
        
        # Crea opzioni per il selectbox con categorie
        bet_options = []
        for i, rec in enumerate(unique_recommendations):
            bet = rec['bet']
            edge = bet.get('edge', 0) * 100
            prob = bet.get('probability', 0) * 100
            bet_options.append(f"{rec['category']}: {bet['type']} {bet['line']} @ {bet['odds']:.2f} (Edge: {edge:.1f}%, Prob: {prob:.1f}%)")
        
        # Aggiungi altre VALUE bets se esistono
        other_bets = []
        for bet in value_bets:
            bet_key = f"{bet['type']}_{bet['line']}"
            if bet_key not in seen_bets:
                other_bets.append({
                    'bet': bet,
                    'category': '💎 VALUE BET',
                    'description': 'Altra opportunità VALUE'
                })
                edge = bet.get('edge', 0) * 100
                prob = bet.get('probability', 0) * 100
                bet_options.append(f"💎 VALUE BET: {bet['type']} {bet['line']} @ {bet['odds']:.2f} (Edge: {edge:.1f}%, Prob: {prob:.1f}%)")
        
        # Combina tutte le opzioni
        all_betting_options = unique_recommendations + other_bets
        
        selected_bet_idx = st.selectbox("Select bet to place:", range(len(bet_options)), 
                                       format_func=lambda x: bet_options[x],
                                       key="select_bet_center")
        
        selected_recommendation = all_betting_options[selected_bet_idx]
        selected_bet = selected_recommendation['bet']
        
        # Mostra dettagli della categoria selezionata
        st.info(f"**{selected_recommendation['category']}**: {selected_recommendation['description']}")
        
        # Bet details
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Bet Type", f"{selected_bet['type']} {selected_bet['line']}")
        with col2:
            st.metric("💰 Odds", f"{selected_bet['odds']:.2f}")
        with col3:
            st.metric("📊 Probability", f"{selected_bet.get('probability', 0)*100:.1f}%")
        
        # Calculate potential returns
        stake = selected_bet.get('stake', 0)
        potential_win = stake * (selected_bet['odds'] - 1)
        
        st.info(f"💵 Recommended Stake: €{stake:.2f} | 🎉 Potential Win: €{potential_win:.2f}")
        
        # Place bet button
        if st.button("🚀 PLACE BET", key="place_bet_center", type="primary", use_container_width=True):
            try:
                game_id = st.session_state['selected_game'].get('game_id', 'UNKNOWN_GAME')
                system.save_pending_bet(selected_bet, game_id)
                st.success(f"✅ Bet placed successfully: {selected_bet['type']} {selected_bet['line']} @ {selected_bet['odds']:.2f}")
                st.balloons()
                
                # Refresh data
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Failed to place bet: {e}")
    
    else:
        st.warning("❌ No VALUE bets available from current analysis")
    
    # Pending bets section
    st.markdown("#### 📋 Pending Bets")
    pending_bets = all_data.get('pending_bets', [])
    
    if pending_bets:
        pending_data = []
        for bet in pending_bets:
            if bet.get('status') == 'pending':
                bet_data = bet.get('bet_data', {})
                pending_data.append({
                    'Game ID': bet.get('game_id', 'N/A'),
                    'Bet': f"{bet_data.get('type', 'N/A')} {bet_data.get('line', 'N/A')}",
                    'Odds': f"{bet_data.get('odds', 0):.2f}",
                    'Stake': f"€{bet_data.get('stake', 0):.2f}",
                    'Timestamp': bet.get('timestamp', 'N/A')[:19].replace('T', ' ')
                })
        
        if pending_data:
            df_pending = pd.DataFrame(pending_data)
            st.dataframe(df_pending, use_container_width=True)
        else:
            st.info("No pending bets")
    else:
        st.info("No pending bets")

# ================================
# 📈 TAB 4: PERFORMANCE
# ================================

def show_performance_tab(all_data):
    st.markdown("### 📈 Performance Dashboard")
    st.markdown("Complete betting performance analysis")
    
    bet_history = all_data.get('bet_history', pd.DataFrame())
    
    if bet_history.empty:
        st.info("No betting history available yet")
        return
    
    # Performance metrics
    if 'Esito' in bet_history.columns:
        total_bets = len(bet_history)
        won_bets = len(bet_history[bet_history['Esito'] == 'Win'])
        lost_bets = len(bet_history[bet_history['Esito'] == 'Loss'])
        win_rate = (won_bets / total_bets * 100) if total_bets > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Bets", total_bets)
        with col2:
            st.metric("🟢 Won Bets", won_bets)
        with col3:
            st.metric("🔴 Lost Bets", lost_bets)
        with col4:
            st.metric("📈 Win Rate", f"{win_rate:.1f}%")
    
    # Profit/Loss analysis
    if 'Profit_Loss' in bet_history.columns:
        total_profit = bet_history['Profit_Loss'].sum()
        avg_profit = bet_history['Profit_Loss'].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💰 Total P&L", format_currency(total_profit), 
                     delta=format_currency(avg_profit) + " avg")
        with col2:
            roi = (total_profit / bet_history['Stake'].sum() * 100) if 'Stake' in bet_history.columns else 0
            st.metric("📊 ROI", f"{roi:.1f}%")
    
    # Betting history table
    st.markdown("#### 📋 Betting History")
    st.dataframe(bet_history, use_container_width=True)
    
    # Performance charts
    if not bet_history.empty and 'Data' in bet_history.columns:
        try:
            bet_history['Data'] = pd.to_datetime(bet_history['Data'])
            
            if 'Profit_Loss' in bet_history.columns:
                bet_history['Cumulative_PL'] = bet_history['Profit_Loss'].cumsum()
                
                fig = px.line(bet_history, x='Data', y='Cumulative_PL', 
                             title='Cumulative Profit/Loss Over Time')
                st.plotly_chart(fig, use_container_width=True)
        except:
            pass

# ================================
# ⚙️ TAB 5: MANAGEMENT
# ================================

def show_management_tab(system, all_data):
    st.markdown("### ⚙️ System Management")
    st.markdown("Bankroll management and system operations")
    
    # Bankroll Management
    st.markdown("#### 💰 Bankroll Management")
    
    current_bankroll = all_data['bankroll']
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("💰 Current Bankroll", format_currency(current_bankroll))
        
        new_bankroll = st.number_input("Adjust Bankroll", 
                                      value=current_bankroll, 
                                      min_value=0.0, 
                                      step=1.0)
        
        if st.button("💾 Update Bankroll", key="update_bankroll_mgmt"):
            try:
                system._save_bankroll(new_bankroll)
                st.session_state['bankroll'] = new_bankroll
                st.success(f"✅ Bankroll updated to {format_currency(new_bankroll)}")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to update bankroll: {e}")
    
    with col2:
        st.markdown("#### 🔄 System Operations")
        
        if st.button("🔄 Check Pending Bets", key="check_pending_bets_mgmt", use_container_width=True):
            with st.spinner("Checking pending bets..."):
                try:
                    system.check_and_update_pending_bets()
                    st.success("✅ Pending bets checked successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error checking bets: {e}")
        
        if st.button("🗑️ Clear Cache", key="clear_cache_mgmt", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ Cache cleared!")
        
        if st.button("🔄 Refresh System", key="refresh_system_mgmt", use_container_width=True):
            st.session_state['system'] = None
            st.rerun()
    
    # System Information
    st.markdown("#### ℹ️ System Information")
    
    info_data = {
        "Component": [
            "Data Provider",
            "Injury Reporter", 
            "Impact Analyzer",
            "Momentum System",
            "Probabilistic Model",
            "Betting Engine"
        ],
        "Status": [
            "✅ Active",
            "✅ Active",
            "✅ VORP v7.0",
            "✅ Real NBA Data" if hasattr(system, 'use_real_momentum') and system.use_real_momentum else "✅ Active",
            "✅ Active",
            "✅ Active"
        ],
        "Version": [
            "NBA API",
            "Dual Source",
            "v7.0",
            "Real NBA" if hasattr(system, 'use_real_momentum') and system.use_real_momentum else "ML",
            "Monte Carlo",
            "Advanced"
        ]
    }
    
    df_info = pd.DataFrame(info_data)
    st.dataframe(df_info, use_container_width=True)

# ================================
# 🚀 APPLICATION ENTRY POINT
# ================================

if __name__ == "__main__":
    main()
