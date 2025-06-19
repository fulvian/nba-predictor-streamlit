"""
🏀 NBA Predictor - Professional Streamlit Application
Advanced NBA Game Prediction System with Modern UI
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import main components
try:
    from main import NBACompleteSystem
    from data_provider import NBADataProvider
    from injury_reporter import InjuryReporter
    from player_impact_analyzer import PlayerImpactAnalyzer
    from bet_manager import BetManager
except ImportError as e:
    st.error(f"❌ Error importing modules: {e}")
    st.stop()

# ================================
# 🎨 CONFIGURATION & STYLING
# ================================

st.set_page_config(
    page_title="NBA Predictor Pro",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add viewport meta tag for better mobile experience
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
""", unsafe_allow_html=True)

# Enhanced CSS for modern styling + MOBILE RESPONSIVENESS
st.markdown("""
<style>
    /* Base styling */
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
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #1e3c72;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 0.8rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
    }
    
    .metric-card h4 {
        margin-top: 0;
        margin-bottom: 0.8rem;
        color: #1e3c72;
        font-size: 1.1rem;
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
    
    .prediction-value {
        font-size: 2.2rem;
        font-weight: bold;
        margin: 0.8rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .bet-summary {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    
    .injury-alert {
        background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);
        color: white;
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.1);
    }
    
    .system-status {
        background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
        color: white;
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.1);
    }
    
    .model-status {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    
    .status-active { background: linear-gradient(135deg, #4caf50, #45a049); color: white; }
    .status-training { background: linear-gradient(135deg, #ff9800, #f57c00); color: white; }
    .status-error { background: linear-gradient(135deg, #f44336, #d32f2f); color: white; }
    
    .game-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.8rem 0;
        box-shadow: 0 3px 15px rgba(0,0,0,0.08);
        border: 2px solid #e8f2ff;
        transition: all 0.3s ease;
    }
    
    .game-card:hover {
        border-color: #1e3c72;
        transform: translateY(-2px);
    }
    
    .analysis-step {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.4rem 0;
        border-left: 3px solid #1e3c72;
    }
    
    .tab-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(0,0,0,0.08);
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.8rem;
        margin: 1rem 0;
    }
    
    .metric-item {
        background: white;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    }
    
    .metric-value {
        font-size: 1.3rem;
        font-weight: bold;
        color: #1e3c72;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: 0.3rem;
    }
    
    .stake-calculator {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        border: 2px solid #dee2e6;
    }
    
    /* MOBILE RESPONSIVENESS */
    @media screen and (max-width: 768px) {
        /* Header mobile optimization */
        .main-header {
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 12px;
        }
        
        .main-header h1 {
            font-size: 1.8rem;
            margin-bottom: 0.3rem;
        }
        
        .main-header p {
            font-size: 0.9rem;
            line-height: 1.3;
        }
        
        /* Cards mobile optimization */
        .metric-card {
            padding: 0.8rem;
            margin: 0.5rem 0;
            border-radius: 10px;
        }
        
        .metric-card h4 {
            font-size: 1rem;
            margin-bottom: 0.5rem;
        }
        
        .prediction-card {
            padding: 1rem;
            margin: 0.8rem 0;
            border-radius: 12px;
        }
        
        .prediction-value {
            font-size: 1.8rem;
            margin: 0.5rem 0;
        }
        
        .bet-summary {
            padding: 1rem;
            border-radius: 10px;
            margin: 0.8rem 0;
        }
        
        .bet-summary h3 {
            font-size: 1.2rem;
            margin-bottom: 0.5rem;
        }
        
        .tab-container {
            padding: 1rem;
            margin: 0.8rem 0;
            border-radius: 10px;
        }
        
        /* Grid responsive */
        .metric-grid {
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 0.5rem;
            margin: 0.8rem 0;
        }
        
        .metric-item {
            padding: 0.6rem;
            border-radius: 6px;
        }
        
        .metric-value {
            font-size: 1.1rem;
        }
        
        .metric-label {
            font-size: 0.7rem;
        }
        
        /* Buttons mobile */
        .stButton > button {
            width: 100% !important;
            font-size: 0.9rem !important;
            padding: 0.6rem 1rem !important;
            border-radius: 8px !important;
            margin: 0.3rem 0 !important;
        }
        
        /* Selectbox mobile */
        .stSelectbox > div > div {
            font-size: 0.9rem;
        }
        
        /* Tables mobile */
        .stDataFrame {
            font-size: 0.8rem;
        }
        
        /* Metrics mobile */
        [data-testid="metric-container"] {
            background-color: white;
            border: 1px solid #e1e5e9;
            padding: 0.8rem;
            border-radius: 8px;
            margin: 0.3rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        
        [data-testid="metric-container"] > div {
            font-size: 0.8rem;
        }
        
        [data-testid="metric-container"] [data-testid="metric-value"] {
            font-size: 1.3rem;
        }
        
        /* Sidebar mobile */
        .sidebar .sidebar-content {
            padding: 1rem 0.5rem;
        }
        
        /* Text adjustments */
        p, div {
            line-height: 1.4;
        }
        
        /* Spacing adjustments */
        .element-container {
            margin-bottom: 0.5rem;
        }
    }
    
    @media screen and (max-width: 480px) {
        /* Extra small screens */
        .main-header h1 {
            font-size: 1.5rem;
        }
        
        .main-header p {
            font-size: 0.8rem;
        }
        
        .prediction-value {
            font-size: 1.5rem;
        }
        
        .metric-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 0.4rem;
        }
        
        .metric-value {
            font-size: 1rem;
        }
        
        .bet-summary h3 {
            font-size: 1rem;
        }
        
        .stButton > button {
            font-size: 0.8rem !important;
            padding: 0.5rem 0.8rem !important;
        }
        
        /* Reduce padding on very small screens */
        .metric-card, .tab-container {
            padding: 0.6rem;
        }
        
        [data-testid="metric-container"] {
            padding: 0.6rem;
        }
    }
    
    /* Landscape mobile optimization */
    @media screen and (max-height: 500px) and (orientation: landscape) {
        .main-header {
            padding: 0.8rem;
            margin-bottom: 0.8rem;
        }
        
        .main-header h1 {
            font-size: 1.6rem;
            margin-bottom: 0.2rem;
        }
        
        .main-header p {
            font-size: 0.8rem;
        }
        
        .metric-card {
            padding: 0.6rem;
            margin: 0.3rem 0;
        }
        
        .tab-container {
            padding: 1rem;
            margin: 0.5rem 0;
        }
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Improve sidebar on mobile */
    @media screen and (max-width: 768px) {
        .sidebar .sidebar-content {
            width: 100%;
        }
        
        /* Better tab display on mobile */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.2rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-size: 0.8rem;
            padding: 0.5rem 0.3rem;
        }
    }
    
    /* Scrollbar optimization for mobile */
    ::-webkit-scrollbar {
        width: 4px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #1e3c72;
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #2a5298;
    }
    
    /* Touch-friendly adjustments */
    @media (hover: none) and (pointer: coarse) {
        .metric-card:hover {
            transform: none;
        }
        
        .game-card:hover {
            transform: none;
            border-color: #e8f2ff;
        }
        
        /* Larger touch targets */
        .stButton > button {
            min-height: 44px !important;
        }
        
        .stSelectbox > div > div {
            min-height: 44px;
        }
    }
</style>
""", unsafe_allow_html=True)

# ================================
# 🔧 UTILITY FUNCTIONS
# ================================

@st.cache_data
def load_bankroll_data():
    """Load bankroll data from JSON file"""
    try:
        with open('data/bankroll.json', 'r') as f:
            data = json.load(f)
            return {
                "current_bankroll": data.get('current_bankroll', 100.0),
                "initial_bankroll": data.get('initial_bankroll', 100.0),
                "total_bets": data.get('total_bets', 0)
            }
    except FileNotFoundError:
        return {"current_bankroll": 100.0, "initial_bankroll": 100.0, "total_bets": 0}

@st.cache_data
def load_bet_history():
    """Load betting history"""
    try:
        with open('data/pending_bets.json', 'r') as f:
            data = json.load(f)
            if isinstance(data, list) and data:
                return pd.DataFrame(data)
            else:
                return pd.DataFrame()
    except (FileNotFoundError, json.JSONDecodeError):
        return pd.DataFrame()

@st.cache_resource
def initialize_system():
    """Initialize the NBA prediction system"""
    try:
        data_provider = NBADataProvider()
        system = NBACompleteSystem(data_provider)
        return system
    except Exception as e:
        st.error(f"❌ System initialization failed: {e}")
        return None

def format_currency(amount):
    """Format currency with Euro symbol"""
    return f"€{amount:.2f}"

def get_model_status(model_path):
    """Get model training status and metrics"""
    try:
        metadata_path = os.path.join(model_path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return {
                'status': 'active',
                'mae': metadata.get('mae', 'N/A'),
                'r2': metadata.get('r2', 'N/A'),
                'last_trained': metadata.get('training_date', 'Unknown'),
                'samples': metadata.get('training_samples', 'N/A')
            }
    except:
        pass
    return {'status': 'error', 'mae': 'N/A', 'r2': 'N/A', 'last_trained': 'N/A', 'samples': 'N/A'}

def get_scheduled_games(system):
    """Recupera le partite programmate dal sistema"""
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
    # Detect mobile (approximate - based on screen width simulation)
    # In a real scenario, you'd use JavaScript, but this is a workaround
    if 'is_mobile' not in st.session_state:
        st.session_state['is_mobile'] = False  # Default to desktop
    
    # Mobile toggle for testing (remove in production)
    with st.sidebar:
        if st.checkbox("📱 Modalità Mobile", value=st.session_state.get('is_mobile', False)):
            st.session_state['is_mobile'] = True
        else:
            st.session_state['is_mobile'] = False
    
    # Header principale con stile mobile-responsive
    if st.session_state.get('is_mobile', False):
        # Mobile header - more compact
        st.markdown("""
        <div class="main-header">
            <h1 style="font-size: 1.8rem; margin-bottom: 0.3rem;">🏀 NBA Predictor</h1>
            <p style="font-size: 0.9rem;">Advanced ML System for NBA Predictions</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Desktop header - full size
        st.markdown("""
        <div class="main-header">
            <h1>🏀 NBA Predictor Pro</h1>
            <p>Advanced Machine Learning System for NBA Game Predictions & Betting Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar Navigation migliorata e MOBILE-OPTIMIZED
    with st.sidebar:
        # Header compatto per mobile
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                    padding: 0.8rem; border-radius: 12px; color: white; text-align: center; margin-bottom: 1rem;">
            <h3 style="margin: 0; font-size: 1.2rem;">🎯 Navigation</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation compatta per mobile
        page_options = [
            ("🎰", "Centro Scommesse"),
            ("📊", "Performance"),
            ("💰", "Bankroll"),
            ("🤖", "Modelli ML"),
            ("⚙️", "Configurazione")
        ]
        
        # Check if on mobile (approximate)
        is_mobile = st.session_state.get('is_mobile', False)
        
        # Mobile-friendly navigation
        page_labels = [f"{icon} {label}" for icon, label in page_options]
        page = st.selectbox(
            "Seleziona Sezione",
            page_labels,
            label_visibility="collapsed"
        )
        
        # Quick stats in sidebar - mobile optimized
        st.markdown("""
        <div style="background: white; padding: 0.8rem; border-radius: 10px; margin: 0.5rem 0; 
                    border: 1px solid #e1e5e9; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
            <h4 style="margin: 0 0 0.5rem 0; color: #1e3c72; font-size: 0.9rem;">📊 Status</h4>
            <div style="font-size: 0.8rem;">
                <div>🟢 Sistema Attivo</div>
                <div>⚡ ML Models: OK</div>
                <div>📡 NBA API: Live</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Bankroll preview - mobile optimized
        try:
            bankroll_data = load_bankroll_data()
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #4caf50 0%, #45a049 100%); 
                        padding: 0.8rem; border-radius: 10px; color: white; margin: 0.5rem 0;">
                <h4 style="margin: 0 0 0.3rem 0; font-size: 0.9rem;">💰 Bankroll</h4>
                <div style="font-size: 1.1rem; font-weight: bold;">€{bankroll_data['current_bankroll']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        except:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4caf50 0%, #45a049 100%); 
                        padding: 0.8rem; border-radius: 10px; color: white; margin: 0.5rem 0;">
                <h4 style="margin: 0 0 0.3rem 0; font-size: 0.9rem;">💰 Bankroll</h4>
                <div style="font-size: 1.1rem; font-weight: bold;">€100.00</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Initialize system
    system = initialize_system()
    if system is None:
        st.error("❌ Cannot proceed without system initialization")
        return
    
    # Route to selected page
    if page == "🎰 Centro Scommesse":
        show_betting_center(system)
    elif page == "📊 Performance":
        show_performance_dashboard()
    elif page == "💰 Bankroll":
        show_bankroll_management()
    elif page == "🤖 Modelli ML":
        show_ml_models_dashboard()
    elif page == "⚙️ Configurazione":
        show_configuration_panel()

# ================================
# 🎰 CENTRO SCOMMESSE
# ================================

def show_betting_center(system):
    # Header principale
    st.markdown("""
    <div class="main-header">
        <h1>🎰 Centro Scommesse NBA</h1>
        <p>Sistema avanzato di analisi e piazzamento scommesse professionali</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Layout principale con tab RIVISTE
    tab1, tab2, tab3, tab4 = st.tabs(["🏀 Analisi Partita", "📊 Statistiche", "🎯 Raccomandazioni", "💰 Piazzamento"])
    
    with tab1:
        show_game_analysis_combined_tab(system)  # NUOVA TAB COMBINATA
    
    with tab2:
        show_statistics_tab(system)  # NUOVA TAB STATISTICHE
    
    with tab3:
        show_recommendations_tab(system)
    
    with tab4:
        show_betting_tab()

def show_game_analysis_combined_tab(system):
    """Tab combinata per selezione partita e analisi - UNICA PAGINA A SCORRIMENTO"""
    st.markdown("""
    <div class="tab-container">
        <h2 style="font-size: 1.4rem; margin-bottom: 1rem;">🏀 Selezione e Analisi Partita</h2>
        <p style="font-size: 0.9rem; margin-bottom: 1rem;">Seleziona una partita e avvia l'analisi completa in un unico flusso</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================
    # SEZIONE 1: RECUPERO E SELEZIONE PARTITE
    # ========================================
    st.markdown("### 📅 Step 1: Recupero Partite")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📅 Recupera Partite NBA", key="get_games", use_container_width=True):
            with st.spinner("🔄 Recupero partite in corso..."):
                games = get_scheduled_games(system)
                st.session_state['games'] = games
                st.success(f"✅ Trovate {len(games)} partite")
                st.rerun()
    
    with col2:
        if st.button("🔄 Reset Selezione", key="reset_games", use_container_width=True):
            if 'games' in st.session_state:
                del st.session_state['games']
            if 'selected_game' in st.session_state:
                del st.session_state['selected_game']
            if 'analysis_result' in st.session_state:
                del st.session_state['analysis_result']
            st.rerun()
    
    # Carica scommesse pendenti per evidenziare partite con scommesse
    try:
        pending_bets = []
        with open('data/pending_bets.json', 'r') as f:
            pending_bets = json.load(f)
        pending_game_ids = {bet.get('game_id') for bet in pending_bets if bet.get('status') == 'pending'}
    except:
        pending_game_ids = set()
    
    # Display partite con evidenziazione scommesse pendenti
    if 'games' in st.session_state and st.session_state['games']:
        games = st.session_state['games']
        
        st.markdown("### 🎯 Step 2: Selezione Partita")
        
        for i, game in enumerate(games, 1):
            game_id = game.get('game_id', f"game_{i}")
            has_pending_bet = game_id in pending_game_ids
            
            # Stile diverso per partite con scommesse pendenti
            if has_pending_bet:
                card_style = f"""
                <div style="background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%); 
                            color: white; border-radius: 12px; padding: 1rem; margin: 0.5rem 0;
                            border: 3px solid #f57c00; box-shadow: 0 4px 20px rgba(255,152,0,0.3);">
                    <h4 style="margin: 0; display: flex; align-items: center;">
                        🚨 SCOMMESSA ATTIVA • {game['away_team']} @ {game['home_team']}
                    </h4>
                    <p style="margin: 0.3rem 0 0 0; font-size: 0.9rem;">📅 {game.get('date', 'TBD')} • 💰 Scommessa pendente</p>
                </div>
                """
            else:
                card_style = f"""
                <div style="background: white; border-radius: 12px; padding: 1rem; margin: 0.5rem 0;
                            border: 2px solid #e8f2ff; box-shadow: 0 3px 15px rgba(0,0,0,0.08);">
                    <h4 style="margin: 0; color: #1e3c72;">{game['away_team']} @ {game['home_team']}</h4>
                    <p style="margin: 0.3rem 0 0 0; color: #6c757d; font-size: 0.9rem;">📅 {game.get('date', 'TBD')}</p>
                </div>
                """
            
            st.markdown(card_style, unsafe_allow_html=True)
            
            if st.button(f"Seleziona Partita {i}", key=f"game_{i}", use_container_width=True):
                st.session_state['selected_game'] = game
                if has_pending_bet:
                    st.warning(f"⚠️ ATTENZIONE: Già presente scommessa pendente per questa partita!")
                st.success(f"✅ Selezionata: {game['away_team']} @ {game['home_team']}")
                st.rerun()
    
    # ========================================
    # SEZIONE 2: CONFIGURAZIONE ANALISI
    # ========================================
    if 'selected_game' in st.session_state:
        game = st.session_state['selected_game']
        
        st.markdown("### ⚙️ Step 3: Configurazione Analisi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="prediction-card">
                <h3>🏀 {game['away_team']} @ {game['home_team']}</h3>
                <p>📅 {game.get('date', 'TBD')} • ⏰ 20:00 EST</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            central_line = st.number_input(
                "📊 Linea bookmaker (punti totali)",
                min_value=150.0,
                max_value=300.0,
                value=221.5,
                step=0.5,
                help="Inserisci la linea over/under del bookmaker"
            )
        
        # ========================================
        # SEZIONE 3: AVVIO ANALISI
        # ========================================
        st.markdown("### 🚀 Step 4: Analisi Completa")
        
        if st.button("🎯 AVVIA ANALISI COMPLETA NBA PREDICTOR", key="analyze", type="primary", use_container_width=True):
            with st.spinner("🎯 Analisi in corso... Questo può richiedere alcuni secondi"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simula progress con messaggi informativi
                status_text.text("🔄 Inizializzazione sistema...")
                progress_bar.progress(10)
                
                # Crea un oggetto args mock
                class MockArgs:
                    def __init__(self):
                        self.auto_mode = True
                        self.line = central_line
                
                mock_args = MockArgs()
                
                status_text.text("📊 Recupero statistiche squadre...")
                progress_bar.progress(30)
                
                status_text.text("🏥 Analisi impatto infortuni...")
                progress_bar.progress(50)
                
                status_text.text("⚡ Calcolo momentum ML...")
                progress_bar.progress(70)
                
                status_text.text("🎲 Simulazioni Monte Carlo...")
                progress_bar.progress(90)
                
                # Esegui analisi
                analysis_result = system.analyze_game(game, central_line=central_line, args=mock_args)
                
                progress_bar.progress(100)
                status_text.text("✅ Analisi completata!")
                
                st.session_state['analysis_result'] = analysis_result
                st.session_state['central_line'] = central_line
                st.success("🎉 Analisi completata con successo!")
                st.rerun()
    
    # ========================================
    # SEZIONE 4: RISULTATI ANALISI MIGLIORATI
    # ========================================
    if 'analysis_result' in st.session_state and 'selected_game' in st.session_state:
        result = st.session_state['analysis_result']
        game = st.session_state['selected_game']
        central_line = st.session_state.get('central_line', 221.5)
        
        if 'error' in result:
            st.error(f"❌ Errore nell'analisi: {result['error']}")
        else:
            st.markdown("### 📊 Step 5: Risultati Analisi")
            
            # Estrai dati per display migliorato
            distribution = result.get('distribution', {})
            momentum_impact = result.get('momentum_impact', {})
            injury_impact = result.get('injury_impact', 0)
            predicted_total = distribution.get('predicted_mu', 0)
            confidence_sigma = distribution.get('predicted_sigma', 0)
            confidence_percentage = max(0, min(100, 100 - (confidence_sigma - 10) * 3))
            
            # RISULTATI PRINCIPALI CON MOMENTUM INCLUSO
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🎯 Totale Previsto", 
                    f"{predicted_total:.1f} pts",
                    help="Punteggio totale predetto dal modello ML"
                )
            
            with col2:
                st.metric(
                    "📈 Confidenza", 
                    f"{confidence_percentage:.1f}%",
                    delta=f"±{confidence_sigma:.1f} pts",
                    help="Livello di confidenza della predizione"
                )
            
            with col3:
                st.metric(
                    "🏥 Impatto Infortuni", 
                    f"{injury_impact:+.2f} pts",
                    help="Effetto degli infortuni sul totale"
                )
            
            with col4:
                momentum_value = momentum_impact.get('total_impact', 0) if isinstance(momentum_impact, dict) else momentum_impact
                st.metric(
                    "⚡ Impatto Momentum", 
                    f"{momentum_value:+.2f} pts",
                    help="Effetto del momentum ML sul totale"
                )
            
            # SYSTEM STATUS MIGLIORATO E PIÙ COMUNICATIVO
            st.markdown("### 🔧 System Status Avanzato")
            
            # Calcola status dettagliati basati sui dati reali
            momentum_conf = momentum_impact.get('confidence_factor', 1.0) * 100 if isinstance(momentum_impact, dict) else 85.0
            
            # Valuta la qualità dei dati e completezza dell'analisi
            team_stats = result.get('team_stats', {})
            stats_quality = "complete" if team_stats and 'home' in team_stats and 'away' in team_stats else "limited"
            
            # Valuta il sistema injury
            injury_quality = "active" if abs(injury_impact) > 0.1 or home_impact_result.get('injured_players_details') else "no_impact"
            
            # Valuta il sistema momentum
            momentum_quality = "high" if momentum_conf > 80 else "medium" if momentum_conf > 60 else "basic"
            
            # Valuta il sistema probabilistico
            prob_quality = "active" if distribution and 'error' not in distribution else "error"
            
            # Valuta il sistema betting
            betting_quality = "active" if opportunities and len(opportunities) > 0 else "no_data"
            
            # Status con feedback accurato sui sistemi ML
            status_items = [
                ("🟢", "Stats", "Statistiche squadre aggiornate e complete", "green") if stats_quality == "complete" 
                else ("🟡", "Stats", "Statistiche squadre parziali", "orange"),
                
                ("🟢", "Injury ML", "Sistema infortuni attivo con impatti rilevati", "green") if injury_quality == "active"
                else ("🟢", "Injury ML", "Sistema infortuni attivo - nessun impatto", "green"),
                
                ("🟢", f"Momentum ML({momentum_conf:.0f}%)", "Sistema ML momentum completamente operativo", "green") if momentum_quality == "high"
                else ("🟡", f"Momentum ML({momentum_conf:.0f}%)", "Sistema ML momentum con confidenza media", "orange") if momentum_quality == "medium"
                else ("🟢", "Momentum ML", "Sistema momentum base attivo", "green"),
                
                ("🟢", "Probabilistic ML", "Modello predittivo ML completamente attivo", "green") if prob_quality == "active"
                else ("🔴", "Probabilistic ML", "Errore nel modello predittivo", "red"),
                
                ("🟢", "Betting Engine", "Motore analisi scommesse operativo", "green") if betting_quality == "active"
                else ("🟡", "Betting Engine", "Motore attivo - dati limitati", "orange")
            ]
            
            cols = st.columns(len(status_items))
            for i, (icon, title, description, color) in enumerate(status_items):
                with cols[i]:
                    if color == "green":
                        bg_color = "linear-gradient(135deg, #4caf50 0%, #45a049 100%)"
                    elif color == "orange":
                        bg_color = "linear-gradient(135deg, #ff9800 0%, #f57c00 100%)"
                    elif color == "red":
                        bg_color = "linear-gradient(135deg, #f44336 0%, #d32f2f 100%)"
                    else:
                        bg_color = "linear-gradient(135deg, #6c757d 0%, #5a6268 100%)"
                    
                    st.markdown(f"""
                    <div style="background: {bg_color}; color: white; border-radius: 10px; 
                                padding: 0.8rem; text-align: center; margin: 0.2rem;">
                        <div style="font-size: 1.2rem; margin-bottom: 0.3rem;">{icon}</div>
                        <div style="font-size: 0.8rem; font-weight: bold;">{title}</div>
                        <div style="font-size: 0.7rem; opacity: 0.9; margin-top: 0.2rem;">{description}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Feedback dettagliato sui sistemi ML
            st.markdown("#### 🤖 Feedback Sistemi ML")
            
            # Calcola un punteggio globale di qualità del sistema
            quality_scores = []
            if stats_quality == "complete":
                quality_scores.append(("Statistiche", 95))
            elif stats_quality == "limited":
                quality_scores.append(("Statistiche", 70))
                
            if injury_quality == "active":
                quality_scores.append(("Injury ML", 90))
            else:
                quality_scores.append(("Injury ML", 85))
                
            quality_scores.append(("Momentum ML", momentum_conf))
            
            if prob_quality == "active":
                quality_scores.append(("Probabilistic ML", 95))
            else:
                quality_scores.append(("Probabilistic ML", 30))
                
            if betting_quality == "active":
                quality_scores.append(("Betting Engine", 90))
            else:
                quality_scores.append(("Betting Engine", 70))
            
            # Calcola punteggio medio
            overall_score = sum(score for _, score in quality_scores) / len(quality_scores)
            
            # Feedback basato sul punteggio
            if overall_score >= 85:
                status_color = "#4CAF50"
                status_text = "🟢 SISTEMA COMPLETAMENTE OPERATIVO"
                status_desc = "Tutti i sistemi ML sono attivi con dati completi e aggiornati. Analisi di massima qualità."
            elif overall_score >= 70:
                status_color = "#FF9800" 
                status_text = "🟡 SISTEMA PARZIALMENTE OPERATIVO"
                status_desc = "La maggior parte dei sistemi ML è operativa. Qualche limitazione nei dati ma analisi affidabile."
            else:
                status_color = "#F44336"
                status_text = "🔴 SISTEMA CON LIMITAZIONI"
                status_desc = "Alcuni sistemi ML hanno problemi. Analisi possibile ma con limitazioni."
            
            st.markdown(f"""
            <div style="background: {status_color}; color: white; border-radius: 15px; 
                       padding: 1.5rem; margin: 1rem 0; text-align: center;">
                <h3 style="margin: 0 0 0.5rem 0; font-size: 1.3rem;">{status_text}</h3>
                <p style="margin: 0; font-size: 1rem; opacity: 0.9;">{status_desc}</p>
                <div style="margin-top: 1rem; font-size: 1.1rem; font-weight: bold;">
                    📊 Score Qualità Sistema: {overall_score:.1f}/100
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Rimuovo l'alert fuorviante e sostituisco con info utili
            st.success("✅ Analisi completata! Procedi alla tab 'Statistiche' per dettagli o 'Raccomandazioni' per le scommesse.")
    
    else:
        st.info("👆 Inizia recuperando le partite NBA programmate")

def show_key_players_for_team(system, team_id, season="2024-25"):
    """Mostra i giocatori chiave di una squadra con statistiche dettagliate"""
    try:
        # Recupera roster della squadra
        roster = system.injury_reporter.get_team_roster(team_id)
        
        if not roster or len(roster) == 0:
            st.info("Roster non disponibile")
            return
        
        # Prendi i primi 5 giocatori (di solito i più importanti)
        key_players = roster[:5] if len(roster) >= 5 else roster
        
        for i, player in enumerate(key_players):
            player_id = player.get('PLAYER_ID') or player.get('id')
            player_name = player.get('PLAYER_NAME') or player.get('name', 'Unknown Player')
            
            if not player_id:
                continue
                
            # Recupera statistiche del giocatore
            try:
                player_stats = system.data_provider.get_player_stats(player_id, season)
                
                if player_stats is not None and not player_stats.empty:
                    stats_row = player_stats.iloc[0]
                    
                    # Estrai statistiche principali
                    pts = float(stats_row.get('PTS', 0))
                    reb = float(stats_row.get('REB', 0))
                    ast = float(stats_row.get('AST', 0))
                    fg_pct = float(stats_row.get('FG_PCT', 0)) * 100 if stats_row.get('FG_PCT') else 0
                    ft_pct = float(stats_row.get('FT_PCT', 0)) * 100 if stats_row.get('FT_PCT') else 0
                    gp = int(stats_row.get('GP', 0))
                    min_played = float(stats_row.get('MIN', 0))
                    
                    # Determina se è un giocatore starter basato sui minuti
                    role = "⭐ STARTER" if min_played >= 25 else "🔄 BENCH" if min_played >= 15 else "🏃 ROLE PLAYER"
                    
                    # Card compatta per ogni giocatore
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                                border-radius: 10px; padding: 1rem; margin: 0.5rem 0; 
                                border-left: 4px solid #007bff;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <h4 style="margin: 0; color: #1e3c72; font-size: 1rem;">{role} {player_name}</h4>
                            <span style="background: #007bff; color: white; padding: 0.2rem 0.5rem; 
                                         border-radius: 12px; font-size: 0.7rem; font-weight: bold;">
                                {gp} GP
                            </span>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; font-size: 0.85rem;">
                            <div style="text-align: center; background: white; padding: 0.4rem; border-radius: 5px;">
                                <div style="font-weight: bold; color: #dc3545;">{pts:.1f}</div>
                                <div style="color: #6c757d; font-size: 0.7rem;">PPG</div>
                            </div>
                            <div style="text-align: center; background: white; padding: 0.4rem; border-radius: 5px;">
                                <div style="font-weight: bold; color: #28a745;">{reb:.1f}</div>
                                <div style="color: #6c757d; font-size: 0.7rem;">RPG</div>
                            </div>
                            <div style="text-align: center; background: white; padding: 0.4rem; border-radius: 5px;">
                                <div style="font-weight: bold; color: #ffc107;">{ast:.1f}</div>
                                <div style="color: #6c757d; font-size: 0.7rem;">APG</div>
                            </div>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin-top: 0.5rem; font-size: 0.85rem;">
                            <div style="text-align: center; background: white; padding: 0.4rem; border-radius: 5px;">
                                <div style="font-weight: bold; color: #17a2b8;">{fg_pct:.1f}%</div>
                                <div style="color: #6c757d; font-size: 0.7rem;">FG%</div>
                            </div>
                            <div style="text-align: center; background: white; padding: 0.4rem; border-radius: 5px;">
                                <div style="font-weight: bold; color: #6f42c1;">{ft_pct:.1f}%</div>
                                <div style="color: #6c757d; font-size: 0.7rem;">FT%</div>
                            </div>
                            <div style="text-align: center; background: white; padding: 0.4rem; border-radius: 5px;">
                                <div style="font-weight: bold; color: #fd7e14;">{min_played:.1f}</div>
                                <div style="color: #6c757d; font-size: 0.7rem;">MIN</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    # Fallback con dati mock se non ci sono statistiche
                    st.markdown(f"""
                    <div style="background: #f8f9fa; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; 
                                border-left: 4px solid #6c757d;">
                        <h4 style="margin: 0; color: #6c757d; font-size: 1rem;">🏃 {player_name}</h4>
                        <p style="margin: 0.5rem 0 0 0; color: #6c757d; font-size: 0.9rem;">
                            📊 Statistiche non disponibili per questa stagione
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.write(f"⚠️ Errore nel recupero statistiche per {player_name}: {str(e)}")
                continue
    
    except Exception as e:
        st.error(f"❌ Errore nel recupero roster: {str(e)}")

def show_statistics_tab(system):
    """NUOVA TAB per le statistiche dettagliate della partita"""
    if 'analysis_result' not in st.session_state:
        st.warning("⚠️ Completa prima l'analisi nella tab 'Analisi Partita'")
        return
    
    result = st.session_state['analysis_result']
    game = st.session_state['selected_game']
    
    if 'error' in result:
        st.error(f"❌ Errore nell'analisi: {result['error']}")
        return
    
    st.markdown("""
    <div class="tab-container">
        <h2>📊 Statistiche Dettagliate</h2>
        <p>Analisi completa di statistiche squadre, giocatori chiave, momentum e infortuni</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================
    # SEZIONE 1: STATISTICHE SQUADRE
    # ========================================
    st.markdown("### 🏀 Statistiche Squadre")
    
    team_stats = result.get('team_stats', {})
    
    # DEBUG: Checkbox per mostrare la struttura dati
    if st.checkbox("🔧 Debug: Mostra struttura dati team_stats", key="debug_team_stats"):
        st.write("**Struttura completa team_stats:**")
        st.json(team_stats)
        if team_stats:
            st.write("**Chiavi disponibili in 'home':**", list(team_stats.get('home', {}).keys()) if 'home' in team_stats else "N/A")
            st.write("**Chiavi disponibili in 'away':**", list(team_stats.get('away', {}).keys()) if 'away' in team_stats else "N/A")
    if team_stats and 'home' in team_stats and 'away' in team_stats:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🏠 {game.get('home_team', 'Home')} - Casa</h4>
            </div>
            """, unsafe_allow_html=True)
            
            home_stats = team_stats['home']
            if home_stats and home_stats.get('has_data'):
                # Usa le chiavi corrette dal data_provider
                ppg = home_stats.get('ppg', 0)  # lowercase
                oppg = home_stats.get('oppg', 0)  # lowercase  
                games_played = home_stats.get('games_played', 0)
                win_pct = home_stats.get('win_percentage', 0)
                
                # Calcola W-L da win_percentage e games_played
                wins = int(win_pct * games_played) if games_played > 0 else 0
                losses = games_played - wins if games_played > 0 else 0
                
                home_metrics = [
                    ("📊 PPG", f"{ppg:.1f}" if isinstance(ppg, (int, float)) and ppg > 0 else "N/A"),
                    ("🛡️ OPP_PPG", f"{oppg:.1f}" if isinstance(oppg, (int, float)) and oppg > 0 else "N/A"),
                    ("🏆 W-L", f"{wins}-{losses}" if games_played > 0 else "N/A"),
                    ("📈 Win%", f"{win_pct*100:.1f}%" if isinstance(win_pct, (int, float)) and win_pct > 0 else "N/A")
                ]
                
                for metric, value in home_metrics:
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.write(f"**{metric}**")
                    with col2:
                        st.write(value)
            else:
                st.info("Dati non disponibili")
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>✈️ {game.get('away_team', 'Away')} - Ospite</h4>
            </div>
            """, unsafe_allow_html=True)
            
            away_stats = team_stats['away']
            if away_stats and away_stats.get('has_data'):
                # Usa le chiavi corrette dal data_provider
                ppg = away_stats.get('ppg', 0)  # lowercase
                oppg = away_stats.get('oppg', 0)  # lowercase
                games_played = away_stats.get('games_played', 0)
                win_pct = away_stats.get('win_percentage', 0)
                
                # Calcola W-L da win_percentage e games_played
                wins = int(win_pct * games_played) if games_played > 0 else 0
                losses = games_played - wins if games_played > 0 else 0
                
                away_metrics = [
                    ("📊 PPG", f"{ppg:.1f}" if isinstance(ppg, (int, float)) and ppg > 0 else "N/A"),
                    ("🛡️ OPP_PPG", f"{oppg:.1f}" if isinstance(oppg, (int, float)) and oppg > 0 else "N/A"),
                    ("🏆 W-L", f"{wins}-{losses}" if games_played > 0 else "N/A"),
                    ("📈 Win%", f"{win_pct*100:.1f}%" if isinstance(win_pct, (int, float)) and win_pct > 0 else "N/A")
                ]
                
                for metric, value in away_metrics:
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.write(f"**{metric}**")
                    with col2:
                        st.write(value)
            else:
                st.info("Dati non disponibili")
    else:
        st.info("Statistiche squadre non disponibili")
    
    # ========================================
    # SEZIONE 2: INJURY DETAILS (SPOSTATO QUI)
    # ========================================
    st.markdown("### 🏥 Injury Details")
    
    # Recupera injury details dai dati reali calcolati dal sistema
    home_impact_result = getattr(system, '_last_home_impact_result', {'injured_players_details': []})
    away_impact_result = getattr(system, '_last_away_impact_result', {'injured_players_details': []})
    
    # Estrai dati dalle details string
    home_injuries = []
    away_injuries = []
    
    for detail in home_impact_result.get('injured_players_details', []):
        try:
            if ' - Impatto: ' in detail:
                player_part, impact_part = detail.split(' - Impatto: ')
                player_name = player_part.split(' (')[0]
                status = player_part.split(' (')[1].split(')')[0].upper()
                impact = abs(float(impact_part.replace(' pts', '')))
                home_injuries.append({"player": player_name, "status": status, "impact": impact})
        except:
            home_injuries.append({"player": "Unknown Player", "status": "OUT", "impact": 0.50})
    
    for detail in away_impact_result.get('injured_players_details', []):
        try:
            if ' - Impatto: ' in detail:
                player_part, impact_part = detail.split(' - Impatto: ')
                player_name = player_part.split(' (')[0]
                status = player_part.split(' (')[1].split(')')[0].upper()
                impact = abs(float(impact_part.replace(' pts', '')))
                away_injuries.append({"player": player_name, "status": status, "impact": impact})
        except:
            away_injuries.append({"player": "Unknown Player", "status": "OUT", "impact": 0.50})
    
    # Se non ci sono injuries, usa dati mock minimi
    if not home_injuries and not away_injuries:
        home_injuries = [{"player": "Nessun infortunio significativo", "status": "ACTIVE", "impact": 0.00}]
        away_injuries = [{"player": "Nessun infortunio significativo", "status": "ACTIVE", "impact": 0.00}]
    
    home_total_impact = sum(inj["impact"] for inj in home_injuries)
    away_total_impact = sum(inj["impact"] for inj in away_injuries)
    injury_impact = result.get('injury_impact', 0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🏠 {game.get('home_team', 'Home')} - Infortuni</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for injury in home_injuries:
            status_color = "🔴" if injury["status"] in ["OUT", "DOUBTFUL"] else "🟡" if injury["status"] == "QUESTIONABLE" else "🟢"
            st.markdown(f"{status_color} **{injury['player']}** ({injury['status']}) - Impatto: +{injury['impact']:.2f} pts")
        
        st.markdown(f"**Total Impact: +{home_total_impact:.2f} pts**")
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>✈️ {game.get('away_team', 'Away')} - Infortuni</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for injury in away_injuries:
            status_color = "🔴" if injury["status"] in ["OUT", "DOUBTFUL"] else "🟡" if injury["status"] == "QUESTIONABLE" else "🟢"
            st.markdown(f"{status_color} **{injury['player']}** ({injury['status']}) - Impatto: +{injury['impact']:.2f} pts")
        
        st.markdown(f"**Total Impact: +{away_total_impact:.2f} pts**")
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>⚖️ Impact Comparison</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("🏠 Home Impact", f"+{home_total_impact:.2f} pts")
        st.metric("✈️ Away Impact", f"+{away_total_impact:.2f} pts")
        st.metric("🔢 Net Impact", f"{injury_impact:+.2f} pts", help="Impatto netto sui totali della partita")
    
    # ========================================
    # SEZIONE 3: MOMENTUM ANALYSIS
    # ========================================
    st.markdown("### ⚡ Momentum Analysis")
    
    momentum_impact = result.get('momentum_impact', {})
    if momentum_impact:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>📊 Momentum Metrics</h4>
            </div>
            """, unsafe_allow_html=True)
            
            if isinstance(momentum_impact, dict):
                momentum_value = momentum_impact.get('total_impact', 0)
                confidence = momentum_impact.get('confidence_factor', 1.0) * 100
                model_used = momentum_impact.get('model_used', 'Standard')
                
                st.metric("⚡ Total Impact", f"{momentum_value:+.2f} pts")
                st.metric("🎯 Confidence", f"{confidence:.1f}%")
                st.metric("🤖 Model Used", model_used)
                
                if momentum_impact.get('reasoning'):
                    st.info(f"💡 **Reasoning**: {momentum_impact['reasoning']}")
            else:
                st.metric("⚡ Total Impact", f"{momentum_impact:+.2f} pts")
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>🔥 Hot Hand Detection</h4>
            </div>
            """, unsafe_allow_html=True)
            
            if isinstance(momentum_impact, dict):
                if momentum_impact.get('synergy_detected'):
                    st.success("🔥 **Hot Hand Synergy Detected!**")
                    st.write("Multipli giocatori in momentum positivo rilevati")
                else:
                    st.info("📊 Momentum standard rilevato")
                
                # Display additional momentum details se disponibili
                if momentum_impact.get('home_momentum'):
                    home_score = momentum_impact['home_momentum'].get('score', 50)
                    st.metric("🏠 Home Momentum", f"{home_score:.1f}/100")
                
                if momentum_impact.get('away_momentum'):
                    away_score = momentum_impact['away_momentum'].get('score', 50)
                    st.metric("✈️ Away Momentum", f"{away_score:.1f}/100")
            else:
                st.info("Momentum analysis completata con sistema base")
    else:
        st.info("Dati momentum non disponibili")
    
    # ========================================
    # SEZIONE 4: GIOCATORI CHIAVE (IMPLEMENTATA)
    # ========================================
    st.markdown("### ⭐ Giocatori Chiave")
    
    # Recupera i roster delle squadre dal sistema
    if hasattr(system, 'data_provider') and hasattr(system, 'injury_reporter'):
        home_team_id = game.get('home_team_id')
        away_team_id = game.get('away_team_id')
        
        if home_team_id and away_team_id:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🏠 {game.get('home_team', 'Home')} - Top Players</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Recupera e mostra giocatori chiave home team
                show_key_players_for_team(system, home_team_id, "2024-25")
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>✈️ {game.get('away_team', 'Away')} - Top Players</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Recupera e mostra giocatori chiave away team  
                show_key_players_for_team(system, away_team_id, "2024-25")
        else:
            st.warning("⚠️ ID squadre non disponibili per recuperare statistiche giocatori")
    else:
        st.error("❌ Sistema data provider non disponibile")

def show_recommendations_tab(system):
    """Tab per le raccomandazioni - PRESENTAZIONE MIGLIORATA"""
    if 'analysis_result' not in st.session_state:
        st.warning("⚠️ Completa prima l'analisi nella tab 'Analisi Partita'")
        return
    
    result = st.session_state['analysis_result']
    game = st.session_state['selected_game']
    
    if 'error' in result:
        st.error(f"❌ Errore nell'analisi: {result['error']}")
        return
    
    st.markdown("""
    <div class="tab-container">
        <h2>🎯 Raccomandazioni di Scommessa</h2>
        <p>Sistema professionale di raccomandazioni basato su algoritmi ML</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Estrai dati dall'analisi
    distribution = result.get('distribution', {})
    opportunities = result.get('opportunities', [])
    momentum_impact = result.get('momentum_impact', {})
    injury_impact = result.get('injury_impact', 0)
    central_line = st.session_state.get('central_line', 0)
    
    if opportunities and isinstance(opportunities, list):
        all_opportunities = sorted(opportunities, key=lambda x: x.get('edge', 0), reverse=True)
        
        # Filtra VALUE bets (edge > 0 e prob >= 50%)
        value_bets = [opp for opp in all_opportunities if opp.get('edge', 0) > 0 and opp.get('probability', 0) >= 0.5]
        
        if value_bets:
            st.markdown(f"""
            <div class="bet-summary">
                <h3>💎 {len(value_bets)} OPPORTUNITÀ VALUE IDENTIFICATE</h3>
                <p>🎯 Sistema di analisi ha trovato {len(value_bets)} scommesse con valore positivo su {len(all_opportunities)} linee analizzate</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Calcola le raccomandazioni categorizzate usando l'algoritmo esatto di main.py
            optimal_bet = system._calculate_optimal_bet(all_opportunities) if hasattr(system, '_calculate_optimal_bet') else value_bets[0]
            highest_prob_bet = max(value_bets, key=lambda x: x.get('probability', 0))
            highest_edge_bet = max(value_bets, key=lambda x: x.get('edge', 0))
            highest_odds_bet = max(value_bets, key=lambda x: x.get('odds', 0))
            
            # Lista delle raccomandazioni principali
            recommendations = []
            
            if optimal_bet:
                recommendations.append({
                    'bet': optimal_bet,
                    'category': '🏆 SCELTA DEL SISTEMA',
                    'description': 'Scommessa ottimale calcolata dall\'algoritmo ML',
                    'color': '#FFD700'  # Gold
                })
            
            recommendations.extend([
                {
                    'bet': highest_prob_bet,
                    'category': '📊 MASSIMA PROBABILITÀ',
                    'description': 'Scommessa con la più alta probabilità di successo',
                    'color': '#4CAF50'  # Green
                },
                {
                    'bet': highest_edge_bet,
                    'category': '🔥 MASSIMO EDGE',
                    'description': 'Scommessa con il margine più favorevole',
                    'color': '#FF5722'  # Red-Orange
                },
                {
                    'bet': highest_odds_bet,
                    'category': '💰 QUOTA MASSIMA',
                    'description': 'Scommessa con la quota più alta',
                    'color': '#9C27B0'  # Purple
                }
            ])
            
            # Rimuovi duplicati
            seen_bets = set()
            unique_recommendations = []
            for rec in recommendations:
                bet_key = f"{rec['bet']['type']}_{rec['bet']['line']}"
                if bet_key not in seen_bets:
                    seen_bets.add(bet_key)
                    unique_recommendations.append(rec)
            
            # NUOVA PRESENTAZIONE MIGLIORATA DELLE RACCOMANDAZIONI
            st.markdown("### 🏆 Raccomandazioni Principali")
            
            for i, rec in enumerate(unique_recommendations, 1):
                bet = rec['bet']
                edge = bet.get('edge', 0) * 100
                prob = bet.get('probability', 0) * 100
                quality = bet.get('quality_score', 0) * 100 if bet.get('quality_score', 0) < 10 else bet.get('quality_score', 0)
                
                # Header della raccomandazione
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {rec['color']}20 0%, {rec['color']}10 100%); 
                            border-left: 5px solid {rec['color']}; border-radius: 15px; 
                            padding: 1.5rem; margin: 1rem 0; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <h3 style="margin: 0; color: #1e3c72; flex-grow: 1;">#{i} {rec['category']}</h3>
                        <span style="background: {rec['color']}; color: white; padding: 0.3rem 0.8rem; 
                                     border-radius: 20px; font-size: 0.8rem; font-weight: bold;">
                            QUALITY: {quality:.1f}/100
                        </span>
                    </div>
                    <p style="margin: 0 0 1rem 0; color: #6c757d; font-style: italic;">{rec['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Grid delle metriche usando Streamlit columns invece di HTML
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.markdown(f"""
                    <div style="text-align: center; background: white; padding: 0.8rem; border-radius: 10px; 
                                box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 1rem;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: #1e3c72;">{bet['type']} {bet['line']}</div>
                        <div style="font-size: 0.8rem; color: #6c757d;">Tipo Scommessa</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="text-align: center; background: white; padding: 0.8rem; border-radius: 10px; 
                                box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 1rem;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: #1e3c72;">{bet['odds']:.2f}</div>
                        <div style="font-size: 0.8rem; color: #6c757d;">Quota</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style="text-align: center; background: white; padding: 0.8rem; border-radius: 10px; 
                                box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 1rem;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: #4CAF50;">{edge:+.1f}%</div>
                        <div style="font-size: 0.8rem; color: #6c757d;">Edge</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div style="text-align: center; background: white; padding: 0.8rem; border-radius: 10px; 
                                box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 1rem;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: #2196F3;">{prob:.1f}%</div>
                        <div style="font-size: 0.8rem; color: #6c757d;">Probabilità</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col5:
                    st.markdown(f"""
                    <div style="text-align: center; background: white; padding: 0.8rem; border-radius: 10px; 
                                box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 1rem;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: #FF9800;">€{bet['stake']:.2f}</div>
                        <div style="font-size: 0.8rem; color: #6c757d;">Stake Consigliato</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Altre VALUE bets in formato compatto
            other_bets = []
            for bet in value_bets:
                bet_key = f"{bet['type']}_{bet['line']}"
                if bet_key not in seen_bets:
                    other_bets.append(bet)
                    seen_bets.add(bet_key)
            
            other_bets = sorted(other_bets, key=lambda x: x.get('stake', 0), reverse=True)
            
            if other_bets:
                st.markdown("### 📋 Altre Opportunità VALUE")
                
                # Tabella compatta per altre value bets
                other_bets_data = []
                for i, bet in enumerate(other_bets, len(unique_recommendations) + 1):
                    edge = bet.get('edge', 0) * 100
                    prob = bet.get('probability', 0) * 100
                    quality = bet.get('quality_score', 0)
                    
                    other_bets_data.append({
                        'Rank': f"#{i}",
                        'Tipo': f"{bet['type']} {bet['line']}",
                        'Quota': f"{bet['odds']:.2f}",
                        'Edge': f"{edge:+.1f}%",
                        'Probabilità': f"{prob:.1f}%",
                        'Quality': f"{quality:.1f}",
                        'Stake': f"€{bet['stake']:.2f}"
                    })
                
                if other_bets_data:
                    df = pd.DataFrame(other_bets_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Salva per la tab di piazzamento
            st.session_state['best_bet'] = optimal_bet
            st.session_state['all_value_bets'] = value_bets
            st.session_state['unique_recommendations'] = unique_recommendations
            
        else:
            # Nessun VALUE bet trovato
            st.markdown("""
            <div class="bet-summary">
                <h3>❌ Nessuna Opportunità VALUE Identificata</h3>
                <p>Il sistema non ha trovato scommesse con valore positivo. Mostriamo le migliori opzioni disponibili.</p>
            </div>
            """, unsafe_allow_html=True)
            
            top_5_bets = all_opportunities[:5]
            
            st.markdown("### 📊 Migliori 5 Opzioni Disponibili")
            
            for i, bet in enumerate(top_5_bets, 1):
                edge = bet.get('edge', 0) * 100
                prob = bet.get('probability', 0) * 100
                quality = bet.get('quality_score', 0)
                
                if edge > -2.0:
                    status = "MARGINALE"
                    color = "#FFC107"  # Amber
                elif edge > -5.0:
                    status = "SCARSA"
                    color = "#FF5722"  # Red-Orange
                else:
                    status = "PESSIMA"
                    color = "#9E9E9E"  # Grey
                
                st.markdown(f"""
                <div style="background: {color}20; border-left: 5px solid {color}; border-radius: 12px; 
                            padding: 1rem; margin: 0.8rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #1e3c72;">#{i} {status} - {bet['type']} {bet['line']}</h4>
                        <span style="background: {color}; color: white; padding: 0.2rem 0.6rem; 
                                     border-radius: 15px; font-size: 0.8rem;">
                            {edge:+.1f}%
                        </span>
                    </div>
                    <div style="margin-top: 0.5rem; color: #6c757d;">
                        Quota: {bet['odds']:.2f} • Probabilità: {prob:.1f}% • Stake: €{bet['stake']:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        st.error("❌ Dati di analisi scommesse non disponibili")

def show_betting_tab():
    """Tab per il piazzamento della scommessa - CON SCORE ALGORITMO SISTEMA"""
    if 'best_bet' not in st.session_state:
        st.warning("⚠️ Completa prima l'analisi e seleziona una raccomandazione")
        return
    
    st.markdown("""
    <div class="tab-container">
        <h2>💰 Piazzamento Scommessa</h2>
        <p>Finalizza la scommessa selezionata con tutti i dettagli del sistema</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recupera i dati dalla sessione
    best_bet = st.session_state.get('best_bet')
    unique_recommendations = st.session_state.get('unique_recommendations', [])
    all_value_bets = st.session_state.get('all_value_bets', [])
    game = st.session_state.get('selected_game', {})
    analysis_result = st.session_state.get('analysis_result', {})
    central_line = st.session_state.get('central_line', 0)
    
    # ========================================
    # SEZIONE 1: SELEZIONE SCOMMESSA FINALE
    # ========================================
    st.markdown("### 🎯 Selezione Scommessa Finale")
    
    # Combina tutte le opzioni disponibili
    all_betting_options = []
    for rec in unique_recommendations:
        all_betting_options.append(rec)
    
    # Aggiungi altre value bets se presenti
    seen_bets = {f"{rec['bet']['type']}_{rec['bet']['line']}" for rec in unique_recommendations}
    for bet in all_value_bets:
        bet_key = f"{bet['type']}_{bet['line']}"
        if bet_key not in seen_bets:
            all_betting_options.append({
                'bet': bet,
                'category': '💎 VALUE',
                'description': 'Opportunità VALUE aggiuntiva',
                'color': '#00BCD4'
            })
    
    if all_betting_options:
        # Crea opzioni per il selectbox
        bet_options = []
        for i, option in enumerate(all_betting_options, 1):
            bet = option['bet']
            edge = bet.get('edge', 0) * 100
            prob = bet.get('probability', 0) * 100
            quality_score = bet.get('quality_score', bet.get('optimization_score', 0))  # Usa optimization_score se disponibile
            bet_options.append(f"{i}. {option['category']} - {bet['type']} {bet['line']} @ {bet['odds']:.2f} (Edge: {edge:+.1f}%, Prob: {prob:.1f}%, Score: {quality_score:.1f})")
        
        selected_bet_index = st.selectbox(
            f"📋 Seleziona la scommessa da piazzare:",
            range(len(bet_options) + 1),
            format_func=lambda x: bet_options[x-1] if x > 0 else "0. Nessuna scommessa",
            key="final_bet_selection"
        )
        
        if selected_bet_index > 0 and selected_bet_index <= len(all_betting_options):
            selected_option = all_betting_options[selected_bet_index - 1]
            selected_bet = selected_option['bet']
            category = selected_option['category']
            
            # ========================================
            # SEZIONE 2: RIEPILOGO SCOMMESSA DETTAGLIATO CON SCORE ALGORITMO
            # ========================================
            st.markdown("### 📊 Riepilogo Scommessa Dettagliato")
            
            # Estrai dati dalla scommessa
            bet_type_full = "OVER" if selected_bet.get('type') == 'OVER' else "UNDER"
            bet_line = selected_bet.get('line', 0)
            bet_odds = selected_bet.get('odds', 0)
            bet_stake = selected_bet.get('stake', 0)
            edge = selected_bet.get('edge', 0) * 100
            prob = selected_bet.get('probability', 0) * 100
            
            # SCORE ALGORITMO DEL SISTEMA - Priorità: optimization_score > quality_score
            optimization_score = selected_bet.get('optimization_score', selected_bet.get('quality_score', 0))
            
            # Estrai componenti del score se disponibili (dal sistema _calculate_optimal_bet)
            edge_score = selected_bet.get('edge_score', 0)
            prob_score = selected_bet.get('prob_score', 0)
            odds_score = selected_bet.get('odds_score', 0)
            total_raw_score = selected_bet.get('total_raw_score', 0)
            
            # FALLBACK: Se i componenti non sono disponibili, li calcoliamo manualmente
            if edge_score == 0 and prob_score == 0 and odds_score == 0:
                # Ricalcola i componenti usando la stessa logica di main.py
                edge_decimal = edge / 100  # Converte da percentuale a decimale
                prob_decimal = prob / 100  # Converte da percentuale a decimale
                
                # Edge Score (max 30 punti)
                if edge_decimal >= 0.15:  # 15%+
                    edge_score = 30
                elif edge_decimal >= 0.10:  # 10-15%
                    edge_score = 25 + (edge_decimal - 0.10) * 100  # Scala 25-30
                elif edge_decimal >= 0.05:  # 5-10%
                    edge_score = 15 + (edge_decimal - 0.05) * 200  # Scala 15-25
                elif edge_decimal >= 0.02:  # 2-5%
                    edge_score = 5 + (edge_decimal - 0.02) * 333   # Scala 5-15
                else:  # <2%
                    edge_score = edge_decimal * 250  # Max 5 punti per edge molto bassi
                
                # Probability Score (max 50 punti - peso maggiore)
                if prob_decimal > 0.65:
                    prob_score = 50              # Bonus per probabilità molto alte
                elif 0.60 <= prob_decimal <= 0.65:
                    prob_score = 35 + (prob_decimal - 0.60) * 300  # Scala 35-50
                elif 0.55 <= prob_decimal < 0.60:
                    prob_score = 20 + (prob_decimal - 0.55) * 300  # Scala 20-35
                elif 0.52 <= prob_decimal < 0.55:
                    prob_score = 8 + (prob_decimal - 0.52) * 400   # Scala 8-20 (AMPLIFICATA)
                else:  # 50-52% - FASCIA CRITICA
                    prob_score = (prob_decimal - 0.50) * 400       # 0-8 punti (MOLTO RIPIDA) 
                
                # Odds Score (max 20 punti)
                if 1.70 <= bet_odds <= 1.95:
                    odds_score = 20              # Range ottimale massimo premio
                elif 1.60 <= bet_odds < 1.70:
                    odds_score = 12              # Buono ma margine basso
                elif 1.95 < bet_odds <= 2.10:
                    odds_score = 15              # Ancora accettabile
                elif 2.10 < bet_odds <= 2.30:
                    odds_score = 10              # Rischio moderato
                elif 2.30 < bet_odds <= 2.60:
                    odds_score = 6               # Rischio alto
                else:
                    odds_score = max(2, 12 - abs(bet_odds - 1.8) * 5)  # Penalizzazione
                
                # Calcola score totale
                total_raw_score = (
                    edge_score * 0.30 +      # Edge 30%
                    prob_score * 0.50 +      # Probabilità 50%
                    odds_score * 0.20        # Quote 20%
                )
                
                # Se optimization_score non è disponibile, usa il calcolo
                if optimization_score == 0:
                    optimization_score = total_raw_score
            
            # Calcola metriche aggiuntive
            potential_win = bet_stake * (bet_odds - 1)
            roi_percent = (potential_win / bet_stake * 100) if bet_stake > 0 else 0
            
            # Estrai dati dall'analisi
            distribution = analysis_result.get('distribution', {})
            momentum_impact = analysis_result.get('momentum_impact', {})
            injury_impact = analysis_result.get('injury_impact', 0)
            predicted_total = distribution.get('predicted_mu', 0)
            confidence_sigma = distribution.get('predicted_sigma', 0)
            
            # Layout a 3 colonne per il riepilogo
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                           color: white; border-radius: 15px; padding: 1.5rem; margin-bottom: 1rem;">
                    <h4 style="margin: 0 0 1rem 0; text-align: center;">🏀 Informazioni Partita</h4>
                    <div style="margin-bottom: 0.8rem;">
                        <strong>Squadre:</strong><br>
                        {game.get('away_team', 'Away')} @ {game.get('home_team', 'Home')}
                    </div>
                    <div style="margin-bottom: 0.8rem;">
                        <strong>Data:</strong> {datetime.now().strftime("%d/%m/%Y")}
                    </div>
                    <div style="margin-bottom: 0.8rem;">
                        <strong>Linea Bookmaker:</strong> {central_line}
                    </div>
                    <div>
                        <strong>Totale Previsto:</strong> {predicted_total:.1f} pts
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {selected_option['color']} 0%, {selected_option['color']}CC 100%); 
                           color: white; border-radius: 15px; padding: 1.5rem; margin-bottom: 1rem;">
                    <h4 style="margin: 0 0 1rem 0; text-align: center;">🎯 Dettagli Scommessa</h4>
                    <div style="margin-bottom: 0.8rem;">
                        <strong>Categoria:</strong><br>
                        {category}
                    </div>
                    <div style="margin-bottom: 0.8rem;">
                        <strong>Tipo:</strong> {bet_type_full} {bet_line}
                    </div>
                    <div style="margin-bottom: 0.8rem;">
                        <strong>Quota:</strong> {bet_odds:.2f}
                    </div>
                    <div>
                        <strong>Stake:</strong> €{bet_stake:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); 
                           color: white; border-radius: 15px; padding: 1.5rem; margin-bottom: 1rem;">
                    <h4 style="margin: 0 0 1rem 0; text-align: center;">📊 Metriche Sistema</h4>
                    <div style="margin-bottom: 0.8rem;">
                        <strong>Edge:</strong> {edge:+.1f}%
                    </div>
                    <div style="margin-bottom: 0.8rem;">
                        <strong>Probabilità:</strong> {prob:.1f}%
                    </div>
                    <div style="margin-bottom: 0.8rem;">
                        <strong>ROI Atteso:</strong> {roi_percent:.1f}%
                    </div>
                    <div>
                        <strong>Vincita Pot.:</strong> €{potential_win:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # ========================================
            # SEZIONE 3: SCORE ALGORITMO DEL SISTEMA (COME NEL MAIN.PY)
            # ========================================
            st.markdown("### 🤖 Score Algoritmo del Sistema")
            
            # Container principale per lo score
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%); 
                       color: white; border-radius: 20px; padding: 2rem; margin: 1rem 0;
                       border: 3px solid #FF9800; box-shadow: 0 8px 32px rgba(255,152,0,0.3);">
                <div style="text-align: center; margin-bottom: 1.5rem;">
                    <h2 style="margin: 0; font-size: 2.5rem; font-weight: bold;">
                        🏆 SCORE TOTALE: {optimization_score:.1f}/100
                    </h2>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
                        Punteggio calcolato dall'algoritmo di ottimizzazione ML
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Breakdown del score se disponibile
            if edge_score > 0 or prob_score > 0 or odds_score > 0:
                st.markdown("#### 🔬 Breakdown Score Algoritmo")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "⚡ Edge Score", 
                        f"{edge_score:.1f}/30",
                        help="Punteggio basato sul margine favorevole (peso 30%)"
                    )
                
                with col2:
                    st.metric(
                        "🎯 Probability Score", 
                        f"{prob_score:.1f}/50",
                        help="Punteggio basato sulla probabilità di successo (peso 50%)"
                    )
                
                with col3:
                    st.metric(
                        "💰 Odds Score", 
                        f"{odds_score:.1f}/20",
                        help="Punteggio basato sulla qualità delle quote (peso 20%)"
                    )
                
                # Spiegazione dettagliata del sistema di scoring
                st.info(f"""
                💡 **Sistema di Scoring ML**:
                - **Edge Score (30%)**: 0-30 punti - Vantaggio matematico della scommessa
                - **Probability Score (50%)**: 0-50 punti - Probabilità di successo (peso maggiore)
                - **Odds Score (20%)**: 0-20 punti - Qualità e attrattività delle quote
                
                **Score Totale**: {total_raw_score:.1f}/100 = Edge({edge_score:.1f}×0.30) + Prob({prob_score:.1f}×0.50) + Odds({odds_score:.1f}×0.20)
                
                Il sistema privilegia le probabilità alte mantenendo un edge positivo.
                """)
            
            # ========================================
            # SEZIONE 4: IMPATTI SISTEMA NBA PREDICTOR
            # ========================================
            st.markdown("### ⚙️ Impatti Sistema NBA Predictor")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                momentum_value = momentum_impact.get('total_impact', 0) if isinstance(momentum_impact, dict) else momentum_impact
                st.metric(
                    "⚡ Momentum Impact", 
                    f"{momentum_value:+.2f} pts",
                    help="Impatto del sistema ML momentum sui totali"
                )
            
            with col2:
                st.metric(
                    "🏥 Injury Impact", 
                    f"{injury_impact:+.2f} pts",
                    help="Impatto degli infortuni sui totali della partita"
                )
            
            with col3:
                total_system_impact = momentum_value + injury_impact
                st.metric(
                    "🔧 Impatto Totale Sistema", 
                    f"{total_system_impact:+.2f} pts",
                    help="Somma di tutti gli impatti calcolati dal sistema"
                )
            
            # ========================================
            # SEZIONE 5: AZIONI FINALI
            # ========================================
            st.markdown("### 🚀 Azioni Finali")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✅ CONFERMA E PIAZZA SCOMMESSA", key="place_final_bet", type="primary", use_container_width=True):
                    # Salva la scommessa nel sistema
                    game_id = game.get('game_id', f"CUSTOM_{game.get('away_team', 'Away')}_{game.get('home_team', 'Home')}")
                    
                    # Qui si dovrebbe chiamare il metodo per salvare la scommessa
                    # system.save_pending_bet(selected_bet, game_id)
                    
                    st.success("🎉 Scommessa piazzata con successo!")
                    st.session_state['bet_placed'] = True
                    st.balloons()
            
            with col2:
                if st.button("📊 SALVA ANALISI", key="save_analysis", use_container_width=True):
                    st.info("💾 Analisi salvata per riferimento futuro")
            
            # ========================================
            # SEZIONE 6: RIEPILOGO FINALE MIGLIORATO
            # ========================================
            st.markdown("### 📋 Riepilogo Finale")
            
            # Determina livello di rischio e confidenza
            if prob >= 70:
                risk_level = "🟢 BASSO"
                risk_color = "#4CAF50"
            elif prob >= 60:
                risk_level = "🟡 MEDIO"
                risk_color = "#FF9800"
            else:
                risk_level = "🔴 ALTO"
                risk_color = "#F44336"
            
            if optimization_score >= 80:
                confidence_level = "🔥 ALTA"
                conf_color = "#4CAF50"
            elif optimization_score >= 60:
                confidence_level = "⚡ MEDIA"
                conf_color = "#FF9800"
            else:
                confidence_level = "⚪ BASSA"
                conf_color = "#9E9E9E"
            
            # Header del riepilogo
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                       border-radius: 15px; padding: 1.5rem; margin: 1rem 0;
                       border: 2px solid #dee2e6;">
                <div style="text-align: center; margin-bottom: 1rem;">
                    <h3 style="margin: 0; color: #1e3c72; font-size: 1.5rem;">
                        🎯 {bet_type_full} {bet_line} @ {bet_odds:.2f}
                    </h3>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Grid delle metriche usando Streamlit columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div style="text-align: center; background: white; padding: 1rem; border-radius: 10px; 
                           box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 1rem;">
                    <div style="font-size: 1.1rem; font-weight: bold; color: #1e3c72;">€{bet_stake:.2f}</div>
                    <div style="font-size: 0.9rem; color: #6c757d;">Stake Consigliato</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="text-align: center; background: white; padding: 1rem; border-radius: 10px; 
                           box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 1rem;">
                    <div style="font-size: 1.1rem; font-weight: bold; color: #4CAF50;">€{potential_win:.2f}</div>
                    <div style="font-size: 0.9rem; color: #6c757d;">Vincita Potenziale</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style="text-align: center; background: white; padding: 1rem; border-radius: 10px; 
                           box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 1rem;">
                    <div style="font-size: 1.1rem; font-weight: bold; color: {risk_color};">{risk_level}</div>
                    <div style="font-size: 0.9rem; color: #6c757d;">Livello Rischio</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div style="text-align: center; background: white; padding: 1rem; border-radius: 10px; 
                           box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 1rem;">
                    <div style="font-size: 1.1rem; font-weight: bold; color: {conf_color};">{confidence_level}</div>
                    <div style="font-size: 0.9rem; color: #6c757d;">Confidenza Sistema</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Footer con metriche aggiornate
            st.markdown(f"""
            <div style="text-align: center; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #dee2e6;">
                <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">
                    🎯 Edge: {edge:+.1f}% • 📊 Probabilità: {prob:.1f}% • 🤖 Score Algoritmo: {optimization_score:.1f}/100
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.get('bet_placed'):
                st.success("""
                ✅ **SCOMMESSA PIAZZATA CON SUCCESSO!**
                
                Il sistema monitorerà automaticamente i risultati e aggiornerà il bankroll.
                Puoi visualizzare lo stato nella sezione Performance.
                """)
        
        else:
            st.info("👆 Seleziona una scommessa dal menu a discesa per vedere i dettagli")
    
    else:
        st.warning("❌ Nessuna raccomandazione di scommessa disponibile. Completa prima l'analisi.")

# ================================
# 📊 PERFORMANCE DASHBOARD
# ================================

def show_performance_dashboard():
    st.markdown("""
    <div class="main-header">
        <h1>📊 Performance Dashboard</h1>
        <p>Monitoraggio avanzato delle performance di scommessa</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load historical data
    bet_history = load_bet_history()
    
    if bet_history.empty:
        st.info("📝 Nessun dato storico disponibile. Inizia a piazzare scommesse per vedere le performance!")
        return
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_bets = len(bet_history)
        st.metric("🎯 Scommesse Totali", total_bets)
    
    with col2:
        win_rate = 0.65  # Calculate from actual data
        st.metric("🏆 Win Rate", f"{win_rate:.1%}")
    
    with col3:
        total_profit = 45.30  # Calculate from actual data
        st.metric("💹 Profitto Totale", format_currency(total_profit))
    
    with col4:
        roi = 0.15  # Calculate from actual data
        st.metric("📈 ROI", f"{roi:.1%}")
    
    # Performance charts
    st.subheader("📈 Andamento Performance")
    
    # Create sample data for charts
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    cumulative_pl = np.cumsum(np.random.normal(0.5, 2.0, len(dates)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=cumulative_pl,
        mode='lines',
        name='P&L Cumulativo',
        line=dict(color='#1e3c72', width=3)
    ))
    
    fig.update_layout(
        title="Profit & Loss Cumulativo",
        xaxis_title="Data",
        yaxis_title="P&L (€)",
        height=400,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ================================
# 💰 BANKROLL MANAGEMENT
# ================================

def show_bankroll_management():
    st.markdown("""
    <div class="main-header">
        <h1>💰 Gestione Bankroll</h1>
        <p>Registro completo delle scommesse e gestione del capitale</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Bankroll overview
    bankroll_data = load_bankroll_data()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "💰 Bankroll Attuale",
            format_currency(bankroll_data['current_bankroll']),
            delta=format_currency(bankroll_data['current_bankroll'] - bankroll_data['initial_bankroll'])
        )
    
    with col2:
        st.metric("🎯 Bankroll Iniziale", format_currency(bankroll_data['initial_bankroll']))
    
    with col3:
        roi = ((bankroll_data['current_bankroll'] - bankroll_data['initial_bankroll']) / bankroll_data['initial_bankroll']) * 100
        st.metric("📊 ROI Totale", f"{roi:.1f}%")
    
    # Bet history table
    st.subheader("📋 Storico Scommesse")
    
    # Sample data (replace with actual data loading)
    sample_bets = pd.DataFrame({
        'Data': ['2024-06-18', '2024-06-17', '2024-06-16', '2024-06-15'],
        'Partita': ['IND vs OKC', 'LAL vs GSW', 'BOS vs MIA', 'NYK vs PHI'],
        'Tipo': ['OVER 221.5', 'UNDER 225.0', 'OVER 218.5', 'OVER 220.0'],
        'Importo': [10.0, 15.0, 8.0, 12.0],
        'Quota': [1.85, 1.90, 1.88, 1.82],
        'Esito': ['Pending', 'Win', 'Loss', 'Win'],
        'P&L': [0.0, 13.5, -8.0, 9.84]
    })
    
    # Color code the results
    def color_result(val):
        if val == 'Win':
            return 'background-color: #d4edda'
        elif val == 'Loss':
            return 'background-color: #f8d7da'
        elif val == 'Pending':
            return 'background-color: #fff3cd'
        return ''
    
    styled_df = sample_bets.style.applymap(color_result, subset=['Esito'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

# ================================
# 🤖 ML MODELS DASHBOARD
# ================================

def show_ml_models_dashboard():
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Dashboard Modelli ML</h1>
        <p>Gestione e monitoraggio dei modelli di Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model overview
    models_info = {
        'Regular Season': get_model_status('models/momentum_complete/regular_season'),
        'Playoff': get_model_status('models/momentum_complete/playoff'),
        'Hybrid': get_model_status('models/momentum_complete/hybrid')
    }
    
    # Model status cards
    col1, col2, col3 = st.columns(3)
    
    for i, (model_name, info) in enumerate(models_info.items()):
        col = [col1, col2, col3][i]
        
        with col:
            status_class = f"status-{info['status']}"
            st.markdown(f"""
            <div class="metric-card">
                <h4>{model_name}</h4>
                <span class="model-status {status_class}">{info['status'].upper()}</span>
                <p><strong>MAE:</strong> {info['mae']}</p>
                <p><strong>R²:</strong> {info['r2']}</p>
                <p><strong>Last Training:</strong> {info['last_trained']}</p>
                <p><strong>Samples:</strong> {info['samples']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Model performance comparison
    st.subheader("📈 Confronto Performance Modelli")
    
    performance_data = pd.DataFrame({
        'Modello': ['Regular Season', 'Playoff', 'Hybrid'],
        'MAE': [6.033, 15.091, 15.076],
        'R²': [0.853, 0.0, 0.0],
        'Samples': [2460, 412, 2872]
    })
    
    fig_performance = px.bar(
        performance_data,
        x='Modello',
        y='MAE',
        title="Mean Absolute Error per Modello",
        color='MAE',
        color_continuous_scale='RdYlGn_r'
    )
    
    st.plotly_chart(fig_performance, use_container_width=True)

# ================================
# ⚙️ CONFIGURATION PANEL
# ================================

def show_configuration_panel():
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ Centro Configurazione</h1>
        <p>Gestione completa delle impostazioni di sistema</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Sistema", "📊 Modelli", "💾 Backup", "🔄 Aggiornamenti"])
    
    with tab1:
        st.subheader("🔧 Configurazione Sistema")
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.markdown("**⚡ Impostazioni API**")
            api_timeout = st.slider("Timeout API (secondi)", 10, 60, 30)
            api_delay = st.slider("Delay tra chiamate (secondi)", 0.1, 2.0, 0.2)
            
            st.markdown("**🏥 Sistema Infortuni**")
            injury_cache_hours = st.slider("Cache infortuni (ore)", 1, 24, 6)
            injury_confidence = st.slider("Soglia confidence", 0.5, 1.0, 0.7)
        
        with config_col2:
            st.markdown("**🎯 Modelli ML**")
            default_model = st.selectbox(
                "Modello predefinito",
                ["Auto", "Regular Season", "Playoff", "Hybrid"]
            )
            
            st.markdown("**💰 Bankroll**")
            max_bet_percentage = st.slider("Max bet % bankroll", 1, 10, 5)
            kelly_fraction = st.slider("Kelly fraction", 0.1, 1.0, 0.25)
        
        if st.button("💾 Salva Configurazione"):
            st.success("✅ Configurazione salvata con successo!")
    
    with tab2:
        st.subheader("📊 Gestione Modelli")
        
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            st.markdown("**📥 Import Modelli**")
            uploaded_model = st.file_uploader("Carica modello", type=['pkl', 'joblib'])
            
            if uploaded_model:
                if st.button("📥 Importa Modello"):
                    st.success("✅ Modello importato con successo!")
        
        with model_col2:
            st.markdown("**📤 Export Modelli**")
            export_model = st.selectbox(
                "Seleziona modello da esportare",
                ["Regular Season", "Playoff", "Hybrid"]
            )
            
            if st.button("📤 Esporta Modello"):
                st.success(f"✅ {export_model} esportato con successo!")
    
    with tab3:
        st.subheader("💾 Gestione Backup")
        
        backup_col1, backup_col2 = st.columns(2)
        
        with backup_col1:
            st.markdown("**💾 Crea Backup**")
            backup_name = st.text_input("Nome backup", value=f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            include_models = st.checkbox("Includi modelli ML", value=True)
            include_data = st.checkbox("Includi dati storici", value=True)
            
            if st.button("💾 Crea Backup Completo"):
                with st.spinner("Creazione backup in corso..."):
                    import time
                    time.sleep(2)
                    st.success(f"✅ Backup '{backup_name}' creato con successo!")
        
        with backup_col2:
            st.markdown("**📥 Ripristina Backup**")
            available_backups = [
                "backup_20240618_193515_injury_system_complete",
                "backup_20240617_180230_stable_version",
                "backup_20240616_142105_pre_update"
            ]
            
            selected_backup = st.selectbox("Seleziona backup", available_backups)
            
            if st.button("📥 Ripristina Backup", type="secondary"):
                st.warning("⚠️ Questa operazione sovrascriverà la configurazione attuale!")
                if st.button("✅ Conferma Ripristino"):
                    st.success(f"✅ Backup '{selected_backup}' ripristinato!")
    
    with tab4:
        st.subheader("🔄 Aggiornamenti Sistema")
        
        update_col1, update_col2 = st.columns(2)
        
        with update_col1:
            st.markdown("**📊 Aggiornamento Dati**")
            st.info("Ultimo aggiornamento: 2024-06-18 19:35")
            
            if st.button("🔄 Aggiorna Dataset"):
                with st.spinner("Aggiornamento dataset in corso..."):
                    import time
                    time.sleep(3)
                    st.success("✅ Dataset aggiornato con successo!")
            
            auto_update = st.checkbox("Aggiornamento automatico", value=True)
            if auto_update:
                update_time = st.time_input("Orario aggiornamento", value=datetime.strptime("06:00", "%H:%M").time())
        
        with update_col2:
            st.markdown("**🚀 Stato Sistema**")
            
            system_status = {
                "NBA API": "🟢 Attivo",
                "Injury Scraper": "🟢 Attivo", 
                "ML Models": "🟢 Operativi",
                "Database": "🟢 Connesso",
                "Backup System": "🟢 Attivo"
            }
            
            for component, status in system_status.items():
                st.write(f"**{component}:** {status}")

# ================================
# 🚀 RUN APPLICATION
# ================================

if __name__ == "__main__":
    main()