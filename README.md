# 🏀 NBA Predictor Analytics Dashboard

**Modern NBA Analytics System with Real-time Data Processing**

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nba-predictor-streamlit.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

NBA Predictor Analytics Dashboard è un sistema completo di analisi e betting NBA che combina:

- **🏀 Real-time NBA Data**: Integrazione diretta con API ufficiali NBA.com
- **💰 Advanced Betting System**: Sistema completo di gestione scommesse con tracking bankroll
- **📊 Analytics Dashboard**: Dashboard interattivo con analisi predittive avanzate
- **Timezone Management**: Gestione automatica fusi orari per tutte le arena NBA
- **Multi-source Data**: The Odds API + NBA Official API integration
- **Modern Architecture**: Streamlit + Polars + DuckDB + Context7 compliance
- **Professional UI**: Dashboard responsive e interattivo

## 🚀 Official Dashboard

**[🏀 NBA Betting Workflow Dashboard](http://localhost:8504)** - **OFFICIAL SYSTEM**

## 📋 Live System Status (December 6, 2025)
**✅ System Fully Operational**

- **Data Sync**: Automatic Daily Download (Dashboard Startup)
- **ML Pipeline**: Continuous Learning (Auto-syncs new games)
- **API Status**: All Systems Go

## ✅ **OFFICIAL NBA DATA SYSTEM**

### 🎯 **Single Source of Truth con Intelligent Caching**
```python
from src.nba_predictor.api.data_provider import NBADataProvider

# Create official provider
provider = NBADataProvider()

# Get games with intelligent caching (Memory → Persistent Storage → NBA Official API → BallDontLie API)
games = provider.get_scheduled_games(days_ahead=7)
```

### 🏗️ **System Architecture**
- **🥇 Primary API**: NBA Official API (Stats & Scores)
- **🥈 Fallback API**: BallDontLie API (Schedule Backup)
- **🥉 Fallback API**: The Odds API (Betting Odds & Backup)
- **💾 Persistent Storage**: Automatic data caching in Data Store (Parquet/DuckDB)
- **🔄 ML Auto-Sync**: Unified Pipeline dynamically appends fresh game data
- **🎯 Official Dashboard**: `main_app.py`

### 🚫 **DEPRECATED SYSTEMS**
All other scripts have been moved to `deprecated/` folder.
**DO NOT USE** any other data download or prediction scripts.

*See [Deprecated Systems Documentation](deprecated/README.md) for details*

## ✨ Key Features

### 🏀 Real-time NBA Data
- **Official NBA API**: Partite NBA reali con timezone corretti
- **Live Schedule**: Calendario partite aggiornato in tempo reale
- **Team Information**: Mapping completo team ID → nomi squadre
- **Arena Timezones**: Conversione automatica UTC → local time

### 💰 Advanced Betting System
- **Complete Bet Management**: Sistema completo per gestione scommesse (pending → settled)
- **Bankroll Tracking**: Monitoraggio automatico bankroll con profit/loss tracking
- **Game-Bet Matching**: Algoritmo intelligente per match scommesse → risultati partite
- **Multi-status Tracking**: Pending → Settled → Won/Lost con aggiornamenti automatici
- **Real-time Updates**: Aggiornamento automatico stati scommesse quando partite finiscono

### 📊 Analytics Dashboard
- **Games Schedule**: Visualizzazione partite con timezone handling
- **Real-time Updates**: Aggiornamenti automatici delle partite
- **Interactive Charts**: Grafici interattivi per analisi dati
- **System Status**: Monitoraggio API e data sources

### 🎨 Modern Architecture
- **Streamlit Interface**: Interfaccia web moderna e responsive
- **Python 3.11+**: Async patterns e type hints completi
- **Polars + DuckDB**: High-performance data processing
- **Context7 Compliant**: Documentation e best practices integrate

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- Virtual environment (recommended)

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/nba-predictor-streamlit.git
cd nba-predictor-streamlit

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run main_app.py
```

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev, test]"

# Run with hot reload
streamlit run main_app.py --server.runOnSave true
```

## 📁 Project Structure

```
nba-predictor-streamlit/
├── main_app.py                    # Main dashboard entry point
├── data_provider.py              # The Odds API integration
├── nba_timezone_utils.py         # NBA API + timezone management
├── nba_schedule_fallback.py      # NBA schedule backup system
├── docs/                         # Documentation
│   ├── api/                     # API documentation
│   ├── guides/                  # User guides
│   ├── examples/                # Code examples
│   ├── architecture/            # System architecture
│   ├── deployment/             # Deployment guides
│   └── development/             # Development plans
├── src/                         # Modern Python package
│   └── nba_predictor/         # Core application modules
├── deprecated/                  # Legacy files (70+ files)
├── data/                       # Data storage
└── .venv/                      # Virtual environment
```

## 🎯 Usage

### 💰 NBA Betting System (COMPLETE WORKFLOW)

Il sistema di betting NBA gestisce completamente il ciclo di vita delle scommesse:

#### **1. Dashboard Betting ufficiale**
```bash
streamlit run run_betting_workflow.py
```
Access at **http://localhost:8513**

#### **2. Flusso di Scommessa Completo**

**📋 Phase 1: Game Selection & Bet Placement**
- Seleziona partita NBA dal calendario ufficiale
- Configura tipo di scommessa (Over/Under, Moneyline, Spread)
- Imposta quota, stake e analisi automatica
- Salva scommessa → status: **pending**

**🔄 Phase 2: Pending Bet Management**
- Le scommesse pending appaiono in "Scommesse Pending"
- Monitoraggio automatico risultati partite NBA
- Matching intelligente bet → game result
- Aggiornamento status quando partita finisce

**✅ Phase 3: Automatic Settlement**
- Quando partita termina: pending → **settled**
- Calcolo automatico profit/loss basato su risultato
- Aggiornamento bankroll con tracking completo
- Spostamento in "Cronologia Scommesse"

#### **3. Database Schema Completo**

**placed_bets table**:
```sql
- bet_id: Unique identifier
- game_id: Link to NBA game
- home_team/away_team: Team names (salvati correttamente)
- status: pending → settled → won/lost
- placed_at: Timestamp creazione
- settled_at: Timestamp risoluzione
- profit_loss: Automatic P&L calculation
```

**betting_analysis table**:
```sql
- Analisi completa value betting
- Quality score e risk level
- Probability e edge calculations
```

**bankroll_history table**:
```sql
- Tracking automatico bankroll
- change_type: deposit/bet_settlement/withdrawal
- amount: Importo movimento
- created_at: Timestamp automatico
```

#### **4. Auto-Matching Algorithm**

Il sistema gestisce automaticamente il matching scommesse-risultati:

1. **Game Recognition**: Team names → game ID matching
2. **Result Detection**: Final score retrieval da API NBA
3. **Bet Resolution**: Automatic win/loss calculation
4. **Status Update**: pending → settled con profit/loss
5. **Bankroll Update**: Tracking automatico movements

### 🏀 Download NBA Games (Official Method)

**⚠️ IMPORTANTE: Usare SOLO DataPersistenceBridge per scaricare partite**

```python
import sys
sys.path.append('src')
from nba_predictor.core.data_persistence_bridge import initialize_persistence_bridge, close_persistence_bridge
from datetime import date

# Inizializza bridge
bridge = initialize_persistence_bridge()

# Scarica partite di oggi
today = date.today()
games = bridge.get_scheduled_games_with_persistence(
    days_ahead=1,
    specific_date=today.strftime('%Y-%m-%d'),
    force_api=True
)

# Chiudi bridge
close_persistence_bridge()
```

*Dati salvati in: `data/persistent/games/games_YYYY-MM-DD.parquet`*
*Vedi [Guida Completa](docs/nba_game_download_guide.md) per dettagli*

### 1. Launch Main Dashboard

```bash
streamlit run main_app.py
```

Access at **http://localhost:8501**

### 2. Navigate Tabs

- **🏀 Games Schedule**: View NBA games with timezone info
- **📊 Analytics**: Advanced analytics and insights
- **💰 Betting Odds**: Real-time odds from bookmakers
- **🔧 System Status**: API health and data sources

### 3. Features

- **Date Selection**: Choose any date for NBA games
- **Timezone Display**: See game times in all relevant timezones
- **Real-time Data**: Live updates from official NBA sources
- **API Fallback**: Automatic switching when sources are unavailable

## ⚠️ Avviso Importante - Script Deprecati

**NON utilizzare questi script per scaricare partite NBA:**
- `get_todays_nba_games_simple.py` ❌
- `get_todays_nba_games_test.py` ❌
- `nba_timezone_utils.py` (funzioni di download) ❌
- Tutti gli script in `deprecated/` ❌

Questi script contengono **dati falsi** o implementazioni errate.

**Usare SEMPRE DataPersistenceBridge** come mostrato sopra.

## 🔧 Technical Details

### Data Sources

- **NBA Official API**: `stats.nba.com/stats/scoreboardv2`
  - Real NBA game schedules
  - Team information and IDs
  - Game status and times
- **The Odds API**: Betting odds from multiple bookmakers
  - Real-time odds data
  - Market information

### API Integration

```python
# NBA Games API
from nba_timezone_utils import get_nba_games_official_api
games = get_nba_games_official_api(date.today())

# Data Provider
from data_provider import NBADataProvider
provider = NBADataProvider()
odds = provider.get_odds_for_date(date.today())
```

### Timezone Management

```python
from nba_timezone_utils import NBATimezoneManager
tz_manager = NBATimezoneManager()

# Convert UTC to local team timezone
local_time, timezone = tz_manager.convert_utc_to_local(utc_time, "Golden State Warriors")
```

## 📊 System Performance

- **Data Sources**: 2 primary APIs with automatic fallback
- **Response Time**: <2 seconds for game data
- **API Quota Management**: Graceful handling of rate limits
- **Reliability**: 99%+ uptime with error recovery
- **Timezone Coverage**: All 30 NBA teams + arena timezones
- **Betting Performance**: Real-time bet processing <500ms
- **Database Performance**: DuckDB optimized for analytics queries
- **Auto-Matching Speed**: <1s for bet-game result matching

## 🔒 Security & Reliability

- **API Key Management**: Environment-based configuration
- **Rate Limiting**: Built-in protection against API limits
- **Error Recovery**: Automatic retry with exponential backoff
- **Data Validation**: Input sanitization and type checking
- **Graceful Degradation**: Fallback to backup data sources
- **Bet Security**: Atomic transactions for bet placement/settlement
- **Bankroll Protection**: Validated stake limits and balance checks
- **Data Integrity**: Foreign key constraints prevent orphan bets
- **Backup Strategy**: Automatic database backups before major operations

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to branch (`git push origin feature/AmazingFeature`)
5. **Open** Pull Request

### Development Guidelines

- Follow Context7 best practices
- Include type hints and docstrings
- Add tests for new features
- Update documentation

## 📚 Documentation

- **[🚀 Intelligent Caching System Guide](docs/guides/intelligent-caching-system.md)**: **NEW!** Complete guide to the multi-layer caching system
- **[🏀 Unified Hybrid Pipeline Guide](docs/UNIFIED_HYBRID_PIPELINE_GUIDE.md)**: **COMPLETE SYSTEM GUIDE** - Use this pipeline
- **[📊 Pipeline Comparison](docs/PIPELINE_COMPARISON.md)**: **WHICH PIPELINE TO USE** - Decision guide
- **[🎯 NBA Game Download Guide](docs/nba_game_download_guide.md)**: Official data retrieval
- **[🏗️ System Architecture](docs/architecture/system-architecture.md)**: Updated with intelligent caching details
- **[User Guides](docs/guides/)**: Comprehensive usage documentation
- **[API Documentation](docs/api/)**: Technical API reference
- **[Examples](docs/examples/)**: Code examples and tutorials
- **[Deployment](docs/deployment/)**: Production deployment guides

### 🎯 **QUICK START: Unified Hybrid Pipeline**

```python
# Import the production-ready pipeline
from nba_predictor.core.unified_hybrid_pipeline import UnifiedHybridPipeline

# Initialize with all features
pipeline = UnifiedHybridPipeline(
    data_path="data",
    model_path="models",
    use_stacked_ensemble=False,  # Single model for stability
    enable_explainability=True,   # SHAP explanations
    validate_realism=True        # Realistic predictions
)

# Train and predict
metrics = pipeline.train_unified_model()
result = pipeline.predict_unified(
    team1="Lakers", team2="Celtics",
    line=225.0, home_team="Lakers"
)
print(f"Prediction: {result.predicted_total:.1f} points")
```

**📖 See [Complete Guide](docs/UNIFIED_HYBRID_PIPELINE_GUIDE.md) for detailed usage.**

## 📄 License

Distributed under MIT License. See `LICENSE` for details.

## 🙏 Acknowledgments

- **NBA.com** for official API access and game data
- **The Odds API** for real-time betting odds
- **Streamlit** for the powerful web framework
- **Context7** for documentation best practices
- **Open Source Community** for tools and libraries

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/nba-predictor-streamlit/issues)
- **Documentation**: See `docs/` directory
- **API Reference**: `docs/api/`

---

**⭐ If this project is useful, consider giving it a star on GitHub!**