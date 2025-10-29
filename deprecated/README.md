# 🚫 DEPRECATED SCRIPTS - SYSTEMS NO LONGER IN USE

## ⚠️ **IMPORTANT WARNING**

All scripts in this directory are **DEPRECATED** and should **NOT** be used.

## ✅ **OFFICIAL SYSTEM (USE THIS INSTEAD)**

The **ONLY** official and supported system is:

### 🏀 **NBA Data Provider with BallDontLie API**
- **Location**: `src/nba_predictor/api/data_provider.py`
- **Primary API**: BallDontLie API (Official NBA Schedule)
- **Fallback API**: The Odds API
- **Dashboard**: `run_betting_workflow.py`
- **Features**:
  - ✅ Intelligent caching (Data Store → API → Cache)
  - ✅ Automatic persistent storage
  - ✅ Rate limiting protection
  - ✅ Real-time NBA schedule data
  - ✅ 10 games found for today (including Houston Rockets @ Toronto Raptors)

### 🚀 **How to Use the Official System**

```python
from src.nba_predictor.api.data_provider import NBADataProvider

# Create provider
provider = NBADataProvider()

# Get today's games (automatically checks cache, then API, then stores)
games = provider.get_scheduled_games(days_ahead=7, specific_date='2025-10-29')

print(f"Found {len(games)} games")
for game in games:
    print(f"{game['away_team']} @ {game['home_team']}")
```

## 🗑️ **Why These Scripts Were Deprecated**

1. **Multiple conflicting APIs** - Created confusion and data inconsistency
2. **No intelligent caching** - Wasted API calls and rate limits
3. **No persistent storage** - Data lost between sessions
4. **Duplicate functionality** - Same code in multiple places
5. **No unified workflow** - Different interfaces for same functionality

## 📁 **What's in This Directory**

### `/download_scripts/`
All manual download and prediction scripts that have been replaced by the official system:
- `main_*.py` - Various main prediction scripts
- `test_*.py` - Test scripts for pipelines
- `angel_prediction_system.py` - Old prediction system
- `real_angel_prediction.py` - Another old prediction system
- `true_angel_complete.py` - Complete old prediction system
- `debug_enhanced_features.py` - Debug script for old features
- `download_*.py` - Manual download scripts

### `/dashboards/`
Old dashboard implementations that have been replaced:
- `app_complete.py` - Old complete dashboard
- `run_dashboard.py` - Old dashboard runner
- `streamlit_app.py` - Old Streamlit app

### Various Data Providers
All old data provider implementations that have been unified into the official `NBADataProvider`.

## 🎯 **Official Workflow**

1. **Dashboard**: `run_betting_workflow.py` (Port 8504)
2. **Data Provider**: `NBADataProvider` in `src/nba_predictor/api/data_provider.py`
3. **Primary API**: BallDontLie API with NBA official schedule
4. **Caching**: Intelligent cache with persistent storage
5. **Fallback**: The Odds API for betting odds

## 📞 **Need Help?**

If you need to use NBA data, always use:
```python
from src.nba_predictor.api.data_provider import NBADataProvider
```

This is the **ONLY** supported method for getting NBA game data in this project.

---

**Last Updated**: 2025-10-29
**Status**: 🚫 FULLY DEPRECATED - USE NBADataProvider INSTEAD