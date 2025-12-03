# 🏀 NBA Predictor Development Startup Guide

## 📋 Overview

This guide provides step-by-step instructions for setting up and running the NBA Predictor system in development mode. All identified issues have been resolved with Context7-compliant solutions.

## ✅ Issues Resolved

### 1. Fixed Circular Import Dependencies
- **Problem**: `sync_dashboard.py` had global `data_store` variable causing circular imports
- **Solution**: Removed global variable and passed `data_store` as parameter
- **Files Modified**: `src/nba_predictor/streamlit/components/sync_dashboard.py`

### 2. Fixed Configuration Issues
- **Problem**: Missing `.env` file and incorrect Streamlit config
- **Solution**: Created proper development configuration files
- **Files Created**: `.env`, updated `.streamlit/config.toml`

### 3. Created Unified Startup System
- **Problem**: Multiple entry points with inconsistent error handling
- **Solution**: Created `startup.py` with comprehensive setup and validation
- **Files Created**: `startup.py`, updated `pyproject.toml`

## 🚀 Quick Start (Recommended)

### Method 1: Using Unified Startup Script (Best)

```bash
# 1. Navigate to project root
cd /path/to/nba-predictor-streamlit

# 2. Create and activate virtual environment
python -m venv .venv_new
source .venv_new/bin/activate  # Linux/Mac
# or
.venv_new\Scripts\activate  # Windows

# 3. Install dependencies
pip install -e .
pip install -r requirements.txt

# 4. Configure environment variables
# Edit .env file with your API keys
nano .env  # or your preferred editor

# 5. Run the application
python startup.py
```

### Method 2: Using pip script (Alternative)

```bash
# After steps 1-3 above
pip install -e .
nba-dev
```

### Method 3: Direct Streamlit (For debugging)

```bash
# After steps 1-3 above
streamlit run src/nba_predictor/streamlit/betting_workflow_dashboard.py
```

## ⚙️ Configuration Setup

### Environment Variables (.env)

Create a `.env` file in your project root:

```bash
# Environment settings
ENV=development
DEBUG=true

# API Keys (replace with your actual keys)
NBA_API_KEY=your_nba_api_key_here
BALLDONTLIE_API_KEY=your_ball_dontlie_api_key_here

# Security (development only)
SECRET_KEY=dev_secret_key_not_for_production_use_only

# Database settings
DATABASE_URL=sqlite:///data/nba_predictor_dev.db

# Cache settings
REDIS_URL=redis://localhost:6379/0
CACHE_ENABLED=true

# Streamlit settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_HEADLESS=false

# Development features
ENABLE_MOCK_DATA=false
ENABLE_API_RATE_LIMITING=true
ENABLE_DEVELOPMENT_TOOLS=true
```

### Streamlit Configuration (.streamlit/config.toml)

The configuration has been updated for development:

```toml
[global]
developmentMode = true

[server]
headless = false
port = 8501
enableCORS = true
enableXsrfProtection = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[client]
showErrorDetails = true
```

## 📦 Dependencies Installation

### Core Dependencies (Required)
```bash
pip install streamlit>=1.28.0
pip install polars>=1.0.0
pip install duckdb>=0.9.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
```

### Optional Dependencies (Enhanced Features)
```bash
pip install nba-api>=1.1.0
pip install plotly>=5.15.0
pip install seaborn>=0.12.0
pip install scikit-learn>=1.3.0
pip install requests>=2.31.0
```

### Development Dependencies
```bash
pip install pytest>=7.4.0
pip install black>=23.7.0
pip install ruff>=0.0.280
pip install mypy>=1.5.0
```

## 🧪 Testing Individual Components

### Test Core Data Store
```python
python -c "
import sys
sys.path.insert(0, 'src')
from nba_predictor.core.data_store import UnifiedDataStore
print('✅ Data Store import successful')
"
```

### Test Sync Engine
```python
python -c "
import sys
sys.path.insert(0, 'src')
from nba_predictor.core.sync_engine import AutomaticSyncEngine
print('✅ Sync Engine import successful')
"
```

### Test Streamlit Components
```python
python -c "
import sys
sys.path.insert(0, 'src')
from nba_predictor.streamlit.components.sync_dashboard import render_sync_dashboard
print('✅ Streamlit Components import successful')
"
```

### Test Main Application
```python
python -c "
import sys
sys.path.insert(0, 'src')
from nba_predictor.streamlit.betting_workflow_dashboard import main
print('✅ Main Application import successful')
"
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'nba_predictor'
# Solution: Ensure src directory is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### 2. Permission Errors
```bash
# Error: Permission denied when creating directories
# Solution: Check directory permissions
chmod 755 data/
chmod 644 .env
```

#### 3. Port Already in Use
```bash
# Error: Port 8501 is already in use
# Solution: Kill existing process or change port
lsof -ti:8501 | xargs kill -9  # Linux/Mac
# or edit .streamlit/config.toml to use different port
```

#### 4. API Key Issues
```bash
# Error: API authentication failed
# Solution: Verify API keys in .env file
# Test API connectivity:
python -c "
import os
from nba_predictor.api.data_provider import NBADataProvider
print('API Key configured:', bool(os.getenv('NBA_API_KEY')))
"
```

#### 5. Database Connection Issues
```bash
# Error: Database connection failed
# Solution: Check database directory permissions
mkdir -p data/persistent
chmod 755 data/persistent
```

## 📊 Project Structure

```
nba-predictor-streamlit/
├── .env                          # Environment variables (created)
├── .streamlit/
│   └── config.toml             # Streamlit config (updated)
├── src/
│   └── nba_predictor/
│       ├── __init__.py
│       ├── core/
│       │   ├── data_store.py
│       │   └── sync_engine.py
│       ├── streamlit/
│       │   ├── __init__.py
│       │   ├── app.py
│       │   ├── betting_workflow_dashboard.py
│       │   └── components/
│       │       ├── __init__.py
│       │       └── sync_dashboard.py    # Fixed circular import
│       └── utils/
├── startup.py                     # Unified startup script (created)
├── run_betting_workflow.py        # Legacy launcher (still works)
├── pyproject.toml                # Updated with nba-dev script
├── requirements.txt
└── DEVELOPMENT_STARTUP_GUIDE.md   # This guide
```

## 🚀 Advanced Usage

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration    # Integration tests only
```

### Code Quality Checks
```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

### Development Server Options
```bash
# Custom port
STREAMLIT_SERVER_PORT=8502 python startup.py

# Debug mode
DEBUG=true python startup.py

# Headless mode
STREAMLIT_HEADLESS=true python startup.py
```

## 📝 Development Best Practices

### 1. Environment Management
- Always use virtual environments
- Keep `.env` files out of version control
- Use different keys for development/production

### 2. Code Quality
- Run `black` and `ruff` before commits
- Write type hints for all functions
- Add docstrings for public functions

### 3. Testing
- Test individual components before integration
- Use mock data for unit tests
- Test with real APIs for integration tests

### 4. Performance
- Enable caching in development
- Monitor memory usage
- Profile slow operations

## 🆘 Getting Help

### Check System Status
```bash
# Check if all components are working
python startup.py --check-only
```

### Enable Debug Logging
```bash
# Set debug level logging
DEBUG=true python startup.py
```

### Common Log Locations
- Application logs: `logs/startup.log`
- Streamlit logs: Console output
- Error logs: Console and log files

## 🎯 Next Steps

After successful startup:

1. **Explore the Dashboard**: Open http://localhost:8501 in your browser
2. **Test Data Sync**: Use the sync controls in the dashboard
3. **Check Predictions**: Verify ML predictions are working
4. **Monitor Performance**: Use the built-in performance monitoring
5. **Review Logs**: Check `logs/startup.log` for any issues

---

## 📞 Support

If you encounter issues not covered in this guide:

1. Check the logs in `logs/startup.log`
2. Verify all environment variables are set correctly
3. Ensure all dependencies are installed
4. Test individual components as shown above

**Remember**: The unified `startup.py` script handles most setup issues automatically and provides detailed error reporting.