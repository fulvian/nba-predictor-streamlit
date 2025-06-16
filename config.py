import os
from dotenv import load_dotenv

# --- API Constants ---
NBA_API_REQUEST_DELAY = 0.9
APISPORTS_REQUEST_DELAY = 1.5
ODDS_API_REQUEST_DELAY = 2.1

# --- Global Path Definitions ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_BASE_DIR = os.path.join(BASE_DIR, 'models')
SETTINGS_FILE = os.path.join(DATA_DIR, 'system_settings.json')

# --- API Keys ---
load_dotenv(os.path.join(BASE_DIR, '.env'))
ODDS_API_KEY = os.getenv('ODDS_API_KEY')
APISPORTS_API_KEY = os.getenv('APISPORTS_API_KEY')

# --- ML Library Availability ---
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# --- Create data directory if it doesn't exist ---
os.makedirs(DATA_DIR, exist_ok=True)

print("✅ Configuration loaded.")
if not XGBOOST_AVAILABLE:
    print("⚠️ XGBoost not installed. Some features may be unavailable.")
if not LIGHTGBM_AVAILABLE:
    print("⚠️ LightGBM not installed. Some features may be unavailable.")

