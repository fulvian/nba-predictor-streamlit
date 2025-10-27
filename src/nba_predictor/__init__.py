"""NBA Predictor package initialization.

This package provides advanced NBA predictive analytics capabilities
including data synchronization, real-time dashboards, and betting odds analysis.

The package follows modern Python src/ layout structure with modular components
for data management, Streamlit UI components, and API integrations.

Example:
    >>> from nba_predictor import UnifiedDataStore, AutomaticSyncEngine
    >>> store = UnifiedDataStore("/data/path")
    >>> engine = AutomaticSyncEngine(store)
    >>> result = await engine.sync_all_data()
    >>> print(f"Synced {result['games_count']} games")

Package Structure:
    core: Core data management and synchronization components
    streamlit: UI components and dashboards
    api: Modern API client integrations
    utils: Utility functions and helpers
"""

__version__ = "2.0.0"
__author__ = "NBA Predictor Team"
__email__ = "team@example.com"
__license__ = "MIT"
__description__ = "Advanced NBA predictive analytics system with real-time data integration"

# Import core components
from .core.data_store import UnifiedDataStore
from .core.sync_engine import AutomaticSyncEngine
# Import streamlit components
from .streamlit import create_main_app, render_sync_dashboard, render_analytics_dashboard
__all__ = [
    "UnifiedDataStore",
    "AutomaticSyncEngine",
    "create_main_app",
    "render_sync_dashboard",
    "render_analytics_dashboard",
]  # Will be populated as components are implemented