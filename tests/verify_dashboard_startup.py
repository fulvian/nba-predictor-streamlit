import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def verify_dashboard_startup():
    """
    Simulate dashboard startup by importing the main module.
    This verifies that:
    1. All dependencies are installed
    2. PYTHONPATH is correct
    3. Core managers (DB, DataStore, ML Bridge) can initialize
    """
    logger.info("🚀 Starting Dashboard Startup Verification...")

    # Add src to path
    project_root = Path(os.getcwd())
    src_path = project_root / "src"
    sys.path.append(str(src_path))
    logger.info(f"📂 Added {src_path} to PYTHONPATH")

    try:
        logger.info("🔄 Attempting to import dashboard module...")
        # We import the module which triggers the top-level initialization code
        import nba_predictor.streamlit.new_wic_dashboard as dashboard

        logger.info("✅ Dashboard module imported successfully")

        # Verify critical objects were created
        if hasattr(dashboard, "db_manager") and dashboard.db_manager:
            logger.info("✅ Database Manager initialized")
        else:
            logger.error("❌ Database Manager failed to initialize")
            return False

        if hasattr(dashboard, "data_store") and dashboard.data_store:
            logger.info("✅ Data Store initialized")
        else:
            logger.error("❌ Data Store failed to initialize")
            return False

        if hasattr(dashboard, "ml_bridge") and dashboard.ml_bridge:
            logger.info("✅ ML Bridge initialized")
        else:
            logger.error("❌ ML Bridge failed to initialize")
            return False

        return True

    except ImportError as e:
        logger.error(f"❌ Import Error: {e}")
        logger.error("This usually means a missing dependency or incorrect path.")
        return False
    except Exception as e:
        logger.error(f"❌ Runtime Error during initialization: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_dashboard_startup()
    if success:
        logger.info("🎉 Dashboard startup check PASSED! System is ready.")
        sys.exit(0)
    else:
        logger.error("⚠️ Dashboard startup check FAILED.")
        sys.exit(1)
