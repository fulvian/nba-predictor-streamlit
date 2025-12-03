import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_imports():
    """Check if critical modules can be imported."""
    logger.info("Checking critical imports...")
    modules = ["pandas", "numpy", "streamlit", "sklearn", "joblib", "nba_api"]

    failed = []
    for module in modules:
        try:
            __import__(module)
            logger.info(f"✅ Imported {module}")
        except ImportError as e:
            logger.error(f"❌ Failed to import {module}: {e}")
            failed.append(module)

    return len(failed) == 0


def check_project_structure():
    """Check if key project directories and files exist."""
    logger.info("Checking project structure...")
    base_path = Path(os.getcwd())

    paths = [
        "src/nba_predictor",
        "src/nba_predictor/core",
        "src/nba_predictor/streamlit",
        "data",
        "models",
    ]

    failed = []
    for p in paths:
        full_path = base_path / p
        if full_path.exists():
            logger.info(f"✅ Found {p}")
        else:
            logger.error(f"❌ Missing {p}")
            failed.append(p)

    return len(failed) == 0


def check_database_connection():
    """Check if the database manager can be initialized."""
    logger.info("Checking database connection...")
    try:
        # Add src to path for imports
        sys.path.append(str(Path(os.getcwd()) / "src"))
        from nba_predictor.utils.betting_database_manager import (
            get_secure_database_manager,
        )

        db_manager = get_secure_database_manager()
        logger.info("✅ Database manager initialized")
        return True
    except Exception as e:
        logger.error(f"❌ Database check failed: {e}")
        return False


def main():
    logger.info("🚀 Starting System Health Check")

    checks = {
        "Imports": check_imports(),
        "Structure": check_project_structure(),
        "Database": check_database_connection(),
    }

    logger.info("-" * 30)
    logger.info("Health Check Summary:")
    all_passed = True
    for name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("🎉 System appears healthy!")
        sys.exit(0)
    else:
        logger.error("⚠️ System has issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
