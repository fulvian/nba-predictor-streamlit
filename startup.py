#!/usr/bin/env python3
"""
🏀 NBA Predictor Development Startup Script
Context7-compliant unified launcher for development environment.
This script handles all setup, path configuration, and error handling.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional

# Configure project root and paths
PROJECT_ROOT = Path(__file__).parent.absolute()
SRC_PATH = PROJECT_ROOT / "src"
DATA_PATH = PROJECT_ROOT / "data"

# Add src to Python path
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "startup.log"),
    ],
)

logger = logging.getLogger(__name__)


def setup_environment() -> bool:
    """
    Setup development environment with proper configuration.

    Returns:
        bool: True if setup successful, False otherwise
    """
    try:
        # Create necessary directories
        directories = [
            DATA_PATH,
            PROJECT_ROOT / "logs",
            PROJECT_ROOT / "data" / "persistent",
            PROJECT_ROOT / "data" / "cache",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")

        # Load environment variables
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            logger.info("Loading environment variables from .env file")
            with open(env_file, "r") as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        key, value = line.strip().split("=", 1)
                        os.environ[key.strip()] = value.strip()
            logger.info("Environment variables loaded successfully")
        else:
            logger.warning(".env file not found, using default values")
            # Set minimal defaults
            os.environ.setdefault("ENV", "development")
            os.environ.setdefault("DEBUG", "true")

        # Verify critical environment variables
        required_vars = ["ENV", "DEBUG"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            return False

        logger.info("Environment setup completed successfully")
        return True

    except Exception as e:
        logger.error(f"Environment setup failed: {e}")
        return False


def verify_dependencies() -> bool:
    """
    Verify that all required dependencies are available.

    Returns:
        bool: True if all dependencies available, False otherwise
    """
    try:
        # Core dependencies
        import streamlit as st
        import polars as pl
        import duckdb
        import pandas as pd
        import numpy as np

        logger.info("✅ Core dependencies verified")

        # Optional dependencies with warnings
        optional_deps = {
            "nba_api": False,
            "requests": False,
            "plotly": False,
            "seaborn": False,
            "sklearn": False,
        }

        try:
            import nba_api

            optional_deps["nba_api"] = True
        except ImportError:
            logger.warning("⚠️ nba_api not available - some features may be limited")

        try:
            import requests

            optional_deps["requests"] = True
        except ImportError:
            logger.warning("⚠️ requests not available - API functionality limited")

        try:
            import plotly

            optional_deps["plotly"] = True
        except ImportError:
            logger.warning("⚠️ plotly not available - visualizations limited")

        try:
            import seaborn

            optional_deps["seaborn"] = True
        except ImportError:
            logger.warning("⚠️ seaborn not available - styling limited")

        try:
            import sklearn

            optional_deps["sklearn"] = True
        except ImportError:
            logger.warning("⚠️ sklearn not available - ML features limited")

        # Report optional dependencies status
        available_count = sum(1 for available in optional_deps.values() if available)
        total_count = len(optional_deps)

        logger.info(
            f"📦 Optional dependencies: {available_count}/{total_count} available"
        )

        return True

    except ImportError as e:
        logger.error(f"❌ Critical dependency missing: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Dependency verification failed: {e}")
        return False


def verify_project_structure() -> bool:
    """
    Verify that the project structure is correct.

    Returns:
        bool: True if structure is valid, False otherwise
    """
    try:
        # Check critical files and directories
        required_structure = {
            "src/nba_predictor": SRC_PATH / "nba_predictor",
            "src/nba_predictor/__init__.py": SRC_PATH / "nba_predictor" / "__init__.py",
            "src/nba_predictor/streamlit": SRC_PATH / "nba_predictor" / "streamlit",
            "src/nba_predictor/streamlit/app.py": SRC_PATH
            / "nba_predictor"
            / "streamlit"
            / "app.py",
            "src/nba_predictor/streamlit/betting_workflow_dashboard.py": SRC_PATH
            / "nba_predictor"
            / "streamlit"
            / "betting_workflow_dashboard.py",
            "src/nba_predictor/core": SRC_PATH / "nba_predictor" / "core",
            "src/nba_predictor/utils": SRC_PATH / "nba_predictor" / "utils",
        }

        missing_items = []
        for name, path in required_structure.items():
            if not path.exists():
                missing_items.append(f"{name} -> {path}")

        if missing_items:
            logger.error("❌ Missing critical project structure:")
            for item in missing_items:
                logger.error(f"   {item}")
            return False

        logger.info("✅ Project structure verified successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Project structure verification failed: {e}")
        return False


def launch_application() -> bool:
    """
    Launch the NBA Predictor application.

    Returns:
        bool: True if launch successful, False otherwise
    """
    try:
        logger.info("🚀 Launching NBA Predictor Development Environment")
        logger.info(f"📁 Project root: {PROJECT_ROOT}")

        # Path to the main application file
        script_path = PROJECT_ROOT / "src" / "nba_predictor" / "streamlit" / "app.py"

        if not script_path.exists():
            logger.error(f"❌ Application file not found: {script_path}")
            return False

        logger.info("📋 Starting Betting Workflow Dashboard...")
        logger.info("🎯 Features: Real NBA Data, ML Predictions, Live Odds")
        logger.info("✅ Context7 Best Practices - Development Mode")

        # Build the command
        import subprocess

        # Use sys.executable to ensure we use the same Python environment (venv)
        cmd = [sys.executable, "-m", "streamlit", "run", str(script_path)]

        # Add any custom arguments if needed
        if os.environ.get("STREAMLIT_HEADLESS", "").lower() == "true":
            cmd.extend(["--server.headless", "true"])

        port = os.environ.get("STREAMLIT_SERVER_PORT")
        if port:
            cmd.extend(["--server.port", port])

        address = os.environ.get("STREAMLIT_SERVER_ADDRESS")
        if address:
            cmd.extend(["--server.address", address])

        logger.info(f"Running command: {' '.join(cmd)}")

        # Ensure PYTHONPATH includes src
        env = os.environ.copy()
        python_path = env.get("PYTHONPATH", "")
        if str(SRC_PATH) not in python_path:
            env["PYTHONPATH"] = (
                f"{str(SRC_PATH)}:{python_path}" if python_path else str(SRC_PATH)
            )

        # Run Streamlit as a subprocess
        # This will block until the user stops the server
        subprocess.run(cmd, check=True, env=env)
        return True

    except KeyboardInterrupt:
        logger.info("⏹️ Application stopped by user")
        return True
    except Exception as e:
        logger.error(f"❌ Application launch failed: {e}")
        return False


def main() -> int:
    """
    Main startup function with comprehensive error handling.

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    try:
        print("🏀 NBA Predictor Development Startup")
        print("=" * 50)
        print(f"📁 Project: {PROJECT_ROOT}")
        print(f"🐍 Python: {sys.version.split()[0]}")
        print("=" * 50)

        # Step 1: Setup environment
        logger.info("🔧 Step 1: Setting up environment...")
        if not setup_environment():
            logger.error("❌ Environment setup failed")
            return 1

        # Step 2: Verify dependencies
        logger.info("📦 Step 2: Verifying dependencies...")
        if not verify_dependencies():
            logger.error("❌ Dependency verification failed")
            return 1

        # Step 3: Verify project structure
        logger.info("📂 Step 3: Verifying project structure...")
        if not verify_project_structure():
            logger.error("❌ Project structure verification failed")
            return 1

        # Step 4: Launch application
        logger.info("🚀 Step 4: Launching application...")
        if not launch_application():
            logger.error("❌ Application launch failed")
            return 1

        logger.info("✅ NBA Predictor started successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("⏹️ Startup interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Unexpected error during startup: {e}")
        logger.error(f"💡 This might be a system issue or configuration problem")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
