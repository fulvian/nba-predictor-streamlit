#!/usr/bin/env python3
"""
🏀 NBA Predictor Development Startup Script
Context7-compliant unified launcher for development environment.
This script handles all setup, path configuration, and error handling.
Updated for PROJECT NEON (Reflex UI).
"""

import sys
import os
import logging
import subprocess
import time
import signal
from pathlib import Path
from typing import Optional, List

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
    """Setup development environment with proper configuration."""
    try:
        # Create necessary directories
        directories = [
            DATA_PATH,
            PROJECT_ROOT / "logs",
            PROJECT_ROOT / "data" / "persistent",
            PROJECT_ROOT / "data" / "cache",
            PROJECT_ROOT / "ui_reflex",
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
    """Verify that all required dependencies are available."""
    try:
        # Core dependencies
        import polars as pl
        import duckdb
        import pandas as pd
        import numpy as np

        logger.info("✅ Core dependencies verified (Polars, DuckDB)")

        # Verify Reflex via CLI instead of import (avoids Py3.9 generator crash)
        try:
            subprocess.run(
                [sys.executable, "-m", "reflex", "--help"],
                check=True,
                capture_output=True,
            )
            logger.info("✅ Reflex verified via CLI")
        except subprocess.CalledProcessError:
            logger.error("❌ Reflex CLI not found or failed")
            return False

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
    """Verify that the project structure is correct."""
    try:
        # Check critical files and directories
        required_structure = {
            "src/nba_predictor": SRC_PATH / "nba_predictor",
            "ui_reflex/rxconfig.py": PROJECT_ROOT / "ui_reflex" / "rxconfig.py",
            "ui_reflex/neon_dashboard/state.py": PROJECT_ROOT
            / "ui_reflex"
            / "neon_dashboard"
            / "state.py",
            "ui_reflex/neon_dashboard/neon_dashboard.py": PROJECT_ROOT
            / "ui_reflex"
            / "neon_dashboard"
            / "neon_dashboard.py",
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


def cleanup_persistent_processes(port: int = 3000) -> None:
    """
    Terminates Reflex processes (port 3000 frontend, 8000 backend).
    """
    try:
        # NOTE: Removed broad 'pkill' commands (e.g. pkill -f node) because they
        # inadvertently kill the IDE/Agent processes.
        # We rely solely on precise port-based cleanup below.

        # 2. Check ports 3000 (Frontend) and 8000 (Backend)
        for p_chk in [3000, 8000]:
            result = subprocess.run(
                ["lsof", "-t", "-i", f":{p_chk}"], capture_output=True, text=True
            )
            pids = result.stdout.strip().split()
            if pids:
                logger.info(f"🧹 Clearing port {p_chk} (PIDs: {pids})...")
                for pid in pids:
                    if pid:
                        try:
                            # Verify it's not our own PID
                            if int(pid) != os.getpid():
                                os.kill(int(pid), signal.SIGTERM)
                        except:
                            pass

    except Exception as e:
        logger.warning(f"⚠️ Failed to cleanup processes: {e}")


def launch_application() -> bool:
    """
    Launch the NBA Predictor application (Reflex Version).
    """
    try:
        cleanup_persistent_processes()

        logger.info("🚀 Launching PROJECT NEON (Reflex UI)")
        logger.info(f"📁 Project root: {PROJECT_ROOT}")

        # Change to ui_reflex directory
        ui_path = PROJECT_ROOT / "ui_reflex"
        if not ui_path.exists():
            logger.error("❌ UI Directory not found")
            return False

        os.chdir(ui_path)

        # Init if needed (silent)
        if not (ui_path / ".web").exists():
            logger.info("⚙️ Initializing Reflex Web Assets (First Run)...")
            subprocess.run([sys.executable, "-m", "reflex", "init"], check=False)

        logger.info("📋 Starting Reflex App...")
        logger.info("✅ Cyberpunk Theme Loaded")

        # Build the command
        cmd = [sys.executable, "-m", "reflex", "run"]

        # Run
        subprocess.run(cmd, check=True)
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
        print("🏀 NBA Predictor - PROJECT NEON - Startup")
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

        logger.info("✅ System started successfully!")
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
