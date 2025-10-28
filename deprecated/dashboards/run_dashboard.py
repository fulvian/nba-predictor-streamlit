#!/usr/bin/env python3
"""
NBA Predictor Dashboard Launcher

Context7-compliant launcher script for the NBA Predictor Streamlit application.
Provides environment detection, dependency checking, and graceful startup.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []

    try:
        import streamlit
    except ImportError:
        missing_deps.append("streamlit")

    try:
        import polars
    except ImportError:
        missing_deps.append("polars")

    try:
        import duckdb
    except ImportError:
        missing_deps.append("duckdb")

    if missing_deps:
        print("❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   • {dep}")
        print("\n📦 Install with:")
        print(f"   pip install {' '.join(missing_deps)}")
        print("   or")
        print("   pip install -r requirements.txt")
        return False

    return True


def check_environment():
    """Check environment configuration."""
    env_vars = {
        "NBA_API_KEY": "Optional: NBA API key for real-time data",
        "ENV": "Optional: Environment (development/staging/production)"
    }

    missing_required = []
    missing_optional = []

    for var, description in env_vars.items():
        if var == "NBA_API_KEY":
            continue  # Optional
        if not os.getenv(var):
            missing_optional.append(f"{var}: {description}")

    if missing_required:
        print("❌ Required environment variables missing:")
        for var in missing_required:
            print(f"   • {var}")
        return False

    if missing_optional:
        print("⚠️  Optional environment variables not set:")
        for var in missing_optional:
            print(f"   • {var}")

    return True


def main():
    """Main launcher function."""
    print("🏀 NBA Predictor Dashboard Launcher")
    print("=" * 40)

    # Check dependencies
    print("📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ Dependencies OK")

    # Check environment
    print("🔧 Checking environment...")
    if not check_environment():
        sys.exit(1)
    print("✅ Environment OK")

    # Check if app.py exists
    app_path = src_path / "nba_predictor" / "streamlit" / "app.py"
    if not app_path.exists():
        print(f"❌ Application file not found: {app_path}")
        sys.exit(1)
    print("✅ Application file found")

    print("\n🚀 Starting NBA Predictor Dashboard...")
    print(f"📁 Project root: {project_root}")
    print(f"🐍 Python path: {sys.path[0]}")

    # Set environment variables
    os.environ.setdefault("ENV", "development")
    os.environ.setdefault("PYTHONPATH", str(src_path))

    # Import and run the app
    try:
        from nba_predictor.streamlit.app import create_main_app
        create_main_app()
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()