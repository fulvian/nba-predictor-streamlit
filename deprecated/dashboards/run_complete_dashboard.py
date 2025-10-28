#!/usr/bin/env python3
"""
🏀 NBA Complete Betting Dashboard Launcher

This script launches the complete NBA betting dashboard with the proper 3-step workflow:
1. Games Schedule (NBA games retrieval)
2. Game Analysis (Individual ML analysis)
3. Betting Lines (Bookmaker integration and value betting)

Context7-compliant implementation with proper error handling and logging.
"""

import sys
import os
import logging
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    import streamlit as st
    from nba_predictor.streamlit.app_complete import main

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    if __name__ == "__main__":
        logger.info("🚀 Launching NBA Complete Betting Dashboard...")
        logger.info("📋 Workflow: Games Schedule → Game Analysis → Betting Lines")
        logger.info("🎯 Features: ML Predictions, Value Betting, Central Line Analysis")

        # Run the main dashboard
        main()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Dashboard error: {e}")
    sys.exit(1)