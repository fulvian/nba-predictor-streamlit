#!/usr/bin/env python3
"""
🏀 NBA Betting Workflow Dashboard Launcher

Context7-compliant launcher for the modular betting workflow dashboard.
Uses existing components and real data APIs - no mock data!
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
    from nba_predictor.streamlit.betting_workflow_dashboard import main

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    if __name__ == "__main__":
        logger.info("🚀 Launching NBA Betting Workflow Dashboard...")
        logger.info("📋 Workflow: Games Schedule → Game Analysis → Betting Lines")
        logger.info("🎯 Features: Real NBA Data, ML Predictions, Live Odds")
        logger.info("✅ Context7 Best Practices - No Mock Data!")

        # Run the betting workflow dashboard
        main()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Dashboard error: {e}")
    sys.exit(1)