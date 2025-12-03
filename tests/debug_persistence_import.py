import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(os.getcwd())
src_path = project_root / "src"
sys.path.append(str(src_path))

print(f"Attempting to import nba_predictor.core.data_persistence_bridge...")
try:
    import nba_predictor.core.data_persistence_bridge

    print("✅ Import successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback

    traceback.print_exc()
