import logging
import joblib
import sys
from pathlib import Path

# Add src to python path to ensure imports work
sys.path.append(str(Path.cwd() / "src"))

from nba_predictor.core.unified_hybrid_pipeline import UnifiedHybridPipeline

# Configure logging to verify explicit errors vs silent failures
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    print("--- Debugging Pickled Pipeline State ---")
    model_path = Path("models/unified_model_latest.pkl")

    if not model_path.exists():
        print(f"❌ Pickled model not found at {model_path}")
        return

    try:
        print(f"🔄 Loading pipeline from {model_path}...")
        with open(model_path, "rb") as f:
            # The pipeline saves a DICT not the object directly in 'model_package' usually?
            # load_model code says: model_package = joblib.load(f)
            # Then self.trained_model = model_package["model"]
            # Wait, does it pickle the PIPELINE or a DICT?
            # load_model logic: model_package = joblib.load(f). It expects a DICT.
            # But the 'pipeline' variable here is expected to be the OBJECT.
            # If the user saves the *whole pipeline* object, then joblib.load returns the object.
            # If the user saves a dict (as load_model implies), then we CANNOT just call predict on it.
            # We need to instantiate a pipeline and call .load_model().

            # Let's inspect what we get.
            loaded_data = joblib.load(f)

        if isinstance(loaded_data, dict) and "model" in loaded_data:
            print(
                "ℹ️ Loaded data is a DICT (Model Package). Instantiating new pipeline and loading this state."
            )
            # Instantiate fresh pipeline
            pipeline = UnifiedHybridPipeline(data_path=Path("data"))
            # We can't call pipeline.load_model() easily with an open file object, checking if we can pass path.
            # pipeline.load_model(str(model_path))
            # But we want to confirm the state.
            pipeline.trained_model = loaded_data["model"]
            pipeline.feature_columns = loaded_data.get("feature_columns", [])
            pipeline._load_team_mapping()  # Ensure map is loaded (modifies in place)
            if "scaler" in loaded_data:
                pipeline.feature_scaler = loaded_data["scaler"]
            pipeline.is_trained = True
            print("✅ Pipeline instantiated and state restored from dict.")
        elif isinstance(loaded_data, UnifiedHybridPipeline):
            print("ℹ️ Loaded data is a Pipeline OBJECT.")
            pipeline = loaded_data
        else:
            print(f"⚠️ Unknown loaded Type: {type(loaded_data)}")
            return

        print("✅ Pipeline ready for testing.")

        # Verify Key State
        print(f"Pipeline Data Path: {pipeline.data_path}")
        print(f"Team Name Map Size: {len(pipeline.team_name_to_id)}")

        # Check if 'L.A. Lakers' is in the map (it shouldn't be, but maybe it is mapped to something wrong?)
        if "L.A. Lakers" in pipeline.team_name_to_id:
            print(
                f"⚠️ 'L.A. Lakers' FOUND in pickle map -> {pipeline.team_name_to_id['L.A. Lakers']}"
            )
        else:
            print(
                "ℹ️ 'L.A. Lakers' NOT in pickle map (Correct behavior if using normalization)"
            )

        # Test Case 2: Valid Teams (Verify if they yield capped result)
        print("\n--- Test Case 2: Valid Teams Prediction (LAL vs GSW) ---")
        try:
            line = 225.0
            result = pipeline.predict_unified(
                team1="Los Angeles Lakers",
                team2="Golden State Warriors",
                line=line,
                validate_prediction=True,
            )
            print(f"Prediction Output: {result.predicted_total}")

            # Check if capped (Prediction == Line - 25 or Line + 25)
            diff = result.predicted_total - line
            print(f"Diff from Line: {diff}")
            if abs(diff) == 25.0:
                print(
                    "🚨 RESULT APPEARS CAPPED! Feature generation likely failed (zeros)."
                )
            else:
                print("✅ Result appears organic (not hard-capped).")

        except Exception as e:
            print(f"❌ VALID TEAM ERROR: {e}")

        # Test Case 1: Run prediction with known problematic input inputs
        print("\n--- Test Case: Prediction with 'L.A. Lakers' (Pickled Model) ---")
        try:
            # We bypass the cache loading logic of predict_unified and assume it uses self.team_name_to_id
            # However, predict_unified calls _create_unified_prediction_features which IS the fixed method.
            # Does the pickled object have the OLD version of the method code?
            # NO: Python pickles objects, but methods are looked up on the class definition in code.
            # UNLESS: The pipeline uses some internal state that overrides the code logic (unlikely here).

            result = pipeline.predict_unified(
                team1="L.A. Lakers",
                team2="Golden State Warriors",
                line=225.0,
                validate_prediction=True,
            )
            print(f"Prediction Output: {result.predicted_total}")
            print(f"Details: {result}")

        except ValueError as e:
            print(f"✅ CAUGHT EXPECTED ERROR: {e}")
        except Exception as e:
            print(f"❌ UNEXPECTED ERROR: {e}")

    except Exception as e:
        print(f"❌ FATAL ERROR LOADING/RUNNING PICKLE: {e}")


if __name__ == "__main__":
    main()
