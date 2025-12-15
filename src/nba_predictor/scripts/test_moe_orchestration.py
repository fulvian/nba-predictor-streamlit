import os
import sys
import json
import logging
import time
import shutil
from pathlib import Path
from dataclasses import asdict

# Add project root
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.nba_predictor.core.unified_hybrid_pipeline import UnifiedHybridPipeline
from src.nba_predictor.intelligence.nanogpt_client import NanoGPTClient
from src.nba_predictor.intelligence.news_aggregator import CompositeNewsAggregator

PYTHON_EXEC = sys.executable

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MoE_Verifier")

# --- MOCK SERVER SCRIPT ---
MOCK_SERVER_PY = """
import sys
import json
import time
import os

def log(msg):
    with open("mock_debug.log", "a") as f:
        f.write(msg + "\\n")

def main():
    log("--- Mock Server Started ---")
    # Read mode from env to simulate timeout
    mode = os.environ.get("MOCK_MODE", "normal")
    log(f"Mode: {mode}")
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                log("Stdin closed, exiting.")
                break
            
            log(f"Received: {line.strip()}")
            msg = json.loads(line)
            
            if "method" in msg:
                if msg["method"] == "initialize":
                    output = {
                        "jsonrpc": "2.0",
                        "id": msg["id"],
                        "result": {
                            "serverInfo": {"name": "mock-server", "version": "1.0"},
                            "capabilities": {"tools": {}}
                        }
                    }
                    print(json.dumps(output))
                    sys.stdout.flush()
                    log(f"Sent Init Response: {json.dumps(output)}")
                
                elif msg["method"] == "notifications/initialized":
                    log("Handshake complete.")
                    
                elif msg["method"] == "tools/call": # Note: official MCP uses 'tools/call'
                    log("Received tool call. Processing...")
                    # Simulate processing
                    if mode == "timeout":
                        time.sleep(20) # Longer than client timeout
                    else:
                        time.sleep(1)
                        
                    # Mock Analysis Result
                    analysis = {
                        "consensus_score": 88,
                        "reasoning": "The mock expert indicates strong alignment with the quantitative model due to recent injury news favoring the under.",
                        "risk_level": "low"
                    }
                    
                    output = {
                        "jsonrpc": "2.0",
                        "id": msg["id"],
                        "result": {
                            "content": [{"type": "text", "text": json.dumps(analysis)}],
                            "isError": False
                        }
                    }
                    print(json.dumps(output))
                    sys.stdout.flush()
                    log(f"Sent Tool Response: {json.dumps(output)}")
                    
        except Exception as e:
            log(f"Error: {e}")
            break

if __name__ == "__main__":
    main()
"""


def setup_mock_environment():
    """Create the mock server file."""
    with open("mock_nanogpt_server.py", "w") as f:
        f.write(MOCK_SERVER_PY)
    logger.info("✅ Created 'mock_nanogpt_server.py'")


def cleanup_mock_environment():
    if os.path.exists("mock_nanogpt_server.py"):
        os.remove("mock_nanogpt_server.py")
        logger.info("🧹 Cleaned up mock server")


def test_pipeline_with_mock(timeout_test=False):
    """Run the pipeline with the mock server."""

    # Override Env Vars for NanoGPTClient
    mock_script = os.path.abspath("mock_nanogpt_server.py")
    os.environ["NANOGPT_CONSENSUS_COMMAND"] = PYTHON_EXEC
    args = ["-u", mock_script]
    os.environ["NANOGPT_CONSENSUS_ARGS"] = json.dumps(args)

    if timeout_test:
        os.environ["MOCK_MODE"] = "timeout"
        logger.info("\n🧪 TEST CASE 2: CONSENSUS TIMEOUT / FALLBACK")
    else:
        os.environ["MOCK_MODE"] = "normal"
        logger.info("\n🧪 TEST CASE 1: NORMAL ORCHESTRATION (SUCCESS)")

    try:
        # Initialize Pipeline
        # We assume data exists. If not, this might fail, but that's part of the "deep test".
        pipeline = UnifiedHybridPipeline(
            data_path="data",
            model_path="models",
            validate_realism=False,  # Disable strict checking for test
        )

        # Override client timeout for faster test if verifying fallback
        if timeout_test:
            pipeline.consensus_client.timeout = 3

        logger.info("⚡️ Pipeline Initialized. Running Prediction...")

        # Run Orchestrated Prediction
        # Using correct full team names
        result = pipeline.predict_unified_with_consensus(
            team1="Los Angeles Lakers",
            team2="Boston Celtics",
            line=225.5,
            home_team="Los Angeles Lakers",
            validate_prediction=False,
        )

        # Verify Results
        output = asdict(result)
        consensus = output.get("consensus_analysis")

        print("\n🔍 --- RESULT INSPECTION ---")
        print(f"Quant Predicted Total: {output['predicted_total']}")
        print(f"Confidence: {output['confidence']}")
        print(f"Consensus Analysis: {json.dumps(consensus, indent=2)}")

        if timeout_test:
            if consensus and consensus.get("fallback") is True:
                logger.info(
                    "✅ SUCCESS: Graceful degradation triggered (Fallback=True)"
                )
            else:
                logger.error("❌ FAILURE: Expected fallback, got success response")
        else:
            if consensus and consensus.get("consensus_score") == 88:
                logger.info("✅ SUCCESS: Consensus score received and merged.")
            else:
                logger.error("❌ FAILURE: Missing or incorrect consensus score")

    except Exception as e:
        logger.error(f"❌ CRITICAL TEST FAILURE: {e}")
        import traceback

        traceback.print_exc()


import subprocess


def test_mock_direct():
    """Verify mock server works with subprocess."""
    # The mock server logs to mock_debug.log, so we don't need to log here.
    logger.info(f"Using python: {PYTHON_EXEC}")
    try:
        proc = subprocess.Popen(
            [PYTHON_EXEC, "-u", "mock_nanogpt_server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "MOCK_MODE": "normal"},
        )

        # Send Init
        init_req = {"jsonrpc": "2.0", "method": "initialize", "id": 1}
        proc.stdin.write(json.dumps(init_req).encode() + b"\n")
        proc.stdin.flush()

        # Read Response
        resp = proc.stdout.readline()
        # logger.info(f"Direct Resp: {resp.decode().strip()}")

        proc.terminate()
        if b"serverInfo" in resp:
            logger.info("✅ Direct Mock Test Passed")
        else:
            logger.error(f"❌ Direct Mock Test Failed: {resp}")

    except Exception as e:
        logger.error(f"❌ Direct Mock Test Error: {e}")


def main():
    # setup_mock_environment()

    try:
        # Pre-check
        test_mock_direct()

        # Test 1: Success Path
        test_pipeline_with_mock(timeout_test=False)

        # Test 2: Timeout/Fallback Path
        # Reset env first just in case
        # time.sleep(1)  # Let processes clean up
        # test_pipeline_with_mock(timeout_test=True)

    finally:
        pass
        # cleanup_mock_environment()


if __name__ == "__main__":
    main()
