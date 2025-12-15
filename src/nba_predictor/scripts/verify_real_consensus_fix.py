from nba_predictor.intelligence.nanogpt_client import NanoGPTClient

import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Set logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NanoGPT_RealVerifier")


def run_real_verification():
    print("🚀 Starting NanoGPT REAL API Verification...")
    print("Testing patched configuration (No :memory suffix, No Flat header).")

    # Clean env
    if "NANOGPT_CONSENSUS_ARGS" in os.environ:
        del os.environ["NANOGPT_CONSENSUS_ARGS"]

    # Create a dummy context
    import uuid

    session_id = f"test_verify_{uuid.uuid4().hex[:8]}"
    print(f"🆔 Using Session ID: {session_id}")

    mcp_env = Path("/Users/fulvioventura/NanoGPT-Consensus-MCP/.env")
    load_dotenv(mcp_env)

    api_key = os.getenv("NANOGPT_API_KEY")
    if api_key:
        print(
            f"🔑 API Key in Verifier: {api_key[:10]}...{api_key[-4:]} (Len: {len(api_key)})"
        )
    else:
        print("❌ API Key is MISSING in Verifier!")

    # Increase timeout
    client = NanoGPTClient(timeout=180)

    context = {
        "team1": "Lakers",
        "team2": "Celtics",
        "predicted_total": 225.5,
        "market_line": 220.5,
        "confidence": "High",
        "stats": {"pace": 98.5},
        "news": [{"type": "injury", "text": "LeBron James is Probable"}],
    }

    try:
        print("\n📩 Sending Query to Consensus (Profile: nba_predictor)...")
        # Ensure asyncio logic works in script
        # nest_asyncio.apply()

        # We call the sync method which handles the loop
        result = client.query_consensus_sync(context, complexity="nba_predictor")

        print("\n📬 Result received:")
        print(json.dumps(result, indent=2))

        if result.get("fallback"):
            print(
                "\n❌ VERIFICATION FAILED: Fallback triggered (likely 402 or timeout)."
            )
        elif "error" in result:
            print("\n❌ VERIFICATION FAILED: Error in result.")
        else:
            print("\n✅ VERIFICATION SUCCESSFUL: Consensus achieved without 402 error!")

    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_real_verification()
