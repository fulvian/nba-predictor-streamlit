import asyncio
import logging
import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.nba_predictor.intelligence.nanogpt_client import NanoGPTClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NanoGPT_Verifier")

# Dummy Server Script Content
DUMMY_SERVER_SCRIPT = """
import sys
import json

def process_message(line):
    try:
        msg = json.loads(line)
    except:
        return

    if "method" in msg and msg["method"] == "initialize":
        # Handshake response
        resp = {
            "jsonrpc": "2.0", 
            "id": msg["id"], 
            "result": {
                "protocolVersion": "2024-11-05", 
                "capabilities": {}, 
                "serverInfo": {"name": "dummy-server", "version": "0.1"}
            }
        }
        print(json.dumps(resp))
        sys.stdout.flush()
    
    elif "method" in msg and msg["method"] == "tools/list":
        # Return capability to ask_consensus
        resp = {
            "jsonrpc": "2.0", 
            "id": msg["id"], 
            "result": {
                "tools": [{
                    "name": "ask_consensus",
                    "description": "Mock Consensus",
                    "inputSchema": {}
                }]
            }
        }
        print(json.dumps(resp))
        sys.stdout.flush()

    elif "method" in msg and msg["method"] == "tools/call":
        # Mock consensus response
        # MCP tool call returns { content: [...], isError: false }
        mock_result_json = {
             "consensus_score": 85,
             "reasoning": "The dummy server confirms the Quantitative signal based on overwhelming evidence.",
             "risk_level": "low"
        }
        
        resp = {
            "jsonrpc": "2.0", 
            "id": msg["id"], 
            "result": {
                "content": [
                    {"type": "text", "text": json.dumps(mock_result_json)}
                ],
                "isError": False
            }
        }
        print(json.dumps(resp))
        sys.stdout.flush()

def main():
    for line in sys.stdin:
        process_message(line)

if __name__ == "__main__":
    main()
"""


def run_verification():
    print("🚀 Starting NanoGPT Client Verification...")

    # 1. Create Dummy Server File
    dummy_path = "dummy_nanogpt_server.py"
    with open(dummy_path, "w") as f:
        f.write(DUMMY_SERVER_SCRIPT)

    print(f"✅ Created dummy server at {dummy_path}")

    # 2. Configure Client to use Dummy
    # We override the command/args via env vars for this test process ONLY
    os.environ["NANOGPT_CONSENSUS_COMMAND"] = "python3"
    os.environ["NANOGPT_CONSENSUS_ARGS"] = json.dumps([dummy_path])

    client = NanoGPTClient(timeout=5)

    # 3. Create Dummy Context
    context = {
        "team1": "Lakers",
        "team2": "Celtics",
        "predicted_total": 225.5,
        "confidence": "High",
        "stats": {"pace": 98.5},
        "news": [{"type": "injury", "text": "LeBron James is Probable"}],
    }

    print("\n📩 Sending Query to Client...")
    result = client.query_consensus_sync(context)

    print("\n📬 Result received:")
    print(json.dumps(result, indent=2))

    # 4. cleanup
    if os.path.exists(dummy_path):
        os.remove(dummy_path)
        print(f"\n🧹 Cleaned up {dummy_path}")

    # Validation
    if result.get("consensus_score") == 85:
        print(
            "\n✅ VERIFICATION SUCCESSFUL: Client communicated with server and parsed response."
        )
    else:
        print("\n❌ VERIFICATION FAILED: Unexpected response.")


if __name__ == "__main__":
    run_verification()
