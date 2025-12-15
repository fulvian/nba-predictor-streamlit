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
logger = logging.getLogger("NanoGPT_FlatVerifier")


def run_header_fuzzing():
    headers_to_try = [
        {},  # Baseline
        {"x-nanogpt-billing-mode": "flat"},
        {"x-nanogpt-billing-mode": "subscription"},
        {"x-nanogpt-billing-mode": "plan"},
        {"x-nanogpt-use-subscription": "true"},
        {"x-billing": "subscription"},
        {"x-billing-mode": "flat"},
    ]

    for h in headers_to_try:
        print(f"\n🧪 Testing Headers: {h}")
        # Loop content just for logging, we run tests in run_direct_api_tests
        pass

    run_direct_api_tests()


def run_direct_api_tests():
    import requests
    import os
    from dotenv import load_dotenv

    mcp_env = Path("/Users/fulvioventura/NanoGPT-Consensus-MCP/.env")
    load_dotenv(mcp_env)
    api_key = os.getenv("NANOGPT_API_KEY")

    url = "https://nano-gpt.com/api/v1/chat/completions"
    base_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "deepseek-r1",  # Known failing model
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant." * 500,
            },  # Huge Prompt (~2000 tokens)
            {
                "role": "user",
                "content": "This is a much longer prompt to simulate the consensus engine size. "
                * 50,
            },
        ],
        "temperature": 0.7,
        "stream": True,
    }

    candidates = [
        {"Accept": "application/json"},
        {"User-Agent": "python-httpx/0.27.0"},
        {"Accept": "application/json", "User-Agent": "python-httpx/0.27.0"},
    ]

    print("\n🚀 Starting Direct API Header Fuzzing (Non-Streaming)...")
    payload["stream"] = False  # Force false

    for h in candidates:
        print(f"👉 Testing: {h}")
        req_headers = {**base_headers, **h}
        try:
            resp = requests.post(
                url, json=payload, headers=req_headers, timeout=30, stream=False
            )
            if resp.status_code == 200:
                print(f"✅ SUCCESS! Working Header: {h}")
                print(f"Response: {resp.text[:200]}...")
                return
            else:
                print(f"❌ Failed ({resp.status_code}): {resp.text[:200]}...")
        except Exception as e:
            print(f"❌ Exception: {e}")


if __name__ == "__main__":
    # run_flat_verification() # Skip the MCP test
    run_direct_api_tests()
