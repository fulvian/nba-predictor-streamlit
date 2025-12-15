import requests
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Robustly find .env in MCP directory
mcp_env_path = Path("/Users/fulvioventura/NanoGPT-Consensus-MCP/.env")
print(f"Loading .env from: {mcp_env_path}")
load_dotenv(dotenv_path=mcp_env_path)


def list_subscription_models():
    api_key = os.getenv("NANOGPT_API_KEY")
    if not api_key:
        print("❌ Error: NANOGPT_API_KEY not found in environment")
        # Fallback for debugging, DO NOT COMMIT REAL KEYS
        # api_key = "..."
        return

    url = "https://nano-gpt.com/api/subscription/v1/models"
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

    print(f"🔍 Querying Subscription Models: {url}")
    try:
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            models = response.json()
            print("\n✅ Subscription Models Found:")
            # It usually returns a list
            if isinstance(models, list):
                print(f"Total Models count: {len(models)}")
                # print(json.dumps(models[:5], indent=2)) # Print first 5 to check structure

                # Check target availability
                model_ids = [m["id"] if isinstance(m, dict) else m for m in models]
                # Print all IDs to a file for review
                with open("subscription_models.txt", "w") as f:
                    f.write("\n".join(sorted(model_ids)))

                targets = [
                    "kimi",
                    "deepseek",
                    "qwen",
                    "moonshot",
                    "hermes",
                    "glm",
                    "llama",
                ]
                found = {t: [] for t in targets}

                for m_id in model_ids:
                    for t in targets:
                        if t.lower() in m_id.lower():
                            found[t].append(m_id)

                print("\n🎯 Target Model Availability (Included in Plan):")
                for t, matches in found.items():
                    print(f"- {t}: {matches}")

            else:
                print(f"⚠️ Response is not a list. Type: {type(models)}")
                if isinstance(models, dict):
                    print(f"Keys: {models.keys()}")
                    if "data" in models:
                        print(f"Found 'data' key with {len(models['data'])} items.")
                        # recursively handle data if it's the list
                        # For now just print it
                        print(json.dumps(models, indent=2))
                else:
                    print(json.dumps(models, indent=2))

        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            print("Trying generic models endpoint as backup...")
            # Backup: check generic models
            url2 = "https://nano-gpt.com/api/v1/models"
            resp2 = requests.get(url2, headers=headers)
            print(f"Generic Models Response: {resp2.status_code}")

    except Exception as e:
        print(f"❌ Exception: {e}")


if __name__ == "__main__":
    list_subscription_models()
