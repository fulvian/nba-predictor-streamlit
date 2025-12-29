import requests
import json
import sys

# User provided SSOID
SSOID = "ViZ3wYJTjQdVBk3X8plwQbxSRPGwUWuWI3vLTs4NW+0="
APP_NAME = "nba_live_bot"

URL = "https://api.betfair.com/exchange/account/json-rpc/v1"

HEADERS = {
    "X-Authentication": SSOID,
    "Content-Type": "application/json",
    "Accept": "application/json",
}


def call_api(method, params={}):
    payload = {
        "jsonrpc": "2.0",
        "method": f"AccountAPING/v1.0/{method}",
        "params": params,
        "id": 1,
    }
    try:
        response = requests.post(URL, headers=HEADERS, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error calling {method}: {e}")
        return None


def main():
    print(f"🔑 Attempting to retrieve keys for SSOID: {SSOID[:5]}...")

    # 1. Try getDeveloperAppKeys
    res = call_api("getDeveloperAppKeys")

    if not res or "error" in res:
        print(f"❌ Error getting keys: {res}")
        # Identify if error is NO_SESSION or INVALID_SESSION
        return

    result = res.get("result", [])

    found_key = None

    if result:
        print(f"✅ Found {len(result)} existing keys.")
        for app in result:
            print(
                f"   - Name: {app['appName']}, Version: {app['appVersions'][0]['version']}, Key: {app['appVersions'][0]['applicationKey']}"
            )
            # Prefer the one named 'nba_live_bot' or take the first one
            if app["appName"] == APP_NAME:
                found_key = app["appVersions"][0]["applicationKey"]

        if not found_key:
            # Just take the first available 'Delay' key if possible, but usually keys are same for live/delay, just usage differs or subscription
            # Just take the first key found
            found_key = result[0]["appVersions"][0]["applicationKey"]
            print(f"👉 Selecting key from '{result[0]['appName']}': {found_key}")

    else:
        print("⚠️ No keys found. Creating new key...")
        # 2. Create Key
        res_create = call_api("createDeveloperAppKeys", {"appName": APP_NAME})
        if res_create and "result" in res_create:
            print("✅ Key Created Successfully!")
            print(json.dumps(res_create["result"], indent=2))
            found_key = res_create["result"]["applicationKey"]
        else:
            print(f"❌ Failed to create key: {res_create}")

    if found_key:
        print(f"\n🎉 SUCCESS! Your Betfair App Key is: {found_key}")
        # Save to a .env file or similar? For now just print.
    else:
        print("\n❌ Could not retrieve or create a key.")


if __name__ == "__main__":
    main()
