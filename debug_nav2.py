#!/usr/bin/env python3
import betfairlightweight

APP_KEY = "QkYxxm82m7tiUQrI"
USERNAME = "fulviold@gmail.com"
PASSWORD = "9#!Vq-!45ukvu&6"
CERTS_PATH = "/Users/fulvioventura/nba-predictor-streamlit/certs"

client = betfairlightweight.APIClient(
    username=USERNAME,
    password=PASSWORD,
    app_key=APP_KEY,
    certs=CERTS_PATH,
    locale="italy",
)
client.login()

nav = client.navigation.list_navigation()
print(f"nav type: {type(nav)}")
print(f"nav length: {len(nav) if hasattr(nav, '__len__') else 'N/A'}")

# Print raw first few chars
if isinstance(nav, list) and nav:
    print(f"First item type: {type(nav[0])}")
    print(f"First item: {str(nav[0])[:200]}")
elif isinstance(nav, dict):
    print(f"nav keys: {nav.keys()}")
elif hasattr(nav, "__dict__"):
    print(f"nav attrs: {dir(nav)}")
else:
    print(f"nav content: {str(nav)[:500]}")
