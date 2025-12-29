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
root_children = nav.get("children", [])

print(f"Root has {len(root_children)} sport categories")

# Find Basketball
basket_node = None
for cat in root_children:
    if isinstance(cat, dict) and cat.get("name") == "Basketball":
        basket_node = cat
        break

if basket_node:
    print("=== Basketball Found ===")
    # Print first level children names (countries/leagues)
    bb_children = basket_node.get("children", [])
    print(f"Basketball has {len(bb_children)} children")
    for child in bb_children[:20]:  # First 20
        if isinstance(child, dict):
            print(f"  - {child.get('name')} (type: {child.get('type')})")
else:
    print("Basketball NOT FOUND")
    print("Available sports:")
    for cat in root_children[:10]:
        if isinstance(cat, dict):
            print(f"  - {cat.get('name')}")
