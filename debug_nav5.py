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

# Find Basket
basket_node = None
for cat in root_children:
    if isinstance(cat, dict) and cat.get("name", "").lower() in (
        "basket",
        "basketball",
    ):
        basket_node = cat
        break

if basket_node:
    print("=== Basket Found ===")
    bb_children = basket_node.get("children", [])
    print(f"Basket has {len(bb_children)} children:")

    # Print ALL children names
    for i, child in enumerate(bb_children):
        if isinstance(child, dict):
            print(f"  [{i}] {child.get('name')} (type: {child.get('type')})")
else:
    print("Basket NOT FOUND")
