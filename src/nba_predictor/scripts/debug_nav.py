import os
import logging
from src.nba_predictor.betfair.client import BetfairClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Credentials
APP_KEY = os.getenv("BETFAIR_APP_KEY", "QkYxxm82m7tiUQrI")
USERNAME = os.getenv("BETFAIR_USERNAME", "fulviold@gmail.com")
PASSWORD = os.getenv("BETFAIR_PASSWORD", "9#!Vq-!45ukvu&6")
CERTS_PATH = os.getenv(
    "BETFAIR_CERTS_PATH", "/Users/fulvioventura/nba-predictor-streamlit/certs"
)


def investigate_nav():
    client = BetfairClient(
        app_key=APP_KEY,
        username=USERNAME,
        password=PASSWORD,
        certs_path=CERTS_PATH,
        locale="italy",
    )
    client.login()

    print("\n--- Investigating Navigation API ---")
    nav = client.client.navigation.list_navigation()

    print(f"Type of nav object: {type(nav)}")

    if isinstance(nav, dict):
        print("Keys:", nav.keys())
        # Print first child if exist
        if "children" in nav:
            print("Children Type:", type(nav["children"]))
            if nav["children"]:
                print("First Child:", nav["children"][0])

    elif hasattr(nav, "__dict__"):
        print("Object Attributes:", nav.__dict__.keys())
    else:
        print("Content:", nav)


if __name__ == "__main__":
    investigate_nav()
