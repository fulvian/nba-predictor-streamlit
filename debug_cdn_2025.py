import requests
import json
from datetime import datetime

url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}

print(f"Fetching {url}...")
try:
    resp = requests.get(url, headers=headers, timeout=10)
    data = resp.json()
    dates = data.get('leagueSchedule', {}).get('gameDates', [])
    
    target = "2025-12-09"
    print(f"Searching for {target}...")
    
    found = False
    for d in dates:
        # d['gameDate'] example: "12/09/2025 00:00:00"
        if target in d['gameDate'] or "12/09/2025" in d['gameDate']:
            found = True
            print(f"FOUND DATA FOR {target}!")
            for g in d.get('games', []):
                h = g['homeTeam']['teamName']
                a = g['awayTeam']['teamName']
                print(f" - {a} @ {h}")
            break
            
    if not found:
        print(f"NO DATA FOUND for {target}.")
        # specific check for 2024 to confirm file is valid otherwise
        print("Checking 2024-12-09 just to verify file works...")
        for d in dates:
             if "12/09/2024" in d['gameDate']:
                 print("Found 2024 data (Knicks vs Raptors etc) - File is valid.")
                 break

except Exception as e:
    print(f"Error: {e}")
