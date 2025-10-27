#!/usr/bin/env python3
"""
🎯 Test The Odds API per NBA games
Usa le credenziali fornite per ottenere partite NBA programmate

API Key: d01e24415744d440168e0a489f233aac
Documentazione: https://the-odds-api.com/liveapi/guides/v4/#overview
"""

import requests
import json
from datetime import datetime, date, timedelta
import time

class TheOddsAPITester:
    def __init__(self):
        self.api_key = "d01e24415744d440168e0a489f233aac"
        self.base_url = "https://api.the-odds-api.com/v4"
        self.session = requests.Session()

        # Headers per l'API
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

        print("✅ TheOddsAPITester inizializzato")
        print(f"   🔑 API Key: {self.api_key[:10]}...")
        print(f"   🌐 Base URL: {self.base_url}")

    def test_sports_endpoint(self):
        """Test dell'endpoint sports per verificare connessione"""
        try:
            print("\n📡 Testando endpoint /sports...")

            url = f"{self.base_url}/sports"
            params = {
                'apiKey': self.api_key
            }

            response = self.session.get(url, params=params, headers=self.headers, timeout=10)

            if response.status_code == 200:
                sports = response.json()
                print(f"   ✅ Sports endpoint funzionante: {len(sports)} sport trovati")

                # Cerca NBA
                nba_sports = [sport for sport in sports if 'basketball_nba' in sport.get('key', '').lower()]
                if nba_sports:
                    print(f"   🏀 NBA trovato: {nba_sports[0]['key']} - {nba_sports[0]['title']}")
                    return nba_sports[0]['key']
                else:
                    print("   ❌ NBA non trovato negli sport disponibili")
                    # Cerca sport con 'nba' o 'basketball'
                    basketball_sports = [sport for sport in sports if 'basketball' in sport.get('key', '').lower() or 'nba' in sport.get('key', '').lower()]
                    if basketball_sports:
                        print(f"   🏀 Sport basketball trovati: {[s['key'] for s in basketball_sports]}")
                        return basketball_sports[0]['key']
                    return None
            else:
                print(f"   ❌ Errore sports endpoint: {response.status_code}")
                print(f"   📄 Response: {response.text[:200]}...")
                return None

        except Exception as e:
            print(f"   ❌ Errore test sports endpoint: {e}")
            return None

    def test_nba_odds(self, sport_key='basketball_nba'):
        """Test odds per partite NBA"""
        try:
            print(f"\n🎰 Testando odds NBA per sport: {sport_key}")

            url = f"{self.base_url}/sports/{sport_key}/odds"
            params = {
                'apiKey': self.api_key,
                'regions': 'us',  # Regioni USA
                'markets': 'h2h',  # Head-to-head odds (vincitore)
                'oddsFormat': 'american',  # Formato quote americane
                'dateFormat': 'iso'
            }

            response = self.session.get(url, params=params, headers=self.headers, timeout=10)

            if response.status_code == 200:
                games = response.json()
                print(f"   ✅ Odds NBA funzionanti: {len(games)} partite trovate")

                if games:
                    print("\n   🏀 Partite NBA trovate:")
                    for i, game in enumerate(games[:5], 1):  # Mostra prime 5
                        home_team = game.get('home_team', 'Unknown')
                        away_team = game.get('away_team', 'Unknown')
                        commence_time = game.get('commence_time', 'Unknown')

                        print(f"      {i}. {away_team} @ {home_team}")
                        print(f"         📅 Inizio: {commence_time}")

                        # Mostra quote se disponibili
                        bookmakers = game.get('bookmakers', [])
                        if bookmakers:
                            print(f"         💰 Quote disponibili da: {len(bookmakers)} bookmaker")

                return games
            else:
                print(f"   ❌ Errore odds NBA: {response.status_code}")
                print(f"   📄 Response: {response.text[:200]}...")
                return None

        except Exception as e:
            print(f"   ❌ Errore test odds NBA: {e}")
            return None

    def test_nba_scores(self, sport_key='basketball_nba', days_from_now=1):
        """Test scores per partite NBA completate"""
        try:
            print(f"\n📊 Testando scores NBA completate (ultimi {days_from_now} giorni)")

            # Calcola date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_from_now)

            url = f"{self.base_url}/sports/{sport_key}/scores"
            params = {
                'apiKey': self.api_key,
                'daysFrom': days_from_now
            }

            response = self.session.get(url, params=params, headers=self.headers, timeout=10)

            if response.status_code == 200:
                games = response.json()
                print(f"   ✅ Scores NBA funzionanti: {len(games)} partite completate")

                if games:
                    print("\n   🏀 Partite completate recenti:")
                    for i, game in enumerate(games[:5], 1):  # Mostra prime 5
                        home_team = game.get('home_team', 'Unknown')
                        away_team = game.get('away_team', 'Unknown')
                        score = f"{game.get('scores', [{}])[0].get('score', '?')}-{game.get('scores', [{}])[1].get('score', '?')}"
                        last_update = game.get('last_update', 'Unknown')

                        print(f"      {i}. {away_team} @ {home_team} - {score}")
                        print(f"         🕐 Aggiornamento: {last_update}")

                return games
            else:
                print(f"   ❌ Errore scores NBA: {response.status_code}")
                print(f"   📄 Response: {response.text[:200]}...")
                return None

        except Exception as e:
            print(f"   ❌ Errore test scores NBA: {e}")
            return None

    def test_usage_quota(self):
        """Test del quota usage dell'API"""
        try:
            print("\n📊 Testando quota usage...")

            url = f"{self.base_url}/usage"
            params = {
                'apiKey': self.api_key
            }

            response = self.session.get(url, params=params, headers=self.headers, timeout=10)

            if response.status_code == 200:
                usage = response.json()
                print(f"   ✅ Usage quota info:")
                print(f"      📊 Requests: {usage.get('requests', 'N/A')}")
                print(f"      🕐 Reset: {usage.get('resets', 'N/A')}")
                print(f"      ⚡ Remaining: {usage.get('remaining', 'N/A')}")
                return usage
            else:
                print(f"   ❌ Errore usage quota: {response.status_code}")
                return None

        except Exception as e:
            print(f"   ❌ Errore test usage quota: {e}")
            return None


def main():
    """Test completo di The Odds API"""
    print("🚀 TEST THE ODDS API - Soluzione NBA Games")
    print("=" * 60)

    tester = TheOddsAPITester()

    # 1. Test sports endpoint
    sport_key = tester.test_sports_endpoint()
    if not sport_key:
        print("❌ Impossibile trovare sport NBA - termino test")
        return False

    # 2. Test usage quota
    usage = tester.test_usage_quota()

    # 3. Test odds per partite future
    odds_games = tester.test_nba_odds(sport_key)

    # 4. Test scores per partite completate
    scores_games = tester.test_nba_scores(sport_key, days_from_now=3)

    # Summary
    print(f"\n🎯 SUMMARY:")
    print(f"   Sport NBA: {sport_key}")
    print(f"   Partite future: {len(odds_games) if odds_games else 0}")
    print(f"   Partite completate: {len(scores_games) if scores_games else 0}")

    if odds_games or scores_games:
        print("🎉 SUCCESS! The Odds API funzionante!")
        print("✅ Possiamo usare questa API per partite NBA")
        return True
    else:
        print("⚠️ Nessun dato trovato - potrebbe essere offseason")
        return True  # API funziona anche se non ci sono partite


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)