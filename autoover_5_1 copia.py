# nba_sistema_automatico_v2.7_with_injury.py (Completo con Injury Report)

# Forza il ricaricamento del modulo injury_reporter
import sys
import importlib
if 'injury_reporter' in sys.modules:
    print("🔄 Forzo il ricaricamento del modulo injury_reporter...")
    importlib.invalidate_caches()
    injury_reporter = importlib.import_module('injury_reporter')
    importlib.reload(injury_reporter)
    print("✅ Modulo injury_reporter ricaricato con successo")
    # Verifica che la classe esista
    if hasattr(injury_reporter, 'PlayerImpactAnalyzer'):
        print(f"✅ PlayerImpactAnalyzer trovato nel modulo ricaricato")
        print(f"   Metodi disponibili: {[m for m in dir(injury_reporter.PlayerImpactAnalyzer) if not m.startswith('_')]}")

import pandas as pd
import numpy as np
import joblib
import os
import json
import warnings
import math
from scipy import stats as scipy_stats
import pickle
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
# --- NUOVO: Import necessario per il calcolo del trend
from sklearn.linear_model import LinearRegression
import argparse
from datetime import datetime, date, timedelta
import time
import re
import http.client
from urllib.parse import urlencode
from sklearn.preprocessing import StandardScaler  # Added import

# Importazioni da nba_api
from nba_api.stats.static import teams as nba_static_teams
from nba_api.stats.endpoints import (
    teamdashboardbygeneralsplits as nba_teamdashboard,
    teamgamelogs as nba_teamgamelogs,
    leaguegamefinder as nba_leaguegamefinder,
    boxscoretraditionalv2 as nba_boxscoretraditionalv2,
    commonteamroster as nba_commonteamroster,
    playergamelogs as nba_playergamelogs,
    playerestimatedmetrics as nba_playerestimatedmetrics,
    playerindex as nba_playerindex
)

# Import locali
from data_provider import NBADataProvider
from injury_reporter import InjuryReporter
from player_impact_analyzer import PlayerImpactAnalyzer
from player_momentum_predictor import PlayerMomentumPredictor
# Spostato l'import qui per risolvere l'errore di inizializzazione
from probabilistic_model_v2 import ProbabilisticModel

# Configurazione del logging

# Per The Odds API
import requests
from dotenv import load_dotenv
from config import ODDS_API_KEY

warnings.filterwarnings('ignore')

from config import (
    NBA_API_REQUEST_DELAY, APISPORTS_REQUEST_DELAY, ODDS_API_REQUEST_DELAY,
    DATA_DIR, MODELS_BASE_DIR, SETTINGS_FILE,
    ODDS_API_KEY, APISPORTS_API_KEY,
    XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE
)

import time
from requests.exceptions import RequestException

def with_retry(max_retries=3, delay=1):
    """
    Decoratore per aggiungere ritry alle chiamate API
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (RequestException, json.JSONDecodeError) as e:
                    print(f"Tentativo {retries+1} fallito: {e}")
                    retries += 1
                    time.sleep(delay)
                    delay *= 2  # Backoff esponenziale
            print(f"Errore dopo {max_retries} tentativi")
            return None
        return wrapper
    return decorator


def is_off_season(check_date=None):
    """Determina se la data corrente è in off-season NBA.
    Off-season: da luglio a settembre.
    """
    if check_date is None:
        check_date = date.today()
    return check_date.month >= 7 and check_date.month <= 9

def get_test_games_for_off_season():
    """Genera partite di test durante l'off-season"""
    print("🏀 OFF-SEASON: Generando partite di test...")
    
    test_games = [
        {
            'GAME_ID': 'TEST_001',
            'date': '2025-10-15',
            'time': '19:00:00',
            'home_team': 'Los Angeles Lakers',
            'away_team': 'Boston Celtics',
            'home_team_id': 1610612747,
            'away_team_id': 1610612738,
            'odds': []
        },
        {
            'GAME_ID': 'TEST_002',
            'date': '2025-10-15',
            'time': '21:30:00',
            'home_team': 'New York Knicks',
            'away_team': 'Golden State Warriors',
            'home_team_id': 1610612752,
            'away_team_id': 1610612744,
            'odds': []
        }
    ]
    
    return test_games


class NBACompleteSystem:


    """Sistema NBA completo per predizioni e analisi scommesse"""
    
    def __init__(self, data_provider=None):
        self.data_provider = data_provider
        self.bankroll = self._load_bankroll()
        # Inizializza i predictor
        self.impact_analyzer = PlayerImpactAnalyzer(self.data_provider)

        # Inizializza il modello probabilistico
        try:
            print("🔄 Inizializzazione modello probabilistico...")
            self.probabilistic_model = ProbabilisticModel()
            print("✅ Modello probabilistico inizializzato con successo")
        except Exception as e:
            print(f"❌ Errore durante l'inizializzazione del modello probabilistico (ProbabilisticModel): {e}")
            self.probabilistic_model = None

        # Inizializza il predictor del momentum con i percorsi corretti
        self.momentum_predictor = PlayerMomentumPredictor(
            data_dir=os.path.join(DATA_DIR, 'momentum'), # Usa DATA_DIR da config
            models_dir=MODELS_BASE_DIR, # Usa MODELS_BASE_DIR da config
            nba_data_provider=self.data_provider  # Aggiungi questa riga
        )
        # Aggiungi un log per verificare lo stato di is_trained dopo l'inizializzazione
        if self.momentum_predictor:
            print(f"🔍 [MOMENTUM_INIT_CHECK] PlayerMomentumPredictor.is_trained: {self.momentum_predictor.is_trained}")
        else:
            print("⚠️ [MOMENTUM_INIT_CHECK] PlayerMomentumPredictor non inizializzato.")
        # Inizializza il sistema di report infortuni
        self.injury_reporter = InjuryReporter(data_provider)
        
        # Imposta l'impact_analyzer nell'injury_reporter se necessario
        if hasattr(self.injury_reporter, 'set_impact_analyzer'):
            self.injury_reporter.set_impact_analyzer(self.impact_analyzer)
        
        # Inizializza l'analizzatore di impatto giocatori
        try:
            print("🔄 Inizializzazione PlayerImpactAnalyzer...")
            self.impact_analyzer = PlayerImpactAnalyzer(data_provider)
            print("✅ PlayerImpactAnalyzer inizializzato con successo")
            print(f"   Metodi disponibili: {[m for m in dir(self.impact_analyzer) if not m.startswith('_')]}")
        except Exception as e:
            print(f"❌ Errore durante l'inizializzazione di PlayerImpactAnalyzer: {e}")
            self.impact_analyzer = None
        
        # Inizializza il sistema probabilistico (già fatto sopra, qui si assegna a probabilistic_system)
        try:
            if self.probabilistic_model:
                self.probabilistic_system = self.probabilistic_model
                print("✅ Sistema probabilistico (self.probabilistic_system) assegnato.")
            else:
                print("⚠️ Modello probabilistico (self.probabilistic_model) non inizializzato, self.probabilistic_system è None.")
                self.probabilistic_system = None
        except Exception as e:
            print(f"❌ Errore assegnazione self.probabilistic_system: {e}")
            self.probabilistic_system = None
    
    def _load_bankroll(self):
        """Carica il bankroll dal file di configurazione"""
        try:
            bankroll_file = os.path.join(DATA_DIR, 'bankroll.json')
            if os.path.exists(bankroll_file):
                with open(bankroll_file, 'r') as f:
                    data = json.load(f)
                    return float(data.get('current_bankroll', 78.88))  # Default 78.88€
        except Exception:
            pass
        return 78.88  # Bankroll iniziale predefinito
    
    def _save_bankroll(self):
        """Salva il bankroll corrente"""
        try:
            bankroll_file = os.path.join(DATA_DIR, 'bankroll.json')
            with open(bankroll_file, 'w') as f:
                json.dump({'current_bankroll': self.bankroll}, f)
        except Exception as e:
            print(f"❌ Errore salvataggio bankroll: {e}")
    
    def get_current_bankroll(self):
        """Restituisce il bankroll corrente"""
        return self.bankroll
        
    def set_bankroll(self, amount):
        """Imposta il bankroll corrente e lo salva"""
        self.bankroll = float(amount)
        self._save_bankroll()
        return self.bankroll
    
    def _extract_team_names(self, game_data):
        """
        Estrae i nomi delle squadre da game_data in modo robusto
        Restituisce una tupla (home_team, away_team)
        """
        home_team = None
        away_team = None
        
        # 1. Prova a ottenere i nomi da game_info
        game_info = game_data.get('game_info', {})
        if isinstance(game_info, dict):
            home_team = game_info.get('home_team')
            away_team = game_info.get('away_team')
        
        # 2. Se manca qualche nome, prova da team_stats
        if not home_team or not away_team:
            team_stats = game_data.get('team_stats', {})
            if isinstance(team_stats, dict):
                if not home_team and 'home' in team_stats and isinstance(team_stats['home'], dict):
                    home_team = team_stats['home'].get('team_name') or team_stats['home'].get('name')
                if not away_team and 'away' in team_stats and isinstance(team_stats['away'], dict):
                    away_team = team_stats['away'].get('team_name') or team_stats['away'].get('name')
        
        # 3. Se manca ancora, prova dalle chiavi di primo livello
        if not home_team:
            home_team = game_data.get('home_team')
        if not away_team:
            away_team = game_data.get('away_team')
        
        # 4. Se ancora manca, prova a cercare in altre chiavi
        if not home_team or not away_team:
            for key, value in game_data.items():
                if isinstance(value, str) and ' vs ' in value:
                    parts = value.split(' vs ')
                    if len(parts) == 2:
                        if not home_team:
                            home_team = parts[0].strip()
                        if not away_team:
                            away_team = parts[1].strip()
                        break
        
        # 5. Se ancora manca, usa i valori di default
        home_team = home_team or 'Squadra Casa'
        away_team = away_team or 'Squadra Ospite'
        
        return home_team, away_team
        
    def predict_complete_game_flow(self, game_data, manual_base_line=None):
        """
        Predice il flusso completo di una partita
        
        Args:
            game_data: Dizionario con i dati della partita
            manual_base_line: Linea centrale opzionale per generare quote multiple
        """
        print("\n" + "="*80)
        print("🏁 INIZIO predict_complete_game_flow")
        print(f"🔍 ID oggetto game_data in input: {id(game_data)}")
        
        # Verifica che game_data contenga le informazioni minime necessarie
        if not isinstance(game_data, dict):
            print("❌ ERRORE: game_data non è un dizionario")
            return None
            
        # Se manca game_info o team_stats, proviamo a recuperarli
        if 'game_info' not in game_data or 'team_stats' not in game_data:
            print("⚠️ game_info o team_stats mancanti in game_data, tentativo di recupero...")
            if hasattr(self, 'data_provider') and hasattr(self.data_provider, 'get_game_details'):
                try:
                    # Prova a recuperare i dettagli del gioco usando l'ID della partita
                    game_id = game_data.get('game_id')
                    if game_id:
                        print(f"🔍 Recupero dettagli partita per ID: {game_id}")
                        game_details = self.data_provider.get_game_details(game_id)
                        if game_details:
                            # Unisci i dettagli con game_data esistente
                            game_data.update(game_details)
                            print("✅ Dettagli partita recuperati con successo")
                        else:
                            print("⚠️ Impossibile recuperare i dettagli della partita")
                    else:
                        print("⚠️ ID partita non disponibile in game_data")
                except Exception as e:
                    print(f"❌ Errore durante il recupero dei dettagli della partita: {e}")
            else:
                print("⚠️ data_provider non disponibile o non ha il metodo get_game_details")
        
        # Usa il metodo per estrarre i nomi delle squadre
        home_team, away_team = self._extract_team_names(game_data)
        
        # Stampa i dettagli di debug
        print(f"📅 Data in game_data: {game_data.get('game_info', {}).get('date', 'N/D')}")
        print(f"🏠 Squadre estratte: {home_team} vs {away_team}")
        print(f"📊 Chiavi in game_data: {list(game_data.keys())}")
        
        # Verifica se team_stats è presente
        print(f"📊 team_stats in game_data: {'presente' if 'team_stats' in game_data else 'assente'}")
        if 'team_stats' in game_data:
            print(f"   - Chiavi in team_stats: {list(game_data['team_stats'].keys())}")
            if 'home' in game_data['team_stats']:
                print(f"   - home team stats: {list(game_data['team_stats']['home'].keys())}")
            if 'away' in game_data['team_stats']:
                print(f"   - away team stats: {list(game_data['team_stats']['away'].keys())}")
        
        if not game_data:
            print("❌ Dati partita non validi")
            return None
            
        # Normalizza la struttura dei dati della partita
        if 'game_info' not in game_data:
            game_data = {
                'game_info': {
                    'home_team': game_data.get('home_team', ''),
                    'away_team': game_data.get('away_team', ''),
                    'home_team_id': game_data.get('home_team_id'),
                    'away_team_id': game_data.get('away_team_id'),
                    'date': game_data.get('date', datetime.now().strftime('%Y-%m-%d')),
                    'time': game_data.get('time', ''),
                    'game_id': game_data.get('game_id', ''),
                    'status': 'SCHEDULED'  # Impostiamo come predefinito
                },
                'team_stats': game_data.get('team_stats', {})
            }
        
        # Verifica se la partita è già stata giocata
        if game_data['game_info'].get('status') == 'COMPLETED':
            print("📊 Partita già giocata, analisi retrospettiva...")
            return self._analyze_completed_game(game_data)
        
        try:
            # Recupera dati momentum
            home_team_id = self._get_team_id(game_data, 'home')
            away_team_id = self._get_team_id(game_data, 'away')
            game_date = datetime.strptime(game_data['game_info']['date'], '%Y-%m-%d')
        except Exception as e:
            print(f"⚠️ Errore durante il recupero dei dati della partita: {e}")
            print(f"   Dettagli partita: {game_data}")
        else:
            print(" data_provider non disponibile o non ha il metodo get_game_details")
        
        # Get team IDs from game_data
        home_team_id = game_data.get('home_team_id') or game_data.get('game_info', {}).get('home_team_id')
        away_team_id = game_data.get('away_team_id') or game_data.get('game_info', {}).get('away_team_id')
        
        # If still not found, try to get from team_stats if available
        if not home_team_id or not away_team_id:
            home_team_id = game_data.get('team_stats', {}).get('home', {}).get('id') or \
                          game_data.get('team_stats', {}).get('home', {}).get('team_id_nba') or \
                          game_data.get('team_stats', {}).get('home', {}).get('team_id')
            
            away_team_id = game_data.get('team_stats', {}).get('away', {}).get('id') or \
                          game_data.get('team_stats', {}).get('away', {}).get('team_id_nba') or \
                          game_data.get('team_stats', {}).get('away', {}).get('team_id')
        
        if not home_team_id or not away_team_id:
            print("⚠️ [WARNING] Team IDs not found in game_data or team_stats")
            print("   Available game_data keys:", list(game_data.keys()))
            if 'game_info' in game_data:
                print("   Available game_info keys:", list(game_data['game_info'].keys()))
            if 'team_stats' in game_data:
                print("   Available team_stats['home'] keys:", list(game_data['team_stats']['home'].keys()) if 'home' in game_data['team_stats'] else "No 'home' in team_stats")
                print("   Available team_stats['away'] keys:", list(game_data['team_stats']['away'].keys()) if 'away' in game_data['team_stats'] else "No 'away' in team_stats")
            return None
            
        # Ensure team IDs are integers
        home_team_id = int(home_team_id)
        away_team_id = int(away_team_id)
        
        # Ottieni i nomi delle squadre da game_data
        home_team_name = game_data.get('home_team', '')
        away_team_name = game_data.get('away_team', '')
        
        print(f"🔍 Recupero roster per {home_team_name} (ID: {home_team_id})...")
        home_roster = self._get_team_roster(home_team_id, home_team_name)
        print(f"✅ Trovati {len(home_roster) if not home_roster.empty else 0} giocatori per {home_team_name}")
        
        print(f"🔍 Recupero roster per {away_team_name} (ID: {away_team_id})...")
        away_roster = self._get_team_roster(away_team_id, away_team_name)
        print(f"✅ Trovati {len(away_roster) if not away_roster.empty else 0} giocatori per {away_team_name}")
        
        # Verifica se i DataFrame sono vuoti
        if home_roster.empty or away_roster.empty:
            print("⚠️ [WARNING] Impossibile recuperare i roster delle squadre")
            return None
        
        # Verifica se le colonne richieste sono presenti
        required_columns = ['PLAYER_ID', 'PLAYER_NAME', 'POSITION', 'MIN']
        missing_columns_home = [col for col in required_columns if col not in home_roster.columns]
        missing_columns_away = [col for col in required_columns if col not in away_roster.columns]
        
        if missing_columns_home or missing_columns_away:
            print(f"⚠️ [WARNING] Colonne mancanti nei roster: Casa={missing_columns_home}, Trasferta={missing_columns_away}")
            # Aggiungi colonne mancanti con valori di default
            for col in missing_columns_home:
                home_roster[col] = 0 if col == 'MIN' else ''
            for col in missing_columns_away:
                away_roster[col] = 0 if col == 'MIN' else ''
        
        # Aggiungi i roster a game_data
        game_data['home_roster'] = home_roster
        game_data['away_roster'] = away_roster
        
        # Aggiungi gli ID delle squadre a game_data per riferimento futuro
        game_data['home_team_id'] = home_team_id
        game_data['away_team_id'] = away_team_id
        
        # Also ensure team_stats has the IDs if they're missing
        if 'team_stats' in game_data and 'home' in game_data['team_stats']:
            game_data['team_stats']['home']['id'] = home_team_id
            game_data['team_stats']['home']['team_id_nba'] = home_team_id
            
        if 'team_stats' in game_data and 'away' in game_data['team_stats']:
            game_data['team_stats']['away']['id'] = away_team_id
            game_data['team_stats']['away']['team_id_nba'] = away_team_id
        
        # Calcola l'impatto del momentum per ogni squadra
        try:
            home_momentum = self.momentum_predictor.predict_team_momentum_impact(home_roster_df)
            away_momentum = self.momentum_predictor.predict_team_momentum_impact(away_roster_df)
            
            # Calcola il fattore di momentum
            momentum_factor = 1 + (home_momentum - away_momentum) * 0.15
            print(f"📊 Momentum factor calcolato: {momentum_factor:.2f}")
            
        except Exception as e:
            print(f"⚠️ Errore nel calcolo del momentum: {e}")
            momentum_factor = 1.0  # Valore di default in caso di errore
            
        # Usa il sistema probabilistico se disponibile, altrimenti usa quello base
        if self.probabilistic_system:
            return self._probabilistic_prediction(game_data, manual_base_line, momentum_factor)
        elif self.probabilistic_model: # Fallback se probabilistic_system non è stato assegnato ma il modello sì
            print("⚠️ self.probabilistic_system è None, ma self.probabilistic_model esiste. Tentativo di usarlo.")
            self.probabilistic_system = self.probabilistic_model
            return self._probabilistic_prediction(game_data, manual_base_line, momentum_factor)
        else:
            return self._basic_prediction(game_data, momentum_factor)
    
    def _analyze_completed_game(self, game_data):
        """Analizza una partita già completata"""
        result = game_data['game_info']['result']
        print(f"🏀 {game_data['game_info']['away_team']} @ {game_data['game_info']['home_team']}")
        print(f"📅 {game_data['game_info']['date']}")
        print(f"🏆 Risultato: {result['away_score']} - {result['home_score']} (Totale: {result['total_score']})")
        
        return {
            'analysis_result': {'actual_total': result['total_score']},
            'system_type': 'retrospective'
        }
    
    def _calculate_prediction_confidence(self, game_data, prediction_result):
        """
        Calcola un punteggio di confidenza per la predizione basato su:
        - Edge della scommessa (fino a 40 punti)
        - Qualità e completezza dei dati (fino a 20 punti)
        - Affidabilità del modello (fino a 20 punti)
        - Stabilità delle squadre (fino a 20 punti)
        """
        # Inizializza i punteggi
        scores = {
            'edge_score': 0.0,
            'data_quality_score': 0.0,
            'model_reliability_score': 0.0,
            'team_stability_score': 0.0
        }
        
        # 1. Calcola punteggio basato sull'edge (0-40% del totale)
        edge = prediction_result.get('edge', 0)
        scores['edge_score'] = min(40, max(0, edge * 400))  # Edge del 10% = 40 punti
        
        # 2. Valuta qualità e completezza dei dati (0-20% del totale)
        # Stampa di debug per analizzare la struttura di game_data
        print("\n=== DEBUG STRUTTURA GAME_DATA ===")
        print("Chiavi disponibili in game_data:", list(game_data.keys()))
        
        # Cerca le statistiche in diverse posizioni possibili
        home_stats = {}
        away_stats = {}
        
        # 1. Controlla se ci sono statistiche in game_data['team_stats']
        if 'team_stats' in game_data and isinstance(game_data['team_stats'], dict):
            if 'home' in game_data['team_stats'] and isinstance(game_data['team_stats']['home'], dict):
                home_stats = game_data['team_stats']['home']
                print("Trovate statistiche in game_data['team_stats']['home']")
                
            if 'away' in game_data['team_stats'] and isinstance(game_data['team_stats']['away'], dict):
                away_stats = game_data['team_stats']['away']
                print("Trovate statistiche in game_data['team_stats']['away']")
        
        # 2. Se non trovate, prova le posizioni alternative
        if not home_stats and 'home_team' in game_data and isinstance(game_data['home_team'], dict):
            if 'stats' in game_data['home_team'] and isinstance(game_data['home_team']['stats'], dict):
                home_stats = game_data['home_team']['stats']
                print("Trovate statistiche in game_data['home_team']['stats']")
            else:
                home_stats = game_data['home_team']
                print("Trovate statistiche in game_data['home_team']")
        
        if not away_stats and 'away_team' in game_data and isinstance(game_data['away_team'], dict):
            if 'stats' in game_data['away_team'] and isinstance(game_data['away_team']['stats'], dict):
                away_stats = game_data['away_team']['stats']
                print("Trovate statistiche in game_data['away_team']['stats']")
            else:
                away_stats = game_data['away_team']
                print("Trovate statistiche in game_data['away_team']")
        
        # 3. Se ancora vuote, prova altre posizioni
        if not home_stats and 'home_team_stats' in game_data and isinstance(game_data['home_team_stats'], dict):
            home_stats = game_data['home_team_stats']
            print("Trovate statistiche in game_data['home_team_stats']")
            
        if not away_stats and 'away_team_stats' in game_data and isinstance(game_data['away_team_stats'], dict):
            away_stats = game_data['away_team_stats']
            print("Trovate statistiche in game_data['away_team_stats']")
            
        # 4. Se ancora vuote, verifica se ci sono statistiche dirette in game_data
        if not home_stats and 'home_stats' in game_data and isinstance(game_data['home_stats'], dict):
            home_stats = game_data['home_stats']
            print("Trovate statistiche in game_data['home_stats']")
            
        if not away_stats and 'away_stats' in game_data and isinstance(game_data['away_stats'], dict):
            away_stats = game_data['away_stats']
            print("Trovate statistiche in game_data['away_stats']")
            
        # Stampa le chiavi delle statistiche trovate per il debug
        print("\n=== STATISTICHE TROVATE ===")
        print(f"Casa ({game_data.get('game_info', {}).get('home_team', 'Sconosciuta')}): {list(home_stats.keys()) if home_stats else 'Nessuna statistica trovata'}")
        print(f"Trasferta ({game_data.get('game_info', {}).get('away_team', 'Sconosciuta')}): {list(away_stats.keys()) if away_stats else 'Nessuna statistica trovata'}")
        
        # Se esiste un dizionario 'stats' all'interno, usalo
        if home_stats and 'stats' in home_stats and isinstance(home_stats['stats'], dict):
            home_stats = home_stats['stats']
            print("Utilizzate statistiche da home_stats['stats']")
            
        if away_stats and 'stats' in away_stats and isinstance(away_stats['stats'], dict):
            away_stats = away_stats['stats']
            print("Utilizzate statistiche da away_stats['stats']")
            
        # Cerca le statistiche disponibili usando varie chiavi alternative
        available_home_stats = set()
        available_away_stats = set()
        
        # Statistiche chiave da verificare (con alias alternativi)
        key_stats = [
            'games_played', 'games', 'gp',
            'points', 'pts', 'pf',
            'points_against', 'opp_pts', 'opp_points',
            'pace', 'pace_rating',
            'off_rating', 'off_eff', 'offensive_rating',
            'def_rating', 'def_eff', 'defensive_rating',
            'win_rate', 'win_pct', 'win_percentage',
            'last_10_games', 'last10', 'last_10'
        ]
        
        # Stampa di debug per le statistiche trovate
        print("\n=== STATISTICHE TROVATE ===")
        if home_stats:
            print(f"\nStatistiche casa ({game_data.get('game_info', {}).get('home_team', 'Sconosciuta')}):")
            for k, v in home_stats.items():
                print(f"  - {k}: {v}")
        
        if away_stats:
            print(f"\nStatistiche trasferta ({game_data.get('game_info', {}).get('away_team', 'Sconosciuta')}):")
            for k, v in away_stats.items():
                print(f"  - {k}: {v}")
        
        # Statistiche richieste per il calcolo della confidenza
        required_stats = [
            'games_played', 'games', 'gp',
            'points', 'pts', 'pf',
            'points_against', 'opp_pts', 'opp_points',
            'pace', 'pace_rating',
            'off_rating', 'off_eff', 'offensive_rating',
            'def_rating', 'def_eff', 'defensive_rating',
            'win_rate', 'win_pct', 'win_percentage',
            'last_10_games', 'last10', 'last_10'
        ]
        
        # Controlla quali statistiche sono disponibili
        def check_available_stats(stats_dict):
            if not isinstance(stats_dict, dict):
                print("  - stats_dict non è un dizionario")
                return set()
                
            available = set()
            
            # Cerca le statistiche direttamente nelle chiavi
            for stat in required_stats:
                # Cerca corrispondenza case-insensitive
                stat_lower = stat.lower()
                for key in stats_dict.keys():
                    if stat_lower == str(key).lower():
                        available.add(stat)
                        break
            
            # Cerca anche nei valori se sono dizionari
            for value in stats_dict.values():
                if isinstance(value, dict):
                    for stat in required_stats:
                        stat_lower = stat.lower()
                        for key in value.keys():
                            if stat_lower == str(key).lower():
                                available.add(stat)
                                break
            
            return available
        
        available_home_stats = check_available_stats(home_stats)
        available_away_stats = check_available_stats(away_stats)
        
        # Trova statistiche mancanti
        home_missing = [s for s in required_stats if s not in available_home_stats]
        away_missing = [s for s in required_stats if s not in available_away_stats]
        
        # Calcola completezza
        home_data_completeness = len(available_home_stats) / len(required_stats)
        away_data_completeness = len(available_away_stats) / len(required_stats)
        
        # Punteggio basato sulla completezza (0-20 punti)
        scores['data_quality_score'] = (home_data_completeness + away_data_completeness) * 10
        
        # Salva i dettagli delle statistiche mancanti
        data_quality_details = {
            'home_missing': home_missing,
            'away_missing': away_missing,
            'total_expected': len(required_stats),
            'home_available': len(available_home_stats),
            'away_available': len(available_away_stats),
            'home_stats_found': list(available_home_stats),
            'away_stats_found': list(available_away_stats)
        }
        
        # 3. Affidabilità del modello avanzata (0-20% del totale)
        # Calcola un punteggio di affidabilità basato su diversi fattori

        # 3.1 Deviazione standard della predizione (massimo 8 punti)
        std_dev = prediction_result.get('prediction', {}).get('predicted_sigma', 15)  # Valore di default 15 punti
        std_reliability = max(0, 1 - (std_dev / 30))  # Normalizza tra 0 e 1
        std_score = std_reliability * 8  # 0-8 punti
        
        # 3.2 Statistiche Head-to-Head (massimo 4 punti)
        h2h_score = 0
        try:
            home_team = game_data['game_info'].get('home_team', '')
            away_team = game_data['game_info'].get('away_team', '')
            
            # Verifica se abbiamo dati head-to-head
            h2h_data = self._get_head_to_head_data(home_team, away_team)
            if h2h_data and h2h_data.get('games', 0) > 0:
                h2h_score = min(4, h2h_data.get('games', 0))  # Max 4 punti
                print(f"   📊 Dati Head-to-Head: {h2h_data.get('games', 0)} partite trovate")
            else:
                print(f"   ⚠️ Nessun dato Head-to-Head trovato per {home_team} vs {away_team}")
        except Exception as e:
            print(f"   ⚠️ Errore recupero dati Head-to-Head: {e}")
            
        # 3.3 Valutazione trend recenti (massimo 4 punti)
        trend_score = 0
        try:
            # Estrai dati ultimi 10 incontri dalle statistiche delle squadre
            home_last10 = game_data['team_stats'].get('home', {}).get('last_10_games', {})
            away_last10 = game_data['team_stats'].get('away', {}).get('last_10_games', {})
            
            # Calcola la consistenza delle prestazioni recenti
            home_win_pct = home_last10.get('wins', 0) / 10 if 'wins' in home_last10 else 0.5
            away_win_pct = away_last10.get('wins', 0) / 10 if 'wins' in away_last10 else 0.5
            
            # Premiamo la consistenza - team che vincono/perdono costantemente sono più prevedibili
            home_consistency = abs(home_win_pct - 0.5) * 2  # 0-1 range
            away_consistency = abs(away_win_pct - 0.5) * 2  # 0-1 range
            
            trend_score = (home_consistency + away_consistency) * 2  # Max 4 punti
            print(f"   📈 Consistenza trend recenti: Casa {home_consistency:.2f}, Trasferta {away_consistency:.2f}")
        except Exception as e:
            print(f"   ⚠️ Errore analisi trend recenti: {e}")
        
        # 3.4 Fattori contestuali/motivazionali (massimo 4 punti)
        context_score = 0
        try:
            # Controlla la data della partita
            game_date = game_data['game_info'].get('date')
            if game_date:
                if isinstance(game_date, str):
                    try:
                        game_date = datetime.strptime(game_date, '%Y-%m-%d').date()
                    except ValueError:
                        game_date = date.today()
                
                # Verifica se siamo in Regular Season o Playoffs
                if game_date.month >= 4 and game_date.month <= 6:
                    # Playoff: le partite sono più prevedibili, maggiore impegno
                    context_score += 2
                    print(f"   🏆 Partita playoff/fine stagione: maggiore affidabilità previsionale")
                elif game_date.month >= 10 or game_date.month <= 1:
                    # Inizio stagione: meno prevedibile
                    context_score += 0.5
                    print(f"   🏀 Inizio stagione: minore affidabilità previsionale")
                else:
                    # Metà stagione: affidabilità standard
                    context_score += 1
                    print(f"   🏀 Metà stagione: affidabilità previsionale standard")
            
            # Verifica la posizione in classifica e l'importanza della partita
            home_win_pct = game_data['team_stats'].get('home', {}).get('win_percentage', 0.5)
            away_win_pct = game_data['team_stats'].get('away', {}).get('win_percentage', 0.5)
            
            # Se entrambe le squadre sono competitive (> 40% vittorie), la partita è più prevedibile
            if home_win_pct > 0.4 and away_win_pct > 0.4:
                context_score += 1
                print(f"   ⚖️ Entrambe le squadre competitive: maggiore affidabilità")
                
            # Se c'è una grande differenza di forza, anche questo è più prevedibile
            if abs(home_win_pct - away_win_pct) > 0.2:
                context_score += 1
                print(f"   ⚖️ Differenza di forza significativa: maggiore affidabilità")
        except Exception as e:
            print(f"   ⚠️ Errore analisi fattori contestuali: {e}")
        
        # Somma tutti i fattori di affidabilità
        model_reliability_score = std_score + h2h_score + trend_score + context_score
        # Limita al massimo di 20 punti
        model_reliability_score = min(20, model_reliability_score)
        
        print(f"   📊 Punteggio affidabilità: {model_reliability_score:.1f}/20")
        print(f"      - Deviazione standard: {std_score:.1f}/8")
        print(f"      - Head-to-Head: {h2h_score:.1f}/4")
        print(f"      - Trend recenti: {trend_score:.1f}/4")
        print(f"      - Contesto: {context_score:.1f}/4")
        
        scores['model_reliability_score'] = model_reliability_score
        
        # 4. Stabilità delle squadre (0-20% del totale)
        # Controlla infortuni e cambi recenti
        home_injuries = len(game_data.get('injury_report', {}).get('home_team', []))
        away_injuries = len(game_data.get('injury_report', {}).get('away_team', []))
        
        # Normalizza il numero di infortuni (0-3 = punteggio pieno, >5 = punteggio basso)
        injury_impact = max(0, 1 - ((home_injuries + away_injuries) / 10))
        scores['team_stability_score'] = injury_impact * 20  # 0-20 punti
        
        # Calcola il punteggio totale (0-100%)
        total_score = sum(scores.values())
        
        # Determina il livello di confidenza
        if total_score >= 80:
            confidence_level = "Alta"
        elif total_score >= 60:
            confidence_level = "Media-Alta"
        elif total_score >= 40:
            confidence_level = "Media"
        elif total_score >= 20:
            confidence_level = "Bassa"
        else:
            confidence_level = "Molto Bassa"
            
        return {
            'score': total_score,
            'level': confidence_level,
            'details': {
                'edge_contribution': scores['edge_score'],
                'data_quality': {
                    'score': scores['data_quality_score'],
                    'home_missing': data_quality_details['home_missing'],
                    'away_missing': data_quality_details['away_missing'],
                    'total_expected': data_quality_details['total_expected'],
                    'home_available': data_quality_details['home_available'],
                    'away_available': data_quality_details['away_available'],
                    'home_stats_found': data_quality_details['home_stats_found'],
                    'away_stats_found': data_quality_details['away_stats_found']
                },
                'model_reliability': scores['model_reliability_score'],
                'team_stability': scores['team_stability_score']
            }
        }
    
    def _probabilistic_prediction(self, game_data, manual_base_line=None, momentum_factor=1):
        """
        Predizione usando il sistema probabilistico
        
        Args:
            game_data: Dizionario con i dati della partita
            manual_base_line: Linea centrale opzionale per generare quote multiple
            momentum_factor: Fattore di momentum (default 1)
        """
        print("\n" + "="*80)
        print("🎲 INIZIO _probabilistic_prediction")
        print(f"🔍 ID oggetto game_data in input: {id(game_data)}")
        
        # Verifica che game_data contenga le informazioni minime necessarie
        if not isinstance(game_data, dict):
            print("❌ ERRORE: game_data non è un dizionario")
            return None
            
        # Se manca game_info o team_stats, proviamo a recuperarli
        if 'game_info' not in game_data or 'team_stats' not in game_data:
            print("⚠️ game_info o team_stats mancanti in game_data, tentativo di recupero...")
            if hasattr(self, 'data_provider') and hasattr(self.data_provider, 'get_game_details'):
                try:
                    # Prova a recuperare i dettagli del gioco usando l'ID della partita
                    game_id = game_data.get('game_id')
                    if game_id:
                        print(f"🔍 Recupero dettagli partita per ID: {game_id}")
                        game_details = self.data_provider.get_game_details(game_id)
                        if game_details:
                            # Unisci i dettagli con game_data esistente
                            game_data.update(game_details)
                            print("✅ Dettagli partita recuperati con successo")
                        else:
                            print("⚠️ Impossibile recuperare i dettagli della partita")
                    else:
                        print("⚠️ ID partita non disponibile in game_data")
                except Exception as e:
                    print(f"❌ Errore durante il recupero dei dettagli della partita: {e}")
            else:
                print("⚠️ data_provider non disponibile o non ha il metodo get_game_details")
        
        # Usa il metodo per estrarre i nomi delle squadre
        home_team, away_team = self._extract_team_names(game_data)
        
        # Stampa i dettagli di debug
        print(f"📅 Data in game_data: {game_data.get('game_info', {}).get('date', 'N/D')}")
        print(f"🏠 Squadre estratte: {home_team} vs {away_team}")
        print(f"📊 Chiavi in game_data: {list(game_data.keys())}")
        
        # Verifica se team_stats è presente
        print(f"📊 team_stats in game_data: {'presente' if 'team_stats' in game_data else 'assente'}")
        if 'team_stats' in game_data:
            print(f"   - Chiavi in team_stats: {list(game_data['team_stats'].keys())}")
            if 'home' in game_data['team_stats']:
                print(f"   - home team stats: {list(game_data['team_stats']['home'].keys())}")
            if 'away' in game_data['team_stats']:
                print(f"   - away team stats: {list(game_data['team_stats']['away'].keys())}")
        
        try:
            # Assicurati che il bankroll corrente sia incluso nei dati della partita
            if 'settings' not in game_data:
                game_data['settings'] = {}
            game_data['settings']['bankroll'] = self.bankroll
            
            # Se disponibili i report infortuni, calcola l'impatto
            if not hasattr(self, 'impact_analyzer') or self.impact_analyzer is None:
                print("⚠️ Attenzione: impact_analyzer non è stato inizializzato correttamente")
            elif 'injury_reports' in game_data:
                print("✅ Report infortuni disponibili nel game_data")
                print(f"   Chiavi disponibili: {list(game_data['injury_reports'].keys())}")
                
                # Verifica l'istanza di impact_analyzer
                print(f"🔍 [DEBUG] Tipo di impact_analyzer: {type(self.impact_analyzer).__name__}")
                print(f"🔍 [DEBUG] ID oggetto impact_analyzer: {id(self.impact_analyzer)}")
                print(f"🔍 [DEBUG] Metodi disponibili in impact_analyzer: {[m for m in dir(self.impact_analyzer) if not m.startswith('_')]}")
                
                home_team_id = game_data.get('team_stats', {}).get('home', {}).get('id')
                away_team_id = game_data.get('team_stats', {}).get('away', {}).get('id')
                
                print(f"🔍 ID squadre - Casa: {home_team_id}, Trasferta: {away_team_id}")
                
                if home_team_id and away_team_id:
                    # Calcola impatto per la squadra di casa
                    home_roster = game_data.get('injury_reports', {}).get('home_nba', {}).get('players', [])
                    home_impact = self.impact_analyzer.calculate_team_impact(home_roster, home_team_id)
                    
                    # Calcola impatto per la squadra in trasferta
                    away_roster = game_data.get('injury_reports', {}).get('away_nba', {}).get('players', [])
                    away_impact = self.impact_analyzer.calculate_team_impact(away_roster, away_team_id)
                    
                    # Aggiungi l'analisi degli impatti ai report infortuni
                    if 'impact_analysis' not in game_data['injury_reports']:
                        game_data['injury_reports']['impact_analysis'] = {}
                    
                    game_data['injury_reports']['impact_analysis'].update({
                        'home_team_impact': home_impact.get('total_impact', 0),
                        'away_team_impact': away_impact.get('total_impact', 0),
                        'home_players_affected': home_impact.get('players_affected', 0),
                        'away_players_affected': away_impact.get('players_affected', 0),
                        'home_players_out': home_impact.get('players_out', 0),
                        'away_players_out': away_impact.get('players_out', 0),
                        'home_players_questionable': home_impact.get('players_questionable', 0),
                        'away_players_questionable': away_impact.get('players_questionable', 0)
                    })
            
            # Calcola l'impatto del momentum per entrambe le squadre
            momentum_impact = 0.0
            if hasattr(self, 'momentum_predictor') and self.momentum_predictor is not None:
                # Log di verifica per self.momentum_predictor
                print(f"🔍 [MOMENTUM_PREDICT_CHECK] ID self.momentum_predictor: {id(self.momentum_predictor)}")
                print(f"🔍 [MOMENTUM_PREDICT_CHECK] self.momentum_predictor.is_trained: {self.momentum_predictor.is_trained}")
                if not self.momentum_predictor.is_trained:
                    print("⚠️ [MOMENTUM_PREDICT_CHECK] Tentativo di ricaricare i modelli per self.momentum_predictor...")
                    self.momentum_predictor.load_models()
                    print(f"🔍 [MOMENTUM_PREDICT_CHECK] Stato dopo ricaricamento: {self.momentum_predictor.is_trained}")

                print("🔍 Calcolo impatto momentum...")
                try:
                    # Ottieni gli ID delle squadre
                    home_team_id = game_data.get('team_stats', {}).get('home', {}).get('id')
                    away_team_id = game_data.get('team_stats', {}).get('away', {}).get('id')
                    home_team_name = game_data.get('team_stats', {}).get('home', {}).get('team_name', 'Squadra Casa')
                    away_team_name = game_data.get('team_stats', {}).get('away', {}).get('team_name', 'Squadra Ospite')
                    
                    # Prova a recuperare i roster dai report infortuni
                    # Assicurati che 'players' contenga una lista di dizionari, non un DataFrame
                    home_roster_list = game_data.get('injury_reports', {}).get('home_nba', {}).get('players', [])
                    away_roster_list = game_data.get('injury_reports', {}).get('away_nba', {}).get('players', [])

                    # Se i roster sono DataFrame, convertili in liste di dizionari
                    if isinstance(home_roster_list, pd.DataFrame):
                        home_roster_list = home_roster_list.to_dict(orient='records')
                    if isinstance(away_roster_list, pd.DataFrame):
                        away_roster_list = away_roster_list.to_dict(orient='records')

                    
                    # Se i roster sono vuoti, prova a recuperarli in altro modo
                    if not home_roster and home_team_id:
                        print(f"\n🔍 Recupero roster per {home_team_name} (ID: {home_team_id})...")
                        home_roster = self._get_team_roster(home_team_id, home_team_name)
                    
                    if not away_roster and away_team_id:
                        print(f"🔍 Recupero roster per {away_team_name} (ID: {away_team_id})...")
                        away_roster = self._get_team_roster(away_team_id, away_team_name)
                    
                    # Debug: Stampa la struttura dei roster
                    print("\n🔍 DEBUG - Roster squadre:")
                    print(f"- {home_team_name}: {len(home_roster_list)} giocatori")
                    print(f"- {away_team_name}: {len(away_roster_list)} giocatori")
                    
                    if home_roster_list:
                        print(f"\n🔍 Primi 3 giocatori {home_team_name}:")
                        for i, p in enumerate(home_roster_list[:3]):
                            print(f"  {i+1}. {p.get('name', 'N/A')} (ID: {p.get('id', 'N/A')}, Ruolo: {p.get('position', 'N/A')}, Min: {p.get('min', 'N/A')})")
                    
                    if away_roster_list:
                        print(f"\n🔍 Primi 3 giocatori {away_team_name}:")
                        for i, p in enumerate(away_roster_list[:3]):
                            print(f"  {i+1}. {p.get('name', 'N/A')} (ID: {p.get('id', 'N/A')}, Ruolo: {p.get('position', 'N/A')}, Min: {p.get('min', 'N/A')})")
                    
                    # Converti in DataFrame
                    home_roster_df = pd.DataFrame(home_roster_list) if home_roster_list else pd.DataFrame()
                    away_roster_df = pd.DataFrame(away_roster_list) if away_roster_list else pd.DataFrame()
                    
                    # Debug: Stampa le colonne dei DataFrame
                    if not home_roster_df.empty:
                        print("\n🔍 Colonne home_roster_df:", home_roster_df.columns.tolist())
                    if not away_roster_df.empty:
                        print("🔍 Colonne away_roster_df:", away_roster_df.columns.tolist())
                    
                    # Calcola l'impatto del momentum
                    if not home_roster_df.empty and not away_roster_df.empty:
                        try:
                            print(f"\n🔍 [MOMENTUM_PREDICT] Chiamata a predict_team_momentum_impact per squadra di casa. Roster df empty: {home_roster_df.empty}")
                            if not home_roster_df.empty:
                                home_momentum = self.momentum_predictor.predict_team_momentum_impact(home_roster_df)
                            else:
                                home_momentum = 0.0
                            print(f"🔍 [MOMENTUM_PREDICT] Chiamata a predict_team_momentum_impact per squadra ospite. Roster df empty: {away_roster_df.empty}")
                            if not away_roster_df.empty:
                                away_momentum = self.momentum_predictor.predict_team_momentum_impact(away_roster_df)
                            else:
                                away_momentum = 0.0
                            momentum_impact = home_momentum - away_momentum
                            print(f"📊 Impatto momentum calcolato: {momentum_impact:.2f} (casa: {home_momentum:.2f}, trasferta: {away_momentum:.2f})")
                        except Exception as e:
                            print(f"❌ Errore durante il calcolo del momentum: {e}")
                            import traceback
                            traceback.print_exc()
                            momentum_impact = 0.0
                    else:
                        print("⚠️ Uno o entrambi i roster DataFrame sono vuoti, impatto momentum impostato a 0")
                        if home_roster_df.empty: print("   - Roster casa DataFrame vuoto.")
                        if away_roster_df.empty: print("   - Roster trasferta DataFrame vuoto.")
                        momentum_impact = 0.0
                except Exception as e:
                    print(f"❌ Errore nel calcolo dell'impatto momentum: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("⚠️ Momentum predictor non disponibile, impatto momentum impostato a 0")
            
            # Aggiungi l'impatto del momentum a game_data
            if 'momentum_impact' not in game_data:
                game_data['momentum_impact'] = {}
            game_data['momentum_impact']['momentum_impact'] = momentum_impact
            
            # Verifica che probabilistic_system sia inizializzato
            if not self.probabilistic_system:
                print("❌ ERRORE CRITICO: self.probabilistic_system non è inizializzato prima della chiamata ad analyze_betting_opportunities_with_injuries.")
                # Potresti voler gestire questo caso in modo più robusto, es. con una predizione di base
                return self._basic_prediction(game_data, momentum_factor if 'momentum_factor' in locals() else 1)

            analysis_result = self.probabilistic_system.analyze_betting_opportunities_with_injuries(game_data, manual_base_line)
            

            
            if analysis_result:
                # Usa best_opportunities per la logica principale
                best_opportunities = analysis_result.get('best_opportunities', [])
                all_opportunities = analysis_result.get('betting_opportunities', [])
                
                # Prendi la migliore opportunità se disponibile
                best_opportunity = best_opportunities[0] if best_opportunities else None
                
                # Calcola la confidenza complessiva della predizione se c'è un'opportunità
                confidence = self._calculate_prediction_confidence(game_data, best_opportunity) if best_opportunity else {'level': 'NO BET'}
                
                # Stampa la tabella delle quote se disponibile
                if all_opportunities:
                    print("\n📊 TABELLA QUOTE (ordinate per edge decrescente)")
                    print("-" * 100)
                    print(f"{'Linea':<8} {'Tipo':<8} {'Quota':<8} {'Prob.':<10} {'Edge':<12} {'Stake':<10} {'Valido'}")
                    print("-" * 100)
                    
                    for quote in all_opportunities:
                        edge_color = '\033[92m' if quote.get('valido', False) else '\033[91m'
                        valido_str = '✅ SI' if quote.get('valido', False) else '❌ NO'
                        reset_color = '\033[0m'
                        
                        print(f"{quote['linea']:<8.1f} {quote['tipo']:<8} {quote['quota']:<8.2f} "
                              f"{quote['probabilita']*100:>5.1f}%  {edge_color}{quote['edge']*100:>6.1f}%{reset_color}  "
                              f"€{quote.get('stake', 0.0):<8.2f} {edge_color}{valido_str}{reset_color}")
                    
                    valide_count = sum(1 for q in all_opportunities if q.get('valido', False))
                    print("-" * 100)
                    if valide_count > 0:
                        print(f"✅ Trovate {valide_count} opportunità con edge > 5%")
                    else:
                        print("⚠️  Nessuna opportunità con edge > 5% trovata")
                    print("\n" + "="*70 + "\n")
                
                # Debug: Stampa le chiavi di analysis_result
                print("\n🔍 DEBUG - Chiavi in analysis_result:", list(analysis_result.keys()))
                
                # Stampa il riepilogo statistico
                print("\n📊 RIEPILOGO STATISTICO")
                print("-" * 50)
                
                # Debug: Stampa la struttura completa di prediction
                print(f"🔍 Struttura di prediction: {analysis_result.get('prediction')}")
                
                # Stampa l'impatto momentum
                if 'momentum_impact' in game_data:
                    momentum_impact = game_data['momentum_impact'].get('momentum_impact', 0)
                    print(f"📊 Impatto momentum nel riepilogo: {momentum_impact:.2f} punti")
                
                # Estrai i valori necessari dalla struttura corretta
                prediction = analysis_result.get('prediction', {})
                
                # Estrai mu e std dalla struttura corretta
                mu = prediction.get('predicted_mu')
                if mu is None:
                    mu = prediction.get('base_mu')
                
                # Prova a ottenere std da predicted_sigma o base_sigma
                std = prediction.get('predicted_sigma')
                if std is None:
                    std = prediction.get('base_sigma')
                
                # Se mu è ancora None, prova a ottenerlo dal game_data
                if mu is None and 'prediction' in game_data and 'mean' in game_data['prediction']:
                    mu = game_data['prediction']['mean']
                    std = game_data['prediction'].get('std', 0)
                
                # Calcola la mu finale
                mu = mu or 0  # Default a 0 se mu è ancora None
                std = std or 0  # Default a 0 se std è ancora None
                
                # Calcola l'impatto degli infortuni
                injury_impact = 0
                if 'injury_reports' in game_data and 'impact_analysis' in game_data['injury_reports']:
                    home_impact = game_data['injury_reports']['impact_analysis'].get('home_team_impact', 0)
                    away_impact = game_data['injury_reports']['impact_analysis'].get('away_team_impact', 0)
                    injury_impact = home_impact + away_impact
                    print(f"🔍 Impatto infortuni - home: {home_impact}, away: {away_impact}, totale: {injury_impact}")
                
                # Calcola l'impatto del momentum (se disponibile)
                momentum_impact = game_data.get('momentum_impact', {}).get('momentum_impact', 0)
                
                final_mu = mu + injury_impact + momentum_impact
                
                # Linea centrale del bookmaker (se disponibile)
                bookmaker_line = manual_base_line if manual_base_line is not None else 'N/D'
                
                # Stampa i valori calcolati
                print(f"μ probabilistica: {mu:.1f}")
                print(f"Dev. standard: {std:.1f}")
                print(f"Impatto injury report: {injury_impact:+.1f} punti")
                print(f"Impatto momentum: {momentum_impact:+.1f} punti")
                print(f"μ finale: {final_mu:.1f}")
                print(f"Linea centrale bookmaker: {bookmaker_line}")
                
                print("-" * 50 + "\n")
                
                # Stampa il riepilogo finale nel formato richiesto
                print("\n" + "="*70)
                
                # Estrai i nomi delle squadre da game_info se presenti
                game_info = game_data.get('game_info', {})
                home_team = game_info.get('home_team')
                away_team = game_info.get('away_team')
                
                # Se non presenti in game_info, cerca in team_stats
                if not home_team:
                    home_team_stats_dict = game_data.get('team_stats', {}).get('home', {})
                    home_team = home_team_stats_dict.get('team_name') if home_team_stats_dict else None
                if not away_team:
                    away_team_stats_dict = game_data.get('team_stats', {}).get('away', {})
                    away_team = away_team_stats_dict.get('team_name') if away_team_stats_dict else None
                
                # Se ancora non trovati, cerca direttamente in game_data
                if not home_team:
                    home_team = game_data.get('home_team')
                if not away_team:
                    away_team = game_data.get('away_team')
                
                # Se ancora non trovati, prova a cercare nell'oggetto game (se esiste)
                # Nota: l'attributo self.game non è gestito in modo consistente, questo fallback è poco affidabile.
                if (not home_team or not away_team) and hasattr(self, 'game'):
                    current_game_attr = getattr(self, 'game', {})
                    if isinstance(current_game_attr, dict):
                        if not home_team: home_team = current_game_attr.get('home_team')
                        if not away_team: away_team = current_game_attr.get('away_team')
                
                # Se ancora non trovati, usa i valori predefiniti
                home_team = home_team or 'Squadra Casa'
                away_team = away_team or 'Squadra Ospite'
                
                # Assicurati che i nomi non siano vuoti
                home_team = home_team or 'Squadra Casa'
                away_team = away_team or 'Squadra Ospite'
                
                # Ottieni data e ora
                game_date = game_info.get('date', '')
                game_time = game_info.get('time', '') or '--:--'
                
                print(f"🏀 {home_team} vs {away_team}  ")
                print(f"📅 {game_date} 🕒 {game_time}")
                
                if best_opportunity:
                    print(f"🎯 {best_opportunity['tipo']} {best_opportunity['linea']:.1f} (Quota: {best_opportunity['quota']:.2f})")
                    print(f"🕛 Media Punti Stimati (μ): {analysis_result.get('prediction', {}).get('mean', 0):.1f}")
                    print(f"📊 Probabilità stimata: {best_opportunity['probabilita']*100:.1f}%")
                    print(f"📈 Edge atteso: {best_opportunity['edge']*100:.1f}%")
                    print(f"💸 Stake: €{best_opportunity['stake']:.2f}")
                    print(f"🔐 Confidenza: {confidence.get('level', 'MEDIA')}")
                    print(f"🧾 Punteggio finale: ")
                    print(f"🧮 Esito: ")
                else:
                    # Ottieni la media punti dalla predizione o da game_data
                    predicted_mean = (
                        analysis_result.get('prediction', {}).get('mean') or 
                        analysis_result.get('prediction', {}).get('predicted_mu') or 
                        game_data.get('prediction', {}).get('mean', 0)
                    )
                    
                    print("🎯 NESSUNA CONDIZIONE SODDISFACENTE")
                    print(f"🕛 Media Punti Stimati (μ): {float(predicted_mean):.1f}" if predicted_mean else "🕛 Media Punti Stimati (μ): --")
                    print(f"📊 Probabilità stimata: --%")
                    print(f"📈 Edge atteso: --%")
                    print(f"💸 Stake: NO BET")
                    print(f"🔐 Confidenza: {confidence.get('level', 'NO BET')}")
                    print(f"🧾 Punteggio finale: ")
                    print(f"🧮 Esito: ")
                
                print("="*70 + "\n")
                
                return {'analysis_result': analysis_result, 'system_type': 'probabilistic'}
            else:
                print("\n" + "="*70)
                print(f"🏀 {game_data.get('home_team', 'Squadra Casa')} vs {game_data.get('away_team', 'Squadra Ospite')}  ")
                print(f"📅 {game_data.get('date', 'Data non disponibile')} 🕒 {game_data.get('time', 'Ora non disponibile')}  ")
                print("🎯 NO BET")
                print("🔐 Confidenza: NO BET")
                print("="*70 + "\n")
                return {'analysis_result': analysis_result, 'system_type': 'probabilistic_no_bet'}
        except Exception as e:
            print(f"❌ Errore predizione probabilistica: {e}")
            return self._basic_prediction(game_data, momentum_factor)
    
    def _basic_prediction(self, game_data, momentum_factor=1):
        """Predizione di base senza sistema probabilistico"""
        print(f"\n🏀 {game_data['game_info']['home_team']} vs {game_data['game_info']['away_team']}")
        print(f"📅 {game_data['game_info']['date']}")
        print("⚠️ Sistema base - predizione limitata")
        
        # Calcolo approssimativo basato sulle medie punti
        home_pts = game_data['team_stats'].get('home', {}).get('PTS_pg', 110)
        away_pts = game_data['team_stats'].get('away', {}).get('PTS_pg', 110)
        estimated_total = (home_pts + away_pts) * momentum_factor
        
        print(f"📊 Totale stimato: {estimated_total:.1f}")
        
        return {
            'analysis_result': {'estimated_total': estimated_total},
            'system_type': 'basic'
        }
    
    def aggiorna_risultati_pendenti(self):
        """Aggiorna i risultati delle partite pendenti"""
        print("🔄 Aggiornamento risultati pendenti...")
        
        try:
            predictions_file = os.path.join(DATA_DIR, 'nba_predictions_history.csv')
            if not os.path.exists(predictions_file):
                print("ℹ️ Nessun file predizioni trovato")
                return
            
            df = pd.read_csv(predictions_file)
            pending_games = df[df['status'] == 'PENDING_RESULT']
            
            if pending_games.empty:
                print("ℹ️ Nessuna partita pendente")
                return
            
            print(f"🔍 Trovate {len(pending_games)} partite pendenti")
            
            for _, game_row in pending_games.iterrows():
                game_details = {
                    'game_date_str': game_row['GAME_DATE'],
                    'home_team_name': game_row['HOME_TEAM'],
                    'away_team_name': game_row['AWAY_TEAM'],
                    'home_team_id': self._get_team_id(game_row['HOME_TEAM']),
                    'away_team_id': self._get_team_id(game_row['AWAY_TEAM'])
                }
                
                if game_details['home_team_id'] and game_details['away_team_id']:
                    result = self.data_provider._fetch_game_result(game_details)
                    if result:
                        # Aggiorna il CSV con il risultato
                        mask = (
                            (df['GAME_DATE'] == game_row['GAME_DATE']) &
                            (df['HOME_TEAM'] == game_row['HOME_TEAM']) &
                            (df['AWAY_TEAM'] == game_row['AWAY_TEAM'])
                        )
                        df.loc[mask, 'status'] = 'COMPLETED'
                        df.loc[mask, 'HOME_SCORE'] = result['home_score']
                        df.loc[mask, 'AWAY_SCORE'] = result['away_score']
                        df.loc[mask, 'TOTAL_SCORE'] = result['total_score']
                        
                        print(f"✅ Aggiornato: {game_row['HOME_TEAM']} vs {game_row['AWAY_TEAM']} - {result['total_score']} punti")
            
            # Salva il CSV aggiornato
            df.to_csv(predictions_file, index=False)
            print("💾 File predizioni aggiornato")
            
        except Exception as e:
            print(f"❌ Errore aggiornamento risultati: {e}")
    
    def _get_team_id(self, game_data, team_type):
        """Ottieni team_id in modo resiliente"""
        # Cerca in game_info se esiste, altrimenti cerca nella radice
        game_info = game_data.get('game_info', game_data)
        
        if team_type == 'home':
            return (game_info.get('home_team_id') or 
                   game_info.get('home', {}).get('id') or 
                   game_info.get('teams', {}).get('home', {}).get('id') or
                   game_info.get('team_stats', {}).get('home', {}).get('id') or
                   game_info.get('home_team_id'))
        else:
            return (game_info.get('away_team_id') or 
                   game_info.get('away', {}).get('id') or 
                   game_info.get('teams', {}).get('away', {}).get('id') or
                   game_info.get('team_stats', {}).get('away', {}).get('id') or
                   game_info.get('away_team_id'))
        
        # Questa parte è stata sostituita dalla nuova implementazione
        return None
        if team_name:
            # Cerca in una mappa di nomi a ID
            team_map = {
                'Indiana Pacers': 1610612754,
                'Oklahoma City Thunder': 1610612760,
                # ... aggiungi altre squadre ...
            }
            return team_map.get(team_name)
        
        return None
    
    def _get_head_to_head_data(self, home_team, away_team):
        """
        Recupera i dati head-to-head tra due squadre
        
        Args:
            home_team (str): Nome squadra di casa
            away_team (str): Nome squadra in trasferta
            
        Returns:
            dict: Dizionario con statistiche head-to-head
        """
        # Cache key per evitare chiamate ripetute
        cache_key = f"h2h_{home_team}_{away_team}"
        if cache_key in getattr(self, 'h2h_cache', {}):
            return self.h2h_cache[cache_key]
            
        # Inizializza la cache se non esiste
        if not hasattr(self, 'h2h_cache'):
            self.h2h_cache = {}
        
        # Per ora simuliamo i dati in base alla similarità dei nomi
        # In un'implementazione reale, qui si farebbe una chiamata API
        h2h_data = {
            'games': 0,
            'home_wins': 0,
            'away_wins': 0,
            'avg_total_points': 0
        }
        
        try:
            # Controlla se abbiamo dati API-Sports per recuperare i dati reali
            if hasattr(self, 'data_provider') and hasattr(self.data_provider, '_make_apisports_request'):
                # Trova gli ID delle squadre
                home_id = self.data_provider.nba_team_name_to_apisports_id.get(home_team)
                away_id = self.data_provider.nba_team_name_to_apisports_id.get(away_team)
                
                if home_id and away_id:
                    # Ottieni la stagione corrente
                    current_season = self.data_provider._get_apisports_season_param_str(date.today())
                    
                    # Parametri per la richiesta H2H
                    params = {
                        'h2h': f"{home_id}-{away_id}",
                        'season': current_season,
                        'league': '12'  # NBA
                    }
                    
                    print(f"   🔍 Richiesta H2H api-sports per {home_team} vs {away_team}")
                    try:
                        response = self.data_provider._make_apisports_request('games/headtohead', params)
                        
                        if response and 'response' in response:
                            h2h_games = response['response']
                            if h2h_games:
                                games_count = len(h2h_games)
                                home_wins = 0
                                total_points = 0
                                
                                for game in h2h_games:
                                    teams = game.get('teams', {})
                                    scores = game.get('scores', {})
                                    
                                    # Identifica quale squadra è home/away in questa partita
                                    is_home_id = teams.get('home', {}).get('id') == home_id
                                    
                                    home_score = scores.get('home', {}).get('total', 0)
                                    away_score = scores.get('away', {}).get('total', 0)
                                    
                                    total_points += home_score + away_score
                                    
                                    # Conta vittorie home team
                                    if (is_home_id and home_score > away_score) or (not is_home_id and away_score > home_score):
                                        home_wins += 1
                                
                                h2h_data = {
                                    'games': games_count,
                                    'home_wins': home_wins,
                                    'away_wins': games_count - home_wins,
                                    'avg_total_points': total_points / games_count if games_count > 0 else 0
                                }
                                
                                print(f"   ✅ Trovate {games_count} partite H2H: {home_team} {home_wins}W - {games_count - home_wins}W {away_team}")
                    except Exception as e:
                        print(f"   ⚠️ Errore API-Sports H2H: {str(e)}")
            
            # Se non abbiamo dati reali, simula alcuni dati
            if h2h_data['games'] == 0:
                # Simulazione basata sui nomi per testing
                same_conf = any(x in home_team and x in away_team for x in ["East", "West"])
                same_div = any(x in home_team and x in away_team for x in ["Atlantic", "Central", "Southeast", "Northwest", "Pacific", "Southwest"])
                
                if same_div:
                    h2h_data['games'] = 4  # Stessa divisione - 4 partite per stagione
                elif same_conf:
                    h2h_data['games'] = 3  # Stessa conference - 3-4 partite per stagione
                else:
                    h2h_data['games'] = 2  # Conference diverse - 2 partite per stagione
                
                # Simula risultati sulla base delle percentuali di vittoria
                home_win_pct = self.get_team_stat(home_team, 'win_percentage', 0.5)
                away_win_pct = self.get_team_stat(away_team, 'win_percentage', 0.5)
                
                # Aggiusta per vantaggio campo (+5%)
                adjusted_home_win_pct = min(1.0, home_win_pct + 0.05)
                
                # Stima quante partite dovrebbe vincere il team di casa
                home_wins = round(h2h_data['games'] * (adjusted_home_win_pct / (adjusted_home_win_pct + away_win_pct)))
                
                h2h_data['home_wins'] = home_wins
                h2h_data['away_wins'] = h2h_data['games'] - home_wins
                
                # Stima media punti
                home_ppg = self.get_team_stat(home_team, 'points_per_game', 110)
                away_ppg = self.get_team_stat(away_team, 'points_per_game', 110)
                h2h_data['avg_total_points'] = home_ppg + away_ppg
                
                print(f"   ℹ️ Dati H2H stimati: {h2h_data['games']} partite per stagione")
        except Exception as e:
            print(f"   ⚠️ Errore recupero dati H2H: {e}")
            
        # Salva in cache
        self.h2h_cache[cache_key] = h2h_data
        return h2h_data
    
    def get_team_stat(self, team_name, stat_key, default_value=0):
        """Helper per recuperare statistiche di squadra"""
        if hasattr(self, 'data_provider'):
            # Cerca prima in cache
            cache_key = f"team_stats_{team_name}"
            if hasattr(self.data_provider, 'team_data_cache') and cache_key in self.data_provider.team_data_cache:
                return self.data_provider.team_data_cache[cache_key].get(stat_key, default_value)
        
        # Fallback a valore di default
        return default_value


    
    def _calculate_momentum_impact(self, home_roster, away_roster):
        """
        Calcola l'impatto del momentum per entrambe le squadre.
        
        Args:
            home_roster: DataFrame con i giocatori della squadra di casa
            away_roster: DataFrame con i giocatori della squadra in trasferta
            
        Returns:
            dict: Dizionario con i risultati del calcolo del momentum
        """
        try:
            print(f"\n🔍 [MOMENTUM] Inizio calcolo impatto momentum...")
            
            # Default result in case of errors
            default_result = {
                'home_momentum': {'momentum_score': 0.0, 'error': 'Errore nel calcolo'},
                'away_momentum': {'momentum_score': 0.0, 'error': 'Errore nel calcolo'},
                'momentum_impact': 0.0,
                'momentum_score': 0.0,
                'error': None
            }
            
            # Verifica che i roster non siano vuoti
            if home_roster is None or away_roster is None:
                error_msg = "Uno o entrambi i roster sono None"
                print(f"⚠️ [MOMENTUM] {error_msg}")
                return {**default_result, 'error': error_msg}
                
            if home_roster.empty or away_roster.empty:
                error_msg = "Uno o entrambi i roster sono vuoti"
                print(f"⚠️ [MOMENTUM] {error_msg}")
                if home_roster.empty:
                    print("   - Roster casa vuoto")
                if away_roster.empty:
                    print("   - Roster trasferta vuoto")
                return {**default_result, 'error': error_msg}
            
            # Log delle dimensioni dei roster
            print(f"   [MOMENTUM] Dimensione roster casa: {len(home_roster)} giocatori")
            print(f"   [MOMENTUM] Dimensione roster trasferta: {len(away_roster)} giocatori")
            
            # Log delle prime righe per debug
            print("   [MOMENTUM] Anteprima roster casa (primi 3):")
            for i, (_, row) in enumerate(home_roster.head(3).iterrows(), 1):
                print(f"      {i}. {row.get('PLAYER_NAME', 'N/A')} - {row.get('POSITION', 'N/A')} - MIN: {row.get('MIN', 0)} - STATUS: {row.get('ROTATION_STATUS', 'N/A')}")
            
            print("   [MOMENTUM] Anteprima roster trasferta (primi 3):")
            for i, (_, row) in enumerate(away_roster.head(3).iterrows(), 1):
                print(f"      {i}. {row.get('PLAYER_NAME', 'N/A')} - {row.get('POSITION', 'N/A')} - MIN: {row.get('MIN', 0)} - STATUS: {row.get('ROTATION_STATUS', 'N/A')}")
            
            # Calcola il momentum per entrambe le squadre
            try:
                print("   [MOMENTUM] Calcolo momentum per la squadra di casa...")
                home_momentum = self.momentum_predictor.predict_team_momentum_impact(home_roster)
                print(f"   [MOMENTUM] Momentum calcolato per la squadra di casa: {home_momentum.get('momentum_score', 0):.2f}")
            except Exception as e:
                error_msg = f"Errore nel calcolo del momentum per la squadra di casa: {str(e)}"
                print(f"❌ [MOMENTUM] {error_msg}")
                home_momentum = {'momentum_score': 0.0, 'error': error_msg}
            
            try:
                print("   [MOMENTUM] Calcolo momentum per la squadra in trasferta...")
                away_momentum = self.momentum_predictor.predict_team_momentum_impact(away_roster)
                print(f"   [MOMENTUM] Momentum calcolato per la squadra in trasferta: {away_momentum.get('momentum_score', 0):.2f}")
            except Exception as e:
                error_msg = f"Errore nel calcolo del momentum per la squadra in trasferta: {str(e)}"
                print(f"❌ [MOMENTUM] {error_msg}")
                away_momentum = {'momentum_score': 0.0, 'error': error_msg}
            
            # Estrai i punteggi di momentum con gestione degli errori
            home_score = home_momentum.get('momentum_score', 0) if isinstance(home_momentum, dict) else 0.0
            away_score = away_momentum.get('momentum_score', 0) if isinstance(away_momentum, dict) else 0.0
            
            # Calcola la differenza di momentum
            momentum_diff = home_score - away_score
            
            print(f"\n📊 [MOMENTUM] Risultati Momentum:")
            print(f"   Casa: {home_score:.2f}")
            print(f"   Trasferta: {away_score:.2f}")
            print(f"   Differenza: {momentum_diff:+.2f}")
            
            # Prepara il risultato con tutte le informazioni rilevanti
            result = {
                'home_momentum': home_momentum if isinstance(home_momentum, dict) else {'momentum_score': 0.0, 'error': 'Formato non valido'},
                'away_momentum': away_momentum if isinstance(away_momentum, dict) else {'momentum_score': 0.0, 'error': 'Formato non valido'},
                'momentum_impact': momentum_diff,
                'momentum_score': momentum_diff,  # Per retrocompatibilità
                'home_players_count': len(home_roster),
                'away_players_count': len(away_roster),
                'timestamp': datetime.now().isoformat(),
                'error': None
            }
            
            # Aggiungi dettagli sugli infortuni se disponibili
            if isinstance(home_momentum, dict) and 'injury_reports' in home_momentum:
                result['home_injuries'] = home_momentum['injury_reports']
            if isinstance(away_momentum, dict) and 'injury_reports' in away_momentum:
                result['away_injuries'] = away_momentum['injury_reports']
            
            return result
            
        except Exception as e:
            error_msg = f"Errore nel calcolo del momentum: {str(e)}"
            print(f"❌ [MOMENTUM] {error_msg}")
            import traceback
            traceback.print_exc()
            
            return {
                'home_momentum': {'momentum_score': 0.0, 'error': error_msg},
                'away_momentum': {'momentum_score': 0.0, 'error': error_msg},
                'momentum_impact': 0.0,
                'momentum_score': 0.0,
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
    
    def _get_team_roster(self, team_id, team_name):
        """
        Recupera il roster di una squadra utilizzando InjuryReporter e lo converte in DataFrame
        con il formato corretto per il momentum predictor.
        
        Args:
            team_id: ID della squadra
            team_name: Nome della squadra (usato per i log e per il recupero degli infortuni)
            
        Returns:
            pd.DataFrame: DataFrame con i giocatori del roster e le colonne necessarie
        """
        try:
            print(f"\n🔍 [ROSTER] Inizio recupero roster per {team_name} (ID: {team_id})")
            
            # Verifichiamo se abbiamo un'istanza di InjuryReporter
            if not (hasattr(self, 'data_provider') and hasattr(self.data_provider, 'injury_reporter')):
                error_msg = "InjuryReporter non disponibile in data_provider"
                print(f"❌ [ROSTER] {error_msg}")
                return pd.DataFrame(columns=['PLAYER_ID', 'PLAYER_NAME', 'POSITION', 'MIN', 'TEAM_ID', 'ROTATION_STATUS', 'INJURY_STATUS'])
            
            # Recupera il roster con gli infortuni
            print(f"   [ROSTER] Recupero dati roster e infortuni per {team_name}...")
            roster = self.data_provider.injury_reporter.get_team_roster(team_id)
            
            if not roster:
                error_msg = f"Nessun giocatore trovato per {team_name} (ID: {team_id})"
                print(f"⚠️ [ROSTER] {error_msg}")
                return pd.DataFrame(columns=['PLAYER_ID', 'PLAYER_NAME', 'POSITION', 'MIN', 'TEAM_ID', 'ROTATION_STATUS', 'INJURY_STATUS'])
            
            print(f"✅ [ROSTER] Trovati {len(roster)} giocatori per {team_name}")
            
            # Converti la lista di dizionari in DataFrame
            roster_df = pd.DataFrame(roster)
            
            # Mappatura dei campi dal formato di InjuryReporter a quello atteso
            column_mapping = {
                'id': 'PLAYER_ID',
                'name': 'PLAYER_NAME',
                'position': 'POSITION',
                'min': 'MIN',
                'team_id': 'TEAM_ID',
                'status': 'INJURY_STATUS',
                'rotation_status': 'ROTATION_STATUS'
            }
            
            # Rinomina le colonne esistenti e crea quelle mancanti
            for old_col, new_col in column_mapping.items():
                if old_col in roster_df.columns:
                    roster_df.rename(columns={old_col: new_col}, inplace=True)
                elif new_col not in roster_df.columns:
                    # Imposta valori di default per le colonne mancanti
                    if new_col == 'MIN':
                        roster_df[new_col] = 0  # Minuti giocati, default 0
                    elif new_col == 'ROTATION_STATUS':
                        # Prova a dedurre lo stato di rotazione in base ai minuti giocati
                        roster_df[new_col] = roster_df.get('MIN', 0).apply(
                            lambda x: 'Starter' if x > 20 else 'Bench' if x > 0 else 'Inactive'
                        )
            
            # Assicurati che le colonne richieste esistano
            required_columns = ['PLAYER_ID', 'PLAYER_NAME', 'POSITION', 'MIN', 'TEAM_ID', 'ROTATION_STATUS', 'INJURY_STATUS']
            for col in required_columns:
                if col not in roster_df.columns:
                    if col == 'INJURY_STATUS':
                        roster_df[col] = 'active'  # Default a 'active' se lo stato infortunio non è specificato
                    else:
                        roster_df[col] = ''  # Stringa vuota per le altre colonne mancanti
            
            # Seleziona solo le colonne necessarie nell'ordine corretto
            roster_df = roster_df[required_columns]
            
            # Log per debug
            print(f"   [ROSTER] Roster elaborato per {team_name}:")
            print(f"      - Giocatori totali: {len(roster_df)}")
            print(f"      - Infortunati: {len(roster_df[roster_df['INJURY_STATUS'] != 'active'])}")
            print(f"      - Starter: {len(roster_df[roster_df['ROTATION_STATUS'] == 'Starter'])}")
            print(f"      - Bench: {len(roster_df[roster_df['ROTATION_STATUS'] == 'Bench'])}")
            
            return roster_df
            
        except Exception as e:
            error_msg = f"Errore durante il recupero del roster per {team_name} (ID: {team_id}): {str(e)}"
            print(f"❌ [ROSTER] {error_msg}")
            import traceback
            traceback.print_exc()
            
            # Crea un DataFrame vuoto con tutte le colonne necessarie
            required_columns = ['PLAYER_ID', 'PLAYER_NAME', 'POSITION', 'MIN', 'TEAM_ID', 'ROTATION_STATUS', 'INJURY_STATUS']
            empty_df = pd.DataFrame(columns=required_columns)
            
            # Aggiungi il messaggio di errore ai metadati del DataFrame
            empty_df.attrs['error'] = error_msg
            empty_df.attrs['traceback'] = traceback.format_exc()
            
            return empty_df

    def _calculate_kelly_stake(self, edge, prob, odds, min_bet=1.0):
        """
        Calcola lo stake secondo Kelly frazionato dinamico
        
        Args:
            edge: Edge percentuale (0.15 per 15%)
            prob: Probabilità stimata (0.65 per 65%)
            odds: Quota decimale (es. 1.95)
            min_bet: Puntata minima (default 1.0€)
            
        Returns:
            float: Stake calcolato
        """
        # Filtro qualità - NO BET se edge < 10%
        if edge < 0.10:
            return 0.0
            
        # Caso 1: Edge < 15% E prob < 65% → Stake fisso 1€
        if edge < 0.15 and prob < 0.65:
            return min_bet
            
        # Caso 2: Edge < 15% OPPURE prob < 65% → Kelly 25% con cap 2.5%
        elif edge < 0.15 or prob < 0.65:
            kelly_fraction = (prob * odds - 1) / (odds - 1)  # Formula Kelly corretta
            stake_raw = self.bankroll * kelly_fraction * 0.25  # Frazionamento al 25%
            max_stake = self.bankroll * 0.025  # Cap al 2.5%
            stake = min(stake_raw, max_stake)
            
        # Caso 3: Edge ≥ 15% E prob ≥ 65% → Kelly 33% con cap 5%
        else:
            kelly_fraction = (prob * odds - 1) / (odds - 1)  # Formula Kelly corretta
            stake_raw = self.bankroll * kelly_fraction * 0.33  # Frazionamento al 33%
            max_stake = self.bankroll * 0.05  # Cap al 5%
            stake = min(stake_raw, max_stake)
        
        # Arrotondamento e verifica minimo
        stake = round(max(stake, min_bet), 1)
        return stake
    
    def _test_momentum_models(self, data_provider, nba_system, max_games=2):
        """
        Testa le predizioni di momentum sulle prossime partite
        
        Args:
            data_provider: Istanza del provider dei dati
            nba_system: Istanza del sistema NBA
            max_games: Numero massimo di partite da testare
        """
        print("🧪 Test predizioni momentum...")
        scheduled_games = data_provider.get_scheduled_games(days_ahead=1)
        
        if not scheduled_games:
            print(" Nessuna partita disponibile per test")
            return
            
        for game in scheduled_games[:max_games]:
            game_data = data_provider.get_injury_adjusted_data_for_game(game)
            if not game_data:
                continue
                
            print("\n Dopo get_injury_adjusted_data_for_game:")
            print(f" ID oggetto game_data: {id(game_data)}")
            print(f" Data in game_data: {game_data.get('game_info', {}).get('date', 'N/D')}")
            print(f" Squadre in game_data: {game_data.get('game_info', {}).get('home_team', 'N/D')} vs {game_data.get('game_info', {}).get('away_team', 'N/D')}")
            print(f" Chiavi in game_data: {list(game_data.keys())}")
            
            # Verifica se team_stats è presente
            print(f" team_stats in game_data: {'presente' if 'team_stats' in game_data else 'assente'}")
            if 'team_stats' in game_data:
                print(f"   - Chiavi in team_stats: {list(game_data['team_stats'].keys())}")
                if 'home' in game_data['team_stats']:
                    print(f"   - home team stats: {list(game_data['team_stats']['home'].keys())}")
                if 'away' in game_data['team_stats']:
                    print(f"   - away team stats: {list(game_data['team_stats']['away'].keys())}")
            
            print(f"\n Test: {game_data['game_info']['away_team']} @ {game_data['game_info']['home_team']}")
            result = nba_system.predict_complete_game_flow(
                game_data, 
                manual_base_line=central_line if central_line is not None else None
            )
            time.sleep(2)
    
    def _train_momentum_models(self):
        """Esegue il training dei modelli di momentum"""
        print("🤖 Training modelli momentum...")
        from player_momentum_predictor import PlayerMomentumPredictor
        predictor = PlayerMomentumPredictor(data_provider)
        success = predictor.train_momentum_models(df_training)
        if success:
            print("✅ Training completato!")
        else:
            print("❌ Errore durante training")
        return success

def main():
    parser = argparse.ArgumentParser(description='Sistema NBA Probabilistico v3.0 con Injury Report e Momentum ML')
    
    # Argomenti esistenti
    parser.add_argument('--addestra-probabilistico', action='store_true', help='Addestra i modelli probabilistici')
    parser.add_argument('--analizza-prossime-partite', action='store_true', help='Analizza le prossime partite')
    parser.add_argument('--giorni', type=int, default=1, help='Numero di giorni da analizzare (default: 1)')
    parser.add_argument('--simula', action='store_true', help='Esegui simulazione completa')
    parser.add_argument('--input-manuale-generale', type=str, help='File JSON con dati manuali generali')
    parser.add_argument('--training-csv', type=str, default='nba_data_with_mu_sigma_for_ml.csv', 
                       help='Nome file CSV per training')
    parser.add_argument('--aggiorna-risultati', action='store_true', help='Aggiorna i risultati delle partite pendenti')
    parser.add_argument('--set-bankroll', type=float, help='Imposta manualmente il bankroll iniziale')
    
    # NUOVO: Argomenti per momentum ML
    parser.add_argument('--setup-momentum', action='store_true', 
                       help='Configura sistema momentum ML (raccolta dati e training)')
    parser.add_argument('--train-momentum', action='store_true', 
                       help='Addestra solo modelli momentum (richiede dati esistenti)')
    parser.add_argument('--test-momentum', action='store_true', 
                       help='Testa predizioni momentum su partite recenti')
    parser.add_argument('--linea-centrale', type=float, help='Linea centrale per generare le quote multiple')
    parser.add_argument('--force-in-season', action='store_true', help="Forza il sistema a comportarsi come in stagione")

    args = parser.parse_args()
    
    # Inizializza il data provider e il sistema
    data_provider = NBADataProvider()
    nba_system = NBACompleteSystem(data_provider=data_provider)
    
    # Gestione comandi di sistema
    if args.set_bankroll is not None:
        nba_system.set_bankroll(args.set_bankroll)
        print(f"💰 Bankroll impostato a {nba_system.get_current_bankroll():.2f}€. Riavviare per altre operazioni.")
        return
    
    # NUOVO: Gestione comandi momentum
    if args.setup_momentum:
        print("🚀 Setup sistema momentum ML...")
        from player_momentum_predictor import PlayerMomentumPredictor
        predictor = PlayerMomentumPredictor(data_provider)
        
        # Pipeline completa
        print("📊 Fase 1: Raccolta dati giocatori...")
        df = predictor.collect_player_data_for_seasons(max_players_per_season=150)
        if df is None:
            print("❌ Errore raccolta dati")
            return
        
        print("🔄 Fase 2: Creazione dataset training...")
        df_training = predictor.create_training_dataset(df)
        if df_training is None:
            print("❌ Errore creazione dataset")
            return
        
        print("🤖 Fase 3: Training modelli...")
        success = predictor.train_momentum_models(df_training)
        if success:
            print("✅ Setup momentum completato!")
        else:
            print("❌ Errore durante training")
        return
    
    def _train_momentum_models(self):
        """Esegue il training dei modelli di momentum"""
        print("🤖 Training modelli momentum...")
        from player_momentum_predictor import PlayerMomentumPredictor
        predictor = PlayerMomentumPredictor(data_provider)
        success = predictor.train_momentum_models(df_training)
        if success:
            print("✅ Training completato!")
        else:
            print("❌ Errore durante training")
        return success

    def _test_momentum_models(self, data_provider, nba_system, max_games=2):
        """
        Testa le predizioni di momentum sulle prossime partite
        
        Args:
            data_provider: Istanza del provider dei dati
            nba_system: Istanza del sistema NBA
            max_games: Numero massimo di partite da testare
        """
        print("🧪 Test predizioni momentum...")
        scheduled_games = data_provider.get_scheduled_games(days_ahead=1)
        
        if not scheduled_games:
            print(" Nessuna partita disponibile per test")
            return
            
        for game in scheduled_games[:max_games]:
            game_data = data_provider.get_injury_adjusted_data_for_game(game)
            if not game_data:
                continue
                
            print("\n Dopo get_injury_adjusted_data_for_game:")
            print(f" ID oggetto game_data: {id(game_data)}")
            print(f" Data in game_data: {game_data.get('game_info', {}).get('date', 'N/D')}")
            print(f" Squadre in game_data: {game_data.get('game_info', {}).get('home_team', 'N/D')} vs {game_data.get('game_info', {}).get('away_team', 'N/D')}")
            print(f" Chiavi in game_data: {list(game_data.keys())}")
            
            # Verifica se team_stats è presente
            print(f" team_stats in game_data: {'presente' if 'team_stats' in game_data else 'assente'}")
            if 'team_stats' in game_data:
                print(f"   - Chiavi in team_stats: {list(game_data['team_stats'].keys())}")
                if 'home' in game_data['team_stats']:
                    print(f"   - home team stats: {list(game_data['team_stats']['home'].keys())}")
                if 'away' in game_data['team_stats']:
                    print(f"   - away team stats: {list(game_data['team_stats']['away'].keys())}")
            
            print(f"\n Test: {game_data['game_info']['away_team']} @ {game_data['game_info']['home_team']}")
            result = nba_system.predict_complete_game_flow(
                game_data, 
                manual_base_line=central_line if central_line is not None else None
            )
            time.sleep(2)
    
    # Gestione comandi da riga di comando
    if args.train_momentum:
        _train_momentum_models()
        sys.exit(0)
    
    if args.test_momentum:
        _test_momentum_models(data_provider, nba_system)
        sys.exit(0)
    
    # Gestione comandi esistenti
    if args.aggiorna_risultati:
        print(" Avvio aggiornamento risultati pendenti...")
        nba_system.aggiorna_risultati_pendenti()
    elif args.addestra_probabilistico:
        nba_system.probabilistic_system.train_probabilistic_models(args.training_csv)
    elif args.analizza_prossime_partite:
        if not args.force_in_season and is_off_season():
            scheduled_games = get_test_games_for_off_season()
        else:
            scheduled_games = data_provider.get_scheduled_games()

        if not scheduled_games:
            print(" Nessuna partita programmata trovata")
            return

        manual_overrides = {}
        if args.input_manuale_generale and os.path.exists(args.input_manuale_generale):
            with open(args.input_manuale_generale, 'r', encoding='utf-8') as f:
                manual_overrides = json.load(f)

        for game_details in scheduled_games:
            try:
                print("\n" + "="*80)
                print(" INIZIO ELABORAZIONE PARTITA")
                print(f" Data partita: {game_details.get('date', 'N/D')}")
                print(f" {game_details.get('home_team', 'N/D')} vs {game_details.get('away_team', 'N/D')}")
                print(f" ID partita: {game_details.get('game_id', 'N/D')}")
                print(f" ID oggetto game_details: {id(game_details)}")
                print(" Contenuto game_details:", {k: v for k, v in game_details.items() if k not in ['odds', 'settings']})
                game_data = data_provider.get_injury_adjusted_data_for_game(game_details, manual_overrides)
                
                print("\n Dopo get_injury_adjusted_data_for_game:")
                print(f" ID oggetto game_data: {id(game_data)}")
                print(f" Data in game_data: {game_data.get('game_info', {}).get('date', 'N/D')}")
                print(f" Squadre in game_data: {game_data.get('game_info', {}).get('home_team', 'N/D')} vs {game_data.get('game_info', {}).get('away_team', 'N/D')}")
                print(f" Chiavi in game_data: {list(game_data.keys())}")
                
                # Verifica se team_stats è presente
                print(f" team_stats in game_data: {'presente' if 'team_stats' in game_data else 'assente'}")
                if 'team_stats' in game_data:
                    print(f"   - Chiavi in team_stats: {list(game_data['team_stats'].keys())}")
                    if 'home' in game_data['team_stats']:
                        print(f"   - home team stats: {list(game_data['team_stats']['home'].keys())}")
                    if 'away' in game_data['team_stats']:
                        print(f"   - away team stats: {list(game_data['team_stats']['away'].keys())}")
            except Exception as e:
                print(f" Errore elaborazione partita: {e}")
                continue
            
            if not game_data:
                continue
            
            # Usa la linea centrale se specificata, altrimenti usa le quote esistenti
            central_line = args.linea_centrale
            
            # Se non è stata specificata una linea centrale, usa quella delle quote esistenti
            if central_line is None and 'odds' in game_details and game_details['odds']:
                central_line = game_details['odds'][0]['line']
            
            # Esegui la predizione con la linea centrale
            prediction = nba_system.predict_complete_game_flow(game_data, manual_base_line=central_line if central_line is not None else None)
            time.sleep(2)
    elif args.simula:
        # Implementa la simulazione completa
        print("🚀 Avvio simulazione completa...")
        # Qui andrebbe il codice per la simulazione
        future_games = data_provider.get_scheduled_games(days_ahead=1)
        for game in future_games:
            game_data = data_provider.get_injury_adjusted_data_for_game(
                game['home_team'], 
                game['away_team'], 
                game_date=game['game_date']
            )
            
            if not game_data:
                print(f"❌ Impossibile analizzare {game['home_team']} vs {game['away_team']}")
                continue
                
            # Aggiungi le quote se disponibili
            if 'odds' in game:
                game_data['odds'] = game['odds']
            
            # Usa la linea centrale specificata o quella delle quote esistenti
            central_line = args.linea_centrale
            if central_line is None and 'odds' in game and game['odds']:
                central_line = game['odds'][0]['line']
            
            # Esegui la predizione con la linea centrale
            prediction = nba_system.predict_complete_game_flow(
                game_data, 
                manual_base_line=central_line if central_line is not None else None
            )
            time.sleep(2)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()