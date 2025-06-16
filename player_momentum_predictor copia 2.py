# player_momentum_predictor.py
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# --- Import delle dipendenze ML necessarie ---
# È buona pratica dichiarare le dipendenze opzionali all'inizio
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor

class PlayerMomentumPredictor:
    """
    Sistema per calcolare il momentum di una squadra NBA basandosi sulle performance recenti dei giocatori.
    Questa classe si occupa solo del calcolo del momentum, non del training o della raccolta dati.
    """
    
    def __init__(self, models_dir='models', nba_data_provider=None):
        """
        Inizializza il predittore di momentum.

        Args:
            models_dir (str): Directory base dove sono salvati i modelli.
            nba_data_provider: Istanza del provider dati per accedere ai game log dei giocatori.
        """
        print("\n🔍 [MOMENTUM] Inizializzazione PlayerMomentumPredictor...")
        
        self.models_dir = os.path.join(models_dir, 'player_momentum')
        self.nba_data_provider = nba_data_provider
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.is_trained = False

        # Pesi per il calcolo del punteggio di momentum, più semplici e robusti
        self.feature_weights = {
            'points_trend': 0.35,
            'plus_minus_trend': 0.30,
            'consistency': 0.20,
            'recent_form': 0.15,
        }
        
        # Pesi per l'importanza di un giocatore nella rotazione
        self.rotation_status_weights = {
            'STARTER': 1.0,
            'BENCH': 0.7,
            'RESERVE': 0.4,
            'INACTIVE': 0.1,
            'UNKNOWN': 0.3,
        }
        
        # Pesi offensivi/difensivi basati sulla posizione
        self.position_weights = {
            'PG': {'off': 0.8, 'def': 0.2}, 'SG': {'off': 0.7, 'def': 0.3}, 'G': {'off': 0.75, 'def': 0.25},
            'SF': {'off': 0.6, 'def': 0.4}, 'PF': {'off': 0.5, 'def': 0.5}, 'F': {'off': 0.55, 'def': 0.45},
            'C': {'off': 0.4, 'def': 0.6}, 'F-C': {'off': 0.45, 'def': 0.55}, 'G-F': {'off': 0.65, 'def': 0.35},
            'UNKNOWN': {'off': 0.5, 'def': 0.5}
        }

        # Carica i modelli all'avvio
        if not self.load_models():
             print("⚠️ [MOMENTUM] Modelli non caricati. Il sistema funzionerà con calcoli basati su regole.")
    
    def load_models(self):
        """Carica modelli, scaler e feature list da un file pickle."""
        models_file = os.path.join(self.models_dir, 'momentum_models.pkl')
        if not os.path.exists(models_file):
            print(f"   [MOMENTUM] File modelli non trovato in: {models_file}")
            return False
        
        try:
            with open(models_file, 'rb') as f:
                data = pickle.load(f)
            
            self.models = data.get('models', {})
            self.scalers = data.get('scalers', {})
            self.feature_columns = data.get('feature_columns', [])
            self.is_trained = data.get('is_trained', False)
            
            if self.is_trained:
                print(f"✅ [MOMENTUM] Modelli caricati con successo. Modelli: {list(self.models.keys())}")
            else:
                print("⚠️ [MOMENTUM] File modello caricato, ma risulta non addestrato.")
            
            return True
        except Exception as e:
            print(f"❌ [MOMENTUM] Errore critico nel caricamento dei modelli: {e}")
            self.is_trained = False
            return False

    def _calculate_trend(self, values, window=5):
        """Calcola la tendenza come pendenza della retta di regressione lineare."""
        if not isinstance(values, list) or len(values) < 2:
            return 0.0
        
        y = np.array(values[-window:])
        if len(y) < 2:
            return 0.0
            
        x = np.arange(len(y))
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope if not np.isnan(slope) else 0.0

    def _calculate_consistency(self, values, window=10):
        """Calcola la consistenza (1 - coefficiente di variazione). Più alto è, meglio è."""
        if not isinstance(values, list) or len(values) < 2:
            return 0.5 # Valore neutro
            
        recent_values = np.array(values[-window:])
        mean = np.mean(recent_values)
        std_dev = np.std(recent_values)
        
        if mean == 0:
            return 0.0
            
        cv = std_dev / mean
        return max(0, 1 - cv) # Il punteggio è tra 0 e 1

    def _get_player_momentum_score(self, player_id, player_name, last_n_games=15):
        """
        Calcola il punteggio di momentum per un singolo giocatore basandosi su regole e statistiche recenti.
        Restituisce un valore da 0 a 100.
        """
        if not self.nba_data_provider:
            print(f"⚠️ [MOMENTUM] Data provider non disponibile per {player_name}. Restituito punteggio neutro.")
            return 50.0

        try:
            # 1. Recupero Game Logs
            game_logs = self.nba_data_provider.get_player_game_logs(player_id, last_n_games=last_n_games)
            if game_logs is None or game_logs.empty or len(game_logs) < 3:
                # print(f"ℹ️ [MOMENTUM] Dati insufficienti per {player_name}. Punteggio neutro.")
                return 50.0

            # 2. Calcolo Metriche Chiave
            points = game_logs['PTS'].astype(float).tolist()
            plus_minus = game_logs['PLUS_MINUS'].astype(float).tolist()
            
            points_trend = self._calculate_trend(points)
            plus_minus_trend = self._calculate_trend(plus_minus)
            consistency = self._calculate_consistency(points)
            recent_form = np.mean(points[-3:]) / (np.mean(points) + 1e-6) # Forma ultime 3 partite vs media

            # 3. Ponderazione e Normalizzazione
            score = (
                (points_trend * 10 * self.feature_weights['points_trend']) +
                (plus_minus_trend * 5 * self.feature_weights['plus_minus_trend']) +
                ((consistency - 0.5) * 50 * self.feature_weights['consistency']) +
                ((recent_form - 1.0) * 50 * self.feature_weights['recent_form'])
            )
            
            # 4. Normalizzazione finale a 0-100
            final_score = 50 + score # Partiamo da una base di 50
            return max(0, min(100, final_score))

        except Exception as e:
            print(f"❌ [MOMENTUM] Errore calcolo momentum per {player_name}: {e}")
            return 50.0 # Ritorna un valore neutro in caso di errore

    def predict_team_momentum_impact(self, team_roster_df):
        """
        Predice l'impatto del momentum per una squadra basandosi sul suo roster.

        Args:
            team_roster_df (pd.DataFrame): DataFrame del roster della squadra. 
                                           Deve contenere PLAYER_ID, PLAYER_NAME, POSITION, ROTATION_STATUS.

        Returns:
            dict: Dizionario con i punteggi di momentum dettagliati per la squadra.
        """
        if not isinstance(team_roster_df, pd.DataFrame) or team_roster_df.empty:
            return {'error': 'Input del roster non valido o vuoto', 'momentum_score': 0}

        player_contributions = []
        
        for _, player in team_roster_df.iterrows():
            player_id = player.get('PLAYER_ID')
            player_name = player.get('PLAYER_NAME', f'ID: {player_id}')
            position = player.get('POSITION', 'UNKNOWN')
            status = player.get('ROTATION_STATUS', 'BENCH')
            
            # Calcola il punteggio individuale
            player_score = self._get_player_momentum_score(player_id, player_name)
            
            # Pondera il contributo del giocatore
            status_weight = self.rotation_status_weights.get(status, 0.5)
            pos_weights = self.position_weights.get(position, self.position_weights['UNKNOWN'])
            
            contribution = {
                'player_name': player_name,
                'player_id': player_id,
                'momentum_score': player_score,
                'status_weight': status_weight,
                'offensive_contribution': player_score * status_weight * pos_weights['off'],
                'defensive_contribution': player_score * status_weight * pos_weights['def']
            }
            player_contributions.append(contribution)

        # Aggrega i risultati
        if not player_contributions:
            return {'error': 'Nessun contributo calcolato', 'momentum_score': 0}

        total_off_contribution = sum(c['offensive_contribution'] for c in player_contributions)
        total_def_contribution = sum(c['defensive_contribution'] for c in player_contributions)
        total_status_weight = sum(c['status_weight'] for c in player_contributions)
        
        # Evita divisione per zero
        if total_status_weight == 0:
            return {'error': 'Peso totale dei giocatori è zero', 'momentum_score': 0}

        # Calcola i punteggi aggregati
        offensive_momentum = total_off_contribution / (total_status_weight * 0.5) # Normalizza
        defensive_momentum = total_def_contribution / (total_status_weight * 0.5) # Normalizza
        
        # Il punteggio finale è una media pesata dei due aspetti
        final_momentum_score = (offensive_momentum * 0.55) + (defensive_momentum * 0.45)
        
        return {
            'momentum_score': final_momentum_score,
            'offensive_momentum': offensive_momentum,
            'defensive_momentum': defensive_momentum,
            'player_contributions': player_contributions,
            'timestamp': datetime.now().isoformat()
        }