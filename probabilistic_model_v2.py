# ==== probabilistic_model.py ====
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from config import DATA_DIR, MODELS_BASE_DIR

class ProbabilisticModel:
    """Sistema probabilistico per predizioni NBA"""
    
    def __init__(self):
        self.model_mu = None
        self.model_sigma = None
        self.scaler = None
        self.is_trained = False
        self.models_dir = os.path.join(MODELS_BASE_DIR, 'probabilistic')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Tenta di caricare modelli esistenti
        self.load_models()
    
    def load_models(self):
        """Carica modelli pre-addestrati"""
        try:
            mu_model_path = os.path.join(self.models_dir, 'mu_model.pkl')
            sigma_model_path = os.path.join(self.models_dir, 'sigma_model.pkl')
            scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
            
            if all(os.path.exists(p) for p in [mu_model_path, sigma_model_path, scaler_path]):
                self.model_mu = joblib.load(mu_model_path)
                self.model_sigma = joblib.load(sigma_model_path)
                self.scaler = joblib.load(scaler_path)
                self.is_trained = True
                print("✅ Modelli probabilistici caricati")
                return True
        except Exception as e:
            print(f"⚠️ Errore caricamento modelli: {e}")
        
        return False
    
    def train_probabilistic_models(self, training_csv_name):
        """Addestra i modelli probabilistici"""
        print("🤖 Inizio training modelli probabilistici...")
        
        training_file = os.path.join(DATA_DIR, training_csv_name)
        if not os.path.exists(training_file):
            print(f"❌ File training non trovato: {training_file}")
            return False
        
        try:
            # Carica dati
            df = pd.read_csv(training_file)
            print(f"📊 Caricati {len(df)} campioni di training")
            
            # Prepara features
            feature_columns = [
                'home_ortg', 'home_drtg', 'home_pace', 'home_efg', 'home_ft_rate',
                'away_ortg', 'away_drtg', 'away_pace', 'away_efg', 'away_ft_rate',
                'home_win_streak', 'away_win_streak'
            ]
            
            # Verifica che le colonne esistano
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                print(f"❌ Colonne mancanti nel dataset: {missing_cols}")
                return False
            
            X = df[feature_columns].fillna(0)
            y_mu = df['target_mu'].fillna(220)  # Default NBA total
            y_sigma = df['target_sigma'].fillna(12)  # Default NBA spread
            
            # Split train/test
            X_train, X_test, y_mu_train, y_mu_test, y_sigma_train, y_sigma_test = train_test_split(
                X, y_mu, y_sigma, test_size=0.2, random_state=42
            )
            
            # Scaling
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model per mu
            print("🔄 Training modello μ...")
            self.model_mu = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model_mu.fit(X_train_scaled, y_mu_train)
            
            # Train model per sigma
            print("🔄 Training modello σ...")
            self.model_sigma = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model_sigma.fit(X_train_scaled, y_sigma_train)
            
            # Valutazione
            mu_pred = self.model_mu.predict(X_test_scaled)
            sigma_pred = self.model_sigma.predict(X_test_scaled)
            
            mu_mae = mean_absolute_error(y_mu_test, mu_pred)
            mu_r2 = r2_score(y_mu_test, mu_pred)
            sigma_mae = mean_absolute_error(y_sigma_test, sigma_pred)
            sigma_r2 = r2_score(y_sigma_test, sigma_pred)
            
            print(f"📊 Risultati μ: MAE={mu_mae:.2f}, R²={mu_r2:.3f}")
            print(f"📊 Risultati σ: MAE={sigma_mae:.2f}, R²={sigma_r2:.3f}")
            
            # Salva modelli
            joblib.dump(self.model_mu, os.path.join(self.models_dir, 'mu_model.pkl'))
            joblib.dump(self.model_sigma, os.path.join(self.models_dir, 'sigma_model.pkl'))
            joblib.dump(self.scaler, os.path.join(self.models_dir, 'scaler.pkl'))
            
            self.is_trained = True
            print("✅ Training completato e modelli salvati")
            return True
            
        except Exception as e:
            print(f"❌ Errore durante training: {e}")
            return False
    
    def predict_game_distribution_with_injuries(self, team_stats, injury_reports=None):
        """Predice la distribuzione di una partita con considerazione infortuni"""
        if not self.is_trained:
            print("❌ Modelli non addestrati")
            return None
        
        try:
            # Prepara features con tutti i 19 parametri attesi dal modello
            home_stats = team_stats.get('home', {})
            away_stats = team_stats.get('away', {})
            
            # Valori di default basati su medie NBA
            features = [
                home_stats.get('ORtg_season', 110.0),       # HOME_ORtg_sAvg
                home_stats.get('DRtg_season', 110.0),       # HOME_DRtg_sAvg
                home_stats.get('Pace_season', 100.0),       # HOME_PACE
                away_stats.get('ORtg_season', 110.0),       # AWAY_ORtg_sAvg
                away_stats.get('DRtg_season', 110.0),       # AWAY_DRtg_sAvg
                away_stats.get('Pace_season', 100.0),       # AWAY_PACE
                home_stats.get('eFG_PCT_season', 0.52),     # HOME_eFG_PCT_sAvg
                home_stats.get('TOV_PCT_season', 12.5),     # HOME_TOV_PCT_sAvg
                home_stats.get('OREB_PCT_season', 25.0),    # HOME_OREB_PCT_sAvg
                home_stats.get('FT_RATE_season', 0.25),     # HOME_FT_RATE_sAvg
                away_stats.get('eFG_PCT_season', 0.52),     # AWAY_eFG_PCT_sAvg
                away_stats.get('TOV_PCT_season', 12.5),     # AWAY_TOV_PCT_sAvg
                away_stats.get('OREB_PCT_season', 25.0),    # AWAY_OREB_PCT_sAvg
                away_stats.get('FT_RATE_season', 0.25),     # AWAY_FT_RATE_sAvg
                home_stats.get('ORtg_L5', 110.0),           # HOME_ORtg_L5Avg
                home_stats.get('DRtg_L5', 110.0),           # HOME_DRtg_L5Avg
                away_stats.get('ORtg_L5', 110.0),           # AWAY_ORtg_L5Avg
                away_stats.get('DRtg_L5', 110.0),           # AWAY_DRtg_L5Avg
                home_stats.get('Pace_season', 100.0)        # GAME_PACE (usiamo il pace di casa come default)
            ]
            
            X = np.array(features).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Predizioni base
            mu_pred = self.model_mu.predict(X_scaled)[0]
            sigma_pred = self.model_sigma.predict(X_scaled)[0]
            
            # Aggiustamenti per infortuni se disponibili
            mu_adjustment = 0.0
            sigma_adjustment = 0.0
            confidence_penalty = 0.0
            
            if injury_reports:
                home_impact = injury_reports.get('home_team_impact', 0)
                away_impact = injury_reports.get('away_team_impact', 0)
                
                # Aggiustamento mu (differenziale)
                mu_adjustment = home_impact - away_impact
                
                # Aggiustamento sigma (aumenta incertezza)
                total_injury_impact = abs(home_impact) + abs(away_impact)
                sigma_adjustment = total_injury_impact * 2.0
                
                # Penalità confidence
                confidence_penalty = min(total_injury_impact * 30, 20)
            
            mu_final = mu_pred + mu_adjustment
            sigma_final = max(sigma_pred + sigma_adjustment, 8.0)  # Minimo 8 punti
            confidence_score = max(85 - confidence_penalty, 30)  # Minimo 30%
            
            return {
                'predicted_mu': mu_final,
                'predicted_sigma': sigma_final,
                'confidence_score': confidence_score,
                'base_mu': mu_pred,
                'base_sigma': sigma_pred,
                'injury_adjustments': {
                    'mu_adjustment': mu_adjustment,
                    'sigma_adjustment': sigma_adjustment,
                    'confidence_penalty': confidence_penalty
                } if injury_reports else None
            }
            
        except Exception as e:
            print(f"❌ Errore predizione: {e}")
            return None
    
    def _generate_quotes_from_central_line(self, central_line):
        """Genera le quote per le linee vicine alla linea centrale"""
        # Schema delle quote in base alla distanza dalla linea centrale
        quote_schema = [
            (-8.0, 1.38, 2.85), (-7.5, 1.40, 2.75), (-7.0, 1.43, 2.65), (-6.5, 1.45, 2.60),
            (-6.0, 1.47, 2.55), (-5.5, 1.50, 2.50), (-5.0, 1.52, 2.40), (-4.5, 1.56, 2.35),
            (-4.0, 1.57, 2.30), (-3.5, 1.62, 2.25), (-3.0, 1.64, 2.20), (-2.5, 1.66, 2.15),
            (-2.0, 1.71, 2.10), (-1.5, 1.74, 2.05), (-1.0, 1.76, 2.00), (-0.5, 1.80, 1.96),
            (0.0, 1.90, 1.90),  # Linea centrale
            (0.5, 1.95, 1.80), (1.0, 2.00, 1.76), (1.5, 2.05, 1.74), (2.0, 2.10, 1.71),
            (2.5, 2.15, 1.66), (3.0, 2.20, 1.64), (3.5, 2.25, 1.62), (4.0, 2.30, 1.57),
            (4.5, 2.35, 1.55), (5.0, 2.40, 1.52), (5.5, 2.50, 1.50), (6.0, 2.55, 1.47),
            (6.5, 2.60, 1.45), (7.0, 2.65, 1.43), (7.5, 2.70, 1.41), (8.0, 2.85, 1.38)
        ]
        
        quotes = []
        for offset, over_quote, under_quote in quote_schema:
            line = central_line + offset
            quotes.append({
                'line': round(line, 1),
                'over_quote': over_quote,
                'under_quote': under_quote,
                'distance_from_center': abs(offset)
            })
        
        return quotes

    def analyze_betting_opportunities_with_injuries(self, game_data, manual_base_line=None):
        """Analizza opportunità di scommessa considerando infortuni"""
        if not self.is_trained:
            print("❌ Modelli non addestrati")
            return None
        
        try:
            home_team = game_data.get('team_stats', {}).get('home', {}).get('name', 'Squadra Casa')
            away_team = game_data.get('team_stats', {}).get('away', {}).get('name', 'Squadra Ospite')
            
            print(f"\n📊 ANALISI SCOMMESSA: {home_team} vs {away_team}")
            print("=" * 70)
            
            # Ottieni predizione
            injury_reports = game_data.get('injury_reports', {}).get('impact_analysis', {})
            prediction = self.predict_game_distribution_with_injuries(
                game_data['team_stats'], injury_reports
            )
            
            if not prediction:
                print("❌ Nessuna predizione disponibile per questa partita")
                return None
            
            # Stampa riepilogo predizione
            print(f"\n📈 PREDIZIONE TOTALE PUNTI: {prediction['predicted_mu']:.1f} ± {prediction['predicted_sigma']:.1f}")
            
            # Analizza quote disponibili
            opportunities = []
            
            if manual_base_line is not None:
                # Genera quote multiple basate sulla linea centrale
                print(f"🔍 Generazione quote multiple basate sulla linea centrale: {manual_base_line}")
                quotes = self._generate_quotes_from_central_line(manual_base_line)
                print(f"   • Generate {len(quotes)} linee di scommessa")
            else:
                # Usa le quote esistenti se disponibili
                quotes = game_data.get('odds', [])
                if not quotes:
                    print("⚠️ Nessuna quota disponibile per l'analisi")
                    return None
                
            # Lista per memorizzare tutte le quote per la tabella
            all_quotes = []
                
            for quote in quotes:
                line = quote.get('line')
                over_odds = quote.get('over_quote', 1.91)
                under_odds = quote.get('under_quote', 1.91)
                
                if line and over_odds and under_odds:
                    # Calcola probabilità implicite
                    mu = prediction['predicted_mu']
                    sigma = prediction['predicted_sigma']
                    
                    # Probabilità over/under usando distribuzione normale
                    prob_over = 1 - stats.norm.cdf(line, mu, sigma)
                    prob_under = stats.norm.cdf(line, mu, sigma)
                    
                    # Calcola edge
                    over_edge = (prob_over * over_odds) - 1
                    under_edge = (prob_under * under_odds) - 1
                    
                    # Kelly criterion per stake
                    bankroll = game_data.get('settings', {}).get('bankroll', 1000.0)
                    
                    # Calcola la frazione di Kelly per OVER
                    kelly_fraction = max(0, over_edge) / (over_odds - 1) if over_odds > 1 else 0
                    
                    # Applica Kelly frazionato in base all'edge
                    if over_edge >= 0.10:  # Edge >= 10%
                        kelly_fraction *= 0.33  # Kelly al 33%
                    elif over_edge >= 0.05:  # Edge tra 5% e 10%
                        kelly_fraction *= 0.25  # Kelly al 25%
                    else:  # Edge < 5%
                        kelly_fraction = 0  # Nessuna scommessa consigliata
                    
                    stake = bankroll * kelly_fraction
                    max_stake = bankroll * 0.05
                    stake = min(stake, max_stake) if kelly_fraction > 0 else 0
                    stake = round(stake, 1)  # Arrotonda a 1 decimale
                    
                    all_quotes.append({
                        'linea': line,
                        'tipo': 'OVER',
                        'quota': over_odds,
                        'probabilita': prob_over,
                        'edge': over_edge,
                        'stake': stake,
                        'valido': over_edge >= 0.05  # Indica se è una scommessa valida (edge >= 5%)
                    })
                    
                    # Calcola la frazione di Kelly per UNDER
                    kelly_fraction = max(0, under_edge) / (under_odds - 1) if under_odds > 1 else 0
                    
                    # Applica Kelly frazionato in base all'edge
                    if under_edge >= 0.10:  # Edge >= 10%
                        kelly_fraction *= 0.33  # Kelly al 33%
                    elif under_edge >= 0.05:  # Edge tra 5% e 10%
                        kelly_fraction *= 0.25  # Kelly al 25%
                    else:  # Edge < 5%
                        kelly_fraction = 0  # Nessuna scommessa consigliata
                    
                    stake = bankroll * kelly_fraction
                    max_stake = bankroll * 0.05
                    stake = min(stake, max_stake) if kelly_fraction > 0 else 0
                    stake = round(stake, 1)  # Arrotonda a 1 decimale
                    
                    all_quotes.append({
                        'linea': line,
                        'tipo': 'UNDER',
                        'quota': under_odds,
                        'probabilita': prob_under,
                        'edge': under_edge,
                        'stake': stake,
                        'valido': under_edge >= 0.05  # Indica se è una scommessa valida (edge >= 5%)
                    })
            
            # Ordina le quote per edge decrescente
            all_quotes.sort(key=lambda x: x['edge'], reverse=True)
            
            # Non stampiamo più la tabella qui, verrà gestita da autoover_5_1.py
            # Conta le opportunità valide
            valide_count = sum(1 for q in all_quotes if q['valido'])
            
            print("\n" + "=" * 90 + "\n")
            
            return {
                'prediction': prediction,
                'betting_opportunities': all_quotes,  # Restituisci tutte le opportunità
                'best_opportunities': [q for q in all_quotes if q['valido']][:3]  # E le migliori 3
            }
            
        except Exception as e:
            print(f"❌ Errore analisi betting: {e}")
            return None

# ==== player_momentum_predictor.py ====
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from config import DATA_DIR, MODELS_BASE_DIR

class PlayerMomentumPredictor:
    """Predittore ML per il momentum dei giocatori"""
    
    def __init__(self):
        self.momentum_model = None
        self.impact_model = None
        self.is_trained = False
        self.models_dir = os.path.join(MODELS_BASE_DIR, 'momentum')
        os.makedirs(self.models_dir, exist_ok=True)
    
    def load_models(self):
        """Carica modelli pre-addestrati"""
        try:
            momentum_path = os.path.join(self.models_dir, 'momentum_model.joblib')
            impact_path = os.path.join(self.models_dir, 'impact_model.joblib')
            
            if os.path.exists(momentum_path) and os.path.exists(impact_path):
                self.momentum_model = joblib.load(momentum_path)
                self.impact_model = joblib.load(impact_path)
                self.is_trained = True
                print("✅ Modelli momentum caricati")
                return True
        except Exception as e:
            print(f"⚠️ Errore caricamento modelli momentum: {e}")
        
        return False
    
    def collect_player_data_for_seasons(self, max_players_per_season=150):
        """Raccoglie dati giocatori per multiple stagioni"""
        print("📊 Raccolta dati giocatori (simulazione)...")
        
        # Simula raccolta dati (in implementazione reale, userebbe NBA API)
        np.random.seed(42)
        n_samples = max_players_per_season * 3  # 3 stagioni simulate
        
        data = {
            'player_id': np.arange(n_samples),
            'season': np.repeat(['2022-23', '2023-24', '2024-25'], max_players_per_season),
            'games_played': np.random.randint(50, 82, n_samples),
            'points_avg': np.random.normal(12, 8, n_samples),
            'assists_avg': np.random.normal(3, 3, n_samples),
            'rebounds_avg': np.random.normal(5, 4, n_samples),
            'usage_rate': np.random.normal(0.2, 0.1, n_samples),
            'plus_minus': np.random.normal(0, 5, n_samples),
            'momentum_score': np.random.normal(0, 2, n_samples)  # Target
        }
        
        df = pd.DataFrame(data)
        
        # Salva dataset
        dataset_path = os.path.join(DATA_DIR, 'player_momentum_data.csv')
        df.to_csv(dataset_path, index=False)
        print(f"💾 Dataset salvato: {dataset_path}")
        
        return df
    
    def create_training_dataset(self):
        """Crea dataset di training dal raw data"""
        dataset_path = os.path.join(DATA_DIR, 'player_momentum_data.csv')
        
        if not os.path.exists(dataset_path):
            print("❌ Dataset player momentum non trovato")
            return None
        
        df = pd.read_csv(dataset_path)
        
        # Feature engineering
        df['efficiency'] = df['points_avg'] + df['assists_avg'] + df['rebounds_avg']
        df['impact_factor'] = df['usage_rate'] * df['plus_minus']
        
        # Salva training dataset
        training_path = os.path.join(DATA_DIR, 'momentum_training_dataset.csv')
        df.to_csv(training_path, index=False)
        print(f"💾 Training dataset creato: {training_path}")
        
        return df
    
    def train_momentum_models(self):
        """Addestra i modelli momentum"""
        training_path = os.path.join(DATA_DIR, 'momentum_training_dataset.csv')
        
        if not os.path.exists(training_path):
            print("❌ Training dataset non trovato")
            return False
        
        try:
            df = pd.read_csv(training_path)
            
            # Features per momentum
            momentum_features = ['games_played', 'points_avg', 'assists_avg', 'rebounds_avg', 'efficiency']
            X_momentum = df[momentum_features].fillna(0)
            y_momentum = df['momentum_score'].fillna(0)
            
            # Features per impact
            impact_features = ['momentum_score', 'usage_rate', 'plus_minus', 'impact_factor']
            X_impact = df[impact_features].fillna(0)
            y_impact = df['points_avg']  # Usa points come proxy per impact
            
            # Train momentum model
            X_train, X_test, y_train, y_test = train_test_split(X_momentum, y_momentum, test_size=0.2, random_state=42)
            
            self.momentum_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.momentum_model.fit(X_train, y_train)
            
            momentum_pred = self.momentum_model.predict(X_test)
            momentum_mae = mean_absolute_error(y_test, momentum_pred)
            
            # Train impact model
            X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(X_impact, y_impact, test_size=0.2, random_state=42)
            
            self.impact_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.impact_model.fit(X_train_imp, y_train_imp)
            
            impact_pred = self.impact_model.predict(X_test_imp)
            impact_mae = mean_absolute_error(y_test_imp, impact_pred)
            
            print(f"📊 Momentum MAE: {momentum_mae:.3f}")
            print(f"📊 Impact MAE: {impact_mae:.3f}")
            
            # Salva modelli
            joblib.dump(self.momentum_model, os.path.join(self.models_dir, 'momentum_model.joblib'))
            joblib.dump(self.impact_model, os.path.join(self.models_dir, 'impact_model.joblib'))
            
            self.is_trained = True
            print("✅ Training momentum completato")
            return True
            
        except Exception as e:
            print(f"❌ Errore training momentum: {e}")
            return False
    
    def predict_team_momentum_impact(self, team_data):
        """Predice l'impatto momentum per una squadra"""
        if not self.is_trained:
            return 0.0
        
        try:
            total_impact = 0.0
            
            for player_data in team_data:
                stats = player_data.get('stats', {})
                
                # Features momentum
                momentum_features = [
                    stats.get('games_played', 50),
                    stats.get('points_avg', 10),
                    stats.get('assists_avg', 3),
                    stats.get('rebounds_avg', 5),
                    stats.get('efficiency', 18)
                ]
                
                momentum_score = self.momentum_model.predict([momentum_features])[0]
                
                # Features impact
                impact_features = [
                    momentum_score,
                    stats.get('usage_rate', 0.2),
                    stats.get('plus_minus', 0),
                    stats.get('impact_factor', 0)
                ]
                
                player_impact = self.impact_model.predict([impact_features])[0]
                total_impact += player_impact * 0.1  # Scale factor
            
            return min(max(total_impact, -5.0), 5.0)  # Clamp tra -5 e +5
            
        except Exception as e:
            print(f"❌ Errore predizione momentum: {e}")
            return 0.0

# ==== utils.py ====
import os
import json
from datetime import datetime, date
from config import DATA_DIR

def ensure_directory_exists(directory_path):
    """Assicura che una directory esista"""
    os.makedirs(directory_path, exist_ok=True)

def save_json_data(data, filename, directory=DATA_DIR):
    """Salva dati in formato JSON"""
    ensure_directory_exists(directory)
    filepath = os.path.join(directory, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return True
    except Exception as e:
        print(f"❌ Errore salvataggio JSON {filename}: {e}")
        return False

def load_json_data(filename, directory=DATA_DIR):
    """Carica dati da file JSON"""
    filepath = os.path.join(directory, filename)
    
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Errore caricamento JSON {filename}: {e}")
        return None

def format_currency(amount):
    """Formatta un importo come valuta"""
    return f"{amount:.2f}€"

def format_percentage(value):
    """Formatta un valore come percentuale"""
    return f"{value*100:+.1f}%"

def safe_float(value, default=0.0):
    """Converte sicuramente un valore in float"""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def calculate_kelly_fraction(probability, odds, max_fraction=0.25):
    """Calcola la frazione Kelly per una scommessa"""
    if probability <= 0 or odds <= 1:
        return 0.0
    
    kelly = (probability * odds - 1) / (odds - 1)
    return min(max(kelly, 0), max_fraction)  # Clamp tra 0 e max_fraction

class DateTimeEncoder(json.JSONEncoder):
    """Encoder JSON per oggetti datetime"""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)
