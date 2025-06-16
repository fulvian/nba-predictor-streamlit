# probabilistic_model.py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

class ProbabilisticModel:
    """
    Sistema probabilistico per predire la distribuzione dei punti totali (mu e sigma) di una partita NBA.
    """
    
    def __init__(self, models_dir='models'):
        print("🔍 [PROBABILISTIC] Inizializzazione ProbabilisticModel...")
        self.model_mu = None
        self.model_sigma = None
        self.scaler = None
        self.is_trained = False
        self.models_dir = os.path.join(models_dir, 'probabilistic')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # --- NUOVO: Definiamo lo schema fisso delle quote basato sull'offset dalla linea centrale ---
        self.QUOTE_SCHEMA = [
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
        
        if not self.load_models():
            print("⚠️ [PROBABILISTIC] Modelli non caricati. Il sistema non potrà effettuare predizioni.")

    # ... (il metodo load_models e predict_distribution rimangono invariati) ...
    def load_models(self):
        mu_model_path, sigma_model_path, scaler_path = os.path.join(self.models_dir, 'mu_model.pkl'), os.path.join(self.models_dir, 'sigma_model.pkl'), os.path.join(self.models_dir, 'scaler.pkl')
        if not all(os.path.exists(p) for p in [mu_model_path, sigma_model_path, scaler_path]): return False
        try:
            self.model_mu, self.model_sigma, self.scaler, self.is_trained = joblib.load(mu_model_path), joblib.load(sigma_model_path), joblib.load(scaler_path), True
            print("✅ [PROBABILISTIC] Modelli (mu, sigma) e scaler caricati con successo.")
            return True
        except Exception as e:
            print(f"❌ [PROBABILISTIC] Errore critico nel caricamento dei modelli: {e}"); self.is_trained = False; return False

    def predict_distribution(self, team_stats, injury_impact, momentum_impact):
        if not self.is_trained: return None
        try:
            home, away = team_stats.get('home', {}), team_stats.get('away', {})
            features = np.array([[home.get('ORtg_season', 112.0), home.get('DRtg_season', 112.0), home.get('Pace_season', 100.0), away.get('ORtg_season', 112.0), away.get('DRtg_season', 112.0), away.get('Pace_season', 100.0), home.get('eFG_PCT_season', 0.53), home.get('TOV_PCT_season', 0.14), home.get('OREB_PCT_season', 0.23), home.get('FT_RATE_season', 0.20), away.get('eFG_PCT_season', 0.53), away.get('TOV_PCT_season', 0.14), away.get('OREB_PCT_season', 0.23), away.get('FT_RATE_season', 0.20), home.get('ORtg_L5', home.get('ORtg_season', 112.0)), home.get('DRtg_L5', home.get('DRtg_season', 112.0)), away.get('ORtg_L5', away.get('ORtg_season', 112.0)), away.get('DRtg_L5', away.get('DRtg_season', 112.0)), home.get('Pace_season', 100.0)]])
            X_scaled = self.scaler.transform(features)
            mu_base, sigma_base = self.model_mu.predict(X_scaled)[0], self.model_sigma.predict(X_scaled)[0]
            mu_final, sigma_final = mu_base + injury_impact + momentum_impact, sigma_base + ((abs(injury_impact) + abs(momentum_impact)) * 0.5)
            return {'predicted_mu': mu_final, 'predicted_sigma': max(sigma_final, 8.0), 'base_mu': mu_base, 'base_sigma': sigma_base, 'injury_adjustment': injury_impact, 'momentum_adjustment': momentum_impact}
        except ValueError as ve: print(f"❌ [PROBABILISTIC] ERRORE DI DIMENSIONE FEATURE: {ve}"); return None
        except Exception as e: print(f"❌ [PROBABILISTIC] Errore durante la predizione: {e}"); return None

    # --- NUOVO METODO ---
    def _generate_odds_from_central_line(self, central_line):
        """Genera una lista di quote basandosi su una linea centrale e lo schema fisso."""
        generated_odds = []
        for offset, over_quote, under_quote in self.QUOTE_SCHEMA:
            generated_odds.append({
                'line': central_line + offset,
                'over_quote': over_quote,
                'under_quote': under_quote
            })
        print(f"✅ Generate {len(generated_odds)} linee di quota attorno alla linea centrale {central_line}")
        return generated_odds

    # --- METODO MODIFICATO ---
    def analyze_betting_opportunities(self, distribution, odds_list=None, central_line=None, bankroll=100.0):
        """
        Analizza TUTTE le linee di scommessa e contrassegna quelle di valore (edge > 5%).
        """
        if not distribution:
            return {'error': "Distribuzione non valida."}

        if central_line is not None:
            odds_list = self._generate_odds_from_central_line(central_line)
        
        if not odds_list:
            print("   -> Nessuna quota disponibile. Impossibile analizzare scommesse.")
            return []

        mu, sigma = distribution['predicted_mu'], distribution['predicted_sigma']
        all_lines_analysis = []

        for quote_info in odds_list:
            line, over_odds, under_odds = quote_info.get('line'), quote_info.get('over_quote'), quote_info.get('under_quote')
            if not all([line, over_odds, under_odds]):
                continue

            # Analisi per OVER
            prob_over = 1 - stats.norm.cdf(line, loc=mu, scale=sigma)
            edge_over = (prob_over * over_odds) - 1
            is_value_over = edge_over > 0.05
            kelly_over = ((prob_over * (over_odds - 1)) - (1 - prob_over)) / (over_odds - 1) if is_value_over else 0
            stake_over = max(0, bankroll * kelly_over * 0.1) if is_value_over else 0
            all_lines_analysis.append({
                'type': 'OVER', 'line': line, 'odds': over_odds, 
                'probability': prob_over, 'edge': edge_over, 
                'stake': round(stake_over, 2), 'is_value': is_value_over
            })

            # Analisi per UNDER
            prob_under = stats.norm.cdf(line, loc=mu, scale=sigma)
            edge_under = (prob_under * under_odds) - 1
            is_value_under = edge_under > 0.05
            kelly_under = ((prob_under * (under_odds - 1)) - (1 - prob_under)) / (under_odds - 1) if is_value_under else 0
            stake_under = max(0, bankroll * kelly_under * 0.1) if is_value_under else 0
            all_lines_analysis.append({
                'type': 'UNDER', 'line': line, 'odds': under_odds,
                'probability': prob_under, 'edge': edge_under,
                'stake': round(stake_under, 2), 'is_value': is_value_under
            })

        # Ordina sempre per edge decrescente per mostrare le migliori in cima
        return sorted(all_lines_analysis, key=lambda x: x['edge'], reverse=True)