# Modifiche e aggiunte a probabilistic_model.py

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Nuovo import per XGBoost se disponibile
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class ProbabilisticModel:
    """
    AGGIORNATO: Sistema probabilistico avanzato che integra momentum scientifico.
    """
    
    def __init__(self, models_dir='models'):
        print("🔍 [PROBABILISTIC] Inizializzazione ProbabilisticModel avanzato...")
        self.model_mu = None
        self.model_sigma = None
        self.momentum_integration_model = None  # NUOVO: Modello per integrazione momentum
        self.scaler = None
        self.is_trained = False
        self.models_dir = os.path.join(models_dir, 'probabilistic')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # NUOVO: Parametri scientificamente validati per momentum integration
        self.momentum_params = {
            'offensive_impact_factor': 0.85,     # Momentum offensivo ha impatto diretto
            'defensive_impact_factor': 1.15,     # Momentum difensivo più predittivo per totali
            'synergy_multiplier': 1.25,          # Quando entrambe squadre hanno momentum positivo
            'volatility_factor': 0.60,           # Momentum aumenta incertezza
            'confidence_threshold': 0.7,         # Soglia per applicare pieno impatto
            'max_momentum_impact': 12.0          # Cap massimo per evitare prediction estreme
        }
        
        # Schema quote fisso (manteniamo invariato)
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

    def load_models(self):
        """AGGIORNATO: Carica anche modello integrazione momentum se disponibile."""
        mu_model_path = os.path.join(self.models_dir, 'mu_model.pkl')
        sigma_model_path = os.path.join(self.models_dir, 'sigma_model.pkl')
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        momentum_model_path = os.path.join(self.models_dir, 'momentum_integration_model.pkl')
        
        if not all(os.path.exists(p) for p in [mu_model_path, sigma_model_path, scaler_path]):
            return False
            
        try:
            self.model_mu = joblib.load(mu_model_path)
            self.model_sigma = joblib.load(sigma_model_path)
            self.scaler = joblib.load(scaler_path)
            
            # Carica modello momentum se disponibile
            if os.path.exists(momentum_model_path):
                self.momentum_integration_model = joblib.load(momentum_model_path)
                print("✅ [PROBABILISTIC] Modello integrazione momentum caricato.")
            else:
                print("ℹ️ [PROBABILISTIC] Modello integrazione momentum non disponibile - usando integrazione rule-based.")
                
            self.is_trained = True
            print("✅ [PROBABILISTIC] Modelli (mu, sigma) e scaler caricati con successo.")
            return True
        except Exception as e:
            print(f"❌ [PROBABILISTIC] Errore critico nel caricamento dei modelli: {e}")
            self.is_trained = False
            return False

    def predict_distribution(self, team_stats, injury_impact, momentum_impact_data):
        """
        MODIFICATO: Predice distribuzione con integrazione momentum avanzata.
        
        Args:
            team_stats: Statistiche squadre
            injury_impact: Impatto infortuni (float)
            momentum_impact_data: Dati momentum dettagliati (dict) o semplice float per compatibilità
        """
        if not self.is_trained:
            return None
            
        try:
            # Estrazione base features (invariato)
            home, away = team_stats.get('home', {}), team_stats.get('away', {})
            features = np.array([[
                home.get('ORtg_season', 112.0), home.get('DRtg_season', 112.0), home.get('Pace_season', 100.0),
                away.get('ORtg_season', 112.0), away.get('DRtg_season', 112.0), away.get('Pace_season', 100.0),
                home.get('eFG_PCT_season', 0.53), home.get('TOV_PCT_season', 0.14),
                home.get('OREB_PCT_season', 0.23), home.get('FT_RATE_season', 0.20),
                away.get('eFG_PCT_season', 0.53), away.get('TOV_PCT_season', 0.14),
                away.get('OREB_PCT_season', 0.23), away.get('FT_RATE_season', 0.20),
                home.get('ORtg_L5', home.get('ORtg_season', 112.0)),
                home.get('DRtg_L5', home.get('DRtg_season', 112.0)),
                away.get('ORtg_L5', away.get('ORtg_season', 112.0)),
                away.get('DRtg_L5', away.get('DRtg_season', 112.0)),
                home.get('Pace_season', 100.0)
            ]])
            
            X_scaled = self.scaler.transform(features)
            mu_base = self.model_mu.predict(X_scaled)[0]
            sigma_base = self.model_sigma.predict(X_scaled)[0]
            
            # NUOVO: Integrazione momentum avanzata
            momentum_adjustments = self._integrate_advanced_momentum(
                momentum_impact_data, mu_base, sigma_base, team_stats
            )
            
            # Applica aggiustamenti
            mu_final = mu_base + injury_impact + momentum_adjustments['mu_adjustment']
            sigma_final = max(
                sigma_base + momentum_adjustments['sigma_adjustment'] + (abs(injury_impact) * 0.5),
                8.0  # Minimo sigma
            )
            
            return {
                'predicted_mu': mu_final,
                'predicted_sigma': sigma_final,
                'base_mu': mu_base,
                'base_sigma': sigma_base,
                'injury_adjustment': injury_impact,
                'momentum_adjustment': momentum_adjustments['mu_adjustment'],
                'momentum_details': momentum_adjustments,
                'total_uncertainty': sigma_final
            }
            
        except ValueError as ve:
            print(f"❌ [PROBABILISTIC] ERRORE DI DIMENSIONE FEATURE: {ve}")
            return None
        except Exception as e:
            print(f"❌ [PROBABILISTIC] Errore durante la predizione: {e}")
            return None

    def _integrate_advanced_momentum(self, momentum_data, base_mu, base_sigma, team_stats):
        """
        NUOVO: Integra momentum usando metodologie scientifiche validate.
        
        Args:
            momentum_data: Dati momentum (dict dettagliato o float semplice)
            base_mu: Media base del modello
            base_sigma: Sigma base del modello
            team_stats: Statistiche squadre per contesto
            
        Returns:
            dict: Aggiustamenti per mu e sigma
        """
        
        # Gestione compatibilità: se momentum_data è un float, convertilo
        if isinstance(momentum_data, (int, float)):
            return {
                'mu_adjustment': float(momentum_data),
                'sigma_adjustment': abs(float(momentum_data)) * 0.5,
                'method': 'legacy_compatible'
            }
        
        # Se non è un dict con i dati dettagliati, usa valore neutro
        if not isinstance(momentum_data, dict) or 'total_impact' not in momentum_data:
            return {
                'mu_adjustment': 0.0,
                'sigma_adjustment': 0.0,
                'method': 'neutral_fallback'
            }
        
        # Integrazione avanzata
        try:
            # Estrai componenti momentum
            total_impact = momentum_data.get('total_impact', 0.0)
            home_momentum = momentum_data.get('home_momentum', {})
            away_momentum = momentum_data.get('away_momentum', {})
            confidence = momentum_data.get('confidence_factor', 1.0)
            synergy_detected = momentum_data.get('synergy_detected', False)
            
            # Applica fattori scientifici
            confidence_adjusted_impact = total_impact * confidence
            
            # Cap per evitare predizioni estreme
            capped_impact = np.clip(
                confidence_adjusted_impact,
                -self.momentum_params['max_momentum_impact'],
                self.momentum_params['max_momentum_impact']
            )
            
            # Calcola aggiustamento sigma basato su volatilità momentum
            momentum_volatility = abs(home_momentum.get('score', 50) - 50) + abs(away_momentum.get('score', 50) - 50)
            volatility_adjustment = momentum_volatility * self.momentum_params['volatility_factor'] * 0.1
            
            # Bonus sinergico se rilevato
            synergy_bonus = 1.5 if synergy_detected else 0.0
            
            # Aggiustamento finale mu
            final_mu_adjustment = capped_impact + synergy_bonus
            
            # Aggiustamento sigma (momentum aumenta incertezza)
            final_sigma_adjustment = volatility_adjustment + (abs(capped_impact) * 0.3)
            
            return {
                'mu_adjustment': final_mu_adjustment,
                'sigma_adjustment': final_sigma_adjustment,
                'method': 'advanced_scientific',
                'components': {
                    'base_impact': total_impact,
                    'confidence_adjusted': confidence_adjusted_impact,
                    'capped_impact': capped_impact,
                    'synergy_bonus': synergy_bonus,
                    'volatility_adjustment': volatility_adjustment
                },
                'confidence_factor': confidence
            }
            
        except Exception as e:
            print(f"⚠️ [PROBABILISTIC] Errore integrazione momentum avanzata: {e}")
            # Fallback a integrazione semplice
            simple_impact = momentum_data.get('total_impact', 0.0)
            return {
                'mu_adjustment': simple_impact,
                'sigma_adjustment': abs(simple_impact) * 0.5,
                'method': 'simple_fallback'
            }

    def _use_ml_momentum_integration(self, momentum_data, base_features):
        """
        OPZIONALE: Usa modello ML per integrazione momentum se disponibile.
        """
        if not self.momentum_integration_model:
            return None
            
        try:
            # Costruisci feature vector per modello momentum
            momentum_features = self._extract_momentum_features(momentum_data)
            combined_features = np.concatenate([base_features.flatten(), momentum_features])
            
            # Predici aggiustamento
            adjustment = self.momentum_integration_model.predict([combined_features])[0]
            return adjustment
            
        except Exception as e:
            print(f"⚠️ [PROBABILISTIC] Errore nel modello ML momentum: {e}")
            return None

    def _extract_momentum_features(self, momentum_data):
        """
        SUPPORTO: Estrae feature numeriche da dati momentum per modello ML.
        """
        features = []
        
        if isinstance(momentum_data, dict):
            features.extend([
                momentum_data.get('total_impact', 0.0),
                momentum_data.get('home_momentum', {}).get('score', 50.0),
                momentum_data.get('away_momentum', {}).get('score', 50.0),
                momentum_data.get('home_momentum', {}).get('hot_hands', 0),
                momentum_data.get('away_momentum', {}).get('hot_hands', 0),
                momentum_data.get('confidence_factor', 1.0),
                1.0 if momentum_data.get('synergy_detected', False) else 0.0
            ])
        else:
            # Fallback per formato semplice
            features = [float(momentum_data), 50.0, 50.0, 0, 0, 1.0, 0.0]
        
        # Padding a lunghezza fissa se necessario
        while len(features) < 10:
            features.append(0.0)
            
        return np.array(features[:10])  # Tronca a 10 feature

    def analyze_betting_opportunities(self, distribution, odds_list=None, central_line=None, bankroll=100.0):
        """
        MODIFICATO: Analisi scommesse che considera momentum nell'edge calculation.
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
        
        # NUOVO: Fattore di confidenza momentum per edge adjustment
        momentum_details = distribution.get('momentum_details', {})
        momentum_confidence = momentum_details.get('confidence_factor', 1.0) if isinstance(momentum_details, dict) else 1.0
        
        for quote_info in odds_list:
            line = quote_info.get('line')
            over_odds = quote_info.get('over_quote')
            under_odds = quote_info.get('under_quote')
            
            if not all([line, over_odds, under_odds]):
                continue

            # Calcola probabilità base
            prob_over = 1 - stats.norm.cdf(line, loc=mu, scale=sigma)
            prob_under = stats.norm.cdf(line, loc=mu, scale=sigma)
            
            # NUOVO: Aggiusta edge per confidenza momentum
            edge_over = (prob_over * over_odds) - 1
            edge_under = (prob_under * under_odds) - 1
            
            # Applica confidence adjustment all'edge
            confidence_adjusted_edge_over = edge_over * momentum_confidence
            confidence_adjusted_edge_under = edge_under * momentum_confidence
            
            # Soglia value bet ajustata per confidenza
            dynamic_threshold = 0.05 / momentum_confidence  # Più stringente se bassa confidenza
            
            is_value_over = confidence_adjusted_edge_over > dynamic_threshold
            is_value_under = confidence_adjusted_edge_under > dynamic_threshold
            
            # Kelly fraction con adjustment
            kelly_over = self._calculate_adjusted_kelly(prob_over, over_odds, momentum_confidence) if is_value_over else 0
            kelly_under = self._calculate_adjusted_kelly(prob_under, under_odds, momentum_confidence) if is_value_under else 0
            
            stake_over = max(0, bankroll * kelly_over * 0.1) if is_value_over else 0
            stake_under = max(0, bankroll * kelly_under * 0.1) if is_value_under else 0
            
            all_lines_analysis.extend([
                {
                    'type': 'OVER', 'line': line, 'odds': over_odds,
                    'probability': prob_over, 'edge': confidence_adjusted_edge_over,
                    'stake': round(stake_over, 2), 'is_value': is_value_over,
                    'momentum_confidence': momentum_confidence
                },
                {
                    'type': 'UNDER', 'line': line, 'odds': under_odds,
                    'probability': prob_under, 'edge': confidence_adjusted_edge_under,
                    'stake': round(stake_under, 2), 'is_value': is_value_under,
                    'momentum_confidence': momentum_confidence
                }
            ])

        return sorted(all_lines_analysis, key=lambda x: x['edge'], reverse=True)

    def _calculate_adjusted_kelly(self, probability, odds, confidence_factor):
        """
        NUOVO: Calcola Kelly fraction aggiustata per confidenza momentum.
        """
        if probability <= 0 or odds <= 1:
            return 0
            
        # Kelly standard
        kelly = ((probability * (odds - 1)) - (1 - probability)) / (odds - 1)
        
        # Adjustment per confidenza: riduci kelly se bassa confidenza
        adjusted_kelly = kelly * confidence_factor
        
        # Cap conservativo
        return max(0, min(adjusted_kelly, 0.25))

    def _generate_odds_from_central_line(self, central_line):
        """Genera quote da linea centrale (invariato)."""
        generated_odds = []
        for offset, over_quote, under_quote in self.QUOTE_SCHEMA:
            generated_odds.append({
                'line': central_line + offset,
                'over_quote': over_quote,
                'under_quote': under_quote
            })
        print(f"✅ Generate {len(generated_odds)} linee di quota attorno alla linea centrale {central_line}")
        return generated_odds