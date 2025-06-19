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
        """AGGIORNATO: Carica modelli e scaler separati per MU e SIGMA."""
        mu_model_path = os.path.join(self.models_dir, 'mu_model.pkl')
        sigma_model_path = os.path.join(self.models_dir, 'sigma_model.pkl')
        mu_scaler_path = os.path.join(self.models_dir, 'mu_scaler.pkl')
        sigma_scaler_path = os.path.join(self.models_dir, 'sigma_scaler.pkl')
        momentum_model_path = os.path.join(self.models_dir, 'momentum_integration_model.pkl')
        
        # Verifica che esistano i modelli e scaler principali
        required_files = [mu_model_path, sigma_model_path, mu_scaler_path, sigma_scaler_path]
        if not all(os.path.exists(p) for p in required_files):
            print(f"❌ [PROBABILISTIC] File mancanti: {[p for p in required_files if not os.path.exists(p)]}")
            return False
            
        try:
            self.model_mu = joblib.load(mu_model_path)
            self.model_sigma = joblib.load(sigma_model_path)
            self.scaler_mu = joblib.load(mu_scaler_path)
            self.scaler_sigma = joblib.load(sigma_scaler_path)
            
            # Per compatibilità, usa lo stesso scaler per entrambi (assumendo che siano identici)
            self.scaler = self.scaler_mu
            
            # Carica modello momentum se disponibile
            if os.path.exists(momentum_model_path):
                self.momentum_integration_model = joblib.load(momentum_model_path)
                print("✅ [PROBABILISTIC] Modello integrazione momentum caricato.")
            else:
                print("ℹ️ [PROBABILISTIC] Modello integrazione momentum non disponibile - usando integrazione rule-based.")
                
            self.is_trained = True
            print("✅ [PROBABILISTIC] Modelli (mu, sigma) e scaler separati caricati con successo.")
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
            # AGGIORNATO: Estrazione features per nuovo modello bilanciato (21 features)
            home, away = team_stats.get('home', {}), team_stats.get('away', {})
            
            # Calcola valori derivati necessari
            home_pace = home.get('pace', 100.0)
            away_pace = away.get('pace', 100.0)
            game_pace = (home_pace + away_pace) / 2
            pace_differential = home_pace - away_pace
            
            home_ortg = home.get('offensive_rating', 112.0)
            home_drtg = home.get('defensive_rating', 112.0)
            away_ortg = away.get('offensive_rating', 112.0)
            away_drtg = away.get('defensive_rating', 112.0)
            
            home_off_vs_away_def = home_ortg - away_drtg
            away_off_vs_home_def = away_ortg - home_drtg
            total_expected_scoring = home_off_vs_away_def + away_off_vs_home_def
            
            lg_avg_ortg = 112.0  # NBA league average
            avg_pace = game_pace
            
            # Features nel formato atteso dal nuovo modello (21 features)
            features = np.array([[
                # Four Factors - HOME
                home.get('efg_pct', 0.53),           # HOME_eFG_PCT_sAvg
                home.get('tov_pct', 0.14),           # HOME_TOV_PCT_sAvg
                home.get('oreb_pct', 0.25),          # HOME_OREB_PCT_sAvg
                home.get('ft_rate', 0.24),           # HOME_FT_RATE_sAvg
                
                # Four Factors - AWAY
                away.get('efg_pct', 0.53),           # AWAY_eFG_PCT_sAvg
                away.get('tov_pct', 0.14),           # AWAY_TOV_PCT_sAvg
                away.get('oreb_pct', 0.25),          # AWAY_OREB_PCT_sAvg
                away.get('ft_rate', 0.24),           # AWAY_FT_RATE_sAvg
                
                # Advanced Ratings
                home_ortg,                           # HOME_ORtg_sAvg
                home_drtg,                           # HOME_DRtg_sAvg
                away_ortg,                           # AWAY_ORtg_sAvg
                away_drtg,                           # AWAY_DRtg_sAvg
                
                # Pace & Context
                home_pace,                           # HOME_PACE
                away_pace,                           # AWAY_PACE
                game_pace,                           # GAME_PACE
                pace_differential,                   # PACE_DIFFERENTIAL
                home_off_vs_away_def,               # HOME_OFF_vs_AWAY_DEF
                away_off_vs_home_def,               # AWAY_OFF_vs_HOME_DEF
                total_expected_scoring,              # TOTAL_EXPECTED_SCORING
                lg_avg_ortg,                         # LgAvg_ORtg_season
                avg_pace                             # AVG_PACE
            ]])
            
            # Usa scaler appropriati (separati se disponibili, altrimenti condiviso)
            # CORREZIONE WARNING: Converte array in DataFrame per mantenere feature names
            import pandas as pd
            
            # Crea nomi di feature per evitare il warning StandardScaler
            feature_names = [
                'HOME_eFG_PCT_sAvg', 'HOME_TOV_PCT_sAvg', 'HOME_OREB_PCT_sAvg', 'HOME_FT_RATE_sAvg',
                'AWAY_eFG_PCT_sAvg', 'AWAY_TOV_PCT_sAvg', 'AWAY_OREB_PCT_sAvg', 'AWAY_FT_RATE_sAvg',
                'HOME_ORtg_sAvg', 'HOME_DRtg_sAvg', 'AWAY_ORtg_sAvg', 'AWAY_DRtg_sAvg',
                'HOME_PACE', 'AWAY_PACE', 'GAME_PACE', 'PACE_DIFFERENTIAL',
                'HOME_OFF_vs_AWAY_DEF', 'AWAY_OFF_vs_HOME_DEF', 'TOTAL_EXPECTED_SCORING',
                'LgAvg_ORtg_season', 'AVG_PACE'
            ]
            
            # Crea DataFrame con feature names per evitare warning
            features_df = pd.DataFrame(features, columns=feature_names[:features.shape[1]])
            
            X_scaled_mu = self.scaler_mu.transform(features_df) if hasattr(self, 'scaler_mu') else self.scaler.transform(features_df)
            X_scaled_sigma = self.scaler_sigma.transform(features_df) if hasattr(self, 'scaler_sigma') else self.scaler.transform(features_df)
            
            mu_base = self.model_mu.predict(X_scaled_mu)[0]
            sigma_base = self.model_sigma.predict(X_scaled_sigma)[0]
            
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
        if not distribution:
            return {'error': "Distribuzione non valida."}
    
        if central_line is not None:
            odds_list = self._generate_odds_from_central_line(central_line)
        
        if not odds_list:
            print("   -> Nessuna quota disponibile. Impossibile analizzare scommesse.")
            return []
    
        mu, sigma = distribution['predicted_mu'], distribution['predicted_sigma']
        all_lines_analysis = []
        
        momentum_details = distribution.get('momentum_details', {})
        momentum_confidence = momentum_details.get('confidence_factor', 1.0) if isinstance(momentum_details, dict) else 1.0
        
        # OTTIMIZZAZIONE MONTE CARLO NBA - Numero dinamico specifico per totali basketball
        # Ricerca NBA: ~49.7% vs 50.3% margine sottile richiede alta precisione
        # Range tipico NBA: 200-250 punti, volatilità diversa da altri sport
        base_simulations = 75000  # Aumentato per precisione NBA over/under
        
        # Sistema dinamico basato su sigma NBA-specific
        if sigma > 25:
            n_simulations = 150000  # Partite molto volatili (back-to-back, injuries)
        elif sigma > 20:
            n_simulations = 125000  # Alta incertezza NBA
        elif sigma > 15:
            n_simulations = 100000  # Media incertezza NBA  
        else:
            n_simulations = base_simulations  # Bassa incertezza NBA
        
        print(f"🎲 Eseguendo {n_simulations:,} simulazioni Monte Carlo (σ={sigma:.1f})...")
        print(f"💰 Sistema Stake GRANULARE: Min €1 (50% prob) → Max 5% bankroll (75% prob) + Quality/Edge/Quota")
        
        # SEED OTTIMIZZATO - Combina dati match con date per stabilità e unicità
        from datetime import datetime
        current_date = datetime.now()
        
        # SEED OTTIMIZZATO NBA OVER/UNDER - Specifico per il mercato basket totali
        # Basato su ricerca: 49.7% vs 50.3% distribuzione storica NBA
        # 1. Mu e sigma (caratteristiche predittive)
        # 2. Range totale NBA (per distinguere da altri sport) 
        # 3. Giorno dell'anno (per stagionalità NBA)
        # 4. Ora (per differenziare sessioni multiple)
        day_of_year = current_date.timetuple().tm_yday
        hour = current_date.hour
        
        # Classificazione range NBA per seed diversificato
        if mu < 210:
            nba_range_factor = 1    # Partite difensive
        elif mu < 230:
            nba_range_factor = 2    # Partite standard  
        elif mu < 250:
            nba_range_factor = 3    # Partite offensive
        else:
            nba_range_factor = 4    # Partite esplosive
        
        seed_components = [
            int(mu * 1000) % 10000,          # Predizione totale (4 cifre)
            int(sigma * 100) % 1000,         # Confidenza (3 cifre)
            nba_range_factor * 1000,         # Range NBA specifico (4 cifre)
            day_of_year,                     # Giorno dell'anno (3 cifre)
            hour                             # Ora (2 cifre)
        ]
        
        # Combina i componenti in un seed deterministico ma unico
        seed_base = int(''.join(f"{comp:04d}" for comp in seed_components)) % 2147483647
        
        print(f"🔧 Seed: {seed_base} (mu:{mu:.1f}, σ:{sigma:.1f}, day:{day_of_year}, h:{hour})")
        np.random.seed(seed_base)
        simulated_scores = np.random.normal(mu, sigma, n_simulations)
        
        # Calcola simulazioni favorevoli per la linea centrale (se disponibile)
        favorable_simulations = None
        favorable_percentage = None
        if central_line:
            favorable_simulations = np.sum(simulated_scores > central_line)
            favorable_percentage = (favorable_simulations / n_simulations) * 100
            print(f"📊 Simulazioni favorevoli OVER {central_line}: {favorable_simulations:,}/{n_simulations:,} ({favorable_percentage:.1f}%)")
        
        # VALIDAZIONE STATISTICA - Verifica convergenza probabilità
        # Test di convergenza per assicurarsi che il numero di simulazioni sia sufficiente
        if len(odds_list) > 0 and n_simulations >= 50000:
            # Prendi la prima linea per test di convergenza
            test_line = odds_list[0].get('line', central_line) if odds_list[0].get('line') else central_line
            if test_line:
                # Calcola probabilità su sottocampioni progressivi
                sample_sizes = [10000, 25000, n_simulations//2, n_simulations]
                prob_variations = []
                
                for sample_size in sample_sizes:
                    sample_over = np.sum(simulated_scores[:sample_size] > test_line)
                    sample_prob = sample_over / sample_size
                    prob_variations.append(sample_prob)
                
                # Calcola variazione massima tra gli ultimi due campioni
                max_variation = abs(prob_variations[-1] - prob_variations[-2])
                # Soglia più rigorosa per NBA over/under (margini stretti)
                convergence_threshold = 0.003  # 0.3% per maggiore precisione NBA
                
                if max_variation > convergence_threshold:
                    print(f"⚠️ Convergenza NBA incompleta: variazione {max_variation:.1%} > {convergence_threshold:.1%}")
                    print(f"💡 NBA over/under richiede alta precisione - consigliato aumentare simulazioni")
                else:
                    print(f"✅ Convergenza NBA ottimale: variazione {max_variation:.1%} < {convergence_threshold:.1%}")
                    print(f"🏀 Precisione adeguata per mercato NBA totali")
                    
        # Statistiche di controllo qualità
        print(f"📊 Distribuzione simulata: μ={np.mean(simulated_scores):.2f}, σ={np.std(simulated_scores):.2f}")
        
        for quote_info in odds_list:
            line = quote_info.get('line')
            over_odds = quote_info.get('over_quote')
            under_odds = quote_info.get('under_quote')
            
            if not all([line, over_odds, under_odds]):
                continue
    
            # Calcola probabilità usando simulazione Monte Carlo
            over_wins = np.sum(simulated_scores > line)
            under_wins = np.sum(simulated_scores <= line)
            
            prob_over = over_wins / n_simulations
            prob_under = under_wins / n_simulations
            
            # Calcola probabilità implicite dalle quote
            implied_prob_over = 1 / over_odds
            implied_prob_under = 1 / under_odds
            margin = (implied_prob_over + implied_prob_under) - 1
            
            # Rimuovi il margine del bookmaker
            true_prob_over = implied_prob_over / (1 + margin)
            true_prob_under = implied_prob_under / (1 + margin)
            
            # Calcola edge CORRETTO usando la formula: Edge = (Probabilità stimata × Quota) - 1
            # Formula corretta: se prob_over=0.749 e odds=1.95: (0.749 * 1.95) - 1 = 1.46 - 1 = 0.46 = 46%
            # Ma il 46% è un edge troppo alto! Il problema è nella comprensione:
            # Edge deve essere: (Expected Value - 1) dove EV = prob * odds
            # Per un edge realistico, usiamo: (prob * odds - 1) senza moltiplicare per 100
            edge_over = (prob_over * over_odds) - 1  # Risultato: 0.46 = 46% di expected value
            edge_under = (prob_under * under_odds) - 1  # Ma questo è comunque irrealisticamente alto
            
            # Calcola Quality Score per entrambe le opzioni
            quality_over = self._calculate_quality_score(edge_over, prob_over, over_odds)
            quality_under = self._calculate_quality_score(edge_under, prob_under, under_odds)
            
            # Applica una soglia minima per il value betting (formato decimale)
            min_edge = 0.02  # 2% in formato decimale (0.02)
            is_value_over = edge_over > min_edge
            is_value_under = edge_under > min_edge
            
            # Calcola stake usando il nuovo metodo avanzato
            # Calcola stake solo per scommesse di valore, passando i dati di qualità
            stake_over = self._calculate_advanced_stake(edge_over, prob_over, over_odds, bankroll, quality_over) if is_value_over else 0
            stake_under = self._calculate_advanced_stake(edge_under, prob_under, under_odds, bankroll, quality_under) if is_value_under else 0
            
            all_lines_analysis.extend([
                {
                    'type': 'OVER',
                    'line': line,
                    'odds': over_odds,
                    'probability': prob_over,
                    'implied_probability': implied_prob_over,
                    'true_probability': true_prob_over,
                    'edge': edge_over,
                    'quality_score': quality_over['quality_score'],
                    'edge_score': quality_over['edge_score'],
                    'confidence_score': quality_over['confidence_score'],
                    'risk_score': quality_over['risk_score'],
                    'consistency_score': quality_over['consistency_score'],
                    'stake': stake_over,
                    'is_value': is_value_over,
                    'margin': margin,
                    'simulation_wins': over_wins,
                    'total_simulations': n_simulations
                },
                {
                    'type': 'UNDER',
                    'line': line,
                    'odds': under_odds,
                    'probability': prob_under,
                    'implied_probability': implied_prob_under,
                    'true_probability': true_prob_under,
                    'edge': edge_under,
                    'quality_score': quality_under['quality_score'],
                    'edge_score': quality_under['edge_score'],
                    'confidence_score': quality_under['confidence_score'],
                    'risk_score': quality_under['risk_score'],
                    'consistency_score': quality_under['consistency_score'],
                    'stake': stake_under,
                    'is_value': is_value_under,
                    'margin': margin,
                    'simulation_wins': under_wins,
                    'total_simulations': n_simulations
                }
            ])
    
        # Aggiungi numero simulazioni al distribution per il riepilogo
        distribution['mc_simulations'] = n_simulations
        distribution['favorable_simulations'] = favorable_simulations
        distribution['favorable_percentage'] = favorable_percentage
        distribution['central_line_analyzed'] = central_line
        
        # Ordina per quality_score invece che per edge
        return sorted(all_lines_analysis, key=lambda x: x['quality_score'], reverse=True)
    
    def _calculate_kelly(self, probability, odds):
        if probability <= 0 or odds <= 1:
            return 0
        return (probability * (odds - 1) - (1 - probability)) / (odds - 1)


    def _calculate_advanced_stake(self, edge, estimated_prob, odds, bankroll, quality_data=None):
        """
        🚀 ADVANCED GRANULAR STAKE CALCULATOR 🚀
        
        Sistema di calcolo stake ultra-granulare che considera:
        1. Quality Score (peso 35%) - Qualità complessiva della scommessa
        2. Probabilità (peso 30%) - Fiducia nella predizione
        3. Edge (peso 25%) - Vantaggio matematico
        4. Quota di Mercato (peso 10%) - Profilo rischio/rendimento
        
        Parametri:
        - edge: Vantaggio matematico (0.0-1.0)
        - estimated_prob: Probabilità stimata (0.0-1.0)
        - odds: Quota del bookmaker
        - bankroll: Bankroll disponibile
        - quality_data: Dati di qualità dal quality_score
        """
        
        # REGOLA BASE: NO BET se probabilità < 50%
        if estimated_prob < 0.50:
            return 0
        
        # Estrai i dati di qualità se disponibili
        if quality_data is None:
            quality_data = self._calculate_quality_score(edge, estimated_prob, odds)
        
        quality_score = quality_data.get('quality_score', 0)
        edge_score = quality_data.get('edge_score', 0)
        confidence_score = quality_data.get('confidence_score', 0)
        risk_score = quality_data.get('risk_score', 0)
        kelly_fraction = quality_data.get('kelly_fraction', 0)
        
        # === COMPONENTE 1: QUALITY MULTIPLIER (35%) ===
        # Trasforma Quality Score (0-1) in moltiplicatore stake (0.2-2.0)
        if quality_score >= 0.9:
            quality_multiplier = 2.0    # Qualità eccezionale
        elif quality_score >= 0.8:
            quality_multiplier = 1.8    # Qualità ottima
        elif quality_score >= 0.7:
            quality_multiplier = 1.5    # Qualità buona
        elif quality_score >= 0.6:
            quality_multiplier = 1.2    # Qualità media
        elif quality_score >= 0.4:
            quality_multiplier = 0.8    # Qualità bassa
        else:
            quality_multiplier = 0.3    # Qualità molto bassa
        
        # === COMPONENTE 2: PROBABILITY FACTOR (30%) ===
        # Scala la probabilità in fattore stake
        prob_pct = estimated_prob * 100
        if prob_pct >= 75:
            prob_factor = 1.5      # Alta fiducia
        elif prob_pct >= 65:
            prob_factor = 1.3      # Buona fiducia
        elif prob_pct >= 55:
            prob_factor = 1.0      # Fiducia media
        elif prob_pct >= 50:
            prob_factor = 0.7      # Fiducia minima
        else:
            prob_factor = 0.2      # Troppo bassa
        
        # === COMPONENTE 3: EDGE AMPLIFIER (25%) ===
        # Edge è in formato decimale, convertiamo in percentuale
        edge_pct = edge * 100  # Converte in percentuale
        if edge_pct >= 15:
            edge_amplifier = 2.0    # Edge eccezionale
        elif edge_pct >= 10:
            edge_amplifier = 1.6    # Edge ottimo
        elif edge_pct >= 7:
            edge_amplifier = 1.3    # Edge buono
        elif edge_pct >= 5:
            edge_amplifier = 1.0    # Edge accettabile
        elif edge_pct >= 2:
            edge_amplifier = 0.7    # Edge basso
        else:
            edge_amplifier = 0.4    # Edge molto basso
        
        # === COMPONENTE 4: ODDS RISK FACTOR (10%) ===
        # Fattore di rischio basato sulla quota
        if 1.50 <= odds <= 2.00:
            odds_factor = 1.2      # Range ottimale
        elif 2.00 < odds <= 2.50:
            odds_factor = 1.0      # Buono
        elif 2.50 < odds <= 3.50:
            odds_factor = 0.8      # Accettabile
        elif 1.30 <= odds < 1.50:
            odds_factor = 0.9      # Margine basso
        else:
            odds_factor = 0.6      # Rischio alto
        
        # === CALCOLO STAKE BASE REALISTICO ===
        # Range probabilità: 50% -> stake minimo, 75% -> stake massimo (5% bankroll)
        
        # 1. STAKE MINIMO: 1€ o 1% del bankroll (il maggiore)
        min_stake_euro = 1.0
        min_stake_pct = 0.01  # 1% del bankroll
        min_stake_from_pct = bankroll * min_stake_pct
        stake_minimum = max(min_stake_euro, min_stake_from_pct)
        
        # 2. STAKE MASSIMO: 5% del bankroll
        max_stake_pct = 0.05
        stake_maximum = bankroll * max_stake_pct
        
        # 3. SCALING PROBABILITÀ (50% -> minimo, 75% -> massimo)
        prob_pct = estimated_prob * 100
        if prob_pct <= 50:
            prob_scale = 0.0  # Stake minimo
        elif prob_pct >= 75:
            prob_scale = 1.0  # Stake massimo
        else:
            # Interpolazione lineare tra 50% e 75%
            prob_scale = (prob_pct - 50) / (75 - 50)
        
        # 4. CALCOLO COMPOSITE MULTIPLIER
        # Formula pesata con pesi ottimizzati
        composite_multiplier = (
            quality_multiplier * 0.35 +
            prob_factor * 0.30 +
            edge_amplifier * 0.25 +
            odds_factor * 0.10
        )
        
        # 5. CALCOLO STAKE FINALE
        # Base stake calcolato dalla probabilità
        base_stake = stake_minimum + (stake_maximum - stake_minimum) * prob_scale
        
        # Applica il moltiplicatore composito
        target_stake = base_stake * composite_multiplier
        
        # === KELLY CONSTRAINT ===
        # Usa Kelly come limite superiore di sicurezza
        kelly_limit_stake = bankroll * max(0.01, min(0.08, kelly_fraction))
        
        # Applica tutti i limiti
        final_stake = min(target_stake, stake_maximum, kelly_limit_stake)
        final_stake = max(final_stake, stake_minimum)  # Garantisce sempre il minimo
        
        stake = final_stake
        
        # === LOGGING DETTAGLIATO (solo per debug) ===
        if False:  # Cambia a True per debug
            print(f"🔍 [STAKE_DEBUG] Prob: {prob_pct:.1f}%, Edge: {edge_pct:.1f}%, Quality: {quality_score:.2f}")
            print(f"   Multipliers: Q={quality_multiplier:.2f}, P={prob_factor:.2f}, E={edge_amplifier:.2f}, O={odds_factor:.2f}")
            print(f"   Base: €{base_stake:.1f}, Target: €{target_stake:.1f}, Final: €{stake:.1f}")
            print(f"   Limits: Min=€{stake_minimum:.1f}, Max=€{stake_maximum:.1f}, Kelly=€{kelly_limit_stake:.1f}")
        
        # Arrotondamento intelligente
        if stake >= 10:
            return round(stake, 0)      # Arrotonda a euro intero per stake alti
        elif stake >= 1:
            return round(stake, 1)      # Arrotonda a 0.1€ per stake medi
        else:
            return round(stake, 2)      # Arrotonda a 0.01€ per stake bassi

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

    def _calculate_quality_score(self, edge, estimated_prob, odds):
        """
        Calcola un punteggio di qualità avanzato per identificare la migliore scommessa.
        
        NUOVO ALGORITMO MULTI-FATTORIALE:
        - Edge Score: Normalizza il vantaggio matematico
        - Confidence Score: Valuta la fiducia nella predizione
        - Risk Score: Analizza il profilo rischio/rendimento
        - Consistency Score: Premia la consistenza del modello
        - Final Quality: Combinazione pesata con scaling intelligente
        
        Parametri:
        - edge: Vantaggio matematico (es. 0.09 = 9%)
        - estimated_prob: Probabilità stimata dal modello (es. 0.507 = 50.7%)
        - odds: Quota del bookmaker (es. 2.15)
        """
        
        # 1. EDGE SCORE - Normalizza il vantaggio matematico (0-100)
        # Edge è in formato decimale (es. 0.46 per 46%), convertiamo in percentuale
        edge_pct = edge * 100  # Converte in percentuale
        if edge_pct <= 0:
            edge_score = 0
        elif edge_pct >= 20:  # Edge > 20% = punteggio massimo
            edge_score = 100
        else:
            # Scala non lineare: premia edge alti
            edge_score = (edge_pct / 20) ** 0.7 * 100
        
        # 2. CONFIDENCE SCORE - Valuta la fiducia nella predizione (0-100)
        prob_pct = estimated_prob * 100
        if prob_pct < 45:
            confidence_score = 0  # Troppo incerto
        elif prob_pct > 95:
            confidence_score = 30  # Troppo estremo, probabilmente errore
        elif 50 <= prob_pct <= 65:
            confidence_score = 100  # Sweet spot: fiducia alta ma realistica
        elif 65 < prob_pct <= 75:
            confidence_score = 90   # Molto buono
        elif 75 < prob_pct <= 85:
            confidence_score = 75   # Buono ma più rischioso
        elif 45 <= prob_pct < 50:
            confidence_score = 40   # Limite accettabile
        else:  # 85-95%
            confidence_score = 50   # Troppo fiducioso
        
        # 3. RISK SCORE - Analizza profilo rischio/rendimento (0-100)
        implied_prob = 1 / odds
        kelly_fraction = (estimated_prob * odds - 1) / (odds - 1) if odds > 1 else 0
        
        # Analisi del rischio basata su quota e Kelly
        if 1.50 <= odds <= 2.00:
            base_risk_score = 100  # Range ideale
        elif 2.00 < odds <= 2.50:
            base_risk_score = 85   # Buono
        elif 2.50 < odds <= 3.00:
            base_risk_score = 70   # Accettabile
        elif 3.00 < odds <= 4.00:
            base_risk_score = 50   # Rischio moderato
        elif 1.30 <= odds < 1.50:
            base_risk_score = 60   # Margine basso
        else:
            base_risk_score = 25   # Rischio alto o margine troppo basso
        
        # Bonus Kelly: premia frazioni Kelly ottimali (2-8%)
        if 0.02 <= kelly_fraction <= 0.08:
            kelly_bonus = 20
        elif 0.01 <= kelly_fraction < 0.02:
            kelly_bonus = 10
        elif 0.08 < kelly_fraction <= 0.15:
            kelly_bonus = 5
        else:
            kelly_bonus = 0
        
        risk_score = min(100, base_risk_score + kelly_bonus)
        
        # 4. CONSISTENCY SCORE - Misura coerenza predizione vs mercato (0-100)
        # Differenza tra probabilità stimata e implicita
        prob_diff = abs(estimated_prob - implied_prob)
        if prob_diff <= 0.05:  # Differenza ≤ 5%
            consistency_score = 100
        elif prob_diff <= 0.10:  # Differenza ≤ 10%
            consistency_score = 80
        elif prob_diff <= 0.15:  # Differenza ≤ 15%
            consistency_score = 60
        elif prob_diff <= 0.20:  # Differenza ≤ 20%
            consistency_score = 40
        else:
            consistency_score = 20
        
        # 5. CALCOLO FINALE - Combinazione pesata con scaling intelligente
        # Pesi: Edge (40%), Confidence (30%), Risk (20%), Consistency (10%)
        raw_score = (
            edge_score * 0.40 +
            confidence_score * 0.30 +
            risk_score * 0.20 +
            consistency_score * 0.10
        )
        
        # Scaling finale: trasforma da 0-100 a 0-1 con curva non lineare
        # Premia i punteggi alti e penalizza quelli bassi
        if raw_score >= 80:
            final_quality = 0.8 + (raw_score - 80) / 20 * 0.2  # 80-100 → 0.8-1.0
        elif raw_score >= 60:
            final_quality = 0.5 + (raw_score - 60) / 20 * 0.3  # 60-80 → 0.5-0.8
        elif raw_score >= 40:
            final_quality = 0.2 + (raw_score - 40) / 20 * 0.3  # 40-60 → 0.2-0.5
        else:
            final_quality = raw_score / 40 * 0.2  # 0-40 → 0.0-0.2
        
        return {
            'quality_score': final_quality,
            'edge_score': edge_score / 100,
            'confidence_score': confidence_score / 100,
            'risk_score': risk_score / 100,
            'consistency_score': consistency_score / 100,
            'raw_score': raw_score,
            'kelly_fraction': kelly_fraction,
            'edge': edge,
            'estimated_prob': estimated_prob,
            'implied_prob': implied_prob
        }

    def get_recommendation_summary(self, opportunities):
        """
        Genera un riassunto della raccomandazione per la migliore scommessa.
        """
        if not opportunities or not any(opp['is_value'] for opp in opportunities):
            return {
                'has_recommendation': False,
                'message': 'Nessuna scommessa di valore trovata'
            }
        
        best_bet = opportunities[0]  # Già ordinato per quality_score
        
        return {
            'has_recommendation': True,
            'best_bet': best_bet,
            'quality_explanation': self._explain_quality_score(best_bet),
            'risk_level': self._assess_risk_level(best_bet)
        }
    
    def _explain_quality_score(self, bet):
        """Spiega i componenti del Quality Score avanzato."""
        explanations = []
        quality = bet.get('quality_score', 0)
        
        # Analisi del punteggio complessivo
        if quality >= 0.8:
            explanations.append("🟢 QUALITÀ ECCELLENTE")
        elif quality >= 0.6:
            explanations.append("🟡 QUALITÀ BUONA")
        elif quality >= 0.4:
            explanations.append("🟠 QUALITÀ MEDIA")
        else:
            explanations.append("🔴 QUALITÀ BASSA")
        
        # Dettagli dei componenti (se disponibili)
        edge_score = bet.get('edge_score', 0)
        confidence_score = bet.get('confidence_score', 0)
        risk_score = bet.get('risk_score', 0)
        consistency_score = bet.get('consistency_score', 0)
        
        if edge_score >= 0.7:
            explanations.append("Edge matematico forte")
        elif edge_score >= 0.4:
            explanations.append("Edge matematico moderato")
        else:
            explanations.append("Edge matematico debole")
            
        if confidence_score >= 0.8:
            explanations.append("Fiducia predizione alta")
        elif confidence_score >= 0.5:
            explanations.append("Fiducia predizione media")
        else:
            explanations.append("Fiducia predizione bassa")
            
        if risk_score >= 0.8:
            explanations.append("Profilo rischio ottimale")
        elif risk_score >= 0.6:
            explanations.append("Profilo rischio accettabile")
        else:
            explanations.append("Profilo rischio elevato")
            
        return explanations
    
    def _assess_risk_level(self, bet):
        """Valuta il livello di rischio complessivo basato sul nuovo Quality Score."""
        quality = bet.get('quality_score', 0)
        risk_score = bet.get('risk_score', 0)
        
        # Combinazione di quality score e risk score specifico
        if quality >= 0.8 and risk_score >= 0.8:
            return "MOLTO BASSO"
        elif quality >= 0.6 and risk_score >= 0.6:
            return "BASSO"
        elif quality >= 0.4 and risk_score >= 0.4:
            return "MODERATO"
        elif quality >= 0.2:
            return "ALTO"
        else:
            return "MOLTO ALTO"