#!/usr/bin/env python3
"""
Momentum Predictor Selector
============================
Sistema intelligente che seleziona automaticamente il modello giusto:
- Regular Season Model: Per partite di stagione regolare
- Playoff Model: Per partite di playoff (con feature aggiuntive)
- Hybrid Model: Quando serve bilanciare entrambi i contesti
"""

import os
import pickle
import json
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

class MomentumPredictorSelector:
    """
    Selettore intelligente per modelli momentum basato sul contesto della partita.
    """
    
    def __init__(self, models_base_dir='models/momentum_complete'):
        """
        Inizializza il selettore.
        
        Args:
            models_base_dir: Directory base dei modelli completi
        """
        self.models_base_dir = models_base_dir
        self.regular_dir = os.path.join(models_base_dir, 'regular_season')
        self.playoff_dir = os.path.join(models_base_dir, 'playoff')
        self.hybrid_dir = os.path.join(models_base_dir, 'hybrid')
        
        # Storage modelli
        self.regular_model = None
        self.regular_scaler = None
        self.regular_metadata = None
        
        self.playoff_model = None
        self.playoff_scaler = None
        self.playoff_metadata = None
        
        self.hybrid_model = None
        self.hybrid_scaler = None
        self.hybrid_metadata = None
        
        self.is_loaded = False
        
        print(f"🔄 MomentumPredictorSelector inizializzato")
        print(f"   📁 Modelli base: {models_base_dir}")
        
        # Carica modelli automaticamente
        self.load_all_models()

    def load_all_models(self):
        """
        Carica tutti i modelli disponibili.
        """
        print("\n📦 Caricamento modelli...")
        
        models_loaded = 0
        
        # 1. Carica Regular Season Model
        if self._load_model_set('regular_season', self.regular_dir):
            models_loaded += 1
        
        # 2. Carica Playoff Model
        if self._load_model_set('playoff', self.playoff_dir):
            models_loaded += 2
        
        # 3. Carica Hybrid Model
        if self._load_model_set('hybrid', self.hybrid_dir):
            models_loaded += 4
        
        self.is_loaded = models_loaded > 0
        
        if self.is_loaded:
            print(f"✅ {models_loaded} set di modelli caricati con successo")
            self._print_model_summary()
        else:
            print("❌ Nessun modello caricato. Esegui prima il training.")
            
        return self.is_loaded

    def _load_model_set(self, model_type, model_dir):
        """
        Carica un set di modelli (model + scaler + metadata).
        """
        model_path = os.path.join(model_dir, 'model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        metadata_path = os.path.join(model_dir, 'metadata.json')
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, metadata_path]):
            print(f"   ⚠️ {model_type.title()} model: File mancanti")
            return False
        
        try:
            # Carica model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Carica scaler
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            # Carica metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Assegna agli attributi corretti
            if model_type == 'regular_season':
                self.regular_model = model
                self.regular_scaler = scaler
                self.regular_metadata = metadata
            elif model_type == 'playoff':
                self.playoff_model = model
                self.playoff_scaler = scaler
                self.playoff_metadata = metadata
            elif model_type == 'hybrid':
                self.hybrid_model = model
                self.hybrid_scaler = scaler
                self.hybrid_metadata = metadata
            
            print(f"   ✅ {model_type.title()}: {metadata.get('model_name', 'Unknown')}")
            return True
            
        except Exception as e:
            print(f"   ❌ {model_type.title()}: Errore caricamento - {e}")
            return False

    def _print_model_summary(self):
        """
        Stampa summary dei modelli caricati.
        """
        print(f"\n📊 SUMMARY MODELLI CARICATI:")
        
        if self.regular_model:
            mae = "N/A"
            if self.regular_metadata and 'progressive_results' in self.regular_metadata:
                # Prendi ultimo step
                steps = [k for k in self.regular_metadata['progressive_results'].keys()]
                if steps:
                    last_step = max(steps)
                    mae = f"{self.regular_metadata['progressive_results'][last_step]['best_result']['test_mae']:.3f}"
            
            print(f"   🏀 Regular Season: {self.regular_metadata.get('model_name', 'Unknown'):<20} | MAE: {mae}")
        
        if self.playoff_model:
            mae = f"{self.playoff_metadata.get('avg_mae', 0):.3f}" if self.playoff_metadata else "N/A"
            print(f"   🏆 Playoff:        {self.playoff_metadata.get('model_name', 'Unknown'):<20} | MAE: {mae}")
        
        if self.hybrid_model:
            mae = f"{self.hybrid_metadata.get('test_mae', 0):.3f}" if self.hybrid_metadata else "N/A"
            print(f"   🔗 Hybrid:         {self.hybrid_metadata.get('model_name', 'Unknown'):<20} | MAE: {mae}")

    def determine_game_context(self, game_date=None, explicit_game_type=None, **kwargs):
        """
        Determina il contesto della partita per scegliere il modello giusto.
        
        Args:
            game_date: Data della partita (datetime o string)
            explicit_game_type: Tipo esplicito ('Regular' o 'Playoff')
            **kwargs: Altri parametri di contesto
            
        Returns:
            dict: Contesto della partita con modello raccomandato
        """
        
        # 1. Se il tipo è esplicito, usalo
        if explicit_game_type:
            if explicit_game_type.lower() == 'playoff':
                recommended_model = 'playoff'
                confidence = 1.0
            else:
                recommended_model = 'regular_season'
                confidence = 1.0
        
        # 2. Determina dal calendario NBA
        elif game_date:
            if isinstance(game_date, str):
                try:
                    game_date = datetime.fromisoformat(game_date.replace('Z', '+00:00'))
                except:
                    try:
                        game_date = datetime.strptime(game_date, '%Y-%m-%d')
                    except:
                        game_date = datetime.now()
            
            # Logic calendario NBA
            if game_date.month >= 4 and game_date.month <= 6:
                if game_date.month == 4 and game_date.day < 15:
                    recommended_model = 'regular_season'
                    confidence = 0.8  # Fine regular season, potrebbe essere playoff
                else:
                    recommended_model = 'playoff'
                    confidence = 0.9
            else:
                recommended_model = 'regular_season'
                confidence = 0.95
        
        # 3. Fallback su hybrid se incerto
        else:
            recommended_model = 'hybrid'
            confidence = 0.5
        
        # 4. Controlla disponibilità modelli
        available_models = []
        if self.regular_model:
            available_models.append('regular_season')
        if self.playoff_model:
            available_models.append('playoff')
        if self.hybrid_model:
            available_models.append('hybrid')
        
        # Se il modello raccomandato non è disponibile
        if recommended_model not in available_models:
            if 'hybrid' in available_models:
                recommended_model = 'hybrid'
                confidence *= 0.7  # Riduci confidence
            elif available_models:
                recommended_model = available_models[0]
                confidence *= 0.5
            else:
                recommended_model = None
                confidence = 0.0
        
        context = {
            'recommended_model': recommended_model,
            'confidence': confidence,
            'available_models': available_models,
            'game_date': game_date,
            'explicit_type': explicit_game_type,
            'reasoning': self._get_reasoning(recommended_model, confidence, game_date)
        }
        
        return context

    def _get_reasoning(self, model, confidence, game_date):
        """
        Fornisce spiegazione della scelta del modello.
        """
        if model == 'playoff':
            if confidence >= 0.9:
                return "Data indica chiaramente playoff (aprile-giugno)"
            else:
                return "Probabile playoff, ma potrebbe essere fine regular season"
        elif model == 'regular_season':
            if confidence >= 0.9:
                return "Data indica chiaramente regular season"
            else:
                return "Probabile regular season"
        elif model == 'hybrid':
            return "Contesto incerto o modello specifico non disponibile"
        else:
            return "Nessun modello disponibile"

    def predict(self, momentum_features, game_context=None, **context_kwargs):
        """
        Predice usando il modello appropriato basato sul contesto.
        
        Args:
            momentum_features: Feature di momentum (dict o array)
            game_context: Contesto pre-determinato (opzionale)
            **context_kwargs: Parametri per determinare contesto automaticamente
            
        Returns:
            dict: Predizione con dettagli del modello usato
        """
        if not self.is_loaded:
            return {
                'error': 'Nessun modello caricato',
                'prediction': None
            }
        
        # Determina contesto se non fornito
        if game_context is None:
            game_context = self.determine_game_context(**context_kwargs)
        
        recommended_model = game_context['recommended_model']
        
        if recommended_model is None:
            return {
                'error': 'Nessun modello disponibile per questo contesto',
                'context': game_context,
                'prediction': None
            }
        
        # Prepara feature per il modello selezionato
        try:
            if recommended_model == 'regular_season':
                prediction = self._predict_with_model(
                    momentum_features, 
                    self.regular_model, 
                    self.regular_scaler, 
                    self.regular_metadata['feature_names']
                )
            elif recommended_model == 'playoff':
                # Per playoff, aggiungi feature mock se non presenti
                enhanced_features = self._enhance_features_for_playoff(momentum_features)
                prediction = self._predict_with_model(
                    enhanced_features,
                    self.playoff_model,
                    self.playoff_scaler,
                    self.playoff_metadata['feature_names']
                )
            elif recommended_model == 'hybrid':
                # Per hybrid, aggiungi flag contesto
                enhanced_features = self._enhance_features_for_hybrid(
                    momentum_features, 
                    game_context
                )
                prediction = self._predict_with_model(
                    enhanced_features,
                    self.hybrid_model,
                    self.hybrid_scaler,
                    self.hybrid_metadata['feature_names']
                )
            
            return {
                'prediction': prediction,
                'model_used': recommended_model,
                'confidence': game_context['confidence'],
                'context': game_context,
                'reasoning': game_context['reasoning']
            }
            
        except Exception as e:
            return {
                'error': f'Errore durante predizione: {str(e)}',
                'model_attempted': recommended_model,
                'context': game_context,
                'prediction': None
            }

    def _predict_with_model(self, features, model, scaler, expected_features):
        """
        Esegue predizione con un modello specifico.
        """
        # Converti features in format array se necessario
        if isinstance(features, dict):
            feature_array = np.array([features.get(feat, 0.0) for feat in expected_features])
        else:
            feature_array = np.array(features)
        
        # Assicurati che sia 2D
        if feature_array.ndim == 1:
            feature_array = feature_array.reshape(1, -1)
        
        # SICUREZZA: Controlla dimensioni prima di scaling
        scaler_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else len(expected_features)
        
        if feature_array.shape[1] != scaler_features:
            print(f"      ⚠️ MISMATCH FEATURE: ricevute {feature_array.shape[1]}, attese {scaler_features}")
            print(f"         Expected features: {expected_features[:5]}...")  # primi 5
            
            # Se dimensioni non corrispondono, padding con zeri o truncate
            if feature_array.shape[1] < scaler_features:
                # Aggiungi zeri per feature mancanti
                missing_count = scaler_features - feature_array.shape[1]
                print(f"         Aggiunto padding per {missing_count} feature mancanti")
                padding = np.zeros((feature_array.shape[0], missing_count))
                feature_array = np.hstack([feature_array, padding])
            else:
                # Truncate se troppe feature
                print(f"         Troncate {feature_array.shape[1] - scaler_features} feature extra")
                feature_array = feature_array[:, :scaler_features]
        
        # Scala features
        scaled_features = scaler.transform(feature_array)
        
        # Predici
        prediction = model.predict(scaled_features)[0]
        
        return float(prediction)

    def _enhance_features_for_playoff(self, base_features):
        """
        Aggiunge feature playoff mock se non presenti.
        """
        enhanced = base_features.copy() if isinstance(base_features, dict) else {}
        
        # Aggiungi feature playoff con valori di default
        playoff_defaults = {
            'home_playoff_experience': 0.6,
            'away_playoff_experience': 0.6,
            'home_seed_advantage': 0.0,
            'away_seed_advantage': 0.0,
            'series_game_number': 3,  # Game medio
            'is_elimination_game': 0,
            'is_series_opener': 0,
            'days_rest': 1,
            'rest_advantage': 0.1,
            'pressure_index': 0.3
        }
        
        for key, default_val in playoff_defaults.items():
            if key not in enhanced:
                enhanced[key] = default_val
        
        # Aggiungi adjusted features (intensità playoff)
        if isinstance(base_features, dict):
            for base_feat in base_features.keys():
                if base_feat.startswith(('home_momentum', 'away_momentum', 'momentum_diff')):
                    enhanced[f'{base_feat}_playoff_adjusted'] = base_features[base_feat] * 1.3
        
        return enhanced

    def _enhance_features_for_hybrid(self, base_features, game_context):
        """
        Aggiunge feature per modello ibrido.
        """
        enhanced = base_features.copy() if isinstance(base_features, dict) else {}
        
        # Flag playoff/regular
        if game_context['recommended_model'] == 'playoff':
            enhanced['is_playoff'] = 1
        else:
            enhanced['is_playoff'] = 0
        
        # Aggiungi feature di base
        enhanced['pressure_index'] = 0.2 if enhanced['is_playoff'] == 0 else 0.5
        enhanced['rest_advantage'] = 0.0
        
        return enhanced

    def get_model_info(self, model_type=None):
        """
        Restituisce informazioni sui modelli caricati.
        
        Args:
            model_type: Tipo specifico ('regular_season', 'playoff', 'hybrid') o None per tutti
            
        Returns:
            dict: Informazioni sui modelli
        """
        info = {}
        
        if model_type is None or model_type == 'regular_season':
            if self.regular_model:
                info['regular_season'] = {
                    'loaded': True,
                    'metadata': self.regular_metadata,
                    'feature_count': len(self.regular_metadata.get('feature_names', [])) if self.regular_metadata else 0
                }
            else:
                info['regular_season'] = {'loaded': False}
        
        if model_type is None or model_type == 'playoff':
            if self.playoff_model:
                info['playoff'] = {
                    'loaded': True,
                    'metadata': self.playoff_metadata,
                    'feature_count': len(self.playoff_metadata.get('feature_names', [])) if self.playoff_metadata else 0
                }
            else:
                info['playoff'] = {'loaded': False}
        
        if model_type is None or model_type == 'hybrid':
            if self.hybrid_model:
                info['hybrid'] = {
                    'loaded': True,
                    'metadata': self.hybrid_metadata,
                    'feature_count': len(self.hybrid_metadata.get('feature_names', [])) if self.hybrid_metadata else 0
                }
            else:
                info['hybrid'] = {'loaded': False}
        
        return info

    def test_prediction_pipeline(self):
        """
        Testa il pipeline di predizione con dati mock.
        """
        print("\n🧪 TEST PIPELINE PREDIZIONE")
        print("=" * 40)
        
        # Dati mock
        mock_features = {
            'home_momentum_score': 55.0,
            'home_hot_hand_players': 2,
            'home_avg_player_momentum': 52.0,
            'home_avg_player_weighted_contribution': 0.48,
            'home_team_offensive_potential': 8.2,
            'home_team_defensive_potential': 7.8,
            'away_momentum_score': 47.0,
            'away_hot_hand_players': 1,
            'away_avg_player_momentum': 49.0,
            'away_avg_player_weighted_contribution': 0.45,
            'away_team_offensive_potential': 7.9,
            'away_team_defensive_potential': 8.1,
            'momentum_diff': 8.0
        }
        
        # Test scenari diversi
        test_scenarios = [
            {
                'name': 'Regular Season (dicembre)',
                'context': {'game_date': '2024-12-15'}
            },
            {
                'name': 'Playoff (maggio)',
                'context': {'game_date': '2024-05-15'}
            },
            {
                'name': 'Tipo esplicito Playoff',
                'context': {'explicit_game_type': 'Playoff'}
            },
            {
                'name': 'Contesto incerto',
                'context': {}
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n🔍 Test: {scenario['name']}")
            
            result = self.predict(mock_features, **scenario['context'])
            
            if 'error' in result:
                print(f"   ❌ Errore: {result['error']}")
            else:
                print(f"   ✅ Predizione: {result['prediction']:.3f}")
                print(f"   🤖 Modello: {result['model_used']}")
                print(f"   🎯 Confidence: {result['confidence']:.2f}")
                print(f"   💭 Reasoning: {result['reasoning']}")
        
        print(f"\n✅ Test pipeline completato")


def main():
    """
    Funzione di test principale.
    """
    # Crea selector
    selector = MomentumPredictorSelector()
    
    if selector.is_loaded:
        # Testa pipeline
        selector.test_prediction_pipeline()
        
        # Mostra info modelli
        print(f"\n📋 INFO MODELLI:")
        model_info = selector.get_model_info()
        for model_type, info in model_info.items():
            status = "✅ Caricato" if info['loaded'] else "❌ Non disponibile"
            features = info.get('feature_count', 0)
            print(f"   {model_type.title()}: {status} ({features} features)")
    else:
        print("\n⚠️ Nessun modello disponibile. Esegui prima il training completo.")


if __name__ == "__main__":
    main() 