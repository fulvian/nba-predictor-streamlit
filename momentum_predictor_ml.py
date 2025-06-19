# momentum_predictor_ml.py
"""
Modulo per utilizzare il modello di Machine Learning addestrato
per predire l'impatto del momentum sul punteggio totale di una partita.
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os
import json

class MomentumPredictorML:
    """
    Carica un modello di momentum addestrato e lo usa per fare previsioni.
    """
    
    def __init__(self, model_dir='models/player_momentum_v2'):
        """
        Inizializza il predittore caricando il modello, lo scaler e i metadati.
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_loaded = False
        
        self._load_artifacts()

    def _load_artifacts(self):
        """Carica il modello, lo scaler e i metadati dal disco."""
        model_path = os.path.join(self.model_dir, 'momentum_impact_model.xgb')
        scaler_path = os.path.join(self.model_dir, 'momentum_scaler.pkl')
        metadata_path = os.path.join(self.model_dir, 'momentum_model_metadata.json')

        if not all(os.path.exists(p) for p in [model_path, scaler_path, metadata_path]):
            print("⚠️ [MomentumPredictorML] Artefatti del modello non trovati. Il predittore è inattivo.")
            return

        try:
            print("✅ [MomentumPredictorML] Caricamento artefatti del modello di momentum...")
            
            self.model = xgb.Booster()
            self.model.load_model(model_path)
            
            self.scaler = joblib.load(scaler_path)
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_columns', [])
            
            self.is_loaded = True
            print("   - Modello di momentum ML caricato e operativo.")

        except Exception as e:
            print(f"❌ [MomentumPredictorML] Errore durante il caricamento del modello: {e}")
            self.is_loaded = False

    def predict_momentum_impact(self, home_momentum_features: dict, away_momentum_features: dict) -> float:
        """
        Predice l'impatto del momentum in punti sulla base delle feature delle due squadre.

        Args:
            home_momentum_features (dict): Feature di momentum per la squadra di casa.
            away_momentum_features (dict): Feature di momentum per la squadra in trasferta.

        Returns:
            float: La deviazione di punteggio predetta (es. +3.5, -1.2).
        """
        if not self.is_loaded:
            # Se il modello non è carico, ritorna un impatto nullo.
            return 0.0

        try:
            # 1. Costruisci il vettore di feature nello stesso ordine del training
            feature_vector = self._create_feature_vector(home_momentum_features, away_momentum_features)
            
            # 2. Applica lo scaler
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # 3. Usa il modello per la previsione
            # XGBoost richiede un DMatrix, ma per una singola previsione possiamo usare il DataFrame
            prediction = self.model.predict(feature_vector_scaled)
            
            # La predizione è un array, prendiamo il primo (e unico) elemento
            impact = float(prediction[0])
            
            # Limita l'impatto entro un range ragionevole per evitare predizioni estreme
            capped_impact = np.clip(impact, -15.0, 15.0)
            
            print(f"   - [MomentumPredictorML] Impatto momentum predetto: {capped_impact:.2f} punti.")
            return capped_impact

        except Exception as e:
            print(f"❌ [MomentumPredictorML] Errore durante la previsione: {e}")
            return 0.0

    def _create_feature_vector(self, home_features: dict, away_features: dict) -> pd.DataFrame:
        """
        Crea un DataFrame a singola riga con le feature nell'ordine corretto.
        """
        # Appiattisci e prefissa le feature
        flat_features = {}
        for key, value in home_features.items():
            flat_features[f'home_{key}'] = value
        for key, value in away_features.items():
            flat_features[f'away_{key}'] = value
            
        # Aggiungi feature di confronto
        flat_features['momentum_diff'] = home_features.get('momentum_score', 50) - away_features.get('momentum_score', 50)
        
        # Crea il DataFrame con le colonne nell'ordine corretto
        ordered_features = {feat: [flat_features.get(feat, 0)] for feat in self.feature_names}
        feature_df = pd.DataFrame(ordered_features)
        
        return feature_df 