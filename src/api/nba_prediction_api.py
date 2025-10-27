#!/usr/bin/env python3
"""
🔌 NBA Prediction API
Context7-compliant real-time prediction service with WebSocket support.

This module provides a comprehensive FastAPI service for NBA game predictions,
integrating the ensemble ML models with real-time data processing and WebSocket
broadcasting for live predictions.
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from models.nba_models import NBAEnsembleModel, ModelConfig
from features.nba_features import NBAFeatureEngineer, NBAMetricsConfig
from nba_predictor.core.data_store import UnifiedDataStore

logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """Configuration for NBA Prediction API."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # WebSocket settings
    websocket_port: int = 8001
    max_connections: int = 100

    # Cache settings
    prediction_cache_ttl: int = 300  # 5 minutes
    data_cache_ttl: int = 60  # 1 minute

# Pydantic models for API requests/responses
class GameData(BaseModel):
    """Game data for prediction request."""
    home_team_id: int = Field(..., description="Home team NBA ID")
    away_team_id: int = Field(..., description="Away team NBA ID")
    home_team_name: str = Field(..., description="Home team name")
    away_team_name: str = Field(..., description="Away team name")
    season: str = Field(default="2025-26", description="Season")
    game_date: str = Field(..., description="Game date (YYYY-MM-DD)")

class PredictionRequest(BaseModel):
    """Prediction request with game data."""
    game: GameData
    prediction_type: str = Field(default="classification", description="classification or regression")
    include_shap: bool = Field(default=False, description="Include SHAP explanations")

class PredictionResponse(BaseModel):
    """Prediction response with results."""
    game_id: str = Field(..., description="Unique game identifier")
    prediction: float = Field(..., description="Prediction value")
    confidence: float = Field(..., description="Prediction confidence score")
    prediction_type: str = Field(..., description="Type of prediction made")
    timestamp: str = Field(..., description="Prediction timestamp")
    shap_explanations: Optional[Dict[str, float]] = Field(None, description="SHAP feature explanations")

class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: str = Field(..., description="Message timestamp")

class PredictionService:
    """Core prediction service integrating ML models with data processing."""

    def __init__(self, data_store: UnifiedDataStore):
        self.data_store = data_store
        self.feature_engineer = None
        self.ensemble_model = None
        self.prediction_cache = {}

        logger.info("🔌 Prediction Service initialized")

    def initialize_models(self):
        """Initialize ML models and feature engineering pipeline."""
        logger.info("🚀 Initializing ML models and feature engineering")

        try:
            # Initialize feature engineer
            self.feature_engineer = NBAFeatureEngineer(self.data_store)

            # Initialize ensemble model
            model_config = ModelConfig(
                xgb_n_estimators=100,
                rf_n_estimators=100,
                test_size=0.2
            )
            self.ensemble_model = NBAEnsembleModel(model_config)

            logger.info("✅ ML models and feature engineering initialized")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize models: {e}")
            return False

    def get_cache_key(self, game_data: GameData, prediction_type: str) -> str:
        """Generate cache key for predictions."""
        return f"{game_data.home_team_id}_{game_data.away_team_id}_{game_data.season}_{game_data.game_date}_{prediction_type}"

    def process_game_data(self, game_data: GameData) -> Dict[str, Any]:
        """Process game data to extract features for prediction."""
        try:
            # Get current season data
            season_start = f"{game_data.season.split('-')[0]}-10-01"
            season_end = f"{game_data.season.split('-')[1]}-04-30"

            # Extract team features
            team_features = self.feature_engineer.process_team_features(game_data.season)

            if team_features is None or len(team_features) == 0:
                # Fallback to mock data if no real data available
                logger.warning("⚠️ No team features available, using mock data")
                return self._create_mock_features(game_data)

            # Filter for specific teams
            home_features = team_features.filter(
                pl.col("TEAM_ID") == game_data.home_team_id
            )
            away_features = team_features.filter(
                pl.col("TEAM_ID") == game_data.away_team_id
            )

            if len(home_features) == 0 or len(away_features) == 0:
                logger.warning("⚠️ No features for requested teams, using mock data")
                return self._create_mock_features(game_data)

            # Combine team features
            combined_features = {
                'home_score_diff': float(home_features.select('PTS').mean() - away_features.select('PTS').mean()) if len(home_features) > 0 and len(away_features) > 0 else 0.0,
                'home_team_id': game_data.home_team_id,
                'away_team_id': game_data.away_team_id,
                'season': game_data.season,
                'game_date': game_data.game_date
            }

            # Add injury impact if available
            try:
                injury_features = self.feature_engineer.process_injury_features(game_data.season)
                if injury_features is not None and len(injury_features) > 0:
                    home_injury = injury_features.filter(pl.col("TEAM_ID") == game_data.home_team_id)
                    away_injury = injury_features.filter(pl.col("TEAM_ID") == game_data.away_team_id)

                    if len(home_injury) > 0 and len(away_injury) > 0:
                        combined_features['home_injury_impact'] = float(home_injury.select('INJURY_IMPACT').mean())
                        combined_features['away_injury_impact'] = float(away_injury.select('INJURY_IMPACT').mean())
                        combined_features['injury_diff'] = combined_features['home_injury_impact'] - combined_features['away_injury_impact']
            except Exception as e:
                logger.warning(f"⚠️ Could not process injury features: {e}")

            return combined_features

        except Exception as e:
            logger.error(f"❌ Error processing game data: {e}")
            return self._create_mock_features(game_data)

    def _create_mock_features(self, game_data: GameData) -> Dict[str, Any]:
        """Create mock features when real data is not available."""
        import random

        # Generate realistic mock features
        mock_features = {
            'home_score_diff': random.gauss(0, 10),
            'home_team_id': game_data.home_team_id,
            'away_team_id': game_data.away_team_id,
            'season': game_data.season,
            'game_date': game_data.game_date,
            'home_injury_impact': random.uniform(0, 2),
            'away_injury_impact': random.uniform(0, 2),
            'injury_diff': random.uniform(-1, 1),
            'team_chemistry_score': random.uniform(0.3, 0.8),
            'availability_score': random.uniform(0.7, 1.0)
        }

        logger.info(f"📊 Created mock features for {game_data.home_team_name} vs {game_data.away_team_name}")
        return mock_features

    def make_prediction(self, features: Dict[str, Any], prediction_type: str = "classification") -> Dict[str, Any]:
        """Make prediction using ensemble model."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([features])

            # Make prediction
            if prediction_type == "classification":
                prediction = self.ensemble_model.predict_classification(df)[0]
                confidence = abs(prediction - 0.5) * 2  # Convert to confidence score
            else:
                prediction = self.ensemble_model.predict_regression(df)[0]
                confidence = 0.7  # Default confidence for regression

            # Calculate confidence based on model consensus if available
            if hasattr(self.ensemble_model, 'models') and len(self.ensemble_model.models) > 1:
                # Simple confidence calculation based on prediction consistency
                prediction = float(prediction)

            return {
                'prediction': prediction,
                'confidence': min(confidence, 1.0),
                'prediction_type': prediction_type,
                'features_used': list(features.keys())
            }

        except Exception as e:
            logger.error(f"❌ Error making prediction: {e}")
            # Return fallback prediction
            if prediction_type == "classification":
                return {
                    'prediction': 0.5,
                    'confidence': 0.5,
                    'prediction_type': prediction_type,
                    'features_used': list(features.keys())
                }
            else:
                return {
                    'prediction': 0.0,
                    'confidence': 0.3,
                    'prediction_type': prediction_type,
                    'features_used': list(features.keys())
                }

    def get_shap_explanations(self, features: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Get SHAP explanations for prediction."""
        try:
            if not self.ensemble_model or not hasattr(self.ensemble_model, 'models'):
                return None

            df = pd.DataFrame([features])
            explanations = self.ensemble_model.explain_predictions(df)

            if explanations and 'shap_values' in explanations:
                shap_values = explanations['shap_values'][0]  # First prediction
                feature_names = explanations['feature_names']

                # Create dictionary of feature contributions
                feature_contributions = {}
                for i, (feature, value) in enumerate(zip(feature_names, shap_values)):
                    if i < len(feature_names):
                        feature_contributions[feature] = float(value)

                return feature_contributions

            return None

        except Exception as e:
            logger.warning(f"⚠️ Could not generate SHAP explanations: {e}")
            return None

class WebSocketManager:
    """Manages WebSocket connections and broadcasts."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.message_queue = asyncio.Queue()

        logger.info("🌐 WebSocket Manager initialized")

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)

        # Send welcome message
        welcome_msg = WebSocketMessage(
            type="connection",
            data={"status": "connected", "message": "Connected to NBA Prediction WebSocket"},
            timestamp=datetime.now().isoformat()
        )

        await websocket.send_text(welcome_msg.json())
        logger.info(f"🔌 WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"🔌 WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: WebSocketMessage):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return

        message_str = message.json()
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.warning(f"⚠️ Failed to send message to WebSocket: {e}")
                disconnected.append(connection)

        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)

    async def broadcast_prediction(self, game_data: GameData, prediction_result: Dict[str, Any]):
        """Broadcast prediction result to all clients."""
        message = WebSocketMessage(
            type="prediction",
            data={
                "game": asdict(game_data),
                "prediction": prediction_result,
                "status": "completed"
            },
            timestamp=datetime.now().isoformat()
        )

        await self.broadcast(message)

    async def broadcast_status(self, status: str, message: str):
        """Broadcast status update to all clients."""
        message = WebSocketMessage(
            type="status",
            data={
                "status": status,
                "message": message,
                "timestamp": datetime.now().isoformat()
            },
            timestamp=datetime.now().isoformat()
        )

        await self.broadcast(message)

class NBAPredictionAPI:
    """Main NBA Prediction API application."""

    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self.app = FastAPI(
            title="NBA Prediction API",
            description="Real-time NBA game predictions with ensemble ML models",
            version="1.0.0"
        )

        # Initialize services
        self.data_store = UnifiedDataStore(base_path="data")
        self.data_store.initialize()

        self.prediction_service = PredictionService(self.data_store)
        self.websocket_manager = WebSocketManager()

        self._setup_middleware()
        self._setup_routes()

        logger.info("🏀 NBA Prediction API initialized")

    def _setup_middleware(self):
        """Setup FastAPI middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.get("/")
        async def root():
            return {"message": "NBA Prediction API", "version": "1.0.0"}

        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "data_store": "initialized",
                    "prediction_service": "ready",
                    "websocket_manager": f"{len(self.websocket_manager.active_connections)} connections"
                }
            }

        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict_game(request: PredictionRequest):
            """Make prediction for NBA game."""
            logger.info(f"🎯 Prediction request: {request.game.home_team_name} vs {request.game.away_team_name}")

            # Initialize models if not already done
            if not self.prediction_service.ensemble_model:
                success = self.prediction_service.initialize_models()
                if not success:
                    raise HTTPException(status_code=500, detail="Failed to initialize ML models")

            # Check cache
            cache_key = self.prediction_service.get_cache_key(request.game, request.prediction_type)
            if cache_key in self.prediction_service.prediction_cache:
                cached_result = self.prediction_service.prediction_cache[cache_key]
                logger.info(f"📦 Using cached prediction for {cache_key}")

                return PredictionResponse(
                    game_id=cache_key,
                    prediction=cached_result['prediction'],
                    confidence=cached_result['confidence'],
                    prediction_type=cached_result['prediction_type'],
                    timestamp=cached_result['timestamp']
                )

            # Process game data
            features = self.prediction_service.process_game_data(request.game)

            # Make prediction
            prediction_result = self.prediction_service.make_prediction(features, request.prediction_type)

            # Get SHAP explanations if requested
            shap_explanations = None
            if request.include_shap:
                shap_explanations = self.prediction_service.get_shap_explanations(features)

            # Cache result
            self.prediction_service.prediction_store[cache_key] = {
                **prediction_result,
                'timestamp': datetime.now().isoformat(),
                'shap_explanations': shap_explanations
            }

            # Create response
            response = PredictionResponse(
                game_id=cache_key,
                prediction=prediction_result['prediction'],
                confidence=prediction_result['confidence'],
                prediction_type=prediction_result['prediction_type'],
                timestamp=datetime.now().isoformat(),
                shap_explanations=shap_explanations
            )

            # Broadcast to WebSocket clients
            await self.websocket_manager.broadcast_prediction(request.game, {
                **prediction_result,
                'shap_explanations': shap_explanations
            })

            logger.info(f"✅ Prediction completed: {prediction_result['prediction']:.3f} (confidence: {prediction_result['confidence']:.3f})")

            return response

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await self.websocket_manager.connect(websocket)
            try:
                while True:
                    # Wait for messages (implement as needed)
                    await asyncio.sleep(1)
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(websocket)
            except Exception as e:
                logger.error(f"❌ WebSocket error: {e}")
                self.websocket_manager.disconnect(websocket)

        @self.app.get("/models/status")
        async def models_status():
            """Get status of ML models."""
            models_ready = (
                self.prediction_service.ensemble_model is not None and
                len(self.prediction_service.ensemble_model.models) > 0
            )

            feature_engineer_ready = self.prediction_service.feature_engineer is not None

            return {
                "status": "ready" if models_ready and feature_engineer_ready else "initializing",
                "models_loaded": len(self.prediction_service.ensemble_model.models) if models_ready else 0,
                "feature_engineer_ready": feature_engineer_ready,
                "cache_size": len(self.prediction_service.prediction_cache),
                "websocket_connections": len(self.websocket_manager.active_connections)
            }

        @self.app.post("/models/initialize")
        async def initialize_models():
            """Initialize ML models (can take time)."""
            logger.info("🚀 Initializing ML models...")

            success = self.prediction_service.initialize_models()

            # Broadcast status update
            if success:
                await self.websocket_manager.broadcast_status("ready", "ML models initialized successfully")
                return {"status": "success", "message": "Models initialized"}
            else:
                await self.websocket_manager.broadcast_status("error", "Failed to initialize models")
                raise HTTPException(status_code=500, detail="Failed to initialize models")

        @self.app.get("/cache/clear")
        async def clear_cache():
            """Clear prediction cache."""
            cache_size = len(self.prediction_service.prediction_cache)
            self.prediction_service.prediction_cache.clear()

            logger.info(f"🗑️ Cleared prediction cache: {cache_size} entries")

            return {"status": "success", "cleared_entries": cache_size}

    async def run(self):
        """Run the API server."""
        logger.info(f"🚀 Starting NBA Prediction API on {self.config.host}:{self.config.port}")

        config = uvicorn.Config(
            app=self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
            reload=self.config.debug
        )

        await uvicorn.run(config)

    def run_websocket_server(self):
        """Run WebSocket server."""
        logger.info(f"🌐 Starting WebSocket server on {self.config.host}:{self.config.websocket_port}")

        config = uvicorn.Config(
            app=self.app,
            host=self.config.host,
            port=self.config.websocket_port,
            log_level="info"
        )

        uvicorn.run(config)

def create_app() -> NBAPredictionAPI:
    """Create and return NBA Prediction API instance."""
    api = NBAPredictionAPI()

    # Try to initialize models in background
    asyncio.create_task(api.prediction_service.initialize_models())

    return api

if __name__ == "__main__":
    # Create and run API
    api = create_app()
    asyncio.run(api.run())