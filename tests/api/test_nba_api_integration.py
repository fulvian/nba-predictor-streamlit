#!/usr/bin/env python3
"""
🧪 NBA API Integration Test
Context7-compliant test for NBA prediction API integration.
"""

import pytest
import asyncio
from datetime import datetime, date
from typing import Dict, Any

# Context7-compliant imports
from api.nba_prediction_api import NBAPredictionAPI, APIConfig, PredictionRequest, GameData
from nba_predictor.core.data_store import UnifiedDataStore

class TestNBAIntegration:
    """Context7-compliant NBA API integration tests."""

    def setup_method(self):
        """Setup test environment."""
        from pathlib import Path
        self.data_store = UnifiedDataStore(base_path=Path("/tmp/nba_test"))
        self.api_config = APIConfig(host="127.0.0.1", port=8000, debug=True)

    def test_data_store_connection(self):
        """Test data store connection."""
        # Test that data store can be instantiated
        assert self.data_store is not None

    def test_api_config_creation(self):
        """Test API configuration creation."""
        config = APIConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.debug == False

    def test_prediction_request_model(self):
        """Test prediction request Pydantic model."""
        request = PredictionRequest(
            game_id="test123",
            home_team_id=1610612747,
            away_team_id=1610612744,
            season="2024-25",
            game_date=date.today().isoformat()
        )

        assert request.game_id == "test123"
        assert request.home_team_id == 1610612747
        assert request.away_team_id == 1610612744
        assert request.season == "2024-25"

    def test_game_data_model(self):
        """Test game data Pydantic model."""
        game_data = GameData(
            home_team_id=1610612747,
            away_team_id=1610612744,
            home_team_name="Los Angeles Lakers",
            away_team_name="Golden State Warriors"
        )

        assert game_data.home_team_id == 1610612747
        assert game_data.away_team_id == 1610612744
        assert game_data.home_team_name == "Los Angeles Lakers"
        assert game_data.away_team_name == "Golden State Warriors"

    def test_api_instantiation(self):
        """Test API instantiation."""
        api = NBAPredictionAPI(self.api_config)
        assert api is not None
        assert api.config == self.api_config

    @pytest.mark.asyncio
    async def test_websocket_components(self):
        """Test WebSocket components."""
        from websocket.nba_websocket import WebSocketManager, NBAWebSocketHandler

        # Test WebSocket manager
        manager = WebSocketManager()
        assert manager is not None

        # Test WebSocket handler
        handler = NBAWebSocketHandler()
        assert handler is not None

        # Test connection stats
        stats = await handler.get_connection_stats()
        assert stats is not None
        assert 'total_connections' in stats

if __name__ == "__main__":
    pytest.main([__file__, "-v"])