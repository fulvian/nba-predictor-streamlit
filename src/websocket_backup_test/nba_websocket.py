#!/usr/bin/env python3
"""
🌐 NBA WebSocket Handler
Context7-compliant WebSocket implementation for real-time NBA predictions.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class WebSocketMessage(BaseModel):
    """WebSocket message structure."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime
    client_id: Optional[str] = None

class PredictionBroadcast(BaseModel):
    """Real-time prediction broadcast message."""
    game_id: str
    home_team: str
    away_team: str
    prediction: Dict[str, Any]
    confidence: float
    timestamp: datetime
    model_version: str

class NBAWebSocketHandler:
    """Context7-compliant WebSocket handler for NBA predictions."""

    def __init__(self):
        """Initialize WebSocket handler."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # game_id -> set of client_ids
        self.client_info: Dict[str, Dict[str, Any]] = {}
        self.message_queue = asyncio.Queue()
        self.broadcast_task = None

    async def connect(self, websocket: WebSocket, client_id: str) -> bool:
        """
        Accept WebSocket connection and register client.

        Args:
            websocket: WebSocket connection
            client_id: Unique client identifier

        Returns:
            True if connection successful, False otherwise
        """
        try:
            await websocket.accept()

            # Store connection
            self.active_connections[client_id] = websocket

            # Initialize client info
            self.client_info[client_id] = {
                'connected_at': datetime.now(),
                'subscriptions': set(),
                'message_count': 0,
                'last_ping': datetime.now()
            }

            logger.info(f"WebSocket client connected: {client_id}")
            logger.info(f"Total active connections: {len(self.active_connections)}")

            # Send welcome message
            await self.send_personal_message({
                'type': 'connection_established',
                'data': {
                    'client_id': client_id,
                    'server_time': datetime.now().isoformat(),
                    'available_subscriptions': ['games', 'predictions', 'scores']
                }
            }, client_id)

            return True

        except Exception as e:
            logger.error(f"WebSocket connection failed for {client_id}: {e}")
            return False

    async def disconnect(self, client_id: str):
        """
        Handle WebSocket disconnection.

        Args:
            client_id: Client identifier
        """
        try:
            # Remove from active connections
            if client_id in self.active_connections:
                del self.active_connections[client_id]

            # Remove from all subscriptions
            for game_id in list(self.subscriptions.keys()):
                if client_id in self.subscriptions[game_id]:
                    self.subscriptions[game_id].remove(client_id)

                    # Clean up empty subscription sets
                    if not self.subscriptions[game_id]:
                        del self.subscriptions[game_id]

            # Remove client info
            if client_id in self.client_info:
                connected_at = self.client_info[client_id]['connected_at']
                duration = datetime.now() - connected_at
                logger.info(f"WebSocket client disconnected: {client_id} (duration: {duration})")
                del self.client_info[client_id]

            logger.info(f"Total active connections: {len(self.active_connections)}")

        except Exception as e:
            logger.error(f"Error during WebSocket disconnect for {client_id}: {e}")

    async def send_personal_message(self, message: Dict[str, Any], client_id: str) -> bool:
        """
        Send message to specific client.

        Args:
            message: Message to send
            client_id: Target client identifier

        Returns:
            True if message sent successfully, False otherwise
        """
        try:
            if client_id not in self.active_connections:
                logger.warning(f"Client {client_id} not connected")
                return False

            websocket = self.active_connections[client_id]

            # Add timestamp and client_id
            message['timestamp'] = datetime.now().isoformat()
            message['client_id'] = client_id

            await websocket.send_text(json.dumps(message))

            # Update client message count
            if client_id in self.client_info:
                self.client_info[client_id]['message_count'] += 1

            return True

        except Exception as e:
            logger.error(f"Failed to send personal message to {client_id}: {e}")

            # Connection might be broken, remove client
            await self.disconnect(client_id)
            return False

    async def broadcast_message(self, message: Dict[str, Any], subscription_type: str = None, game_id: str = None) -> int:
        """
        Broadcast message to subscribed clients.

        Args:
            message: Message to broadcast
            subscription_type: Type of subscription ('games', 'predictions', 'scores')
            game_id: Specific game ID for targeted broadcasts

        Returns:
            Number of clients message was sent to
        """
        try:
            # Determine target clients
            target_clients = set()

            if game_id and game_id in self.subscriptions:
                # Clients subscribed to specific game
                target_clients.update(self.subscriptions[game_id])
            elif subscription_type:
                # Clients subscribed to type
                for client_id, info in self.client_info.items():
                    if subscription_type in info.get('subscriptions', set()):
                        target_clients.add(client_id)
            else:
                # All connected clients
                target_clients = set(self.active_connections.keys())

            # Send message to all target clients
            message['timestamp'] = datetime.now().isoformat()
            message['broadcast_type'] = subscription_type or 'all'

            if game_id:
                message['game_id'] = game_id

            sent_count = 0
            failed_clients = []

            for client_id in target_clients:
                success = await self.send_personal_message(message, client_id)
                if success:
                    sent_count += 1
                else:
                    failed_clients.append(client_id)

            # Clean up failed clients
            for client_id in failed_clients:
                await self.disconnect(client_id)

            logger.info(f"Broadcast message sent to {sent_count} clients "
                       f"(subscription: {subscription_type}, game_id: {game_id})")

            return sent_count

        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
            return 0

    async def handle_subscription(self, client_id: str, subscription_data: Dict[str, Any]) -> bool:
        """
        Handle client subscription request.

        Args:
            client_id: Client identifier
            subscription_data: Subscription details

        Returns:
            True if subscription handled successfully, False otherwise
        """
        try:
            action = subscription_data.get('action')  # 'subscribe' or 'unsubscribe'
            subscription_type = subscription_data.get('type')  # 'games', 'predictions', 'scores'
            game_id = subscription_data.get('game_id')  # optional specific game

            if client_id not in self.client_info:
                logger.warning(f"Subscription attempt from unknown client: {client_id}")
                return False

            client_subscriptions = self.client_info[client_id]['subscriptions']

            if action == 'subscribe':
                if game_id:
                    # Subscribe to specific game
                    if game_id not in self.subscriptions:
                        self.subscriptions[game_id] = set()
                    self.subscriptions[game_id].add(client_id)

                    # Add to client's game subscriptions
                    if 'games' not in client_subscriptions:
                        client_subscriptions.add('games')

                if subscription_type:
                    client_subscriptions.add(subscription_type)

                logger.info(f"Client {client_id} subscribed to {subscription_type or 'game:' + game_id}")

                # Send confirmation
                await self.send_personal_message({
                    'type': 'subscription_confirmed',
                    'data': {
                        'action': action,
                        'subscription_type': subscription_type,
                        'game_id': game_id,
                        'current_subscriptions': list(client_subscriptions)
                    }
                }, client_id)

            elif action == 'unsubscribe':
                if game_id and game_id in self.subscriptions:
                    self.subscriptions[game_id].discard(client_id)

                    # Clean up empty subscription sets
                    if not self.subscriptions[game_id]:
                        del self.subscriptions[game_id]

                if subscription_type:
                    client_subscriptions.discard(subscription_type)

                logger.info(f"Client {client_id} unsubscribed from {subscription_type or 'game:' + game_id}")

                # Send confirmation
                await self.send_personal_message({
                    'type': 'unsubscription_confirmed',
                    'data': {
                        'action': action,
                        'subscription_type': subscription_type,
                        'game_id': game_id,
                        'current_subscriptions': list(client_subscriptions)
                    }
                }, client_id)

            return True

        except Exception as e:
            logger.error(f"Failed to handle subscription for {client_id}: {e}")
            return False

    async def broadcast_prediction(self, prediction: PredictionBroadcast) -> int:
        """
        Broadcast real-time prediction to subscribers.

        Args:
            prediction: Prediction data to broadcast

        Returns:
            Number of clients prediction was sent to
        """
        try:
            message = {
                'type': 'prediction_update',
                'data': {
                    'game_id': prediction.game_id,
                    'home_team': prediction.home_team,
                    'away_team': prediction.away_team,
                    'prediction': prediction.prediction,
                    'confidence': prediction.confidence,
                    'model_version': prediction.model_version,
                    'prediction_time': prediction.timestamp.isoformat()
                }
            }

            return await self.broadcast_message(
                message=message,
                subscription_type='predictions',
                game_id=prediction.game_id
            )

        except Exception as e:
            logger.error(f"Failed to broadcast prediction: {e}")
            return 0

    async def broadcast_score_update(self, game_data: Dict[str, Any]) -> int:
        """
        Broadcast live score update to subscribers.

        Args:
            game_data: Live game data with scores

        Returns:
            Number of clients score update was sent to
        """
        try:
            message = {
                'type': 'score_update',
                'data': game_data
            }

            return await self.broadcast_message(
                message=message,
                subscription_type='scores',
                game_id=game_data.get('game_id')
            )

        except Exception as e:
            logger.error(f"Failed to broadcast score update: {e}")
            return 0

    async def send_ping_to_all(self) -> int:
        """
        Send ping message to all connected clients.

        Returns:
            Number of successful pings
        """
        try:
            message = {
                'type': 'ping',
                'data': {
                    'server_time': datetime.now().isoformat(),
                    'active_connections': len(self.active_connections)
                }
            }

            return await self.broadcast_message(message)

        except Exception as e:
            logger.error(f"Failed to send ping: {e}")
            return 0

    async def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get WebSocket connection statistics.

        Returns:
            Connection statistics
        """
        try:
            stats = {
                'total_connections': len(self.active_connections),
                'total_subscriptions': len(self.subscriptions),
                'client_details': []
            }

            for client_id, info in self.client_info.items():
                duration = datetime.now() - info['connected_at']
                client_detail = {
                    'client_id': client_id,
                    'connected_at': info['connected_at'].isoformat(),
                    'duration_seconds': int(duration.total_seconds()),
                    'subscriptions': list(info['subscriptions']),
                    'message_count': info['message_count'],
                    'last_ping': info['last_ping'].isoformat()
                }
                stats['client_details'].append(client_detail)

            # Subscription breakdown
            subscription_stats = {}
            for client_info in self.client_info.values():
                for sub in client_info['subscriptions']:
                    subscription_stats[sub] = subscription_stats.get(sub, 0) + 1

            stats['subscription_breakdown'] = subscription_stats

            return stats

        except Exception as e:
            logger.error(f"Failed to get connection stats: {e}")
            return {'error': str(e)}

    async def cleanup_stale_connections(self, max_idle_minutes: int = 30) -> int:
        """
        Clean up stale connections.

        Args:
            max_idle_minutes: Maximum idle time before disconnection

        Returns:
            Number of connections cleaned up
        """
        try:
            cleaned_count = 0
            current_time = datetime.now()
            stale_clients = []

            for client_id, info in self.client_info.items():
                idle_time = current_time - info['last_ping']
                if idle_time.total_seconds() > (max_idle_minutes * 60):
                    stale_clients.append(client_id)

            for client_id in stale_clients:
                await self.disconnect(client_id)
                cleaned_count += 1

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} stale WebSocket connections")

            return cleaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup stale connections: {e}")
            return 0

class WebSocketManager:
    """Manager class for WebSocket operations."""

    def __init__(self):
        """Initialize WebSocket manager."""
        self.handler = NBAWebSocketHandler()
        self._running = False
        self._ping_task = None
        self._cleanup_task = None

    async def start_background_tasks(self):
        """Start background tasks for WebSocket management."""
        try:
            self._running = True

            # Start ping task (every 30 seconds)
            self._ping_task = asyncio.create_task(self._ping_loop())

            # Start cleanup task (every 5 minutes)
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            logger.info("WebSocket background tasks started")

        except Exception as e:
            logger.error(f"Failed to start WebSocket background tasks: {e}")

    async def stop_background_tasks(self):
        """Stop background tasks."""
        try:
            self._running = False

            if self._ping_task:
                self._ping_task.cancel()
                try:
                    await self._ping_task
                except asyncio.CancelledError:
                    pass

            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            logger.info("WebSocket background tasks stopped")

        except Exception as e:
            logger.error(f"Failed to stop WebSocket background tasks: {e}")

    async def _ping_loop(self):
        """Background task to send periodic pings."""
        while self._running:
            try:
                await asyncio.sleep(30)  # 30 seconds
                if self._running:
                    await self.handler.send_ping_to_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ping loop: {e}")

    async def _cleanup_loop(self):
        """Background task to cleanup stale connections."""
        while self._running:
            try:
                await asyncio.sleep(300)  # 5 minutes
                if self._running:
                    await self.handler.cleanup_stale_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

# Global WebSocket manager instance
websocket_manager = WebSocketManager()