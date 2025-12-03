#!/usr/bin/env python3
"""
🏥 NBA Injury Reporter - Bridge System for Legacy Compatibility
Integrates the existing NBAInjuryTracker with the legacy data provider system.
Provides unified injury data for predictive analytics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import sys

# Add src to path for NBAInjuryTracker import
sys.path.append(str(Path(__file__).parent.parent / "src"))

from nba_predictor.core.injury_tracker import NBAInjuryTracker, InjuryDataConfig

logger = logging.getLogger(__name__)

class InjuryReporter:
    """
    Bridge class that provides legacy-compatible injury reporting
    using the modern NBAInjuryTracker system.
    """

    def __init__(self, nba_data_provider=None):
        """
        Initialize injury reporter with optional data provider integration.

        Args:
            nba_data_provider: Legacy NBA data provider instance
        """
        self.data_provider = nba_data_provider

        # Initialize modern injury tracker
        config = InjuryDataConfig(
            rate_limit=0.5,  # Conservative rate limiting
            sources=['nba_api', 'basketball_reference'],
            validate_data=True
        )

        self.injury_tracker = NBAInjuryTracker(
            data_dir="data",
            config=config
        )

        # Cache for team rosters with injury status
        self.roster_cache = {}
        self.cache_expiry = 1800  # 30 minutes
        self.last_update = {}

        logger.info("🏥 InjuryReporter initialized with NBAInjuryTracker backend")

    def get_team_roster(self, team_id: int, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get team roster with injury status integrated.

        Args:
            team_id: NBA team ID
            force_refresh: Force refresh of cached data

        Returns:
            List of player dictionaries with injury status
        """
        current_time = datetime.now()

        # Check cache
        if (not force_refresh and
            team_id in self.roster_cache and
            team_id in self.last_update and
            (current_time - self.last_update[team_id]).seconds < self.cache_expiry):

            logger.debug(f"📋 Using cached roster for team {team_id}")
            return self.roster_cache[team_id]

        try:
            # Get base roster from data provider if available
            if self.data_provider:
                roster_data = self._get_roster_from_provider(team_id)
            else:
                roster_data = self._get_roster_from_injury_tracker(team_id)

            # Enhance with injury data
            enhanced_roster = self._enhance_roster_with_injuries(roster_data, team_id)

            # Cache results
            self.roster_cache[team_id] = enhanced_roster
            self.last_update[team_id] = current_time

            logger.info(f"📋 Updated roster for team {team_id}: {len(enhanced_roster)} players")
            return enhanced_roster

        except Exception as e:
            logger.error(f"❌ Failed to get team roster for {team_id}: {e}")
            return []

    def get_team_injuries(self, team_id: int) -> Dict[str, Any]:
        """
        Get comprehensive injury report for a team.

        Args:
            team_id: NBA team ID

        Returns:
            Dictionary with injury analysis and impact
        """
        try:
            # Get recent injuries from tracker
            end_date = date.today()
            start_date = end_date - timedelta(days=7)  # Last 7 days

            # Get current injury data
            roster = self.get_team_roster(team_id)
            injured_players = [p for p in roster if p.get('status', '').lower() != 'active']

            # Calculate impact metrics
            injury_impact = self._calculate_team_injury_impact(injured_players)

            return {
                'team_id': team_id,
                'injury_date': end_date,
                'injured_players': injured_players,
                'total_injured': len(injured_players),
                'injury_impact_score': injury_impact['total_impact'],
                'key_players_injured': injury_impact['key_players_injured'],
                'availability_percentage': injury_impact['availability_percentage'],
                'injury_details': {
                    player.get('PLAYER_NAME', player.get('name', 'Unknown')): {
                        'status': player.get('status', 'Unknown'),
                        'impact': player.get('injury_impact', 0),
                        'position': player.get('POSITION', player.get('position', 'Unknown'))
                    }
                    for player in injured_players
                }
            }

        except Exception as e:
            logger.error(f"❌ Failed to get team injuries for {team_id}: {e}")
            return {'team_id': team_id, 'injured_players': [], 'total_injured': 0}

    def _get_roster_from_provider(self, team_id: int) -> List[Dict[str, Any]]:
        """Get roster from legacy data provider."""
        try:
            # Use the data provider's roster functionality
            if hasattr(self.data_provider, 'get_team_roster'):
                return self.data_provider.get_team_roster(team_id)
            else:
                # Fallback: get basic team info and create minimal roster
                return self._create_minimal_roster(team_id)
        except Exception as e:
            logger.warning(f"⚠️ Could not get roster from provider: {e}")
            return self._create_minimal_roster(team_id)

    def _get_roster_from_injury_tracker(self, team_id: int) -> List[Dict[str, Any]]:
        """Get roster from injury tracker system."""
        try:
            # Get NBA teams to find team info
            teams = self.injury_tracker.get_nba_teams()
            team_info = next((t for t in teams if t['id'] == team_id), None)

            if not team_info:
                logger.warning(f"⚠️ Team {team_id} not found in NBA teams")
                return []

            # Create minimal roster structure
            return self._create_minimal_roster(team_id, team_info)

        except Exception as e:
            logger.error(f"❌ Failed to get roster from injury tracker: {e}")
            return []

    def _create_minimal_roster(self, team_id: int, team_info: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Create minimal roster structure when full data unavailable."""
        # This is a fallback - in real implementation, you'd want
        # to integrate with a proper roster API
        return []

    def _enhance_roster_with_injuries(self, roster_data: List[Dict], team_id: int) -> List[Dict[str, Any]]:
        """Enhance roster data with injury information from tracker."""
        try:
            # Get injury data from tracker
            injury_report = self.get_team_injuries(team_id)
            injured_players = injury_report.get('injured_players', [])

            # Create lookup for injury data
            injury_lookup = {}
            for player in injured_players:
                name = player.get('PLAYER_NAME', player.get('name', ''))
                if name:
                    injury_lookup[name.lower()] = player

            # Enhance roster entries
            enhanced_roster = []
            for player in roster_data:
                player_name = player.get('PLAYER_NAME', player.get('name', ''))

                # Check if player is injured
                injury_data = injury_lookup.get(player_name.lower(), {})

                # Merge player data with injury data
                enhanced_player = {
                    **player,
                    'status': injury_data.get('status', 'Active'),
                    'injury_impact': injury_data.get('injury_impact', 0.0),
                    'injury_type': injury_data.get('injury_type', 'None'),
                    'availability_probability': injury_data.get('availability_probability', 1.0)
                }

                enhanced_roster.append(enhanced_player)

            return enhanced_roster

        except Exception as e:
            logger.error(f"❌ Failed to enhance roster with injuries: {e}")
            return roster_data

    def _calculate_team_injury_impact(self, injured_players: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive injury impact metrics."""
        total_impact = 0.0
        key_players_injured = 0
        total_players = len(injured_players)

        if total_players == 0:
            return {
                'total_impact': 0.0,
                'key_players_injured': 0,
                'availability_percentage': 100.0
            }

        for player in injured_players:
            impact = player.get('injury_impact', 0.0)
            status = player.get('status', '').lower()

            # Weight impact by injury status
            if 'out' in status:
                impact_weight = 1.0
            elif 'doubtful' in status:
                impact_weight = 0.8
            elif 'questionable' in status:
                impact_weight = 0.4
            elif 'probable' in status:
                impact_weight = 0.1
            else:
                impact_weight = 0.5

            weighted_impact = impact * impact_weight
            total_impact += weighted_impact

            # Consider key players (impact > 5.0)
            if weighted_impact > 5.0:
                key_players_injured += 1

        # Calculate team availability percentage
        max_possible_impact = total_players * 10.0  # Max 10.0 impact per player
        availability_percentage = max(0, 100 - (total_impact / max_possible_impact * 100))

        return {
            'total_impact': round(total_impact, 2),
            'key_players_injured': key_players_injured,
            'availability_percentage': round(availability_percentage, 1)
        }

    def update_injury_data(self, team_id: Optional[int] = None) -> bool:
        """
        Force update of injury data.

        Args:
            team_id: Specific team ID to update, or None for all teams

        Returns:
            Success status
        """
        try:
            if team_id:
                # Update specific team
                self.get_team_roster(team_id, force_refresh=True)
                logger.info(f"🔄 Updated injury data for team {team_id}")
            else:
                # Update all cached teams
                for cached_team_id in list(self.roster_cache.keys()):
                    self.get_team_roster(cached_team_id, force_refresh=True)
                logger.info("🔄 Updated injury data for all cached teams")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to update injury data: {e}")
            return False

    def get_injury_summary(self) -> Dict[str, Any]:
        """Get overall injury system summary."""
        try:
            return {
                'system_status': 'active',
                'cached_teams': len(self.roster_cache),
                'last_updates': {
                    str(team_id): last_update.strftime('%Y-%m-%d %H:%M:%S')
                    for team_id, last_update in self.last_update.items()
                },
                'injury_tracker_stats': getattr(self.injury_tracker, 'stats', {}),
                'cache_expiry_minutes': self.cache_expiry // 60
            }
        except Exception as e:
            logger.error(f"❌ Failed to get injury summary: {e}")
            return {'system_status': 'error', 'error': str(e)}