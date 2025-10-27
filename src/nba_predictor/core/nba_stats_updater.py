#!/usr/bin/env python3
"""
🏀 NBA Stats Updater - Sistema di Download Statistiche Aggiornate

Sistema automatico per mantenere aggiornate le statistiche di squadre e giocatori:
- Verifica data ultima update nel persistent storage
- Scarica statistiche aggiornate da NBA API
- Salva nel data store persistente (UnifiedDataStore)
- Fornisce dati sempre aggiornati per analisi predittiva
"""

import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import polars as pl
import requests

# Import existing components
from data_provider import NBADataProvider
from src.nba_predictor.core.data_store import UnifiedDataStore

logger = logging.getLogger(__name__)

class NBAStatsUpdater:
    """
    Sistema per aggiornare statistiche NBA da API e salvarle nel persistent storage.
    """

    def __init__(self, data_provider: Optional[NBADataProvider] = None):
        """
        Inizializza l'updater di statistiche NBA.

        Args:
            data_provider: Provider dati NBA con API access
        """
        self.data_provider = data_provider or NBADataProvider()
        self.data_store = UnifiedDataStore(
            base_path="data/persistent",
            cache_enabled=True
        )
        self.data_store.initialize()

        # Configurazione API
        self.nba_api_base = "https://stats.nba.com"
        self.headers = {
            'Host': 'stats.nba.com',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'x-nba-stats-origin': 'stats',
            'Connection': 'keep-alive'
        }

        logger.info("NBA Stats Updater initialized")

    def check_and_update_stats(self, force_update: bool = False) -> Dict[str, Any]:
        """
        Verifica se le statistiche sono aggiornate e le aggiorna se necessario.

        Args:
            force_update: Forza aggiornamento anche se dati recenti

        Returns:
            Dict con risultati dell'aggiornamento
        """
        try:
            # Controlla ultima data di update
            last_update = self._get_last_stats_update()
            today = date.today()
            days_since_update = (today - last_update).days if last_update else 999

            logger.info(f"Last stats update: {last_update} ({days_since_update} days ago)")

            # Decidi se aggiornare
            should_update = force_update or days_since_update >= 1

            if should_update:
                logger.info("🔄 Starting NBA stats update...")
                return self._update_all_stats()
            else:
                logger.info("✅ Stats are up to date")
                return {
                    'updated': False,
                    'reason': 'Stats already up to date',
                    'last_update': last_update.isoformat(),
                    'days_old': days_since_update
                }

        except Exception as e:
            logger.error(f"Error in check_and_update_stats: {e}")
            return {
                'updated': False,
                'error': str(e),
                'last_update': None
            }

    def _get_last_stats_update(self) -> Optional[date]:
        """Ottiene data ultimo aggiornamento statistiche dal persistent storage."""
        try:
            # Controlla se esiste metadata per le statistiche
            metadata = self.data_store.get_metadata()

            # Filtra per statistiche giocatori
            player_stats_meta = metadata.filter(
                pl.col("table_name").str.contains("player_stats")
            )

            if player_stats_meta.height > 0:
                # Trova la data più recente
                latest_date = player_stats_meta["last_updated"].max()
                if latest_date:
                    return latest_date.date()

            return None

        except Exception as e:
            logger.error(f"Error getting last stats update: {e}")
            return None

    def _update_all_stats(self) -> Dict[str, Any]:
        """Aggiorna tutte le statistiche (giocatori e squadre)."""
        try:
            results = {
                'updated': False,
                'start_time': datetime.now().isoformat(),
                'updates': {}
            }

            # 1. Aggiorna statistiche giocatori
            player_stats_result = self._update_player_stats()
            results['updates']['player_stats'] = player_stats_result

            # 2. Aggiorna statistiche squadre
            team_stats_result = self._update_team_stats()
            results['updates']['team_stats'] = team_stats_result

            results['end_time'] = datetime.now().isoformat()
            results['updated'] = True

            logger.info("✅ NBA stats update completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error updating all stats: {e}")
            return {
                'updated': False,
                'error': str(e),
                'start_time': datetime.now().isoformat()
            }

    def _update_player_stats(self) -> Dict[str, Any]:
        """Aggiorna statistiche giocatori da NBA API."""
        try:
            logger.info("🏀 Updating player stats from NBA API...")

            # Ottieni stagione corrente (2024-25)
            current_season = 2024

            # Endpoint per statistiche giocatori
            url = f"{self.nba_api_base}/stats/leaguegamelog"
            params = {
                'LeagueID': '00',  # NBA
                'Season': current_season,
                'SeasonType': 'Regular Season',
                'PlayerOrTeam': 'P'  # Player stats
            }

            response = requests.get(url, params=params, headers=self.headers, timeout=30)
            response.raise_for_status()

            data = response.json()

            if not data.get('resultSets'):
                raise ValueError("No data returned from NBA API")

            # Processa i dati
            player_stats = self._process_player_stats_data(data, current_season)

            # Salva nel persistent storage
            today_str = date.today().strftime('%Y-%m-%d')
            file_path = self.data_store.store_player_stats(player_stats, today_str)

            logger.info(f"✅ Player stats updated: {len(player_stats)} players")
            return {
                'success': True,
                'players_count': len(player_stats),
                'season': current_season,
                'file_path': file_path
            }

        except Exception as e:
            logger.error(f"Error updating player stats: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _process_player_stats_data(self, api_data: Dict, season: int) -> pl.DataFrame:
        """Processa dati statistiche giocatori da NBA API."""
        try:
            # Estrai il dataset principale
            result_sets = api_data.get('resultSets', [])
            if not result_sets:
                raise ValueError("No result sets in API response")

            # Usa il primo result set (statistiche base giocatori)
            headers = result_sets[0]['headers']
            row_set = result_sets[0]['rowSet']

            # Converti in DataFrame
            df = pl.DataFrame(row_set, schema=[h['columnAlias'] for h in headers])

            # Filtra solo giocatori con statistiche
            if 'MIN' in df.columns:
                df = df.filter(pl.col('MIN') > 0)

            # Rinomina colonne per consistenza
            column_mapping = {
                'PLAYER_ID': 'player_id',
                'PLAYER_NAME': 'player_name',
                'TEAM_ID': 'team_id',
                'TEAM_ABBREVIATION': 'team_abbreviation',
                'AGE': 'age',
                'GP': 'games_played',
                'MIN': 'minutes_avg',
                'PTS': 'points_avg',
                'REB': 'rebounds_avg',
                'AST': 'assists_avg',
                'STL': 'steals_avg',
                'BLK': 'blocks_avg',
                'TOV': 'turnovers_avg',
                'PLUS_MINUS': 'plus_minus_avg',
                'FG_PCT': 'fg_pct',
                'FG3_PCT': 'fg3_pct',
                'FT_PCT': 'ft_pct'
            }

            # Rinomina le colonne esistenti
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename({old_col: new_col})

            # Aggiungi colonne aggiuntive
            df = df.with_columns([
                pl.lit(season).alias('season'),
                pl.lit(date.today()).alias('last_updated'),
                pl.lit("NBA_API").alias('source')
            ])

            return df

        except Exception as e:
            logger.error(f"Error processing player stats data: {e}")
            raise

    def _update_team_stats(self) -> Dict[str, Any]:
        """Aggiorna statistiche squadre da NBA API."""
        try:
            logger.info("🏀 Updating team stats from NBA API...")

            # Ottieni stagione corrente
            current_season = 2024

            # Endpoint per statistiche squadre
            url = f"{self.nba_api_base}/stats/leaguedashteamstats"
            params = {
                'LeagueID': '00',  # NBA
                'Season': current_season,
                'SeasonType': 'Regular Season',
                'PerMode': 'PerGame'
            }

            response = requests.get(url, params=params, headers=self.headers, timeout=30)
            response.raise_for_status()

            data = response.json()

            if not data.get('resultSets'):
                raise ValueError("No team data returned from NBA API")

            # Processa i dati
            team_stats = self._process_team_stats_data(data, current_season)

            # Salva nel persistent storage
            today_str = date.today().strftime('%Y-%m-%d')
            file_path = self.data_store.store_team_stats(team_stats, today_str)

            logger.info(f"✅ Team stats updated: {len(team_stats)} teams")
            return {
                'success': True,
                'teams_count': len(team_stats),
                'season': current_season,
                'file_path': file_path
            }

        except Exception as e:
            logger.error(f"Error updating team stats: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _process_team_stats_data(self, api_data: Dict, season: int) -> pl.DataFrame:
        """Processa dati statistiche squadre da NBA API."""
        try:
            result_sets = api_data.get('resultSets', [])
            if not result_sets:
                raise ValueError("No team result sets in API response")

            headers = result_sets[0]['headers']
            row_set = result_sets[0]['rowSet']

            # Converti in DataFrame
            df = pl.DataFrame(row_set, schema=[h['columnAlias'] for h in headers])

            # Mappa colonne importanti
            column_mapping = {
                'TEAM_ID': 'team_id',
                'TEAM_NAME': 'team_name',
                'GP': 'games_played',
                'W': 'wins',
                'L': 'losses',
                'W_PCT': 'win_percentage',
                'MIN': 'minutes_avg',
                'PTS': 'points_avg',
                'REB': 'rebounds_avg',
                'AST': 'assists_avg',
                'STL': 'steals_avg',
                'BLK': 'blocks_avg',
                'TOV': 'turnovers_avg',
                'PLUS_MINUS': 'plus_minus_avg',
                'FG_PCT': 'fg_pct',
                'FG3_PCT': 'fg3_pct',
                'FT_PCT': 'ft_pct'
            }

            # Rinomina colonne
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename({old_col: new_col})

            # Aggiungi colonne aggiuntive
            df = df.with_columns([
                pl.lit(season).alias('season'),
                pl.lit(date.today()).alias('last_updated'),
                pl.lit("NBA_API").alias('source')
            ])

            return df

        except Exception as e:
            logger.error(f"Error processing team stats data: {e}")
            raise

    def get_latest_player_stats(self) -> pl.DataFrame:
        """Ottieni le statistiche giocatori più recenti dal persistent storage."""
        try:
            today_str = date.today().strftime('%Y-%m-%d')

            # Prova a ottenere statistiche di oggi
            stats = self.data_store.get_player_stats(date_range=(today_str, today_str))

            if stats.height > 0:
                return stats

            # Se non ci sono statistiche di oggi, prova gli ultimi 7 giorni
            week_ago = (date.today() - timedelta(days=7)).strftime('%Y-%m-%d')
            stats = self.data_store.get_player_stats(date_range=(week_ago, today_str))

            if stats.height > 0:
                logger.info(f"Using stats from last 7 days (latest: {stats.height} records)")
                return stats

            # Fallback: usa i dati più recenti disponibili
            metadata = self.data_store.get_metadata()
            player_stats_meta = metadata.filter(
                pl.col("table_name").str.contains("player_stats")
            )

            if player_stats_meta.height > 0:
                latest_date = player_stats_meta["last_updated"].max()
                if latest_date:
                    latest_str = latest_date.strftime('%Y-%m-%d')
                    stats = self.data_store.get_player_stats(date_range=(latest_str, latest_str))
                    logger.info(f"Using fallback stats from {latest_str}")
                    return stats

            logger.warning("No player stats found in persistent storage")
            return pl.DataFrame()

        except Exception as e:
            logger.error(f"Error getting latest player stats: {e}")
            return pl.DataFrame()

    def get_latest_team_stats(self) -> pl.DataFrame:
        """Ottieni le statistiche squadre più recenti dal persistent storage."""
        try:
            today_str = date.today().strftime('%Y-%m-%d')

            # Prova a ottenere statistiche di oggi
            stats = self.data_store.get_team_stats(date_range=(today_str, today_str))

            if stats.height > 0:
                return stats

            # Fallback: usa dati più recenti
            metadata = self.data_store.get_metadata()
            team_stats_meta = metadata.filter(
                pl.col("table_name").str.contains("team_stats")
            )

            if team_stats_meta.height > 0:
                latest_date = team_stats_meta["last_updated"].max()
                if latest_date:
                    latest_str = latest_date.strftime('%Y-%m-%d')
                    stats = self.data_store.get_team_stats(date_range=(latest_str, latest_str))
                    logger.info(f"Using team stats from {latest_str}")
                    return stats

            logger.warning("No team stats found in persistent storage")
            return pl.DataFrame()

        except Exception as e:
            logger.error(f"Error getting latest team stats: {e}")
            return pl.DataFrame()


# Funzione principale per aggiornare statistiche
def update_nba_stats(force: bool = False) -> Dict[str, Any]:
    """
    Funzione principale per aggiornare statistiche NBA.

    Args:
        force: Forza aggiornamento anche se dati recenti

    Returns:
        Risultati dell'aggiornamento
    """
    try:
        updater = NBAStatsUpdater()
        return updater.check_and_update_stats(force_update=force)

    except Exception as e:
        logger.error(f"Error in update_nba_stats: {e}")
        return {
            'updated': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # Test dell'updater
    print("🏀 Test NBA Stats Updater")

    # Verifica e aggiorna statistiche
    result = update_nba_stats()

    print(f"Update result: {result}")

    if result.get('updated'):
        print("✅ Stats updated successfully")
    else:
        print(f"ℹ️ Stats not updated: {result.get('reason', 'Unknown reason')}")