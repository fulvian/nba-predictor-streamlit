#!/usr/bin/env python3
"""
🏀 COMPREHENSIVE NBA DATASET BUILDER
Sistema completo per ricostruire dataset NBA con coverage totale
Utilizza nba-api per recuperare TUTTE le partite delle stagioni target
"""

import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import nba-api
from nba_api.stats.endpoints import (
    leaguegamefinder,
    boxscoretraditionalv2,
    boxscoreadvancedv2,
    leaguedashteamstats,
    teamgamelog
)
from nba_api.stats.static import teams as nba_teams

from data_provider import NBADataProvider

class ComprehensiveDatasetBuilder:
    """Costruttore dataset NBA completo con coverage totale"""
    
    def __init__(self):
        self.data_provider = NBADataProvider()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'data')
        
        # Configurazione stagioni TARGET
        self.target_seasons = [
            "2019-20",  # Pre-COVID baseline
            "2020-21",  # COVID season (ridotta)
            "2021-22",  # Ritorno normalità
            "2022-23",  # Stabilizzazione
            "2023-24"   # Ultima completa
        ]
        
        self.season_types = ["Regular Season", "Playoffs"]
        self.request_delay = 0.6
        
        print("🏀 COMPREHENSIVE DATASET BUILDER INIZIALIZZATO")
        print(f"📅 Stagioni target: {', '.join(self.target_seasons)}")
        print(f"🎯 Obiettivo: ~6,000+ partite complete")

    def build_complete_dataset(self) -> pd.DataFrame:
        """Costruisce dataset completo con tutte le partite target"""
        
        print("\n🚀 === INIZIO COSTRUZIONE DATASET COMPLETO ===")
        
        all_games_data = []
        
        for season in self.target_seasons:
            season_year = int(season.split('-')[0])
            print(f"\n📅 ELABORAZIONE STAGIONE {season} ({season_year})...")
            
            season_games = self._get_all_games_for_season(season)
            if season_games:
                all_games_data.extend(season_games)
                print(f"   ✅ Raccolte {len(season_games)} partite per {season}")
            else:
                print(f"   ❌ Errore nella raccolta dati per {season}")
        
        if not all_games_data:
            print("❌ ERRORE: Nessun dato raccolto!")
            return pd.DataFrame()
        
        # Costruisci DataFrame finale
        print(f"\n🔄 COSTRUZIONE DATAFRAME FINALE...")
        final_df = pd.DataFrame(all_games_data)
        
        # Aggiungi features derivate
        final_df = self._add_derived_features(final_df)
        
        # Pulizia e validazione
        final_df = self._clean_and_validate(final_df)
        
        # Salva dataset
        output_path = os.path.join(self.data_dir, 'nba_complete_dataset.csv')
        final_df.to_csv(output_path, index=False)
        
        print(f"\n🎉 === DATASET COMPLETO COSTRUITO ===")
        print(f"📊 TOTALE PARTITE: {len(final_df):,}")
        print(f"📅 STAGIONI: {final_df['SEASON'].nunique()}")
        print(f"💾 SALVATO IN: {output_path}")
        
        # Statistiche per stagione
        season_stats = final_df['SEASON'].value_counts().sort_index()
        print(f"\n📈 DISTRIBUZIONE PER STAGIONE:")
        for season, count in season_stats.items():
            print(f"   {season}: {count:,} partite")
        
        return final_df

    def _get_all_games_for_season(self, season: str) -> List[Dict]:
        """Recupera tutte le partite per una stagione specifica"""
        
        season_games = []
        season_year = int(season.split('-')[0])
        
        for season_type in self.season_types:
            print(f"   🔄 Recupero {season_type} per {season}...")
            
            try:
                time.sleep(self.request_delay)
                
                # Usa leaguegamefinder per ottenere tutte le partite
                game_finder = leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    season_type_nullable=season_type,
                    league_id_nullable='00'
                )
                
                games_df = game_finder.get_data_frames()[0]
                
                if games_df.empty:
                    print(f"     ℹ️ Nessuna partita trovata per {season_type}")
                    continue
                
                # Filtra solo partite a casa (per evitare duplicati)
                home_games = games_df[games_df['MATCHUP'].str.contains('vs.')]
                
                print(f"     📊 Trovate {len(home_games)} partite uniche")
                
                # Processa ogni partita
                for _, game_row in home_games.iterrows():
                    game_data = self._process_single_game(game_row, season_year)
                    if game_data:
                        season_games.append(game_data)
                        
                print(f"     ✅ Processate {len(home_games)} partite per {season_type}")
                
            except Exception as e:
                print(f"     ❌ Errore in {season_type}: {e}")
                continue
        
        return season_games

    def _process_single_game(self, game_row: pd.Series, season_year: int) -> Optional[Dict]:
        """Processa una singola partita per estrarre features"""
        
        try:
            # Informazioni base partita
            game_id = game_row['GAME_ID']
            game_date = game_row['GAME_DATE']
            team_id = game_row['TEAM_ID']
            
            # Determina home/away team
            matchup = game_row['MATCHUP']
            is_home = 'vs.' in matchup
            
            # Ottieni ID squadra avversaria
            opponent_abbr = matchup.split(' vs. ' if is_home else ' @ ')[1]
            opponent_team = self._get_team_by_abbreviation(opponent_abbr)
            
            if not opponent_team:
                return None
            
            # Recupera statistiche avanzate per entrambe le squadre
            home_team_id = team_id if is_home else opponent_team['id']
            away_team_id = opponent_team['id'] if is_home else team_id
            
            # Ottieni statistiche pre-partita per entrambe le squadre
            home_stats = self._get_team_season_stats(home_team_id, season_year)
            away_stats = self._get_team_season_stats(away_team_id, season_year)
            
            if not home_stats or not away_stats:
                return None
            
            # Calcola target (punteggio totale)
            home_score = game_row['PTS'] if is_home else self._get_opponent_score(game_id, team_id)
            away_score = self._get_opponent_score(game_id, team_id) if is_home else game_row['PTS']
            
            if home_score is None or away_score is None:
                return None
            
            total_score = home_score + away_score
            
            # Costruisci record dati
            game_data = {
                'GAME_ID': game_id,
                'GAME_DATE': game_date,
                'SEASON': season_year,
                'HOME_TEAM_ID': home_team_id,
                'AWAY_TEAM_ID': away_team_id,
                'HOME_TEAM_NAME': self._get_team_name(home_team_id),
                'AWAY_TEAM_NAME': self._get_team_name(away_team_id),
                
                # Features squadra casa
                'HOME_eFG_PCT_sAvg': home_stats.get('EFG_PCT', 0.5),
                'HOME_TOV_PCT_sAvg': home_stats.get('TOV_PCT', 0.14),
                'HOME_OREB_PCT_sAvg': home_stats.get('OREB_PCT', 0.25),
                'HOME_FT_RATE_sAvg': home_stats.get('FT_RATE', 0.2),
                'HOME_ORtg_sAvg': home_stats.get('OFF_RATING', 110.0),
                'HOME_DRtg_sAvg': home_stats.get('DEF_RATING', 110.0),
                'HOME_PACE': home_stats.get('PACE', 100.0),
                
                # Features squadra trasferta
                'AWAY_eFG_PCT_sAvg': away_stats.get('EFG_PCT', 0.5),
                'AWAY_TOV_PCT_sAvg': away_stats.get('TOV_PCT', 0.14),
                'AWAY_OREB_PCT_sAvg': away_stats.get('OREB_PCT', 0.25),
                'AWAY_FT_RATE_sAvg': away_stats.get('FT_RATE', 0.2),
                'AWAY_ORtg_sAvg': away_stats.get('OFF_RATING', 110.0),
                'AWAY_DRtg_sAvg': away_stats.get('DEF_RATING', 110.0),
                'AWAY_PACE': away_stats.get('PACE', 100.0),
                
                # Features derivate
                'GAME_PACE': (home_stats.get('PACE', 100.0) + away_stats.get('PACE', 100.0)) / 2,
                'PACE_DIFFERENTIAL': home_stats.get('PACE', 100.0) - away_stats.get('PACE', 100.0),
                'HOME_OFF_vs_AWAY_DEF': home_stats.get('OFF_RATING', 110.0) - away_stats.get('DEF_RATING', 110.0),
                'AWAY_OFF_vs_HOME_DEF': away_stats.get('OFF_RATING', 110.0) - home_stats.get('DEF_RATING', 110.0),
                
                # Target
                'target_mu': total_score,
                'target_sigma': self._calculate_game_volatility(home_stats, away_stats),
                'TOTAL_SCORE': total_score,
                'HOME_SCORE': home_score,
                'AWAY_SCORE': away_score
            }
            
            return game_data
            
        except Exception as e:
            print(f"     ⚠️ Errore processing game {game_row.get('GAME_ID', 'unknown')}: {e}")
            return None

    def _get_team_season_stats(self, team_id: int, season_year: int) -> Optional[Dict]:
        """Recupera statistiche di squadra per la stagione"""
        
        season_str = f"{season_year}-{str(season_year + 1)[-2:]}"
        cache_key = f"{team_id}_{season_str}"
        
        if hasattr(self, '_team_stats_cache') and cache_key in self._team_stats_cache:
            return self._team_stats_cache[cache_key]
        
        try:
            time.sleep(self.request_delay * 0.5)  # Delay ridotto per stats
            
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season_str,
                season_type_all_star='Regular Season',
                measure_type_detailed_defense='Advanced'
            )
            
            stats_df = team_stats.get_data_frames()[0]
            team_row = stats_df[stats_df['TEAM_ID'] == team_id]
            
            if team_row.empty:
                return None
            
            row = team_row.iloc[0]
            
            stats = {
                'EFG_PCT': float(row.get('EFG_PCT', 0.5)),
                'TOV_PCT': float(row.get('TM_TOV_PCT', 0.14)),
                'OREB_PCT': float(row.get('OREB_PCT', 0.25)),
                'FT_RATE': float(row.get('FTA_RATE', 0.2)),
                'OFF_RATING': float(row.get('OFF_RATING', 110.0)),
                'DEF_RATING': float(row.get('DEF_RATING', 110.0)),
                'PACE': float(row.get('PACE', 100.0))
            }
            
            # Cache risultato
            if not hasattr(self, '_team_stats_cache'):
                self._team_stats_cache = {}
            self._team_stats_cache[cache_key] = stats
            
            return stats
            
        except Exception as e:
            print(f"     ⚠️ Errore statistiche team {team_id}: {e}")
            return None

    def _get_team_by_abbreviation(self, abbr: str) -> Optional[Dict]:
        """Trova team per abbreviazione"""
        teams = nba_teams.get_teams()
        for team in teams:
            if team['abbreviation'] == abbr:
                return team
        return None

    def _get_team_name(self, team_id: int) -> str:
        """Ottieni nome team da ID"""
        teams = nba_teams.get_teams()
        for team in teams:
            if team['id'] == team_id:
                return team['full_name']
        return "Unknown Team"

    def _get_opponent_score(self, game_id: str, team_id: int) -> Optional[int]:
        """Recupera punteggio squadra avversaria"""
        try:
            time.sleep(self.request_delay * 0.3)
            
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            team_stats = boxscore.team_stats.get_data_frame()
            
            opponent_row = team_stats[team_stats['TEAM_ID'] != team_id]
            if opponent_row.empty:
                return None
            
            return int(opponent_row.iloc[0]['PTS'])
            
        except Exception as e:
            print(f"     ⚠️ Errore recupero punteggio avversario: {e}")
            return None

    def _calculate_game_volatility(self, home_stats: Dict, away_stats: Dict) -> float:
        """Calcola volatilità prevista della partita"""
        
        # Fattori di volatilità
        pace_factor = (home_stats.get('PACE', 100) + away_stats.get('PACE', 100)) / 2
        off_efficiency = (home_stats.get('OFF_RATING', 110) + away_stats.get('OFF_RATING', 110)) / 2
        def_efficiency = (home_stats.get('DEF_RATING', 110) + away_stats.get('DEF_RATING', 110)) / 2
        
        # Formula volatilità basata su pace e efficienza
        base_volatility = 12.0  # NBA baseline
        pace_adjustment = (pace_factor - 100) * 0.1
        efficiency_adjustment = abs(off_efficiency - def_efficiency) * 0.05
        
        volatility = base_volatility + pace_adjustment + efficiency_adjustment
        
        # Clamp tra valori ragionevoli
        return max(8.0, min(20.0, volatility))

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggiunge features derivate al dataset"""
        
        print("🔧 Aggiunta features derivate...")
        
        # Media league per stagione
        df['LgAvg_ORtg_season'] = df.groupby('SEASON')['HOME_ORtg_sAvg'].transform('mean')
        
        # Features H2H (placeholder - da implementare con storico)
        df['H2H_L3_Avg_TotalScore'] = 220.0  # Default NBA average
        df['H2H_L3_Var_TotalScore'] = 15.0   # Default variance
        
        # Features momentum (placeholder - L5 games average)
        df['HOME_ORtg_L5Avg'] = df['HOME_ORtg_sAvg']  # Placeholder
        df['HOME_DRtg_L5Avg'] = df['HOME_DRtg_sAvg']  # Placeholder
        df['AWAY_ORtg_L5Avg'] = df['AWAY_ORtg_sAvg']  # Placeholder
        df['AWAY_DRtg_L5Avg'] = df['AWAY_DRtg_sAvg']  # Placeholder
        
        # Features scoring expectation
        df['TOTAL_EXPECTED_SCORING'] = (
            df['HOME_OFF_vs_AWAY_DEF'] + df['AWAY_OFF_vs_HOME_DEF']
        )
        
        # Average pace
        df['AVG_PACE'] = (df['HOME_PACE'] + df['AWAY_PACE']) / 2
        
        return df

    def _clean_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pulizia e validazione dataset"""
        
        print("🧹 Pulizia e validazione dataset...")
        
        initial_count = len(df)
        
        # Rimuovi righe con valori target invalidi
        df = df.dropna(subset=['target_mu', 'TOTAL_SCORE'])
        df = df[(df['target_mu'] >= 140) & (df['target_mu'] <= 350)]  # Range NBA realistico
        
        # Rimuovi duplicati
        df = df.drop_duplicates(subset=['GAME_ID'])
        
        # Ordina per data
        df = df.sort_values(['SEASON', 'GAME_DATE']).reset_index(drop=True)
        
        final_count = len(df)
        removed = initial_count - final_count
        
        print(f"   🧹 Rimosse {removed} righe invalide/duplicate")
        print(f"   ✅ Dataset finale: {final_count:,} partite valide")
        
        return df

    def validate_dataset_quality(self, df: pd.DataFrame) -> Dict:
        """Valida qualità del dataset costruito"""
        
        print("\n🔍 === VALIDAZIONE QUALITÀ DATASET ===")
        
        validation_results = {
            'total_games': len(df),
            'seasons_covered': df['SEASON'].nunique(),
            'date_range': {
                'start': df['GAME_DATE'].min(),
                'end': df['GAME_DATE'].max()
            },
            'average_games_per_season': len(df) / df['SEASON'].nunique(),
            'missing_values': df.isnull().sum().sum(),
            'score_distribution': {
                'mean': df['target_mu'].mean(),
                'std': df['target_mu'].std(),
                'min': df['target_mu'].min(),
                'max': df['target_mu'].max()
            }
        }
        
        # Stampa risultati
        print(f"📊 Partite totali: {validation_results['total_games']:,}")
        print(f"📅 Stagioni coperte: {validation_results['seasons_covered']}")
        print(f"🏀 Media partite/stagione: {validation_results['average_games_per_season']:.0f}")
        print(f"❌ Valori mancanti: {validation_results['missing_values']}")
        
        score_stats = validation_results['score_distribution']
        print(f"🎯 Punteggi - Media: {score_stats['mean']:.1f}, "
              f"Range: {score_stats['min']}-{score_stats['max']}")
        
        return validation_results


def main():
    """Funzione principale per costruire dataset completo"""
    
    builder = ComprehensiveDatasetBuilder()
    
    print("🚀 AVVIO COSTRUZIONE DATASET COMPLETO...")
    print("⏱️  Tempo stimato: 15-20 minuti")
    print("🔄 Questo processo recupererà ~6,000+ partite da nba-api")
    
    # Costruisci dataset
    complete_dataset = builder.build_complete_dataset()
    
    if not complete_dataset.empty:
        # Valida qualità
        validation = builder.validate_dataset_quality(complete_dataset)
        
        print("\n🎉 === COSTRUZIONE COMPLETATA ===")
        print("✅ Dataset NBA completo creato con successo!")
        print("📁 File salvato: data/nba_complete_dataset.csv")
        
        return complete_dataset
    else:
        print("\n❌ === ERRORE COSTRUZIONE ===")
        print("Impossibile costruire il dataset completo")
        return None


if __name__ == "__main__":
    main() 