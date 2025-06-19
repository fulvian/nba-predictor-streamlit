# momentum_dataset_builder.py
"""
Script per costruire un dataset di training per il modello di momentum ML.
Questo script genera un dataset dove ogni riga rappresenta una partita e contiene
le feature di momentum pre-partita per entrambe le squadre e una variabile target
che rappresenta la deviazione del punteggio finale rispetto a una linea di base.
"""

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
from data_provider import NBADataProvider
from advanced_player_momentum_predictor import AdvancedPlayerMomentumPredictor
import warnings
import sys

warnings.filterwarnings('ignore', category=FutureWarning)

class MomentumDatasetBuilder:
    """
    Costruisce un dataset per l'addestramento di un modello di momentum.
    """
    
    def __init__(self, data_provider: NBADataProvider):
        """
        Inizializza il builder.

        Args:
            data_provider (NBADataProvider): Istanza per accedere ai dati NBA.
        """
        self.data_provider = data_provider
        self.momentum_predictor = AdvancedPlayerMomentumPredictor(nba_data_provider=data_provider)
        self.output_dir = os.path.join(os.path.dirname(__file__), 'data', 'momentum_v2')
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.output_dir, 'momentum_training_checkpoint.csv')

    def build_dataset_for_seasons(self, seasons: list[str], include_playoffs: bool = True):
        """
        Costruisce un dataset di momentum per le stagioni specificate.
        
        Args:
            seasons: Lista delle stagioni da includere (es. ['2024-25', '2023-24'])
            include_playoffs: Se True include anche i playoff, se False solo Regular Season
        """
        playoffs_text = "Regular Season + Playoffs" if include_playoffs else "Solo Regular Season"
        print(f"🚀 Avvio costruzione dataset di momentum per le stagioni: {seasons}")
        print(f"   📋 Modalità: {playoffs_text}")
        
        # STEP 1: Carica checkpoint esistente se presente
        processed_games_df = pd.DataFrame()
        processed_game_ids = set()
        
        if os.path.exists(self.checkpoint_path):
            print(f"🔄 Trovato file di checkpoint. Caricamento progressi da: {self.checkpoint_path}")
            try:
                processed_games_df = pd.read_csv(self.checkpoint_path)
                if not processed_games_df.empty:
                    # Pulisci e normalizza i dati esistenti
                    processed_games_df = processed_games_df.drop_duplicates(subset=['game_id'], keep='last')
                    processed_games_df['game_id'] = processed_games_df['game_id'].astype(str).str.lstrip('0')
                    processed_game_ids = set(processed_games_df['game_id'])
                    print(f"   - Caricati dati di {len(processed_game_ids)} partite processate.")
                else:
                    print(f"   - File checkpoint vuoto, ripartendo da zero.")
            except Exception as e:
                print(f"   - ⚠️ Errore caricamento checkpoint: {e}. Ripartendo da zero.")
                processed_games_df = pd.DataFrame()
                processed_game_ids = set()
        else:
            print("   - Nessun checkpoint trovato, creazione nuovo dataset.")
            # SOLO qui puliamo il cache per forzare nuovi dati freschi
            for season in seasons:
                cache_key = f"season_log_{season}"
                if hasattr(self.data_provider, cache_key):
                    delattr(self.data_provider, cache_key)
                    print(f"🔄 Cache pulito per stagione {season}")

        # STEP 2: Recupera le partite delle stagioni (configurabile)
        all_season_games = []
        for season in seasons:
            print(f"\nRecupero partite per la stagione {season}...")
            if include_playoffs:
                # Include sia Regular Season che Playoffs
                game_log = self.data_provider.get_season_game_log(season)
            else:
                # Solo Regular Season
                game_log = self.data_provider.get_season_game_log(season, season_type='Regular Season')
            
            if game_log is not None and not game_log.empty:
                all_season_games.append(game_log)
        
        if not all_season_games:
            print("❌ Nessuna partita trovata per le stagioni specificate.")
            return
            
        # Combina e ordina tutte le partite in modo deterministico
        combined_game_log = pd.concat(all_season_games, ignore_index=True)
        combined_game_log = combined_game_log.sort_values(by=['GAME_DATE', 'GAME_ID']).reset_index(drop=True)
        
        # Crea una lista ordinata di tutti i game_id per un tracking consistente
        all_game_ids_ordered = [str(gid).lstrip('0') for gid in combined_game_log['GAME_ID']]
        
        total_games_count = len(combined_game_log)
        print(f"   - Trovate {total_games_count} partite totali nelle stagioni selezionate.")
        print(f"   - Partite già processate: {len(processed_game_ids)}")
        print(f"   - Partite rimanenti: {total_games_count - len(processed_game_ids)}")

        # 🚀 STEP 2.5: Pre-carica roster per le squadre coinvolte (OTTIMIZZAZIONE)
        print("🚀 Pre-caricamento roster delle squadre per ottimizzare il processo...")
        unique_team_ids = set()
        for _, game in combined_game_log.iterrows():
            unique_team_ids.add(game['HOME_TEAM_ID'])
            unique_team_ids.add(game['AWAY_TEAM_ID'])
        
        print(f"   - Trovate {len(unique_team_ids)} squadre uniche da pre-caricare")
        failed_rosters = 0
        for i, team_id in enumerate(unique_team_ids, 1):
            try:
                roster = self.data_provider.get_team_roster(team_id)
                if roster is not None:
                    print(f"   - ✅ [{i}/{len(unique_team_ids)}] Roster caricato per team {team_id}")
                else:
                    failed_rosters += 1
                    print(f"   - ⚠️ [{i}/{len(unique_team_ids)}] Fallito roster per team {team_id}")
            except KeyboardInterrupt:
                print("   - ⚠️ Pre-caricamento interrotto dall'utente")
                break
            except Exception as e:
                failed_rosters += 1
                print(f"   - ❌ [{i}/{len(unique_team_ids)}] Errore roster team {team_id}: {e}")
        
        if failed_rosters == 0:
            print("🎉 Tutti i roster pre-caricati con successo! Velocità massima abilitata.")
        else:
            print(f"⚠️ {failed_rosters} roster falliti, ma possiamo comunque procedere.")

        # STEP 3: Processa le partite in ordine sequenziale deterministico
        new_games_processed = 0
        batch_size = 5  # Salva ogni 5 partite
        
        try:
            for index, game in combined_game_log.iterrows():
                # Normalizza game_id per confronto
                game_id_str = str(game['GAME_ID']).lstrip('0')
                
                # Determina il numero di partita corrente (1-based) per il display
                current_progress = index + 1
                
                # Salta se già processata
                if game_id_str in processed_game_ids:
                    print(f"  - Saltando Partita {current_progress}/{total_games_count}: {game['MATCHUP']} il {game['GAME_DATE']} (già processata)")
                    continue

                # Trova il log della stagione corretta
                game_season = game['SEASON_YEAR']
                current_season_log = next((log for log in all_season_games 
                                         if not log.empty and log['SEASON_YEAR'].iloc[0] == game_season), None)

                if current_season_log is None:
                    print(f"  - ⚠️ Log stagione {game_season} non trovato per game {game['GAME_ID']}. Salto.")
                    continue

                print(f"  - Processando Partita {current_progress}/{total_games_count}: {game['MATCHUP']} il {game['GAME_DATE']}...")
                
                try:
                    game_features = self._extract_features_for_game(game, current_season_log)
                    if game_features:
                        # Aggiungi alle partite processate in memoria ATOMICAMENTE
                        new_row = pd.DataFrame([game_features])
                        processed_games_df = pd.concat([processed_games_df, new_row], ignore_index=True)
                        processed_game_ids.add(game_id_str)
                        
                        # Salva checkpoint ogni batch_size partite processate in questa sessione
                        if new_games_processed % batch_size == 0:
                            self._save_checkpoint(processed_games_df)
                            print(f"     💾 Checkpoint salvato: {len(processed_games_df)} partite totali.")
                    else:
                        # Se fallisce l'estrazione features, non incrementiamo il contatore
                        new_games_processed -= 1
                        print(f"    - ⚠️ Saltata partita {current_progress} per errore estrazione features.")
                        
                except KeyboardInterrupt:
                    # Se l'interruzione avviene durante l'elaborazione, salva subito
                    print(f"    - ⚠️ Interruzione durante elaborazione partita {current_progress}. Salvataggio immediato...")
                    self._save_checkpoint(processed_games_df)
                    raise  # Rilancia l'eccezione per uscire dal loop principale
                
                # 🚀 RIMOSSO: time.sleep(0.1) - Non necessario, il rate limiting è gestito dal data provider

        except KeyboardInterrupt:
            print(f"\n⚠️ Interruzione utente rilevata. Salvataggio checkpoint finale...")
        except Exception as e:
            print(f"\n❌ Errore durante elaborazione: {e}. Salvataggio checkpoint...")
        finally:
            # Salvataggio finale SEMPRE garantito
            if new_games_processed > 0:  # Salva solo se abbiamo processato nuove partite
                self._save_checkpoint(processed_games_df)
                print(f"💾 Checkpoint finale salvato con {len(processed_games_df)} partite.")

        # STEP 4: Verifica completamento e finalizzazione
        if len(processed_games_df) >= total_games_count:
            print("\n✅ Tutte le partite processate! Finalizzazione dataset...")
            final_output_path = os.path.join(self.output_dir, 'momentum_training_dataset.csv')
            processed_games_df.to_csv(final_output_path, index=False)
            print(f"   - Dataset finale salvato in: {final_output_path}")
            
            # Rimuovi checkpoint
            if os.path.exists(self.checkpoint_path):
                os.remove(self.checkpoint_path)
                print(f"   - Checkpoint rimosso.")
        else:
            print(f"\n📊 Sessione completata: {len(processed_games_df)}/{total_games_count} partite processate.")
            
            # 🚀 STATISTICHE PERFORMANCE
            if hasattr(self.data_provider, 'api_call_times') and self.data_provider.api_call_times:
                calls_last_minute = len(self.data_provider.api_call_times)
                avg_delay = self.data_provider.current_delay
                print(f"   ⚡ Performance API: {calls_last_minute} chiamate/min, delay medio: {avg_delay:.2f}s")
                print(f"   🎯 Cache roster hit: {len(self.data_provider.roster_cache)} team cachati")
            
            print(f"   - Per continuare, riesegui lo script.")

    def _save_checkpoint(self, df: pd.DataFrame):
        """
        Salva il checkpoint in modo atomico e sicuro.
        """
        try:
            # Salvataggio atomico con file temporaneo
            temp_path = self.checkpoint_path + '.tmp'
            df.to_csv(temp_path, index=False)
            
            # Sposta il file temporaneo sostituendo quello vecchio
            if os.path.exists(temp_path):
                os.replace(temp_path, self.checkpoint_path)
        except Exception as e:
            print(f"❌ Errore salvamento checkpoint: {e}")
            # Cleanup in caso di errore
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _extract_features_for_game(self, game_row: pd.Series, season_log: pd.DataFrame) -> dict:
        """
        Estrae le feature per una singola partita.
        """
        try:
            home_team_id = game_row['HOME_TEAM_ID']
            away_team_id = game_row['AWAY_TEAM_ID']
            game_date = pd.to_datetime(game_row['GAME_DATE'])

            # Calcola le feature di momentum pre-partita
            home_momentum = self._get_team_momentum_features(home_team_id, game_date, season_log)
            away_momentum = self._get_team_momentum_features(away_team_id, game_date, season_log)

            # Calcola la linea di base e il target
            baseline_total = self._calculate_baseline_total(home_team_id, away_team_id, game_date, season_log)
            actual_total = game_row['PTS_home'] + game_row['PTS_away']
            score_deviation = actual_total - baseline_total
            
            # Combina le feature
            features = {
                'game_id': str(game_row['GAME_ID']).lstrip('0'),  # Normalizza rimuovendo zeri iniziali
                'game_date': game_date,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'baseline_total': baseline_total,
                'actual_total': actual_total,
                'score_deviation': score_deviation  # Target Variable
            }
            
            # Aggiunge le feature di momentum con prefissi
            for key, value in home_momentum.items():
                features[f'home_{key}'] = value
            for key, value in away_momentum.items():
                features[f'away_{key}'] = value
                
            # Aggiunge feature di confronto
            features['momentum_diff'] = home_momentum.get('momentum_score', 50) - away_momentum.get('momentum_score', 50)
            
            return features

        except Exception as e:
            print(f"    - ⚠️ Errore estrazione features per game {game_row.get('GAME_ID')}: {e}")
            return None

    def _get_team_momentum_features(self, team_id: int, game_date: datetime, season_log: pd.DataFrame) -> dict:
        """
        Calcola un set di feature di momentum aggregate per un team prima di una partita.
        🚀 VERSIONE OTTIMIZZATA: Usa dati statici del roster senza chiamate API ai game logs dei giocatori.
        """
        # Ottieni il roster dal cache (già pre-caricato)
        roster_df = self.data_provider.get_team_roster(team_id)
        if roster_df is None or roster_df.empty:
            return self._get_default_momentum_features()

        # 🚀 CALCOLO MOMENTUM SEMPLIFICATO SENZA CHIAMATE API
        # Invece di usare game logs individuali, usa statistiche aggregate basate su posizione e status
        try:
            features = self._calculate_lightweight_momentum(roster_df, team_id, game_date, season_log)
            return features
        except Exception as e:
            print(f"    - ⚠️ Errore calcolo momentum per team {team_id}: {e}")
            return self._get_default_momentum_features()

    def _get_default_momentum_features(self) -> dict:
        """Feature di default quando non è possibile calcolare il momentum."""
        return {
            'momentum_score': 50.0,
            'hot_hand_players': 0,
            'avg_player_momentum': 50.0,
            'avg_player_weighted_contribution': 50.0,
            'team_offensive_potential': 50.0,
            'team_defensive_potential': 50.0
        }

    def _calculate_lightweight_momentum(self, roster_df: pd.DataFrame, team_id: int, game_date: datetime, season_log: pd.DataFrame) -> dict:
        """
        Calcola feature di momentum usando solo dati del roster e statistiche aggregate della squadra.
        NON fa chiamate API ai game logs individuali.
        """
        # Analizza la composizione del roster
        starters_count = len(roster_df[roster_df.get('ROTATION_STATUS', 'BENCH') == 'STARTER'])
        bench_players = len(roster_df[roster_df.get('ROTATION_STATUS', 'BENCH') == 'BENCH'])
        
        # Calcola un punteggio basato sulla "qualità" del roster
        # Usa statistiche semplici come anni di esperienza, posizioni
        avg_experience = roster_df.get('SEASON_EXP', 3).mean() if 'SEASON_EXP' in roster_df.columns else 3.0
        total_players = len(roster_df)
        
        # Ottieni performance recente della squadra dai game logs
        team_recent_performance = self._get_team_recent_performance(team_id, game_date, season_log)
        
        # Combina fattori per creare un momentum score semplificato
        roster_quality_score = min(100, 40 + (avg_experience * 2) + (starters_count * 3))
        recent_performance_score = team_recent_performance.get('performance_score', 50.0)
        
        # Weighted average
        momentum_score = (roster_quality_score * 0.4) + (recent_performance_score * 0.6)
        
        # Simula hot hand players basato su performance recente
        hot_hand_estimate = max(0, int((recent_performance_score - 50) / 10))
        
        features = {
            'momentum_score': float(momentum_score),
            'hot_hand_players': hot_hand_estimate,
            'avg_player_momentum': float(momentum_score),
            'avg_player_weighted_contribution': float(momentum_score * 0.8),
            'team_offensive_potential': float(team_recent_performance.get('offensive_rating', 110)),
            'team_defensive_potential': float(team_recent_performance.get('defensive_rating', 110))
        }
        
        return features

    def _get_team_recent_performance(self, team_id: int, game_date: datetime, season_log: pd.DataFrame, last_n_games: int = 5) -> dict:
        """
        Calcola performance recente di una squadra dai game logs disponibili.
        """
        # Filtra le partite della squadra prima della data corrente
        team_games = season_log[
            (season_log['TEAM_ID'] == team_id) & 
            (pd.to_datetime(season_log['GAME_DATE']) < game_date)
        ].sort_values('GAME_DATE', ascending=False).head(last_n_games)
        
        if team_games.empty:
            return {'performance_score': 50.0, 'offensive_rating': 110.0, 'defensive_rating': 110.0}
        
        # Calcola statistiche semplici
        avg_points = team_games['PTS'].mean()
        avg_fg_pct = team_games['FG_PCT'].mean() if 'FG_PCT' in team_games.columns else 0.45
        avg_plus_minus = team_games['PLUS_MINUS'].mean() if 'PLUS_MINUS' in team_games.columns else 0
        
        # Converti in performance score (30-70 range tipico)
        performance_score = 50 + (avg_plus_minus * 2) + ((avg_fg_pct - 0.45) * 100)
        performance_score = max(30, min(70, performance_score))
        
        # Stima rating approssimativo
        offensive_rating = 105 + (avg_points - 110) * 0.5
        defensive_rating = 115 - (avg_plus_minus * 0.5)
        
        return {
            'performance_score': float(performance_score),
            'offensive_rating': float(offensive_rating),
            'defensive_rating': float(defensive_rating)
        }

    def _calculate_baseline_total(self, home_team_id: int, away_team_id: int, game_date: datetime, season_log: pd.DataFrame) -> float:
        """
        Calcola un punteggio totale di base usando le medie stagionali (Ortg, Drtg, Pace) fino a quella data.
        """
        historical_games = season_log[pd.to_datetime(season_log['GAME_DATE']) < game_date]
        
        # Statistiche Home Team
        home_stats = self._get_team_historical_advanced_stats(home_team_id, historical_games)
        
        # Statistiche Away Team
        away_stats = self._get_team_historical_advanced_stats(away_team_id, historical_games)
        
        # Calcolo della linea di base secondo la formula di Dean Oliver
        avg_pace = (home_stats['pace'] + away_stats['pace']) / 2
        
        home_expected_score = (home_stats['ortg'] / 100) * avg_pace
        away_expected_score = (away_stats['ortg'] / 100) * avg_pace
        
        baseline = home_expected_score + away_expected_score
        return baseline if not np.isnan(baseline) else 220.0

    def _get_team_historical_advanced_stats(self, team_id: int, historical_games: pd.DataFrame) -> dict:
        """Helper per calcolare le medie delle statistiche avanzate di un team."""
        team_games = historical_games[historical_games['TEAM_ID'] == team_id]
        
        if team_games.empty or len(team_games) < 5: # Minimo 5 partite per avere medie stabili
            return {'ortg': 110, 'drtg': 110, 'pace': 100}

        # Calcolo approssimativo di Ortg, Drtg, Pace dai game log
        # Nota: La NBA API non fornisce Ortg/Drtg/Pace per singola partita nel leaguegamelog
        # Usiamo un'approssimazione basata sui punti
        possessions_approx = team_games['FGA'] - team_games['OREB'] + team_games['TOV'] + (0.44 * team_games['FTA'])
        
        # Filtra possessi non validi
        valid_possessions = possessions_approx[possessions_approx > 0]
        if valid_possessions.empty:
            return {'ortg': 110, 'drtg': 110, 'pace': 100}
            
        # Punti segnati e subiti dalla prospettiva del TEAM_ID
        pts_for = team_games['PTS']
        
        # Per i punti subiti, dobbiamo trovare la partita dell'avversario
        opponent_pts_list = []
        for _, game in team_games.iterrows():
            opponent_row = historical_games[(historical_games['GAME_ID'] == game['GAME_ID']) & (historical_games['TEAM_ID'] != team_id)]
            if not opponent_row.empty:
                opponent_pts_list.append(opponent_row.iloc[0]['PTS'])
        
        if not opponent_pts_list:
            return {'ortg': 110, 'drtg': 110, 'pace': 100}

        pts_against = pd.Series(opponent_pts_list, index=valid_possessions.index)

        ortg = (pts_for / valid_possessions * 100).mean()
        drtg = (pts_against / valid_possessions * 100).mean()
        
        # Pace (possessi per 48 minuti)
        minutes_played = team_games['MIN']
        pace = (possessions_approx / minutes_played * 48).mean()
        
        return {
            'ortg': ortg if not np.isnan(ortg) else 110,
            'drtg': drtg if not np.isnan(drtg) else 110,
            'pace': pace if not np.isnan(pace) else 100
        }


def main():
    """
    Funzione principale per eseguire il processo di costruzione del dataset.
    """
    print("Inizializzazione del Data Provider...")
    data_provider = NBADataProvider()
    
    builder = MomentumDatasetBuilder(data_provider)
    
    # Definisci le stagioni per cui costruire il dataset
    # Best Practice: Usare le 2-3 stagioni complete più recenti.
    seasons_to_build = ['2024-25', '2023-24'] 
    
    # 🔧 FALLBACK: Se l'API ha problemi, prova con stagioni precedenti
    print("🔧 Testando connessione API NBA...")
    test_data = data_provider.get_season_game_log('2022-23', 'Regular Season')
    if test_data is None or test_data.empty:
        print("⚠️ API NBA non risponde correttamente, provo con stagioni alternative...")
        seasons_to_build = ['2022-23', '2021-22']  # Fallback a stagioni precedenti
    else:
        print("✅ API NBA funzionante, procedo con stagioni pianificate")
    
    # 🏀 CONFIGURA QUI: include_playoffs=True per Regular Season + Playoffs
    #                  include_playoffs=False per solo Regular Season
    include_playoffs_flag = True  # CAMBIA QUESTO per scegliere la modalità
    
    builder.build_dataset_for_seasons(seasons_to_build, include_playoffs=include_playoffs_flag)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❗️ Processo interrotto dall'utente. Uscita pulita.")
        sys.exit(0) 