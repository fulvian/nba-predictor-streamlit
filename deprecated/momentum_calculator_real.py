# momentum_calculator_real.py
"""
Sistema di calcolo momentum basato su dati NBA reali.
Implementa l'approccio professionale usando game logs, plus/minus e trend recenti.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from nba_api.stats.endpoints import playergamelog, teamgamelog
from nba_api.stats.static import teams

class RealMomentumCalculator:
    """
    Calcola momentum reale basato su game logs NBA e plus/minus.
    """
    
    def __init__(self, window_size=5, api_delay=0.6):
        """
        Inizializza il calcolatore momentum.
        
        Args:
            window_size: Numero di partite per calcolare la media mobile
            api_delay: Delay tra chiamate API per evitare rate limiting
        """
        self.window_size = window_size
        self.api_delay = api_delay
        self.season = '2024-25'
        
        print(f"🔄 RealMomentumCalculator inizializzato")
        print(f"   📊 Window size: {window_size} partite")
        print(f"   ⏱️ API delay: {api_delay}s")

    def fetch_player_game_log(self, player_id, limit_games=15):
        """
        Recupera game log per un giocatore specifico.
        
        Args:
            player_id: ID del giocatore NBA
            limit_games: Numero massimo di partite da recuperare
            
        Returns:
            DataFrame con game log o None se errore
        """
        try:
            print(f"   📊 [NBA_API] Recupero game log per player_id={player_id}")
            time.sleep(self.api_delay)
            
            gl = playergamelog.PlayerGameLog(
                player_id=player_id, 
                season=self.season,
                timeout=30
            )
            df = gl.get_data_frames()[0]
            
            if df.empty:
                print(f"   ⚠️ Nessun game log trovato per player {player_id}")
                return None
            
            # Prendi solo le colonne necessarie se disponibili
            available_cols = df.columns.tolist()
            cols_needed = ['GAME_DATE', 'PTS', 'PLUS_MINUS', 'MIN']
            cols_to_use = [col for col in cols_needed if col in available_cols]
            
            # Aggiungi GAME_ID se disponibile
            if 'GAME_ID' in available_cols:
                cols_to_use.append('GAME_ID')
            
            if not cols_to_use:
                print(f"   ⚠️ Nessuna colonna necessaria trovata per player {player_id}")
                print(f"   📊 Colonne disponibili: {available_cols}")
                return None
                
            df = df[cols_to_use].copy()
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values('GAME_DATE', ascending=False)
            
            # Limita il numero di partite per performance
            if limit_games and len(df) > limit_games:
                df = df.head(limit_games)
            
            print(f"   ✅ Trovate {len(df)} partite per player {player_id}")
            return df
            
        except Exception as e:
            print(f"   ❌ Errore recupero game log per player {player_id}: {e}")
            return None

    def compute_player_momentum(self, game_log_df):
        """
        Calcola momentum per un singolo giocatore.
        
        Args:
            game_log_df: DataFrame con game log del giocatore
            
        Returns:
            dict con metriche momentum
        """
        if game_log_df is None or len(game_log_df) < self.window_size:
            return {
                'momentum_score': 50.0,  # Neutro
                'pts_trend': 0.0,
                'plus_minus_trend': 0.0,
                'recent_performance': 'insufficient_data',
                'games_analyzed': 0
            }
        
        df = game_log_df.copy()
        
        # Calcola medie mobili (momentum)
        df['momentum_pts'] = df['PTS'].rolling(window=self.window_size, min_periods=2).mean()
        df['momentum_pm'] = df['PLUS_MINUS'].rolling(window=self.window_size, min_periods=2).mean()
        
        # Calcola delta (performance attuale vs momentum)
        df['delta_pts'] = df['PTS'] - df['momentum_pts']
        df['delta_pm'] = df['PLUS_MINUS'] - df['momentum_pm']
        
        # Prendi i valori più recenti
        latest = df.iloc[0]  # Più recente (già ordinato)
        
        # Debug: verifica che ci siano dati validi
        if 'PTS' not in df.columns or 'PLUS_MINUS' not in df.columns:
            print(f"   ⚠️ Colonne mancanti: PTS={('PTS' in df.columns)}, PLUS_MINUS={('PLUS_MINUS' in df.columns)}")
            return {
                'momentum_score': 50.0,
                'pts_trend': 0.0,
                'plus_minus_trend': 0.0,
                'recent_performance': 'missing_columns',
                'games_analyzed': 0
            }
        
        # Calcola score momentum (basato su plus/minus trend)
        recent_pm_avg = df['PLUS_MINUS'].head(self.window_size).mean()
        season_pm_avg = df['PLUS_MINUS'].mean()
        
        # Debug dettagliato
        print(f"     🔧 Debug momentum: {len(df)} partite, PM recente: {recent_pm_avg:.2f}, PM stagione: {season_pm_avg:.2f}")
        
        # Score da 0 a 100 (50 = neutro)
        momentum_score = 50 + (recent_pm_avg - season_pm_avg) * 2
        momentum_score = np.clip(momentum_score, 0, 100)
        
        # Trend delle ultime partite
        recent_games = df.head(self.window_size)
        pts_trend = recent_games['PTS'].mean() - df['PTS'].mean()
        pm_trend = recent_games['PLUS_MINUS'].mean() - df['PLUS_MINUS'].mean()
        
        # Classifica performance recente
        if recent_pm_avg > 5:
            performance = 'hot'
        elif recent_pm_avg > 0:
            performance = 'positive'
        elif recent_pm_avg > -5:
            performance = 'neutral'
        else:
            performance = 'cold'
        
        return {
            'momentum_score': float(momentum_score),
            'pts_trend': float(pts_trend),
            'plus_minus_trend': float(pm_trend),
            'recent_performance': performance,
            'games_analyzed': len(df),
            'recent_pm_avg': float(recent_pm_avg),
            'season_pm_avg': float(season_pm_avg),
            'latest_pts': float(latest.get('PTS', 0)),
            'latest_pm': float(latest.get('PLUS_MINUS', 0))
        }

    def calculate_team_momentum_impact(self, roster_df, team_name):
        """
        Calcola momentum impact per una squadra completa.
        
        Args:
            roster_df: DataFrame con roster della squadra
            team_name: Nome della squadra
            
        Returns:
            dict con impact totale e dettagli giocatori
        """
        print(f"   🔄 Calcolo momentum REALE per {team_name}...")
        
        if roster_df.empty:
            print(f"   ⚠️ Roster vuoto per {team_name}")
            return {
                'total_impact': 0.0,
                'avg_momentum_score': 50.0,
                'hot_players': 0,
                'cold_players': 0,
                'players_analyzed': 0,
                'team_performance': 'unknown'
            }
        
        player_momentums = []
        analyzed_count = 0
        
                 # Analizza ogni giocatore nel roster
        for idx, player_row in roster_df.iterrows():
            try:
                # Converti la row in un dict sicuro per evitare problemi con Series
                player = player_row.to_dict() if hasattr(player_row, 'to_dict') else player_row
                
                # Estrai player_id sicuro
                player_id = None
                for id_field in ['PLAYER_ID', 'id', 'player_id']:
                    if id_field in player and player[id_field] is not None:
                        player_id = player[id_field]
                        break
                
                # Converti a tipo python nativo se necessario
                if hasattr(player_id, 'item'):
                    player_id = player_id.item()
                elif isinstance(player_id, (list, tuple)) and len(player_id) > 0:
                    player_id = player_id[0]
                
                # Estrai player_name sicuro
                player_name = 'Unknown'
                for name_field in ['PLAYER', 'full_name', 'player_name', 'name']:
                    if name_field in player and player[name_field] is not None:
                        player_name = str(player[name_field])
                        break
                
                # Converti a tipo python nativo se necessario
                if hasattr(player_name, 'item'):
                    player_name = str(player_name.item())
                elif isinstance(player_name, (list, tuple)) and len(player_name) > 0:
                    player_name = str(player_name[0])
                
                print(f"   🔧 Debug: Processing player_id={player_id}, name={player_name}")
                    
            except Exception as e:
                print(f"   ⚠️ Errore nell'estrazione dati giocatore {idx}: {e}")
                continue
                
            if not player_id:
                continue
            
            print(f"   📊 Analizzando momentum per {player_name}...")
            
            # Recupera game log
            game_log = self.fetch_player_game_log(player_id, limit_games=12)
            
            # Calcola momentum
            momentum = self.compute_player_momentum(game_log)
            momentum['player_name'] = player_name
            momentum['player_id'] = player_id
            
            player_momentums.append(momentum)
            analyzed_count += 1
            
            # Limite per evitare troppi API calls
            if analyzed_count >= 8:  # Top 8 giocatori
                break
        
        if not player_momentums:
            print(f"   ❌ Nessun momentum calcolato per {team_name}")
            return {
                'total_impact': 0.0,
                'avg_momentum_score': 50.0,
                'hot_players': 0,
                'cold_players': 0,
                'players_analyzed': 0,
                'team_performance': 'unknown'
            }
        
        # Calcola metriche aggregate
        momentum_scores = [p['momentum_score'] for p in player_momentums]
        pm_trends = [p['plus_minus_trend'] for p in player_momentums]
        
        avg_momentum = np.mean(momentum_scores)
        avg_pm_trend = np.mean(pm_trends)
        
        # Conta giocatori hot/cold
        hot_players = sum(1 for p in player_momentums if p['recent_performance'] == 'hot')
        cold_players = sum(1 for p in player_momentums if p['recent_performance'] == 'cold')
        
        # Calcola impact sui punti totali
        # Formula: (avg_pm_trend * players_weight) con scala appropriata
        players_weight = min(analyzed_count / 8.0, 1.0)  # Max weight per 8 giocatori
        total_impact = avg_pm_trend * players_weight * 0.5  # Scala conservativa
        
        # Classifica team performance
        if avg_momentum > 60:
            team_performance = 'hot'
        elif avg_momentum > 45:
            team_performance = 'positive'
        elif avg_momentum > 40:
            team_performance = 'neutral'
        else:
            team_performance = 'cold'
        
        print(f"   ✅ {team_name} momentum: {avg_momentum:.1f}/100 | Impact: {total_impact:+.2f} pts")
        print(f"   🔥 Hot players: {hot_players} | 🧊 Cold players: {cold_players}")
        
        return {
            'total_impact': float(total_impact),
            'avg_momentum_score': float(avg_momentum),
            'avg_pm_trend': float(avg_pm_trend),
            'hot_players': hot_players,
            'cold_players': cold_players,
            'players_analyzed': analyzed_count,
            'team_performance': team_performance,
            'player_details': player_momentums[:5]  # Top 5 per summary
        }

    def calculate_game_momentum_differential(self, home_roster_df, away_roster_df, 
                                           home_team_name, away_team_name):
        """
        Calcola il differenziale di momentum per una partita.
        
        Args:
            home_roster_df: Roster squadra di casa
            away_roster_df: Roster squadra in trasferta
            home_team_name: Nome squadra di casa
            away_team_name: Nome squadra in trasferta
            
        Returns:
            dict con impact differenziale e dettagli
        """
        print(f"🔄 Calcolo momentum differenziale: {away_team_name} @ {home_team_name}")
        
        # Calcola momentum per entrambe le squadre
        home_momentum = self.calculate_team_momentum_impact(home_roster_df, home_team_name)
        away_momentum = self.calculate_team_momentum_impact(away_roster_df, away_team_name)
        
        # Calcola differenziale
        impact_differential = home_momentum['total_impact'] - away_momentum['total_impact']
        momentum_differential = home_momentum['avg_momentum_score'] - away_momentum['avg_momentum_score']
        
        # Confidence basato sul numero di giocatori analizzati
        total_players_analyzed = home_momentum['players_analyzed'] + away_momentum['players_analyzed']
        confidence = min(total_players_analyzed / 12.0, 1.0)  # Max confidence con 12+ giocatori
        
        print(f"📊 MOMENTUM SUMMARY:")
        print(f"   🏠 {home_team_name}: {home_momentum['avg_momentum_score']:.1f}/100 ({home_momentum['team_performance']})")
        print(f"   🛫 {away_team_name}: {away_momentum['avg_momentum_score']:.1f}/100 ({away_momentum['team_performance']})")
        print(f"   ⚖️ Differenziale: {impact_differential:+.2f} pts | Confidence: {confidence:.1%}")
        
        return {
            'total_impact': impact_differential,
            'home_momentum': home_momentum,
            'away_momentum': away_momentum,
            'momentum_differential': momentum_differential,
            'confidence_factor': confidence,
            'real_data_system': True,
            'summary': f"Home: {home_momentum['team_performance']}, Away: {away_momentum['team_performance']}"
        } 