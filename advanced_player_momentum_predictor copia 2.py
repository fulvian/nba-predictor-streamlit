# advanced_player_momentum_predictor.py (Versione Revisionata e Corretta)
import pandas as pd
import numpy as np
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Imports per XGBoost (opzionale, gestiremo l'assenza)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class AdvancedPlayerMomentumPredictor:
    """
    Sistema avanzato per il calcolo del momentum basato su ricerca scientifica.
    Implementa metodologie validate per catturare hot hand, mean reversion,
    e impatto differenziale offensivo/difensivo.
    """
    
    def __init__(self, models_dir='models', nba_data_provider=None):
        print("\n🚀 [ADVANCED_MOMENTUM] Inizializzazione sistema avanzato...")
        
        self.models_dir = os.path.join(models_dir, 'advanced_momentum')
        self.nba_data_provider = nba_data_provider
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        
        self.temporal_windows = {
            'short': 3, 'medium': 5, 'long': 10
        }
        self.momentum_weights = {
            'offensive_momentum': 0.35, 'defensive_momentum': 0.45, 'usage_momentum': 0.20
        }
        self.position_impact_weights = {
            'PG': {'offensive': 0.75, 'defensive': 0.25, 'usage_sensitivity': 1.2},
            'SG': {'offensive': 0.85, 'defensive': 0.15, 'usage_sensitivity': 1.1}, 
            'SF': {'offensive': 0.70, 'defensive': 0.30, 'usage_sensitivity': 1.0},
            'PF': {'offensive': 0.60, 'defensive': 0.40, 'usage_sensitivity': 0.9},
            'C': {'offensive': 0.45, 'defensive': 0.55, 'usage_sensitivity': 0.8},
            'UNKNOWN': {'offensive': 0.65, 'defensive': 0.35, 'usage_sensitivity': 1.0}
        }
        self.hot_hand_thresholds = {
            'ts_pct_improvement': 0.05, 'usage_increase': 0.03, 
            'consistency_floor': 0.6, 'sample_size_min': 3
        }
        self.mean_reversion_factors = {
            'extreme_performance_threshold': 1.5, 'reversion_strength': 0.3, 'hot_streak_decay': 0.85
        }

    def _safe_get(self, data, key, default=None):
        """Helper centralizzato per accedere a dati in dizionari o oggetti pandas."""
        if hasattr(data, 'get'):  # Funziona per dizionari e pandas Series/DataFrame
            return data.get(key, default)
        return default

    def _calculate_advanced_offensive_metrics(self, game_logs, player_season_avg=None):
        if game_logs is None or len(game_logs) < 2:
            return self._get_neutral_offensive_metrics()
            
        games = game_logs.copy()
        ts_pct_games, usage_games, efg_games = [], [], []
        
        for _, game in games.iterrows():
            pts, fga, fta = float(game.get('PTS', 0)), float(game.get('FGA', 0)), float(game.get('FTA', 0))
            ts_pct = pts / (2 * (fga + 0.44 * fta)) if fga + 0.44 * fta > 0 else 0.0
            ts_pct_games.append(ts_pct)
            
            fgm, fg3m = float(game.get('FGM', 0)), float(game.get('FG3M', 0))
            efg_pct = (fgm + 0.5 * fg3m) / fga if fga > 0 else 0.0
            efg_games.append(efg_pct)
            
            usage = float(game.get('USG_PCT', game.get('USAGE_PCT', 0)))
            if usage == 0 and fga > 0:
                min_played = float(game.get('MIN', 1))
                usage = (fga + 0.44 * fta) * 100 / (min_played * 5) if min_played > 0 else 0
            usage_games.append(usage)
        
        metrics = {}
        for window_name, window_size in self.temporal_windows.items():
            ts_w = ts_pct_games[-window_size:]
            metrics[f'{window_name}_window'] = {
                'ts_pct_mean': np.mean(ts_w) if ts_w else 0,
                'ts_trend': self._calculate_linear_trend(ts_w),
                'consistency': self._calculate_consistency(ts_w),
                'sample_size': len(ts_w)
            }
        
        reversion_signal = 0
        if player_season_avg and 'TS_PCT' in player_season_avg:
            season_ts = player_season_avg['TS_PCT']
            recent_ts = metrics['short_window']['ts_pct_mean']
            if season_ts > 0:
                deviation = (recent_ts - season_ts) / season_ts
                if abs(deviation) > self.mean_reversion_factors['extreme_performance_threshold']:
                    reversion_signal = -deviation * self.mean_reversion_factors['reversion_strength']
        
        offensive_score = self._aggregate_offensive_score(metrics, reversion_signal)
        
        return {
            'offensive_momentum_score': offensive_score,
            'detailed_metrics': metrics,
            'mean_reversion_signal': reversion_signal,
            'hot_hand_detected': self._detect_hot_hand(metrics['medium_window'])
        }

    def _calculate_advanced_defensive_metrics(self, game_logs, player_name, player_season_avg=None):
        if game_logs is None or len(game_logs) < 2:
            return self._get_neutral_defensive_metrics()
        
        games = game_logs.copy()
        def_metrics_games, plus_minus_games = [], []
        
        for _, game in games.iterrows():
            plus_minus_games.append(float(game.get('PLUS_MINUS', 0)))
            stl, blk, dreb, pf = float(game.get('STL', 0)), float(game.get('BLK', 0)), float(game.get('DREB', 0)), float(game.get('PF', 0))
            def_metrics_games.append(stl * 2 + blk * 2.5 + dreb * 0.5 - pf * 0.5)
        
        metrics = {}
        for window_name, window_size in self.temporal_windows.items():
            pm_w = plus_minus_games[-window_size:]
            metrics[f'{window_name}_window'] = {
                'plus_minus_mean': np.mean(pm_w) if pm_w else 0,
                'plus_minus_trend': self._calculate_linear_trend(pm_w),
                'consistency': self._calculate_consistency(pm_w),
                'sample_size': len(pm_w)
            }
        
        defensive_score = self._aggregate_defensive_score(metrics)
        
        return {
            'defensive_momentum_score': defensive_score,
            'detailed_metrics': metrics,
            'consistency_advantage': metrics['medium_window']['consistency']
        }

    def _calculate_usage_momentum(self, game_logs, player_season_avg=None):
        if game_logs is None or len(game_logs) < 3:
            return {'usage_momentum_score': 50.0, 'usage_trend': 0.0}
        
        usage_games = [float(g.get('USG_PCT', g.get('USAGE_PCT', 0))) for _, g in game_logs.iterrows()]
        usage_trend = self._calculate_linear_trend(usage_games[-5:])
        
        season_usage = self._safe_get(player_season_avg, 'USG_PCT', np.mean(usage_games))
        recent_usage = np.mean(usage_games[-3:])
        usage_change = (recent_usage - season_usage) / max(season_usage, 1.0) if season_usage > 0 else 0
        
        usage_momentum = 50 + (usage_trend * 20) + (usage_change * 15)
        
        return {
            'usage_momentum_score': max(0, min(100, usage_momentum)),
            'usage_trend': usage_trend,
            'usage_change_pct': usage_change
        }

    def _detect_hot_hand(self, window_metrics):
        return (window_metrics.get('ts_pct_mean', 0) > 0.55 and
                window_metrics.get('ts_trend', 0) > 0.01 and
                window_metrics.get('consistency', 0) > self.hot_hand_thresholds['consistency_floor'] and
                window_metrics.get('sample_size', 0) >= self.hot_hand_thresholds['sample_size_min'])

    def _aggregate_offensive_score(self, metrics, reversion_signal):
        medium_window = metrics['medium_window']
        base_score = (medium_window['ts_pct_mean'] * 100) + (medium_window['ts_trend'] * 50)
        consistency_bonus = medium_window['consistency'] * 10
        hot_hand_bonus = 5 if self._detect_hot_hand(medium_window) else 0
        reversion_adjustment = reversion_signal * 10
        final_score = base_score + consistency_bonus + hot_hand_bonus + reversion_adjustment
        return max(0, min(100, final_score))

    def _aggregate_defensive_score(self, metrics):
        medium_window = metrics['medium_window']
        base_score = 50 + (medium_window['plus_minus_mean'] * 2) + (medium_window['plus_minus_trend'] * 15)
        consistency_bonus = medium_window['consistency'] * 15
        final_score = base_score + consistency_bonus
        return max(0, min(100, final_score))

    def _get_player_momentum_score_advanced(self, player_id, player_name, player_season_avg=None, last_n_games=10):
        default_result = {
            'total_score': 50.0, 'component_scores': {'offensive': 50.0, 'defensive': 50.0, 'usage': 50.0},
            'detailed_analysis': {'hot_hand_detected': False, 'mean_reversion_signal': 0, 'defensive_consistency': 0}
        }
        if not self.nba_data_provider: return default_result
        
        try:
            game_logs = self.nba_data_provider.get_player_game_logs(player_id, last_n_games=last_n_games)
            if game_logs is None or len(game_logs) < 2: return default_result
            
            offensive_metrics = self._calculate_advanced_offensive_metrics(game_logs, player_season_avg)
            defensive_metrics = self._calculate_advanced_defensive_metrics(game_logs, player_name, player_season_avg)
            usage_metrics = self._calculate_usage_momentum(game_logs, player_season_avg)
            
            off_score = float(self._safe_get(offensive_metrics, 'offensive_momentum_score', 50.0))
            def_score = float(self._safe_get(defensive_metrics, 'defensive_momentum_score', 50.0))
            use_score = float(self._safe_get(usage_metrics, 'usage_momentum_score', 50.0))
            
            total_score = (off_score * self.momentum_weights['offensive_momentum'] +
                           def_score * self.momentum_weights['defensive_momentum'] +
                           use_score * self.momentum_weights['usage_momentum'])
            
            return {
                'total_score': max(0.0, min(100.0, total_score)),
                'component_scores': {'offensive': off_score, 'defensive': def_score, 'usage': use_score},
                'detailed_analysis': {
                    'hot_hand_detected': bool(self._safe_get(offensive_metrics, 'hot_hand_detected', False)),
                    'mean_reversion_signal': float(self._safe_get(offensive_metrics, 'mean_reversion_signal', 0.0)),
                    'defensive_consistency': float(self._safe_get(defensive_metrics, 'consistency_advantage', 0.0))
                }
            }
        except Exception as e:
            print(f"❌ [ADVANCED_MOMENTUM] Errore per {player_name}: {e}")
            return default_result

    def predict_team_momentum_impact_advanced(self, team_roster_df):
        if not isinstance(team_roster_df, pd.DataFrame) or team_roster_df.empty:
            return {'momentum_score': 50.0, 'impact_on_totals': 0.0, 'player_contributions': [], 'hot_hand_players_count': 0, 'total_weighted_contribution': 0.0}
        
        team_contributions = []
        for _, player_series in team_roster_df.iterrows():
            player_id = self._safe_get(player_series, 'PLAYER_ID')
            player_name = self._safe_get(player_series, 'PLAYER_NAME', f'ID: {player_id}')
            
            try:
                player_season_avg = self._get_player_season_averages(player_series)
                momentum_result = self._get_player_momentum_score_advanced(player_id, player_name, player_season_avg)
                
                component_scores = self._safe_get(momentum_result, 'component_scores', {})
                pos_weights = self.position_impact_weights.get(self._safe_get(player_series, 'POSITION', 'UNKNOWN'))
                rotation_weight = self._get_rotation_weight(self._safe_get(player_series, 'ROTATION_STATUS', 'BENCH'))
                
                offensive_score = self._safe_get(component_scores, 'offensive', 50)
                defensive_score = self._safe_get(component_scores, 'defensive', 50)
                
                weighted_contribution = (offensive_score * pos_weights['offensive'] * rotation_weight +
                                         defensive_score * pos_weights['defensive'] * rotation_weight) * pos_weights['usage_sensitivity']
                
                detailed_analysis = self._safe_get(momentum_result, 'detailed_analysis', {})
                team_contributions.append({
                    'player_name': player_name,
                    'momentum_score': self._safe_get(momentum_result, 'total_score', 50.0),
                    'weighted_contribution': weighted_contribution,
                    'hot_hand_detected': self._safe_get(detailed_analysis, 'hot_hand_detected', False)
                })
            except Exception as e:
                print(f"⚠️ Errore nell'elaborazione del giocatore {player_name}: {e}")
                continue
        
        if not team_contributions:
            return {'momentum_score': 50.0, 'impact_on_totals': 0.0, 'player_contributions': [], 'hot_hand_players_count': 0, 'total_weighted_contribution': 0.0}
        
        total_weighted_contribution = sum(p['weighted_contribution'] for p in team_contributions)
        # --- CORREZIONE BUG: USARE iterrows() INVECE DI itertuples() ---
        total_weights = sum(self._get_rotation_weight(p.get('ROTATION_STATUS', 'BENCH')) for _, p in team_roster_df.iterrows())
        
        team_momentum_score = (total_weighted_contribution / total_weights) if total_weights > 0 else 50.0
        impact_on_totals = (team_momentum_score - 50) * 0.16
        
        hot_hand_players = sum(1 for p in team_contributions if p['hot_hand_detected'])
        if hot_hand_players >= 2:
            impact_on_totals += 1.5
        
        return {
            'momentum_score': float(team_momentum_score),
            'impact_on_totals': float(impact_on_totals),
            'player_contributions': team_contributions,
            'hot_hand_players_count': int(hot_hand_players),
            'total_weighted_contribution': float(total_weighted_contribution)
        }

    def _calculate_linear_trend(self, values):
        if len(values) < 2: return 0.0
        x = np.arange(len(values))
        try:
            slope, _, _, _, _ = stats.linregress(x, values)
            return slope if not np.isnan(slope) else 0.0
        except: return 0.0

    def _calculate_consistency(self, values):
        if not isinstance(values, list) or len(values) < 2: return 0.5
        mean_val = np.mean(values)
        if mean_val == 0: return 0.0
        cv = np.std(values) / abs(mean_val)
        return max(0, min(1, 1 - cv))

    def _get_rotation_weight(self, status):
        return {'STARTER': 1.0, 'BENCH': 0.7, 'RESERVE': 0.4, 'INACTIVE': 0.1}.get(status, 0.5)

    def _get_player_season_averages(self, player_row):
        player_dict = player_row.to_dict() if hasattr(player_row, 'to_dict') else player_row
        if isinstance(player_dict, dict):
            return {
                'TS_PCT': player_dict.get('TS_PCT', 0.55),
                'USG_PCT': player_dict.get('USG_PCT', 20.0),
                'PTS': player_dict.get('PTS', 10.0)
            }
        return {'TS_PCT': 0.55, 'USG_PCT': 20.0, 'PTS': 10.0}

    def _get_neutral_offensive_metrics(self):
        return {'offensive_momentum_score': 50.0, 'detailed_metrics': {}, 'mean_reversion_signal': 0.0, 'hot_hand_detected': False}

    def _get_neutral_defensive_metrics(self):
        return {'defensive_momentum_score': 50.0, 'detailed_metrics': {}, 'consistency_advantage': 0.5}