# momentum_monitoring_system.py
"""
Sistema completo di monitoring per il momentum system avanzato.
Include logging, metriche real-time, e dashboard di controllo.
"""

import pandas as pd
import numpy as np
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict, deque
import os
import threading
import time

class MomentumPerformanceMonitor:
    """
    Monitor delle performance del sistema momentum in real-time.
    """
    
    def __init__(self, db_path='momentum_monitoring.db', window_size=100):
        self.db_path = db_path
        self.window_size = window_size  # Finestra mobile per metriche real-time
        
        # Buffer per metriche real-time
        self.recent_predictions = deque(maxlen=window_size)
        self.recent_outcomes = deque(maxlen=window_size)
        self.recent_edges = deque(maxlen=window_size)
        self.recent_confidence = deque(maxlen=window_size)
        
        # Contatori performance
        self.daily_stats = defaultdict(lambda: {
            'predictions': 0,
            'correct_predictions': 0,
            'total_edge': 0.0,
            'avg_confidence': 0.0,
            'value_bets_found': 0,
            'momentum_triggers': {
                'hot_hand_detected': 0,
                'mean_reversion_applied': 0,
                'synergy_bonus_triggered': 0
            }
        })
        
        # Setup database e logging
        self._setup_database()
        self._setup_logging()
        
        print("🔍 [MONITOR] Sistema di monitoring inizializzato")
    
    def _setup_database(self):
        """Inizializza database SQLite per storage persistente."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabella predizioni
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                game_date DATE,
                home_team TEXT,
                away_team TEXT,
                predicted_total REAL,
                actual_total REAL,
                prediction_error REAL,
                momentum_impact REAL,
                confidence_factor REAL,
                hot_hands_detected INTEGER,
                synergy_detected BOOLEAN,
                value_bets_count INTEGER,
                max_edge REAL,
                outcome_known BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Tabella scommesse raccomandate
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommended_bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                bet_type TEXT,
                line REAL,
                odds REAL,
                probability REAL,
                edge REAL,
                stake REAL,
                is_value BOOLEAN,
                outcome TEXT,
                profit_loss REAL,
                FOREIGN KEY (prediction_id) REFERENCES predictions (id)
            )
        ''')
        
        # Tabella metriche giornaliere
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_metrics (
                date DATE PRIMARY KEY,
                total_predictions INTEGER,
                correct_predictions INTEGER,
                win_rate REAL,
                avg_edge REAL,
                avg_confidence REAL,
                total_value_bets INTEGER,
                estimated_roi REAL,
                momentum_analysis TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _setup_logging(self):
        """Configura logging avanzato."""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # Logger principale
        self.logger = logging.getLogger('momentum_monitor')
        self.logger.setLevel(logging.INFO)
        
        # Handler per file giornaliero
        today = datetime.now().strftime('%Y-%m-%d')
        file_handler = logging.FileHandler(f'{log_dir}/momentum_{today}.log')
        file_handler.setLevel(logging.INFO)
        
        # Formatter dettagliato
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def log_prediction(self, game_details, prediction_result, momentum_details):
        """
        Registra una predizione completa con tutti i dettagli.
        """
        timestamp = datetime.now()
        
        # Estrai dati chiave
        predicted_total = prediction_result.get('predicted_mu', 0)
        momentum_impact = momentum_details.get('total_impact', 0)
        confidence = momentum_details.get('confidence_factor', 1.0)
        hot_hands = momentum_details.get('home_momentum', {}).get('hot_hands', 0) + \
                   momentum_details.get('away_momentum', {}).get('hot_hands', 0)
        synergy = momentum_details.get('synergy_detected', False)
        
        # Conta value bets
        opportunities = game_details.get('betting_opportunities', [])
        value_bets = [opp for opp in opportunities if opp.get('is_value', False)]
        max_edge = max([opp.get('edge', 0) for opp in opportunities], default=0)
        
        # Salva nel database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (
                timestamp, game_date, home_team, away_team, predicted_total,
                momentum_impact, confidence_factor, hot_hands_detected,
                synergy_detected, value_bets_count, max_edge
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp, game_details.get('date'), game_details.get('home_team'),
            game_details.get('away_team'), predicted_total, momentum_impact,
            confidence, hot_hands, synergy, len(value_bets), max_edge
        ))
        
        prediction_id = cursor.lastrowid
        
        # Salva scommesse raccomandate
        for bet in value_bets:
            cursor.execute('''
                INSERT INTO recommended_bets (
                    prediction_id, bet_type, line, odds, probability, edge, stake, is_value
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction_id, bet['type'], bet['line'], bet['odds'],
                bet['probability'], bet['edge'], bet['stake'], bet['is_value']
            ))
        
        conn.commit()
        conn.close()
        
        # Log textual
        self.logger.info(f"PREDICTION | {game_details.get('away_team')} @ {game_details.get('home_team')} | "
                        f"Total: {predicted_total:.1f} | Momentum: {momentum_impact:+.2f} | "
                        f"Confidence: {confidence:.2f} | Value bets: {len(value_bets)}")
        
        # Aggiorna metriche real-time
        self.recent_predictions.append(predicted_total)
        self.recent_confidence.append(confidence)
        if opportunities:
            self.recent_edges.append(max_edge)
        
        # Aggiorna stats giornalieri
        today = datetime.now().date()
        self.daily_stats[today]['predictions'] += 1
        self.daily_stats[today]['total_edge'] += max_edge
        self.daily_stats[today]['avg_confidence'] += confidence
        self.daily_stats[today]['value_bets_found'] += len(value_bets)
        
        # Contatori momentum
        if hot_hands > 0:
            self.daily_stats[today]['momentum_triggers']['hot_hand_detected'] += 1
        if synergy:
            self.daily_stats[today]['momentum_triggers']['synergy_bonus_triggered'] += 1
        
        return prediction_id
    
    def update_outcome(self, prediction_id, actual_total):
        """
        Aggiorna con il risultato reale di una partita.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Recupera predizione
        cursor.execute('SELECT predicted_total FROM predictions WHERE id = ?', (prediction_id,))
        result = cursor.fetchone()
        
        if result:
            predicted_total = result[0]
            prediction_error = abs(predicted_total - actual_total)
            
            # Aggiorna predizione
            cursor.execute('''
                UPDATE predictions 
                SET actual_total = ?, prediction_error = ?, outcome_known = TRUE 
                WHERE id = ?
            ''', (actual_total, prediction_error, prediction_id))
            
            # Aggiorna scommesse
            cursor.execute('SELECT * FROM recommended_bets WHERE prediction_id = ?', (prediction_id,))
            bets = cursor.fetchall()
            
            for bet in bets:
                bet_id, _, bet_type, line, odds, prob, edge, stake, is_value = bet[:9]
                
                # Determina outcome
                if bet_type == 'OVER':
                    won = actual_total > line
                else:  # UNDER
                    won = actual_total < line
                
                profit_loss = stake * (odds - 1) if won else -stake
                outcome = 'WIN' if won else 'LOSS'
                
                cursor.execute('''
                    UPDATE recommended_bets 
                    SET outcome = ?, profit_loss = ? 
                    WHERE id = ?
                ''', (outcome, profit_loss, bet_id))
            
            conn.commit()
            
            # Log outcome
            self.logger.info(f"OUTCOME | Prediction ID: {prediction_id} | "
                           f"Predicted: {predicted_total:.1f} | Actual: {actual_total} | "
                           f"Error: {prediction_error:.1f}")
            
            # Aggiorna metriche real-time
            self.recent_outcomes.append(prediction_error)
            
            # Aggiorna stats giornalieri se predizione era accurata
            today = datetime.now().date()
            if prediction_error < 10.0:  # Soglia accuratezza
                self.daily_stats[today]['correct_predictions'] += 1
        
        conn.close()
        return prediction_error if result else None
    
    def get_realtime_metrics(self):
        """
        Restituisce metriche real-time per dashboard.
        """
        if not self.recent_predictions:
            return {'status': 'no_data'}
        
        # Metriche rolling window
        avg_confidence = np.mean(self.recent_confidence) if self.recent_confidence else 0
        avg_edge = np.mean(self.recent_edges) if self.recent_edges else 0
        avg_prediction_error = np.mean(self.recent_outcomes) if self.recent_outcomes else 0
        
        # Stats giornalieri
        today = datetime.now().date()
        today_stats = self.daily_stats[today]
        
        accuracy_rate = (today_stats['correct_predictions'] / max(today_stats['predictions'], 1)) * 100
        
        return {
            'timestamp': datetime.now().isoformat(),
            'rolling_window_metrics': {
                'avg_confidence': round(avg_confidence, 3),
                'avg_edge': round(avg_edge, 4),
                'avg_prediction_error': round(avg_prediction_error, 2),
                'sample_size': len(self.recent_predictions)
            },
            'daily_metrics': {
                'predictions_count': today_stats['predictions'],
                'accuracy_rate': round(accuracy_rate, 1),
                'value_bets_found': today_stats['value_bets_found'],
                'momentum_triggers': today_stats['momentum_triggers']
            },
            'health_indicators': {
                'system_status': 'healthy' if avg_confidence > 0.5 else 'warning',
                'prediction_quality': 'good' if avg_prediction_error < 12 else 'poor',
                'edge_detection': 'active' if avg_edge > 0.02 else 'low'
            }
        }
    
    def generate_daily_report(self, target_date=None):
        """
        Genera report completo per una giornata.
        """
        if target_date is None:
            target_date = datetime.now().date()
        
        conn = sqlite3.connect(self.db_path)
        
        # Query predizioni del giorno
        predictions_df = pd.read_sql_query('''
            SELECT * FROM predictions 
            WHERE date(game_date) = ?
            ORDER BY timestamp
        ''', conn, params=[target_date])
        
        # Query scommesse del giorno
        bets_df = pd.read_sql_query('''
            SELECT rb.* FROM recommended_bets rb
            JOIN predictions p ON rb.prediction_id = p.id
            WHERE date(p.game_date) = ?
        ''', conn, params=[target_date])
        
        conn.close()
        
        if predictions_df.empty:
            return {'date': str(target_date), 'status': 'no_games'}
        
        # Calcola metriche giornaliere
        total_predictions = len(predictions_df)
        predictions_with_outcome = predictions_df[predictions_df['outcome_known'] == True]
        
        if not predictions_with_outcome.empty:
            avg_error = predictions_with_outcome['prediction_error'].mean()
            accuracy_rate = (predictions_with_outcome['prediction_error'] < 10).mean() * 100
        else:
            avg_error = None
            accuracy_rate = None
        
        # Analisi momentum
        momentum_analysis = {
            'avg_momentum_impact': predictions_df['momentum_impact'].mean(),
            'avg_confidence': predictions_df['confidence_factor'].mean(),
            'hot_hands_games': (predictions_df['hot_hands_detected'] > 0).sum(),
            'synergy_games': predictions_df['synergy_detected'].sum()
        }
        
        # Analisi betting
        if not bets_df.empty:
            total_value_bets = len(bets_df)
            bets_with_outcome = bets_df[bets_df['outcome'].notna()]
            
            if not bets_with_outcome.empty:
                winning_bets = (bets_with_outcome['outcome'] == 'WIN').sum()
                win_rate = winning_bets / len(bets_with_outcome) * 100
                total_profit = bets_with_outcome['profit_loss'].sum()
                roi = total_profit / bets_with_outcome['stake'].sum() * 100 if bets_with_outcome['stake'].sum() > 0 else 0
            else:
                win_rate = None
                total_profit = None
                roi = None
        else:
            total_value_bets = 0
            win_rate = None
            total_profit = None
            roi = None
        
        report = {
            'date': str(target_date),
            'games_analyzed': total_predictions,
            'prediction_metrics': {
                'avg_prediction_error': round(avg_error, 2) if avg_error else None,
                'accuracy_rate_pct': round(accuracy_rate, 1) if accuracy_rate else None,
                'predictions_with_outcome': len(predictions_with_outcome)
            },
            'momentum_metrics': {
                k: round(v, 3) if isinstance(v, float) else v 
                for k, v in momentum_analysis.items()
            },
            'betting_metrics': {
                'total_value_bets': total_value_bets,
                'win_rate_pct': round(win_rate, 1) if win_rate else None,
                'total_profit': round(total_profit, 2) if total_profit else None,
                'roi_pct': round(roi, 1) if roi else None
            }
        }
        
        return report
    
    def get_performance_trend(self, days=7):
        """
        Analizza trend di performance negli ultimi N giorni.
        """
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days-1)
        
        daily_reports = []
        for i in range(days):
            current_date = start_date + timedelta(days=i)
            report = self.generate_daily_report(current_date)
            daily_reports.append(report)
        
        # Calcola trend
        accuracy_trend = [r['prediction_metrics']['accuracy_rate_pct'] 
                         for r in daily_reports 
                         if r['prediction_metrics']['accuracy_rate_pct'] is not None]
        
        roi_trend = [r['betting_metrics']['roi_pct'] 
                    for r in daily_reports 
                    if r['betting_metrics']['roi_pct'] is not None]
        
        momentum_impact_trend = [r['momentum_metrics']['avg_momentum_impact'] 
                               for r in daily_reports]
        
        return {
            'period': f"{start_date} to {end_date}",
            'daily_reports': daily_reports,
            'trends': {
                'accuracy_trend': accuracy_trend,
                'roi_trend': roi_trend,
                'momentum_impact_trend': momentum_impact_trend,
                'trend_direction': {
                    'accuracy': 'improving' if len(accuracy_trend) > 1 and accuracy_trend[-1] > accuracy_trend[0] else 'declining',
                    'roi': 'improving' if len(roi_trend) > 1 and roi_trend[-1] > roi_trend[0] else 'declining'
                }
            }
        }

# Sistema di alert intelligenti
class MomentumAlertSystem:
    """
    Sistema di alert per anomalie e opportunità nel momentum system.
    """
    
    def __init__(self, monitor, alert_config=None):
        self.monitor = monitor
        self.alert_config = alert_config or self._default_alert_config()
        self.last_alerts = {}
        
    def _default_alert_config(self):
        return {
            'low_confidence_threshold': 0.4,
            'high_edge_threshold': 0.15,
            'accuracy_drop_threshold': 0.6,
            'momentum_extreme_threshold': 8.0,
            'alert_cooldown_minutes': 30
        }
    
    def check_alerts(self):
        """
        Controlla condizioni di alert e genera notifiche.
        """
        metrics = self.monitor.get_realtime_metrics()
        
        if metrics.get('status') == 'no_data':
            return []
        
        alerts = []
        current_time = datetime.now()
        
        # Alert 1: Confidenza sistema bassa
        avg_confidence = metrics['rolling_window_metrics']['avg_confidence']
        if avg_confidence < self.alert_config['low_confidence_threshold']:
            alert_key = 'low_confidence'
            if self._should_send_alert(alert_key, current_time):
                alerts.append({
                    'type': 'WARNING',
                    'category': 'system_confidence',
                    'message': f"Confidenza sistema bassa: {avg_confidence:.2f}",
                    'recommendation': "Verifica qualità dati input e sample size",
                    'timestamp': current_time.isoformat()
                })
                self.last_alerts[alert_key] = current_time
        
        # Alert 2: Edge eccezionalmente alto (opportunità)
        avg_edge = metrics['rolling_window_metrics']['avg_edge']
        if avg_edge > self.alert_config['high_edge_threshold']:
            alert_key = 'high_edge'
            if self._should_send_alert(alert_key, current_time):
                alerts.append({
                    'type': 'OPPORTUNITY',
                    'category': 'high_value',
                    'message': f"Edge elevato rilevato: {avg_edge:.3f}",
                    'recommendation': "Considera aumento stake per opportunità eccezionale",
                    'timestamp': current_time.isoformat()
                })
                self.last_alerts[alert_key] = current_time
        
        # Alert 3: Drop accuratezza
        accuracy_rate = metrics['daily_metrics']['accuracy_rate']
        if accuracy_rate < self.alert_config['accuracy_drop_threshold'] * 100:
            alert_key = 'accuracy_drop'
            if self._should_send_alert(alert_key, current_time):
                alerts.append({
                    'type': 'CRITICAL',
                    'category': 'performance_drop',
                    'message': f"Calo accuratezza: {accuracy_rate:.1f}%",
                    'recommendation': "Rivedere configurazione momentum e sample size",
                    'timestamp': current_time.isoformat()
                })
                self.last_alerts[alert_key] = current_time
        
        # Alert 4: Momentum estremo (potenziale outlier)
        # Questo richiederebbe accesso ai dati delle ultime predizioni
        
        return alerts
    
    def _should_send_alert(self, alert_key, current_time):
        """Controlla cooldown per evitare spam di alert."""
        if alert_key not in self.last_alerts:
            return True
        
        time_diff = current_time - self.last_alerts[alert_key]
        cooldown = timedelta(minutes=self.alert_config['alert_cooldown_minutes'])
        
        return time_diff > cooldown

# Dashboard web semplice con Flask (opzionale)
def create_monitoring_dashboard(monitor, alert_system, host='localhost', port=5000):
    """
    Crea dashboard web per monitoring real-time.
    Richiede: pip install flask
    """
    try:
        from flask import Flask, jsonify, render_template_string
        
        app = Flask(__name__)
        
        # Template HTML semplice
        dashboard_template = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>NBA Momentum Monitor</title>
            <meta http-equiv="refresh" content="30">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric-box { border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 5px; }
                .healthy { background-color: #d4edda; }
                .warning { background-color: #fff3cd; }
                .critical { background-color: #f8d7da; }
            </style>
        </head>
        <body>
            <h1>🏀 NBA Momentum System Monitor</h1>
            <div id="metrics"></div>
            
            <script>
                function loadMetrics() {
                    fetch('/api/metrics')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('metrics').innerHTML = formatMetrics(data);
                        });
                }
                
                function formatMetrics(data) {
                    if (data.status === 'no_data') {
                        return '<p>Nessun dato disponibile</p>';
                    }
                    
                    let html = '<h2>Metriche Real-time</h2>';
                    html += '<div class="metric-box healthy">';
                    html += '<h3>Sistema</h3>';
                    html += '<p>Confidenza Media: ' + (data.rolling_window_metrics.avg_confidence * 100).toFixed(1) + '%</p>';
                    html += '<p>Edge Medio: ' + (data.rolling_window_metrics.avg_edge * 100).toFixed(2) + '%</p>';
                    html += '<p>Errore Predizione: ' + data.rolling_window_metrics.avg_prediction_error.toFixed(1) + ' punti</p>';
                    html += '</div>';
                    
                    html += '<div class="metric-box">';
                    html += '<h3>Giornaliere</h3>';
                    html += '<p>Predizioni: ' + data.daily_metrics.predictions_count + '</p>';
                    html += '<p>Accuratezza: ' + data.daily_metrics.accuracy_rate + '%</p>';
                    html += '<p>Value Bets: ' + data.daily_metrics.value_bets_found + '</p>';
                    html += '</div>';
                    
                    return html;
                }
                
                // Carica inizialmente e poi ogni 30 secondi
                loadMetrics();
                setInterval(loadMetrics, 30000);
            </script>
        </body>
        </html>
        '''
        
        @app.route('/')
        def dashboard():
            return render_template_string(dashboard_template)
        
        @app.route('/api/metrics')
        def api_metrics():
            return jsonify(monitor.get_realtime_metrics())
        
        @app.route('/api/alerts')
        def api_alerts():
            return jsonify(alert_system.check_alerts())
        
        @app.route('/api/report/<date>')
        def api_daily_report(date):
            try:
                target_date = datetime.strptime(date, '%Y-%m-%d').date()
                return jsonify(monitor.generate_daily_report(target_date))
            except ValueError:
                return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
        
        print(f"🌐 Dashboard disponibile su: http://{host}:{port}")
        app.run(host=host, port=port, debug=False)
        
    except ImportError:
        print("❌ Flask non installato. Dashboard web non disponibile.")
        print("   Installa con: pip install flask")

# Esempio di utilizzo completo
def setup_monitoring_system(nba_system):
    """
    Setup completo del sistema di monitoring.
    """
    print("🔧 Setup sistema di monitoring...")
    
    # Inizializza componenti
    monitor = MomentumPerformanceMonitor()
    alert_system = MomentumAlertSystem(monitor)
    
    # Hook nel sistema NBA per logging automatico
    original_analyze_game = nba_system.analyze_game
    
    def analyze_game_with_logging(*args, **kwargs):
        # Esegui analisi originale
        result = original_analyze_game(*args, **kwargs)
        
        # Log predizione (questo richiederà adattamento ai tuoi dati)
        # monitor.log_prediction(game_details, prediction_result, momentum_details)
        
        return result
    
    # Sostituisci metodo con versione instrumentata
    nba_system.analyze_game = analyze_game_with_logging
    
    print("✅ Sistema di monitoring attivo")
    
    return monitor, alert_system

if __name__ == "__main__":
    # Test del sistema di monitoring
    monitor = MomentumPerformanceMonitor()
    alert_system = MomentumAlertSystem(monitor)
    
    # Simula alcune predizioni
    mock_game = {
        'date': '2024-01-15',
        'home_team': 'Lakers',
        'away_team': 'Warriors'
    }
    
    mock_prediction = {
        'predicted_mu': 225.5
    }
    
    mock_momentum = {
        'total_impact': 3.2,
        'confidence_factor': 0.85,
        'home_momentum': {'hot_hands': 1},
        'away_momentum': {'hot_hands': 0},
        'synergy_detected': False
    }
    
    prediction_id = monitor.log_prediction(mock_game, mock_prediction, mock_momentum)
    print(f"Predizione loggata con ID: {prediction_id}")
    
    # Simula outcome
    monitor.update_outcome(prediction_id, 228.0)
    
    # Mostra metriche
    metrics = monitor.get_realtime_metrics()
    print("\n📊 Metriche real-time:")
    print(json.dumps(metrics, indent=2))
    
    # Controlla alert
    alerts = alert_system.check_alerts()
    if alerts:
        print("\n🚨 Alert:")
        for alert in alerts:
            print(f"   {alert['type']}: {alert['message']}")
    else:
        print("\n✅ Nessun alert attivo")