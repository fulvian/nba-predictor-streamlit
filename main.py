# main.py - VERSIONE COMPLETA CON SISTEMA MOMENTUM AVANZATO
import pandas as pd
import numpy as np
import os
import json
import time
import argparse
from datetime import datetime
import sys

# --- Import dei moduli del sistema ---
from data_provider import NBADataProvider
from injury_reporter import InjuryReporter
from player_impact_analyzer import PlayerImpactAnalyzer

# MODIFICA: Usa il nuovo sistema momentum predictor selector
try:
    from momentum_calculator_real import RealMomentumCalculator
    REAL_MOMENTUM_AVAILABLE = True
    print("🎯 Sistema momentum REALE caricato (NBA game logs)")
except ImportError as e:
    REAL_MOMENTUM_AVAILABLE = False
    print(f"⚠️ Sistema momentum reale non disponibile: {e}")

try:
    from momentum_predictor_selector import MomentumPredictorSelector
    MOMENTUM_SELECTOR_AVAILABLE = True
    print("✅ Sistema momentum selector ML caricato")
except ImportError as e:
    print(f"⚠️ Sistema momentum selector non disponibile: {e}")
    print("   Falling back al sistema avanzato...")
    try:
        from advanced_player_momentum_predictor import AdvancedPlayerMomentumPredictor
        MOMENTUM_SELECTOR_AVAILABLE = False
        ADVANCED_MOMENTUM_AVAILABLE = True
        print("📊 Sistema momentum avanzato caricato")
    except ImportError as e2:
        print(f"⚠️ Sistema momentum avanzato non disponibile: {e2}")
        print("   Falling back al sistema base...")
        try:
            from player_momentum_predictor import PlayerMomentumPredictor
            MOMENTUM_SELECTOR_AVAILABLE = False
            ADVANCED_MOMENTUM_AVAILABLE = False
            print("📊 Sistema momentum base caricato")
        except ImportError as e3:
            print(f"❌ Nessun sistema momentum disponibile: {e3}")
            MOMENTUM_SELECTOR_AVAILABLE = False
            ADVANCED_MOMENTUM_AVAILABLE = None

# Modulo probabilistico sempre disponibile
from probabilistic_model import ProbabilisticModel

class NBACompleteSystem:
    def __init__(self, data_provider, auto_mode=False):
        print("🚀 Inizializzazione NBACompleteSystem...")
        self.data_provider = data_provider
        self.auto_mode = auto_mode
        self.bankroll = self._load_bankroll()
        self.impact_analyzer = PlayerImpactAnalyzer(self.data_provider)
        self.injury_reporter = InjuryReporter(self.data_provider)
        
        # Inizializza sistema momentum appropriato - PRIORITÀ AL SISTEMA REALE
        if REAL_MOMENTUM_AVAILABLE:
            self.momentum_predictor = RealMomentumCalculator()
            self.use_real_momentum = True
            self.use_momentum_selector = False
            self.use_advanced_momentum = False
            print("🎯 Sistema momentum REALE attivato (NBA game logs)")
        elif MOMENTUM_SELECTOR_AVAILABLE:
            self.momentum_predictor = MomentumPredictorSelector()
            self.use_real_momentum = False
            self.use_momentum_selector = True
            self.use_advanced_momentum = False
            print("🚀 Sistema momentum selector ML attivato")
        elif ADVANCED_MOMENTUM_AVAILABLE is True:
            self.momentum_predictor = AdvancedPlayerMomentumPredictor(nba_data_provider=self.data_provider)
            self.use_real_momentum = False
            self.use_momentum_selector = False
            self.use_advanced_momentum = True
            print("🔬 Sistema momentum avanzato attivato")
        elif ADVANCED_MOMENTUM_AVAILABLE is False:
            self.momentum_predictor = PlayerMomentumPredictor(nba_data_provider=self.data_provider)
            self.use_real_momentum = False
            self.use_momentum_selector = False
            self.use_advanced_momentum = False
            print("📊 Sistema momentum base attivato")
        else:
            self.momentum_predictor = None
            self.use_real_momentum = False
            self.use_momentum_selector = False
            self.use_advanced_momentum = False
            print("⚠️ Nessun sistema momentum disponibile - continuando senza momentum")
        
        self.probabilistic_model = ProbabilisticModel()

    def _load_bankroll(self, default=89.48):
        """Carica il bankroll dal file JSON, o usa valore di default."""
        # Prova prima data/bankroll.json poi bankroll.json
        bankroll_paths = ['data/bankroll.json', 'bankroll.json']
        
        for bankroll_path in bankroll_paths:
            try:
                with open(bankroll_path, 'r') as f: 
                    data = json.load(f)
                    bankroll_value = float(data.get('current_bankroll', default))
                    print(f"💰 Bankroll caricato da {bankroll_path}: €{bankroll_value:.2f}")
                    return bankroll_value
            except (FileNotFoundError, json.JSONDecodeError) as e:
                continue
        
        print(f"ℹ️ Nessun file bankroll trovato. Usando valore default: €{default}")
        return default

    def _save_bankroll(self, new_bankroll):
        """Salva il bankroll aggiornato nel file JSON."""
        try:
            bankroll_data = {'current_bankroll': float(new_bankroll)}
            
            # Salva nel file principale
            with open('bankroll.json', 'w') as f:
                json.dump(bankroll_data, f, indent=2)
            
            # Salva anche nel backup in data/
            os.makedirs('data', exist_ok=True)
            with open('data/bankroll.json', 'w') as f:
                json.dump(bankroll_data, f, indent=2)
            
            self.bankroll = new_bankroll
            print(f"💰 Bankroll aggiornato e salvato: €{new_bankroll:.2f}")
            
        except Exception as e:
            print(f"⚠️ Errore nel salvataggio del bankroll: {e}")

    def update_bankroll_from_bet(self, bet_result, actual_total=None):
        """
        Aggiorna il bankroll basandosi sul risultato di una scommessa.
        
        Args:
            bet_result: Dict con 'type', 'line', 'odds', 'stake'
            actual_total: Punteggio totale reale della partita
        """
        if not actual_total or not bet_result:
            return
        
        bet_type = bet_result.get('type')
        line = bet_result.get('line')
        odds = bet_result.get('odds')
        stake = bet_result.get('stake')
        
        if not all([bet_type, line, odds, stake]):
            print("⚠️ Informazioni scommessa incomplete per aggiornamento bankroll")
            return
            
        # Determina se la scommessa è vinta
        if bet_type == 'OVER':
            bet_won = actual_total > line
        else:  # UNDER
            bet_won = actual_total <= line
        
        # Calcola profit/loss
        if bet_won:
            profit = stake * (odds - 1)
            new_bankroll = self.bankroll + profit
            print(f"🟢 SCOMMESSA VINTA! Profit: €{profit:.2f}")
        else:
            loss = stake
            new_bankroll = self.bankroll - loss
            print(f"🔴 Scommessa persa. Loss: €{loss:.2f}")
        
        # Salva il nuovo bankroll
        self._save_bankroll(new_bankroll)
        
        return {
            'bet_won': bet_won,
            'profit_loss': profit if bet_won else -stake,
            'new_bankroll': new_bankroll
        }

    def analyze_game(self, game, central_line=None, args=None):
        """
        Analisi completa di una partita NBA con momentum avanzato.
        """
        print(f"\n{'='*80}")
        print(f"🏀 ANALISI PARTITA: {game.get('away_team', 'Away')} @ {game.get('home_team', 'Home')}")
        print(f"{'='*80}")

        # Passo 1: Acquisizione statistiche squadre
        print("1. Acquisizione statistiche squadre...")
        team_stats = self.data_provider.get_team_stats_for_game(
            game.get('home_team', 'Home'), 
            game.get('away_team', 'Away')
        )
        if not team_stats:
            print("   🔴 ❌ Statistiche squadre NON DISPONIBILI - Fallback")
            return {'error': "Impossibile ottenere statistiche squadre"}
        else:
            # Verifica completezza dati
            stats_completeness = self._check_team_stats_completeness(team_stats)
            if stats_completeness >= 0.8:
                print("   🟢 ✅ Statistiche squadre COMPLETE - Sistema attivo")
            elif stats_completeness >= 0.5:
                print("   🟡 ⚠️  Statistiche squadre PARZIALI - Funzionamento ridotto")
            else:
                print("   🔴 ⚠️  Statistiche squadre INCOMPLETE - Dati limitati")

        # Passo 2: Acquisizione roster giocatori
        print("2. Acquisizione roster giocatori...")
        try:
            # Usa injury_reporter per ottenere i roster
            home_roster_list = self.injury_reporter.get_team_roster(game['home_team_id'])
            away_roster_list = self.injury_reporter.get_team_roster(game['away_team_id'])
        
            home_roster_df = pd.DataFrame(home_roster_list) if home_roster_list else pd.DataFrame()
            away_roster_df = pd.DataFrame(away_roster_list) if away_roster_list else pd.DataFrame()
            
            # Rinomina colonne se necessario
            if not home_roster_df.empty and 'id' in home_roster_df.columns: 
                home_roster_df = home_roster_df.rename(columns={'id': 'PLAYER_ID'})
            if not away_roster_df.empty and 'id' in away_roster_df.columns: 
                away_roster_df = away_roster_df.rename(columns={'id': 'PLAYER_ID'})
                
            # Valuta status acquisizione roster
            if not home_roster_df.empty and not away_roster_df.empty:
                roster_completeness = (len(home_roster_df) + len(away_roster_df)) / 30  # ~15 giocatori per squadra
                if roster_completeness >= 0.8:
                    print("   🟢 ✅ Roster giocatori COMPLETI - Sistema injury/momentum attivo")
                elif roster_completeness >= 0.5:
                    print("   🟡 ⚠️  Roster giocatori PARZIALI - Analisi injury/momentum ridotta")
                else:
                    print("   🟡 ⚠️  Roster giocatori LIMITATI - Dati minimi disponibili")
            elif home_roster_df.empty or away_roster_df.empty:
                print("   🔴 ⚠️  Roster INCOMPLETO - Sistema injury/momentum in fallback")
                
        except Exception as e:
            print(f"   🔴 ❌ Errore nell'acquisizione roster: {e} - Sistema in fallback")
            # Continuiamo con roster vuoti invece di interrompere
            home_roster_df = pd.DataFrame()
            away_roster_df = pd.DataFrame()

        # Passo 3: Analisi impatto infortuni
        print("3. Analisi impatto infortuni...")
        try:
            home_impact_result = self.impact_analyzer.calculate_team_impact(home_roster_df, team_id=game.get('home_team_id'))
            away_impact_result = self.impact_analyzer.calculate_team_impact(away_roster_df, team_id=game.get('away_team_id'))
            home_injury_pts = home_impact_result.get('total_impact', 0.0)
            away_injury_pts = away_impact_result.get('total_impact', 0.0)
            injury_impact = home_injury_pts - away_injury_pts
            
            # Salva i risultati per il display finale
            self._last_home_impact_result = home_impact_result
            self._last_away_impact_result = away_impact_result
            
            # Log dettagliato degli infortuni
            self._log_injury_details(game.get('home_team'), home_impact_result)
            self._log_injury_details(game.get('away_team'), away_impact_result)
            
            # Valuta qualità dell'analisi infortuni
            if not home_roster_df.empty and not away_roster_df.empty and self.impact_analyzer:
                if abs(injury_impact) > 0.1:  # Impatto significativo rilevato
                    print(f"   🟢 ⚡ Sistema INJURY ATTIVO - Impatto: Casa={home_injury_pts:+.2f}, Ospite={away_injury_pts:+.2f} | Diff: {injury_impact:+.2f} pts")
                else:
                    print(f"   🟢 ✅ Sistema INJURY ATTIVO - Nessun impatto significativo rilevato")
            elif home_roster_df.empty or away_roster_df.empty:
                print(f"   🟡 ⚠️  Sistema INJURY PARZIALE - Dati roster limitati | Diff: {injury_impact:+.2f} pts")
            else:
                print(f"   🟢 ✅ Sistema INJURY ATTIVO - Impatto: Casa={home_injury_pts:+.2f}, Ospite={away_injury_pts:+.2f} | Diff: {injury_impact:+.2f} pts")
                
        except Exception as e: 
            print(f"   🔴 ❌ Sistema INJURY FALLBACK - Errore: {e} | Impatto azzerato")
            injury_impact = 0.0

        # Passo 4: Calcolo momentum
        print("4. Calcolo momentum giocatori...")
        if self.momentum_predictor:
            if self.use_real_momentum:
                print("   🔄 Inizializzazione sistema MOMENTUM REALE (NBA game logs)...")
                momentum_impact = self._calculate_real_momentum_impact(
                    home_roster_df, away_roster_df, 
                    game.get('home_team'), game.get('away_team')
                )
            elif self.use_momentum_selector:
                print("   🔄 Inizializzazione sistema MOMENTUM SELECTOR ML...")
                momentum_impact = self._calculate_momentum_selector_impact(
                    home_roster_df, away_roster_df, 
                    game.get('home_team'), game.get('away_team'), 
                    team_stats, game
                )
            elif self.use_advanced_momentum:
                print("   🔄 Inizializzazione sistema MOMENTUM AVANZATO ML...")
                momentum_impact = self._calculate_advanced_momentum_impact(
                    home_roster_df, away_roster_df, 
                    game.get('home_team'), game.get('away_team'), 
                    team_stats
                )
            else:
                print("   🔄 Inizializzazione sistema MOMENTUM BASE...")
                try:
                    momentum_value = self.momentum_predictor.calculate_team_momentum(
                        game['home_team_id'], game['away_team_id']
                    )
                    momentum_impact = {'total_impact': momentum_value}
                    print(f"   🟢 ✅ Sistema MOMENTUM BASE attivo - Impatto: {momentum_value:+.2f} pts")
                except Exception as e:
                    print(f"   🔴 ❌ Sistema MOMENTUM FALLBACK - Errore: {e}")
                    momentum_impact = {'total_impact': 0.0}
        else:
            print("   🔴 ❌ Sistema MOMENTUM NON DISPONIBILE - Fallback")
            momentum_impact = {'total_impact': 0.0}

        # Passo 5: Esecuzione modello probabilistico
        print("5. Esecuzione modello probabilistico...")
        try:
            print("   🔄 Caricamento MODELLO PROBABILISTICO ML...")
            distribution = self.probabilistic_model.predict_distribution(
                team_stats, injury_impact, momentum_impact
            )
            if 'error' in distribution:
                print(f"   🔴 ❌ MODELLO PROBABILISTICO FALLBACK - Errore: {distribution['error']}")
            else:
                predicted_total = distribution.get('predicted_mu', 0)
                confidence = distribution.get('predicted_sigma', 0)
                print(f"   🟢 ✅ MODELLO PROBABILISTICO ATTIVO - Predizione: {predicted_total:.1f}±{confidence:.1f}")
        except Exception as e:
            print(f"   🔴 ❌ MODELLO PROBABILISTICO FALLBACK - Errore: {e}")
            distribution = {'error': str(e)}

        # Passo 6: Analisi opportunità scommesse
        print("6. Analisi opportunità scommesse...")
        try:
            print("   🔄 Inizializzazione MOTORE BETTING ANALYSIS...")
            odds_list = game.get('odds', [])
            opportunities = self.probabilistic_model.analyze_betting_opportunities(
                distribution,
                odds_list=odds_list,
                central_line=central_line,
                bankroll=self.bankroll
            )
            
            if opportunities and len(opportunities) > 0:
                value_bets = [opp for opp in opportunities if opp.get('edge', 0) > 0]
                if value_bets:
                    print(f"   🟢 🎯 BETTING ANALYSIS ATTIVO - {len(value_bets)} VALUE bets su {len(opportunities)} linee")
                else:
                    print(f"   🟡 ⚠️  BETTING ANALYSIS ATTIVO - Nessuna VALUE bet su {len(opportunities)} linee")
            else:
                print("   🔴 ❌ BETTING ANALYSIS FALLBACK - Nessuna opportunità generata")
                
        except Exception as e:
            print(f"   🔴 ❌ BETTING ANALYSIS FALLBACK - Errore: {e}")
            opportunities = []
        
        # Passo 7: Mostra riepilogo finale
        self._display_system_status_summary(team_stats, home_roster_df, away_roster_df, 
                                           distribution, opportunities, momentum_impact)
        self.display_final_summary(game, distribution, opportunities, args, momentum_impact, injury_impact)

        return {
            'team_stats': team_stats,
            'injury_impact': injury_impact,
            'momentum_impact': momentum_impact,
            'distribution': distribution,
            'opportunities': opportunities
        }

    def _calculate_momentum_selector_impact(self, home_roster_df, away_roster_df, home_team_name, away_team_name, team_stats, game):
        """Calcola momentum impact usando il nuovo sistema selector ML."""
        try:
            # Prepara le feature mock per il momentum selector
            # In futuro si potranno estrarre direttamente dai dati team_stats
            mock_features = {
                'home_momentum_score': 55.0,
                'home_hot_hand_players': 2,
                'home_avg_player_momentum': 52.0,
                'home_avg_player_weighted_contribution': 0.48,
                'home_team_offensive_potential': 8.2,
                'home_team_defensive_potential': 7.8,
                'away_momentum_score': 47.0,
                'away_hot_hand_players': 1,
                'away_avg_player_momentum': 49.0,
                'away_avg_player_weighted_contribution': 0.45,
                'away_team_offensive_potential': 7.9,
                'away_team_defensive_potential': 8.1,
                'momentum_diff': 8.0
            }
            
            # Estrai feature reali dai team_stats se disponibili
            if team_stats and 'home' in team_stats and 'away' in team_stats:
                home_stats = team_stats['home']
                away_stats = team_stats['away']
                
                # Aggiorna con statistiche reali se disponibili
                if home_stats.get('PPG'):
                    mock_features['home_team_offensive_potential'] = float(home_stats['PPG']) / 10
                if away_stats.get('PPG'):
                    mock_features['away_team_offensive_potential'] = float(away_stats['PPG']) / 10
                if home_stats.get('OPP_PPG'):
                    mock_features['home_team_defensive_potential'] = 12.0 - float(home_stats['OPP_PPG']) / 10
                if away_stats.get('OPP_PPG'):
                    mock_features['away_team_defensive_potential'] = 12.0 - float(away_stats['OPP_PPG']) / 10
            
            # Determina il contesto della partita per il selector
            game_context = {}
            if 'date' in game:
                game_context['game_date'] = game['date']
            
            # Usa il momentum selector per la predizione
            result = self.momentum_predictor.predict(mock_features, **game_context)
            
            if 'error' in result:
                print(f"   🔴 ❌ Sistema MOMENTUM SELECTOR FALLBACK - Errore: {result['error']}")
                return {'total_impact': 0.0, 'error': result['error']}
            
            prediction = result.get('prediction', 0.0)
            model_used = result.get('model_used', 'unknown')
            confidence = result.get('confidence', 0.0)
            reasoning = result.get('reasoning', 'N/A')
            
            # CORREZIONE CRITICA: Il modello predice valori nella scala sbagliata
            # prediction = score_deviation nel formato modello (es. -454.45)
            # Dobbiamo normalizzare e scalare correttamente per l'impatto NBA
            
            # STEP 1: Normalizza la predizione del modello (che è su scala diversa)
            # Se la predizione è estrema, probabilmente è su scala diversa (centinaia vs singole unità)
            if abs(prediction) > 100:
                # Scala da centinaia a unità singole (divide per fattore appropriato)
                normalized_prediction = prediction / 50.0  # Fattore di calibrazione empirico
            else:
                normalized_prediction = prediction
            
            # STEP 2: Applica un clamp realistico per momentum NBA (massimo ±8 punti)
            # I migliori team NBA raramente hanno momentum superiore a 5-8 punti di impatto
            momentum_impact_total = np.clip(normalized_prediction, -8.0, 8.0)
            
            # Calcola confidence basato sul modello usato e accuracy
            if model_used == 'regular_season':
                effective_confidence = confidence * 0.95  # Alta confidenza per regular
            elif model_used == 'playoff':
                effective_confidence = confidence * 0.85  # Buona confidenza per playoff
            else:
                effective_confidence = confidence * 0.75  # Media confidenza per hybrid
            
            # Valuta qualità del sistema
            # DEBUG: Mostra sempre la trasformazione per verificare la correzione
            print(f"   🔧 Debug Momentum: Raw={prediction:+.2f} → Normalized={normalized_prediction:+.2f} → Final={momentum_impact_total:+.2f}")
            
            if momentum_impact_total > 2.0 and effective_confidence > 0.8:
                print(f"   🟢 🚀 Sistema MOMENTUM SELECTOR ML ATTIVO - Modello: {model_used.upper()}")
                print(f"      📊 Predizione Raw: {prediction:+.2f} | Confidence: {confidence:.1%} | Reasoning: {reasoning}")
                print(f"      ⚡ Impatto Finale: {momentum_impact_total:+.2f} pts")
            elif effective_confidence > 0.6:
                print(f"   🟢 ⚖️  Sistema MOMENTUM SELECTOR EQUILIBRATO - Modello: {model_used.upper()}")
                print(f"      📊 Predizione Raw: {prediction:+.2f} | Confidence: {confidence:.1%}")
                print(f"      ⚡ Impatto Finale: {momentum_impact_total:+.2f} pts")
            else:
                print(f"   🟡 ⚠️  Sistema MOMENTUM SELECTOR INCERTO - Modello: {model_used.upper()}")
                print(f"      📊 Predizione Raw: {prediction:+.2f} | Confidence: {confidence:.1%}")
                print(f"      ⚡ Impatto Finale: {momentum_impact_total:+.2f} pts")
            
            return {
                'total_impact': momentum_impact_total,
                'prediction': prediction,
                'model_used': model_used,
                'confidence_factor': effective_confidence,
                'reasoning': reasoning,
                'selector_system': True
            }
            
        except Exception as e:
            print(f"   🔴 ❌ Sistema MOMENTUM SELECTOR FALLBACK - Errore: {e}")
            return {'total_impact': 0.0, 'error': str(e)}

    def _calculate_advanced_momentum_impact(self, home_roster_df, away_roster_df, home_team_name, away_team_name, team_stats):
        """Calcola momentum impact usando il sistema avanzato."""
        try:
            # Combina i roster e calcola per entrambe le squadre
            home_result = self.momentum_predictor.predict_team_momentum_impact_advanced(home_roster_df)
            away_result = self.momentum_predictor.predict_team_momentum_impact_advanced(away_roster_df)
            
            # Calcola l'impatto e formatta nel formato atteso
            home_impact = home_result.get('impact_on_totals', 0.0)
            away_impact = away_result.get('impact_on_totals', 0.0)
            home_momentum_score = home_result.get('momentum_score', 50.0)
            away_momentum_score = away_result.get('momentum_score', 50.0)
            
            # NUOVO: Calcola confidence_factor reale basato sulla qualità dei dati
            confidence_factor = self._calculate_momentum_confidence_factor(
                home_roster_df, away_roster_df, home_result, away_result
            )
            
            result = {
                'home_team_momentum': {'final_score': home_momentum_score},
                'away_team_momentum': {'final_score': away_momentum_score},
                'differential_impact': home_impact + away_impact,
                'confidence_factor': confidence_factor,  # Ora calcolato realmente
                'synergy_detected': (home_result.get('hot_hand_players_count', 0) + away_result.get('hot_hand_players_count', 0)) >= 3
            }
            
            if 'error' in result:
                print(f"   🔴 ❌ Sistema MOMENTUM AVANZATO FALLBACK - Errore: {result['error']}")
                return {'total_impact': 0.0, 'error': result['error']}
            
            # Estrai i dati dal risultato avanzato
            home_momentum_score = result.get('home_team_momentum', {}).get('final_score', 50.0)
            away_momentum_score = result.get('away_team_momentum', {}).get('final_score', 50.0)
            
            # Calcola l'impatto finale
            momentum_differential = (home_momentum_score - away_momentum_score) * 0.15  # Fattore di scala per sistema avanzato
            
            # Valuta qualità del sistema momentum avanzato
            if not home_roster_df.empty and not away_roster_df.empty:
                synergy_factor = result.get('synergy_detected', False)
                confidence = result.get('confidence_factor', 1.0)
                
                if abs(momentum_differential) > 1.0 and confidence > 0.7:
                    print(f"   🟢 🚀 Sistema MOMENTUM AVANZATO ML ATTIVO - Casa={home_momentum_score:.2f}, Ospite={away_momentum_score:.2f} | Impatto: {momentum_differential:+.2f} pts | Confidence: {confidence:.1%}")
                    if synergy_factor:
                        print(f"   🟢 ⚡ SINERGIA HOT-HAND rilevata - Confidence: {confidence:.1%}")
                elif confidence > 0.5:
                    if abs(momentum_differential) < 1.0:
                        print(f"   🟢 ⚖️  Sistema MOMENTUM AVANZATO EQUILIBRATO - Casa={home_momentum_score:.2f}, Ospite={away_momentum_score:.2f} | Impatto minimo: {momentum_differential:+.2f} pts | Confidence: {confidence:.1%}")
                    else:
                        print(f"   🟡 ⚠️  Sistema MOMENTUM AVANZATO MODERATO - Casa={home_momentum_score:.2f}, Ospite={away_momentum_score:.2f} | Impatto: {momentum_differential:+.2f} pts | Confidence: {confidence:.1%}")
                else:
                    print(f"   🟡 ⚠️  Sistema MOMENTUM AVANZATO INCERTO - Casa={home_momentum_score:.2f}, Ospite={away_momentum_score:.2f} | Impatto: {momentum_differential:+.2f} pts | Confidence: {confidence:.1%}")
            else:
                print(f"   🟡 ⚠️  Sistema MOMENTUM AVANZATO LIMITATO - Dati roster incompleti | Impatto: {momentum_differential:+.2f} pts")
            
            return {
                'total_impact': momentum_differential,
                'home_momentum': {'score': home_momentum_score},
                'away_momentum': {'score': away_momentum_score},
                'confidence_factor': confidence_factor,
                'synergy_detected': result.get('synergy_detected', False),
                'advanced_system': True
            }
            
        except Exception as e:
            print(f"   🔴 ❌ Sistema MOMENTUM AVANZATO FALLBACK - Errore: {e}")
            return {'total_impact': 0.0, 'error': str(e)}

    def _calculate_momentum_confidence_factor(self, home_roster_df, away_roster_df, home_result, away_result):
        """
        Calcola un fattore di confidenza reale basato sulla qualità dei dati del roster
        e della completezza delle analisi momentum.
        
        Returns:
            float: Valore tra 0.0 e 1.0 che rappresenta la confidenza nel sistema momentum
        """
        confidence_components = []
        
        # 1. Qualità del roster data (25% del peso)
        roster_quality = 0.0
        home_roster_size = len(home_roster_df) if not home_roster_df.empty else 0
        away_roster_size = len(away_roster_df) if not away_roster_df.empty else 0
        
        if home_roster_size >= 10 and away_roster_size >= 10:
            roster_quality = 1.0  # Roster completi
        elif home_roster_size >= 7 and away_roster_size >= 7:
            roster_quality = 0.8  # Roster sufficienti
        elif home_roster_size >= 5 and away_roster_size >= 5:
            roster_quality = 0.6  # Roster minimi
        else:
            roster_quality = 0.3  # Roster insufficienti
        
        confidence_components.append(('roster_quality', roster_quality, 0.25))
        
        # 2. Completezza dell'analisi momentum (30% del peso)
        home_contributions = home_result.get('player_contributions', [])
        away_contributions = away_result.get('player_contributions', [])
        
        analysis_completeness = 0.0
        total_analyzed_players = len(home_contributions) + len(away_contributions)
        
        if total_analyzed_players >= 16:  # ~8 per squadra
            analysis_completeness = 1.0
        elif total_analyzed_players >= 12:  # ~6 per squadra
            analysis_completeness = 0.8
        elif total_analyzed_players >= 8:   # ~4 per squadra
            analysis_completeness = 0.6
        else:
            analysis_completeness = 0.4
        
        confidence_components.append(('analysis_completeness', analysis_completeness, 0.30))
        
        # 3. Qualità dei momentum scores (25% del peso)
        home_momentum = home_result.get('momentum_score', 50.0)
        away_momentum = away_result.get('momentum_score', 50.0)
        
        # Score che si allontanano da 50 (neutro) indicano segnali più forti
        home_signal_strength = min(abs(home_momentum - 50) / 50, 1.0)
        away_signal_strength = min(abs(away_momentum - 50) / 50, 1.0)
        signal_quality = (home_signal_strength + away_signal_strength) / 2
        
        confidence_components.append(('signal_quality', signal_quality, 0.25))
        
        # 4. Presenza di hot hands e sinergie (20% del peso)
        hot_hands_home = home_result.get('hot_hand_players_count', 0)
        hot_hands_away = away_result.get('hot_hand_players_count', 0)
        total_hot_hands = hot_hands_home + hot_hands_away
        
        hot_hands_bonus = 0.0
        if total_hot_hands >= 3:
            hot_hands_bonus = 1.0  # Molti hot hands = alta confidenza
        elif total_hot_hands >= 2:
            hot_hands_bonus = 0.8
        elif total_hot_hands >= 1:
            hot_hands_bonus = 0.6
        else:
            hot_hands_bonus = 0.4  # Nessun hot hand = confidenza media
        
        confidence_components.append(('hot_hands_bonus', hot_hands_bonus, 0.20))
        
        # Calcola la confidenza finale come media pesata
        final_confidence = sum(score * weight for _, score, weight in confidence_components)
        
        # Applica un minimo di confidenza del 30% per evitare valori troppo bassi
        final_confidence = max(0.3, min(1.0, final_confidence))
        
        # Debug: mostra i componenti della confidenza
        if final_confidence < 0.7:  # Solo se non è alta confidenza
            component_details = ", ".join([f"{name}: {score:.2f}" for name, score, _ in confidence_components])
            print(f"   🔧 Debug Confidence: {component_details} → Final: {final_confidence:.2f}")
        
        return final_confidence

    def _check_team_stats_completeness(self, team_stats):
        """
        Verifica la completezza delle statistiche squadre.
        Returns: float tra 0 e 1 (percentuale di completezza)
        """
        if not team_stats:
            return 0.0
        
        present_fields = 0
        
        # Il controllo ora è più semplice: basta che i dizionari principali esistano.
        # Le chiavi corrette sono 'home' e 'away', non 'home_team_stats'.
        if 'home' in team_stats and team_stats.get('home', {}).get('has_data'):
            present_fields += 1
        if 'away' in team_stats and team_stats.get('away', {}).get('has_data'):
            present_fields += 1
        
        # Un check più robusto: entrambi devono essere presenti e validi.
        completeness = present_fields / 2.0
        return completeness

    def _display_system_status_summary(self, team_stats, home_roster_df, away_roster_df, 
                                      distribution, opportunities, momentum_impact):
        """
        Mostra un riepilogo visuale dello stato di tutti i sistemi.
        """
        print(f"\n🔧 RIEPILOGO STATO SISTEMI")
        print("="*60)
        
        # Sistema Statistiche squadre
        stats_completeness = self._check_team_stats_completeness(team_stats)
        if stats_completeness >= 1.0:
            print("🟢 STATISTICHE SQUADRE:    ✅ COMPLETE")
        elif stats_completeness > 0:
            print("🟡 STATISTICHE SQUADRE:    ⚠️  PARZIALI")
        else:
            print("🔴 STATISTICHE SQUADRE:    ❌ INCOMPLETE")
        
        # Sistema Roster/Injury
        roster_available = not home_roster_df.empty and not away_roster_df.empty
        if roster_available:
            roster_size = len(home_roster_df) + len(away_roster_df)
            if roster_size >= 24:  # ~12 per squadra
                print("🟢 SISTEMA INJURY:         ✅ ATTIVO")
            else:
                print("🟡 SISTEMA INJURY:         ⚠️  PARZIALE")
        else:
            print("🔴 SISTEMA INJURY:         ❌ FALLBACK")
        
        # Sistema Momentum
        if momentum_impact and momentum_impact.get('total_impact') is not None:
            if momentum_impact.get('real_data_system'):
                confidence = momentum_impact.get('confidence_factor', 0)
                total_impact = momentum_impact.get('total_impact', 0)
                home_perf = momentum_impact.get('home_performance', 'unknown')
                away_perf = momentum_impact.get('away_performance', 'unknown')
                if abs(total_impact) > 2.0 and confidence > 0.8:
                    print(f"🟢 SISTEMA MOMENTUM:       🎯 NBA REALE ATTIVO ({home_perf}/{away_perf}, Conf: {confidence:.0%})")
                elif confidence > 0.6:
                    print(f"🟢 SISTEMA MOMENTUM:       ⚖️  NBA REALE EQUILIBRATO ({home_perf}/{away_perf}, Conf: {confidence:.0%})")
                else:
                    print(f"🟡 SISTEMA MOMENTUM:       ⚠️  NBA REALE INCERTO ({home_perf}/{away_perf}, Conf: {confidence:.0%})")
            elif momentum_impact.get('selector_system'):
                confidence = momentum_impact.get('confidence_factor', 0)
                model_used = momentum_impact.get('model_used', 'unknown')
                total_impact = momentum_impact.get('total_impact', 0)
                if abs(total_impact) > 2.0 and confidence > 0.8:
                    print(f"🟢 SISTEMA MOMENTUM:       🚀 ML SELECTOR ATTIVO ({model_used.upper()}, Conf: {confidence:.0%})")
                elif confidence > 0.6:
                    print(f"🟢 SISTEMA MOMENTUM:       ⚖️  ML SELECTOR EQUILIBRATO ({model_used.upper()}, Conf: {confidence:.0%})")
                else:
                    print(f"🟡 SISTEMA MOMENTUM:       ⚠️  ML SELECTOR INCERTO ({model_used.upper()}, Conf: {confidence:.0%})")
            elif momentum_impact.get('advanced_system'):
                confidence = momentum_impact.get('confidence_factor', 0)
                total_impact = momentum_impact.get('total_impact', 0)
                if abs(total_impact) > 1.0 and confidence > 0.7:
                    print(f"🟢 SISTEMA MOMENTUM:       🚀 ML AVANZATO ATTIVO (Conf: {confidence:.0%})")
                elif confidence > 0.5:
                    if abs(total_impact) < 1.0:
                        print(f"🟢 SISTEMA MOMENTUM:       ⚖️  ML AVANZATO EQUILIBRATO (Conf: {confidence:.0%})")
                    else:
                        print(f"🟡 SISTEMA MOMENTUM:       ⚠️  ML AVANZATO MODERATO (Conf: {confidence:.0%})")
                else:
                    print(f"🟡 SISTEMA MOMENTUM:       ⚠️  ML AVANZATO INCERTO (Conf: {confidence:.0%})")
            elif 'base_system' in momentum_impact:
                print("🟢 SISTEMA MOMENTUM:       ✅ BASE ATTIVO")
            else:
                print("🔴 SISTEMA MOMENTUM:       ❌ FALLBACK")
        else:
            print("🔴 SISTEMA MOMENTUM:       ❌ NON DISPONIBILE")

        
        # Modello Probabilistico
        if distribution and 'error' not in distribution:
            print("🟢 MODELLO PROBABILISTICO: ✅ ML ATTIVO")
        else:
            print("🔴 MODELLO PROBABILISTICO: ❌ FALLBACK")
        
        # Sistema Betting
        if opportunities and len(opportunities) > 0:
            value_bets = [opp for opp in opportunities if opp.get('edge', 0) > 0 and opp.get('probability', 0) >= 0.50]
            if value_bets:
                print("🟢 BETTING ANALYSIS:       🎯 ATTIVO + VALUE")
            else:
                print("🟡 BETTING ANALYSIS:       ⚠️  ATTIVO - NO VALUE")
        else:
            print("🔴 BETTING ANALYSIS:       ❌ FALLBACK")
        
        print("="*60)

    def _log_injury_details(self, team_name, impact_result):
        """Logga i dettagli degli infortuni per una squadra."""
        injured_players_details = impact_result.get('injured_players_details', [])
        total_impact = impact_result.get('total_impact', 0.0)

        if total_impact == 0 and not injured_players_details:
             print(f"   🟢 {team_name}: Nessun infortunio con impatto rilevato.")
             return
        
        print(f"   ⚪ Dettaglio Infortuni per {team_name} (Impatto Totale: {total_impact:.2f} pts):")

        if not injured_players_details:
            print("     - Nessun dettaglio giocatore disponibile.")
            return

        for detail_string in injured_players_details:
            print(f"     - {detail_string}")

    def _calculate_real_momentum_impact(self, home_roster_df, away_roster_df, home_team_name, away_team_name):
        """Calcola momentum impact usando dati NBA reali (game logs + plus/minus)."""
        try:
            # Usa il RealMomentumCalculator per calcolare l'impatto
            result = self.momentum_predictor.calculate_game_momentum_differential(
                home_roster_df, away_roster_df, home_team_name, away_team_name
            )
            
            total_impact = result.get('total_impact', 0.0)
            confidence = result.get('confidence_factor', 1.0)
            home_performance = result.get('home_momentum', {}).get('team_performance', 'unknown')
            away_performance = result.get('away_momentum', {}).get('team_performance', 'unknown')
            
            # Valuta qualità del sistema momentum reale
            if abs(total_impact) > 2.0 and confidence > 0.8:
                print(f"   🟢 🎯 Sistema MOMENTUM REALE ATTIVO - Casa: {home_performance}, Ospite: {away_performance}")
                print(f"      📊 Impatto: {total_impact:+.2f} pts | Confidence: {confidence:.1%}")
            elif confidence > 0.6:
                print(f"   🟢 ⚖️  Sistema MOMENTUM REALE EQUILIBRATO - Casa: {home_performance}, Ospite: {away_performance}")
                print(f"      📊 Impatto: {total_impact:+.2f} pts | Confidence: {confidence:.1%}")
            else:
                print(f"   🟡 ⚠️  Sistema MOMENTUM REALE INCERTO - Casa: {home_performance}, Ospite: {away_performance}")
                print(f"      📊 Impatto limitato: {total_impact:+.2f} pts | Confidence: {confidence:.1%}")
            
            return {
                'total_impact': total_impact,
                'home_momentum': result.get('home_momentum', {}),
                'away_momentum': result.get('away_momentum', {}),
                'confidence_factor': confidence,
                'real_data_system': True,
                'home_performance': home_performance,
                'away_performance': away_performance
            }
            
        except Exception as e:
            print(f"   🔴 ❌ Sistema MOMENTUM REALE FALLBACK - Errore: {e}")
            return {'total_impact': 0.0, 'error': str(e)}

    def _calculate_base_momentum_impact(self, home_team_id, away_team_id):
        """Calcola momentum impact usando il sistema base."""
        try:
            momentum_value = self.momentum_predictor.calculate_team_momentum(home_team_id, away_team_id)
            print(f"   🟢 ✅ Sistema MOMENTUM BASE attivo - Impatto: {momentum_value:+.2f} pts")
            
            return {
                'total_impact': momentum_value,
                'base_system': True
            }
            
        except Exception as e:
            print(f"   🔴 ❌ Sistema MOMENTUM BASE FALLBACK - Errore: {e}")
            return {'total_impact': 0.0, 'error': str(e)}

    def display_final_summary(self, game, distribution, opportunities, args, momentum_impact, injury_impact=0.0):
        """
        Mostra il riepilogo finale con opzioni di scommessa.
        """
        optimal_bet = None
        print(f"\n🎯 RIEPILOGO FINALE")
        print("="*80)
        
        if 'error' in distribution:
            print(f"❌ Errore nella distribuzione: {distribution['error']}")
            return
        
        predicted_total = distribution['predicted_mu']
        confidence_sigma = distribution['predicted_sigma']
        
        # Estrai informazioni dettagliate
        home_team = game.get('home_team', 'Home')
        away_team = game.get('away_team', 'Away')
        
        # Calcola score predetti (assumendo split 50/50 con variazioni)
        home_predicted = predicted_total / 2 + (momentum_impact.get('total_impact', 0) / 2) + (injury_impact / 2)
        away_predicted = predicted_total / 2 - (momentum_impact.get('total_impact', 0) / 2) - (injury_impact / 2)
        
        # Calcola confidence percentage (inverso della sigma, normalizzato)
        confidence_percentage = max(0, min(100, 100 - (confidence_sigma - 10) * 3))
        
        print(f"🏀 Partita: {away_team} @ {home_team}")
        print(f"📊 Score Predetto: {away_team} {away_predicted:.1f} - {home_predicted:.1f} {home_team}")
        print(f"📊 Totale Predetto: {predicted_total:.1f} punti")
        print(f"📈 Confidenza Predizione: {confidence_percentage:.1f}% (σ: {confidence_sigma:.1f})")
        print(f"🏥 Injury Impact: {injury_impact:+.2f} punti")
        print(f"⚡ Momentum Impact: {momentum_impact.get('total_impact', 0):+.2f} punti")
        print(f"💰 Bankroll Attuale: €{self.bankroll:.2f}")

        # Status compatto su una riga
        momentum_conf = momentum_impact.get('confidence_factor', 1.0) * 100
        print(f"┃ \033[92m🟢\033[0m Stats  \033[92m🟢\033[0m Injury  \033[92m🟢\033[0m Momentum({momentum_conf:.0f}%)  \033[92m🟢\033[0m Probabilistic  \033[93m🟡\033[0m Betting ┃")
        print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
        
        # TABELLA INJURY DETAILS
        print(f"\n┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
        print(f"┃                              🏥 INJURY DETAILS                               ┃")
        print(f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫")
        
        # Recupera injury details dai dati reali calcolati dal sistema
        # Usa gli impact result che sono stati calcolati durante l'analisi
        home_impact_result = getattr(self, '_last_home_impact_result', {'injured_players_details': []})
        away_impact_result = getattr(self, '_last_away_impact_result', {'injured_players_details': []})
        
        # Estrai dati dalle details string (formato: "Nome (status) - Impatto: -X.XX pts")
        home_injuries = []
        away_injuries = []
        
        # PROCESSING HOME INJURIES - Estrai dati reali NBA
        for detail in home_impact_result.get('injured_players_details', []):
            try:
                # Parse del formato: "Jarace Walker (Out) - Impatto: -0.53 pts [NBA Data]"
                if ' - Impatto: ' in detail:
                    player_part, impact_part = detail.split(' - Impatto: ')
                    player_name = player_part.split(' (')[0]
                    status = player_part.split(' (')[1].split(')')[0].upper()
                    # Rimuovi sia ' pts' che eventuali tag come '[NBA Data]'
                    impact_clean = impact_part.replace(' pts', '').split('[')[0].strip()
                    impact = abs(float(impact_clean))
                    home_injuries.append({"player": player_name, "status": status, "impact": impact})
            except Exception as e:
                # Ignora errori di parsing - mantieni solo dati validi
                continue
        
        # PROCESSING AWAY INJURIES - Estrai dati reali NBA
        for detail in away_impact_result.get('injured_players_details', []):
            try:
                if ' - Impatto: ' in detail:
                    player_part, impact_part = detail.split(' - Impatto: ')
                    player_name = player_part.split(' (')[0]
                    status = player_part.split(' (')[1].split(')')[0].upper()
                    # Rimuovi sia ' pts' che eventuali tag come '[NBA Data]'
                    impact_clean = impact_part.replace(' pts', '').split('[')[0].strip()
                    impact = abs(float(impact_clean))
                    away_injuries.append({"player": player_name, "status": status, "impact": impact})
            except Exception as e:
                # Ignora errori di parsing - mantieni solo dati validi
                continue
        
        # Se non ci sono injuries REALI, mostra messaggio appropriato
        if not home_injuries and not away_injuries:
            home_injuries = [{"player": "Nessun infortunio", "status": "ACTIVE", "impact": 0.00}]
            away_injuries = [{"player": "Nessun infortunio", "status": "ACTIVE", "impact": 0.00}]
        elif not home_injuries:
            home_injuries = [{"player": "Nessun infortunio", "status": "ACTIVE", "impact": 0.00}]
        elif not away_injuries:
            away_injuries = [{"player": "Nessun infortunio", "status": "ACTIVE", "impact": 0.00}]
        
        home_total_impact = sum(inj["impact"] for inj in home_injuries)
        away_total_impact = sum(inj["impact"] for inj in away_injuries)
        
        print(f"┃ 🏠 {home_team:<15} │ 🛫 {away_team:<15} │ Impact Comparison         ┃")
        print(f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫")
        
        max_injuries = max(len(home_injuries), len(away_injuries))
        for i in range(max_injuries):
            home_player = home_injuries[i]["player"][:15] if i < len(home_injuries) else ""
            home_impact = f"+{home_injuries[i]['impact']:.2f}" if i < len(home_injuries) else ""
            
            away_player = away_injuries[i]["player"][:15] if i < len(away_injuries) else ""
            away_impact = f"+{away_injuries[i]['impact']:.2f}" if i < len(away_injuries) else ""
            
            print(f"┃ \033[91m🚨\033[0m {home_player:<13} {home_impact:<6} │ \033[91m🚨\033[0m {away_player:<13} {away_impact:<6} │                           ┃")
        
        print(f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫")
        print(f"┃ \033[1mTotal Impact: \033[92m+{home_total_impact:.2f} pts\033[0m │ \033[1mTotal Impact: \033[92m+{away_total_impact:.2f} pts\033[0m │ Net: {injury_impact:+.2f} pts        ┃")
        print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
        
        # SISTEMA STATUS COMPATTO
        print(f"\n┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
        print(f"┃                              🔧 SYSTEM STATUS                               ┃")
        print(f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫")
        
        # Status compatto su una riga
        momentum_conf = momentum_impact.get('confidence_factor', 1.0) * 100
        print(f"┃ \033[92m🟢\033[0m Stats  \033[92m🟢\033[0m Injury  \033[92m🟢\033[0m Momentum({momentum_conf:.0f}%)  \033[92m🟢\033[0m Probabilistic  \033[93m🟡\033[0m Betting ┃")
        print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")

        # ANALISI SCOMMESSE COMPLETA
        if opportunities and isinstance(opportunities, list):
            all_opportunities = sorted(opportunities, key=lambda x: x.get('edge', 0), reverse=True)
            
            # Filtra VALUE bets (edge > 0 e prob >= 50%)
            value_bets = [opp for opp in all_opportunities if opp.get('edge', 0) > 0 and opp.get('probability', 0) >= 0.5]
            
            print(f"\n┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            print(f"┃                        💎 ANALISI SCOMMESSE                                ┃")
            print(f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫")
            
            all_betting_options = []  # Inizializza qui
            
            if value_bets:
                # CASO 1: Ci sono VALUE bets - sistema di raccomandazioni categorizzate
                print(f"┃ 🎯 Trovate {len(value_bets)} opportunità VALUE su {len(all_opportunities)} linee analizzate        ┃")
                print(f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫")
                print(f"┃ #  CATEGORIA                  LINE    ODDS   EDGE    PROB   QUAL   STAKE   ┃")
                print(f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫")
                
                # Calcola le raccomandazioni categorizzate
                optimal_bet = self._calculate_optimal_bet(all_opportunities)
                highest_prob_bet = max(value_bets, key=lambda x: x.get('probability', 0))
                highest_edge_bet = max(value_bets, key=lambda x: x.get('edge', 0))
                # Quota più alta solo tra VALUE bets
                highest_odds_bet = max(value_bets, key=lambda x: x.get('odds', 0))
                
                # Lista delle raccomandazioni principali
                recommendations = []
                
                # 1. SCELTA DEL SISTEMA (Ottimale)
                if optimal_bet:
                    recommendations.append({
                        'bet': optimal_bet,
                        'category': '🏆 SCELTA DEL SISTEMA',
                        'color': '\033[93m'  # Giallo oro
                    })
                
                # 2. PIÙ PROBABILE
                recommendations.append({
                    'bet': highest_prob_bet,
                    'category': '📊 MASSIMA PROBABILITÀ',
                    'color': '\033[92m'  # Verde
                })
                
                # 3. MASSIMO EDGE
                recommendations.append({
                    'bet': highest_edge_bet,
                    'category': '🔥 MASSIMO EDGE',
                    'color': '\033[91m'  # Rosso
                })
                
                # 4. QUOTA MAGGIORE
                recommendations.append({
                    'bet': highest_odds_bet,
                    'category': '💰 QUOTA MASSIMA',
                    'color': '\033[95m'  # Magenta
                })
                
                # Rimuovi duplicati mantenendo l'ordine
                seen_bets = set()
                unique_recommendations = []
                for rec in recommendations:
                    bet_key = f"{rec['bet']['type']}_{rec['bet']['line']}"
                    if bet_key not in seen_bets:
                        seen_bets.add(bet_key)
                        unique_recommendations.append(rec)
                
                # Mostra le raccomandazioni principali
                for i, rec in enumerate(unique_recommendations, 1):
                    bet = rec['bet']
                    edge = bet.get('edge', 0) * 100
                    prob = bet.get('probability', 0) * 100
                    quality = bet.get('quality_score', 0)
                    
                    print(f"┃ {rec['color']}{i:<2} {rec['category']:<24} {bet['type']} {bet['line']:<6} {bet['odds']:<6.2f} {edge:<6.1f}% {prob:<5.1f}% {quality:<5.1f} €{bet['stake']:<6.1f}\033[0m ┃")
                
                # Aggiungi le altre VALUE bets
                other_bets = []
                for bet in value_bets:
                    bet_key = f"{bet['type']}_{bet['line']}"
                    if bet_key not in seen_bets:
                        other_bets.append(bet)
                        seen_bets.add(bet_key)
                
                # Ordina le altre per STAKE decrescente (dal maggiore al minore)
                other_bets = sorted(other_bets, key=lambda x: x.get('stake', 0), reverse=True)
                
                if other_bets:
                    print(f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫")
                    
                    for i, bet in enumerate(other_bets, len(unique_recommendations) + 1):
                        edge = bet.get('edge', 0) * 100
                        prob = bet.get('probability', 0) * 100
                        quality = bet.get('quality_score', 0)
                        
                        print(f"┃ \033[96m{i:<2} ALTRE VALUE BETS        {bet['type']} {bet['line']:<6} {bet['odds']:<6.2f} {edge:<6.1f}% {prob:<5.1f}% {quality:<5.1f} €{bet['stake']:<6.1f}\033[0m ┃")
                
                all_betting_options = unique_recommendations + [{'bet': bet, 'category': 'VALUE', 'color': '\033[96m'} for bet in other_bets]
                
            else:
                # CASO 2: Nessun VALUE bet - mostra le migliori 5
                print(f"┃ ❌ Nessuna opportunità VALUE trovata - prime 5 opzioni migliori             ┃")
                print(f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫")
                print(f"┃ #  TIPO                  LINE    ODDS   EDGE    PROB   QUAL   STAKE        ┃")
                print(f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫")
                
                top_5_bets = all_opportunities[:5]
                
                for i, bet in enumerate(top_5_bets, 1):
                    edge = bet.get('edge', 0) * 100
                    prob = bet.get('probability', 0) * 100
                    quality = bet.get('quality_score', 0)
                    
                    # Colori basati sull'edge (tutti negativi in questo caso)
                    if edge > -2.0:
                        row_color = "\033[93m"  # Giallo (migliore tra i negativi)
                        status = "MARGINALE"
                    elif edge > -5.0:
                        row_color = "\033[91m"  # Rosso chiaro
                        status = "SCARSA"
                    else:
                        row_color = "\033[90m"  # Grigio
                        status = "PESSIMA"
                    
                    print(f"┃ {row_color}{i:<2} {status:<19} {bet['type']} {bet['line']:<6} {bet['odds']:<6.2f} {edge:<6.1f}% {prob:<5.1f}% {quality:<5.1f} €{bet['stake']:<6.1f}\033[0m ┃")
                    
                    all_betting_options.append({
                        'bet': bet,
                        'category': status,
                        'color': row_color
                    })
            
            print(f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫")
            print(f"┃ 💡 VALUE = Edge > 0% AND Probabilità ≥ 50%                                  ┃")
            print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
            
            # PRIMO RIEPILOGO - SCOMMESSA CONSIGLIATA DAL SISTEMA (SCELTA DEL SISTEMA)
            # Usa la stessa scommessa già calcolata per la tabella per mantenere coerenza
            if optimal_bet:
                self._show_bet_summary(game, distribution, optimal_bet, args, "🏆 SCELTA DEL SISTEMA", momentum_impact, injury_impact)

            # SELEZIONE INTERATTIVA
            if all_betting_options and not args.auto_mode:
                print(f"\n🎯 CENTRO COMANDO SCOMMESSE")
                print("="*80)
                
                while True:
                    try:
                        choice_text = f"\nSeleziona il numero della raccomandazione (1-{len(all_betting_options)}) o 0 per nessuna scommessa: "
                        choice = input(choice_text).strip()
                        
                        if not choice:
                            continue
                        
                        choice = int(choice)
                    
                        if choice == 0:
                            print("\n❌ Nessuna scommessa selezionata")
                            break
                        elif 1 <= choice <= len(all_betting_options):
                            selected_option = all_betting_options[choice - 1]
                            selected_bet = selected_option['bet']
                            category = selected_option['category']
                            
                            print(f"\n✅ Hai selezionato:")
                            print(f"   📋 Categoria: {category}")
                            print(f"   🎯 Scommessa: {selected_bet['type']} {selected_bet['line']} @ {selected_bet['odds']:.2f}")
                            print(f"   💰 Stake: €{selected_bet['stake']:.2f}")
                            print(f"   📊 Edge: {selected_bet.get('edge', 0)*100:.1f}% | Probabilità: {selected_bet.get('probability', 0)*100:.1f}%")
                            
                            confirm = input("\nConfermi questa scommessa? (y/N): ").strip().lower()
                            if confirm == 'y':
                                game_id = game.get('game_id')
                                if game_id:
                                    self.save_pending_bet(selected_bet, game_id)
                                    print("📲 Scommessa salvata! Il sistema aggiornerà automaticamente il bankroll.")
                                print("✅ Scommessa confermata!")
                                
                                # SECONDO RIEPILOGO - SCOMMESSA EFFETTIVAMENTE SELEZIONATA
                                self._show_bet_summary(game, distribution, selected_bet, args, f"✅ {category}", momentum_impact, injury_impact)
                                break # Esce dal loop dopo conferma
                            else:
                                print("❌ Scommessa annullata")
                                # Non esce dal loop, permette altra scelta
                        else:
                            print(f"❌ Numero non valido. Scegli tra 1-{len(all_betting_options)} o 0.")
                    except ValueError:
                        print("❌ Inserisci un numero valido")
                    except EOFError:
                        print("\n❌ Input interrotto. Nessuna scommessa selezionata.")
                        break
        
        else:
            print(f"\n┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            print(f"┃                           💎 BETTING ANALYSIS                               ┃")
            print(f"┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫")
            print(f"┃ ❌ No betting analysis available                                             ┃")
            print(f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
        
        # FOOTER
        print(f"\n\033[92m▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓\033[0m")
        print("✅ Analysis completed successfully!\n")

    def _show_bet_summary(self, game, distribution, bet, args, category_title, momentum_impact, injury_impact):
        """
        Mostra il riepilogo di una scommessa specifica con formattazione migliorata.
        """
        away_team = game.get('away_team', 'Away')
        home_team = game.get('home_team', 'Home')
        predicted_total = distribution.get('predicted_mu', 0)
        confidence_sigma = distribution.get('predicted_sigma', 0)
        
        # Estrai dati scommessa
        bet_type_full = "OVER" if bet.get('type') == 'OVER' else "UNDER"
        bet_line = bet.get('line', 0)
        bet_odds = bet.get('odds', 0)
        bet_stake = bet.get('stake', 0)
        opt_edge = bet.get('edge', 0) * 100
        opt_prob = bet.get('probability', 0) * 100
        opt_quality = bet.get('quality_score', 0) * 100
        
        # Calcola potenziale vincita e ROI
        potential_win = bet_stake * (bet_odds - 1)
        roi_percent = (potential_win / bet_stake * 100) if bet_stake > 0 else 0
        
        # Informazioni partita - usa data reale dal game object
        if 'date' in game and game['date']:
            # Converte da formato '2025-06-22' a '22/06/2025'
            try:
                game_date_obj = datetime.strptime(game['date'], '%Y-%m-%d')
                game_date = game_date_obj.strftime("%d/%m/%Y")
            except (ValueError, TypeError):
                game_date = game.get('date', datetime.now().strftime("%d/%m/%Y"))
        else:
            game_date = datetime.now().strftime("%d/%m/%Y")
        
        game_time = game.get('time', "20:00 EST")  # Usa orario dal game se disponibile
        central_line = args.line if args and hasattr(args, 'line') and args.line else "N/A"
        
        # Header con titolo centrato
        print(f"\n╔══════════════════════════════════════════════════════════════════════════════╗")
        print(f"║                          {category_title:<38}                          ║")
        print(f"╠══════════════════════════════════════════════════════════════════════════════╣")
        
        # SEZIONE 1: INFORMAZIONI PARTITA
        print(f"║  📊 INFORMAZIONI PARTITA                                                    ║")
        print(f"║                                                                              ║")
        print(f"║    🏀 Squadre:         {away_team:<35} @ {home_team:<15} ║")
        print(f"║    📅 Data partita:    {game_date:<50} ║")
        print(f"║    ⏰ Orario:          {game_time:<50} ║")
        print(f"║    📊 Linea bookmaker: {central_line:<50} ║")
        print(f"╠══════════════════════════════════════════════════════════════════════════════╣")
        
        # SEZIONE 2: PREDIZIONI SISTEMA
        print(f"║  🔮 PREDIZIONI SISTEMA NBA PREDICTOR                                        ║")
        print(f"║                                                                              ║")
        print(f"║    🎯 Totale previsto:    {predicted_total:.1f} punti                                        ║")
        print(f"║    📈 Confidenza (σ):     ±{confidence_sigma:.1f} punti                                       ║")
        # Usa il numero effettivo di simulazioni dal distribution o default
        mc_simulations = distribution.get('mc_simulations', 25000)
        favorable_sims = distribution.get('favorable_simulations')
        favorable_pct = distribution.get('favorable_percentage')
        analyzed_line = distribution.get('central_line_analyzed')
        
        print(f"║    🎲 Simulazioni MC:     {mc_simulations:,} iterazioni                                ║")
        
        # Mostra simulazioni favorevoli se disponibili
        if favorable_sims is not None and analyzed_line is not None:
            print(f"║    📊 Simulazioni favor.: {favorable_sims:,} OVER {analyzed_line} ({favorable_pct:.1f}%)                           ║")
        # Gestione corretta dei valori di impatto
        injury_value = injury_impact if isinstance(injury_impact, (int, float)) else 0.0
        momentum_value = momentum_impact if isinstance(momentum_impact, (int, float)) else momentum_impact.get('total_impact', 0.0) if isinstance(momentum_impact, dict) else 0.0
        
        print(f"║    🏥 Impatto infortuni:  {injury_value:+.2f} punti                                      ║")
        print(f"║    ⚡ Impatto momentum:   {momentum_value:+.2f} punti                                      ║")
        print(f"╠══════════════════════════════════════════════════════════════════════════════╣")
        
        # SEZIONE 3: ANALISI SCOMMESSA
        print(f"║  🎰 ANALISI SCOMMESSA CONSIGLIATA                                           ║")
        print(f"║                                                                              ║")
        print(f"║    🎯 Tipo:              {bet_type_full} {bet_line}                                            ║")
        print(f"║    💰 Quota:             {bet_odds:.2f}                                                 ║")
        print(f"║    🎲 Probabilità:       {opt_prob:.1f}%                                               ║")
        print(f"║    ⚡ Edge:              {opt_edge:+.1f}%                                               ║")
        print(f"║    🌟 Quality Score:     {opt_quality:.1f}/100                                          ║")
        print(f"╠══════════════════════════════════════════════════════════════════════════════╣")
        
        # SEZIONE 4: GESTIONE BANKROLL
        print(f"║  💼 GESTIONE BANKROLL E RISULTATI ATTESI                                    ║")
        print(f"║                                                                              ║")
        print(f"║    💵 Stake consigliato: €{bet_stake:.2f}                                            ║")
        print(f"║    💰 Potenziale vincita: €{potential_win:.2f}                                            ║")
        print(f"║    📈 ROI atteso:        {roi_percent:.1f}%                                            ║")
        print(f"║    🔄 Stake × Odds:      €{bet_stake:.2f} × {bet_odds:.2f} = €{bet_stake * bet_odds:.2f}                           ║")
        print(f"╚══════════════════════════════════════════════════════════════════════════════╝")
        
        # Riepilogo compatto sotto la tabella
        risk_level = "🟢 BASSO" if opt_prob >= 70 else "🟡 MEDIO" if opt_prob >= 60 else "🔴 ALTO"
        confidence_level = "🔥 ALTA" if opt_quality >= 80 else "⚡ MEDIA" if opt_quality >= 60 else "⚪ BASSA"
        
        print(f"\n🎯 RIEPILOGO: {bet_type_full} {bet_line} @ {bet_odds:.2f} • Prob: {opt_prob:.1f}% • Edge: {opt_edge:+.1f}% • Stake: €{bet_stake:.2f}")
        print(f"📊 Livello rischio: {risk_level} • Confidenza: {confidence_level} • Vincita potenziale: €{potential_win:.2f}")

    def _calculate_optimal_bet(self, opportunities):
        """
        Calcola la scommessa ottimale usando un algoritmo razionale.
        Ora considera solo scommesse con probabilità >= 50%.
        """
        try:
            # SOGLIA EDGE PIÙ ALTA: Minimo 1% per essere considerato VALUE
            value_bets = [opp for opp in opportunities if opp.get('edge', 0) > 0.01 and opp.get('probability', 0) >= 0.50]
            if not value_bets:
                return None
            
            scored_bets = []
            
            for bet in value_bets:
                edge = bet.get('edge', 0)
                probability = bet.get('probability', 0)
                odds = bet.get('odds', 1.0)
                
                if edge <= 0 or probability <= 0 or odds <= 1.0:
                    continue
                
                # CORREZIONE DRASTICA: Edge deve essere significativo per ottenere punti alti
                if edge >= 0.15:  # 15%+
                    edge_score = 30
                elif edge >= 0.10:  # 10-15%
                    edge_score = 25 + (edge - 0.10) * 100  # Scala 25-30
                elif edge >= 0.05:  # 5-10%
                    edge_score = 15 + (edge - 0.05) * 200  # Scala 15-25
                elif edge >= 0.02:  # 2-5%
                    edge_score = 5 + (edge - 0.02) * 333   # Scala 5-15
                else:  # <2%
                    edge_score = edge * 250  # Max 5 punti per edge molto bassi
                
                # SISTEMA AMPLIFICATO - Ogni % conta nella fascia critica 50-55%
                if probability > 0.65:
                    prob_score = 35              # Bonus per probabilità molto alte
                elif 0.60 <= probability <= 0.65:
                    prob_score = 25 + (probability - 0.60) * 200  # Scala 25-35
                elif 0.55 <= probability < 0.60:
                    prob_score = 15 + (probability - 0.55) * 200  # Scala 15-25
                elif 0.52 <= probability < 0.55:
                    prob_score = 5 + (probability - 0.52) * 333   # Scala 5-15 (AMPLIFICATA)
                else:  # 50-52% - FASCIA CRITICA
                    prob_score = (probability - 0.50) * 250       # 0-5 punti (MOLTO RIPIDA) 
                
                # SISTEMA QUOTE POTENZIATO - Range ottimale privilegiato (20% peso)
                if 1.70 <= odds <= 1.95:
                    odds_score = 30              # POTENZIATO: Range ottimale massimo premio
                elif 1.60 <= odds < 1.70:
                    odds_score = 18              # Buono ma margine basso
                elif 1.95 < odds <= 2.10:
                    odds_score = 20              # Ancora accettabile
                elif 2.10 < odds <= 2.30:
                    odds_score = 12              # Rischio moderato
                elif 2.30 < odds <= 2.60:
                    odds_score = 8               # Rischio alto
                else:
                    odds_score = max(3, 15 - abs(odds - 1.8) * 8)  # Penalizzazione severa
                
                # KELLY ELIMINATO - È un derivato dell'edge che aumenta surrettiziamente il suo peso
                
                # SISTEMA PULITO - Solo 3 componenti indipendenti
                total_score = (
                    edge_score * 0.30 +      # Edge 
                    prob_score * 0.50 +      # Probabilità dominante
                    odds_score * 0.20        # Quote potenziate (+5% dal Kelly eliminato)
                )

                bet_copy = bet.copy()
                # Normalizzazione corretta: max possibile = 35+30+25+10 = 100
                normalized_score = total_score  # Già su scala 0-100
                
                bet_copy.update({
                    'optimization_score': normalized_score,
                    'edge_score': edge_score,
                    'prob_score': prob_score,
                    'odds_score': odds_score,
                    'total_raw_score': total_score
                })
                scored_bets.append(bet_copy)

            if not scored_bets:
                return None



            best_bet = max(scored_bets, key=lambda x: x['optimization_score'])
            self.last_scored_bets = sorted(scored_bets, key=lambda x: x['optimization_score'], reverse=True)
            return best_bet
            
        except Exception as e:
            print(f"⚠️ Errore nel calcolo scommessa ottimale: {e}")
            self.last_scored_bets = []
            return None

    def get_game_result_automatically(self, game_id):
        """Recupera automaticamente il risultato di una partita tramite API NBA."""
        try:
            from nba_api.stats.endpoints import boxscoresummaryv2
            import time
            
            print(f"🔍 Recupero risultato automatico per game_id: {game_id}")
            time.sleep(0.6)
            boxscore = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id, headers={'User-Agent': 'Mozilla/5.0'})
            game_summary = boxscore.game_summary.get_data_frame()
            if game_summary.empty:
                print("   ❌ Nessun dato trovato per questa partita")
                return None
            
            summary = game_summary.iloc[0]
            game_status = summary.get('GAME_STATUS_ID', 1)
            
            if game_status != 3:
                print(f"   ⏳ Partita non ancora completata (status: {game_status})")
                return None
            
            line_score = boxscore.line_score.get_data_frame()
            if line_score.empty:
                print("   ❌ Punteggi non disponibili")
                return None
            
            scores = {team['TEAM_ID']: int(team['PTS']) for _, team in line_score.iterrows()}
            if len(scores) != 2:
                print("   ❌ Errore nel recupero dei punteggi delle squadre")
                return None
            
            score_values = list(scores.values())
            total_score = sum(score_values)
            
            result = {
                'total_score': total_score,
                'home_score': score_values[0],
                'away_score': score_values[1],
                'status': 'COMPLETED'
            }
            print(f"   ✅ Risultato: {result['away_score']} - {result['home_score']} (Totale: {total_score})")
            return result
            
        except Exception as e:
            print(f"   ⚠️ Errore nel recupero automatico del risultato: {e}")
            return None

    def save_pending_bet(self, bet_data, game_id):
        """Salva una scommessa in attesa di risultato."""
        try:
            pending_file = 'data/pending_bets.json'
            os.makedirs('data', exist_ok=True)
            
            # Converte i dati in tipi JSON serializzabili
            clean_bet_data = {}
            for key, value in bet_data.items():
                if isinstance(value, (int, float, str)):
                    clean_bet_data[key] = value
                elif hasattr(value, 'item'):  # NumPy scalars
                    clean_bet_data[key] = value.item()
                elif isinstance(value, bool):
                    clean_bet_data[key] = bool(value)
                else:
                    clean_bet_data[key] = float(value) if value is not None else 0.0
            
            try:
                with open(pending_file, 'r') as f:
                    pending_bets = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                pending_bets = []
            
            # Controlla se esiste già una scommessa per questo game_id
            existing_bet_index = None
            for i, bet in enumerate(pending_bets):
                if bet.get('game_id') == game_id and bet.get('status') == 'pending':
                    existing_bet_index = i
                    break
            
            if existing_bet_index is not None:
                old_bet_data = pending_bets[existing_bet_index]['bet_data']
                print(f"\n⚠️  SCOMMESSA ESISTENTE TROVATA per {game_id}:")
                print(f"   ATTUALE: {old_bet_data.get('type', 'N/A')} {old_bet_data.get('line', 'N/A')} @ {old_bet_data.get('odds', 'N/A')} (€{old_bet_data.get('stake', 0):.2f})")
                print(f"   NUOVA:   {clean_bet_data.get('type', 'N/A')} {clean_bet_data.get('line', 'N/A')} @ {clean_bet_data.get('odds', 'N/A')} (€{clean_bet_data.get('stake', 0):.2f})")
                
                choice = input("\nCosa vuoi fare?\n1. Sostituisci scommessa esistente\n2. Aggiungi come nuova scommessa\n3. Annulla\nScelta (1/2/3): ").strip()
                
                if choice == '1':
                    # Sostituisci la scommessa esistente
                    pending_bets[existing_bet_index] = {
                        'bet_id': f"{game_id}_{clean_bet_data['type']}_{clean_bet_data['line']}",
                        'game_id': game_id,
                        'bet_data': clean_bet_data,
                        'timestamp': datetime.now().isoformat(),
                        'status': 'pending',
                        'replaced_at': datetime.now().isoformat(),
                        'original_bet': old_bet_data
                    }
                    print(f"🔄 Scommessa sostituita: {clean_bet_data['type']} {clean_bet_data['line']}")
                elif choice == '2':
                    # Aggiungi come nuova scommessa
                    pending_bet = {
                        'bet_id': f"{game_id}_{clean_bet_data['type']}_{clean_bet_data['line']}_{len(pending_bets)}",
                        'game_id': game_id,
                        'bet_data': clean_bet_data,
                        'timestamp': datetime.now().isoformat(),
                        'status': 'pending'
                    }
                    pending_bets.append(pending_bet)
                    print(f"➕ Nuova scommessa aggiunta: {clean_bet_data['type']} {clean_bet_data['line']}")
                else:
                    print("❌ Salvataggio annullato")
                    return
            else:
                # Nessuna scommessa esistente, aggiungi normalmente
                pending_bet = {
                    'bet_id': f"{game_id}_{clean_bet_data['type']}_{clean_bet_data['line']}",
                    'game_id': game_id,
                    'bet_data': clean_bet_data,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'pending'
                }
                pending_bets.append(pending_bet)
                print(f"💾 Scommessa salvata in attesa di risultato: {clean_bet_data['type']} {clean_bet_data['line']}")
            
            with open(pending_file, 'w') as f:
                json.dump(pending_bets, f, indent=2)
            
        except Exception as e:
            print(f"⚠️ Errore nel salvataggio scommessa pendente: {e}")

    def check_and_update_pending_bets(self):
        """Controlla tutte le scommesse pendenti e aggiorna il bankroll."""
        try:
            pending_file = 'data/pending_bets.json'
            if not os.path.exists(pending_file):
                print("📝 Nessuna scommessa pendente trovata")
                return
            
            with open(pending_file, 'r') as f:
                pending_bets = json.load(f)
            
            if not pending_bets:
                print("📝 Nessuna scommessa pendente")
                return
            
            print(f"🔄 Controllo {len(pending_bets)} scommesse pendenti...")
            
            updated_bets = []
            bankroll_updates = 0
            
            for bet in pending_bets:
                if bet['status'] != 'pending':
                    updated_bets.append(bet)
                    continue
                
                game_id = bet['game_id']
                bet_data = bet['bet_data']
                
                print(f"\n🎯 Controllo partita {game_id}...")
                result = self.get_game_result_automatically(game_id)
                
                if result and result['status'] == 'COMPLETED':
                    update_result = self.update_bankroll_from_bet(bet_data, result['total_score'])
                    if update_result:
                        bet['status'] = 'completed'
                        bet['result'] = {
                            'actual_total': result['total_score'],
                            'bet_won': update_result['bet_won'],
                            'profit_loss': update_result['profit_loss'],
                            'completed_at': datetime.now().isoformat()
                        }
                        bankroll_updates += 1
                        print(f"   ✅ Scommessa aggiornata automaticamente!")
                    else:
                        print(f"   ⚠️ Errore nell'aggiornamento del bankroll")
                else:
                    print(f"   ⏳ Partita non ancora completata o risultato non disponibile")
                
                updated_bets.append(bet)
                time.sleep(1)
            
            with open(pending_file, 'w') as f:
                json.dump(updated_bets, f, indent=2)
            
            if bankroll_updates > 0:
                print(f"\n🎉 {bankroll_updates} scommesse aggiornate automaticamente!")
                print(f"💰 Bankroll attuale: €{self.bankroll:.2f}")
            else:
                print("\n📋 Nessuna scommessa da aggiornare al momento")
                
        except Exception as e:
            print(f"⚠️ Errore nel controllo scommesse pendenti: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='🏀 NBA Predictor - Sistema Completo di Analisi e Predizione',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ESEMPI D'USO:
    python main.py                                           # Analizza partite dal calendario NBA
    python main.py --team1 "Thunder" --team2 "Pacers"       # Analisi partita personalizzata
    python main.py --team1 "Lakers" --team2 "Warriors" --line 225.0  # Con linea specifica
    python main.py --check-bets                              # Controlla risultati scommesse pendenti
    python main.py --auto-mode --team1 "Celtics" --team2 "Heat"      # Modalità automatica

FLUSSO TIPICO:
    1. python main.py --team1 "Thunder" --team2 "Pacers"    # Fai analisi e salva scommessa
    2. python bet_manager.py                                # Visualizza scommesse salvate
    3. python main.py --check-bets                          # Controlla risultati dopo le partite

SISTEMI ATTIVI:
    🟢 Sistema Injury Reporting    - Analisi infortuni multi-fonte
    🟢 Sistema Momentum Avanzato   - ML-based con confidence scoring
    🟢 Modello Probabilistico      - Predizioni accurate con Monte Carlo
    🟢 Betting Analysis            - Raccomandazioni categorizzate VALUE
    🟢 Bankroll Management        - Gestione automatica stake e profitti
        """
    )
    
    parser.add_argument('--team1', 
                       type=str, 
                       metavar='SQUADRA',
                       help='🏀 Prima squadra per analisi personalizzata (es: "Thunder", "Lakers")')
    
    parser.add_argument('--team2', 
                       type=str, 
                       metavar='SQUADRA',
                       help='🏀 Seconda squadra per analisi personalizzata (es: "Pacers", "Warriors")')
    
    parser.add_argument('--line', 
                       type=float, 
                       metavar='PUNTI',
                       help='📊 Linea centrale per generazione quote (es: 225.5, 210.0)')
    
    parser.add_argument('--auto-mode', 
                       action='store_true', 
                       help='🤖 Modalità automatica senza interazione utente')
    
    parser.add_argument('--check-bets', 
                       action='store_true', 
                       help='🔄 Controlla e aggiorna automaticamente le scommesse pendenti')
    
    # Parametri deprecati
    parser.add_argument('--giorni', 
                       type=int, 
                       help='⚠️  DEPRECATO: usa --team1/--team2 per partite specifiche')
    
    parser.add_argument('--linea-centrale', 
                       type=float, 
                       help='⚠️  DEPRECATO: usa --line invece')
    
    args = parser.parse_args()

    print("🚀 Avvio Sistema NBA Predictor Completo")
    print("="*50)
    
    try:
        data_provider = NBADataProvider()
        system = NBACompleteSystem(data_provider, auto_mode=args.auto_mode)
        
        if args.check_bets:
            print("🔄 Controllo scommesse pendenti...")
            system.check_and_update_pending_bets()
            return

        if args.giorni:
            print("⚠️ AVVISO: --giorni è deprecato. Il nuovo sistema analizza partite specifiche.")
            print("   Usa: python main.py --team1 Lakers --team2 Warriors")
            return

        if hasattr(args, 'linea_centrale') and args.linea_centrale:
            args.line = args.linea_centrale
            print("⚠️ AVVISO: --linea-centrale è deprecato. Usa --line invece.")
        
        game = None
        if args.team1 and args.team2:
            # Ottieni gli ID delle squadre usando il metodo corretto
            away_team_info = data_provider._find_team_by_name(args.team1)
            home_team_info = data_provider._find_team_by_name(args.team2)
            
            if not away_team_info or not home_team_info:
                print(f"❌ Squadra non trovata: {args.team1 if not away_team_info else args.team2}")
                return
            
            game = {
                'away_team': args.team1,
                'home_team': args.team2,
                'away_team_id': away_team_info['id'],
                'home_team_id': home_team_info['id'],
                'game_id': f"CUSTOM_{args.team1}_{args.team2}",
                'odds': []
            }
            if not args.line and not game.get('odds'):
                args.line = 225.0
                print(f"ℹ️ Nessuna quota o linea centrale fornita. Usando linea di default: {args.line}")
            print(f"🎯 Analisi partita personalizzata: {args.team1} @ {args.team2}")
        else:
            print("📅 Recupero partite dal calendario NBA...")
            try:
                scheduled_games = data_provider.get_scheduled_games(days_ahead=3)
                if scheduled_games:
                    print(f"🏀 Trovate {len(scheduled_games)} partite nei prossimi 3 giorni:")
                    for i, g in enumerate(scheduled_games[:5], 1):
                        print(f"   {i}. {g['away_team']} @ {g['home_team']} ({g['date']})")
                    
                    if not args.auto_mode:
                        choice = input(f"\nScegli una partita (1-{min(len(scheduled_games), 5)}) o premi Enter per la prima: ")
                        game_index = int(choice) - 1 if choice.strip() and choice.isdigit() and 0 <= int(choice) - 1 < len(scheduled_games) else 0
                        game = scheduled_games[game_index]
                    else:
                        game = scheduled_games[0]
                else:
                    print("⚠️ Nessuna partita trovata, uso partita di esempio")
                    game = {
                        'away_team': 'Lakers', 'home_team': 'Warriors', 
                        'away_team_id': 1610612747, 'home_team_id': 1610612744,
                        'game_id': 'EXAMPLE_GAME', 'odds': []
                    }
            except Exception as e:
                print(f"⚠️ Errore nel recupero partite: {e}")
                game = {
                    'away_team': 'Lakers', 'home_team': 'Warriors', 
                    'away_team_id': 1610612747, 'home_team_id': 1610612744,
                    'game_id': 'EXAMPLE_GAME', 'odds': []
                }
        
        if not game:
            print("⚠️ Nessuna partita disponibile, uso partita di esempio")
            game = {
                'away_team': 'Lakers', 'home_team': 'Warriors', 
                'away_team_id': 1610612747, 'home_team_id': 1610612744,
                'game_id': 'EXAMPLE_GAME', 'odds': []
            }
        
        if not args.line and not game.get('odds'):
            args.line = 225.0
            print(f"ℹ️ Nessuna quota o linea centrale fornita. Usando linea di default: {args.line}")

        system.analyze_game(game, central_line=args.line, args=args)
        
        print("✅ Analisi completata con successo!")
        
    except Exception as e:
        print(f"❌ Errore critico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()