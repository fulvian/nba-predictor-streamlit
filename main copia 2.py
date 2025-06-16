# main.py - VERSIONE COMPLETA CON SISTEMA MOMENTUM AVANZATO
import pandas as pd
import os
import json
import time
import argparse
from datetime import datetime

# --- Import dei moduli del sistema ---
from data_provider import NBADataProvider
from injury_reporter import InjuryReporter
from player_impact_analyzer import PlayerImpactAnalyzer

# MODIFICA: Usa il nuovo sistema momentum avanzato con fallback
try:
    from advanced_player_momentum_predictor import AdvancedPlayerMomentumPredictor
    ADVANCED_MOMENTUM_AVAILABLE = True
    print("✅ Sistema momentum avanzato caricato")
except ImportError as e:
    print(f"⚠️ Sistema momentum avanzato non disponibile: {e}")
    print("   Falling back al sistema base...")
    try:
        from player_momentum_predictor import PlayerMomentumPredictor
        ADVANCED_MOMENTUM_AVAILABLE = False
        print("📊 Sistema momentum base caricato")
    except ImportError:
        print("❌ Nessun sistema momentum disponibile!")
        ADVANCED_MOMENTUM_AVAILABLE = None

from probabilistic_model import ProbabilisticModel

class NBACompleteSystem:
    def __init__(self, data_provider):
        print("🚀 Inizializzazione NBACompleteSystem...")
        self.data_provider = data_provider
        self.bankroll = self._load_bankroll()
        self.impact_analyzer = PlayerImpactAnalyzer(self.data_provider)
        self.injury_reporter = InjuryReporter(self.data_provider)
        
        # Inizializza sistema momentum appropriato
        if ADVANCED_MOMENTUM_AVAILABLE is True:
            self.momentum_predictor = AdvancedPlayerMomentumPredictor(nba_data_provider=self.data_provider)
            self.use_advanced_momentum = True
            print("🔬 Sistema momentum avanzato attivato")
        elif ADVANCED_MOMENTUM_AVAILABLE is False:
            self.momentum_predictor = PlayerMomentumPredictor(nba_data_provider=self.data_provider)
            self.use_advanced_momentum = False
            print("📊 Sistema momentum base attivato")
        else:
            self.momentum_predictor = None
            self.use_advanced_momentum = False
            print("⚠️ Nessun sistema momentum disponibile - continuando senza momentum")
        
        self.probabilistic_model = ProbabilisticModel()

    def _load_bankroll(self, default=100.0):
        """Carica il bankroll dal file JSON, o usa valore di default."""
        try:
            with open('bankroll.json', 'r') as f: 
                data = json.load(f)
                return float(data.get('current_bankroll', default))
        except (FileNotFoundError, json.JSONDecodeError): 
            print(f"ℹ️ File bankroll.json non trovato. Usando valore default: €{default}")
            return default

    def analyze_game(self, game_details, central_line=None):
        """
        Flusso completo di analisi per una singola partita.
        Accetta una linea centrale opzionale per generare le quote.
        """
        home_team_id, away_team_id = game_details['home_team_id'], game_details['away_team_id']
        home_team_name, away_team_name = game_details['home_team'], game_details['away_team']
        
        print("\n" + "="*80)
        print(f"🏀 Analisi Partita: {away_team_name} @ {home_team_name} ({game_details['date']})")
        if central_line:
            print(f"   📌 Utilizzando linea centrale manuale: {central_line}")
        print("="*80)

        # Passo 1: Recupero statistiche di squadra
        print("1. Recupero statistiche di squadra...")
        team_stats = self.data_provider.get_team_stats_for_game(home_team_name, away_team_name)
        if not team_stats: 
            print("❌ Statistiche squadra non disponibili. Analisi interrotta.")
            return
            
        # Passo 2: Recupero roster e dati infortuni
        print("2. Recupero roster e dati infortuni...")
        home_roster_list = self.injury_reporter.get_team_roster(home_team_id)
        away_roster_list = self.injury_reporter.get_team_roster(away_team_id)
        
        home_roster_df = pd.DataFrame(home_roster_list) if home_roster_list else pd.DataFrame()
        away_roster_df = pd.DataFrame(away_roster_list) if away_roster_list else pd.DataFrame()
        
        # Rinomina colonne se necessario
        if not home_roster_df.empty and 'id' in home_roster_df.columns: 
            home_roster_df = home_roster_df.rename(columns={'id': 'PLAYER_ID'})
        if not away_roster_df.empty and 'id' in away_roster_df.columns: 
            away_roster_df = away_roster_df.rename(columns={'id': 'PLAYER_ID'})
            
        if home_roster_df.empty or away_roster_df.empty: 
            print("❌ Impossibile recuperare uno o entrambi i roster. Analisi interrotta.")
            return

        # Passo 3: Calcolo impatto infortuni
        print("3. Calcolo impatto infortuni...")
        injury_impact = 0.0
        try:
            home_impact_result = self.impact_analyzer.calculate_team_impact(home_roster_df)
            away_impact_result = self.impact_analyzer.calculate_team_impact(away_roster_df)
            home_injury_pts = home_impact_result.get('total_impact', 0.0)
            away_injury_pts = away_impact_result.get('total_impact', 0.0)
            injury_impact = home_injury_pts - away_injury_pts
            print(f"   -> Impatto Infortuni: Casa={home_injury_pts:+.2f}, Ospite={away_injury_pts:+.2f} | Differenziale: {injury_impact:+.2f} punti")
        except Exception as e: 
            print(f"⚠️ Errore nel calcolo dell'impatto infortuni: {e}. L'impatto verrà considerato nullo.")
            injury_impact = 0.0

        # Passo 4: Calcolo momentum (avanzato o base)
        print("4. Calcolo impatto momentum...")
        if self.momentum_predictor is None:
            print("   -> Nessun sistema momentum disponibile. Impatto momentum = 0")
            momentum_impact = {'total_impact': 0.0, 'no_momentum_system': True}
            momentum_value = 0.0
        elif self.use_advanced_momentum:
            momentum_impact = self._calculate_advanced_momentum_impact(
                home_roster_df, away_roster_df, home_team_name, away_team_name, team_stats
            )
            momentum_value = momentum_impact['total_impact']
        else:
            # Sistema base
            home_momentum_result = self.momentum_predictor.predict_team_momentum_impact(home_roster_df)
            away_momentum_result = self.momentum_predictor.predict_team_momentum_impact(away_roster_df)
            home_momentum_score = home_momentum_result.get('momentum_score', 50.0)
            away_momentum_score = away_momentum_result.get('momentum_score', 50.0)
            momentum_value = (home_momentum_score - away_momentum_score) * 0.10
            momentum_impact = {'total_impact': momentum_value, 'basic_system': True}
            print(f"   -> Punteggio Momentum: Casa={home_momentum_score:.2f}, Ospite={away_momentum_score:.2f} | Impatto: {momentum_value:+.2f} punti")

        # Passo 5: Esecuzione modello probabilistico
        print("5. Esecuzione modello probabilistico...")
        distribution = self.probabilistic_model.predict_distribution(
            team_stats, injury_impact, momentum_value
        )
        if not distribution: 
            print("❌ Predizione probabilistica fallita. Analisi interrotta.")
            return
            
        print(f"   -> Predizione Base: μ={distribution['base_mu']:.2f}, σ={distribution['base_sigma']:.2f}")
        print(f"   -> Predizione Finale: μ={distribution['predicted_mu']:.2f}, σ={distribution['predicted_sigma']:.2f}")

        # Passo 6: Analisi opportunità di scommessa
        print("6. Analisi opportunità di scommessa...")
        odds_list = game_details.get('odds', [])
        opportunities = self.probabilistic_model.analyze_betting_opportunities(
            distribution,
            odds_list=odds_list,
            central_line=central_line,
            bankroll=self.bankroll
        )
        
        # Mostra riepilogo finale
        self.display_final_summary(game_details, distribution, opportunities, momentum_details=momentum_impact)

    def _calculate_advanced_momentum_impact(self, home_roster_df, away_roster_df, home_team_name, away_team_name, team_stats):
        """
        NUOVO METODO: Calcola impatto momentum usando sistema avanzato basato su ricerca scientifica.
        
        Returns:
            dict: Impatto momentum dettagliato con componenti separate
        """
        def safe_get(data, key, default=None):
            """Helper per gestire l'accesso a dizionari o oggetti pandas"""
            if hasattr(data, 'get'):
                return data.get(key, default)
            elif hasattr(data, 'to_dict'):
                data_dict = data.to_dict()
                return data_dict.get(key, default)
            return default
            
        try:
            # Verifica se il metodo avanzato è disponibile
            if not hasattr(self.momentum_predictor, 'predict_team_momentum_impact_advanced'):
                raise AttributeError("Metodo avanzato non disponibile nel predictor")
                
            # Calcola momentum per entrambe le squadre con nuovo sistema
            print("   🔬 Applicando metodologie scientifiche validate...")
            
            try:
                home_momentum_result = self.momentum_predictor.predict_team_momentum_impact_advanced(home_roster_df)
                away_momentum_result = self.momentum_predictor.predict_team_momentum_impact_advanced(away_roster_df)
                
                # Converti in dizionario se è un oggetto pandas
                if hasattr(home_momentum_result, 'to_dict'):
                    home_momentum_result = home_momentum_result.to_dict()
                if hasattr(away_momentum_result, 'to_dict'):
                    away_momentum_result = away_momentum_result.to_dict()
                    
                # Estrai componenti dettagliate con valori di default sicuri
                home_score = safe_get(home_momentum_result, 'momentum_score', 50.0)
                away_score = safe_get(away_momentum_result, 'momentum_score', 50.0)
                home_totals_impact = safe_get(home_momentum_result, 'impact_on_totals', 0.0)
                away_totals_impact = safe_get(away_momentum_result, 'impact_on_totals', 0.0)
                home_hot_hands = safe_get(home_momentum_result, 'hot_hand_players_count', 0)
                away_hot_hands = safe_get(away_momentum_result, 'hot_hand_players_count', 0)
                
                # Calcola impatto differenziale scientificamente validato
                differential_impact = home_totals_impact - away_totals_impact
                
                # Effetto moltiplicativo quando entrambe le squadre hanno momentum positivo
                synergy_detected = home_score > 55 and away_score > 55
                if synergy_detected:
                    synergy_bonus = 1.5  # Partite ad alto punteggio quando entrambe sono "hot"
                    differential_impact += synergy_bonus
                    print(f"   🔥 Rilevato momentum positivo bilaterale - Bonus sinergico: +{synergy_bonus:.1f} punti")
                
                try:
                    # Fattore di confidenza basato su sample size e consistency
                    confidence_factor = self._calculate_momentum_confidence(home_momentum_result, away_momentum_result)
                except Exception as e:
                    print(f"   ⚠️ Errore nel calcolo confidenza: {e}. Usando confidenza predefinita (0.7)")
                    confidence_factor = 0.7
                
                # Applica fattore di confidenza all'impatto
                final_impact = differential_impact * confidence_factor
                
                # Log dettagliato
                print(f"   -> Momentum Casa: {home_score:.1f} (Hot hands: {home_hot_hands}) | Impatto: {home_totals_impact:+.2f}")
                print(f"   -> Momentum Ospite: {away_score:.1f} (Hot hands: {away_hot_hands}) | Impatto: {away_totals_impact:+.2f}")
                print(f"   -> Impatto Differenziale: {differential_impact:+.2f} | Confidenza: {confidence_factor:.2f}")
                print(f"   -> Impatto Finale Aggiustato: {final_impact:+.2f} punti")
                
                return {
                    'total_impact': final_impact,
                    'home_momentum': {
                        'score': home_score,
                        'impact': home_totals_impact,
                        'hot_hands': home_hot_hands,
                        'details': home_momentum_result
                    },
                    'away_momentum': {
                        'score': away_score, 
                        'impact': away_totals_impact,
                        'hot_hands': away_hot_hands,
                        'details': away_momentum_result
                    },
                    'differential_impact': differential_impact,
                    'confidence_factor': confidence_factor,
                    'synergy_detected': synergy_detected,
                    'advanced_system': True
                }
                
            except Exception as e:
                print(f"   ❌ Errore durante l'elaborazione dei risultati momentum: {e}")
                raise  # Rilancia per gestione errori esterna
                
        except Exception as e:
            print(f"⚠️ Errore nel calcolo momentum avanzato: {e}")
            # Fallback al sistema base se il nuovo sistema fallisce
            try:
                if hasattr(self.momentum_predictor, 'predict_team_momentum_impact'):
                    home_basic = self.momentum_predictor.predict_team_momentum_impact(home_roster_df)
                    away_basic = self.momentum_predictor.predict_team_momentum_impact(away_roster_df)
                    basic_impact = (home_basic.get('momentum_score', 50.0) - away_basic.get('momentum_score', 50.0)) * 0.10
                    print(f"   -> Usando sistema fallback: {basic_impact:+.2f} punti")
                    
                    return {
                        'total_impact': basic_impact,
                        'fallback_mode': True
                    }
                else:
                    raise Exception("Metodo predict_team_momentum_impact non disponibile")
            except:
                print("   -> Usando impatto momentum neutro")
                return {'total_impact': 0.0, 'fallback_mode': True}

    def _calculate_momentum_confidence(self, home_result, away_result):
        """
        Calcola fattore di confidenza per l'impatto momentum basato su:
        - Sample size dei dati
        - Consistency delle performance 
        - Strength dei segnali
        """
        # Fattori di confidenza per casa e ospite
        home_confidence = 1.0
        away_confidence = 1.0
        
        # Analizza home team
        home_contributions = home_result.get('player_contributions', [])
        if home_contributions:
            # Confidence basata su sample size medio dei giocatori chiave
            key_players = [p for p in home_contributions if p.get('rotation_weight', 0) > 0.6]
            if key_players:
                # Piena fiducia con 3+ key players, scalata linearmente
                home_confidence = min(1.0, len(key_players) / 3.0)
        
        # Stesso per away team  
        away_contributions = away_result.get('player_contributions', [])
        if away_contributions:
            key_players = [p for p in away_contributions if p.get('rotation_weight', 0) > 0.6]
            if key_players:
                away_confidence = min(1.0, len(key_players) / 3.0)
        
        # Confidence finale è la media, con minimo di 0.3 per evitare impatti troppo ridotti
        final_confidence = max(0.3, (home_confidence + away_confidence) / 2.0)
        
        return final_confidence

    # In main.py

    def display_final_summary(self, game_details, distribution, opportunities, momentum_details=None):
        """
        Mostra un riepilogo formattato dell'analisi, includendo dettagli momentum avanzati.
        MODIFICATO per includere indicatore di confidenza e predizione finale esplicita.
        """
        # --- INIZIO MODIFICA: Mostra la predizione base all'inizio del riepilogo ---
        print("\n" + "—"*30 + " RIEPILOGO FINALE " + "—"*30)
        print(f"Partita: {game_details['away_team']} @ {game_details['home_team']} | Data: {game_details['date']}")
        # La prima riga ora mostra la predizione PRIMA degli aggiustamenti
        print(f"Predizione Base (μ): {distribution['base_mu']:.2f} ± {distribution['base_sigma']:.2f}")

        # Sezione momentum dettagliata
        if momentum_details:
            if momentum_details.get('advanced_system', False):
                print(f"\n🔬 ANALISI MOMENTUM AVANZATA:")
                # L'impatto degli infortuni viene considerato nel modello, qui mostriamo solo momentum
                print(f"Impatto Momentum Stimato: {momentum_details['total_impact']:+.2f} punti")
                
                home_mom = momentum_details['home_momentum']
                away_mom = momentum_details['away_momentum']
                print(f"Casa - Score: {home_mom['score']:.1f}, Hot Hands: {home_mom['hot_hands']}")
                print(f"Ospite - Score: {away_mom['score']:.1f}, Hot Hands: {away_mom['hot_hands']}")
                
                if momentum_details.get('synergy_detected', False):
                    print("🔥 Effetto sinergico rilevato - Aspettarsi punteggio elevato")
                
                # --- INIZIO MODIFICA 1: Indicatore sintetico di confidenza ---
                confidence_percentage = momentum_details.get('confidence_factor', 0) * 100
                if confidence_percentage > 80:
                    confidence_label = "Alta"
                elif confidence_percentage >= 50:
                    confidence_label = "Media"
                else:
                    confidence_label = "Bassa"
                print(f"Confidenza: {confidence_percentage:.1f}% ({confidence_label})")
                
                # --- INIZIO MODIFICA 2: Riga esplicita per la predizione finale ---
                print(f"Punti Totali Predetti finale (μ): {distribution['predicted_mu']:.2f} ± {distribution['predicted_sigma']:.2f}")

            elif momentum_details.get('basic_system', False) or momentum_details.get('fallback_mode', False):
                source_label = "Sistema Base" if momentum_details.get('basic_system') else "Modalità Fallback"
                print(f"\n📊 ANALISI MOMENTUM ({source_label}):")
                print(f"Impatto Totale: {momentum_details['total_impact']:+.2f} punti")
                print(f"Punti Totali Predetti finale (μ): {distribution['predicted_mu']:.2f} ± {distribution['predicted_sigma']:.2f}")

            elif momentum_details.get('no_momentum_system', False):
                print(f"\n❌ Nessun sistema momentum disponibile")
                print(f"Punti Totali Predetti finale (μ): {distribution['predicted_mu']:.2f} ± {distribution['predicted_sigma']:.2f}")

        # Analisi delle quote (invariata)
        print("\n📊 ANALISI DELLE QUOTE:")
        print("-" * 78)
        print(f"{'':<2} {'Tipo':<8} {'Linea':<8} {'Quota':<8} {'Prob.':<12} {'Edge':<12} {'Stake (€)':<10}")
        print("-" * 78)

        if not opportunities:
            print("Nessuna quota da analizzare.")
        else:
            value_bets_found = 0
            for opp in opportunities:
                value_marker = "✅" if opp['is_value'] else "  "
                prob_str = f"{opp['probability']*100:.2f}%"
                edge_str = f"{opp['edge']*100:+.2f}%"
                
                print(f"{value_marker:<2} {opp['type']:<8} {opp['line']:<8.1f} {opp['odds']:<8.2f} {prob_str:<12} {edge_str:<12} {opp['stake']:<10.2f}")
                
                if opp['is_value']:
                    value_bets_found += 1
            
            print("-" * 78)
            if value_bets_found > 0:
                print(f"✅ Trovate {value_bets_found} opportunità di scommessa con edge > 5%.")
            else:
                print("🚫 Nessuna opportunità di scommessa valida trovata (Edge > 5%).")

        print("—"*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Sistema completo di predizioni NBA v5.4 - Momentum Avanzato")
    parser.add_argument('--giorni', type=int, default=1, help='Numero di giorni futuri da analizzare.')
    # NUOVO ARGOMENTO per linea centrale
    parser.add_argument('--linea-centrale', type=float, help='Specifica una linea di punteggio centrale per generare le quote (es. 228.5).')
    
    args = parser.parse_args()

    # Inizializza sistema
    try:
        data_provider = NBADataProvider()
        nba_system = NBACompleteSystem(data_provider)

        print(f"\nAnalisi delle partite per i prossimi {args.giorni} giorni...")
        scheduled_games = data_provider.get_scheduled_games(days_ahead=args.giorni)

        if not scheduled_games:
            print("Nessuna partita trovata nel periodo specificato.")
            return

        # Analizza ogni partita
        for i, game in enumerate(scheduled_games, 1):
            print(f"\n{'='*20} PARTITA {i}/{len(scheduled_games)} {'='*20}")
            nba_system.analyze_game(game, central_line=args.linea_centrale)
            
            # Pausa tra le analisi per evitare rate limiting
            if i < len(scheduled_games):
                time.sleep(2)
                
        print(f"\n🎉 Analisi completata per {len(scheduled_games)} partite!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Analisi interrotta dall'utente.")
    except Exception as e:
        print(f"\n❌ Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()