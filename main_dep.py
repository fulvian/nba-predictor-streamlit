# main.py - v7.2 Finale e Corretta
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
from advanced_player_momentum_predictor import AdvancedPlayerMomentumPredictor
from probabilistic_model import ProbabilisticModel

class NBACompleteSystem:
    def __init__(self, data_provider):
        print("🚀 Inizializzazione NBACompleteSystem...")
        self.data_provider = data_provider
        self.injury_reporter = InjuryReporter(self.data_provider)
        self.impact_analyzer = PlayerImpactAnalyzer(self.data_provider)
        self.momentum_predictor = AdvancedPlayerMomentumPredictor(nba_data_provider=self.data_provider)
        self.probabilistic_model = ProbabilisticModel()
        self.bankroll = self._load_bankroll() # Funzione ripristinata

    def _load_bankroll(self, default=100.0):
        """Carica il bankroll dal file JSON, o usa valore di default."""
        try:
            with open('bankroll.json', 'r') as f:
                data = json.load(f)
                return float(data.get('current_bankroll', default))
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"ℹ️ File bankroll.json non trovato. Usando valore default: €{default}")
            return default

    def analyze_game(self, game_details, league_injuries_map, central_line=None):
        """
        Flusso completo di analisi per una singola partita, utilizzando la mappa degli infortuni pre-caricata.
        """
        home_team_id = game_details['home_team_id']
        away_team_id = game_details['away_team_id']
        home_team_name = game_details['home_team']
        away_team_name = game_details['away_team']
        
        print("\n" + "="*80)
        print(f"🏀 Analisi Partita: {away_team_name} @ {home_team_name} ({game_details['date']})")
        print("="*80)

        # 1. Statistiche Squadra
        print("1. Recupero statistiche di squadra...")
        team_stats = self.data_provider.get_team_stats_for_game(home_team_name, away_team_name)
        if not team_stats: return

        # 2. Roster + Infortuni
        print("2. Recupero roster e integrazione dati infortuni...")
        home_roster_list = self.injury_reporter.get_team_roster_with_injuries(home_team_id, league_injuries_map)
        away_roster_list = self.injury_reporter.get_team_roster_with_injuries(away_team_id, league_injuries_map)
        
        home_roster_df = pd.DataFrame(home_roster_list) if home_roster_list else pd.DataFrame()
        away_roster_df = pd.DataFrame(away_roster_list) if away_roster_list else pd.DataFrame()
        
        if home_roster_df.empty or away_roster_df.empty: return

        # 3. Calcolo Impatto Infortuni
        print("3. Calcolo impatto infortuni...")
        injury_impact = 0.0
        try:
            home_impact = self.impact_analyzer.calculate_team_impact(home_roster_df)
            away_impact = self.impact_analyzer.calculate_team_impact(away_roster_df)
            injury_impact = (home_impact.get('total_impact', 0.0) + away_impact.get('total_impact', 0.0))
            print(f"   -> Impatto Infortuni: Casa={home_impact.get('total_impact', 0.0):+.2f}, Ospite={away_impact.get('total_impact', 0.0):+.2f} | Aggiustamento Totale: {injury_impact:+.2f} punti")
        except Exception as e:
            print(f"⚠️ Errore calcolo impatto infortuni: {e}. Impatto considerato nullo.")
            injury_impact = 0.0

        # 4. Calcolo Momentum con logica aggregata semplice e robusta
        print("4. Calcolo impatto momentum...")
        home_momentum = self.momentum_predictor.predict_team_momentum_impact_advanced(home_roster_df)
        away_momentum = self.momentum_predictor.predict_team_momentum_impact_advanced(away_roster_df)
        
        momentum_details = {
            'total_impact': home_momentum.get('impact_on_totals', 0.0) + away_momentum.get('impact_on_totals', 0.0),
            'advanced_system': True,
            'home_momentum': home_momentum,
            'away_momentum': away_momentum,
            'synergy_detected': home_momentum.get('momentum_score', 50) > 55 and away_momentum.get('momentum_score', 50) > 55,
            'confidence_factor': 1.0 
        }

        # 5. Modello Probabilistico
        print("5. Esecuzione modello probabilistico...")
        distribution = self.probabilistic_model.predict_distribution(team_stats, injury_impact, momentum_details)
        if not distribution: return
            
        # 6. Analisi Scommesse e Riepilogo
        print("6. Analisi opportunità di scommessa...")
        opportunities = self.probabilistic_model.analyze_betting_opportunities(
            distribution, odds_list=game_details.get('odds', []),
            central_line=central_line, bankroll=self.bankroll
        )
        self.display_final_summary(game_details, distribution, opportunities, momentum_details)

    def display_final_summary(self, game_details, distribution, opportunities, momentum_details=None):
        # Questo metodo non richiede modifiche
        print("\n" + "—"*30 + " RIEPILOGO FINALE " + "—"*30)
        print(f"Partita: {game_details['away_team']} @ {game_details['home_team']} | Data: {game_details['date']}")
        print(f"Predizione Base (μ): {distribution['base_mu']:.2f} ± {distribution['base_sigma']:.2f}")
        if momentum_details and momentum_details.get('advanced_system'):
            print(f"\n🔬 ANALISI MOMENTUM AVANZATA:")
            print(f"Impatto Momentum Stimato: {momentum_details.get('total_impact', 0):+.2f} punti")
            home_mom = momentum_details.get('home_momentum', {})
            away_mom = momentum_details.get('away_momentum', {})
            print(f"Casa - Score: {home_mom.get('momentum_score', 50):.1f}, Hot Hands: {home_mom.get('hot_hand_players_count', 0)}")
            print(f"Ospite - Score: {away_mom.get('momentum_score', 50):.1f}, Hot Hands: {away_mom.get('hot_hand_players_count', 0)}")
            if momentum_details.get('synergy_detected'): print("🔥 Effetto sinergico rilevato - Aspettarsi punteggio elevato")
            
            confidence_percentage = momentum_details.get('confidence_factor', 1.0) * 100
            if confidence_percentage > 80: confidence_label = "Alta"
            elif confidence_percentage >= 50: confidence_label = "Media"
            else: confidence_label = "Bassa"
            print(f"Confidenza: {confidence_percentage:.1f}% ({confidence_label})")
            
            print(f"Punti Totali Predetti finale (μ): {distribution['predicted_mu']:.2f} ± {distribution['predicted_sigma']:.2f}")

        print("\n📊 ANALISI DELLE QUOTE:")
        print("-" * 78)
        print(f"{'':<2} {'Tipo':<8} {'Linea':<8} {'Quota':<8} {'Prob.':<12} {'Edge':<12} {'Stake (€)':<10}")
        print("-" * 78)
        if not opportunities:
            print("Nessuna quota da analizzare.")
        else:
            value_bets_found = 0
            for opp in opportunities:
                value_marker = "✅" if opp.get('is_value') else "  "
                prob_str = f"{opp.get('probability', 0)*100:.2f}%"
                edge_str = f"{opp.get('edge', 0)*100:+.2f}%"
                print(f"{value_marker:<2} {opp.get('type', ''):<8} {opp.get('line', 0):<8.1f} {opp.get('odds', 0):<8.2f} {prob_str:<12} {edge_str:<12} {opp.get('stake', 0):<10.2f}")
                if opp.get('is_value'): value_bets_found += 1
            print("-" * 78)
            if value_bets_found > 0: print(f"✅ Trovate {value_bets_found} opportunità di scommessa valide.")
            else: print("🚫 Nessuna opportunità di scommessa valida trovata.")
        print("—"*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Sistema predizioni NBA v7.2 - Finale")
    parser.add_argument('--giorni', type=int, default=1, help='Giorni futuri da analizzare.')
    parser.add_argument('--linea-centrale', type=float, help='Linea di punteggio centrale (es. 228.5).')
    args = parser.parse_args()

    try:
        data_provider = NBADataProvider()
        nba_system = NBACompleteSystem(data_provider)

        print("--- Inizializzazione Sistema ---")
        league_injuries = nba_system.injury_reporter._fetch_injuries_from_rotowire()
        if not league_injuries:
            print("⚠️ ATTENZIONE: Impossibile recuperare i dati sugli infortuni. Le analisi potrebbero essere meno accurate.")
        print("--- Sistema Pronto ---")

        print(f"\nAnalisi delle partite per i prossimi {args.giorni} giorni...")
        scheduled_games = data_provider.get_scheduled_games(days_ahead=args.giorni)
        if not scheduled_games:
            print("Nessuna partita trovata.")
            return

        for i, game in enumerate(scheduled_games, 1):
            nba_system.analyze_game(game, league_injuries, central_line=args.linea_centrale)
            if i < len(scheduled_games): 
                time.sleep(2)
                
        print(f"\n🎉 Analisi completata per {len(scheduled_games)} partite!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Analisi interrotta dall'utente.")
    except Exception as e:
        print(f"\n❌ Errore critico durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()