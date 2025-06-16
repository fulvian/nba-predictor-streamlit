# main.py
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
from player_momentum_predictor import PlayerMomentumPredictor
from probabilistic_model import ProbabilisticModel

class NBACompleteSystem:
    def __init__(self, data_provider):
        print("🚀 Inizializzazione NBACompleteSystem...")
        self.data_provider, self.bankroll = data_provider, self._load_bankroll()
        self.impact_analyzer = PlayerImpactAnalyzer(self.data_provider)
        self.injury_reporter = InjuryReporter(self.data_provider)
        self.momentum_predictor = PlayerMomentumPredictor(nba_data_provider=self.data_provider)
        self.probabilistic_model = ProbabilisticModel()

    def _load_bankroll(self, default=100.0):
        try:
            with open('bankroll.json', 'r') as f: return float(json.load(f).get('current_bankroll', default))
        except (FileNotFoundError, json.JSONDecodeError): return default

    # --- METODO MODIFICATO ---
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

        # ... (Passo 1, 2, 3, 4 rimangono identici) ...
        print("1. Recupero statistiche di squadra...")
        team_stats = self.data_provider.get_team_stats_for_game(home_team_name, away_team_name)
        if not team_stats: print("❌ Statistiche squadra non disponibili. Analisi interrotta."); return
        print("2. Recupero roster e dati infortuni...")
        home_roster_list, away_roster_list = self.injury_reporter.get_team_roster(home_team_id), self.injury_reporter.get_team_roster(away_team_id)
        home_roster_df, away_roster_df = (pd.DataFrame(home_roster_list) if home_roster_list else pd.DataFrame()), (pd.DataFrame(away_roster_list) if away_roster_list else pd.DataFrame())
        if not home_roster_df.empty and 'id' in home_roster_df.columns: home_roster_df = home_roster_df.rename(columns={'id': 'PLAYER_ID'})
        if not away_roster_df.empty and 'id' in away_roster_df.columns: away_roster_df = away_roster_df.rename(columns={'id': 'PLAYER_ID'})
        if home_roster_df.empty or away_roster_df.empty: print("❌ Impossibile recuperare uno o entrambi i roster. Analisi interrotta."); return
        print("3. Calcolo impatto infortuni...")
        injury_impact = 0.0
        try:
            home_impact_result, away_impact_result = self.impact_analyzer.calculate_team_impact(home_roster_df), self.impact_analyzer.calculate_team_impact(away_roster_df)
            home_injury_pts, away_injury_pts = home_impact_result.get('total_impact', 0.0), away_impact_result.get('total_impact', 0.0)
            injury_impact = home_injury_pts - away_injury_pts
            print(f"   -> Impatto Infortuni: Casa={home_injury_pts:+.2f}, Ospite={away_injury_pts:+.2f} | Differenziale: {injury_impact:+.2f} punti")
        except Exception as e: print(f"⚠️ Errore nel calcolo dell'impatto infortuni: {e}. L'impatto verrà considerato nullo."); injury_impact = 0.0
        print("4. Calcolo impatto momentum...")
        home_momentum_result, away_momentum_result = self.momentum_predictor.predict_team_momentum_impact(home_roster_df), self.momentum_predictor.predict_team_momentum_impact(away_roster_df)
        home_momentum_score, away_momentum_score = home_momentum_result.get('momentum_score', 50.0), away_momentum_result.get('momentum_score', 50.0)
        momentum_impact = (home_momentum_score - away_momentum_score) * 0.10
        print(f"   -> Punteggio Momentum: Casa={home_momentum_score:.2f}, Ospite={away_momentum_score:.2f} | Impatto: {momentum_impact:+.2f} punti")

        print("5. Esecuzione modello probabilistico...")
        distribution = self.probabilistic_model.predict_distribution(team_stats, injury_impact, momentum_impact)
        if not distribution: print("❌ Predizione probabilistica fallita. Analisi interrotta."); return
        print(f"   -> Predizione Base: μ={distribution['base_mu']:.2f}, σ={distribution['base_sigma']:.2f} | Predizione Finale: μ={distribution['predicted_mu']:.2f}, σ={distribution['predicted_sigma']:.2f}")

        print("6. Analisi opportunità di scommessa...")
        # --- MODIFICATO: Passiamo sia la lista di quote che la linea centrale al metodo ---
        odds_list = game_details.get('odds', [])
        opportunities = self.probabilistic_model.analyze_betting_opportunities(
            distribution,
            odds_list=odds_list,
            central_line=central_line,
            bankroll=self.bankroll
        )
        
        self.display_final_summary(game_details, distribution, opportunities)
        
    def display_final_summary(self, game_details, distribution, opportunities):
        """Mostra un riepilogo formattato dell'analisi, stampando sempre la tabella delle quote."""
        print("\n" + "—"*30 + " RIEPILOGO FINALE " + "—"*30)
        print(f"Partita: {game_details['away_team']} @ {game_details['home_team']} | Data: {game_details['date']}")
        print(f"Punti Totali Predetti (μ): {distribution['predicted_mu']:.2f} ± {distribution['predicted_sigma']:.2f}")
        
        print("\n📊 ANALISI DELLE QUOTE:")
        print("-" * 78)
        print(f"{'':<2} {'Tipo':<8} {'Linea':<8} {'Quota':<8} {'Prob.':<12} {'Edge':<12} {'Stake (€)':<10}")
        print("-" * 78)

        if not opportunities:
            print("Nessuna quota da analizzare.")
        else:
            value_bets_found = 0
            for opp in opportunities:
                # Contrassegna solo le scommesse di valore
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
    parser = argparse.ArgumentParser(description="Sistema completo di predizioni NBA v5.4")
    parser.add_argument('--giorni', type=int, default=1, help='Numero di giorni futuri da analizzare.')
    # --- NUOVO ARGOMENTO ---
    parser.add_argument('--linea-centrale', type=float, help='Specifica una linea di punteggio centrale per generare le quote (es. 228.5).')
    
    args = parser.parse_args()

    data_provider = NBADataProvider()
    nba_system = NBACompleteSystem(data_provider)

    print(f"Analisi delle partite per i prossimi {args.giorni} giorni...")
    scheduled_games = data_provider.get_scheduled_games(days_ahead=args.giorni)

    if not scheduled_games:
        print("Nessuna partita trovata nel periodo specificato.")
        return

    # --- MODIFICATO: Passiamo il nuovo argomento alla funzione di analisi ---
    for game in scheduled_games:
        nba_system.analyze_game(game, central_line=args.linea_centrale)
        time.sleep(2)

if __name__ == "__main__":
    main()