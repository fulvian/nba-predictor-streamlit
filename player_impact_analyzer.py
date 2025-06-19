# player_impact_analyzer.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("🔄 Modulo player_impact_analyzer caricato con successo!")

class PlayerImpactAnalyzer:
    """
    Analizza l'impatto di un giocatore sulla sua squadra usando metriche dettagliate.
    Versione Professionale 4.0 - Allineata alle best practice del betting market
    """
    
    def __init__(self, nba_data_provider=None):
        print("\n🔍 [DEBUG] Inizializzazione PlayerImpactAnalyzer...")
        self.nba_data_provider = nba_data_provider
        self.player_impact_cache = {}
        self.team_impact_cache = {}
        self.cache_expiry = 3600  # 1 ora

        # SISTEMA PROFESSIONALE: Position weights basati su studi di mercato
        self.position_weights = {
            'PG': 1.35,  # Point Guard - massimo impatto su pace e sistema di gioco
            'SG': 1.15,  # Shooting Guard - impatto scoring significativo  
            'SF': 1.20,  # Small Forward - versatilità offensiva/difensiva
            'PF': 1.10,  # Power Forward - impatto su rimbalzi e paint
            'C': 1.25,   # Center - controllo area, rim protection
            'G': 1.25,   # Generic Guard
            'F': 1.15    # Generic Forward
        }
        self.default_pos_weight = 1.15

        # SISTEMA PROFESSIONALE: Status weights calibrati su analisi mercato
        self.status_weights = {
            'out': 1.0,                    # 100% impatto
            'doubtful': 0.80,              # 80% probabilità assenza
            'questionable': 0.45,          # 45% probabilità assenza (mercato professionale)
            'probable': 0.15,              # 15% probabilità assenza
            'game-time-decision': 0.50,    # 50% probabilità assenza
            'active': 0.0                  # Nessun impatto
        }
        self.default_status_weight = 0.45
        
        # COEFFICIENTI PROFESSIONALI per calcolo impatto
        self.star_player_threshold = 20.0   # PIE > 20 = superstar
        self.key_player_threshold = 15.0    # PIE > 15 = key player
        self.role_player_threshold = 10.0   # PIE > 10 = role player
        
        print("✅ [DEBUG] Inizializzazione PlayerImpactAnalyzer completata.")

    def _get_professional_player_impact(self, player_row):
        """
        Calcolo dell'impatto utilizzando metodologie professionali allineate ai mercati di scommesse.
        Combina PIE (Player Impact Estimate) + Usage Rate + Team Context
        """
        # Statistiche base
        pts = player_row.get('PTS', 0)
        fgm = player_row.get('FGM', 0)
        ftm = player_row.get('FTM', 0)
        fga = player_row.get('FGA', 1)
        fta = player_row.get('FTA', 1) 
        reb = player_row.get('REB', 0)
        ast = player_row.get('AST', 0)
        stl = player_row.get('STL', 0)
        blk = player_row.get('BLK', 0)
        tov = player_row.get('TOV', 0)
        pf = player_row.get('PF', 0)
        minutes = player_row.get('MIN', 20)  # Default 20 min se mancante

        # STEP 1: PIE (Player Impact Estimate) - Formula NBA ufficiale migliorata
        pie_score = (pts + fgm + ftm - fga - fta + reb + (0.5 * ast) + stl + (0.5 * blk) - pf - tov)
        
        # STEP 2: Usage Rate Impact (fondamentale per i professionisti)
        # Stima Usage Rate basata su FGA, FTA, TOV, AST
        estimated_usage = (fga + (0.44 * fta) + tov + (0.33 * ast)) / max(minutes, 1) * 48
        usage_multiplier = min(1.5, estimated_usage / 20)  # Cap a 1.5x per usage molto alta
        
        # STEP 3: Pace Impact Factor (quanto il giocatore influenza il ritmo)
        pace_factor = 1.0
        if ast >= 5:  # Playmaker principale
            pace_factor = 1.25
        elif ast >= 3:  # Playmaker secondario
            pace_factor = 1.10
            
        # STEP 4: Two-Way Impact (offense + defense)
        defensive_factor = 1.0 + (stl + blk) * 0.15  # Bonus per difensori
        
        # STEP 5: Calcolo impatto finale con scaling professionale
        base_impact = pie_score * 0.35  # Scaling PIE per punti di impatto
        
        # Applica tutti i moltiplicatori professionali
        professional_impact = base_impact * usage_multiplier * pace_factor * defensive_factor
        
        # STEP 6: Tiering basato su standard professionali
        if pie_score >= self.star_player_threshold:
            # SUPERSTAR: 4-8 punti di impatto (standard professionale)
            return np.clip(professional_impact, 4.0, 8.0)
        elif pie_score >= self.key_player_threshold:
            # KEY PLAYER: 2-4 punti di impatto
            return np.clip(professional_impact, 2.0, 4.0)
        elif pie_score >= self.role_player_threshold:
            # ROLE PLAYER: 0.5-2 punti di impatto
            return np.clip(professional_impact, 0.5, 2.0)
        else:
            # BENCH PLAYER: 0-1 punto di impatto
            return np.clip(professional_impact, 0.0, 1.0)

    def _get_player_base_impact(self, player_row):
        """
        Calcola un punteggio di impatto base per un giocatore usando metodologie professionali.
        Versione legacy mantenuta per compatibilità, ma usa il nuovo algoritmo.
        """
        return self._get_professional_player_impact(player_row)

    def calculate_team_impact(self, team_roster_df, team_id=None):
        """
        Calcola l'impatto totale degli infortuni per una squadra utilizzando standard professionali.
        Restituisce un dizionario con l'impatto totale e i dettagli dei giocatori.
        
        Args:
            team_roster_df (pd.DataFrame): Il roster della squadra.
            team_id (int, optional): L'ID della squadra per il caching.
        """
        if team_roster_df is None or team_roster_df.empty:
            return {'total_impact': 0.0, 'players_affected': 0, 'injured_players_details': []}

        if 'status' not in team_roster_df.columns:
            return {'total_impact': 0.0, 'players_affected': 0, 'injured_players_details': []}
            
        injured_players = team_roster_df[team_roster_df['status'].str.lower() != 'active']

        if injured_players.empty:
            return {'total_impact': 0.0, 'players_affected': 0, 'injured_players_details': []}

        total_impact = 0.0
        players_affected = 0
        injured_players_details = []

        for _, player_row in injured_players.iterrows():
            player_name = player_row.get('PLAYER_NAME', 'Sconosciuto')
            player_status = player_row.get('status', 'questionable').lower()
            
            # USA ALGORITMO PROFESSIONALE
            base_impact = self._get_professional_player_impact(player_row)
            position = player_row.get('POSITION', 'F')
            pos_weight = self.position_weights.get(position, self.default_pos_weight)
            status_weight = self.status_weights.get(player_status, self.default_status_weight)
            
            # Calcolo finale con pesi professionali
            final_player_impact = base_impact * pos_weight * status_weight
            
            detail_string = f"{player_name} ({player_status.capitalize()}) - Impatto: {-(final_player_impact):.2f} pts"
            injured_players_details.append(detail_string)

            total_impact += final_player_impact
            players_affected += 1
        
        # GESTIONE STACK DI INFORTUNI (best practice professionale)
        if players_affected >= 3:
            # Effetto compounding per multiple injuries
            total_impact *= 1.15  # +15% per effetto compounding
        elif players_affected >= 2:
            total_impact *= 1.05  # +5% per double injury
            
        final_total_impact = -round(total_impact, 2)
        
        return {
            'total_impact': final_total_impact,
            'players_affected': players_affected,
            'injured_players_details': injured_players_details,
            'methodology': 'Professional v4.0 - Market Aligned'
        } 