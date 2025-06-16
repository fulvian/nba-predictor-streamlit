# player_impact_analyzer.py
import pandas as pd
from datetime import datetime, timedelta

print("🔄 Modulo player_impact_analyzer caricato con successo!")

class PlayerImpactAnalyzer:
    """
    Analizza l'impatto di un giocatore sulla sua squadra usando metriche dettagliate.
    Versione Definitiva 3.0
    """
    
    def __init__(self, nba_data_provider=None):
        print("\n🔍 [DEBUG] Inizializzazione PlayerImpactAnalyzer...")
        self.nba_data_provider = nba_data_provider
        self.player_impact_cache = {}
        self.team_impact_cache = {}
        self.cache_expiry = 3600  # 1 ora

        self.position_weights = {
            'G': 1.15, 'F': 1.05, 'C': 1.0, 'PG': 1.2, 'SG': 1.1, 'SF': 1.0, 'PF': 1.0
        }
        self.default_pos_weight = 1.0

        # Fattori di impatto per status. 1.0 = impatto pieno, 0.5 = impatto dimezzato
        self.status_weights = {
            'out': 1.0,
            'doubtful': 0.75,
            'questionable': 0.5,
            'probable': 0.25,
            'game-time-decision': 0.5,
            'active': 0.0
        }
        self.default_status_weight = 0.5
        
        print("✅ [DEBUG] Inizializzazione PlayerImpactAnalyzer completata.")

    def _get_player_base_impact(self, player_row):
        """
        Calcola un punteggio di impatto base per un giocatore usando le sue statistiche.
        Utilizza una formula simile al PIE (Player Impact Estimate).
        """
        # Usa i dati dalla riga del DataFrame, con fallback a 0 se mancano
        pts = player_row.get('PTS', 0)
        fgm = player_row.get('FGM', 0)
        ftm = player_row.get('FTM', 0)
        fga = player_row.get('FGA', 1) # Evita divisione per zero
        fta = player_row.get('FTA', 1) # Evita divisione per zero
        reb = player_row.get('REB', 0)
        ast = player_row.get('AST', 0)
        stl = player_row.get('STL', 0)
        blk = player_row.get('BLK', 0)
        tov = player_row.get('TOV', 0)
        pf = player_row.get('PF', 0)

        # Formula semplificata del PIE, normalizzata per dare un valore di impatto
        player_impact = (pts + fgm + ftm - fga - fta + reb + (0.5 * ast) + stl + (0.5 * blk) - pf - tov)
        
        # Normalizza l'impatto in "punti persi a partita"
        # Un giocatore d'elite ha un PIE > 15, un giocatore medio ~10
        # Scaliamo questo valore per rappresentare i punti persi se il giocatore è assente
        return player_impact * 0.25 # Es: un PIE di 20 diventa 5 punti di impatto

    def calculate_team_impact(self, team_roster_df):
        """
        Calcola l'impatto totale degli infortuni per una squadra.
        Restituisce un valore negativo che rappresenta i punti persi.
        
        Args:
            team_roster_df (pd.DataFrame): Il roster della squadra, inclusa la colonna 'status'.
        """
        # Controllo robusto per DataFrame
        if team_roster_df is None or team_roster_df.empty:
            print("⚠️ [TEAM_IMPACT] Roster non fornito o vuoto. Impatto nullo.")
            return {'total_impact': 0.0, 'players_affected': 0}

        # Assicurati che la colonna 'status' esista
        if 'status' not in team_roster_df.columns:
            print("⚠️ [TEAM_IMPACT] Colonna 'status' mancante nel roster. Impatto nullo.")
            return {'total_impact': 0.0, 'players_affected': 0}
            
        # Filtra solo i giocatori il cui status NON è 'active'
        injured_players = team_roster_df[team_roster_df['status'].str.lower() != 'active']

        if injured_players.empty:
            return {'total_impact': 0.0, 'players_affected': 0}

        print(f"🔍 [TEAM_IMPACT] Trovati {len(injured_players)} giocatori non attivi. Inizio calcolo impatto.")

        total_impact = 0.0
        players_affected = 0

        for _, player_row in injured_players.iterrows():
            player_name = player_row.get('PLAYER_NAME', 'Sconosciuto')
            player_status = player_row.get('status', 'questionable').lower()

            # Calcola l'impatto base del giocatore
            base_impact = self._get_player_base_impact(player_row)
            
            # Pondera l'impatto in base alla posizione
            position = player_row.get('POSITION', 'F')
            pos_weight = self.position_weights.get(position, self.default_pos_weight)
            
            # Pondera l'impatto in base allo status dell'infortunio
            status_weight = self.status_weights.get(player_status, self.default_status_weight)
            
            # L'impatto finale del giocatore è una combinazione di questi fattori
            final_player_impact = base_impact * pos_weight * status_weight
            final_player_impact = round(final_player_impact, 2)

            print(f"   -> Giocatore: {player_name}, Status: {player_status}, Impatto: {final_player_impact:.2f} pts")

            total_impact += final_player_impact
            players_affected += 1
        
        # L'impatto è negativo perché rappresenta una perdita di punti sul totale atteso
        final_total_impact = -round(total_impact, 2)
        
        print(f"✅ [TEAM_IMPACT] Calcolo completato. Impatto totale: {final_total_impact:.2f} punti, Giocatori considerati: {players_affected}")
        
        return {
            'total_impact': final_total_impact,
            'players_affected': players_affected
        }