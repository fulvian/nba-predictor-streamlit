"""
Test per la classe NBADataProvider.

Questo script testa le funzionalità principali del modulo data_provider.
"""

import os
import sys
import json
import time
import unittest
from datetime import datetime, date
from unittest.mock import patch, MagicMock

# Aggiungi la directory principale al path per permettere l'importazione dei moduli
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_provider import NBADataProvider, Config

class TestNBADataProvider(unittest.TestCase):
    """Test case per la classe NBADataProvider."""
    
    @classmethod
    def setUpClass(cls):
        """Configurazione iniziale per tutti i test."""
        cls.provider = NBADataProvider(use_cache=False)  # Disattiva la cache per i test
    
    def setUp(self):
        """Prepara l'ambiente per ogni test."""
        # Pulisce la cache prima di ogni test
        if hasattr(self.provider, 'clear_cache'):
            self.provider.clear_cache()

    def test_get_team_info(self):
        """Test per il recupero delle informazioni di una squadra."""
        print("\n=== Test: get_team_info ===")
        
        # Test con nome completo
        team = self.provider.get_team_info("Los Angeles Lakers")
        self.assertIsNotNone(team, "Dovrebbe trovare i Lakers con il nome completo")
        
        if team:
            print(f"Trovata squadra: {team['full_name']} ({team['abbreviation']})")
        
        # Test con abbreviazione
        team = self.provider.get_team_info("LAL")
        self.assertIsNotNone(team, "Dovrebbe trovare i Lakers con l'abbreviazione")
        
        # Test con nome non esistente
        team = self.provider.get_team_info("Squadra Inesistente")
        self.assertIsNone(team, "Non dovrebbe trovare una squadra inesistente")
    
    def test_get_team_id(self):
        """Test per il recupero dell'ID di una squadra."""
        print("\n=== Test: get_team_id ===")
        
        team_id = self.provider.get_team_id("Los Angeles Lakers")
        self.assertIsNotNone(team_id, "Dovrebbe trovare l'ID dei Lakers")
        print(f"ID dei Los Angeles Lakers: {team_id}")
        
        team_id = self.provider.get_team_id("Squadra Inesistente")
        self.assertIsNone(team_id, "Non dovrebbe trovare l'ID di una squadra inesistente")
    
    def test_get_team_roster(self):
        """Test per il recupero del roster di una squadra."""
        print("\n=== Test: get_team_roster ===")
        
        roster = self.provider.get_team_roster("Los Angeles Lakers", "2023-24")
        self.assertIsNotNone(roster, "Dovrebbe restituire il roster")
        self.assertIsInstance(roster, list, "Il roster dovrebbe essere una lista")
        
        if roster:
            print(f"Trovati {len(roster)} giocatori nel roster dei Lakers 2023-24")
            print(f"Primo giocatore: {roster[0].get('PLAYER', 'N/A')}")
    
    def test_get_team_stats(self):
        """Test per il recupero delle statistiche di una squadra."""
        print("\n=== Test: get_team_stats ===")
        
        stats = self.provider.get_team_stats(
            "Los Angeles Lakers", 
            "2023-24", 
            "Regular Season", 
            "Base"
        )
        
        self.assertIsNotNone(stats, "Dovrebbe restituire le statistiche")
        self.assertIsInstance(stats, dict, "Le statistiche dovrebbero essere un dizionario")
        
        if stats:
            print(f"Statistiche dei Lakers 2023-24 (Regular Season):")
            print(f"- Partite giocate: {stats.get('W', 'N/A')}V - {stats.get('L', 'N/A')}P")
            print(f"- Punti per partita: {stats.get('PTS', 'N/A')}")
    
    @patch('data_provider.requests.get')
    def test_cache_behavior(self, mock_get):
        """Test il comportamento della cache."""
        print("\n=== Test: Comportamento della cache ===")
        
        # Configura il mock per restituire una risposta fittizia
        mock_response = MagicMock()
        mock_response.json.return_value = {'response': [{'id': 1, 'name': 'Lakers'}]}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Crea un nuovo provider con la cache abilitata
        provider = NBADataProvider(use_cache=True)
        
        # Prima chiamata - dovrebbe chiamare l'API
        result1 = provider._make_apisports_request('teams', {'league': '12', 'season': '2023'})
        
        # Seconda chiamata con gli stessi parametri - dovrebbe usare la cache
        result2 = provider._make_apisports_request('teams', {'league': '12', 'season': '2023'})
        
        # Verifica che l'API sia stata chiamata solo una volta
        mock_get.assert_called_once()
        
        # Verifica che i risultati siano gli stessi
        self.assertEqual(result1, result2, "I risultati dovrebbero essere identici")
        
        # Verifica che la cache sia stata utilizzata
        stats = provider.get_usage_stats()
        self.assertEqual(stats['cache_hits'], 1, "Dovrebbe esserci un hit in cache")
        self.assertEqual(stats['cache_misses'], 1, "Dovrebbe esserci un miss in cache")
        
        print("Test del comportamento della cache completato con successo")

    def test_head_to_head(self):
        """Test per il recupero delle statistiche degli scontri diretti."""
        print("\n=== Test: get_head_to_head_stats ===")
        
        team1 = "Los Angeles Lakers"
        team2 = "Boston Celtics"
        
        # Usa la stagione 2023-24
        h2h = self.provider.get_head_to_head_stats(team1, team2, season="2023-24", last_n_games=3)
        
        if h2h:
            print(f"Statistiche degli scontri diretti tra {team1} e {team2}:")
            print(f"- Ultimi scontri: {h2h.get('last_meetings', [])}")
            print(f"- Vittorie {team1}: {h2h.get('team1_wins', 0)}")
            print(f"- Vittorie {team2}: {h2h.get('team2_wins', 0)}")
            print(f"- Media punti {team1}: {h2h.get('team1_avg_points', 0):.1f}")
            print(f"- Media punti {team2}: {h2h.get('team2_avg_points', 0):.1f}")
        else:
            print(f"❌ Impossibile recuperare le statistiche degli scontri diretti")
    
    def test_recent_form(self):
        """Test per il recupero della forma recente di una squadra."""
        print("\n=== Test: get_recent_form ===")
        
        team_name = "Los Angeles Lakers"
        last_n_games = 5
        
        form = self.provider.get_recent_form(team_name, last_n_games=last_n_games)
        
        if form:
            print(f"Forma recente degli ultimi {last_n_games} gare per {team_name}:")
            print(f"- Vittorie: {form.get('wins', 0)}")
            print(f"- Sconfitte: {form.get('losses', 0)}")
            print(f"- Percentuale vittorie: {form.get('win_percentage', 0):.1%}")
            print(f"- Media punti fatti: {form.get('points_for_avg', 0):.1f}")
            print(f"- Media punti subiti: {form.get('points_against_avg', 0):.1f}")
            print(f"- Differenziale punti: {form.get('point_differential', 0):.1f}")
        else:
            print(f"❌ Impossibile recuperare la forma recente per {team_name}")
    
    def test_prepare_prediction_data(self):
        """Test per la preparazione dei dati per la previsione."""
        print("\n=== Test: prepare_prediction_data ===")
        
        team1 = "Los Angeles Lakers"
        team2 = "Boston Celtics"
        
        # Usa una data di esempio
        game_date = "2024-12-25"
        
        prediction_data = self.provider.prepare_prediction_data(team1, team2, game_date)
        
        if prediction_data:
            print("Dati per la previsione generati con successo!")
            print(f"Partita: {team1} vs {team2}")
            print(f"Data: {game_date}")
            print("\nDettagli completi:")
            
            # Salva i dati in un file per l'ispezione
            with open('prediction_data_dump.json', 'w') as f:
                json.dump(prediction_data, f, indent=2, default=str)
            print("\nDati completi salvati in 'prediction_data_dump.json'")
        else:
            print("❌ Impossibile preparare i dati per la previsione")

if __name__ == "__main__":
    print("=== Test NBADataProvider ===\n")
    
    # Esegui i test con unittest
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    print("\n=== Tutti i test completati ===")
    print("Nota: Controlla i file di log per eventuali avvisi o errori.")
