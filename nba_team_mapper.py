#!/usr/bin/env python3
"""
🏀 NBA Team ID Mapper - Real Data Only
Niente mock, solo dati reali da NBA API
"""

from nba_api.stats.static import teams as nba_teams
from typing import Dict, Optional

class NBATeamMapper:
    """
    Mapping professionale tra Team ID NBA e abbreviazioni/numeri.
    Solo dati reali, niente mock.
    """

    def __init__(self):
        """Inizializza con dati reali NBA API"""
        print("🔄 Caricamento team mapping da NBA API...")

        # Ottieni dati reali da NBA API
        self.nba_teams = nba_teams.get_teams()

        # Crea mapping dictionaries
        self.id_to_abbreviation = {}
        self.id_to_full_name = {}
        self.abbreviation_to_id = {}
        self.full_name_to_id = {}

        # Popola mapping con dati reali
        for team in self.nba_teams:
            team_id = team['id']
            abbreviation = team['abbreviation']
            full_name = team['full_name']

            self.id_to_abbreviation[team_id] = abbreviation
            self.id_to_full_name[team_id] = full_name
            self.abbreviation_to_id[abbreviation] = team_id
            self.full_name_to_id[full_name] = team_id

        print(f"✅ Team mapping caricato: {len(self.nba_teams)} squadre reali")

    def get_abbreviation(self, team_id: int) -> Optional[str]:
        """
        Converte Team ID in abbreviation es. 1610612754 → IND

        Args:
            team_id: NBA team ID (es. 1610612754)

        Returns:
            Team abbreviation (es. "IND") o None se non trovato
        """
        return self.id_to_abbreviation.get(team_id)

    def get_full_name(self, team_id: int) -> Optional[str]:
        """
        Converte Team ID in full name es. 1610612754 → Indiana Pacers

        Args:
            team_id: NBA team ID (es. 1610612754)

        Returns:
            Team full name (es. "Indiana Pacers") o None se non trovato
        """
        return self.id_to_full_name.get(team_id)

    def get_team_id(self, team_identifier: str) -> Optional[int]:
        """
        Converte abbreviation o full name in Team ID

        Args:
            team_identifier: "IND" o "Indiana Pacers"

        Returns:
            Team ID (es. 1610612754) o None se non trovato
        """
        # Prova prima come abbreviation
        if team_identifier.upper() in self.abbreviation_to_id:
            return self.abbreviation_to_id[team_identifier.upper()]

        # Prova come full name
        if team_identifier in self.full_name_to_id:
            return self.full_name_to_id[team_identifier]

        return None

    def validate_team_id(self, team_id: int) -> bool:
        """
        Verifica se un team ID è valido

        Args:
            team_id: Team ID da validare

        Returns:
            True se il team ID esiste, False altrimenti
        """
        return team_id in self.id_to_abbreviation

    def get_team_info(self, team_id: int) -> Optional[Dict]:
        """
        Ottiene informazioni complete su un team

        Args:
            team_id: NBA team ID

        Returns:
            Dictionary con tutte le info del team o None
        """
        if not self.validate_team_id(team_id):
            return None

        return {
            'team_id': team_id,
            'abbreviation': self.get_abbreviation(team_id),
            'full_name': self.get_full_name(team_id),
            'city': None,  # Potremmo aggiungere queste info in futuro
            'state': None
        }

    def print_all_teams(self):
        """Stampa tutte le squadre disponibili (debug)"""
        print("\n🏀 NBA Teams Available:")
        print("=" * 60)
        for team_id in sorted(self.id_to_abbreviation.keys()):
            abbreviation = self.id_to_abbreviation[team_id]
            full_name = self.id_to_full_name[team_id]
            print(f"{team_id:10d} | {abbreviation:3s} | {full_name}")


# Singleton per uso globale
_team_mapper = None

def get_team_mapper() -> NBATeamMapper:
    """Ottieni istanza del team mapper (singleton)"""
    global _team_mapper
    if _team_mapper is None:
        _team_mapper = NBATeamMapper()
    return _team_mapper


# Funzioni helper per accesso rapido
def team_id_to_abbreviation(team_id: int) -> Optional[str]:
    """Converte team ID in abbreviation"""
    return get_team_mapper().get_abbreviation(team_id)


def team_id_to_full_name(team_id: int) -> Optional[str]:
    """Converte team ID in full name"""
    return get_team_mapper().get_full_name(team_id)


def team_identifier_to_id(team_identifier: str) -> Optional[int]:
    """Converte abbreviation/full name in team ID"""
    return get_team_mapper().get_team_id(team_identifier)


if __name__ == "__main__":
    # Test del mapper
    mapper = NBATeamMapper()

    # Test mapping ID → abbreviation
    print("🔍 Test Mapping:")
    test_ids = [1610612754, 1610612760, 1610612747, 1610612737]  # IND, OKC, LAL, BOS

    for team_id in test_ids:
        abbreviation = mapper.get_abbreviation(team_id)
        full_name = mapper.get_full_name(team_id)
        print(f"  {team_id} → {abbreviation} ({full_name})")

    # Test reverse mapping
    print("\n🔍 Test Reverse Mapping:")
    test_identifiers = ["IND", "Los Angeles Lakers", "OKC"]

    for identifier in test_identifiers:
        team_id = mapper.get_team_id(identifier)
        print(f"  '{identifier}' → {team_id}")