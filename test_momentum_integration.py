#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test di integrazione per il calcolo del momentum dei giocatori NBA.
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Aggiungi la directory principale al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider import NBADataProvider
from player_momentum_predictor import PlayerMomentumPredictor

def main():
    # Inizializza il data provider
    print("🔄 Inizializzazione del data provider...")
    data_provider = NBADataProvider()
    
    # Inizializza il predittore di momentum
    print("🔄 Inizializzazione del predittore di momentum...")
    momentum_predictor = PlayerMomentumPredictor(
        data_dir='data', 
        models_dir='models', 
        nba_data_provider=data_provider
    )
    
    # Scegli una squadra di test (es. Los Angeles Lakers)
    team_name = "Lakers"
    print(f"\n🏀 Test con la squadra: {team_name}")
    
    try:
        # Recupera il roster della squadra
        print(f"\n📋 Recupero il roster di {team_name}...")
        roster = data_provider.get_team_roster(team_name)
        
        if not roster:
            print("❌ Impossibile recuperare il roster della squadra")
            return
        
        # Stampa un riepilogo del roster
        print(f"\n📊 Roster di {team_name} (primi 5 giocatori):")
        for i, player in enumerate(roster[:5], 1):
            print(f"   {i}. {player['PLAYER_NAME']} ({player['POSITION']}) - {player['ROTATION_STATUS']}")
        
        # Calcola l'impatto del momentum per la squadra
        print(f"\n📈 Calcolo l'impatto del momentum per {team_name}...")
        momentum_result = momentum_predictor.predict_team_momentum_impact(roster)
        
        # Stampa i risultati
        print("\n🎯 Risultati del momentum:")
        print(f"   • Punteggio complessivo: {momentum_result['momentum_score']:.2f}")
        print(f"   • Momentum offensivo: {momentum_result['offensive_momentum']:.2f}")
        print(f"   • Momentum difensivo: {momentum_result['defensive_momentum']:.2f}")
        print(f"   • Consistenza: {momentum_result['consistency']:.2f}")
        print(f"   • Tendenza (ultime 5 partite): {momentum_result['trend_5_games']:+.2f}")
        
        # Stampa i contributi dei giocatori principali
        if 'player_contributions' in momentum_result and momentum_result['player_contributions']:
            print("\n🏆 Contributi dei giocatori principali:")
            top_players = sorted(
                momentum_result['player_contributions'], 
                key=lambda x: x['weighted_contribution'], 
                reverse=True
            )[:5]  # Prendi i primi 5 giocatori per contributo
            
            for i, player in enumerate(top_players, 1):
                print(f"   {i}. {player['player_name']}:")
                print(f"      • Punteggio: {player['momentum_score']:.2f}")
                print(f"      • Contributo: {player['weighted_contribution']:.2f}")
                print(f"      • Ruolo: {player['position']} ({player['status']})")
        
        # Se c'è un errore, stampalo
        if 'error' in momentum_result:
            print(f"\n⚠️ Attenzione: {momentum_result['error']}")
        
        print("\n✅ Test completato con successo!")
        
    except Exception as e:
        print(f"\n❌ Si è verificato un errore durante il test:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
