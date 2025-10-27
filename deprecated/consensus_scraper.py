# consensus_scraper.py
# Modulo per recuperare dati di consenso tramite web scraping

import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import re

def scrape_actionnetwork_consensus():
    """
    Scrape dati di consenso da Action Network (sezione pubblica)
    """
    consensus_data = []
    
    try:
        # URL pubblico di Action Network per dati NBA
        url = "https://www.actionnetwork.com/nba/public-betting"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Cerca elementi con dati di consenso
        # Questo è un esempio - la struttura HTML va adattata
        games = soup.find_all('div', class_='game-row')  # Esempio
        
        for game in games:
            # Estrai dati specifici (da adattare alla struttura reale)
            teams = game.find('span', class_='teams')
            consensus = game.find('span', class_='public-percentage')
            
            if teams and consensus:
                consensus_data.append({
                    'game_id': f"action_{teams.text.replace(' ', '_')}",
                    'source': 'ActionNetwork',
                    'public_money_percentage_over': extract_percentage(consensus.text),
                    'timestamp': datetime.now()
                })
        
    except Exception as e:
        print(f"❌ Errore scraping Action Network: {e}")
    
    return consensus_data

def scrape_covers_consensus():
    """
    Scrape dati di consenso da Covers.com (gratuito)
    """
    consensus_data = []
    
    try:
        url = "https://www.covers.com/sport/basketball/nba/odds"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Implementa logic parsing per Covers
        # Struttura da adattare in base al sito
        
    except Exception as e:
        print(f"❌ Errore scraping Covers: {e}")
    
    return consensus_data

def extract_percentage(text):
    """Estrae percentuale da testo"""
    if text:
        match = re.search(r'(\d+(?:\.\d+)?)%', text)
        if match:
            return float(match.group(1))
    return 50.0  # Default

def get_consensus_data():
    """Orchestratore principale per raccogliere dati consenso"""
    print("📊 Raccogliendo dati consenso tramite scraping...")
    
    all_consensus = []
    
    # Action Network
    action_data = scrape_actionnetwork_consensus()
    all_consensus.extend(action_data)
    
    # Covers
    covers_data = scrape_covers_consensus()
    all_consensus.extend(covers_data)
    
    print(f"✅ Raccolti {len(all_consensus)} record di consenso")
    return all_consensus

if __name__ == '__main__':
    # Test del scraper
    data = get_consensus_data()
    for item in data:
        print(f"📊 {item['source']}: {item['public_money_percentage_over']}% pubblico sull'Over") 