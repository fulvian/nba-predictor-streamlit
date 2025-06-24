# config.py

import os
from dotenv import load_dotenv

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# --- API Keys ---
# The Odds API (CRUCIALE per quote NBA primarie)
THE_ODDS_API_KEY = os.getenv('THE_ODDS_API_KEY', 'd01e24415744d440168e0a489f233aac')
# TheRundown API tramite RapidAPI
THERUNDOWN_API_KEY = os.getenv('THERUNDOWN_API_KEY', '72377a40f0msh3ea874a8b0e9b42p18e27cjsn28903f0d8874')
# API-Sports tramite RapidAPI (usa la stessa key di RapidAPI)
API_SPORTS_KEY = os.getenv('API_SPORTS_KEY', '72377a40f0msh3ea874a8b0e9b42p18e27cjsn28903f0d8874')
# BallDontLie API per dati NBA
BALLDONTLIE_API_KEY = os.getenv('BALLDONTLIE_API_KEY', '0baa5751-350b-44b1-bb0b-7808683e4c96')

# --- Telegram Bot Configuration ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '7802329679:AAG3nkMX03vGSqWuAI1OuLtV4MMphEPJomw')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '987117252')

# --- Consensus Data APIs (per analisi RLM avanzata) ---
# Sports Insights API per dati di consenso pubblico/sharp
SPORTS_INSIGHTS_API_KEY = os.getenv('SPORTS_INSIGHTS_API_KEY', '')
# VSIN API per dati sharp money
VSIN_API_KEY = os.getenv('VSIN_API_KEY', '')
# Action Network API per percentuali scommesse
ACTION_NETWORK_API_KEY = os.getenv('ACTION_NETWORK_API_KEY', '')

# --- Additional Betting APIs ---
# Pinnacle API (per dati di riferimento sharp)
PINNACLE_API_KEY = os.getenv('PINNACLE_API_KEY', '')
# FanDuel API (per completare copertura bookmaker)
FANDUEL_API_KEY = os.getenv('FANDUEL_API_KEY', '')
# DraftKings API
DRAFTKINGS_API_KEY = os.getenv('DRAFTKINGS_API_KEY', '')

# --- Database Configuration ---
# Usa SQLite per semplicità, ma può essere configurato tramite .env se necessario
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///odds_movements.db')

# --- Monitoring Settings ---
# Mercato di interesse principale
MARKET_TO_MONITOR = os.getenv('MARKET_TO_MONITOR', 'totals')  # h2h, spreads, totals
# Bookmaker di riferimento (quelli più reattivi o 'sharp' come Pinnacle)
TARGET_BOOKMAKERS = os.getenv('TARGET_BOOKMAKERS', 'pinnacle,betfair,draftkings,fanduel,betmgm').split(',')
# Regioni di interesse per le quote
TARGET_REGIONS = os.getenv('TARGET_REGIONS', 'eu')  # us, uk, eu, au

# --- Scheduler Settings ---
# Frequenza di polling in minuti (può essere configurata tramite .env)
POLLING_INTERVAL_MINUTES = int(os.getenv('POLLING_INTERVAL_MINUTES', '15'))

# --- Validazione configurazione ---
def validate_config():
    """Verifica che tutte le configurazioni necessarie siano presenti"""
    required_vars = {
        'TELEGRAM_BOT_TOKEN': TELEGRAM_BOT_TOKEN,
        'TELEGRAM_CHAT_ID': TELEGRAM_CHAT_ID
    }
    
    # API Keys principali per quote
    primary_apis = {
        'THE_ODDS_API_KEY': THE_ODDS_API_KEY,
        'THERUNDOWN_API_KEY': THERUNDOWN_API_KEY,
        'API_SPORTS_KEY': API_SPORTS_KEY,
        'BALLDONTLIE_API_KEY': BALLDONTLIE_API_KEY
    }
    
    # API Keys per dati di consenso (opzionali ma importanti)
    consensus_apis = {
        'SPORTS_INSIGHTS_API_KEY': SPORTS_INSIGHTS_API_KEY,
        'VSIN_API_KEY': VSIN_API_KEY,
        'ACTION_NETWORK_API_KEY': ACTION_NETWORK_API_KEY
    }
    
    # API Keys bookmaker aggiuntive (opzionali)
    bookmaker_apis = {
        'PINNACLE_API_KEY': PINNACLE_API_KEY,
        'FANDUEL_API_KEY': FANDUEL_API_KEY,
        'DRAFTKINGS_API_KEY': DRAFTKINGS_API_KEY
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    available_primary = [name for name, key in primary_apis.items() if key]
    available_consensus = [name for name, key in consensus_apis.items() if key]
    available_bookmakers = [name for name, key in bookmaker_apis.items() if key]
    
    if missing_vars:
        raise ValueError(f"Variabili d'ambiente mancanti nel file .env: {', '.join(missing_vars)}")
    
    if not available_primary:
        raise ValueError("Nessuna API per quote configurata!")
    
    print("✅ Configurazione validata correttamente")
    print(f"📡 API Quote: {', '.join(available_primary)}")
    
    if available_consensus:
        print(f"📊 API Consenso: {', '.join(available_consensus)}")
    else:
        print("⚠️  Nessuna API consenso configurata (RLM sarà meno preciso)")
    
    if available_bookmakers:
        print(f"🏪 API Bookmaker: {', '.join(available_bookmakers)}")
    
    return True

if __name__ == '__main__':
    validate_config() 