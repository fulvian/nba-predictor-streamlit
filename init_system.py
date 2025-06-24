#!/usr/bin/env python3
# init_system.py - Script di inizializzazione e test del sistema NBA Predictor

import os
import sys
from dotenv import load_dotenv

def check_dependencies():
    """Verifica che tutte le dipendenze siano installate."""
    print("🔍 Controllo dipendenze...")
    
    required_packages = [
        ('requests', 'requests'), 
        ('pandas', 'pandas'), 
        ('sqlalchemy', 'sqlalchemy'), 
        ('apscheduler', 'apscheduler'),
        ('python-telegram-bot', 'telegram'), 
        ('beautifulsoup4', 'bs4'), 
        ('lxml', 'lxml')
    ]
    
    missing = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            missing.append(package_name)
            print(f"  ❌ {package_name}")
    
    if missing:
        print(f"\n❌ Dipendenze mancanti: {', '.join(missing)}")
        print("💡 Esegui: pip install -r requirements.txt")
        return False
    
    print("✅ Tutte le dipendenze sono installate")
    return True

def check_environment():
    """Verifica la configurazione usando config.py."""
    print("\n🔧 Controllo configurazione...")
    
    try:
        import config
        
        # Variabili obbligatorie
        required_configs = {
            'TELEGRAM_BOT_TOKEN': config.TELEGRAM_BOT_TOKEN,
            'TELEGRAM_CHAT_ID': config.TELEGRAM_CHAT_ID
        }
        
        # Variabili API (almeno una necessaria)
        api_configs = {
            'THE_ODDS_API_KEY': config.THE_ODDS_API_KEY,
            'THERUNDOWN_API_KEY': config.THERUNDOWN_API_KEY,
            'API_SPORTS_KEY': config.API_SPORTS_KEY,
            'BALLDONTLIE_API_KEY': config.BALLDONTLIE_API_KEY
        }
        
        missing_required = []
        for var, value in required_configs.items():
            if not value or value in ['your_telegram_bot_token', 'your_telegram_chat_id']:
                missing_required.append(var)
                print(f"  ❌ {var}")
            else:
                print(f"  ✅ {var}: {'*' * (len(str(value)) - 4) + str(value)[-4:]}")
        
        # Controlla che almeno una API sia configurata
        configured_apis = [name for name, key in api_configs.items() if key and 'your_' not in str(key)]
        
        if missing_required:
            print(f"\n❌ Configurazioni obbligatorie mancanti:")
            for var in missing_required:
                print(f"   - {var}")
            return False
        
        if not configured_apis:
            print(f"\n⚠️ Nessuna API configurata!")
            print("💡 Configura almeno una API in config.py")
            return False
        
        print(f"\n✅ API configurate: {len(configured_apis)} su {len(api_configs)}")
        for api in configured_apis:
            print(f"   - {api}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Errore importazione config.py: {e}")
        return False

def create_env_template():
    """Crea un template del file .env."""
    template = """# NBA Predictor - Configurazione Environment Variables

# --- API Keys (almeno una obbligatoria) ---
THE_ODDS_API_KEY='your_odds_api_key_here'
THERUNDOWN_API_KEY='your_therundown_api_key_here'
API_SPORTS_KEY='your_apisports_key_here'

# --- Telegram Bot (obbligatorio) ---
TELEGRAM_BOT_TOKEN='your_telegram_bot_token'
TELEGRAM_CHAT_ID='your_telegram_chat_id'

# --- Configurazione Opzionale ---
DATABASE_URL='sqlite:///market_movements.db'
POLLING_INTERVAL_MINUTES='15'
MARKET_TO_MONITOR='totals'
TARGET_REGIONS='eu'
TARGET_BOOKMAKERS='pinnacle,betfair,draftkings,fanduel,betmgm'
"""
    
    with open('.env.template', 'w') as f:
        f.write(template)
    
    print(f"\n📄 Template creato: .env.template")
    print("💡 Copialo in .env e inserisci le tue credenziali")

def test_database():
    """Testa l'inizializzazione del database."""
    print("\n🗄️ Test database...")
    
    try:
        from database_manager import initialize_database, get_all_active_games
        initialize_database()
        print("  ✅ Database inizializzato correttamente")
        
        # Test query
        games = get_all_active_games()
        print(f"  📊 Partite attive nel database: {len(games)}")
        return True
        
    except Exception as e:
        print(f"  ❌ Errore database: {e}")
        return False

def test_telegram():
    """Testa la connessione Telegram."""
    print("\n📱 Test notifiche Telegram...")
    
    try:
        from notification_manager import send_system_status_notification
        
        print("  📤 Invio notifica di test...")
        send_system_status_notification(
            status="TEST SISTEMA",
            details="✅ Sistema NBA Predictor inizializzato correttamente!\n🚀 Pronto per il monitoraggio"
        )
        print("  ✅ Notifica inviata (controlla Telegram)")
        return True
        
    except Exception as e:
        print(f"  ❌ Errore Telegram: {e}")
        return False

def test_api_connectivity():
    """Testa la connettività alle API."""
    print("\n🌐 Test connettività API...")
    
    try:
        from data_ingestion_worker import fetch_from_theoddsapi, fetch_from_therundown
        
        # Test The Odds API
        if os.getenv('THE_ODDS_API_KEY'):
            print("  📡 Test The Odds API...")
            data = fetch_from_theoddsapi()
            if data:
                print(f"    ✅ The Odds API: {len(data)} record recuperati")
            else:
                print("    ⚠️ The Odds API: Nessun dato (normale se non ci sono partite)")
        
        # Test TheRundown API  
        if os.getenv('THERUNDOWN_API_KEY'):
            print("  📡 Test TheRundown API...")
            data = fetch_from_therundown()
            if data:
                print(f"    ✅ TheRundown API: {len(data)} record recuperati")
            else:
                print("    ⚠️ TheRundown API: Nessun dato (normale se non ci sono partite)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Errore test API: {e}")
        return False

def main():
    """Funzione principale di inizializzazione."""
    print("🏀 NBA PREDICTOR - Inizializzazione Sistema")
    print("=" * 50)
    
    # Step 1: Dipendenze
    if not check_dependencies():
        sys.exit(1)
    
    # Step 2: Configurazione
    if not check_environment():
        sys.exit(1)
    
    # Step 3: Database
    if not test_database():
        print("⚠️ Problemi con il database, ma si può continuare...")
    
    # Step 4: Telegram
    if not test_telegram():
        print("⚠️ Problemi con Telegram, controlla la configurazione...")
    
    # Step 5: API
    if not test_api_connectivity():
        print("⚠️ Problemi con le API, controlla le chiavi...")
    
    print("\n" + "=" * 50)
    print("🎯 INIZIALIZZAZIONE COMPLETATA")
    print("\n📚 Prossimi passi:")
    print("   1. Controlla che la notifica Telegram sia arrivata")
    print("   2. Avvia il sistema: python odds_monitor_service.py")
    print("   3. Monitora i log per verificare il funzionamento")
    print("\n📖 Per aiuto dettagliato: README_ADVANCED_MONITORING.md")

if __name__ == '__main__':
    main() 