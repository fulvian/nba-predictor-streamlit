# odds_monitor_service.py

import time
import os
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from data_ingestion_worker import run_ingestion_cycle
from odds_analysis_engine import run_analysis_for_all_games
from database_manager import initialize_database
from notification_manager import send_system_status_notification
import config

load_dotenv()
POLLING_INTERVAL_MINUTES = int(os.getenv('POLLING_INTERVAL_MINUTES', 15))

def validate_environment():
    """Verifica che tutte le configurazioni necessarie siano presenti"""
    required_configs = {
        'TELEGRAM_BOT_TOKEN': config.TELEGRAM_BOT_TOKEN,
        'TELEGRAM_CHAT_ID': config.TELEGRAM_CHAT_ID
    }
    
    api_configs = {
        'THE_ODDS_API_KEY': config.THE_ODDS_API_KEY,
        'THERUNDOWN_API_KEY': config.THERUNDOWN_API_KEY,
        'API_SPORTS_KEY': config.API_SPORTS_KEY
    }
    
    missing_required = [var for var, value in required_configs.items() if not value or 'your_' in str(value)]
    missing_optional = [var for var, value in api_configs.items() if not value or 'your_' in str(value)]
    
    if missing_required:
        raise ValueError(f"Configurazioni OBBLIGATORIE mancanti: {', '.join(missing_required)}")
    
    if len(missing_optional) == len(api_configs):
        raise ValueError("Almeno una API deve essere configurata!")
    
    if missing_optional:
        print(f"⚠️ API opzionali non configurate: {', '.join(missing_optional)}")
        print("   Il sistema funzionerà con le API disponibili")
    
    print("✅ Configurazione validata correttamente")
    return True

def monitoring_job():
    """
    Il lavoro principale eseguito periodicamente dallo scheduler.
    Orchestra l'intero processo di monitoraggio quote e analisi.
    """
    print("🚀 --- Avvio ciclo di monitoraggio completo ---")

    try:
        # 1. Fase di Ingestione: Recupera dati da tutte le API disponibili
        print("📡 Fase 1: Ingestione dati da multiple API...")
        run_ingestion_cycle()
        
        # 2. Fase di Analisi: Analizza tutti i dati per rilevare opportunità
        print("🔍 Fase 2: Analisi avanzata dei movimenti...")
        run_analysis_for_all_games()
        
        print("✅ --- Ciclo di monitoraggio completato con successo ---")
        
    except Exception as e:
        error_message = f"❌ Errore durante il ciclo di monitoraggio: {str(e)}"
        print(error_message)
        
        # Invia notifica di errore
        send_system_status_notification(
            status="ERRORE",
            details=f"Errore nel ciclo di monitoraggio:\n{str(e)[:500]}"
        )

def send_startup_notification():
    """Invia una notifica di avvio del sistema."""
    startup_details = (
        f"🔧 Configurazione:\n"
        f"• Intervallo polling: {POLLING_INTERVAL_MINUTES} minuti\n"
        f"• Database: {config.DATABASE_URL}\n"
        f"• API configurate: {get_configured_apis()}"
    )
    
    send_system_status_notification(
        status="AVVIATO",
        details=startup_details
    )

def get_configured_apis():
    """Restituisce una lista delle API configurate."""
    apis = []
    if config.THE_ODDS_API_KEY and 'your_' not in config.THE_ODDS_API_KEY:
        apis.append("The Odds API")
    if config.THERUNDOWN_API_KEY and 'your_' not in config.THERUNDOWN_API_KEY:
        apis.append("TheRundown")
    if config.API_SPORTS_KEY and 'your_' not in config.API_SPORTS_KEY:
        apis.append("API-Sports")
    
    return ", ".join(apis) if apis else "Nessuna API configurata"

def run_initial_cycle():
    """Esegue un ciclo iniziale di test al primo avvio."""
    print("🧪 Esecuzione ciclo iniziale di test...")
    try:
        monitoring_job()
        print("✅ Ciclo iniziale completato con successo")
    except Exception as e:
        print(f"❌ Errore nel ciclo iniziale: {e}")
        print("⚠️ Il servizio continuerà comunque...")

if __name__ == "__main__":
    print("🏀 NBA PREDICTOR - Servizio di Monitoraggio Quote Avanzato")
    print("=" * 60)
    
    # Valida la configurazione prima di avviare il servizio
    try:
        validate_environment()
    except ValueError as e:
        print(f"❌ Errore di configurazione: {e}")
        print("💡 Assicurati di aver configurato correttamente il file .env")
        print("\n📋 Variabili richieste nel file .env:")
        print("   - TELEGRAM_BOT_TOKEN")
        print("   - TELEGRAM_CHAT_ID")
        print("   - THE_ODDS_API_KEY (almeno una delle API)")
        print("   - THERUNDOWN_API_KEY (opzionale)")
        print("   - API_SPORTS_KEY (opzionale)")
        exit(1)

    # Inizializza il database all'avvio del servizio
    print("🗄️ Inizializzazione database...")
    initialize_database()

    # Invia notifica di avvio
    send_startup_notification()

    # Esegui un ciclo iniziale per testare il sistema
    run_initial_cycle()

    # Configura lo scheduler per eseguire il lavoro in background
    print(f"⏰ Configurazione scheduler (ogni {POLLING_INTERVAL_MINUTES} minuti)...")
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        monitoring_job, 
        'interval', 
        minutes=POLLING_INTERVAL_MINUTES,
        id='nba_monitoring_job',
        max_instances=1  # Previene sovrapposizioni
    )
    scheduler.start()

    print("🚀 Servizio di Monitoraggio Quote NBA avviato con successo!")
    print(f"📊 Il controllo verrà eseguito ogni {POLLING_INTERVAL_MINUTES} minuti")
    print("🔔 Le notifiche saranno inviate su Telegram")
    print("⌨️ Premi Ctrl+C per terminare il servizio")
    print("=" * 60)

    try:
        # Mantieni lo script in esecuzione
        while True:
            time.sleep(60)  # Check ogni minuto per un clean shutdown
    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Arresto del servizio in corso...")
        scheduler.shutdown(wait=True)
        
        # Invia notifica di spegnimento
        send_system_status_notification(
            status="SPENTO",
            details="Servizio di monitoraggio arrestato manualmente"
        )
        
        print("✅ Servizio di monitoraggio terminato correttamente")
        print("👋 Arrivederci!") 