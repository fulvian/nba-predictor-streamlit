# notification_manager.py

import requests
import os
from dotenv import load_dotenv

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '7802329679:AAG3nkMX03vGSqWuAI1OuLtV4MMphEPJomw')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '987117252')

def send_telegram_notification(message: str):
    """
    Invia un messaggio a una chat Telegram specificata tramite un bot.
    Compatibile con python-telegram-bot v20+
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Credenziali Telegram non configurate, messaggio non inviato")
        print(f"📄 Messaggio che sarebbe stato inviato:\n{message}")
        return False
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # Usa data invece di json per compatibilità
    data = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown',
        'disable_web_page_preview': True
    }
    
    try:
        response = requests.post(url, data=data, timeout=10)
        response.raise_for_status()
        print("✅ Notifica Telegram inviata con successo")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Errore invio notifica Telegram: {e}")
        # Fallback: stampa il messaggio in console
        print(f"📄 Messaggio non inviato:\n{message}")
        return False

def format_bet_alert(game, bet_type, line, odds, reason, confidence, stake_suggestion):
    """
    Crea un messaggio ben formattato e dettagliato per la notifica.
    Include emoji e formattazione per una migliore leggibilità.
    """
    # Formatta l'header con emoji appropriati
    if "RLM FORTE" in reason:
        header = "🚨🔥 *ALLARME RLM FORTE* 🔥🚨"
    elif "STEAM MOVE" in reason:
        header = "💨⚡ *STEAM MOVE RILEVATO* ⚡💨"
    else:
        header = "🚨 *Opportunità di Scommessa NBA* 🚨"
    
    # Emoji per il tipo di scommessa
    bet_emoji = "📈" if bet_type.upper() == "OVER" else "📉"
    
    # Colore per la confidenza
    if "MOLTO ALTA" in confidence:
        conf_emoji = "🔥🔥🔥"
    elif "ALTA" in confidence:
        conf_emoji = "🔥🔥"
    elif "MEDIA" in confidence:
        conf_emoji = "🔥"
    else:
        conf_emoji = "⚠️"
    
    message = (
        f"{header}\n\n"
        f"🏀 *Partita:* `{game.get('away_team', 'Away')} @ {game.get('home_team', 'Home')}`\n"
        f"{bet_emoji} *Scommessa:* `{bet_type.upper()} {line}`\n"
        f"💰 *Quota:* `{odds}`\n"
        f"{conf_emoji} *Confidenza:* `{confidence}`\n"
        f"💸 *Stake Suggerito:* `{stake_suggestion}`\n\n"
        f"🔍 *Analisi Dettagliata:*\n{reason}\n\n"
        f"⚠️ *Ricorda:* Scommetti sempre responsabilmente!\n"
        f"📊 *Timestamp:* `{get_current_timestamp()}`"
    )
    return message

def get_current_timestamp():
    """Restituisce il timestamp corrente formattato."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def send_system_status_notification(status: str, details: str = ""):
    """
    Invia notifiche di stato del sistema (avvio, errori, statistiche).
    """
    message = (
        f"🤖 *Sistema NBA Predictor*\n\n"
        f"📊 *Status:* `{status}`\n"
        f"⏰ *Timestamp:* `{get_current_timestamp()}`"
    )
    
    if details:
        message += f"\n\n📝 *Dettagli:*\n{details}"
    
    send_telegram_notification(message)

if __name__ == '__main__':
    # Test di invio notifica
    test_message = format_bet_alert(
        game={'home_team': 'Lakers', 'away_team': 'Clippers'},
        bet_type='Over',
        line=220.5,
        odds=1.91,
        reason='🚨 RLM FORTE sull\'OVER: Linea salita da 218.5 a 220.5 (+2.0) nonostante 78.3% del denaro pubblico e 72.1% dei ticket siano sull\'UNDER. Sharp money chiaramente sull\'OVER.',
        confidence='MOLTO ALTA (Multiple conferme)',
        stake_suggestion='2.0 Unità (Max)'
    )
    print("📱 Test invio notifica...")
    send_telegram_notification(test_message) 