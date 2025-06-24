# odds_analysis_engine.py

from database_manager import get_odds_history_for_game, get_consensus_history, get_all_active_games
from notification_manager import send_telegram_notification, format_bet_alert
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
TARGET_BOOKMAKERS = os.getenv('TARGET_BOOKMAKERS', 'pinnacle,betfair,draftkings').split(',')

def detect_true_reverse_line_movement(df_odds: pd.DataFrame, df_consensus: pd.DataFrame):
    """
    Rileva RLM confrontando i movimenti di linea con le scommesse del pubblico.
    Un vero RLM si verifica quando:
    1. La linea si muove in una direzione
    2. Ma la maggioranza del pubblico sta scommettendo nella direzione opposta
    3. Questo suggerisce che gli "sharp bettors" stanno influenzando la linea
    
    Args:
        df_odds: DataFrame con lo storico delle quote
        df_consensus: DataFrame con i dati di consenso pubblico
    
    Returns: 
        Una stringa descrittiva se viene rilevato RLM, altrimenti None.
    """
    if df_odds.empty or len(df_odds) < 3:
        return None

    # Prendi i dati più recenti e iniziali
    latest_odds = df_odds.iloc[-1]
    initial_odds = df_odds.iloc[0]
    
    line_change = latest_odds['line'] - initial_odds['line']
    
    # Se abbiamo dati di consenso, usali per un'analisi più precisa
    if not df_consensus.empty:
        latest_consensus = df_consensus.iloc[-1]
        public_money_on_over = latest_consensus.get('public_money_percentage_over', 50.0)
        public_tickets_on_over = latest_consensus.get('public_tickets_percentage_over', 50.0)
        
        # RLM sull'UNDER: linea scende (meno punti attesi) MA il pubblico sta massicciamente sull'OVER
        # Questo indica che gli sharp stanno pesantemente sull'UNDER
        if line_change < -1.0 and public_money_on_over > 70:
            return (f"🚨 RLM FORTE sull'UNDER: Linea scesa da {initial_odds['line']} a {latest_odds['line']} "
                   f"(-{abs(line_change):.1f}) nonostante {public_money_on_over:.1f}% del denaro pubblico "
                   f"e {public_tickets_on_over:.1f}% dei ticket siano sull'OVER. "
                   f"Sharp money chiaramente sull'UNDER.")

        # RLM sull'OVER: linea sale (più punti attesi) MA il pubblico sta massicciamente sull'UNDER  
        public_money_on_under = 100 - public_money_on_over
        public_tickets_on_under = 100 - public_tickets_on_over
        if line_change > 1.0 and public_money_on_under > 70:
            return (f"🚨 RLM FORTE sull'OVER: Linea salita da {initial_odds['line']} a {latest_odds['line']} "
                   f"(+{line_change:.1f}) nonostante {public_money_on_under:.1f}% del denaro pubblico "
                   f"e {public_tickets_on_under:.1f}% dei ticket siano sull'UNDER. "
                   f"Sharp money chiaramente sull'OVER.")
    else:
        # Fallback: analisi senza dati di consenso (logica precedente migliorata)
        over_odds_change = latest_odds['odds_over'] - initial_odds['odds_over']
        under_odds_change = latest_odds['odds_under'] - initial_odds['odds_under']
        
        # RLM potenziale basato solo su movimenti di quota inusuali
        if line_change > 1.5 and over_odds_change < -0.15:
            return (f"⚠️ RLM Potenziale (OVER): Linea salita di {line_change:.1f} punti "
                   f"mentre quota Over scesa (da {initial_odds['odds_over']:.2f} a {latest_odds['odds_over']:.2f}). "
                   f"Possibile sharp action sull'OVER.")

        if line_change < -1.5 and under_odds_change < -0.15:
            return (f"⚠️ RLM Potenziale (UNDER): Linea scesa di {abs(line_change):.1f} punti "
                   f"mentre quota Under scesa (da {initial_odds['odds_under']:.2f} a {latest_odds['odds_under']:.2f}). "
                   f"Possibile sharp action sull'UNDER.")
                
    return None

def detect_steam_moves(df_odds: pd.DataFrame):
    """
    Rileva "Steam Moves" - movimenti di quota rapidi e significativi 
    che spesso indicano scommesse importanti da parte di sharp bettors.
    """
    if len(df_odds) < 3:
        return None
    
    # Calcola i movimenti di quota negli ultimi 3 rilevamenti
    recent_data = df_odds.tail(3)
    
    # Steam move su OVER: quota scende rapidamente
    if len(recent_data) >= 2:
        over_change = recent_data.iloc[-1]['odds_over'] - recent_data.iloc[0]['odds_over']
        line_change = recent_data.iloc[-1]['line'] - recent_data.iloc[0]['line']
        
        # Steam move significativo: quota scende di almeno 0.10 in poco tempo
        if over_change <= -0.10:
            return (f"💨 STEAM MOVE su OVER: Quota scesa rapidamente di {abs(over_change):.2f} "
                   f"(da {recent_data.iloc[0]['odds_over']:.2f} a {recent_data.iloc[-1]['odds_over']:.2f}). "
                   f"Linea: {recent_data.iloc[-1]['line']}")
        
        under_change = recent_data.iloc[-1]['odds_under'] - recent_data.iloc[0]['odds_under']
        if under_change <= -0.10:
            return (f"💨 STEAM MOVE su UNDER: Quota scesa rapidamente di {abs(under_change):.2f} "
                   f"(da {recent_data.iloc[0]['odds_under']:.2f} a {recent_data.iloc[-1]['odds_under']:.2f}). "
                   f"Linea: {recent_data.iloc[-1]['line']}")
    
    return None

def detect_late_game_value(df_odds: pd.DataFrame, hours_to_game: float = None):
    """
    Rileva opportunità di valore nelle ultime ore prima della partita.
    Gli sharp bettors spesso aspettano fino all'ultimo per piazzare le loro scommesse.
    """
    if len(df_odds) < 2:
        return None
        
    # Se abbiamo informazioni sui tempi, considera solo movimenti nelle ultime 4 ore
    # Per ora, analizziamo gli ultimi movimenti disponibili
    latest = df_odds.iloc[-1]
    
    # Cerca discrepanze tra quote di bookmaker diversi
    # (questo richiederebbe dati da multiple bookmaker nella stessa query)
    
    return None

def calculate_confidence_score(rlm_reason: str, steam_reason: str, df_consensus: pd.DataFrame) -> str:
    """
    Calcola un punteggio di confidenza basato sui segnali rilevati.
    """
    confidence_factors = []
    
    if rlm_reason and "RLM FORTE" in rlm_reason:
        confidence_factors.append("RLM con dati consenso")
    elif rlm_reason:
        confidence_factors.append("RLM senza dati consenso")
        
    if steam_reason:
        confidence_factors.append("Steam Move")
        
    if not df_consensus.empty:
        confidence_factors.append("Dati consenso disponibili")
    
    if len(confidence_factors) >= 3:
        return "MOLTO ALTA (Multiple conferme)"
    elif len(confidence_factors) == 2:
        return "ALTA (Doppia conferma)"
    elif len(confidence_factors) == 1:
        return "MEDIA (Singolo segnale)"
    else:
        return "BASSA"

def analyze_game_odds(game_id: str, game_info: dict):
    """
    Analizza una partita usando sia lo storico quote che i dati di consenso.
    Implementa un'analisi più sofisticata correlando multiple fonti di informazione.
    """
    print(f"🔍 Analizzando: {game_info.get('away_team', 'Away')} @ {game_info.get('home_team', 'Home')}")

    # Preferisci Pinnacle come bookmaker di riferimento (più affidabile per RLM)
    preferred_bookmaker = 'pinnacle'
    df_odds = get_odds_history_for_game(game_id=game_id, bookmaker=preferred_bookmaker)
    
    # Se Pinnacle non disponibile, usa il primo bookmaker disponibile
    if df_odds.empty:
        for bookmaker in TARGET_BOOKMAKERS:
            df_odds = get_odds_history_for_game(game_id=game_id, bookmaker=bookmaker)
            if not df_odds.empty:
                preferred_bookmaker = bookmaker
                break
    
    if df_odds.empty:
        print(f"⚠️ Nessun dato quote trovato per {game_id}")
        return

    # Recupera dati di consenso
    df_consensus = get_consensus_history(game_id)
    
    # Applica le strategie di analisi
    rlm_reason = detect_true_reverse_line_movement(df_odds, df_consensus)
    steam_reason = detect_steam_moves(df_odds)
    
    # Se troviamo segnali significativi, prepara la notifica
    primary_reason = rlm_reason or steam_reason
    
    if primary_reason:
        latest_quote = df_odds.iloc[-1]
        
        # Determina il tipo di scommessa raccomandato
        if "OVER" in primary_reason.upper():
            bet_type = "Over"
            odds = latest_quote['odds_over']
        else:
            bet_type = "Under" 
            odds = latest_quote['odds_under']
        
        # Calcola confidenza e stake
        confidence = calculate_confidence_score(rlm_reason, steam_reason, df_consensus)
        
        # Suggerimento stake basato sulla confidenza
        if "MOLTO ALTA" in confidence:
            stake_suggestion = "2.0 Unità (Max)"
        elif "ALTA" in confidence:
            stake_suggestion = "1.5 Unità"
        elif "MEDIA" in confidence:
            stake_suggestion = "1.0 Unità"
        else:
            stake_suggestion = "0.5 Unità (Cauto)"
        
        # Componi il messaggio completo
        full_reason = primary_reason
        if rlm_reason and steam_reason:
            full_reason += f"\n\n🔥 CONFERMA AGGIUNTIVA: {steam_reason}"
        
        alert_message = format_bet_alert(
            game=game_info,
            bet_type=bet_type,
            line=latest_quote['line'],
            odds=odds,
            reason=full_reason,
            confidence=confidence,
            stake_suggestion=stake_suggestion
        )
        
        # Aggiungi info sul bookmaker e fonte dati
        alert_message += f"\n\n📊 Bookmaker analizzato: {preferred_bookmaker.upper()}"
        if not df_consensus.empty:
            alert_message += f"\n📈 Dati consenso: Disponibili"
        else:
            alert_message += f"\n⚠️ Dati consenso: Non disponibili (analisi basata solo su quote)"
        
        send_telegram_notification(alert_message)
        print(f"🚨 ALERT INVIATO per {game_id}: {bet_type} {latest_quote['line']}")
        return True
    
    print(f"✅ Nessuna opportunità rilevata per {game_id}")
    return False

def run_analysis_for_all_games():
    """
    Esegue l'analisi per tutte le partite attive nel database.
    """
    print("🎯 Avvio analisi per tutte le partite attive...")
    
    active_games = get_all_active_games()
    alerts_sent = 0
    
    for game_id in active_games:
        # Recupera informazioni di base sulla partita
        df_game_info = get_odds_history_for_game(game_id, bookmaker=None)
        if not df_game_info.empty:
            latest_info = df_game_info.iloc[-1]
            game_info = {
                'id': game_id,
                'home_team': latest_info['home_team'],
                'away_team': latest_info['away_team']
            }
            
            if analyze_game_odds(game_id, game_info):
                alerts_sent += 1
    
    print(f"📱 Analisi completata: {alerts_sent} alert inviati su {len(active_games)} partite")

if __name__ == '__main__':
    # Test del modulo
    run_analysis_for_all_games() 