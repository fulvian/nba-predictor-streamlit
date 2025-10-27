# 🏀 NBA Predictor - Sistema di Monitoraggio Quote Avanzato

## 📋 Panoramica

Questo sistema di monitoraggio avanzato integra **multiple API** per massimizzare la copertura delle quote NBA e analizza i **dati di consenso pubblico/sharp** per rilevare opportunità di scommessa di alto valore basate su strategie verificate come il **Reverse Line Movement (RLM)**.

### 🎯 Caratteristiche Principali

- **🔄 Multi-API Integration**: The Odds API, TheRundown, API-Sports
- **📊 Analisi del Consenso**: Correlazione Public vs Sharp Money  
- **🚨 RLM Detection**: Rilevamento Reverse Line Movement avanzato
- **💨 Steam Moves**: Identificazione movimenti rapidi di quote
- **📱 Notifiche Telegram**: Alert dettagliati in tempo reale
- **🗄️ Persistenza Dati**: Storicizzazione completa di quote e consenso
- **⚡ Orchestrazione Automatica**: Scheduler intelligente con gestione errori

## 🏗️ Architettura

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Sources   │    │   Data Layer    │    │  Analysis Layer │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • The Odds API  │───▶│ • odds_history  │───▶│ • RLM Detection │
│ • TheRundown    │    │ • consensus_    │    │ • Steam Moves   │
│ • API-Sports    │    │   history       │    │ • Value Spots   │
│ • Consensus Src │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Database Manager│    │ Telegram Alerts │
                       └─────────────────┘    └─────────────────┘
```

## 🚀 Installazione e Configurazione

### 1. Dipendenze

```bash
pip install -r requirements.txt
```

### 2. Configurazione Environment Variables

Crea un file `.env` nella root del progetto:

```env
# API Keys (almeno una obbligatoria)
THE_ODDS_API_KEY='your_odds_api_key_here'
THERUNDOWN_API_KEY='your_therundown_api_key_here'
API_SPORTS_KEY='your_apisports_key_here'

# Telegram (obbligatorio)
TELEGRAM_BOT_TOKEN='your_telegram_bot_token'
TELEGRAM_CHAT_ID='your_telegram_chat_id'

# Configurazione (opzionale)
DATABASE_URL='sqlite:///market_movements.db'
POLLING_INTERVAL_MINUTES='15'
MARKET_TO_MONITOR='totals'
TARGET_REGIONS='eu'
TARGET_BOOKMAKERS='pinnacle,betfair,draftkings'
```

### 3. Setup Telegram Bot

1. Contatta [@BotFather](https://t.me/botfather) su Telegram
2. Crea un nuovo bot con `/newbot`
3. Ottieni il token e inseriscilo in `TELEGRAM_BOT_TOKEN`
4. Per ottenere il Chat ID:
   - Invia un messaggio al tuo bot
   - Vai su: `https://api.telegram.org/bot<TOKEN>/getUpdates`
   - Trova il `"chat":{"id": NUMBER}`

### 4. Ottenimento API Keys

#### The Odds API (Primaria)
- Registrati su [the-odds-api.com](https://the-odds-api.com/)
- Piano gratuito: 500 requests/mese
- Supporta NBA totals con bookmaker multipli

#### TheRundown API (Alternativa)
- Registrati su [RapidAPI](https://rapidapi.com/therundown/api/therundown)
- Spesso più generoso con i limiti
- Ottima per aumentare la copertura

#### API-Sports (Integrativa)
- Registrati su [api-sports.io](https://api-sports.io/)
- Ampia copertura di sport e mercati

## 🎮 Utilizzo

### Avvio del Sistema

```bash
python odds_monitor_service.py
```

### Test dei Moduli Singolarmente

```bash
# Test database
python database_manager.py

# Test ingestione dati
python data_ingestion_worker.py

# Test analisi
python odds_analysis_engine.py

# Test notifiche
python notification_manager.py
```

### Monitoraggio Logs

Il sistema fornisce log dettagliati con emoji per facilità di lettura:

- 🚀 Avvio cicli
- 📡 Fetch API
- ✅ Successi
- ❌ Errori
- 🚨 Alert rilevati
- 📱 Notifiche inviate

## 🔬 Sistema di Analisi

### Reverse Line Movement (RLM)

Il sistema rileva RLM **vero** correlando:

1. **Movimenti di Linea**: Variazioni della linea totale (es. da 215.5 a 218.5)
2. **Dati di Consenso**: % di scommesse pubbliche vs. direzione della linea
3. **Sharp Indicators**: Segnali di attività da sharp bettors

**Esempio di RLM Forte:**
```
🚨 RLM FORTE sull'UNDER: Linea scesa da 220.5 a 218.5 (-2.0) 
nonostante 78.3% del denaro pubblico e 72.1% dei ticket 
siano sull'OVER. Sharp money chiaramente sull'UNDER.
```

### Steam Moves

Rileva movimenti rapidi di quota (>0.10 in poco tempo) che indicano:
- Scommesse di grandi dimensioni
- Informazioni privilegiate
- Correzioni di mercato

### Scoring di Confidenza

Il sistema assegna livelli di confidenza basati su:
- **MOLTO ALTA**: RLM + Dati consenso + Steam move
- **ALTA**: RLM con dati consenso OR multiple conferme
- **MEDIA**: Singolo segnale forte
- **BASSA**: Segnali deboli o contraddittori

## 📊 Database Schema

### Tabella `odds_history`
```sql
- game_id (String, Index)
- sport_key, home_team, away_team
- bookmaker, market_key
- line, odds_over, odds_under
- api_source (theoddsapi/therundown/apisports)
- timestamp
```

### Tabella `betting_consensus_history`
```sql
- game_id (String, Index)
- source (SportsInsights/VSIN/ActionNetwork)
- public_tickets_percentage_over
- public_money_percentage_over
- sharp_money_indicator
- reverse_line_movement_detected
- steam_move_detected
- line_at_time
- timestamp
```

## 🎯 Strategia di Gestione API

### Orchestrazione Intelligente

1. **TheRundown** (Prima scelta): Limiti più generosi
2. **The Odds API** (Integrazione): Dati più affidabili  
3. **API-Sports** (Completamento): Copertura aggiuntiva

### Gestione Rate Limits

- Deduplicazione automatica dei dati
- Fallback tra API in caso di esaurimento limiti
- Rispetto dei rate limits con timeout appropriati

## 📱 Sistema di Notifiche

### Tipi di Alert

1. **🚨 Alert Scommesse**: Opportunità rilevate
2. **🤖 Status Sistema**: Avvio/arresto/errori
3. **📊 Statistiche**: Riassunti periodici

### Formato Notifiche

Le notifiche includono:
- **Emoji descrittivi** per tipo di segnale
- **Dettagli tecnici** dell'analisi
- **Suggerimenti stake** calibrati sulla confidenza
- **Timestamp** e fonte dati
- **Disclaimer** responsabilità

## ⚙️ Personalizzazione

### Modifica Intervalli

```env
POLLING_INTERVAL_MINUTES='10'  # Controllo ogni 10 minuti
```

### Filtri Bookmaker

```env
TARGET_BOOKMAKERS='pinnacle,betfair'  # Solo bookmaker sharp
```

### Soglie di Rilevamento

Nel file `odds_analysis_engine.py`, modifica:
```python
# Soglia RLM (default: 1.0 punto di linea)
if line_change < -1.0 and public_money_on_over > 70:

# Soglia Steam Move (default: 0.10 di quota)
if over_change <= -0.10:
```

## 🛠️ Troubleshooting

### Errori Comuni

1. **API Key mancanti**
   - Verifica che almeno una API key sia configurata
   - Controlla validità delle key

2. **Telegram non funziona**
   - Verifica BOT_TOKEN e CHAT_ID
   - Assicurati che il bot sia avviato

3. **Nessun dato**
   - Controlla connessione internet
   - Verifica limiti API non esauriti
   - Controlla se ci sono partite NBA in programma

4. **Database errors**
   - Elimina `market_movements.db` e riavvia
   - Verifica permessi di scrittura

### Debug Mode

Per debug dettagliato, aggiungi al tuo `.env`:
```env
DEBUG=true
```

## 📈 Roadmap

- [ ] **Machine Learning**: Predizione accuratezza RLM
- [ ] **Web Dashboard**: Interfaccia web per monitoraggio
- [ ] **Backtesting**: Sistema di test storici
- [ ] **Kelly Criterion**: Calcolo stake ottimale automatico
- [ ] **Multi-Sport**: Estensione ad altri sport
- [ ] **Alert Filters**: Filtri personalizzabili per alert

## 🤝 Contributi

Questo sistema è modulare e extensibile. Per contribuire:

1. Fork del repository
2. Crea feature branch
3. Implementa modifiche
4. Test approfonditi
5. Pull request

## ⚠️ Disclaimer

Questo sistema è per scopi educativi e di ricerca. Le scommesse comportano rischi finanziari. Scommetti sempre responsabilmente e nei limiti delle tue possibilità economiche.

## 📞 Supporto

Per supporto tecnico o domande:
- Apri un issue su GitHub
- Controlla la documentazione dei singoli moduli
- Verifica i log di sistema per errori dettagliati 