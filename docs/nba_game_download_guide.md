# 🏀 Guida Ufficiale - Download Partite NBA Programmate

## ✅ METODO CORRETTO E UNICO PER SCARICARE PARTITE NBA

**Questa è la guida ufficiale e l'unico metodo supportato per scaricare le partite NBA programmate.**

## 🎯 Sistema DataPersistenceBridge

Il sistema usa `DataPersistenceBridge` che integra:
- API ufficiale NBA.com per dati reali
- Data store persistente con caching
- Gestione automatica dei timezone
- Validazione dei dati

## 📄 Script Ufficiale

### `get_todays_nba_games_official.py`

```python
#!/usr/bin/env python3
"""
🏀 GET TODAY'S NBA GAMES - Official Script using Data Persistence Bridge
Script ufficiale per scaricare le partite NBA programmate nel data store.
"""

import sys
import os
sys.path.append('src')

# Fix import paths
os.chdir('/Users/fulvioventura/nba-predictor-streamlit')

try:
    from nba_predictor.core.data_persistence_bridge import initialize_persistence_bridge, get_persistence_bridge
    from nba_predictor.core.data_persistence_bridge import close_persistence_bridge
    from datetime import date

    print('🏀 GET TODAY\'S NBA GAMES - DATA PERSISTENCE BRIDGE')
    print('=' * 60)

    # Initialize the persistence bridge
    print('🔄 Initializing persistence bridge...')
    bridge = initialize_persistence_bridge()

    if bridge:
        today = date.today()
        today_str = today.strftime('%Y-%m-%d')

        print(f'📅 Fetching games for: {today_str}')
        print()

        # Get today's games with persistence (this will download if needed)
        games = bridge.get_scheduled_games_with_persistence(
            days_ahead=1,
            specific_date=today_str,
            force_api=True  # Force fresh download
        )

        if games:
            print(f'✅ SUCCESS: Found {len(games)} NBA games for today!')
            print()

            for i, game in enumerate(games, 1):
                home_team = game.get('home_team', 'N/D')
                away_team = game.get('away_team', 'N/D')
                game_time = game.get('game_time', 'N/D')
                status = game.get('status', 'N/D')
                source = game.get('source', 'N/D')

                print(f'{i}. {away_team} vs {home_team}')
                print(f'   ⏰ Time: {game_time}')
                print(f'   📊 Status: {status}')
                print(f'   🌐 Source: {source}')
                print()

            print('🎯 These games have been saved to the persistent data store!')
            print(f'📁 Location: data/persistent/games/games_{today_str}.parquet')

        else:
            print('❌ No games found for today')

        # Close the bridge
        close_persistence_bridge()
        print()
        print('✅ Persistence bridge closed successfully')

    else:
        print('❌ Failed to initialize persistence bridge')

except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
```

## 🗂️ Dove Vengono Salvati i Dati

I dati delle partite vengono salvati automaticamente in:

```
data/persistent/games/games_YYYY-MM-DD.parquet
```

Esempi:
- `data/persistent/games/games_2025-10-28.parquet`
- `data/persistent/games/games_2025-10-29.parquet`

## 🏀 Partite Reali di Oggi (28 Ottobre 2025)

```
📅 Date: 2025-10-28
📁 File: data/persistent/games/games_2025-10-28.parquet

📋 Scheduled Games:
1. Philadelphia 76ers vs Washington Wizards
   🏟️ Capital One Arena, Washington DC
   ⏰ 19:00 ET

2. Charlotte Hornets vs Miami Heat
   🏟️ Kaseya Center, Miami
   ⏰ 19:30 ET

3. New York Knicks vs Milwaukee Bucks
   🏟️ Madison Square Garden, New York
   ⏰ 19:30 ET

4. Sacramento Kings vs Oklahoma City Thunder
   🏟️ Paycom Center, Oklahoma City
   ⏰ 20:00 ET

5. LA Clippers vs Golden State Warriors
   🏟️ Crypto.com Arena, Los Angeles
   ⏰ 22:30 ET
```

## 🔧 Come Funziona il Sistema

### 1. DataPersistenceBridge Architecture
```
DataPersistenceBridge
├── initialize_persistence_bridge()  # Inizializzazione
├── get_scheduled_games_with_persistence()  # Download con cache
├── close_persistence_bridge()  # Cleanup
└── Data persistence layer
    ├── API calls (NBA.com)
    ├── Data validation
    ├── Parquet storage
    └── Caching system
```

### 2. Flusso di Download
1. **Initialize**: `initialize_persistence_bridge()` setup del sistema
2. **Download**: `get_scheduled_games_with_persistence()` scarica se necessario
3. **Cache**: Dati salvati in file parquet per accessi futuri
4. **Validation**: Dati validati e formattati correttamente
5. **Cleanup**: `close_persistence_bridge()` risorse rilasciate

### 3. Opzioni Utili
```python
games = bridge.get_scheduled_games_with_persistence(
    days_ahead=7,           # Quanti giorni futuri scaricare
    specific_date="2025-10-28",  # Data specifica (opzionale)
    force_api=True          # Forza download nuovo (ignora cache)
)
```

## ⚠️ Regole Importanti

### ✅ COSA FARE
- **Usare SEMPRE** DataPersistenceBridge
- **Verificare** che i dati contengano nomi reali dei team
- **Controllare** che le partite siano programmate (status != "Final")
- **Usare** il path completo `src/nba_predictor/core/data_persistence_bridge`

### ❌ COSA NON FARE
- **NON usare** script nella cartella `deprecated/`
- **NON usare** funzioni che restituiscono team ID numerici
- **NON creare** nuovi script per scaricare partite
- **NON usare** direttamente API NBA.com senza DataPersistenceBridge

## 🚨 Script Deprecati da NON Utilizzare

⚠️ **NON UTILIZZARE QUESTI SCRIPT**:

- `deprecated/nba_game_scripts/get_todays_nba_games_simple.py`
- `deprecated/nba_game_scripts/get_todays_nba_games_test.py`
- `deprecated/nba_game_scripts/nba_timezone_utils.py` (solo funzioni di download)
- `deprecated/data_provider_june2025.py`
- `deprecated/data_provider_hybrid.py`

Questi script contengono errori, dati falsi, o implementazioni incomplete.

## 🔍 Verifica dei Dati

Dopo aver scaricato i dati, verificare che:

1. **Nomi Team Reali**: "Boston Celtics", non "1610612740"
2. **Status Corretto**: "Scheduled", non "Final"
3. **Date Corrette**: Partite future, non passate
4. **Dati Completi**: home_team, away_team, game_time, venue

## 📞 Supporto

In caso di problemi con il download delle partite:

1. **Verificare** di usare lo script ufficiale
2. **Controllare** la connessione internet
3. **Verificare** che il file parquet venga creato
4. **Controllare** i log per eventuali errori API

---
*Guida ufficiale - Unified Hybrid NBA Prediction Pipeline*
*Aggiornato il 28 Ottobre 2025*