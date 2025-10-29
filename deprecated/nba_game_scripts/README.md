# Deprecated NBA Game Download Scripts

## ⚠️ DEPRECATED - NON UTILIZZARE QUESTI SCRIPT

Questi script sono stati deprecati perché **non funzionano correttamente** per il download delle partite NBA programmate.

## Script Deprecati

### 1. `get_todays_nba_games_simple.py`
- **Problema**: Tentava di usare DataPersistenceBridge ma con implementazione incompleta
- **Status**: DEPRECATED - Non utilizzare

### 2. `get_todays_nba_games_test.py`
- **Problema**: Solo legge dal data store senza scaricare dati nuovi
- **Status**: DEPRECATED - Non utilizzare

### 3. `nba_timezone_utils.py`
- **Problema**: La funzione `get_nba_games_official_api()` restituisce dati **FALSI**
- **Problema**: Restituisce team ID numerici invece di nomi reali dei team
- **Problema**: Dati di test invece di partite reali programmate
- **Status**: DEPRECATED - Non utilizzare per scaricare partite
- **Nota**: Le funzioni timezone possono ancora essere utilizzate, ma non le funzioni di download partite

## ✅ METODO CORRETTO PER SCARICARE LE PARTITE NBA

Usare **esclusivamente** il sistema `DataPersistenceBridge`:

```python
import sys
import os
sys.path.append('src')

from nba_predictor.core.data_persistence_bridge import initialize_persistence_bridge, get_persistence_bridge, close_persistence_bridge
from datetime import date

# Inizializza il bridge
bridge = initialize_persistence_bridge()

# Scarica le partite di oggi
today = date.today()
games = bridge.get_scheduled_games_with_persistence(
    days_ahead=1,
    specific_date=today_str,
    force_api=True  # Forza download fresco
)

# Chiudi il bridge
close_persistence_bridge()
```

## 🔍 Dove Vengono Salvati i Dati

I dati reali delle partite vengono salvati in:
```
data/persistent/games/games_YYYY-MM-DD.parquet
```

Esempio: `data/persistent/games/games_2025-10-28.parquet`

## 🏀 Partite Reali di Oggi (28 Ottobre 2025)

Le partite reali disponibili nel data store:
1. Philadelphia 76ers vs Washington Wizards
2. Charlotte Hornets vs Miami Heat
3. New York Knicks vs Milwaukee Bucks
4. Sacramento Kings vs Oklahoma City Thunder
5. LA Clippers vs Golden State Warriors

## 📋 Perché Questi Script Sono Stati Deprecati

1. **Dati Falsi**: Molti script restituivano team ID (es. "1610612766") invece di nomi reali
2. **Dati Passati**: Alcuni script restituivano partite già completate invece di partite programmate
3. **Implementazioni Multiple**: Troppi script che facevano la stessa cosa in modo errato
4. **Confusione**: Impossibile capire quale script fosse quello corretto

## 🎯 Regola d'Oro

**Usare SEMPRE e SOLO DataPersistenceBridge per scaricare le partite NBA**

Qualsiasi altro script che promette di scaricare partite NBA è probabilmente errato o deprecato.

---
*Aggiornato il 28 Ottobre 2025 - Sistema Unified Hybrid NBA Prediction*