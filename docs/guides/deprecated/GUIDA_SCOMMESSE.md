# 🎯 GUIDA COMPLETA - Sistema Gestione Scommesse NBA

## 📋 Come funziona il sistema

### 1. 💾 **Salvataggio Scommesse**

Quando selezioni una scommessa dal sistema principale, viene automaticamente salvata in:
```
data/pending_bets.json
```

**Struttura del file:**
```json
[
  {
    "bet_id": "CUSTOM_Thunder_Pacers_OVER_210.5",
    "game_id": "CUSTOM_Thunder_Pacers", 
    "bet_data": {
      "type": "OVER",
      "line": 210.5,
      "odds": 1.95,
      "stake": 3.6,
      "edge": 0.363,
      "probability": 0.699,
      "quality_score": 0.4
    },
    "timestamp": "2025-06-17T19:45:30.123456",
    "status": "pending"
  }
]
```

### 2. 👀 **Visualizzazione Scommesse**

**Comando base:**
```bash
python bet_manager.py
# oppure
python bet_manager.py --view
```

**Output esempio:**
```
🎯 SCOMMESSE SALVATE (2 totali)
================================================================================

⏳ SCOMMESSE PENDENTI (1)
--------------------------------------------------

1. 🎲 OVER 210.5 @ 1.95
   📅 Data: 17/06/2025 19:45
   🏀 Partita: CUSTOM_Thunder_Pacers
   💰 Stake: €3.60
   📊 Edge: 36.3% | Prob: 69.9%

✅ SCOMMESSE COMPLETATE (1)
--------------------------------------------------

1. 🎲 UNDER 225.0 @ 2.10
   📅 Data: 16/06/2025 18:30
   🏀 Partita: CUSTOM_Lakers_Warriors
   💰 Stake: €2.50
   📊 Edge: 15.2% | Prob: 55.4%
   🎯 Risultato: 🟢 VINTA | P&L: +€2.75
   📈 Totale Reale: 218
```

### 3. 🔄 **Sistema Anti-Duplicati e Sovrascrittura**

**Quando salvi una nuova scommessa per la stessa partita:**

Il sistema rileva automaticamente se esiste già una scommessa per lo stesso `game_id` e ti chiede:

```
⚠️  SCOMMESSA ESISTENTE TROVATA per CUSTOM_Thunder_Pacers:
   ATTUALE: OVER 210.5 @ 1.95 (€3.60)
   NUOVA:   OVER 215.0 @ 2.20 (€2.80)

Cosa vuoi fare?
1. Sostituisci scommessa esistente
2. Aggiungi come nuova scommessa  
3. Annulla
Scelta (1/2/3): 
```

**Opzioni:**
- **1. Sostituisci**: La vecchia scommessa viene completamente sostituita
- **2. Aggiungi**: Entrambe le scommesse vengono mantenute
- **3. Annulla**: Non viene salvato nulla

### 4. 🗑️ **Eliminazione Scommesse**

**Elimina scommessa specifica:**
```bash
python bet_manager.py --delete 1
```

**Pulisci tutte le scommesse completate:**
```bash
python bet_manager.py --clean
```

### 5. 🔍 **Ricerca per Partita**

**Trova scommesse per una partita specifica:**
```bash
python bet_manager.py --game-id "CUSTOM_Thunder_Pacers"
```

### 6. 🔄 **Controllo Automatico Risultati**

**Il sistema controlla automaticamente i risultati:**
```bash
python main.py --check-bets
```

**Cosa fa:**
- Controlla tutte le scommesse con status "pending"
- Recupera i risultati delle partite finite
- Aggiorna automaticamente il bankroll
- Cambia lo status a "completed"

### 7. 📊 **Flusso Completo di Utilizzo**

#### **Passo 1: Fai un pronostico**
```bash
python main.py --team1 "Thunder" --team2 "Pacers" --line 225.0
```

#### **Passo 2: Seleziona e salva scommessa**
- Il sistema mostra le raccomandazioni categorizzate
- Selezioni il numero della raccomandazione (1-N)
- Confermi il salvataggio

#### **Passo 3: Visualizza scommesse salvate**
```bash
python bet_manager.py
```

#### **Passo 4: (Opzionale) Cambia idea**
- Rifai il pronostico per la stessa partita
- Il sistema rileva la scommessa esistente
- Scegli se sostituire o aggiungere

#### **Passo 5: Controlla risultati**
```bash
python main.py --check-bets
```

### 8. 🛡️ **Gestione Errori e Sicurezza**

**Il sistema gestisce automaticamente:**
- ✅ File JSON corrotti o mancanti
- ✅ Duplicati per la stessa partita
- ✅ Backup automatico prima delle modifiche
- ✅ Validazione dei dati di input
- ✅ Rollback in caso di errori

**Backup automatico:**
Ogni volta che modifichi le scommesse, il sistema mantiene:
- `original_bet`: Dati della scommessa originale (se sostituita)
- `replaced_at`: Timestamp della sostituzione
- `timestamp`: Timestamp originale della scommessa

### 9. 📁 **Struttura File**

```
data/
├── pending_bets.json     # Scommesse pendenti e completate
├── bankroll.json         # Stato del bankroll
└── system_settings.json  # Impostazioni sistema
```

### 10. 🚀 **Comandi Rapidi**

```bash
# Visualizza tutte le scommesse
python bet_manager.py

# Fai un nuovo pronostico
python main.py --team1 "Lakers" --team2 "Warriors"

# Controlla risultati
python main.py --check-bets

# Elimina scommessa numero 2
python bet_manager.py --delete 2

# Pulisci scommesse completate
python bet_manager.py --clean

# Cerca scommesse per partita
python bet_manager.py --game-id "CUSTOM_Lakers_Warriors"
```

### 11. 💡 **Consigli d'Uso**

1. **Controlla regolarmente** le scommesse pendenti con `python bet_manager.py`
2. **Usa --check-bets** dopo ogni giornata di partite per aggiornare il bankroll
3. **Pulisci periodicamente** le scommesse completate con `--clean`
4. **Fai backup** del file `data/pending_bets.json` prima di modifiche importanti
5. **Usa la sostituzione** quando cambi idea su una scommessa, non aggiungere duplicati

---

## 🎯 **Esempio Pratico Completo**

```bash
# 1. Fai pronostico Thunder vs Pacers
python main.py --team1 "Thunder" --team2 "Pacers" --line 225.0
# Selezioni raccomandazione 1 (SCELTA DEL SISTEMA)
# Confermi il salvataggio

# 2. Visualizza scommesse
python bet_manager.py
# Vedi: 1. OVER 225.5 @ 2.10 (€3.20)

# 3. Cambi idea, rifai il pronostico
python main.py --team1 "Thunder" --team2 "Pacers" --line 220.0  
# Sistema rileva scommessa esistente
# Scegli "1" per sostituire
# Confermi nuova scommessa

# 4. Controlla risultati dopo la partita
python main.py --check-bets
# Sistema aggiorna automaticamente bankroll

# 5. Pulisci scommesse completate
python bet_manager.py --clean
```

Il sistema è progettato per essere **sicuro**, **intuitivo** e **completo**! 🎯 