# 💰 Sistema di Aggiornamento Automatico Bankroll

## 🔄 **Come Funziona**

Il sistema ora offre **tre modalità** per gestire il bankroll:

### **1. 📲 Salvataggio Automatico Scommesse**
Quando selezioni una scommessa, il sistema:
- ✅ Salva automaticamente la scommessa in `data/pending_bets.json`
- 🎯 Associa la scommessa al `game_id` della partita NBA
- ⏳ Marca la scommessa come "pending" (in attesa di risultato)

### **2. 🤖 Recupero Automatico Risultati**
Il sistema può recuperare automaticamente i risultati tramite **NBA API**:
- 🔍 Controlla lo stato della partita (`GAME_STATUS_ID`)
- 📊 Recupera i punteggi finali quando `status = COMPLETED`
- 💰 Aggiorna automaticamente il bankroll (profit/loss)

### **3. 🕐 Controlli Periodici Automatici**
Esegui controlli automatici programmati per tutte le scommesse pendenti.

---

## 🎮 **Come Usare il Sistema**

### **Opzione 1: Controllo Manuale Singolo**
```bash
# Controlla una volta tutte le scommesse pendenti
python main.py --check-bets
```

### **Opzione 2: Script Dedicato**
```bash
# Controllo singolo
python bankroll_updater.py

# Modalità daemon (controllo ogni ora)
python bankroll_updater.py --daemon

# Modalità daemon personalizzata (ogni 30 minuti)
python bankroll_updater.py --daemon --interval 30
```

### **Opzione 3: Durante Analisi Partita**
Quando analizzi una partita e selezioni una scommessa:
1. Il sistema salva automaticamente la scommessa
2. Ti chiede se conosci già il risultato
3. Se no, aggiornerà automaticamente in futuro

---

## 📁 **File e Struttura Dati**

### **Bankroll File:**
- `bankroll.json` - File principale del bankroll
- `data/bankroll.json` - Backup automatico

### **Scommesse Pendenti:**
- `data/pending_bets.json` - Lista scommesse in attesa di risultato

**Esempio struttura scommessa:**
```json
{
  "bet_id": "0022400123_OVER_220.5",
  "game_id": "0022400123", 
  "bet_data": {
    "type": "OVER",
    "line": 220.5,
    "odds": 1.90,
    "stake": 25.0
  },
  "timestamp": "2024-01-15T20:30:00",
  "status": "pending"
}
```

### **Scommessa Completata:**
```json
{
  "status": "completed",
  "result": {
    "actual_total": 225,
    "bet_won": true,
    "profit_loss": 22.5,
    "completed_at": "2024-01-16T02:45:00"
  }
}
```

---

## 🎯 **Flusso Completo Esempio**

### **1. Analisi e Scommessa**
```bash
python main.py --team1 Lakers --team2 Warriors --line 220.5
```
- Sistema analizza la partita
- Mostra opportunità value betting
- Utente seleziona: "OVER 220.5 @ 1.90 (Stake: €25.0)"
- **Sistema salva automaticamente** in `pending_bets.json`

### **2. Controllo Automatico**
```bash
python bankroll_updater.py --daemon
```
- Script gira in background
- Ogni ora controlla tutte le scommesse pendenti
- Recupera risultati tramite NBA API
- Aggiorna bankroll automaticamente

### **3. Risultato Partita**
```
🔍 Controllo partita 0022400123...
✅ Risultato: 110 - 115 (Totale: 225)
🟢 SCOMMESSA VINTA! Profit: €22.50
💰 Bankroll aggiornato e salvato: €122.50
```

---

## ⚙️ **Configurazione Avanzata**

### **Modalità Daemon Personalizzata**
```bash
# Controllo ogni 15 minuti
python bankroll_updater.py --daemon --interval 15

# Controllo ogni 2 ore  
python bankroll_updater.py --daemon --interval 120
```

### **Integrazione con Cron (Linux/Mac)**
```bash
# Aggiungi al crontab per controllo ogni ora
0 * * * * cd /path/to/nba-predictor && python bankroll_updater.py
```

### **Integrazione con Task Scheduler (Windows)**
- Apri Task Scheduler
- Crea attività di base
- Programma: `python bankroll_updater.py`
- Frequenza: ogni ora

---

## 🚨 **Limitazioni e Note Importanti**

### **Rate Limiting NBA API**
- Il sistema rispetta i limiti API (pausa 0.6s tra chiamate)
- Non sovraccaricare con controlli troppo frequenti

### **Affidabilità Game ID**
- Funziona solo con partite che hanno un `game_id` valido
- Fallback manuale disponibile per partite senza ID

### **Connessione Internet**
- Richiede connessione stabile per recuperare risultati
- Errori di rete vengono gestiti gracefully

### **Backup Bankroll**
- Sempre mantenere backup dei file bankroll
- Sistema crea automaticamente backup in `data/`

---

## 🛠️ **Troubleshooting**

### **Problema: Scommessa non aggiornata**
```bash
# Controllo manuale debug
python main.py --check-bets
```

### **Problema: API NBA non risponde**  
- Verifica connessione internet
- Riprova dopo alcuni minuti
- Aggiorna manualmente se necessario

### **Problema: File corrotti**
```bash
# Ripristina da backup
cp data/bankroll.json bankroll.json
```

---

## 📊 **Monitoraggio Performance**

Il sistema logga tutte le operazioni:
- ✅ Scommesse salvate
- 🔍 Controlli eseguiti  
- 💰 Aggiornamenti bankroll
- ❌ Errori e fallimenti

Usa questi log per monitorare l'efficacia del sistema automatico.

---

**🎯 Il sistema ora è completamente automatizzato!** 

Seleziona le scommesse durante l'analisi e lascia che il sistema aggiorni automaticamente il bankroll quando le partite finiscono. 🚀 