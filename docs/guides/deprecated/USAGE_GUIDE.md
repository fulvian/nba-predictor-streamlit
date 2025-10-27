# 🏀 Guida d'Uso NBA Predictor Aggiornato

## 📋 **Comandi Principali**

### **🎯 Analisi Partita Specifica**
```bash
# Analizza partita specifica con linea centrale
python main.py --team1 Lakers --team2 Warriors --line 220.5

# Analisi senza linea (usa quote generate automaticamente)  
python main.py --team1 Celtics --team2 Heat
```

### **📅 Analisi Partite del Calendario**
```bash
# Mostra partite programmate e permette di scegliere
python main.py

# Modalità automatica (analizza prima partita disponibile)
python main.py --auto-mode
```

### **💰 Gestione Bankroll**
```bash
# Controlla scommesse pendenti (controllo singolo)
python main.py --check-bets

# Avvia monitoraggio automatico continuo (ogni ora)
python bankroll_updater.py --daemon

# Monitoraggio personalizzato (ogni 30 minuti)
python bankroll_updater.py --daemon --interval 30
```

---

## 🔄 **Migrazione dai Vecchi Comandi**

### **Comando Vecchio → Nuovo**

| **VECCHIO** | **NUOVO** | **Note** |
|-------------|-----------|----------|
| `--giorni 4` | `python main.py` | Il sistema ora mostra partite disponibili |
| `--linea-centrale 225.0` | `--line 225.0` | Parametro rinominato |
| `--data 2024-01-15` | `python main.py` | Seleziona dalla lista partite |

### **Esempi di Migrazione**

#### **Prima:**
```bash
python main.py --giorni 4 --linea-centrale 225.0
```

#### **Ora:**
```bash
# Opzione 1: Analisi partita specifica
python main.py --team1 Lakers --team2 Warriors --line 225.0

# Opzione 2: Scegli dal calendario
python main.py
# > Scegli partita dalla lista mostrata
```

---

## 🎮 **Flusso di Utilizzo Completo**

### **1. Analisi Partita**
```bash
python main.py --team1 Lakers --team2 Warriors --line 220.5
```

**Output:**
```
🎯 Analisi partita personalizzata: Lakers @ Warriors
🏀 ANALISI PARTITA: Lakers @ Warriors
================================================================================
1. Acquisizione statistiche squadre...
2. Acquisizione roster giocatori...
3. Analisi impatto infortuni...
4. Calcolo momentum giocatori...
5. Esecuzione modello probabilistico...
6. Analisi opportunità scommesse...

🎯 RIEPILOGO FINALE
================================================================================
🏀 Partita: Lakers @ Warriors
📊 Totale Predetto: 218.3 punti
📈 Confidenza (σ): 12.1
⚡ Momentum Impact: +2.3 punti
💰 Bankroll Attuale: €100.00

💎 OPPORTUNITÀ VALUE BETTING (2 trovate)
--------------------------------------------------------------------------------
🏆 [1] OVER 220.5 @ 1.90 (Edge: 8.2%, Stake: €15.0)
   [2] UNDER 216.5 @ 1.85 (Edge: 5.1%, Stake: €10.0)

🎮 SELEZIONE SCOMMESSA
================================================================================
Scegli una scommessa inserendo il numero corrispondente:
🏆 [1] OVER 220.5 @ 1.90 (Stake: €15.0)
   [2] UNDER 216.5 @ 1.85 (Stake: €10.0)
[0] Nessuna scommessa

Inserisci il numero della tua scelta: 1

✅ Hai selezionato: OVER 220.5 @ 1.90
💰 Stake: €15.0
📲 Scommessa salvata! Il sistema aggiornerà automaticamente il bankroll quando la partita finirà.

Conosci già il risultato della partita? (y/n): n
```

### **2. Monitoraggio Automatico**
```bash
# Terminal separato per monitoraggio continuo
python bankroll_updater.py --daemon
```

**Output:**
```
🤖 Avvio modalità daemon - Controllo automatico ogni 60 minuti
   Premi Ctrl+C per interrompere

============================================================
🕐 2024-01-15 22:00:00 - Controllo Automatico Bankroll
============================================================
🔄 Controllo 1 scommesse pendenti...

🎯 Controllo partita CUSTOM_Lakers_Warriors...
🔍 Recupero risultato automatico per game_id: CUSTOM_Lakers_Warriors
✅ Risultato: 112 - 109 (Totale: 221)
🟢 SCOMMESSA VINTA! Profit: €13.50
💰 Bankroll aggiornato e salvato: €113.50
   ✅ Scommessa aggiornata automaticamente!

🎉 1 scommesse aggiornate automaticamente!
💰 Bankroll attuale: €113.50
✅ Controllo completato alle 22:00:15
```

### **3. Controllo Manuale Occasionale**
```bash
python main.py --check-bets
```

---

## ⚙️ **Parametri Avanzati**

### **Modalità Automatica**
```bash
# Analizza automaticamente prima partita disponibile
python main.py --auto-mode

# Con linea specifica
python main.py --auto-mode --line 225.0
```

### **Team Specifici (esempi)**
```bash
python main.py --team1 "Los Angeles Lakers" --team2 "Golden State Warriors"
python main.py --team1 Celtics --team2 Heat
python main.py --team1 Bucks --team2 Nets
```

### **Configurazione Daemon**
```bash
# Controllo ogni 15 minuti
python bankroll_updater.py --daemon --interval 15

# Controllo ogni 2 ore
python bankroll_updater.py --daemon --interval 120

# Controllo singolo immediato
python bankroll_updater.py
```

---

## 🚨 **Risoluzione Problemi Comuni**

### **Errore: "unrecognized arguments"**
**Problema:** Stai usando i vecchi parametri
```bash
# ❌ VECCHIO (non funziona più)
python main.py --giorni 4 --linea-centrale 225.0

# ✅ NUOVO (funziona)  
python main.py --team1 Lakers --team2 Warriors --line 225.0
```

### **Nessuna Partita Trovata**
```bash
# Il sistema mostra automaticamente partite disponibili
python main.py
# Se nessuna partita è trovata, usa partita di esempio
```

### **Scommesse Non Aggiornate**
```bash
# Controllo manuale debug
python main.py --check-bets

# Verifica file pendenti
ls -la data/pending_bets.json
```

---

## 📊 **File Importanti**

| **File** | **Funzione** |
|----------|-------------|
| `bankroll.json` | Bankroll principale |
| `data/bankroll.json` | Backup bankroll |
| `data/pending_bets.json` | Scommesse in attesa |
| `data/probabilistic_predictions.json` | Storico predizioni |

---

**🎯 Il sistema ora è più semplice e automatizzato!** 

Usa `python main.py` per iniziare e il sistema ti guiderà attraverso le opzioni disponibili. 🚀 