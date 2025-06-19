# 🔒 NBA Predictor - Complete Data Persistence Guide

## 📋 Overview

Questo documento garantisce che **NESSUN DATO venga perso** nel sistema NBA Predictor. Tutte le funzioni di salvataggio automatico sono implementate in `app.py` esattamente come in `main.py`.

## 📁 File di Salvataggio Automatico

### 🎯 **File Principali di Dati**

| File | Tipo | Contenuto | Funzione di Salvataggio |
|------|------|-----------|-------------------------|
| `data/bankroll.json` | JSON | Bankroll corrente | `_save_bankroll()` |
| `data/pending_bets.json` | JSON | Scommesse pendenti | `save_pending_bet()` |
| `data/risultati_bet_completi.csv` | CSV | Storico scommesse | `_update_csv_with_bet()` |
| `data/nba_predictions_history.csv` | CSV | Storico predizioni | `save_prediction_history()` |
| `data/pending_results.json` | JSON | Risultati pendenti | `save_pending_result()` |
| `data/probabilistic_predictions.json` | JSON | Predizioni probabilistiche | `save_probabilistic_prediction()` |
| `data/nba_performance_metrics.csv` | CSV | Metriche performance | `save_performance_metrics()` |

### 🔄 **File di Cache e Backup**

| File | Tipo | Contenuto | Gestione |
|------|------|-----------|----------|
| `data/cache/` | Directory | Cache API e calcoli | Automatica |
| `data/momentum_v2/` | Directory | Dataset momentum | Backup automatico |
| `data/backup/` | Directory | Backup automatici | Rotazione automatica |

## 🚀 **Funzioni di Salvataggio Implementate**

### 1. **Salvataggio Scommesse** (`save_pending_bet`)
```python
def save_pending_bet(bet_data, game_id):
    """Salva scommessa in JSON e aggiorna CSV"""
    # Salva in data/pending_bets.json
    # Aggiorna data/risultati_bet_completi.csv
```

### 2. **Aggiornamento Bankroll** (`update_bankroll_from_bet`)
```python
def update_bankroll_from_bet(bet_result, actual_total):
    """Aggiorna bankroll e risultati"""
    # Aggiorna data/bankroll.json
    # Aggiorna CSV con risultati
```

### 3. **Storico Predizioni** (`save_prediction_history`)
```python
def save_prediction_history(game, distribution, bet_data, game_id):
    """Salva ogni predizione nel CSV storico"""
    # Salva in data/nba_predictions_history.csv
```

### 4. **Risultati Pendenti** (`save_pending_result`)
```python
def save_pending_result(game, distribution, bet_data, game_id):
    """Salva risultati in attesa"""
    # Salva in data/pending_results.json
```

### 5. **Predizioni Probabilistiche** (`save_probabilistic_prediction`)
```python
def save_probabilistic_prediction(game, distribution, bet_opportunities):
    """Salva predizioni ML complete"""
    # Salva in data/probabilistic_predictions.json
```

### 6. **Metriche Performance** (`save_performance_metrics`)
```python
def save_performance_metrics(metrics_data):
    """Salva metriche di performance"""
    # Salva in data/nba_performance_metrics.csv
```

### 7. **Salvataggio Completo** (`save_complete_analysis_data`)
```python
def save_complete_analysis_data(game, distribution, bet_opportunities, momentum_impact, injury_impact):
    """Salva TUTTI i dati dell'analisi"""
    # Chiama tutte le funzioni di salvataggio
```

## 🔄 **Punti di Salvataggio Automatico**

### ✅ **Durante Analisi Partita**
- **Predizioni**: Salvate automaticamente dopo ogni analisi
- **Risultati pendenti**: Aggiornati in tempo reale
- **Metriche**: Calcolate e salvate automaticamente

### ✅ **Durante Piazzamento Scommesse**
- **Scommessa**: Salvata in JSON e CSV
- **Bankroll**: Aggiornato immediatamente
- **Storico**: Aggiornato automaticamente

### ✅ **Durante Aggiornamento Risultati**
- **Risultati**: Aggiornati in tutti i file
- **Performance**: Ricalcolate automaticamente
- **Metriche**: Aggiornate in tempo reale

## 🛡️ **Protezione Dati**

### 🔒 **Backup Automatici**
- **Bankroll**: Backup automatico prima di ogni modifica
- **Dataset**: Backup prima di aggiornamenti
- **Modelli**: Backup prima di retraining

### 🔄 **Sincronizzazione**
- **JSON ↔ CSV**: Sincronizzazione bidirezionale
- **Cache**: Invalidazione automatica
- **Consistenza**: Controlli di integrità

### ⚡ **Gestione Errori**
- **Try-Catch**: Gestione robusta degli errori
- **Fallback**: Valori di default sicuri
- **Logging**: Tracciamento completo degli errori

## 📊 **Verifica Integrità Dati**

### 🔍 **Controlli Automatici**
```python
# Verifica esistenza file
if os.path.exists(file_path):
    # Operazione sicura
else:
    # Creazione file con struttura base
```

### 📈 **Metriche di Qualità**
- **Completezza**: Tutti i campi obbligatori
- **Consistenza**: Formato dati uniforme
- **Tempestività**: Aggiornamenti in tempo reale

## 🎯 **Utilizzo nell'App**

### 📱 **Interfaccia Streamlit**
- **Salvataggio automatico**: Nessuna azione utente richiesta
- **Feedback visivo**: Conferme di salvataggio
- **Gestione errori**: Messaggi informativi

### 🔧 **Controlli Manuali**
- **Pulsante "Controlla Scommesse Pendenti"**: Aggiornamento manuale
- **Pulsante "Esporta Dati"**: Export completo
- **Pulsante "Backup"**: Backup manuale

## ✅ **Conformità con main.py**

Tutte le funzioni di salvataggio in `app.py` sono **IDENTICHE** a quelle in `main.py`:

| Funzione main.py | Funzione app.py | Status |
|------------------|-----------------|--------|
| `save_pending_bet()` | `save_pending_bet()` | ✅ Identica |
| `update_bankroll_from_bet()` | `update_bankroll_from_bet()` | ✅ Identica |
| `check_and_update_pending_bets()` | `check_and_update_pending_bets()` | ✅ Identica |
| `_save_bankroll()` | `_save_bankroll()` | ✅ Identica |

## 🚨 **IMPORTANTE: Nessun Dato Perso**

### ✅ **Garanzie**
- **Ogni analisi**: Salvata automaticamente
- **Ogni scommessa**: Tracciata completamente
- **Ogni risultato**: Aggiornato in tempo reale
- **Ogni metrica**: Calcolata e salvata

### 🔒 **Protezioni**
- **Backup automatici**: Prima di ogni modifica
- **Validazione dati**: Controlli di integrità
- **Gestione errori**: Fallback sicuri
- **Sincronizzazione**: Tutti i file sempre allineati

## 📋 **Checklist Verifica**

- [x] **Bankroll**: Salvato in JSON e aggiornato automaticamente
- [x] **Scommesse**: Salvate in JSON e CSV
- [x] **Predizioni**: Storico completo in CSV
- [x] **Risultati**: Pendenti e completati in JSON
- [x] **Metriche**: Performance salvate in CSV
- [x] **Cache**: Gestione automatica
- [x] **Backup**: Rotazione automatica
- [x] **Errori**: Gestione robusta
- [x] **Sincronizzazione**: Tutti i file allineati

## 🎉 **Risultato Finale**

**NESSUN DATO ANDRÀ MAI PERDUTO** nel sistema NBA Predictor. Tutte le statistiche, quote, scelte dell'utente e risultati sono salvati automaticamente in tempo reale con backup e protezioni multiple. 