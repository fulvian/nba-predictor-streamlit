# Deprecated Dashboards

⚠️ **ATTENZIONE**: Questa cartella contiene dashboard deprecate che non devono essere utilizzate.

## Dashboard Funzionante

La dashboard ufficiale e funzionante è:
- **File**: `../run_betting_workflow.py` (root del progetto)
- **URL**: http://localhost:8504
- **Porta**: 8504

## Dashboard Deprecate

1. **run_dashboard.py** - Dashboard base con dati fittizi
2. **run_complete_dashboard.py** - Versione completa con bug di formattazione
3. **app_complete.py** - App streamlit completa deprecata

## Problemi Risolti

Le dashboard deprecate avevano:
- ❌ Bug formattazione confidence (9500.0% invece di 95.0%)
- ❌ Intervallo confidenza errato (199.3-200.7)
- ❌ Delta probabilità sbagliato (+-45.0%)
- ❌ Lock database DuckDB
- ❌ Dati fittizi invece di dati reali NBA

## Dashboard Corretta

La dashboard `run_betting_workflow.py` ha tutte le correzioni applicate:
- ✅ Formattazione corretta di tutti i valori
- ✅ Intervalli di confidenza realistici
- ✅ Calcoli probabilità funzionanti
- ✅ Dati NBA reali
- ✅ Workflow completo 3-step

**UTILIZZARE SOLO**: `../run_betting_workflow.py`