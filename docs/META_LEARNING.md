# Integrazione NanoGPT: Meta-Learning Feedback Loop

## Panoramica
Questa applicazione utilizza ora un modulo di **Meta-Learning** avanzato (`src/nba_predictor/intelligence/feedback_loop.py`) per "imparare dai propri errori".

## Architettura del Feedback Loop

### 1. Il Problema
I modelli statistici possono sviluppare bias nascosti (es. sovrastimare sempre i Lakers quando giocano fuori casa). Senza feedback, il modello ripete l'errore.

### 2. La Soluzione: EMA Bias Correction
Prima di generare una previsione per "Team A vs Team B", il sistema esegue:

1.  **Historical Query**: Recupera le ultime **10 partite** dal DB locale (`filtered by status='SETTLED'`).
2.  **Parsing Errore**: Confronta `predicted_total` (salvato nel JSON della scommessa) con il punteggio reale (`home + away`).
### 2. Bias Calculation (EMA)
- The system calculates an *Exponential Moving Average (EMA)* of the prediction errors.
- **Alpha**: `0.25` (Standard) - Gives more weight to recent games (approx last 4 games) without chasing noise.
- **Formula**: `EMA_today = (Error_today * Alpha) + (EMA_yesterday * (1 - Alpha))`

### 3. Threshold Trigger
- The Meta-Learning Prompt is **ONLY** injected if the absolute bias exceeds a significant threshold.
- **Threshold**: `5.0` points.
- *Reasoning*: NBA totals variance is high. We only want to correct for persistent, structural model bias, not random nightly variance.

### 3. Prompt Injection
Il Consensus Engine riceve un blocco di testo speciale:
> "SYSTEM NOTIFICATION: We differ logic... Detected BIAS: OVERESTIMATING Total Score by 5.2 points. ADJUST PREDICTION."

## Manutenzione
*   **Database**: Il sistema dipende dalla tabella `bets` e dallo stato `SETTLED`. Assicurarsi che `auto_bet_settlement.py` giri regolarmente.
*   **Parsing**: La logica di parsing JSON in `feedback_loop.py` è critica. Se cambia il formato del JSON in `prediction`, bisogna aggiornare quel file.
