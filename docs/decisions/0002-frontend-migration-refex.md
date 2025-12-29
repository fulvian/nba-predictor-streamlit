# ADR 002: Migrazione Frontend da Streamlit a Framework "Pro"

## Contesto
L'attuale dashboard in Streamlit (`04_Live_Betting_Monitor.py`) non soddisfa i requisiti estetici e funzionali di un "Professional Trading Terminal".
- **Problemi**: Contrasto limitato, layout rigido, componenti standard "data science" non adatti al trading in tempo reale, difficoltà nello styling avanzato (Dark Mode pixel-perfect).
- **Obiettivo**: Creare un'interfaccia "Bloomberg meets Cyberpunk" con aggiornamenti real-time <100ms e controllo totale del CSS.

## Ricerca & Opzioni (2025)

### 1. Streamlit (Attuale)
- **Pro**: Semplicità estrema, zero HTML/JS.
- **Contro**: Styling limitato (solo CSS hacks), Performance websocket non native, Layout a blocchi rigidi.
- **Verdetto**: ❌ Inadatto per UI High-End.

### 2. Reflex (Ex Pynecone) - 🏆 RACCOMANDATO
- **Tecnologia**: Compila Python in React.js (Frontend) + FastAPI (Backend).
- **Pro**: 
    - Styling CSS/Tailwind completo (può fare *esattamente* il look Cyberpunk).
    - Componenti React reali (grafici fluidi, griglie dati professionali).
    - Websocket nativi (aggiornamenti istantanei).
    - Scritto 100% in Python.
- **Contro**: Richiede Node.js installato nel sistema (per la compilazione). Curva di apprendimento superiore a Streamlit.

### 3. NiceGUI
- **Tecnologia**: Vue.js + FastAPI.
- **Pro**: Molto leggero (niente Node.js richiesto), styling CSS flessibile (Tailwind/Quasar), aggiornamenti veloci.
- **Contro**: Meno "strutturato" di Reflex per app grandi, estetica default "Material Design" (richiede lavoro per il look Cyberpunk).

## Proposta di Implementazione: "Project NEON"

Si consiglia di passare a **Reflex** se l'ambiente lo supporta (Node.js), altrimenti **NiceGUI**.

### Stack Proposto (Reflex)
- **Frontend**: Reflex Components (Wrapping React).
- **Styling**: Tailwind CSS (integrato in Reflex) per il look "Neon/Dark".
- **Backend**: Riutilizzo immediato di `BetfairService` (già scritto in Python).
- **Data Grid**: `ag-grid` (integrato in Reflex) per tabelle "Excel-like" professionali.

## Piano d'Azione
1. Installare Reflex (`pip install reflex`).
2. Inizializzare progetto (`reflex init`).
3. Porting della logica: Collegare `BetfairService` allo State di Reflex.
4. Design UI: Costruire il layout a griglia (HUD, Market Ladder, Execution Deck).
