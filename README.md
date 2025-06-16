# NBA Performance Predictor

Un'applicazione avanzata per l'analisi e la predizione delle prestazioni dei giocatori NBA utilizzando tecniche di Machine Learning.

## 🚀 Funzionalità

- **Analisi Statistiche**: Analisi dettagliata delle statistiche dei giocatori NBA
- **Predizione Prestazioni**: Modelli ML per predire le prestazioni future
- **Monitoraggio Momentum**: Analisi del momentum dei giocatori
- **Gestione Infortuni**: Report e analisi sugli infortuni
- **Dashboard Interattiva**: Interfaccia web con Streamlit
- **API REST**: Endpoint per l'integrazione con altri servizi

## 🛠️ Installazione

### Prerequisiti

- Python 3.8+
- pip (Python package manager)
- Git

### Configurazione

1. Clona il repository:
   ```bash
   git clone https://github.com/fulvian/autoover.git
   cd autoover
   ```

2. Crea e attiva un ambiente virtuale (consigliato):
   ```bash
   # Su macOS/Linux
   python -m venv venv
   source venv/bin/activate
   
   # Su Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```

4. Imposta le variabili d'ambiente:
   Crea un file `.env` nella directory radice con le tue chiavi API:
   ```
   NBA_API_KEY=your_nba_api_key
   # Altre variabili d'ambiente
   ```

## 💻 Utilizzo

### Avvio dell'applicazione Streamlit

```bash
streamlit run src/main.py
```

### Esecuzione dei test

```bash
pytest tests/
```

### Installazione in modalità sviluppo

```bash
pip install -e .
```

## 📊 Struttura del Progetto

```
nba-predictor/
├── .github/               # Configurazione GitHub Actions
├── data/                  # Dati grezzi e dataset
│   ├── cache/             # Cache delle richieste API
│   └── processed/         # Dati processati
├── models/                # Modelli ML addestrati
├── notebooks/             # Jupyter notebooks per analisi
├── src/                   # Codice sorgente
│   ├── core/              # Logica di business principale
│   ├── data/              # Gestione e pulizia dei dati
│   ├── models/            # Modelli ML e training
│   ├── utils/             # Funzioni di utilità
│   └── web/               # Frontend Streamlit
├── tests/                 # Test automatici
├── .env.example          # Template variabili d'ambiente
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

## 🔧 Sviluppo

### Strumenti consigliati

- **Editor di codice**: VS Code o PyCharm
- **Ambiente virtuale**: `venv` o `conda`
- **Formattazione codice**: Black
- **Linting**: Flake8
- **Type checking**: mypy

### Convenzioni di codice

- Seguire le linee guida PEP 8
- Usare type hints per tutte le funzioni
- Documentare le funzioni con docstring
- Scrivere test per nuovo codice

## 🤝 Contributi

I contributi sono benvenuti! Ecco come puoi contribuire:

1. Crea un fork del progetto
2. Crea un branch per la tua feature (`git checkout -b feature/AmazingFeature`)
3. Fai commit delle tue modifiche (`git commit -m 'Aggiungi qualche AmazingFeature'`)
4. Pusha il branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request

## 📄 Licenza

Questo progetto è concesso in licenza con la licenza MIT - vedi il file [LICENSE](LICENSE) per i dettagli.

## 📧 Contatti

Fulvio Ventura - [@tuo_profilo](https://twitter.com/tuo_profilo) - fulviold@gmail.com

Link al progetto: [https://github.com/fulvian/autoover](https://github.com/fulvian/autoover)

