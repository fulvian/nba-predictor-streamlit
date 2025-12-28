#!/bin/bash

# Script di monitoring automatico per lo scraping NBA
# Controlla ogni 10 minuti lo stato e notifica al completamento

DATA_DIR="/Users/fulvioventura/nba-predictor-streamlit/data/odds"
LOG_FILE="/Users/fulvioventura/scraping_master.log"
NOTIFY_FILE="/Users/fulvioventura/scraping_completed.flag"

echo "🔍 Monitoring scraping in background..."
echo "📊 Checking every 10 minutes for completion"
echo ""

while true; do
    # Conta i file JSON completati
    completed=$(ls "$DATA_DIR"/scraped_*.json 2>/dev/null | wc -l | xargs)
    
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] Stagioni completate: $completed/5"
    
    # Se tutte e 5 le stagioni sono completate
    if [ "$completed" -eq 5 ]; then
        echo ""
        echo "✅ TUTTE LE STAGIONI COMPLETATE!"
        echo ""
        
        # Genera report di riepilogo
        echo "📊 RIEPILOGO FINALE:" > "$NOTIFY_FILE"
        echo "" >> "$NOTIFY_FILE"
        
        for season_file in "$DATA_DIR"/scraped_*.json; do
            season=$(basename "$season_file" .json | sed 's/scraped_//')
            size=$(ls -lh "$season_file" | awk '{print $5}')
            matches=$(python3 -c "import json; print(len(json.load(open('$season_file'))))")
            echo "✅ $season: $matches partite ($size)" >> "$NOTIFY_FILE"
        done
        
        echo "" >> "$NOTIFY_FILE"
        echo "Completato alle: $timestamp" >> "$NOTIFY_FILE"
        
        cat "$NOTIFY_FILE"
        exit 0
    fi
    
    # Attendi 10 minuti prima del prossimo check
    sleep 600
done
