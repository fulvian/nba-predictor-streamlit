#!/bin/bash

# RECOVERY SCRIPT - Scrapes ONLY seasons with missing/incomplete data
# Uses patched OddsHarvester with checkpointing
# FIXED: Uses correct python executable and path context

PROJECT_DIR="/Users/fulvioventura/nba-predictor-streamlit"
DATA_DIR="$PROJECT_DIR/data/odds"
LOG_FILE="/Users/fulvioventura/scraping_recovery.log"
PYTHON_EXEC="/Users/fulvioventura/.pyenv/versions/oddsharvester-env/bin/python"
SITE_PACKAGES="/Users/fulvioventura/.pyenv/versions/oddsharvester-env/lib/python3.12/site-packages"
PROXY="socks5://127.0.0.1:9050"

echo "🔄 Starting Recovery Scraping (Smart Resume)..." > $LOG_FILE
echo "🎯 Targets: 2022-2023, 2023-2024, 2024-2025" >> $LOG_FILE
echo "🛡️  Checkpointing ENABLED (Batch size: 50)" >> $LOG_FILE

# Function to scrape a specific season
scrape_season() {
    SEASON=$1
    OUTPUT_FILE="$DATA_DIR/scraped_${SEASON//-/_}.json"
    
    echo "----------------------------------------------------------------" >> $LOG_FILE
    echo "🚀 RECOVERY: Starting Season $SEASON" >> $LOG_FILE
    echo "----------------------------------------------------------------" >> $LOG_FILE
    
    # CRITICAL: Change to site-packages for correct module resolution
    cd "$SITE_PACKAGES"
    
    $PYTHON_EXEC -m src.main scrape_historic \
        --sport basketball \
        --leagues nba \
        --season "$SEASON" \
        --markets over_under \
        --headless \
        --format json \
        --proxies "$PROXY" \
        --preview_submarkets_only \
        --file_path "$OUTPUT_FILE" >> $LOG_FILE 2>&1
        
    RET_CODE=$?
    
    # Return to project dir
    cd "$PROJECT_DIR"
        
    if [ $RET_CODE -eq 0 ]; then
        echo "✅ SUCCESS: Season $SEASON recovered." >> $LOG_FILE
    else
        echo "❌ FAILURE: Season $SEASON crashed. Check logs." >> $LOG_FILE
    fi
    
    # Random sleep to let Tor/Network cool down
    sleep 30
}

# Execute recovery/scraping for target seasons
scrape_season "2023-2024"
scrape_season "2024-2025"
scrape_season "2025-2026"

echo "🎉 Scraping process completed." >> $LOG_FILE
