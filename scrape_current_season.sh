#!/bin/bash
# Scrape CURRENT NBA season (uses /nba/results/ instead of /nba-YYYY-YYYY/results/)

set -e

# Configuration
PYTHON_EXEC="/Users/fulvioventura/.pyenv/versions/oddsharvester-env/bin/python"
SITE_PACKAGES="/Users/fulvioventura/.pyenv/versions/oddsharvester-env/lib/python3.12/site-packages"
OUTPUT_DIR="/Users/fulvioventura/nba-predictor-streamlit/data/odds"
PROXY="socks5://127.0.0.1:9050"
LOG_FILE="/Users/fulvioventura/scraping_current_season.log"

echo "🏀 Starting CURRENT SEASON scrape @ $(date)" | tee -a $LOG_FILE

# Check Tor
if ! nc -z 127.0.0.1 9050 2>/dev/null; then
    echo "❌ Tor proxy not running on port 9050" | tee -a $LOG_FILE
    exit 1
fi
echo "✅ Tor proxy verified" | tee -a $LOG_FILE

# Clear old checkpoint to force fresh scrape
rm -f "$OUTPUT_DIR/checkpoints/checkpoint_basketball_latest.json"
echo "🗑️ Cleared old checkpoint" | tee -a $LOG_FILE

# Navigate to site-packages for module resolution
cd "$SITE_PACKAGES"

# Scrape current season (omit --season to use /nba/results/)
echo "📥 Scraping current NBA season (no --season = current)..." | tee -a $LOG_FILE

$PYTHON_EXEC -m src.main scrape_historic \
    --sport basketball \
    --leagues nba \
    --markets over_under \
    --headless \
    --format json \
    --proxies "$PROXY" \
    --preview_submarkets_only \
    --file_path "$OUTPUT_DIR/scraped_current_season.json" >> $LOG_FILE 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Current season scrape completed!" | tee -a $LOG_FILE
    # Count records
    RECORDS=$(python3 -c "import json; print(len(json.load(open('$OUTPUT_DIR/scraped_current_season.json'))))" 2>/dev/null || echo "0")
    echo "📊 Records scraped: $RECORDS" | tee -a $LOG_FILE
else
    echo "❌ Scrape failed" | tee -a $LOG_FILE
    exit 1
fi

echo "🎉 Done @ $(date)" | tee -a $LOG_FILE
