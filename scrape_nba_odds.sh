#!/bin/bash

# Activate the virtual environment if not already activated
# Assuming the user is running this from the project root or has pyenv set up
# We use the full path to the python executable we identified to be safe
PYTHON_EXEC="/Users/fulvioventura/.pyenv/versions/oddsharvester-env/bin/python"
PROJECT_DIR="/Users/fulvioventura/nba-predictor-streamlit"
DATA_DIR="$PROJECT_DIR/data/odds"
PROXY="socks5://127.0.0.1:9050"

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

echo "Starting NBA Odds Scraping for seasons 2020-2025..."
echo "Using Proxy: $PROXY"

# Function to scrape a season
scrape_season() {
    SEASON="$1"
    OUTPUT_FILE="$DATA_DIR/scraped_${SEASON//-/_}.json"
    
    echo "----------------------------------------------------------------"
    echo "Scraping Season: $SEASON"
    echo "Output: $OUTPUT_FILE"
    
    cd ~/.pyenv/versions/oddsharvester-env/lib/python3.12/site-packages
    
    $PYTHON_EXEC -m src.main scrape_historic \
        --sport basketball \
        --leagues nba \
        --season "$SEASON" \
        --markets over_under \
        --headless \
        --format json \
        --file_path "$OUTPUT_FILE" \
        --proxies "$PROXY" \
        --preview_submarkets_only \
        --save_logs
        
    if [ $? -eq 0 ]; then
        echo "✅ Successfully scraped $SEASON"
    else
        echo "❌ Failed to scrape $SEASON"
    fi
}

# Run scraping for each season sequentially
# Seasons: 2020-2021, 2021-2022, 2022-2023, 2023-2024, 2024-2025

scrape_season "2020-2021"
scrape_season "2021-2022"
scrape_season "2022-2023"
scrape_season "2023-2024"
scrape_season "2024-2025"

echo "----------------------------------------------------------------"
echo "🎉 All scraping tasks completed."
