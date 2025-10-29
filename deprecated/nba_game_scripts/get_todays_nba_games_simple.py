#!/usr/bin/env python3
"""
🏀 GET TODAY'S NBA GAMES - Simple Script using Data Persistence Bridge
Lo script corretto che scarica e salva le partite NBA programmate nel data store.
"""

import sys
import os
sys.path.append('src')

# Fix import paths
os.chdir('/Users/fulvioventura/nba-predictor-streamlit')

try:
    from nba_predictor.core.data_persistence_bridge import initialize_persistence_bridge, get_persistence_bridge
    from nba_predictor.core.data_persistence_bridge import close_persistence_bridge
    from datetime import date
    
    print('🏀 GET TODAY\'S NBA GAMES - DATA PERSISTENCE BRIDGE')
    print('=' * 60)
    
    # Initialize the persistence bridge
    print('🔄 Initializing persistence bridge...')
    bridge = initialize_persistence_bridge()
    
    if bridge:
        today = date.today()
        today_str = today.strftime('%Y-%m-%d')
        
        print(f'📅 Fetching games for: {today_str}')
        print()
        
        # Get today's games with persistence (this will download if needed)
        games = bridge.get_scheduled_games_with_persistence(
            days_ahead=1,
            specific_date=today_str,
            force_api=True  # Force fresh download
        )
        
        if games:
            print(f'✅ SUCCESS: Found {len(games)} NBA games for today!')
            print()
            
            for i, game in enumerate(games, 1):
                home_team = game.get('home_team', 'N/D')
                away_team = game.get('away_team', 'N/D')
                game_time = game.get('game_time', 'N/D')
                status = game.get('status', 'N/D')
                source = game.get('source', 'N/D')
                
                print(f'{i}. {away_team} vs {home_team}')
                print(f'   ⏰ Time: {game_time}')
                print(f'   📊 Status: {status}')
                print(f'   🌐 Source: {source}')
                print()
            
            print('🎯 These games have been saved to the persistent data store!')
            print(f'📁 Location: data/persistent/games/games_{today_str}.parquet')
            
        else:
            print('❌ No games found for today')
        
        # Close the bridge
        close_persistence_bridge()
        print()
        print('✅ Persistence bridge closed successfully')
        
    else:
        print('❌ Failed to initialize persistence bridge')
        
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
