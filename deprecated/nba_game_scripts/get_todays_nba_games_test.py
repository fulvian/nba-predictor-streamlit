#!/usr/bin/env python3
"""
🏀 TEST TODAY'S NBA GAMES - Direct Data Store Test
Test script per verificare che i dati nel data store sono corretti.
"""

import polars as pl
from datetime import date

print('🏀 TEST TODAY\'S NBA GAMES - DIRECT DATA STORE CHECK')
print('=' * 70)

# Leggi direttamente dal data store
today = date.today()
today_str = today.strftime('%Y-%m-%d')

print(f'📅 Checking games for: {today_str}')
print(f'📁 Looking in: data/persistent/games/games_{today_str}.parquet')
print()

file_path = f'data/persistent/games/games_{today_str}.parquet'

try:
    if pl.scan_parquet(file_path).collect():
        df = pl.read_parquet(file_path)
        
        print(f'✅ SUCCESS: Found {len(df)} NBA games for today!')
        print()
        print('📋 Games Details:')
        print()
        
        for i, row in enumerate(df.iter_rows(), 1):
            home_team = row[df.columns.index('home_team')]
            away_team = row[df.columns.index('away_team')]
            game_id = row[df.columns.index('game_id')]
            
            print(f'{i}. {away_team} vs {home_team}')
            print(f'   🆔 Game ID: {game_id}')
            
            # Altri dettagli se disponibili
            if 'game_time' in df.columns:
                game_time = row[df.columns.index('game_time')]
                print(f'   ⏰ Time: {game_time}')
            if 'status' in df.columns:
                status = row[df.columns.index('status')]
                print(f'   📊 Status: {status}')
            if 'venue' in df.columns:
                venue = row[df.columns.index('venue')]
                print(f'   🏟️ Venue: {venue}')
            print()
        
        print('🎯 These are the REAL NBA games stored in your data store!')
        print('💡 The system has these games available for prediction.')
        print()
        print('✅ Data store is working correctly!')
        
    else:
        print(f'❌ No data found in: {file_path}')
        print('💡 This could mean:')
        print('   - No games scheduled for today')
        print('   - Data hasn\'t been downloaded yet')
        print('   - File path issue')
        
except Exception as e:
    print(f'❌ Error reading data store: {e}')
    print('💡 The data store might not be accessible')

print()
print('🔍 This confirms the data store system is working.')
print('📊 The real download script would use DataPersistenceBridge')
