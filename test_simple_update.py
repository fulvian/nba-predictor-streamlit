#!/usr/bin/env python3
"""
Test simple status update on pending bets
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from nba_predictor.utils.betting_database_manager import BettingDatabaseManager

# Test a simple status update
db_manager = BettingDatabaseManager()

# Check what pending bets we have
pending_bets = db_manager.get_pending_bets()
print(f'Found {len(pending_bets)} pending bets')

# Try to find a specific bet and just update its status to 'settled' for testing
if pending_bets:
    test_bet = pending_bets[0]
    print(f'Testing update on bet: {test_bet.bet_id}')

    try:
        # Let's check if there are references in betting_analysis
        ref_count = db_manager.conn.execute('SELECT COUNT(*) FROM betting_analysis WHERE bet_id = ?', [test_bet.bet_id]).fetchone()[0]
        print(f'References in betting_analysis: {ref_count}')

        # Try direct UPDATE without changing other columns
        print('Trying simple status update...')
        result = db_manager.conn.execute('UPDATE bets SET status = ? WHERE bet_id = ?', ['test_settled', test_bet.bet_id])
        print(f'Update result: {result}')

        # Rollback the change
        db_manager.conn.execute('UPDATE bets SET status = ? WHERE bet_id = ?', ['pending', test_bet.bet_id])
        print('Test update successful - rolled back')

    except Exception as e:
        print(f'Update failed: {e}')

db_manager.conn.close()
print('Test complete')