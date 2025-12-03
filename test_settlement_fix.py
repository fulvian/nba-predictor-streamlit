#!/usr/bin/env python3
"""
🎯 TEST - NBA Betting Settlement System Fix

Script di test per verificare e correggere i problemi del sistema di auto-settlement.
"""

import sys
sys.path.append('src')

from nba_predictor.utils.betting_database_manager import BettingDatabaseManager
from nba_predictor.utils.auto_bet_settlement import AutoBetSettlement
import requests
from datetime import datetime

def get_real_nba_scores():
    """
    Ottiene punteggi reali da NBA boxscore API.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com'
    }

    # Game IDs for our pending bets (from NBA API investigation)
    game_mappings = {
        'Knicks @ Bulls': '0022500023',      # Chicago Bulls vs New York Knicks
        'Lakers @ Grizzlies': '0022500024',  # Memphis Grizzlies vs Los Angeles Lakers
        'Nuggets @ Trail Blazers': '0022500026', # Portland Trail Blazers vs Denver Nuggets
        'Celtics @ 76ers': '0022500021',     # Boston Celtics (not in our bets)
        'Hawks @ Pacers': '0022500020',      # (not in our bets)
        'Raptors @ Cavaliers': '0022500022', # (not in our bets)
        'Jazz @ Suns': '0022500025',         # (not in our bets)
        'Pelicans @ Clippers': '0022500027'  # (not in our bets)
    }

    real_scores = {}

    for game_desc, game_id in game_mappings.items():
        try:
            # Get boxscore data
            url = f'https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json'
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if 'game' in data:
                    game_data = data['game']
                    home_team = game_data.get('homeTeam', {})
                    away_team = game_data.get('awayTeam', {})

                    home_score = home_team.get('score', 0)
                    away_score = away_team.get('score', 0)
                    home_name = home_team.get('teamName', '')
                    away_name = away_team.get('teamName', '')

                    if home_score > 0 and away_score > 0:
                        real_scores[game_id] = {
                            'game_desc': f"{away_name} @ {home_name}",
                            'home_score': int(home_score),
                            'away_score': int(away_score),
                            'total_points': int(home_score) + int(away_score),
                            'home_team': home_name,
                            'away_team': away_name
                        }
                        print(f"✅ Found scores for {game_desc}: {away_score}-{home_score} (Total: {home_score + away_score})")
                    else:
                        print(f"⚠️ No valid scores for {game_desc}: {away_score}-{home_score}")
                else:
                    print(f"❌ No game data for {game_desc}")
            else:
                print(f"❌ Failed to get {game_desc}: HTTP {response.status_code}")

        except Exception as e:
            print(f"❌ Error getting {game_desc}: {e}")

    return real_scores

def map_manual_bets_to_nba_games(betting_db, real_scores):
    """
    Mappa le scommesse manuali ai game ID NBA.
    """
    print("\n🔍 Mapping manual bets to NBA games...")

    # NBA team name mappings
    team_mappings = {
        'Knicks': 'New York Knicks',
        'Bulls': 'Chicago Bulls',
        'Lakers': 'Los Angeles Lakers',
        'Grizzlies': 'Memphis Grizzlies',
        'Nuggets': 'Denver Nuggets',
        'Trail Blazers': 'Portland Trail Blazers',
        'Celtics': 'Boston Celtics',
        '76ers': 'Philadelphia 76ers',
        'Rockets': 'Houston Rockets',
        'Hawks': 'Atlanta Hawks',
        'Pacers': 'Indiana Pacers',
        'Raptors': 'Toronto Raptors',
        'Cavaliers': 'Cleveland Cavaliers',
        'Jazz': 'Utah Jazz',
        'Suns': 'Phoenix Suns',
        'Pelicans': 'New Orleans Pelicans',
        'Clippers': 'LA Clippers'
    }

    # Get pending bets
    result = betting_db.conn.execute('''
        SELECT bet_id, game_id, bet_type, line, home_team, away_team, stake, odds
        FROM placed_bets
        WHERE status = 'pending'
        ORDER BY placed_at DESC
    ''').fetchall()

    mappings = []

    for bet_row in result:
        bet_id, manual_game_id, bet_type, line, home_team, away_team, stake, odds = bet_row

        print(f"\n🎲 Processing bet: {bet_id}")
        print(f"   Manual teams: {home_team} vs {away_team}")
        print(f"   Bet: {bet_type} {line} @ {odds} (€{stake:.2f})")

        # Find matching NBA game
        found_match = False
        for nba_game_id, game_data in real_scores.items():
            nba_home = game_data['home_team']
            nba_away = game_data['away_team']

            # Improved team name matching
            home_match = (home_team.lower() in nba_home.lower() or nba_home.lower() in home_team.lower())
            away_match = (away_team.lower() in nba_away.lower() or nba_away.lower() in away_team.lower())
            reverse_home_match = (home_team.lower() in nba_away.lower() or nba_away.lower() in home_team.lower())
            reverse_away_match = (away_team.lower() in nba_home.lower() or nba_home.lower() in away_team.lower())

            if (home_match and away_match) or (reverse_home_match and reverse_away_match):
                print(f"   ✅ NBA Match Found: {game_data['game_desc']} ({nba_game_id})")
                print(f"   Scores: {game_data['away_score']}-{game_data['home_score']} (Total: {game_data['total_points']})")

                # Calculate bet outcome
                total_points = game_data['total_points']
                if bet_type.upper() == 'OVER':
                    outcome = 'WON' if total_points > line else 'LOST' if total_points < line else 'VOID'
                elif bet_type.upper() == 'UNDER':
                    outcome = 'WON' if total_points < line else 'LOST' if total_points > line else 'VOID'
                else:
                    outcome = 'UNKNOWN'

                print(f"   🎯 Bet Result: {outcome} (Total {total_points} vs Line {line})")

                # Calculate profit/loss using actual odds
                if outcome == 'WON':
                    result_amount = stake * odds
                    profit_loss = result_amount - stake
                elif outcome == 'LOST':
                    profit_loss = -stake
                    result_amount = 0
                else:  # VOID
                    profit_loss = 0
                    result_amount = stake

                print(f"   💰 Profit/Loss: €{profit_loss:.2f} (Return: €{result_amount:.2f})")

                mappings.append({
                    'bet_id': bet_id,
                    'manual_game_id': manual_game_id,
                    'nba_game_id': nba_game_id,
                    'bet_type': bet_type,
                    'line': line,
                    'odds': odds,
                    'stake': stake,
                    'outcome': outcome,
                    'profit_loss': profit_loss,
                    'result_amount': result_amount,
                    'home_score': game_data['home_score'],
                    'away_score': game_data['away_score']
                })

                found_match = True
                break

        if not found_match:
            print(f"   ❌ No NBA match found for this bet")

    return mappings

def settle_bets_manually(betting_db, mappings):
    """
    Esegue il settlement manuale delle scommesse.
    """
    print(f"\n🔧 Settling {len(mappings)} bets...")

    settled_count = 0

    for mapping in mappings:
        bet_id = mapping['bet_id']
        outcome = mapping['outcome'].lower()
        result_amount = mapping['result_amount']
        profit_loss = mapping['profit_loss']

        print(f"\n⚡ Settling bet: {bet_id}")
        print(f"   Outcome: {outcome}")
        print(f"   Return: €{result_amount:.2f}")
        print(f"   P&L: €{profit_loss:.2f}")

        try:
            # Update bet status using our betting_db manager
            success = betting_db.settle_bet(bet_id, outcome, None)  # We don't need final_score parameter here

            if success:
                settled_count += 1
                print(f"   ✅ Bet settled successfully!")
            else:
                print(f"   ❌ Failed to settle bet")

        except Exception as e:
            print(f"   ❌ Error settling bet: {e}")

    return settled_count

def main():
    """
    Funzione principale di test e fix.
    """
    print("🎯 NBA BETTING SETTLEMENT SYSTEM - TEST & FIX")
    print("=" * 50)

    try:
        # Initialize database
        betting_db = BettingDatabaseManager()

        # Step 1: Get real NBA scores
        print("\n📡 Step 1: Getting real NBA scores...")
        real_scores = get_real_nba_scores()

        print(f"\n📊 Found {len(real_scores)} games with real scores:")
        for game_id, data in real_scores.items():
            print(f"   {game_id}: {data['game_desc']} - {data['total_points']} points")

        # Step 2: Map manual bets to NBA games
        print(f"\n🔗 Step 2: Mapping manual bets to NBA games...")
        mappings = map_manual_bets_to_nba_games(betting_db, real_scores)

        # Step 3: Show summary
        print(f"\n📋 Step 3: Settlement Summary")
        print(f"   Total bets to settle: {len(mappings)}")

        if mappings:
            outcomes = {}
            total_pl = 0
            for mapping in mappings:
                outcome = mapping['outcome']
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
                total_pl += mapping['profit_loss']

            print(f"   Outcomes:")
            for outcome, count in outcomes.items():
                print(f"     {outcome}: {count} bets")
            print(f"   Total P&L: €{total_pl:.2f}")

        # Step 4: Execute settlement
        if mappings:
            print(f"\n⚡ Step 4: Executing settlement...")
            print(f"\n⚠️  AUTO-SETTLEMENT: Settling {len(mappings)} bets and updating bankroll!")

            settled_count = settle_bets_manually(betting_db, mappings)
            print(f"\n✅ Settlement complete: {settled_count}/{len(mappings)} bets settled")
        else:
            print("\nℹ️ No bets to settle")

        # Step 5: Show final bankroll status
        print(f"\n💰 Final Bankroll Status:")
        bankroll = betting_db.get_bankroll_status()
        print(f"   Current: €{bankroll.get('current_bankroll', 0):.2f}")
        print(f"   Available: €{bankroll.get('available_bankroll', 0):.2f}")
        print(f"   Total bets: {bankroll.get('total_bets', 0)}")
        print(f"   Pending bets: {bankroll.get('pending_bets_count', 0)}")

    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()