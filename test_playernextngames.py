#!/usr/bin/env python3
"""
🎯 TEST PlayerNextNGames - Soluzione anti-timeout per trovare partite di oggi
Usa PlayerNextNGames endpoint per trovare partite programmate senza timeouts
"""

from nba_api.stats.endpoints import playernextngames
from nba_api.stats.static import players
from datetime import datetime, date, timedelta
import time

def get_player_id_by_name(player_name):
    """Ottieni player ID dal nome"""
    try:
        players_list = players.get_players()
        for player in players_list:
            if player['full_name'].lower() == player_name.lower():
                return player['id']
        return None
    except Exception as e:
        print(f"   ❌ Errore ricerca player ID: {e}")
        return None

def test_playernextngames_for_today_games():
    """
    Test PlayerNextNGames per trovare le 2 partite di oggi:
    1. Oklahoma City Thunder @ Indiana Pacers
    2. Denver Nuggets @ Golden State Warriors
    """
    print("🏀 TEST PlayerNextNGames - Soluzione anti-timeout")
    print("=" * 60)

    # Giocatori chiave delle partite di oggi
    key_players = [
        "Shai Gilgeous-Alexander",      # Oklahoma City Thunder
        "Tyrese Haliburton",            # Indiana Pacers
        "Nikola Jokić",                 # Denver Nuggets
        "Stephen Curry"                 # Golden State Warriors
    ]

    today = date.today()
    today_str = today.strftime('%Y-%m-%d')
    season = "2025-26"  # Stagione NBA corrente

    print(f"📅 Ricerca partite per oggi: {today_str}")
    print(f"🏀 Season: {season}")
    print(f"👥 Giocatori chiave da testare: {len(key_players)}")

    all_games = []
    game_ids_found = set()

    for i, player_name in enumerate(key_players, 1):
        print(f"\n{i}. 🏀 Testando giocatore: {player_name}")

        # Ottieni player ID
        player_id = get_player_id_by_name(player_name)
        if not player_id:
            print(f"   ❌ Player ID non trovato per {player_name}")
            continue

        print(f"   ✅ Player ID trovato: {player_id}")

        # Rate limiting
        time.sleep(0.5)

        try:
            print(f"   📡 Chiamata PlayerNextNGames API...")

            # Chiama PlayerNextNGames API
            next_n_games = playernextngames.PlayerNextNGames(
                player_id=player_id,
                season_all=season,
                season_type_all_star="Regular Season",
                number_of_games=10
            )

            games_df = next_n_games.get_data_frames()[0]
            print(f"   📊 Risposta API: {len(games_df)} games found")

            # Filtra per partite di oggi
            today_games = []
            for _, game in games_df.iterrows():
                game_date_str = game['GAME_DATE']

                # Formato data NBA: "2025-10-23"
                try:
                    game_date = datetime.strptime(game_date_str, '%Y-%m-%d').date()

                    if game_date == today:
                        game_id = game['GAME_ID']
                        if game_id not in game_ids_found:
                            game_info = {
                                'game_id': game_id,
                                'date': game_date_str,
                                'away_team': game.get('VISITOR_TEAM_NAME', 'Unknown'),
                                'home_team': game.get('HOME_TEAM_NAME', 'Unknown'),
                                'player_found': player_name,
                                'source': 'PlayerNextNGames API'
                            }
                            today_games.append(game_info)
                            game_ids_found.add(game_id)
                            print(f"   ✅ PARTITA TROVATA: {game_info['away_team']} @ {game_info['home_team']}")
                        else:
                            print(f"   ℹ️ Partita già trovata: {game.get('VISITOR_TEAM_NAME', 'Unknown')} @ {game.get('HOME_TEAM_NAME', 'Unknown')}")
                except Exception as e:
                    print(f"   ⚠️ Errore parsing data: {game_date_str} - {e}")
                    continue

            all_games.extend(today_games)

        except Exception as e:
            print(f"   ❌ Errore PlayerNextNGames API: {e}")
            continue

    # Risultati finali
    print(f"\n🎉 RISULTATI FINALI:")
    print(f"   📊 Partite uniche trovate: {len(all_games)}")

    if all_games:
        print(f"   🏀 Partite di oggi ({today_str}):")
        for i, game in enumerate(all_games, 1):
            print(f"      {i}. {game['away_team']} @ {game['home_team']}")
            print(f"         📡 Scoperto tramite: {game['player_found']}")
            print(f"         🔗 Game ID: {game['game_id']}")

        # Verifica se abbiamo trovato le partite attese
        teams_found = []
        for game in all_games:
            teams_found.extend([game['away_team'], game['home_team']])

        expected_teams = [
            'Oklahoma City Thunder', 'Indiana Pacers',
            'Denver Nuggets', 'Golden State Warriors'
        ]

        print(f"\n🔍 VERIFICA SQUADRE ATTESE:")
        for team in expected_teams:
            if team in teams_found:
                print(f"   ✅ {team}")
            else:
                print(f"   ❌ {team} - MANCANTE")

        if len(all_games) == 2 and all(team in teams_found for team in expected_teams):
            print(f"\n🎉 SUCCESSO COMPLETO!")
            print(f"   ✅ PlayerNextNGames ha trovato le 2 partite reali!")
            print(f"   ✅ Soluzone funzionante senza timeouts!")
            return True
        else:
            print(f"\n⚠️ Risultato parziale:")
            print(f"   ✅ Trovate {len(all_games)} partite")
            print(f"   📊 API PlayerNextNGames funzionante")
            return False
    else:
        print(f"   ❌ Nessuna partita trovata per oggi")
        return False

if __name__ == "__main__":
    success = test_playernextngames_for_today_games()

    if success:
        print(f"\n✅ PlayerNextNGames è la soluzione corretta!")
        print(f"   🚀 Implementare in data_provider.py")
    else:
        print(f"\n⚠️ PlayerNextNGames需要调整或需要其他方法")