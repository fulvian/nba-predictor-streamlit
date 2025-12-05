from nba_api.stats.endpoints import scoreboardv2
import pandas as pd


def get_2024_schedule():
    print("Fetching 2024-12-03 Schedule...")
    try:
        board = scoreboardv2.ScoreboardV2(game_date="2024-12-03")
        games = board.game_header.get_dict()["data"]
        # game[2] is GAME_ID, game[3] is STATUS_ID
        # We need team IDs to get names, or use LineScore
        line_score = board.line_score.get_dict()["data"]
        # LineScore has TEAM_ABBREVIATION (index 4) and TEAM_NAME (index 5 usually)
        # Let's print what we find
        print(f"Found {len(games)} games.")

        # Map Game ID to Teams
        game_teams = {}
        for ls in line_score:
            game_id = ls[2]  # GAME_ID
            team_abbr = ls[4]  # TEAM_ABBREVIATION
            if game_id not in game_teams:
                game_teams[game_id] = []
            game_teams[game_id].append(team_abbr)

        for gid, teams in game_teams.items():
            print(f"Game {gid}: {teams}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    get_2024_schedule()
