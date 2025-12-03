import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_dynamic_team_lists(df, n_games=5, top_n=10, bottom_n=8):
    """
    Calculate high and low performance teams based on average points in last n_games.
    """
    try:
        # Ensure date is datetime
        if "GAME_DATE_EST" in df.columns:
            df["GAME_DATE_EST"] = pd.to_datetime(df["GAME_DATE_EST"])
            df = df.sort_values("GAME_DATE_EST", ascending=False)

        team_stats = {}

        # Get all unique teams
        all_teams = set(df["home_team"].unique()) | set(df["away_team"].unique())

        for team in all_teams:
            # Get last n games for this team
            team_games = df[(df["home_team"] == team) | (df["away_team"] == team)].head(
                n_games
            )

            if len(team_games) < 3:  # Skip teams with too few games
                continue

            scores = []
            for _, game in team_games.iterrows():
                if game["home_team"] == team:
                    scores.append(game["HOME_SCORE"])
                else:
                    scores.append(game["AWAY_SCORE"])

            avg_score = sum(scores) / len(scores)
            team_stats[team] = avg_score

        # Sort teams by average score
        sorted_teams = sorted(team_stats.items(), key=lambda x: x[1], reverse=True)

        high_perf = [t[0] for t in sorted_teams[:top_n]]
        low_perf = [t[0] for t in sorted_teams[-bottom_n:]]

        return high_perf, low_perf, sorted_teams

    except Exception as e:
        logger.error(f"Error calculating dynamic teams: {e}")
        return [], [], []


def main():
    data_path = Path("data/nba_data_with_mu_sigma_for_ml.csv")
    if not data_path.exists():
        print(f"File not found: {data_path}")
        return

    # Team ID mapping (copied from UnifiedHybridPipeline)
    team_id_to_name = {
        1610612737: "Atlanta Hawks",
        1610612738: "Boston Celtics",
        1610612739: "Cleveland Cavaliers",
        1610612740: "New Orleans Pelicans",
        1610612741: "Chicago Bulls",
        1610612742: "Dallas Mavericks",
        1610612743: "Denver Nuggets",
        1610612744: "Golden State Warriors",
        1610612745: "Houston Rockets",
        1610612746: "Los Angeles Clippers",
        1610612747: "Los Angeles Lakers",
        1610612748: "Miami Heat",
        1610612749: "Milwaukee Bucks",
        1610612750: "Minnesota Timberwolves",
        1610612751: "Brooklyn Nets",
        1610612752: "New York Knicks",
        1610612753: "Orlando Magic",
        1610612754: "Indiana Pacers",
        1610612755: "Philadelphia 76ers",
        1610612756: "Phoenix Suns",
        1610612757: "Portland Trail Blazers",
        1610612758: "Sacramento Kings",
        1610612759: "San Antonio Spurs",
        1610612760: "Oklahoma City Thunder",
        1610612761: "Toronto Raptors",
        1610612762: "Utah Jazz",
        1610612763: "Memphis Grizzlies",
        1610612764: "Washington Wizards",
        1610612765: "Detroit Pistons",
        1610612766: "Charlotte Hornets",
    }

    print("Loading data...")
    df = pd.read_csv(data_path)

    print(f"Total games: {len(df)}")

    # Ensure date is datetime
    if "GAME_DATE_EST" in df.columns:
        df["GAME_DATE_EST"] = pd.to_datetime(df["GAME_DATE_EST"])

    # Use IDs
    all_teams_ids = set(df["HOME_TEAM_ID"].unique()) | set(df["AWAY_TEAM_ID"].unique())

    team_stats = {}

    n_games = 5  # Default value from get_dynamic_team_lists
    top_n = 10  # Default value from get_dynamic_team_lists
    bottom_n = 8  # Default value from get_dynamic_team_lists

    for team_id in all_teams_ids:
        # Get last n games for this team
        team_games = (
            df[(df["HOME_TEAM_ID"] == team_id) | (df["AWAY_TEAM_ID"] == team_id)]
            .sort_values("GAME_DATE_EST", ascending=False)
            .head(n_games)
        )

        if len(team_games) < 3:
            continue

        scores = []
        for _, game in team_games.iterrows():
            if game["HOME_TEAM_ID"] == team_id:
                scores.append(game["HOME_SCORE"])
            else:
                scores.append(game["AWAY_SCORE"])

        avg_score = sum(scores) / len(scores)
        team_name = team_id_to_name.get(team_id, f"Unknown ID {team_id}")
        team_stats[team_name] = avg_score

    # Calculate percentiles
    import numpy as np

    avg_scores = list(team_stats.values())
    if not avg_scores:
        print("No team stats calculated.")
        return

    p75 = np.percentile(avg_scores, 75)
    p25 = np.percentile(avg_scores, 25)

    print(f"\n📊 Statistics (Last {n_games} Games):")
    print(f"   75th Percentile (High Perf Threshold): {p75:.1f} ppg")
    print(f"   25th Percentile (Low Perf Threshold):  {p25:.1f} ppg")
    print(f"   Average League Score: {np.mean(avg_scores):.1f} ppg")

    high_perf = [team for team, score in team_stats.items() if score >= p75]
    low_perf = [team for team, score in team_stats.items() if score <= p25]

    # Sort for display
    high_perf.sort(key=lambda x: team_stats[x], reverse=True)
    low_perf.sort(key=lambda x: team_stats[x])  # Ascending for low perf

    print(f"\n🏆 High Performance Teams (>= {p75:.1f} ppg):")
    for i, team in enumerate(high_perf, 1):
        print(f"{i}. {team}: {team_stats[team]:.1f} ppg")

    print(f"\n📉 Low Performance Teams (<= {p25:.1f} ppg):")
    for i, team in enumerate(low_perf, 1):
        print(f"{i}. {team}: {team_stats[team]:.1f} ppg")


if __name__ == "__main__":
    main()
