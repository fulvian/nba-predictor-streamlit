#!/usr/bin/env python3
"""
Script di verifica completezza dati NBA - Da eseguire dopo il completamento dello scraping
Confronta i dati scrappati con le stagioni NBA ufficiali
"""

import json
from pathlib import Path
from datetime import datetime

# Informazioni ufficiali stagioni NBA
SEASONS_INFO = {
    "2020_2021": {
        "expected_games": 1080,  # COVID-shortened: 72 games/team × 30 teams / 2
        "start_date": "2020-12-22",
        "end_date": "2021-05-16",
        "playoffs_start": "2021-05-22",
        "notes": "COVID-shortened season (72 games per team)",
    },
    "2021_2022": {
        "expected_games": 1230,  # Full season: 82 games/team × 30 teams / 2
        "start_date": "2021-10-19",
        "end_date": "2022-04-10",
        "playoffs_start": "2022-04-16",
        "notes": "Full season",
    },
    "2022_2023": {
        "expected_games": 1230,
        "start_date": "2022-10-18",
        "end_date": "2023-04-09",
        "playoffs_start": "2023-04-15",
        "notes": "Full season",
    },
    "2023_2024": {
        "expected_games": 1230,
        "start_date": "2023-10-24",
        "end_date": "2024-04-14",
        "playoffs_start": "2024-04-20",
        "notes": "Full season",
    },
    "2024_2025": {
        "expected_games": 1230,
        "start_date": "2024-10-22",
        "end_date": "2025-04-13",
        "playoffs_start": "2025-04-19",
        "notes": "Full season (in corso)",
    },
}


def analyze_season(season_name: str, data: list) -> dict:
    """Analizza una stagione e restituisce metriche di completezza."""
    info = SEASONS_INFO[season_name]

    # Conta partite con dati Over/Under
    matches_with_ou = sum(1 for m in data if m.get("over_under_market"))

    # Calcola coverage
    actual_games = len(data)
    expected_games = info["expected_games"]
    coverage_pct = (actual_games / expected_games) * 100
    missing_games = expected_games - actual_games

    # Analizza date
    dates = [m.get("match_date") for m in data if m.get("match_date")]
    date_range = f"{min(dates)[:10]} → {max(dates)[:10]}" if dates else "N/A"

    return {
        "season": season_name.replace("_", "-"),
        "actual_games": actual_games,
        "expected_games": expected_games,
        "coverage_pct": coverage_pct,
        "missing_games": missing_games,
        "matches_with_ou": matches_with_ou,
        "ou_coverage_pct": (matches_with_ou / actual_games * 100)
        if actual_games > 0
        else 0,
        "date_range": date_range,
        "notes": info["notes"],
    }


def main():
    odds_dir = Path("/Users/fulvioventura/nba-predictor-streamlit/data/odds")

    print("=" * 80)
    print("📊 VERIFICA COMPLETEZZA DATI NBA - REPORT DETTAGLIATO")
    print("=" * 80)
    print()

    all_analyses = []
    total_actual = 0
    total_expected = 0

    for season_file in sorted(SEASONS_INFO.keys()):
        file_path = odds_dir / f"scraped_{season_file}.json"

        if not file_path.exists():
            print(f"⚠️  MANCANTE: {season_file.replace('_', '-')}")
            continue

        with open(file_path) as f:
            data = json.load(f)

        analysis = analyze_season(season_file, data)
        all_analyses.append(analysis)

        total_actual += analysis["actual_games"]
        total_expected += analysis["expected_games"]

        # Stampa dettagli stagione
        status = "✅" if analysis["coverage_pct"] >= 95 else "⚠️"
        print(f"{status} STAGIONE {analysis['season']} ({analysis['notes']})")
        print(
            f"   Partite: {analysis['actual_games']}/{analysis['expected_games']} ({analysis['coverage_pct']:.1f}%)"
        )
        print(f"   Mancanti: {analysis['missing_games']}")
        print(
            f"   Con O/U: {analysis['matches_with_ou']} ({analysis['ou_coverage_pct']:.1f}%)"
        )
        print(f"   Range: {analysis['date_range']}")
        print()

    # Riepilogo totale
    print("=" * 80)
    print("📈 TOTALI COMPLESSIVI:")
    print(
        f"   Partite totali: {total_actual}/{total_expected} ({total_actual / total_expected * 100:.1f}%)"
    )
    print(f"   Mancanti: {total_expected - total_actual}")
    print()

    # Identificazione stagioni problematiche
    problematic = [a for a in all_analyses if a["coverage_pct"] < 95]
    if problematic:
        print("⚠️  STAGIONI CON COVERAGE < 95%:")
        for a in problematic:
            print(
                f"   - {a['season']}: {a['coverage_pct']:.1f}% ({a['missing_games']} partite mancanti)"
            )
        print()
        print("💡 RACCOMANDAZIONI:")
        print("   1. Verificare i log per errori durante lo scraping")
        print("   2. Controllare OddsPortal per disponibilità dati")
        print(
            "   3. Eventualmente integrare con fonti alternative (es. Kaggle dataset)"
        )
    else:
        print("✅ Tutte le stagioni hanno coverage >= 95%")

    print("=" * 80)


if __name__ == "__main__":
    main()
