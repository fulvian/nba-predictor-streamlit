import json
from pathlib import Path
from datetime import datetime

# Configurazione date ufficiali per filtro rigoroso
SEASON_CONFIG = {
    "2020_2021": {"start": "2020-12-01", "end": "2021-08-01"},
    "2021_2022": {"start": "2021-10-01", "end": "2022-07-01"},
    "2022_2023": {"start": "2022-09-01", "end": "2023-07-01"},
    "2023_2024": {"start": "2023-09-01", "end": "2024-07-01"},
    "2024_2025": {"start": "2024-09-01", "end": "2025-07-01"},
    "2025_2026": {"start": "2025-09-01", "end": "2026-07-01"},
}

DATA_DIR = Path("/Users/fulvioventura/nba-predictor-streamlit/data/odds")


def clean_and_split_data():
    print("🧹 INIZIO PULIZIA E DEDUPLICAZIONE DATI...")

    # 1. Carica TUTTI i dati in un unico calderone per poi ridistribuirli
    all_matches = []
    source_files = list(DATA_DIR.glob("scraped_*.json"))

    print(f"📖 Caricamento da {len(source_files)} file...")
    for f in source_files:
        try:
            data = json.load(open(f))
            print(f"   - {f.name}: {len(data)} records")
            all_matches.extend(data)
        except Exception as e:
            print(f"   ⚠️ Errore lettura {f.name}: {e}")

    print(f"📦 Totale records grezzi: {len(all_matches)}")

    # 2. Deduplicazione (chiave: data + home + away)
    unique_matches = {}
    duplicates = 0

    for m in all_matches:
        # Crea chiave univoca
        date = m.get("match_date")
        home = m.get("home_team")
        away = m.get("away_team")

        if not date or not home or not away:
            continue

        key = f"{date}_{home}_{away}"

        # Mantieni il record con più dati (es. se uno ha O/U field e l'altro no)
        if key in unique_matches:
            existing = unique_matches[key]
            # Logica semplice: se il nuovo ha field over_under e il vecchio no, sovrascrivi
            if not existing.get("over_under_market") and m.get("over_under_market"):
                unique_matches[key] = m
            duplicates += 1
        else:
            unique_matches[key] = m

    print(
        f"✨ Records unici dopo deduplica: {len(unique_matches)} (rimossi {duplicates} duplicati)"
    )

    # 3. Ridistribuzione nelle stagioni corrette
    season_buckets = {k: [] for k in SEASON_CONFIG.keys()}
    unsorted = []

    for key, m in unique_matches.items():
        date_str = m.get("match_date")[:10]  # YYYY-MM-DD
        assigned = False

        for season, rng in SEASON_CONFIG.items():
            if rng["start"] <= date_str <= rng["end"]:
                season_buckets[season].append(m)
                assigned = True
                break

        if not assigned:
            unsorted.append(m)

    # 4. Salvataggio
    print("\n💾 SALVATAGGIO FILE PULITI:")
    for season, matches in season_buckets.items():
        out_file = DATA_DIR / f"scraped_{season}_CLEAN.json"

        # Ordina per data
        matches.sort(key=lambda x: x.get("match_date", ""))

        with open(out_file, "w") as f:
            json.dump(matches, f, indent=2)

        print(f"   ✅ {season}: {len(matches)} partite -> {out_file.name}")

    if unsorted:
        print(
            f"\n⚠️ {len(unsorted)} partite fuori target salvate in 'unsorted_matches.json'"
        )
        with open(DATA_DIR / "unsorted_matches.json", "w") as f:
            json.dump(unsorted, f, indent=2)


if __name__ == "__main__":
    clean_and_split_data()
