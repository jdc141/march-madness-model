"""Build ML training data from historical tournament results + KenPom archive ratings.

Usage:
    python scripts/build_training_data.py

Requires:
    - KENPOM_BEARER_TOKEN set in environment (or .env file)
    - Historical tournament results CSV at data/historical_results.csv
      (download from Kaggle: "March Madness Historical DataSet 2002-2026")

The script pairs each historical tournament game with pre-tournament KenPom
ratings for both teams, computes matchup features, and writes the result to
data/training_data.csv.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from utils.features import ML_FEATURE_NAMES

_DATA_DIR = Path(__file__).parent.parent / "data"
_RESULTS_FILE = _DATA_DIR / "historical_results.csv"
_OUTPUT_FILE = _DATA_DIR / "training_data.csv"

# Pre-tournament snapshot dates (Selection Sunday, roughly)
_SEASON_SNAPSHOT_DATES = {
    2010: "2010-03-14",
    2011: "2011-03-13",
    2012: "2012-03-11",
    2013: "2013-03-17",
    2014: "2014-03-16",
    2015: "2015-03-15",
    2016: "2016-03-13",
    2017: "2017-03-12",
    2018: "2018-03-11",
    2019: "2019-03-17",
    2021: "2021-03-14",  # no 2020 tournament
    2022: "2022-03-13",
    2023: "2023-03-12",
    2024: "2024-03-17",
    2025: "2025-03-16",
}


def _get_kenpom_archive(date_str: str) -> dict[str, dict]:
    """Fetch KenPom ratings archive for a given date."""
    from kenpom import KenpomClient

    token = os.environ.get("KENPOM_BEARER_TOKEN", "")
    if not token:
        raise ValueError("KENPOM_BEARER_TOKEN not set")

    client = KenpomClient(bearer_token=token)
    ratings = client.get_ratings_archive_by_date(date=date_str)

    team_lookup = {}
    for r in ratings:
        name = r.get("team_name", r.get("team", ""))
        if name:
            team_lookup[name.lower()] = r
    return team_lookup


def _build_feature_row(team_a: dict, team_b: dict) -> list[float]:
    """Build a feature vector from two team stat dicts."""

    def g(d, k, default=0.0):
        v = d.get(k, default)
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    return [
        g(team_a, "adj_em") - g(team_b, "adj_em"),
        g(team_a, "adj_o") - g(team_b, "adj_d"),
        g(team_b, "adj_o") - g(team_a, "adj_d"),
        g(team_a, "tempo") - g(team_b, "tempo"),
        g(team_b, "seed", 8) - g(team_a, "seed", 8),
        g(team_a, "sos") - g(team_b, "sos"),
        g(team_a, "luck") - g(team_b, "luck"),
    ]


def main():
    if not _RESULTS_FILE.exists():
        print(f"ERROR: Historical results file not found at {_RESULTS_FILE}")
        print("Download from Kaggle and place at data/historical_results.csv")
        print("Expected columns: season, round, team_a, seed_a, score_a, team_b, seed_b, score_b")
        sys.exit(1)

    rows_out = []
    seasons_processed = 0
    games_processed = 0

    with open(_RESULTS_FILE) as f:
        reader = csv.DictReader(f)
        games_by_season: dict[int, list[dict]] = {}
        for row in reader:
            try:
                season = int(row.get("season", 0))
            except ValueError:
                continue
            if season not in _SEASON_SNAPSHOT_DATES:
                continue
            games_by_season.setdefault(season, []).append(row)

    for season in sorted(games_by_season.keys()):
        snapshot_date = _SEASON_SNAPSHOT_DATES[season]
        print(f"Processing {season} (snapshot: {snapshot_date})...")

        try:
            kenpom = _get_kenpom_archive(snapshot_date)
        except Exception as e:
            print(f"  Skipping {season}: {e}")
            continue

        seasons_processed += 1

        for game in games_by_season[season]:
            team_a_name = game.get("team_a", "").strip().lower()
            team_b_name = game.get("team_b", "").strip().lower()

            ka = kenpom.get(team_a_name)
            kb = kenpom.get(team_b_name)
            if ka is None or kb is None:
                continue

            try:
                score_a = int(game.get("score_a", 0))
                score_b = int(game.get("score_b", 0))
            except ValueError:
                continue

            features = _build_feature_row(ka, kb)
            team_a_won = 1 if score_a > score_b else 0

            rows_out.append(features + [team_a_won, season])
            games_processed += 1

        # Rate limit: KenPom API
        time.sleep(1)

    if not rows_out:
        print("No training data generated. Check input file format and KenPom access.")
        sys.exit(1)

    _OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ML_FEATURE_NAMES + ["team_a_won", "season"])
        writer.writerows(rows_out)

    print(f"\nDone: {games_processed} games across {seasons_processed} seasons")
    print(f"Output: {_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
