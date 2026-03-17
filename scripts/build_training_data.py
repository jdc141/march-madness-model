"""Build ML training data from real historical tournament results + KenPom archive snapshots.

Fetches actual NCAA tournament game results from ESPN for each season,
pairs them with pre-tournament KenPom ratings from the archive API,
computes feature differentials, and writes the result to data/training_data.csv.

Usage:
    python scripts/build_training_data.py

Requires:
    - KENPOM_BEARER_TOKEN set in environment (or .env file)
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from utils.features import ML_FEATURE_NAMES
from utils.team_names import normalize

_DATA_DIR = Path(__file__).parent.parent / "data"
_OUTPUT_FILE = _DATA_DIR / "training_data.csv"

# Pre-tournament snapshot dates (Selection Sunday) and tournament date ranges
_SEASONS = {
    2014: {"snapshot": "2014-03-16", "dates": ["20140318", "20140320", "20140321", "20140322", "20140323", "20140327", "20140328", "20140329", "20140330", "20140405", "20140407"]},
    2015: {"snapshot": "2015-03-15", "dates": ["20150317", "20150319", "20150320", "20150321", "20150322", "20150326", "20150327", "20150328", "20150329", "20150404", "20150406"]},
    2016: {"snapshot": "2016-03-13", "dates": ["20160315", "20160317", "20160318", "20160319", "20160320", "20160324", "20160325", "20160326", "20160327", "20160402", "20160404"]},
    2017: {"snapshot": "2017-03-12", "dates": ["20170314", "20170316", "20170317", "20170318", "20170319", "20170323", "20170324", "20170325", "20170326", "20170401", "20170403"]},
    2018: {"snapshot": "2018-03-11", "dates": ["20180313", "20180315", "20180316", "20180317", "20180318", "20180322", "20180323", "20180324", "20180325", "20180331", "20180402"]},
    2019: {"snapshot": "2019-03-17", "dates": ["20190319", "20190321", "20190322", "20190323", "20190324", "20190328", "20190329", "20190330", "20190331", "20190406", "20190408"]},
    2021: {"snapshot": "2021-03-14", "dates": ["20210318", "20210319", "20210320", "20210321", "20210322", "20210327", "20210328", "20210329", "20210330", "20210403", "20210405"]},
    2022: {"snapshot": "2022-03-13", "dates": ["20220315", "20220317", "20220318", "20220319", "20220320", "20220324", "20220325", "20220326", "20220327", "20220402", "20220404"]},
    2023: {"snapshot": "2023-03-12", "dates": ["20230314", "20230316", "20230317", "20230318", "20230319", "20230323", "20230324", "20230325", "20230326", "20230401", "20230403"]},
    2024: {"snapshot": "2024-03-17", "dates": ["20240319", "20240321", "20240322", "20240323", "20240324", "20240328", "20240329", "20240330", "20240331", "20240406", "20240408"]},
    2025: {"snapshot": "2025-03-16", "dates": ["20250318", "20250320", "20250321", "20250322", "20250323", "20250327", "20250328", "20250329", "20250330", "20250405", "20250407"]},
}

_ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"

# KenPom archive field mapping
_ARCHIVE_FIELD_MAP = {
    "TeamName": "team",
    "AdjEM": "adj_em",
    "AdjOE": "adj_o",
    "AdjDE": "adj_d",
    "AdjTempo": "tempo",
    "Seed": "seed",
    "RankAdjEM": "rank",
}


def _get_kenpom_archive(date_str: str) -> dict[str, dict]:
    """Fetch KenPom ratings archive for a given date, keyed by normalized name."""
    from kenpom import KenpomClient

    token = os.environ.get("KENPOM_BEARER_TOKEN", "")
    if not token:
        raise ValueError("KENPOM_BEARER_TOKEN not set")

    client = KenpomClient(bearer_token=token)
    ratings = client.get_ratings_archive_by_date(date=date_str)

    team_lookup = {}
    for r in ratings:
        mapped = {}
        for old_key, new_key in _ARCHIVE_FIELD_MAP.items():
            if old_key in r:
                mapped[new_key] = r[old_key]

        raw_name = r.get("TeamName", "")
        if raw_name:
            mapped["team"] = raw_name
            norm = normalize(raw_name)
            team_lookup[norm] = mapped
            team_lookup[raw_name.lower()] = mapped

    return team_lookup


def _fetch_espn_tournament_games(dates: list[str]) -> list[dict]:
    """Fetch completed NCAA tournament games from ESPN for given dates."""
    games = []
    seen_ids = set()

    for date_str in dates:
        try:
            resp = requests.get(_ESPN_URL, params={
                "dates": date_str,
                "groups": 50,
                "limit": 100,
            }, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    ESPN error for {date_str}: {e}")
            continue

        for event in data.get("events", []):
            eid = event.get("id", "")
            if eid in seen_ids:
                continue
            seen_ids.add(eid)

            comp = event.get("competitions", [{}])[0]
            status_type = comp.get("status", {}).get("type", {}).get("name", "")
            if status_type != "STATUS_FINAL":
                continue

            # Only NCAA tournament
            season_type = event.get("season", {}).get("type", 0)
            notes = [n.get("headline", "") for n in comp.get("notes", [])]
            is_tourney = season_type == 3 or any(
                kw in " ".join(notes).lower()
                for kw in ["ncaa", "tournament", "march madness", "sweet 16", "elite", "final four"]
            )
            if not is_tourney:
                continue

            competitors = comp.get("competitors", [])
            if len(competitors) != 2:
                continue

            teams = {}
            for t in competitors:
                ha = t.get("homeAway", "")
                teams[ha] = {
                    "name": t.get("team", {}).get("displayName", ""),
                    "short": t.get("team", {}).get("shortDisplayName", ""),
                    "score": int(t.get("score", 0)),
                    "winner": t.get("winner", False),
                    "seed": t.get("curatedRank", {}).get("current"),
                }

            if "home" in teams and "away" in teams:
                games.append({
                    "away": teams["away"],
                    "home": teams["home"],
                })

        time.sleep(0.3)

    return games


def _g(d: dict, k: str, default: float = 0.0) -> float:
    v = d.get(k, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _build_feature_row(team_a: dict, team_b: dict) -> list[float]:
    """Build a 20-feature vector. Archive only has core efficiency stats,
    so four factors / shooting / height features default to 0.0 (no signal,
    won't help or hurt the model)."""
    return [
        _g(team_a, "adj_em") - _g(team_b, "adj_em"),
        _g(team_a, "adj_o", 100) - _g(team_b, "adj_d", 100),
        _g(team_b, "adj_o", 100) - _g(team_a, "adj_d", 100),
        _g(team_a, "tempo", 67.5) - _g(team_b, "tempo", 67.5),
        _g(team_b, "seed", 8) - _g(team_a, "seed", 8),
        0.0,  # sos (not in archive)
        0.0,  # luck (not in archive)
        0.0,  # off_efg_diff
        0.0,  # def_efg_diff
        0.0,  # off_to_diff
        0.0,  # off_orb_diff
        0.0,  # fg3_pct_diff
        0.0,  # ft_pct_diff
        0.0,  # experience_diff
        0.0,  # avg_hgt_diff
        0.0,  # bench_diff
        0.0,  # continuity_diff
        0.0,  # stl_rate_diff
        0.0,  # block_pct_diff
        0.0,  # ast_rate_diff
    ]


def main():
    rows_out = []
    total_games = 0
    seasons_ok = 0

    for season in sorted(_SEASONS.keys()):
        cfg = _SEASONS[season]
        print(f"\n{'='*50}")
        print(f"Season {season} (snapshot: {cfg['snapshot']})")
        print(f"{'='*50}")

        # Fetch KenPom archive
        try:
            kenpom = _get_kenpom_archive(cfg["snapshot"])
            print(f"  KenPom: {len(kenpom)} team entries loaded")
        except Exception as e:
            print(f"  KenPom FAILED: {e} — skipping season")
            continue

        # Fetch ESPN results
        games = _fetch_espn_tournament_games(cfg["dates"])
        print(f"  ESPN: {len(games)} completed tournament games found")

        if not games:
            print("  No games — skipping")
            continue

        matched = 0
        unmatched_teams = set()

        for game in games:
            away = game["away"]
            home = game["home"]

            # Try multiple name variants to match
            ka = None
            for name_variant in [away["name"], away["short"], away["name"].lower()]:
                norm = normalize(name_variant)
                ka = kenpom.get(norm) or kenpom.get(name_variant.lower())
                if ka:
                    break

            kb = None
            for name_variant in [home["name"], home["short"], home["name"].lower()]:
                norm = normalize(name_variant)
                kb = kenpom.get(norm) or kenpom.get(name_variant.lower())
                if kb:
                    break

            if ka is None:
                unmatched_teams.add(away["name"])
                continue
            if kb is None:
                unmatched_teams.add(home["name"])
                continue

            # Assign seeds from ESPN if KenPom didn't have them
            if away.get("seed") and away["seed"] != 99:
                ka["seed"] = away["seed"]
            if home.get("seed") and home["seed"] != 99:
                kb["seed"] = home["seed"]

            features = _build_feature_row(ka, kb)
            team_a_won = 1 if away["winner"] else 0

            rows_out.append(features + [team_a_won, season])
            matched += 1

        total_games += matched
        seasons_ok += 1
        print(f"  Matched: {matched} games")
        if unmatched_teams:
            print(f"  Unmatched teams: {', '.join(sorted(unmatched_teams))}")

        # Rate limit between seasons
        time.sleep(1.5)

    if not rows_out:
        print("\nNo training data generated!")
        sys.exit(1)

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ML_FEATURE_NAMES + ["team_a_won", "season"])
        writer.writerows(rows_out)

    print(f"\n{'='*50}")
    print(f"DONE: {total_games} real tournament games across {seasons_ok} seasons")
    print(f"Output: {_OUTPUT_FILE}")
    print(f"Win rate (away team): {sum(r[-2] for r in rows_out) / len(rows_out):.3f}")


if __name__ == "__main__":
    main()
