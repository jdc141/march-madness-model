"""Build ML training data from real historical tournament results + full KenPom data.

Fetches actual NCAA tournament game results from ESPN for each season,
pairs them with pre-tournament KenPom data (ratings + four factors + height
+ misc stats), computes feature differentials, and writes training_data.csv.

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

from utils.team_names import normalize

_DATA_DIR = Path(__file__).parent.parent / "data"
_OUTPUT_FILE = _DATA_DIR / "training_data.csv"

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

# Archive ratings field map
_RATINGS_MAP = {
    "TeamName": "team", "AdjEM": "adj_em", "AdjOE": "adj_o",
    "AdjDE": "adj_d", "AdjTempo": "tempo", "Seed": "seed",
}

# Four factors field map
_FF_MAP = {
    "eFG_Pct": "off_efg", "TO_Pct": "off_to", "OR_Pct": "off_orb",
    "FT_Rate": "off_ftr", "DeFG_Pct": "def_efg", "DTO_Pct": "def_to",
    "DOR_Pct": "def_orb", "DFT_Rate": "def_ftr",
}

# Misc stats field map
_MISC_MAP = {
    "FG3Pct": "fg3_pct", "FG2Pct": "fg2_pct", "FTPct": "ft_pct",
    "BlockPct": "block_pct", "StlRate": "stl_rate", "ARate": "ast_rate",
    "F3GRate": "fg3_rate", "OppFG3Pct": "opp_fg3_pct", "OppFG2Pct": "opp_fg2_pct",
}

# Height field map
_HEIGHT_MAP = {
    "AvgHgt": "avg_hgt", "Exp": "experience", "Bench": "bench",
    "Continuity": "continuity", "HgtEff": "hgt_eff",
}

FEATURE_NAMES = [
    "adj_em_diff",
    "adj_o_edge_a",
    "adj_o_edge_b",
    "tempo_diff",
    "seed_diff",
    # Four factors
    "off_efg_diff",
    "def_efg_diff",
    "off_to_diff",
    "off_orb_diff",
    "off_ftr_diff",
    "def_to_diff",
    # Shooting
    "fg3_pct_diff",
    "fg2_pct_diff",
    "ft_pct_diff",
    "fg3_rate_diff",
    "ast_rate_diff",
    # Defense
    "block_pct_diff",
    "stl_rate_diff",
    "opp_fg3_pct_diff",
    "opp_fg2_pct_diff",
    # Roster
    "avg_hgt_diff",
    "experience_diff",
    "bench_diff",
    "continuity_diff",
    # Engineered
    "seed_matchup",         # 1 vs 16, 2 vs 15 interaction
    "tempo_mismatch",       # absolute tempo difference
    "off_def_asymmetry_a",  # team_a offense vs team_b defense gap
    "off_def_asymmetry_b",  # team_b offense vs team_a defense gap
    "efg_margin",           # (off_efg_a - def_efg_b) - (off_efg_b - def_efg_a)
]


def _get_client():
    from kenpom import KenpomClient
    token = os.environ.get("KENPOM_BEARER_TOKEN", "")
    if not token:
        raise ValueError("KENPOM_BEARER_TOKEN not set")
    return KenpomClient(bearer_token=token)


def _build_team_dict(raw: dict, field_map: dict) -> dict:
    out = {}
    for old_key, new_key in field_map.items():
        if old_key in raw:
            out[new_key] = raw[old_key]
    return out


def _get_full_kenpom(season: int, snapshot_date: str) -> dict[str, dict]:
    """Fetch all KenPom data for a season and merge into one dict per team."""
    client = _get_client()

    # Ratings archive (snapshot date)
    ratings = client.get_ratings_archive_by_date(date=snapshot_date)
    teams: dict[str, dict] = {}
    for r in ratings:
        name = r.get("TeamName", "")
        if not name:
            continue
        d = _build_team_dict(r, _RATINGS_MAP)
        d["team"] = name
        norm = normalize(name)
        teams[norm] = d
        teams[name.lower()] = d

    time.sleep(0.5)

    # Four factors (by year — end-of-season data, close enough)
    try:
        ff = client.get_four_factors(year=season)
        for r in ff:
            name = r.get("TeamName", "")
            norm = normalize(name)
            target = teams.get(norm) or teams.get(name.lower())
            if target:
                target.update(_build_team_dict(r, _FF_MAP))
    except Exception as e:
        print(f"    Four factors failed: {e}")

    time.sleep(0.5)

    # Misc stats
    try:
        ms = client.get_misc_stats(year=season)
        for r in ms:
            name = r.get("TeamName", "")
            norm = normalize(name)
            target = teams.get(norm) or teams.get(name.lower())
            if target:
                target.update(_build_team_dict(r, _MISC_MAP))
    except Exception as e:
        print(f"    Misc stats failed: {e}")

    time.sleep(0.5)

    # Height / experience
    try:
        ht = client.get_height(year=season)
        for r in ht:
            name = r.get("TeamName", "")
            norm = normalize(name)
            target = teams.get(norm) or teams.get(name.lower())
            if target:
                target.update(_build_team_dict(r, _HEIGHT_MAP))
    except Exception as e:
        print(f"    Height failed: {e}")

    return teams


def _fetch_espn_games(dates: list[str]) -> list[dict]:
    """Fetch completed NCAA tournament games from ESPN."""
    games = []
    seen_ids = set()
    for date_str in dates:
        try:
            resp = requests.get(_ESPN_URL, params={"dates": date_str, "groups": 50, "limit": 100}, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    ESPN error {date_str}: {e}")
            continue

        for event in data.get("events", []):
            eid = event.get("id", "")
            if eid in seen_ids:
                continue
            seen_ids.add(eid)
            comp = event.get("competitions", [{}])[0]
            if comp.get("status", {}).get("type", {}).get("name") != "STATUS_FINAL":
                continue
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
                games.append({"away": teams["away"], "home": teams["home"]})
        time.sleep(0.3)
    return games


def _g(d: dict, k: str, default: float = 0.0) -> float:
    v = d.get(k, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _build_feature_row(a: dict, b: dict) -> list[float]:
    """Build the full feature vector."""
    adj_o_a = _g(a, "adj_o", 100)
    adj_d_a = _g(a, "adj_d", 100)
    adj_o_b = _g(b, "adj_o", 100)
    adj_d_b = _g(b, "adj_d", 100)
    seed_a = _g(a, "seed", 8)
    seed_b = _g(b, "seed", 8)
    off_efg_a = _g(a, "off_efg")
    off_efg_b = _g(b, "off_efg")
    def_efg_a = _g(a, "def_efg")
    def_efg_b = _g(b, "def_efg")
    tempo_a = _g(a, "tempo", 67.5)
    tempo_b = _g(b, "tempo", 67.5)

    return [
        # Core efficiency
        _g(a, "adj_em") - _g(b, "adj_em"),
        adj_o_a - adj_d_b,
        adj_o_b - adj_d_a,
        tempo_a - tempo_b,
        seed_b - seed_a,
        # Four factors
        off_efg_a - off_efg_b,
        def_efg_a - def_efg_b,
        _g(a, "off_to") - _g(b, "off_to"),
        _g(a, "off_orb") - _g(b, "off_orb"),
        _g(a, "off_ftr") - _g(b, "off_ftr"),
        _g(a, "def_to") - _g(b, "def_to"),
        # Shooting
        _g(a, "fg3_pct") - _g(b, "fg3_pct"),
        _g(a, "fg2_pct") - _g(b, "fg2_pct"),
        _g(a, "ft_pct") - _g(b, "ft_pct"),
        _g(a, "fg3_rate") - _g(b, "fg3_rate"),
        _g(a, "ast_rate") - _g(b, "ast_rate"),
        # Defense
        _g(a, "block_pct") - _g(b, "block_pct"),
        _g(a, "stl_rate") - _g(b, "stl_rate"),
        _g(a, "opp_fg3_pct") - _g(b, "opp_fg3_pct"),
        _g(a, "opp_fg2_pct") - _g(b, "opp_fg2_pct"),
        # Roster
        _g(a, "avg_hgt") - _g(b, "avg_hgt"),
        _g(a, "experience") - _g(b, "experience"),
        _g(a, "bench") - _g(b, "bench"),
        _g(a, "continuity") - _g(b, "continuity"),
        # Engineered: seed matchup interaction
        seed_a * seed_b,
        # Engineered: tempo mismatch
        abs(tempo_a - tempo_b),
        # Engineered: offense vs defense asymmetry
        (adj_o_a - adj_d_b) - (adj_o_b - adj_d_a),
        (adj_o_b - adj_d_a) - (adj_o_a - adj_d_b),
        # Engineered: eFG margin
        (off_efg_a - def_efg_b) - (off_efg_b - def_efg_a),
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

        try:
            kenpom = _get_full_kenpom(season, cfg["snapshot"])
            print(f"  KenPom: {len(kenpom)} entries (ratings + four factors + height + misc)")
        except Exception as e:
            print(f"  KenPom FAILED: {e}")
            continue

        games = _fetch_espn_games(cfg["dates"])
        print(f"  ESPN: {len(games)} completed tournament games")

        if not games:
            continue

        matched = 0
        unmatched = set()

        for game in games:
            away, home = game["away"], game["home"]

            ka = None
            for v in [away["name"], away["short"]]:
                norm = normalize(v)
                ka = kenpom.get(norm) or kenpom.get(v.lower())
                if ka:
                    break

            kb = None
            for v in [home["name"], home["short"]]:
                norm = normalize(v)
                kb = kenpom.get(norm) or kenpom.get(v.lower())
                if kb:
                    break

            if not ka:
                unmatched.add(away["name"])
                continue
            if not kb:
                unmatched.add(home["name"])
                continue

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
        if unmatched:
            print(f"  Unmatched: {', '.join(sorted(unmatched))}")

        time.sleep(2)

    if not rows_out:
        print("\nNo training data generated!")
        sys.exit(1)

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_NAMES + ["team_a_won", "season"])
        writer.writerows(rows_out)

    # Quick check: how many features have actual data
    import numpy as np
    arr = np.array([r[:-2] for r in rows_out])
    nonzero_pct = (arr != 0).mean(axis=0) * 100
    print(f"\n{'='*50}")
    print(f"DONE: {total_games} games across {seasons_ok} seasons")
    print(f"Output: {_OUTPUT_FILE}")
    print(f"Features: {len(FEATURE_NAMES)}")
    print(f"\nFeature coverage (% non-zero):")
    for name, pct in zip(FEATURE_NAMES, nonzero_pct):
        print(f"  {name:30s} {pct:5.1f}%")


if __name__ == "__main__":
    main()
