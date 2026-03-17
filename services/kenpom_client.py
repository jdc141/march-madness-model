"""KenPom API client with Streamlit caching and rate-limit handling."""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd
import streamlit as st

from utils.team_names import normalize

_YEAR = 2026
_log = logging.getLogger(__name__)

# Track degraded state so the sidebar can display it
_last_error: str | None = None


def _get_client():
    from kenpom import KenpomClient

    token = os.environ.get("KENPOM_BEARER_TOKEN", "")
    if not token:
        return None
    return KenpomClient(bearer_token=token)


def is_available() -> bool:
    return bool(os.environ.get("KENPOM_BEARER_TOKEN", ""))


def last_error() -> str | None:
    return _last_error


def _handle_error(source: str, exc: Exception) -> None:
    """Log the error and surface a one-time sidebar warning."""
    global _last_error
    msg = str(exc)
    is_rate_limit = "429" in msg or "rate" in msg.lower() or "too many" in msg.lower()

    if is_rate_limit:
        _last_error = "Rate limited — using cached data"
        _log.warning("KenPom %s: rate limited (%s)", source, msg)
        st.warning(f"KenPom rate limit hit for {source}. Using cached data.")
    else:
        _last_error = f"{source}: {msg[:80]}"
        _log.warning("KenPom %s failed: %s", source, msg)
        st.warning(f"KenPom {source} unavailable: {msg[:120]}")


@st.cache_data(ttl=900, show_spinner="Fetching KenPom ratings...")
def get_ratings() -> list[dict[str, Any]] | None:
    client = _get_client()
    if client is None:
        return None
    try:
        return client.get_ratings(year=_YEAR)
    except Exception as e:
        _handle_error("ratings", e)
        return None


@st.cache_data(ttl=900, show_spinner="Fetching four factors...")
def get_four_factors() -> list[dict[str, Any]] | None:
    client = _get_client()
    if client is None:
        return None
    try:
        return client.get_four_factors(year=_YEAR)
    except Exception as e:
        _handle_error("four factors", e)
        return None


@st.cache_data(ttl=900, show_spinner="Fetching misc stats...")
def get_misc_stats() -> list[dict[str, Any]] | None:
    client = _get_client()
    if client is None:
        return None
    try:
        return client.get_misc_stats(year=_YEAR)
    except Exception as e:
        _handle_error("misc stats", e)
        return None


@st.cache_data(ttl=900, show_spinner="Fetching height/experience data...")
def get_height() -> list[dict[str, Any]] | None:
    client = _get_client()
    if client is None:
        return None
    try:
        return client.get_height(year=_YEAR)
    except Exception as e:
        _handle_error("height", e)
        return None


@st.cache_data(ttl=300, show_spinner="Fetching FanMatch predictions...")
def get_fanmatch(date_str: str) -> list[dict[str, Any]] | None:
    client = _get_client()
    if client is None:
        return None
    try:
        return client.get_fanmatch(date=date_str)
    except Exception as e:
        _handle_error("FanMatch", e)
        return None


_FIELD_MAP = {
    # --- Ratings ---
    "TeamName": "team",
    "Seed": "seed",
    "ConfShort": "conference",
    "AdjEM": "adj_em",
    "RankAdjEM": "rank_adj_em",
    "AdjOE": "adj_o",
    "RankAdjOE": "rank_adj_o",
    "AdjDE": "adj_d",
    "RankAdjDE": "rank_adj_d",
    # Use adjusted tempo as the canonical pace input so live data matches
    # historical training snapshots.
    "Tempo": "raw_tempo",
    "AdjTempo": "tempo",
    "RankAdjTempo": "rank_adj_tempo",
    "Luck": "luck",
    "RankLuck": "rank_luck",
    "SOS": "sos",
    "RankSOS": "rank_sos",
    "NCSOS": "ncsos",
    "RankNCSOS": "rank_ncsos",
    "Pythag": "pythag",
    "Wins": "wins",
    "Losses": "losses",
    "Coach": "coach",
    "Event": "event",
    "SOSO": "sos_o",
    "SOSD": "sos_d",
    "OE": "raw_oe",
    "DE": "raw_de",
    "APL_Off": "apl_off",
    "APL_Def": "apl_def",
    # --- Four Factors ---
    "eFG_Pct": "off_efg",
    "TO_Pct": "off_to",
    "OR_Pct": "off_orb",
    "FT_Rate": "off_ftr",
    "DeFG_Pct": "def_efg",
    "DTO_Pct": "def_to",
    "DOR_Pct": "def_orb",
    "DFT_Rate": "def_ftr",
    "RankeFG_Pct": "rank_off_efg",
    "RankTO_Pct": "rank_off_to",
    "RankOR_Pct": "rank_off_orb",
    "RankFT_Rate": "rank_off_ftr",
    "RankDeFG_Pct": "rank_def_efg",
    "RankDTO_Pct": "rank_def_to",
    "RankDOR_Pct": "rank_def_orb",
    "RankDFT_Rate": "rank_def_ftr",
    # --- Misc Stats ---
    "FG3Pct": "fg3_pct",
    "FG2Pct": "fg2_pct",
    "FTPct": "ft_pct",
    "BlockPct": "block_pct",
    "StlRate": "stl_rate",
    "NSTRate": "nst_rate",
    "ARate": "ast_rate",
    "F3GRate": "fg3_rate",
    "Avg2PADist": "avg_2pa_dist",
    "OppFG3Pct": "opp_fg3_pct",
    "OppFG2Pct": "opp_fg2_pct",
    "OppFTPct": "opp_ft_pct",
    "OppBlockPct": "opp_block_pct",
    "OppStlRate": "opp_stl_rate",
    "OppNSTRate": "opp_nst_rate",
    "OppARate": "opp_ast_rate",
    "OppF3GRate": "opp_fg3_rate",
    "OppAvg2PADist": "opp_avg_2pa_dist",
    "RankFG3Pct": "rank_fg3_pct",
    "RankFG2Pct": "rank_fg2_pct",
    "RankFTPct": "rank_ft_pct",
    "RankBlockPct": "rank_block_pct",
    "RankStlRate": "rank_stl_rate",
    "RankARate": "rank_ast_rate",
    "RankF3GRate": "rank_fg3_rate",
    # --- Height / Experience ---
    "AvgHgt": "avg_hgt",
    "AvgHgtRank": "rank_avg_hgt",
    "HgtEff": "hgt_eff",
    "HgtEffRank": "rank_hgt_eff",
    "Hgt5": "hgt_c",
    "Hgt4": "hgt_pf",
    "Hgt3": "hgt_sf",
    "Hgt2": "hgt_sg",
    "Hgt1": "hgt_pg",
    "Hgt5Rank": "rank_hgt_c",
    "Hgt4Rank": "rank_hgt_pf",
    "Hgt3Rank": "rank_hgt_sf",
    "Hgt2Rank": "rank_hgt_sg",
    "Hgt1Rank": "rank_hgt_pg",
    "Exp": "experience",
    "ExpRank": "rank_experience",
    "Bench": "bench",
    "BenchRank": "rank_bench",
    "Continuity": "continuity",
    "RankContinuity": "rank_continuity",
    # --- Point Distribution ---
    "OffFt": "pts_pct_ft",
    "OffFg2": "pts_pct_2pt",
    "OffFg3": "pts_pct_3pt",
    "DefFt": "opp_pts_pct_ft",
    "DefFg2": "opp_pts_pct_2pt",
    "DefFg3": "opp_pts_pct_3pt",
    "RankOffFt": "rank_pts_pct_ft",
    "RankOffFg2": "rank_pts_pct_2pt",
    "RankOffFg3": "rank_pts_pct_3pt",
    "RankDefFt": "rank_opp_pts_pct_ft",
    "RankDefFg2": "rank_opp_pts_pct_2pt",
    "RankDefFg3": "rank_opp_pts_pct_3pt",
}


def _normalize_rating(r: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, val in r.items():
        mapped = _FIELD_MAP.get(key, key)
        out[mapped] = val

    raw_name = out.get("team", "")
    if raw_name:
        out["team"] = normalize(str(raw_name))

    return out


@st.cache_data(ttl=900, show_spinner="Fetching point distribution...")
def get_point_distribution() -> list[dict[str, Any]] | None:
    client = _get_client()
    if client is None:
        return None
    try:
        return client.get_point_distribution(year=_YEAR)
    except Exception as e:
        _handle_error("point distribution", e)
        return None


def get_all_team_stats() -> dict[str, dict[str, Any]]:
    """Build a unified team profile dict keyed by normalized team name.

    Merges ratings, four factors, misc stats, and height data.
    Returns empty dict if API is unavailable — caller should fall back to CSV.
    """
    try:
        ratings = get_ratings()
    except Exception as e:
        _handle_error("ratings (outer)", e)
        return {}

    if ratings is None:
        return {}

    teams: dict[str, dict[str, Any]] = {}
    for r in ratings:
        try:
            r = _normalize_rating(r)
            name = r.get("team", "")
            if name:
                teams[name] = r
        except Exception:
            continue

    for source_fn in [get_four_factors, get_misc_stats, get_height, get_point_distribution]:
        try:
            data = source_fn()
        except Exception:
            continue
        if data is None:
            continue
        for row in data:
            try:
                row = _normalize_rating(row)
                name = row.get("team", "")
                if name in teams:
                    teams[name].update(row)
            except Exception:
                continue

    return teams


def get_team_names() -> list[str]:
    stats = get_all_team_stats()
    return sorted(stats.keys())
