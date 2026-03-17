"""Fallback CSV loader for offline/demo mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from utils.team_names import normalize

_DATA_DIR = Path(__file__).parent.parent / "data"


@st.cache_data(show_spinner="Loading team data from CSV...")
def load_team_stats_csv(path: str | None = None) -> dict[str, dict[str, Any]]:
    """Load team stats from a CSV file into a team-keyed dict.

    Works with both the bundled fallback CSV and user-uploaded files.
    """
    csv_path = Path(path) if path else _DATA_DIR / "teams_2026.csv"
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    if "team" not in df.columns:
        return {}

    df["team"] = df["team"].apply(normalize)
    teams: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        rd = row.to_dict()
        name = rd.get("team", "")
        if name:
            teams[name] = rd
    return teams


@st.cache_data(show_spinner="Loading schedule from CSV...")
def load_schedule_csv(path: str | None = None) -> list[dict[str, Any]]:
    """Load game schedule from a CSV file into a list of game dicts."""
    csv_path = Path(path) if path else _DATA_DIR / "games_2026.csv"
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    games: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rd = row.to_dict()
        team_a = normalize(str(rd.get("team_a", "")))
        team_b = normalize(str(rd.get("team_b", "")))
        games.append({
            "game_id": str(rd.get("game_id", "")),
            "date": str(rd.get("game_time_et", "")),
            "state": str(rd.get("status", "pre")),
            "status_detail": str(rd.get("game_time_et", "")),
            "round": str(rd.get("round", "")),
            "region": str(rd.get("region", "")),
            "home_team": {
                "name": team_a,
                "short_name": team_a,
                "abbreviation": "",
                "logo_url": "",
                "color": "333333",
                "seed": rd.get("seed_a"),
                "record": "",
            },
            "away_team": {
                "name": team_b,
                "short_name": team_b,
                "abbreviation": "",
                "logo_url": "",
                "color": "333333",
                "seed": rd.get("seed_b"),
                "record": "",
            },
            "venue": str(rd.get("venue", "")),
            "broadcast": "",
            "odds": None,
            "home_score": rd.get("team_a_score"),
            "away_score": rd.get("team_b_score"),
        })
    return games


def load_uploaded_csv(uploaded_file) -> dict[str, dict[str, Any]]:
    """Parse a user-uploaded CSV (from Streamlit file_uploader) into team stats."""
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()
    if "team" not in df.columns:
        st.error("Uploaded CSV must have a 'team' column.")
        return {}
    df["team"] = df["team"].apply(normalize)
    teams: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        rd = row.to_dict()
        name = rd.get("team", "")
        if name:
            teams[name] = rd
    return teams
