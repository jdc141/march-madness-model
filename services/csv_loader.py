"""CSV loader for user-provided team data."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from utils.team_names import normalize


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
