"""The Odds API client for multi-sportsbook odds (DraftKings, FanDuel, etc.).

Free tier: 500 requests/month. Set THE_ODDS_API_KEY in .env.
https://the-odds-api.com/
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests
import streamlit as st

_BASE = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
_log = logging.getLogger(__name__)

_last_error: str | None = None
_remaining_requests: int | None = None


def is_available() -> bool:
    return bool(os.environ.get("THE_ODDS_API_KEY", ""))


def last_error() -> str | None:
    return _last_error


def remaining_requests() -> int | None:
    return _remaining_requests


@st.cache_data(ttl=120, show_spinner="Fetching sportsbook odds...")
def get_ncaab_odds() -> dict[str, dict[str, Any]]:
    """Fetch NCAAB odds from multiple books, keyed by a normalized matchup key.

    Returns empty dict on any error (rate limit, network, auth, etc.)
    so the app can continue without odds.
    """
    global _last_error, _remaining_requests

    api_key = os.environ.get("THE_ODDS_API_KEY", "")
    if not api_key:
        return {}

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "bookmakers": "draftkings,fanduel",
        "oddsFormat": "american",
    }

    try:
        resp = requests.get(_BASE, params=params, timeout=15)

        # Track quota from response headers
        remaining = resp.headers.get("x-requests-remaining")
        if remaining is not None:
            try:
                _remaining_requests = int(remaining)
            except ValueError:
                pass

        if resp.status_code == 429:
            _last_error = "Rate limited — odds temporarily unavailable"
            _log.warning("Odds API rate limited (429). Using cached data.")
            st.warning("Odds API rate limit reached. Odds may be stale.")
            return {}

        if resp.status_code == 401:
            _last_error = "Invalid API key"
            _log.error("Odds API auth failed (401)")
            return {}

        resp.raise_for_status()
        data = resp.json()
        _last_error = None
    except requests.exceptions.Timeout:
        _last_error = "Request timed out"
        _log.warning("Odds API timeout")
        return {}
    except requests.exceptions.ConnectionError:
        _last_error = "Connection failed"
        _log.warning("Odds API connection error")
        return {}
    except Exception as e:
        _last_error = str(e)[:80]
        _log.warning("Odds API error: %s", e)
        return {}

    result: dict[str, dict[str, Any]] = {}

    for event in data:
        try:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            key = f"{away.lower()} vs {home.lower()}"

            books: dict[str, Any] = {}
            for bookmaker in event.get("bookmakers", []):
                bk_key = bookmaker.get("key", "")
                if bk_key not in ("draftkings", "fanduel"):
                    continue

                book_data: dict[str, Any] = {"name": bookmaker.get("title", bk_key)}

                for market in bookmaker.get("markets", []):
                    mkey = market.get("key", "")
                    outcomes = market.get("outcomes", [])

                    if mkey == "h2h":
                        for o in outcomes:
                            if o.get("name") == home:
                                book_data["ml_home"] = o.get("price")
                            elif o.get("name") == away:
                                book_data["ml_away"] = o.get("price")

                    elif mkey == "spreads":
                        for o in outcomes:
                            if o.get("name") == home:
                                book_data["spread"] = o.get("point")
                                book_data["spread_price_home"] = o.get("price")
                            elif o.get("name") == away:
                                book_data["spread_price_away"] = o.get("price")

                    elif mkey == "totals":
                        for o in outcomes:
                            if o.get("name") == "Over":
                                book_data["total"] = o.get("point")
                                book_data["total_over_price"] = o.get("price")
                            elif o.get("name") == "Under":
                                book_data["total_under_price"] = o.get("price")

                books[bk_key] = book_data

            if books:
                result[key] = books
        except Exception:
            continue

    return result
