"""ESPN Scoreboard API client for NCAA tournament schedule, scores, and odds."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

_ET = timezone(timedelta(hours=-4))  # EDT (March = daylight saving)

import requests
import streamlit as st

from utils.team_names import normalize

_BASE_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/scoreboard"
)

# Full tournament window — First Four through Championship
_TOURNAMENT_START = "20260317"
_TOURNAMENT_END = "20260407"


def _parse_team(competitor: dict) -> dict[str, Any]:
    """Extract a clean team dict from an ESPN competitor object."""
    team_info = competitor.get("team", {})
    records = competitor.get("records", [])
    record_str = records[0].get("summary", "") if records else ""

    seed_raw = competitor.get("curatedRank", {}).get("current", None)
    seed = seed_raw if seed_raw is not None and seed_raw <= 16 else None

    return {
        "espn_id": team_info.get("id", ""),
        "name": normalize(team_info.get("displayName", "")),
        "short_name": normalize(team_info.get("shortDisplayName", "")),
        "abbreviation": team_info.get("abbreviation", ""),
        "logo_url": team_info.get("logo", ""),
        "color": team_info.get("color", "333333"),
        "seed": seed,
        "record": record_str,
        "winner": competitor.get("winner", False),
    }


def _parse_odds(odds_list: list) -> dict[str, Any] | None:
    """Extract DraftKings odds from the ESPN odds array."""
    if not odds_list:
        return None
    odds = odds_list[0]
    result: dict[str, Any] = {
        "provider": odds.get("provider", {}).get("name", ""),
        "spread": odds.get("spread"),
        "spread_detail": odds.get("details", ""),
        "over_under": odds.get("overUnder"),
    }

    ml = odds.get("moneyline", {})
    home_ml = ml.get("home", {}).get("close", {}).get("odds", "")
    away_ml = ml.get("away", {}).get("close", {}).get("odds", "")
    result["ml_home"] = home_ml
    result["ml_away"] = away_ml

    ps = odds.get("pointSpread", {})
    home_spread_line = ps.get("home", {}).get("close", {}).get("line", "")
    away_spread_line = ps.get("away", {}).get("close", {}).get("line", "")
    result["spread_home_line"] = home_spread_line
    result["spread_away_line"] = away_spread_line

    total_data = odds.get("total", {})
    over_line = total_data.get("over", {}).get("close", {}).get("line", "")
    under_line = total_data.get("under", {}).get("close", {}).get("line", "")
    result["total_over_line"] = over_line
    result["total_under_line"] = under_line

    return result


def _parse_round_region(notes: list) -> tuple[str, str]:
    """Extract round name and region from ESPN notes headline."""
    if not notes:
        return ("", "")
    headline = notes[0].get("headline", "")
    # Typical: "NCAA Men's Basketball Championship - East Region - Round of 64"
    parts = [p.strip() for p in headline.split(" - ")]
    round_name = ""
    region = ""
    for part in parts:
        lower = part.lower()
        if "region" in lower:
            region = part.replace(" Region", "")
        elif "championship" not in lower and "ncaa" not in lower:
            round_name = part
    return (round_name, region)


def _parse_game(event: dict) -> dict[str, Any]:
    """Parse a single ESPN event into a clean game dict."""
    competition = event.get("competitions", [{}])[0]
    competitors = competition.get("competitors", [])

    home = None
    away = None
    for c in competitors:
        if c.get("homeAway") == "home":
            home = _parse_team(c)
        else:
            away = _parse_team(c)

    if home is None or away is None:
        home = home or _parse_team(competitors[0]) if competitors else {}
        away = away or _parse_team(competitors[1]) if len(competitors) > 1 else {}

    status_obj = competition.get("status", event.get("status", {}))
    status_type = status_obj.get("type", {})
    state = status_type.get("state", "pre")
    status_detail = status_type.get("shortDetail", status_type.get("detail", ""))

    venue_info = competition.get("venue", {})
    venue_name = venue_info.get("fullName", "")
    venue_city = venue_info.get("address", {}).get("city", "")
    venue_state = venue_info.get("address", {}).get("state", "")

    broadcasts = competition.get("broadcasts", [])
    broadcast = broadcasts[0].get("names", [""])[0] if broadcasts else ""

    notes = competition.get("notes", [])
    round_name, region = _parse_round_region(notes)

    odds = _parse_odds(competition.get("odds", []))

    home_score = None
    away_score = None
    for c in competitors:
        score = c.get("score", "")
        if score and score != "0":
            if c.get("homeAway") == "home":
                home_score = score
            else:
                away_score = score

    return {
        "game_id": event.get("id", ""),
        "date": event.get("date", ""),
        "state": state,
        "status_detail": status_detail,
        "round": round_name,
        "region": region,
        "home_team": home,
        "away_team": away,
        "venue": f"{venue_name}, {venue_city}, {venue_state}".strip(", "),
        "broadcast": broadcast,
        "odds": odds,
        "home_score": home_score,
        "away_score": away_score,
    }


def _is_ncaa_tournament(event: dict) -> bool:
    """Return True only for NCAA Men's Basketball Championship games."""
    notes = event.get("competitions", [{}])[0].get("notes", [])
    if notes:
        headline = notes[0].get("headline", "")
        if "NIT" in headline or "Crown" in headline:
            return False
        if "NCAA" in headline or "Championship" in headline:
            return True
    return True  # default to include if no notes


@st.cache_data(ttl=120, show_spinner="Fetching ESPN tournament schedule...")
def get_tournament_games() -> list[dict[str, Any]]:
    """Fetch NCAA tournament games from ESPN, per-day to include odds."""
    from datetime import datetime as dt, timedelta

    all_games: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    today = dt.now()
    for delta in range(5):
        day = today + timedelta(days=delta)
        day_str = day.strftime("%Y%m%d")
        params = {"seasontype": 3, "dates": day_str, "groups": 50, "limit": 100}
        try:
            resp = requests.get(_BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            for event in data.get("events", []):
                gid = event.get("id", "")
                if gid not in seen_ids and _is_ncaa_tournament(event):
                    seen_ids.add(gid)
                    all_games.append(_parse_game(event))
        except Exception:
            continue

    params = {
        "seasontype": 3,
        "dates": f"{_TOURNAMENT_START}-{_TOURNAMENT_END}",
        "groups": 50,
        "limit": 500,
    }
    try:
        resp = requests.get(_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for event in data.get("events", []):
            gid = event.get("id", "")
            if gid not in seen_ids and _is_ncaa_tournament(event):
                seen_ids.add(gid)
                all_games.append(_parse_game(event))
    except Exception as e:
        if not all_games:
            st.warning(f"ESPN schedule unavailable: {e}")

    return all_games


def get_game_display_label(game: dict[str, Any]) -> str:
    """Build a human-readable label for the matchup selector dropdown."""
    dt_str = game.get("date", "")
    try:
        dt_utc = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        dt_et = dt_utc.astimezone(_ET)
        time_label = dt_et.strftime("%a %-I:%M %p ET")
    except (ValueError, TypeError):
        time_label = dt_str[:16] if dt_str else "TBD"

    home = game.get("home_team", {})
    away = game.get("away_team", {})
    home_name = home.get("short_name", home.get("name", "TBD"))
    away_name = away.get("short_name", away.get("name", "TBD"))

    home_seed = f"({home.get('seed')})" if home.get("seed") else ""
    away_seed = f"({away.get('seed')})" if away.get("seed") else ""

    round_name = game.get("round", "")
    region = game.get("region", "")

    state_map = {"pre": "Upcoming", "in": "LIVE", "post": "Final"}
    status = state_map.get(game.get("state", ""), game.get("state", ""))

    parts = [time_label]
    if round_name:
        parts.append(round_name)
    if region:
        parts.append(region)
    parts.append(f"{away_seed} {away_name} vs {home_seed} {home_name}")
    parts.append(status)

    return " | ".join(parts)


def filter_games(
    games: list[dict[str, Any]],
    *,
    round_filter: str | None = None,
    status_filter: str | None = None,
    region_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Filter games by round, status, and/or region."""
    filtered = games
    if round_filter and round_filter != "All":
        filtered = [g for g in filtered if round_filter.lower() in g.get("round", "").lower()]
    if status_filter and status_filter != "All":
        state_map = {"Upcoming": "pre", "Live": "in", "Final": "post"}
        target = state_map.get(status_filter, status_filter.lower())
        filtered = [g for g in filtered if g.get("state") == target]
    if region_filter and region_filter != "All":
        filtered = [g for g in filtered if region_filter.lower() in g.get("region", "").lower()]
    return filtered


# ---------------------------------------------------------------------------
# NIT schedule (groups=54)
# ---------------------------------------------------------------------------

_NIT_GROUP = 54


@st.cache_data(ttl=120, show_spinner="Fetching NIT schedule...")
def get_nit_games() -> list[dict[str, Any]]:
    """Fetch NIT tournament games from ESPN."""
    from datetime import datetime as dt, timedelta

    all_games: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    today = dt.now()
    for delta in range(5):
        day = today + timedelta(days=delta)
        day_str = day.strftime("%Y%m%d")
        params = {"seasontype": 3, "dates": day_str, "groups": _NIT_GROUP, "limit": 100}
        try:
            resp = requests.get(_BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            for event in resp.json().get("events", []):
                gid = event.get("id", "")
                if gid not in seen_ids:
                    seen_ids.add(gid)
                    all_games.append(_parse_game(event))
        except Exception:
            continue

    params = {
        "seasontype": 3,
        "dates": f"{_TOURNAMENT_START}-{_TOURNAMENT_END}",
        "groups": _NIT_GROUP,
        "limit": 200,
    }
    try:
        resp = requests.get(_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        for event in resp.json().get("events", []):
            gid = event.get("id", "")
            if gid not in seen_ids:
                seen_ids.add(gid)
                all_games.append(_parse_game(event))
    except Exception:
        pass

    return all_games


# ---------------------------------------------------------------------------
# Game summary (news, leaders, predictor)
# ---------------------------------------------------------------------------

_SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/summary"
)


@st.cache_data(ttl=3600, show_spinner=False)
def get_game_odds(game_id: str) -> dict[str, Any] | None:
    """Fetch closing odds from ESPN game summary pickcenter.

    Works for both upcoming and completed games — ESPN keeps pickcenter data
    even after a game ends (unlike the scoreboard odds field which disappears).
    Returns a dict in the same format as the scoreboard `odds` field, or None.
    """
    if not game_id:
        return None
    try:
        resp = requests.get(_SUMMARY_URL, params={"event": game_id}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    pickcenter = data.get("pickcenter", [])
    if not pickcenter:
        return None

    pc = pickcenter[0]  # DraftKings is priority 1

    result: dict[str, Any] = {}

    spread = pc.get("spread")
    if spread is not None:
        result["spread"] = spread                    # home perspective (neg = home favored)
        result["spread_away_line"] = -spread         # away perspective

    over_under = pc.get("overUnder")
    if over_under is not None:
        result["over_under"] = over_under

    home_ml = pc.get("homeTeamOdds", {}).get("moneyLine")
    away_ml = pc.get("awayTeamOdds", {}).get("moneyLine")
    if home_ml is not None:
        result["ml_home"] = home_ml
    if away_ml is not None:
        result["ml_away"] = away_ml

    return result if result else None


@st.cache_data(ttl=300, show_spinner=False)
def get_game_summary(game_id: str) -> dict[str, Any]:
    """Fetch ESPN game summary including news and stat leaders.

    Returns a dict with 'news' (list of articles) and 'leaders' (per-team).
    Returns empty dict on any error.
    """
    if not game_id:
        return {}
    try:
        resp = requests.get(_SUMMARY_URL, params={"event": game_id}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {}

    result: dict[str, Any] = {}

    # News articles
    articles = data.get("news", {}).get("articles", [])
    news = []
    for a in articles[:6]:
        headline = a.get("headline", "")
        link = a.get("links", {}).get("web", {}).get("href", "")
        img = ""
        images = a.get("images", [])
        if images:
            img = images[0].get("url", "")
        if headline:
            news.append({"headline": headline, "link": link, "image": img})
    result["news"] = news

    # Stat leaders
    leaders_raw = data.get("leaders", [])
    leaders = []
    for team_block in leaders_raw:
        team_name = team_block.get("team", {}).get("displayName", "")
        team_logo = team_block.get("team", {}).get("logo", "")
        cats = []
        for cat in team_block.get("leaders", []):
            cat_name = cat.get("displayName", "")
            top_leaders = cat.get("leaders", [])
            if top_leaders:
                top = top_leaders[0]
                athlete = top.get("athlete", {})
                cats.append({
                    "category": cat_name,
                    "player": athlete.get("displayName", ""),
                    "jersey": athlete.get("jersey", ""),
                    "position": athlete.get("position", {}).get("abbreviation", ""),
                    "headshot": athlete.get("headshot", {}).get("href", ""),
                    "value": top.get("displayValue", ""),
                })
        leaders.append({"team": team_name, "logo": team_logo, "stats": cats})
    result["leaders"] = leaders

    # ESPN's own win probability
    predictor = data.get("predictor", {})
    if predictor:
        result["espn_predictor"] = predictor

    return result
