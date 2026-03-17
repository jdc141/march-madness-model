"""March Madness Prediction Engine — Streamlit App."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from predictor import predict_matchup, MatchupPrediction
from services import espn_client, kenpom_client, csv_loader
from services import odds_client
from utils.features import compute_matchup_stats, MatchupStats
from utils.formatting import (
    fmt_moneyline,
    fmt_spread,
    fmt_probability,
    confidence_color,
)
from utils.team_names import normalize, fuzzy_match
from utils.bracket import render_full_bracket

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="March Madness Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 1rem; }

    [data-testid="stMetric"] {
        background: rgba(30, 30, 46, 0.6);
        border: 1px solid rgba(100, 100, 140, 0.25);
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; }

    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a5b4fc;
        border-bottom: 1px solid rgba(100, 100, 140, 0.3);
        padding-bottom: 6px;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }

    .agree-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .agree { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .disagree { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }

    .conf-label {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    .prob-bar-container {
        background: rgba(50, 50, 70, 0.5);
        border-radius: 8px;
        overflow: hidden;
        height: 28px;
        display: flex;
        margin: 4px 0 8px 0;
    }
    .prob-bar-a, .prob-bar-b {
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
    }
    .prob-bar-b { flex: 1; }

    .good-bet {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 8px;
        padding: 8px 12px;
    }
    .bad-bet {
        background: rgba(50, 50, 70, 0.3);
        border: 1px solid rgba(100, 100, 140, 0.2);
        border-radius: 8px;
        padding: 8px 12px;
    }

    .news-card {
        background: rgba(30, 30, 46, 0.5);
        border: 1px solid rgba(100, 100, 140, 0.2);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .news-card a { color: #93c5fd; text-decoration: none; }
    .news-card a:hover { text-decoration: underline; }

    .leader-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 6px 0;
        border-bottom: 1px solid rgba(100, 100, 140, 0.15);
    }
    .leader-row img { width: 32px; height: 32px; border-radius: 50%; object-fit: cover; }

    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.75rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(100, 100, 140, 0.2);
    }

    /* ---- Comparison row (pure HTML, responsive) ---- */
    .cmp-header {
        display: flex; justify-content: space-between; align-items: center;
        padding: 4px 0; margin-bottom: 4px;
    }
    .cmp-header-team { font-weight: 600; font-size: 0.95rem; color: #e5e7eb; flex: 1; }
    .cmp-header-team:last-child { text-align: right; }
    .cmp-header-mid { text-align: center; color: #6b7280; font-size: 0.8rem; flex: 0 0 80px; }

    .cmp-row {
        display: flex; align-items: center; gap: 8px;
        margin-bottom: 6px;
    }
    .cmp-val {
        flex: 1;
        background: rgba(30, 30, 46, 0.6);
        border: 1px solid rgba(100, 100, 140, 0.25);
        border-radius: 8px;
        padding: 10px 12px;
        font-size: 1.15rem;
        font-weight: 600;
        text-align: center;
    }
    .cmp-label {
        flex: 0 0 90px;
        text-align: center;
        color: #a5b4fc;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* ---- Tab bar: scrollable on mobile ---- */
    [data-testid="stTabs"] > div[role="tablist"] {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: none;
        flex-wrap: nowrap !important;
    }
    [data-testid="stTabs"] > div[role="tablist"]::-webkit-scrollbar {
        display: none;
    }
    [data-testid="stTabs"] > div[role="tablist"] button {
        white-space: nowrap;
        flex-shrink: 0;
    }

    /* ---- Mobile responsiveness ---- */
    @media (max-width: 768px) {
        .block-container { padding-top: 1rem; padding-left: 0.5rem; padding-right: 0.5rem; }

        /* Tab bar: smaller text, tighter padding */
        [data-testid="stTabs"] > div[role="tablist"] button {
            font-size: 0.78rem !important;
            padding: 8px 10px !important;
        }

        [data-testid="stMetric"] { padding: 8px 10px; }
        [data-testid="stMetricLabel"] { font-size: 0.7rem; }
        [data-testid="stMetricValue"] { font-size: 1rem; }

        .section-header { font-size: 0.95rem; margin-top: 1rem; }

        h1 { font-size: 1.4rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1rem !important; }

        .prob-bar-a, .prob-bar-b { font-size: 0.65rem; }
        .prob-bar-container { height: 24px; }

        .good-bet, .bad-bet { padding: 6px 8px; }
        .good-bet div, .bad-bet div { font-size: 0.85rem; }

        [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 0.5rem; }

        [data-testid="stSidebar"] { min-width: 200px !important; }

        .news-card { padding: 8px 10px; }

        [data-testid="stSelectbox"] { min-height: 44px; }
        button { min-height: 44px !important; }

        .cmp-row { gap: 4px; margin-bottom: 4px; }
        .cmp-val { padding: 8px 6px; font-size: 0.9rem; }
        .cmp-label { flex: 0 0 60px; font-size: 0.7rem; }
        .cmp-header-team { font-size: 0.85rem; }
        .cmp-header-mid { flex: 0 0 60px; font-size: 0.7rem; }
    }

    @media (max-width: 480px) {
        .block-container { padding-left: 0.25rem; padding-right: 0.25rem; }
        h1 { font-size: 1.2rem !important; }
        [data-testid="stMetric"] { padding: 6px 8px; }
        [data-testid="stMetricValue"] { font-size: 0.9rem; }

        .cmp-val { padding: 6px 4px; font-size: 0.82rem; border-radius: 6px; }
        .cmp-label { flex: 0 0 50px; font-size: 0.65rem; }
        .cmp-header-mid { flex: 0 0 50px; }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Data Sources")

    kenpom_ok = kenpom_client.is_available()
    if kenpom_ok:
        st.success("KenPom API: Connected", icon="✅")
    else:
        st.warning("KenPom API: No token", icon="⚠️")
        st.caption("Set `KENPOM_BEARER_TOKEN` in `.env`")

    st.info("ESPN API: Connected", icon="📡")

    odds_ok = odds_client.is_available()
    if odds_ok:
        odds_err = odds_client.last_error()
        if odds_err:
            st.warning(f"Odds API: {odds_err}", icon="⚠️")
        else:
            remaining = odds_client.remaining_requests()
            extra = f" ({remaining} req left)" if remaining is not None else ""
            st.success(f"Odds API: DK + FanDuel{extra}", icon="✅")
    else:
        st.caption("Set `THE_ODDS_API_KEY` in `.env` for FanDuel odds")

    st.divider()
    st.subheader("Manual Data Override")
    uploaded = st.file_uploader(
        "Upload KenPom CSV",
        type=["csv"],
        help="CSV with columns: team, seed, adj_em, adj_o, adj_d, tempo, sos, luck",
    )

    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Last refreshed: {datetime.now().strftime('%I:%M %p ET')}")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

team_stats: dict = {}
stats_source = "None"

try:
    if uploaded:
        team_stats = csv_loader.load_uploaded_csv(uploaded)
        stats_source = "Uploaded CSV"
    elif kenpom_ok:
        team_stats = kenpom_client.get_all_team_stats()
        if team_stats:
            stats_source = "KenPom API"
except Exception as _e:
    st.sidebar.error(f"Team stats error: {_e}")

if not team_stats:
    try:
        team_stats = csv_loader.load_team_stats_csv()
        if team_stats:
            stats_source = "Fallback CSV"
    except Exception:
        team_stats = {}
        stats_source = "None (all sources failed)"

all_games: list = []
sched_source = "None"
try:
    all_games = espn_client.get_tournament_games()
    if all_games:
        sched_source = "ESPN API"
except Exception as _e:
    st.sidebar.error(f"ESPN error: {_e}")

if not all_games:
    try:
        all_games = csv_loader.load_schedule_csv()
        if all_games:
            sched_source = "Fallback CSV"
    except Exception:
        all_games = []

multi_odds: dict = {}
try:
    if odds_ok:
        multi_odds = odds_client.get_ncaab_odds()
except Exception:
    multi_odds = {}

nit_games: list = []
try:
    nit_games = espn_client.get_nit_games()
except Exception:
    nit_games = []

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("# 🏀 March Madness Prediction Engine")
st.caption("Live NCAA tournament matchup explorer · KenPom ratings · Dual-model predictions · Sportsbook odds")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_live, tab_bracket, tab_nit, tab_stats, tab_deep_dive, tab_lab = st.tabs([
    "🏟️ March Madness",
    "📊 Live Bracket",
    "🏆 NIT",
    "📈 Stats Deep Dive",
    "🔍 Team Profile",
    "🧪 Future Matchup Lab",
])


# ===========================================================================
# Helpers
# ===========================================================================

_ROUND_ORDER = [
    "First Four",
    "1st Round",
    "2nd Round",
    "Sweet 16",
    "Elite 8",
    "Final Four",
    "National Championship",
]


def _sort_rounds(rounds: list[str]) -> list[str]:
    """Sort round names in tournament progression order."""
    order = {name: i for i, name in enumerate(_ROUND_ORDER)}
    return sorted(rounds, key=lambda r: order.get(r, 999))


def _lookup_team(name: str) -> dict | None:
    if not team_stats:
        return None
    norm = normalize(name)
    if norm in team_stats:
        return team_stats[norm]
    match = fuzzy_match(norm, list(team_stats.keys()))
    return team_stats[match] if match else None


def _render_probability_bar(
    name_a: str, name_b: str, prob_a: float, color_a: str, color_b: str, label: str = "",
):
    pct_a = max(2, min(98, int(prob_a * 100)))
    pct_b = 100 - pct_a
    st.markdown(f"""
    <div style="font-size:0.8rem; color:#9ca3af; margin-bottom:2px;">{label}</div>
    <div class="prob-bar-container">
        <div class="prob-bar-a" style="width:{pct_a}%; background:#{color_a};">{name_a} {pct_a}%</div>
        <div class="prob-bar-b" style="background:#{color_b};">{name_b} {pct_b}%</div>
    </div>
    """, unsafe_allow_html=True)


def _fmt_height(inches: float | None) -> str:
    """Convert total inches (e.g. 75.3) to feet-inches string (e.g. 6'3\")."""
    if inches is None or not isinstance(inches, (int, float)):
        return "—"
    total = round(inches)
    ft = total // 12
    in_ = total % 12
    return f"{ft}'{in_}\""


def _fmt_deviation(val: float | None) -> str:
    """Format a height deviation value (inches above/below D-I average)."""
    if val is None or not isinstance(val, (int, float)):
        return "—"
    return f"{val:+.1f} in"


def _colored_val(val_a, val_b, key: str, fmt: str = ".1f") -> tuple[str, str]:
    """Return (html_a, html_b) with green/red coloring based on which value is better.

    For most stats higher=better. For adj_d, seed, losses, and def/opp stats lower=better.
    Returns neutral gray for ties or non-numeric values.
    """
    lower_is_better = key in (
        "adj_d", "seed", "losses", "def_efg", "def_ftr", "def_orb",
        "opp_fg3_pct", "opp_fg2_pct", "opp_ft_pct", "opp_block_pct",
        "opp_stl_rate", "opp_fg3_rate", "off_to", "nst_rate",
        "opp_pts_pct_ft", "opp_pts_pct_2pt", "opp_pts_pct_3pt",
    )

    green = "#34d399"
    red = "#f87171"
    neutral = "#e5e7eb"

    if not isinstance(val_a, (int, float)) or not isinstance(val_b, (int, float)):
        s_a = str(val_a) if val_a not in (None, "", "—") else "—"
        s_b = str(val_b) if val_b not in (None, "", "—") else "—"
        return (f'<span style="color:{neutral}">{s_a}</span>',
                f'<span style="color:{neutral}">{s_b}</span>')

    if key == "seed":
        s_a = str(int(val_a)) if val_a else "—"
        s_b = str(int(val_b)) if val_b else "—"
    else:
        s_a = f"{val_a:{fmt}}"
        s_b = f"{val_b:{fmt}}"

    if val_a == val_b:
        return (f'<span style="color:{neutral}">{s_a}</span>',
                f'<span style="color:{neutral}">{s_b}</span>')

    if lower_is_better:
        a_better = val_a < val_b
    else:
        a_better = val_a > val_b

    c_a = green if a_better else red
    c_b = red if a_better else green
    return (f'<span style="color:{c_a}">{s_a}</span>',
            f'<span style="color:{c_b}">{s_b}</span>')


def _cmp_row_html(label: str, val_a, val_b, key: str, fmt: str = ".1f") -> str:
    """Return HTML for a single comparison row."""
    html_a, html_b = _colored_val(val_a, val_b, key, fmt)
    return (
        f'<div class="cmp-row">'
        f'<div class="cmp-val">{html_a}</div>'
        f'<div class="cmp-label">{label}</div>'
        f'<div class="cmp-val">{html_b}</div>'
        f'</div>'
    )


def _cmp_header_html(name_a: str, name_b: str, mid: str = "Stat") -> str:
    return (
        f'<div class="cmp-header">'
        f'<div class="cmp-header-team">{name_a}</div>'
        f'<div class="cmp-header-mid">{mid}</div>'
        f'<div class="cmp-header-team" style="text-align:right;">{name_b}</div>'
        f'</div>'
    )


def _hgt_cmp_row_html(label: str, a_val, b_val, formatter) -> str:
    """Return HTML for a height comparison row with a custom formatter."""
    if a_val is None and b_val is None:
        return ""
    a_disp = formatter(a_val) if isinstance(a_val, (int, float)) else "—"
    b_disp = formatter(b_val) if isinstance(b_val, (int, float)) else "—"
    both_num = isinstance(a_val, (int, float)) and isinstance(b_val, (int, float))
    green, red, neutral = "#34d399", "#f87171", "#e5e7eb"
    c_a = green if both_num and a_val > b_val else (red if both_num and a_val < b_val else neutral)
    c_b = green if both_num and b_val > a_val else (red if both_num and b_val < a_val else neutral)
    return (
        f'<div class="cmp-row">'
        f'<div class="cmp-val" style="color:{c_a};">{a_disp}</div>'
        f'<div class="cmp-label">{label}</div>'
        f'<div class="cmp-val" style="color:{c_b};">{b_disp}</div>'
        f'</div>'
    )


_STAT_GLOSSARY = {
    "Seed": "Tournament seed (1 = best, 16 = worst in each region)",
    "AdjEM": "Adjusted Efficiency Margin — point differential per 100 possessions, adjusted for opponent strength. The single best measure of team quality.",
    "AdjO": "Adjusted Offensive Efficiency — points scored per 100 possessions, adjusted for opponent defense. Higher = better offense.",
    "AdjD": "Adjusted Defensive Efficiency — points allowed per 100 possessions, adjusted for opponent offense. Lower = better defense.",
    "Tempo": "Possessions per 40 minutes — how fast or slow a team plays.",
    "SOS": "Strength of Schedule — overall difficulty of opponents faced, adjusted for results.",
    "Luck": "Luck rating — measures how much a team's record deviates from what their efficiency stats predict. Near 0 = team is who their stats say they are.",
    "eFG%": "Effective Field Goal % — adjusts FG% to account for 3-pointers being worth more. A 3PT make counts as 1.5 FG makes.",
    "3PT%": "Three-point field goal percentage.",
    "2PT%": "Two-point field goal percentage.",
    "FT%": "Free throw percentage.",
    "3PT Rate": "Percentage of field goal attempts that are 3-pointers. High = perimeter-oriented offense.",
    "Ast Rate": "Assist rate — percentage of made field goals that were assisted. High = good ball movement.",
    "TO%": "Turnover percentage — turnovers per 100 possessions. Lower = takes better care of the ball.",
    "Off Reb%": "Offensive rebound percentage — % of available offensive rebounds grabbed.",
    "FT Rate": "Free throw rate — free throw attempts per field goal attempt. High = gets to the line a lot.",
    "Blk%": "Block percentage — % of opponent 2PT attempts blocked.",
    "Stl Rate": "Steal rate — steals per 100 possessions.",
    "Opp eFG%": "Opponent effective FG% allowed. Lower = better perimeter and interior defense.",
    "Opp 3PT%": "Opponent 3-point % allowed. Lower = better 3PT defense.",
    "Opp 2PT%": "Opponent 2-point % allowed. Lower = better interior defense.",
    "Opp FT Rate": "Opponent free throw rate allowed. Lower = avoids fouling.",
    "Opp TO%": "Opponent turnovers forced per 100 possessions. Higher = more disruptive defense.",
    "Avg Height": "Average team height in feet and inches, weighted by minutes played.",
    "Effective Hgt": "Height advantage vs the D-I average, weighted by minutes. Positive = taller than average, negative = shorter. Shown as +/- inches.",
    "PG Height": "Point guard height vs D-I average (in inches). Based on the shortest 20% of a team's minutes.",
    "SG Height": "Shooting guard height vs D-I average (in inches).",
    "SF Height": "Small forward height vs D-I average (in inches).",
    "PF Height": "Power forward height vs D-I average (in inches).",
    "C Height": "Center height vs D-I average (in inches). Based on the tallest 20% of a team's minutes.",
    "Experience": "Average years of college experience. Higher = more veteran roster.",
    "Bench": "Bench strength — minutes-weighted contribution from non-starters.",
    "Continuity": "Roster continuity — % of minutes from returning players. Higher = more chemistry.",
}


def _render_team_comparison(team_a: dict, team_b: dict):
    st.markdown('<div class="section-header">Team Comparison</div>', unsafe_allow_html=True)

    name_a = team_a.get("team", "Team A")
    name_b = team_b.get("team", "Team B")

    core_metrics = [
        ("Seed", "seed"),
        ("AdjEM", "adj_em"),
        ("AdjO", "adj_o"),
        ("AdjD", "adj_d"),
        ("Tempo", "tempo"),
        ("SOS", "sos"),
        ("Luck", "luck"),
    ]

    html = _cmp_header_html(name_a, name_b)
    for label, key in core_metrics:
        html += _cmp_row_html(label, team_a.get(key, "—"), team_b.get(key, "—"), key)

    st.markdown(html, unsafe_allow_html=True)

    with st.expander("📖 What do these stats mean?"):
        for label, _ in core_metrics:
            desc = _STAT_GLOSSARY.get(label, "")
            if desc:
                st.markdown(f"**{label}** — {desc}")


def _render_shooting_stats(team_a: dict, team_b: dict):
    """Render shooting, scoring, and four-factors comparison with color coding."""
    has_data = any(team_a.get(k) is not None for k in ["off_efg", "fg3_pct", "fg2_pct", "ft_pct"])
    if not has_data:
        return

    name_a = team_a.get("team", "Team A")
    name_b = team_b.get("team", "Team B")

    st.markdown('<div class="section-header">Shooting & Scoring</div>', unsafe_allow_html=True)

    shooting_metrics = [
        ("eFG%", "off_efg", ".1f"),
        ("3PT%", "fg3_pct", ".1f"),
        ("2PT%", "fg2_pct", ".1f"),
        ("FT%", "ft_pct", ".1f"),
        ("3PT Rate", "fg3_rate", ".1f"),
        ("Ast Rate", "ast_rate", ".1f"),
        ("TO%", "off_to", ".1f"),
        ("Off Reb%", "off_orb", ".1f"),
        ("FT Rate", "off_ftr", ".1f"),
        ("Blk%", "block_pct", ".1f"),
        ("Stl Rate", "stl_rate", ".1f"),
    ]

    html = _cmp_header_html(name_a, name_b)
    for label, key, fmt in shooting_metrics:
        a_val = team_a.get(key)
        b_val = team_b.get(key)
        if a_val is not None or b_val is not None:
            html += _cmp_row_html(label, a_val if a_val is not None else "—", b_val if b_val is not None else "—", key, fmt)
    st.markdown(html, unsafe_allow_html=True)

    # Defensive shooting
    def_metrics = [
        ("Opp eFG%", "def_efg", ".1f"),
        ("Opp 3PT%", "opp_fg3_pct", ".1f"),
        ("Opp 2PT%", "opp_fg2_pct", ".1f"),
        ("Opp FT Rate", "def_ftr", ".1f"),
        ("Opp TO%", "def_to", ".1f"),
    ]

    has_def = any(team_a.get(k) is not None for _, k, _ in def_metrics)
    if has_def:
        st.markdown('<div class="section-header">Defensive Shooting</div>', unsafe_allow_html=True)
        html_def = _cmp_header_html(name_a, name_b)
        for label, key, fmt in def_metrics:
            a_val = team_a.get(key)
            b_val = team_b.get(key)
            if a_val is not None or b_val is not None:
                html_def += _cmp_row_html(label, a_val if a_val is not None else "—", b_val if b_val is not None else "—", key, fmt)
        st.markdown(html_def, unsafe_allow_html=True)

    # Point distribution
    has_pts = team_a.get("pts_pct_3pt") is not None
    if has_pts:
        st.markdown('<div class="section-header">Scoring Breakdown (% of points)</div>', unsafe_allow_html=True)
        html_pts = _cmp_header_html(name_a, name_b, "Source")
        for label, key in [("From 3PT", "pts_pct_3pt"), ("From 2PT", "pts_pct_2pt"), ("From FT", "pts_pct_ft")]:
            a_val = team_a.get(key)
            b_val = team_b.get(key)
            if a_val is not None or b_val is not None:
                html_pts += _cmp_row_html(label, a_val if a_val is not None else "—", b_val if b_val is not None else "—", key, ".1f")
        st.markdown(html_pts, unsafe_allow_html=True)

    all_shooting_labels = [l for l, _, _ in shooting_metrics] + [l for l, _, _ in def_metrics]
    with st.expander("📖 What do these stats mean?"):
        for label in all_shooting_labels:
            desc = _STAT_GLOSSARY.get(label, "")
            if desc:
                st.markdown(f"**{label}** — {desc}")


def _render_height_comparison(team_a: dict, team_b: dict):
    """Render height, experience, and roster build comparison."""
    has_data = any(team_a.get(k) is not None for k in ["avg_hgt", "hgt_c", "hgt_pg", "experience"])
    if not has_data:
        return

    name_a = team_a.get("team", "Team A")
    name_b = team_b.get("team", "Team B")

    st.markdown('<div class="section-header">Height & Roster Build</div>', unsafe_allow_html=True)

    html = _cmp_header_html(name_a, name_b, "Metric")

    html += _hgt_cmp_row_html("Avg Height", team_a.get("avg_hgt"), team_b.get("avg_hgt"), _fmt_height)
    html += _hgt_cmp_row_html("Effective Hgt", team_a.get("hgt_eff"), team_b.get("hgt_eff"), _fmt_deviation)

    st.markdown(html, unsafe_allow_html=True)

    pos_heights = [
        ("PG Height", "hgt_pg"),
        ("SG Height", "hgt_sg"),
        ("SF Height", "hgt_sf"),
        ("PF Height", "hgt_pf"),
        ("C Height", "hgt_c"),
    ]

    has_pos = any(team_a.get(k) is not None for _, k in pos_heights)
    if has_pos:
        st.markdown("**By Position** *(vs D-I average)*")
        html_pos = ""
        for label, key in pos_heights:
            html_pos += _hgt_cmp_row_html(label, team_a.get(key), team_b.get(key), _fmt_deviation)
        st.markdown(html_pos, unsafe_allow_html=True)

    # Experience, bench, continuity
    roster_metrics = [
        ("Experience", "experience"),
        ("Bench", "bench"),
        ("Continuity", "continuity"),
    ]

    has_roster = any(team_a.get(k) is not None for _, k in roster_metrics)
    if has_roster:
        st.markdown("**Roster Composition**")
        html_roster = ""
        for label, key in roster_metrics:
            html_roster += _cmp_row_html(label, team_a.get(key, "—"), team_b.get(key, "—"), key)
        st.markdown(html_roster, unsafe_allow_html=True)

    height_labels = ["Avg Height", "Effective Hgt", "PG Height", "SG Height", "SF Height", "PF Height", "C Height", "Experience", "Bench", "Continuity"]
    with st.expander("📖 What do these stats mean?"):
        for label in height_labels:
            desc = _STAT_GLOSSARY.get(label, "")
            if desc:
                st.markdown(f"**{label}** — {desc}")


def _render_derived_stats(stats: MatchupStats, name_a: str, name_b: str):
    st.markdown('<div class="section-header">Matchup Analytics</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        leader = name_a if stats.adj_em_diff > 0 else name_b
        st.metric("AdjEM Edge", f"{leader} +{abs(stats.adj_em_diff):.1f}")
    with c2:
        st.metric("Tempo Projection", f"{stats.tempo_projection:.1f} poss")
    with c3:
        st.metric("Projected Score", f"{stats.proj_score_a:.0f} – {stats.proj_score_b:.0f}")
    with c4:
        winner = name_a if stats.proj_margin > 0 else name_b
        st.metric("Projected Margin", f"{winner} by {abs(stats.proj_margin):.1f}")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric(f"{name_a} Off Edge", f"{stats.adj_o_edge_a:+.1f}")
    with c6:
        st.metric(f"{name_b} Off Edge", f"{stats.adj_o_edge_b:+.1f}")
    with c7:
        st.metric("Projected Total", f"{stats.proj_total:.1f}")
    with c8:
        if stats.upset_flag:
            st.metric("Upset Alert", f"🚨 {stats.upset_team}")
        else:
            st.metric("Seed Advantage", f"{stats.seed_diff:+.0f}")


def _render_prediction(prediction: MatchupPrediction, name_a: str, name_b: str, color_a: str, color_b: str):
    st.markdown('<div class="section-header">Model Predictions</div>', unsafe_allow_html=True)

    f = prediction.formula
    ml = prediction.ml

    if ml is not None:
        if prediction.models_agree:
            st.markdown(f'<span class="agree-badge agree">Models Agree: {prediction.consensus_winner}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="agree-badge disagree">Models Disagree — Formula: {f.predicted_winner} · ML: {ml.predicted_winner}</span>', unsafe_allow_html=True)

    _render_probability_bar(name_a, name_b, f.win_prob_a, color_a, color_b, "Formula Model")
    if ml:
        _render_probability_bar(name_a, name_b, ml.win_prob_a, color_a, color_b, "ML Model")

    if ml:
        col_f, col_ml = st.columns(2)
    else:
        col_f = st.container()
        col_ml = None

    with col_f:
        st.markdown("**Formula Model**")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.metric("Winner", f.predicted_winner)
        with fc2:
            st.metric("Win Prob", fmt_probability(f.win_prob_a if f.predicted_winner == name_a else f.win_prob_b))
        with fc3:
            st.metric("Confidence", f.confidence)

        fc4, fc5, fc6 = st.columns(3)
        with fc4:
            st.metric("Projected Score", f"{f.score_a} – {f.score_b}")
        with fc5:
            fav = name_a if f.margin > 0 else name_b
            st.metric("Fair Spread", fmt_spread(f.margin, fav))
        with fc6:
            st.metric("Fair ML", f"{f.fair_ml_a} / {f.fair_ml_b}")

    if col_ml and ml:
        with col_ml:
            st.markdown("**ML Model**")
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("Winner", ml.predicted_winner)
            with mc2:
                st.metric("Win Prob", fmt_probability(ml.win_prob_a if ml.predicted_winner == name_a else ml.win_prob_b))
            with mc3:
                st.metric("Confidence", ml.confidence)

    with st.expander("📖 What do the confidence levels mean?"):
        st.markdown(
            "**Strong Lean** — The model gives the favored team a **65%+ win probability**. "
            "This is a high-confidence pick — the stats strongly favor one side.\n\n"
            "**Solid** — Win probability between **57–65%**. "
            "The model sees a clear advantage, but there's meaningful upset potential.\n\n"
            "**Lean** — Win probability below **57%**. "
            "This is close to a toss-up — the model slightly favors one team but wouldn't be surprised by either outcome."
        )


def _pick_html(pick: str, rationale: str, is_good: bool) -> str:
    css = "good-bet" if is_good else "bad-bet"
    icon = "✅" if is_good else "⚪"
    return (
        f'<div class="{css}" style="margin-bottom:8px;">'
        f'<div style="font-size:1.05rem;font-weight:700;">{icon} {pick}</div>'
        f'<div style="font-size:0.78rem;color:#9ca3af;margin-top:2px;">{rationale}</div>'
        f'</div>'
    )


def _render_market_comparison(prediction: MatchupPrediction, espn_odds: dict | None, multi_book: dict | None, name_a: str, name_b: str):
    """Render actionable betting picks per sportsbook."""
    has_espn = espn_odds and (espn_odds.get("spread") is not None or espn_odds.get("ml_home"))
    has_multi = bool(multi_book)

    if not has_espn and not has_multi:
        return

    st.markdown('<div class="section-header">Sportsbook Picks</div>', unsafe_allow_html=True)

    model_spread = prediction.formula.fair_spread
    model_total = prediction.formula.total
    model_winner = prediction.formula.predicted_winner
    model_prob_a = prediction.formula.win_prob_a

    books: list[tuple[str, dict]] = []

    if has_espn:
        dk: dict = {
            "name": "DraftKings",
            "spread": espn_odds.get("spread"),
            "spread_detail": espn_odds.get("spread_detail", ""),
            "ml_home": espn_odds.get("ml_home", ""),
            "ml_away": espn_odds.get("ml_away", ""),
            "total": espn_odds.get("over_under"),
        }
        books.append(("draftkings", dk))

    if has_multi:
        for bk_key in ["draftkings", "fanduel"]:
            if bk_key in multi_book:
                bk = multi_book[bk_key]
                entry = {
                    "name": bk.get("name", bk_key.title()),
                    "spread": bk.get("spread"),
                    "spread_detail": f"{bk.get('spread', '')}" if bk.get("spread") is not None else "",
                    "ml_home": str(bk.get("ml_home", "")),
                    "ml_away": str(bk.get("ml_away", "")),
                    "total": bk.get("total"),
                }
                existing = [i for i, (k, _) in enumerate(books) if k == bk_key]
                if existing:
                    books[existing[0]] = (bk_key, entry)
                else:
                    books.append((bk_key, entry))

    if not books:
        return

    cols = st.columns(len(books))

    for col, (bk_key, bk) in zip(cols, books):
        with col:
            st.markdown(f"**{bk['name']}**")

            # --- Spread pick ---
            mkt_spread = bk.get("spread")
            if mkt_spread is not None:
                try:
                    mkt_spread = float(mkt_spread)
                except (TypeError, ValueError):
                    mkt_spread = None

            if mkt_spread is not None:
                spread_edge = model_spread - mkt_spread
                spread_detail = bk.get("spread_detail", "")

                if abs(spread_edge) >= 1.5:
                    if spread_edge > 0:
                        pick_team = name_b
                        reasoning = f"Market has {spread_detail}, but model says the fair line is {model_spread:+.1f} — {abs(spread_edge):.1f} pts of value on {name_b}"
                    else:
                        pick_team = name_a
                        reasoning = f"Market has {spread_detail}, but model says the fair line is {model_spread:+.1f} — {abs(spread_edge):.1f} pts of value on {name_a}"
                    st.markdown(_pick_html(f"Take {pick_team} {spread_detail}", reasoning, True), unsafe_allow_html=True)
                else:
                    st.markdown(_pick_html(f"Spread: {spread_detail}", f"Model line: {model_spread:+.1f} — no significant edge ({spread_edge:+.1f})", False), unsafe_allow_html=True)
            else:
                st.markdown(_pick_html("Spread: N/A", "No spread available", False), unsafe_allow_html=True)

            # --- Moneyline pick ---
            ml_h = bk.get("ml_home", "")
            ml_a = bk.get("ml_away", "")

            if ml_h and ml_a:
                ml_pick = model_winner
                model_prob = model_prob_a if model_winner == name_a else (1 - model_prob_a)
                fair_ml = prediction.formula.fair_ml_a if model_winner == name_a else prediction.formula.fair_ml_b
                market_ml = ml_a if model_winner == name_a else ml_h
                other_team = name_b if model_winner == name_a else name_a

                try:
                    mkt_ml_val = float(str(market_ml).replace("EVEN", "100").replace("+", ""))
                    if mkt_ml_val >= 0:
                        implied = 100 / (mkt_ml_val + 100)
                    else:
                        implied = abs(mkt_ml_val) / (abs(mkt_ml_val) + 100)
                except (ValueError, ZeroDivisionError):
                    implied = None

                if implied is not None:
                    edge_pct = (model_prob - implied) * 100
                    if edge_pct > 3:
                        st.markdown(_pick_html(
                            f"Take {ml_pick} ML ({market_ml})",
                            f"Model gives {ml_pick} a {model_prob:.0%} chance (fair ML: {fair_ml}) — {edge_pct:.1f}% edge over implied {implied:.0%}",
                            True,
                        ), unsafe_allow_html=True)
                    elif edge_pct < -3:
                        other_ml = ml_h if model_winner == name_a else ml_a
                        st.markdown(_pick_html(
                            f"Lean {other_team} ML ({other_ml})",
                            f"Model says {ml_pick} is overpriced at {market_ml} (implied {implied:.0%}, model {model_prob:.0%})",
                            True,
                        ), unsafe_allow_html=True)
                    else:
                        st.markdown(_pick_html(
                            f"ML: {name_a} {ml_a} / {name_b} {ml_h}",
                            f"Model fair: {prediction.formula.fair_ml_a} / {prediction.formula.fair_ml_b} — no clear ML edge",
                            False,
                        ), unsafe_allow_html=True)
                else:
                    st.markdown(_pick_html(f"ML: {name_a} {ml_a} / {name_b} {ml_h}", f"Fair: {prediction.formula.fair_ml_a} / {prediction.formula.fair_ml_b}", False), unsafe_allow_html=True)
            else:
                st.markdown(_pick_html("Moneyline: N/A", "No moneyline available", False), unsafe_allow_html=True)

            # --- Total pick ---
            mkt_total = bk.get("total")
            if mkt_total is not None:
                try:
                    mkt_total = float(mkt_total)
                except (TypeError, ValueError):
                    mkt_total = None

            if mkt_total is not None:
                total_edge = model_total - mkt_total
                if total_edge >= 2.0:
                    st.markdown(_pick_html(
                        f"Take the OVER {mkt_total}",
                        f"Model projects {model_total} total points — {total_edge:.1f} pts above the line",
                        True,
                    ), unsafe_allow_html=True)
                elif total_edge <= -2.0:
                    st.markdown(_pick_html(
                        f"Take the UNDER {mkt_total}",
                        f"Model projects {model_total} total points — {abs(total_edge):.1f} pts below the line",
                        True,
                    ), unsafe_allow_html=True)
                else:
                    st.markdown(_pick_html(
                        f"Total: {mkt_total}",
                        f"Model projects {model_total} — no significant edge ({total_edge:+.1f})",
                        False,
                    ), unsafe_allow_html=True)
            else:
                st.markdown(_pick_html("Total: N/A", "No total available", False), unsafe_allow_html=True)


def _get_multi_book_odds(away_name: str, home_name: str) -> dict | None:
    """Look up multi-book odds for a game by team names."""
    if not multi_odds:
        return None
    key = f"{away_name.lower()} vs {home_name.lower()}"
    if key in multi_odds:
        return multi_odds[key]
    for k, v in multi_odds.items():
        if away_name.lower() in k and home_name.lower() in k:
            return v
    return None


def _render_news(news: list[dict]) -> None:
    """Render game-related news articles."""
    if not news:
        return
    st.markdown('<div class="section-header">Related News</div>', unsafe_allow_html=True)
    for article in news[:4]:
        headline = article.get("headline", "")
        link = article.get("link", "")
        if link:
            st.markdown(f'<div class="news-card"><a href="{link}" target="_blank">{headline} →</a></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="news-card">{headline}</div>', unsafe_allow_html=True)


def _render_leaders(leaders: list[dict]) -> None:
    """Render per-team stat leaders (points, assists, rebounds)."""
    if not leaders:
        return
    st.markdown('<div class="section-header">Top Players</div>', unsafe_allow_html=True)

    cols = st.columns(len(leaders)) if len(leaders) > 1 else [st.container()]

    for col, team_block in zip(cols, leaders):
        with col:
            team_name = team_block.get("team", "")
            st.markdown(f"**{team_name}**")
            for stat in team_block.get("stats", []):
                player = stat.get("player", "")
                pos = stat.get("position", "")
                jersey = stat.get("jersey", "")
                value = stat.get("value", "")
                cat = stat.get("category", "")
                headshot = stat.get("headshot", "")

                player_label = f"#{jersey} {player}" if jersey else player
                if pos:
                    player_label += f" ({pos})"

                if headshot:
                    st.markdown(
                        f'<div class="leader-row">'
                        f'<img src="{headshot}" alt="{player}">'
                        f'<div><strong>{cat}</strong>: {player_label} — {value}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"**{cat}**: {player_label} — {value}")


def _render_full_matchup(team_a: dict, team_b: dict, espn_odds: dict | None = None, multi_book: dict | None = None, game_id: str = ""):
    name_a = team_a.get("team", "Team A")
    name_b = team_b.get("team", "Team B")
    color_a = team_a.get("color", "3b82f6")
    color_b = team_b.get("color", "ef4444")

    try:
        _render_team_comparison(team_a, team_b)
    except Exception as e:
        st.error(f"Could not render team comparison: {e}")

    try:
        _render_shooting_stats(team_a, team_b)
    except Exception as e:
        st.caption(f"Shooting stats unavailable: {e}")

    try:
        _render_height_comparison(team_a, team_b)
    except Exception as e:
        st.caption(f"Height comparison unavailable: {e}")

    try:
        stats = compute_matchup_stats(team_a, team_b)
        _render_derived_stats(stats, name_a, name_b)
    except Exception as e:
        st.error(f"Could not compute matchup analytics: {e}")

    try:
        prediction = predict_matchup(team_a, team_b)
        _render_prediction(prediction, name_a, name_b, color_a, color_b)
    except Exception as e:
        st.error(f"Could not generate prediction: {e}")
        return

    try:
        _render_market_comparison(prediction, espn_odds, multi_book, name_a, name_b)
    except Exception as e:
        st.caption(f"Market odds unavailable: {e}")

    # Game summary: news + stat leaders
    if game_id:
        try:
            summary = espn_client.get_game_summary(game_id)
            if summary:
                _render_leaders(summary.get("leaders", []))
                _render_news(summary.get("news", []))
        except Exception:
            pass


# ===========================================================================
# Tab 1: Live Matchups
# ===========================================================================

with tab_live:
    if not all_games:
        st.info("No tournament games found. Check back when the bracket is released.")
    else:
        rounds = _sort_rounds([r for r in {g.get("round", "") for g in all_games} if r])
        regions = sorted({g.get("region", "") for g in all_games if g.get("region")})

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            round_sel = st.selectbox("Round", ["All"] + rounds, key="live_round")
        with fc2:
            status_sel = st.selectbox("Status", ["All", "Upcoming", "Live", "Final"], key="live_status")
        with fc3:
            region_sel = st.selectbox("Region", ["All"] + regions, key="live_region")

        filtered = espn_client.filter_games(
            all_games, round_filter=round_sel, status_filter=status_sel, region_filter=region_sel,
        )

        if not filtered:
            st.warning("No games match the current filters.")
        else:
            labels = {espn_client.get_game_display_label(g): g for g in filtered}
            selected_label = st.selectbox("Select Matchup", list(labels.keys()), key="live_matchup")
            game = labels[selected_label]

            home = game["home_team"]
            away = game["away_team"]

            # Game header
            st.divider()
            hc1, hc2, hc3 = st.columns([1, 3, 1])
            with hc1:
                if away.get("logo_url"):
                    st.image(away["logo_url"], width=80)
                seed_a = f"({away['seed']}) " if away.get("seed") else ""
                st.markdown(f"### {seed_a}{away.get('short_name', away.get('name', ''))}")
            with hc2:
                st.markdown("<h2 style='text-align:center;'>vs</h2>", unsafe_allow_html=True)
                meta_parts = []
                if game.get("round"):
                    meta_parts.append(game["round"])
                if game.get("region"):
                    meta_parts.append(f"{game['region']} Region")
                if game.get("venue"):
                    meta_parts.append(game["venue"])
                st.markdown(f"<p style='text-align:center; color:#9ca3af;'>{' · '.join(meta_parts)}</p>", unsafe_allow_html=True)

                sub_parts = []
                if game.get("status_detail"):
                    sub_parts.append(game["status_detail"])
                if game.get("broadcast"):
                    sub_parts.append(f"📺 {game['broadcast']}")
                if sub_parts:
                    st.markdown(f"<p style='text-align:center; color:#6b7280; font-size:0.85rem;'>{' · '.join(sub_parts)}</p>", unsafe_allow_html=True)

                if game.get("home_score") and game.get("away_score"):
                    st.markdown(f"<h3 style='text-align:center;'>{game['away_score']} – {game['home_score']}</h3>", unsafe_allow_html=True)
            with hc3:
                if home.get("logo_url"):
                    st.image(home["logo_url"], width=80)
                seed_b = f"({home['seed']}) " if home.get("seed") else ""
                st.markdown(f"### {seed_b}{home.get('short_name', home.get('name', ''))}")

            stats_a = _lookup_team(away.get("name", ""))
            stats_b = _lookup_team(home.get("name", ""))

            if stats_a and stats_b:
                stats_a.setdefault("team", away.get("name", "Team A"))
                stats_b.setdefault("team", home.get("name", "Team B"))
                stats_a.setdefault("color", away.get("color", "3b82f6"))
                stats_b.setdefault("color", home.get("color", "ef4444"))
                if away.get("seed"):
                    stats_a["seed"] = away["seed"]
                if home.get("seed"):
                    stats_b["seed"] = home["seed"]

                mb = _get_multi_book_odds(away.get("name", ""), home.get("name", ""))
                _render_full_matchup(stats_a, stats_b, game.get("odds"), mb, game_id=game.get("game_id", ""))
            else:
                missing = [n for n, s in [(away.get("name"), stats_a), (home.get("name"), stats_b)] if not s]
                st.warning(f"KenPom data not found for: {', '.join(missing)}. Data source: {stats_source}.")


# ===========================================================================
# Tab 2: Live Bracket
# ===========================================================================

with tab_bracket:
    if not all_games:
        st.info("No bracket data available yet. Check back when the bracket is released.")
    else:
        st.caption("Scroll right within each region to see later rounds. Winners advance automatically as games finish.")

        # Region selector: All or individual
        region_opts = ["Full Bracket"] + sorted({g.get("region", "") for g in all_games if g.get("region")})
        bracket_region = st.selectbox("View", region_opts, key="bracket_region")

        if bracket_region == "Full Bracket":
            bracket_html = render_full_bracket(all_games)
        else:
            from utils.bracket import build_region_bracket, build_final_four
            region_games = [g for g in all_games if g.get("region") == bracket_region]
            bracket_html = build_region_bracket(region_games, bracket_region)
            # Also show Final Four if relevant
            ff = [g for g in all_games if "Final Four" in g.get("round", "") or "National Championship" in g.get("round", "")]
            if ff:
                bracket_html += build_final_four(ff)

        import streamlit.components.v1 as components
        components.html(bracket_html, height=max(600, len(all_games) * 8), scrolling=True)


# ===========================================================================
# Tab 3: NIT
# ===========================================================================

with tab_nit:
    if not nit_games:
        st.info("No NIT games found yet. Check back after Selection Sunday.")
    else:
        nit_rounds = _sort_rounds([r for r in {g.get("round", "") for g in nit_games} if r])

        nc1, nc2 = st.columns(2)
        with nc1:
            nit_round_sel = st.selectbox("Round", ["All"] + nit_rounds, key="nit_round")
        with nc2:
            nit_status_sel = st.selectbox("Status", ["All", "Upcoming", "Live", "Final"], key="nit_status")

        nit_filtered = espn_client.filter_games(
            nit_games, round_filter=nit_round_sel, status_filter=nit_status_sel,
        )

        if not nit_filtered:
            st.warning("No NIT games match the current filters.")
        else:
            nit_labels = {espn_client.get_game_display_label(g): g for g in nit_filtered}
            nit_label = st.selectbox("Select NIT Matchup", list(nit_labels.keys()), key="nit_matchup")
            game = nit_labels[nit_label]

            home = game["home_team"]
            away = game["away_team"]

            st.divider()
            hc1, hc2, hc3 = st.columns([1, 3, 1])
            with hc1:
                if away.get("logo_url"):
                    st.image(away["logo_url"], width=80)
                st.markdown(f"### {away.get('short_name', away.get('name', ''))}")
            with hc2:
                st.markdown("<h2 style='text-align:center;'>vs</h2>", unsafe_allow_html=True)
                meta_parts = [p for p in [game.get("round", ""), game.get("venue", "")] if p]
                st.markdown(f"<p style='text-align:center; color:#9ca3af;'>{' · '.join(meta_parts)}</p>", unsafe_allow_html=True)
                sub_parts = []
                if game.get("status_detail"):
                    sub_parts.append(game["status_detail"])
                if game.get("broadcast"):
                    sub_parts.append(f"📺 {game['broadcast']}")
                if sub_parts:
                    st.markdown(f"<p style='text-align:center; color:#6b7280; font-size:0.85rem;'>{' · '.join(sub_parts)}</p>", unsafe_allow_html=True)
                if game.get("home_score") and game.get("away_score"):
                    st.markdown(f"<h3 style='text-align:center;'>{game['away_score']} – {game['home_score']}</h3>", unsafe_allow_html=True)
            with hc3:
                if home.get("logo_url"):
                    st.image(home["logo_url"], width=80)
                st.markdown(f"### {home.get('short_name', home.get('name', ''))}")

            nit_stats_a = _lookup_team(away.get("name", ""))
            nit_stats_b = _lookup_team(home.get("name", ""))

            if nit_stats_a and nit_stats_b:
                nit_stats_a = dict(nit_stats_a)
                nit_stats_b = dict(nit_stats_b)
                nit_stats_a.setdefault("team", away.get("name", "Team A"))
                nit_stats_b.setdefault("team", home.get("name", "Team B"))
                nit_stats_a.setdefault("color", away.get("color", "3b82f6"))
                nit_stats_b.setdefault("color", home.get("color", "ef4444"))

                _render_full_matchup(nit_stats_a, nit_stats_b, game.get("odds"), game_id=game.get("game_id", ""))
            else:
                missing = [n for n, s in [(away.get("name"), nit_stats_a), (home.get("name"), nit_stats_b)] if not s]
                st.warning(f"KenPom data not found for: {', '.join(missing)}.")


# ===========================================================================
# Tab 4: Stats Deep Dive
# ===========================================================================

with tab_stats:
    st.markdown("### KenPom Stats Explorer")
    st.caption("Sort, filter, and compare every D-I team. Stack multiple filters to find exactly what you're looking for.")

    if not team_stats:
        st.info("No team data loaded. Connect KenPom API or upload a CSV.")
    else:
        import pandas as pd

        _STAT_COLS = [
            ("AdjEM", "adj_em", True),
            ("AdjO", "adj_o", True),
            ("AdjD", "adj_d", False),
            ("Tempo", "tempo", None),
            ("SOS", "sos", True),
            ("NCSOS", "ncsos", True),
            ("Luck", "luck", None),
            ("Wins", "wins", True),
            ("Losses", "losses", False),
        ]

        rows = []
        for name, data in team_stats.items():
            row: dict = {"Team": name}
            row["Seed"] = data.get("seed", "")
            row["Conf"] = data.get("conference", "")
            row["Record"] = f"{data.get('wins', '?')}-{data.get('losses', '?')}" if data.get("wins") else ""
            for label, key, _ in _STAT_COLS:
                val = data.get(key)
                row[label] = round(float(val), 2) if isinstance(val, (int, float)) else None
            rows.append(row)

        stats_df = pd.DataFrame(rows)

        # --- Conference filter ---
        all_confs = sorted(stats_df["Conf"].dropna().unique().tolist())
        all_confs = [c for c in all_confs if c]

        filter_conf = st.multiselect("Filter by Conference", all_confs, default=[], key="stats_conf")
        if filter_conf:
            stats_df = stats_df[stats_df["Conf"].isin(filter_conf)]

        # --- Multi-sort builder ---
        st.markdown('<div class="section-header">Sort & Filter</div>', unsafe_allow_html=True)
        st.caption("Add multiple sort rules. They apply in order — first sort is primary, second is tiebreaker, etc.")

        sortable_labels = [label for label, _, _ in _STAT_COLS]

        if "sort_rules" not in st.session_state:
            st.session_state.sort_rules = [{"col": "AdjEM", "dir": "Highest first"}]

        def _add_sort_rule():
            st.session_state.sort_rules.append({"col": "AdjEM", "dir": "Highest first"})

        def _remove_sort_rule(idx):
            if len(st.session_state.sort_rules) > 1:
                st.session_state.sort_rules.pop(idx)

        for i, rule in enumerate(st.session_state.sort_rules):
            rc1, rc2, rc3 = st.columns([3, 3, 1])
            with rc1:
                rule["col"] = st.selectbox(
                    f"Sort by" if i == 0 else f"Then by",
                    sortable_labels,
                    index=sortable_labels.index(rule["col"]) if rule["col"] in sortable_labels else 0,
                    key=f"sort_col_{i}",
                )
            with rc2:
                rule["dir"] = st.selectbox(
                    "Direction",
                    ["Highest first", "Lowest first"],
                    index=0 if rule["dir"] == "Highest first" else 1,
                    key=f"sort_dir_{i}",
                )
            with rc3:
                st.markdown("<div style='padding-top:28px;'></div>", unsafe_allow_html=True)
                if i > 0:
                    st.button("✕", key=f"rm_sort_{i}", on_click=_remove_sort_rule, args=(i,))

        if len(st.session_state.sort_rules) < 4:
            st.button("+ Add sort rule", on_click=_add_sort_rule, key="add_sort")

        # --- Stat range filters ---
        st.markdown('<div class="section-header">Stat Filters</div>', unsafe_allow_html=True)
        st.caption("Narrow the table to teams within specific stat ranges.")

        filter_cols = st.multiselect(
            "Choose stats to filter",
            sortable_labels,
            default=[],
            key="stat_filter_cols",
        )

        range_filters: list[tuple[str, float, float]] = []
        if filter_cols:
            for fc in filter_cols:
                col_data = stats_df[fc].dropna()
                if col_data.empty:
                    continue
                col_min = float(col_data.min())
                col_max = float(col_data.max())
                fmin, fmax = st.slider(
                    f"{fc} range",
                    min_value=col_min,
                    max_value=col_max,
                    value=(col_min, col_max),
                    key=f"filter_range_{fc}",
                )
                range_filters.append((fc, fmin, fmax))

        # --- Apply range filters ---
        for fc, fmin, fmax in range_filters:
            stats_df = stats_df[stats_df[fc].between(fmin, fmax)]

        # --- Apply sort ---
        sort_by = [r["col"] for r in st.session_state.sort_rules if r["col"] in stats_df.columns]
        sort_asc = [r["dir"] == "Lowest first" for r in st.session_state.sort_rules if r["col"] in stats_df.columns]

        if sort_by:
            stats_df = stats_df.sort_values(by=sort_by, ascending=sort_asc, na_position="last")

        stats_df = stats_df.reset_index(drop=True)
        stats_df.index = stats_df.index + 1
        stats_df.index.name = "#"

        # --- Summary metrics ---
        st.markdown(f"**{len(stats_df)} teams** matching your filters")

        if len(stats_df) > 0 and sort_by:
            primary_col = sort_by[0]
            primary_asc = sort_asc[0]
            mc1, mc2, mc3, mc4 = st.columns(4)
            col_vals = stats_df[primary_col].dropna()
            if not col_vals.empty:
                with mc1:
                    best_idx = col_vals.idxmin() if primary_asc else col_vals.idxmax()
                    best_team = stats_df.loc[best_idx, "Team"]
                    best_val = col_vals.loc[best_idx]
                    st.metric(f"#1 {primary_col}", f"{best_val:.1f}", help=best_team)
                with mc2:
                    st.metric(f"Avg {primary_col}", f"{col_vals.mean():.1f}")
                with mc3:
                    st.metric(f"Median {primary_col}", f"{col_vals.median():.1f}")
                with mc4:
                    st.metric(f"Std Dev", f"{col_vals.std():.2f}")

        # --- Display table ---
        display_cols = ["Team", "Seed", "Conf", "Record"] + sortable_labels
        display_cols = [c for c in display_cols if c in stats_df.columns]

        st.dataframe(
            stats_df[display_cols],
            use_container_width=True,
            height=min(700, 38 + len(stats_df) * 35),
        )


# ===========================================================================
# Tab 5: Team Profile
# ===========================================================================

with tab_deep_dive:
    st.markdown("### Full KenPom Team Profile")
    st.caption("Select any team to view their complete KenPom ratings and advanced stats.")

    if not team_stats:
        st.info("No team data loaded. Connect KenPom API or upload a CSV.")
    else:
        try:
            all_names = sorted(team_stats.keys())
            selected_team = st.selectbox("Select Team", all_names, key="dd_team")

            if selected_team:
                t = team_stats.get(selected_team, {})

                st.markdown(f"## {selected_team}")
                seed_str = f"Seed: {t['seed']}" if t.get("seed") else ""
                conf_str = f"Conference: {t.get('conference', '—')}"
                record_str = f"Record: {t.get('wins', '?')}-{t.get('losses', '?')}" if t.get("wins") else ""
                coach_str = f"Coach: {t.get('coach', '')}" if t.get("coach") else ""
                parts = [p for p in [seed_str, conf_str, record_str, coach_str] if p]
                st.caption(" · ".join(parts))

                st.divider()

                st.markdown('<div class="section-header">Efficiency Ratings</div>', unsafe_allow_html=True)
                rc1, rc2, rc3, rc4, rc5, rc6, rc7 = st.columns(7)
                for col, (lbl, key) in zip(
                    [rc1, rc2, rc3, rc4, rc5, rc6, rc7],
                    [("AdjEM", "adj_em"), ("AdjO", "adj_o"), ("AdjD", "adj_d"),
                     ("Tempo", "tempo"), ("SOS", "sos"), ("NCSOS", "ncsos"), ("Luck", "luck")],
                ):
                    with col:
                        val = t.get(key, "—")
                        display = f"{val:.2f}" if isinstance(val, float) else str(val)
                        st.metric(lbl, display)

                with st.expander("All Available Data"):
                    import pandas as pd
                    display_data = {k: str(v) for k, v in t.items() if k != "team"}
                    df = pd.DataFrame(list(display_data.items()), columns=["Metric", "Value"])
                    st.dataframe(df, use_container_width=False, width=600, hide_index=True)
        except Exception as e:
            st.error(f"Error loading team profile: {e}")


# ===========================================================================
# Tab 3: Future Matchup Lab
# ===========================================================================

with tab_lab:
    st.markdown("### Hypothetical Matchup Lab")
    st.caption("Pick any two teams and run the prediction engine — great for future rounds and what-if scenarios.")

    if not team_stats:
        st.info("No team data loaded. Connect KenPom API or upload a CSV.")
    else:
        all_names = sorted(team_stats.keys())

        lc1, lc2, lc3 = st.columns([2, 1, 2])
        with lc1:
            team_a_name = st.selectbox("Team A", all_names, index=0, key="lab_a")
        with lc2:
            st.markdown("<div style='text-align:center; padding-top:28px; font-size:1.5rem;'>vs</div>", unsafe_allow_html=True)
        with lc3:
            team_b_name = st.selectbox("Team B", all_names, index=min(1, len(all_names) - 1), key="lab_b")

        if st.button("Predict", use_container_width=True, type="primary"):
            if team_a_name == team_b_name:
                st.error("Please select two different teams.")
            else:
                sa = team_stats.get(team_a_name)
                sb = team_stats.get(team_b_name)
                if sa and sb:
                    sa = dict(sa)
                    sb = dict(sb)
                    sa["team"] = team_a_name
                    sb["team"] = team_b_name
                    st.divider()
                    _render_full_matchup(sa, sb)
                else:
                    st.error("Could not load stats for one or both teams.")


# ===========================================================================
# Footer
# ===========================================================================

st.markdown(f"""
<div class="footer">
    March Madness Prediction Engine · Data: {stats_source} + {sched_source}<br>
    Predictions are for entertainment purposes only · Not gambling advice<br>
    Built with Streamlit · <a href="https://joecharland.dev" style="color:#6b7280;">joecharland.dev</a>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Auto-refresh every 2 minutes (scores + odds)
# ---------------------------------------------------------------------------

@st.fragment(run_every=120)
def _auto_refresh():
    """Silent fragment that triggers a cache-busting rerun on interval."""
    pass

_auto_refresh()
