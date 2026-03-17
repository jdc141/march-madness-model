"""HTML bracket renderer for the NCAA tournament."""

from __future__ import annotations

from typing import Any

_ROUND_ORDER = ["First Four", "1st Round", "2nd Round", "Sweet 16", "Elite 8"]
_FINAL_ROUNDS = ["Final Four", "National Championship"]

_REGIONS = ["East", "West", "South", "Midwest"]


def _team_cell(team: dict | None, is_winner: bool, state: str) -> str:
    """Render a single team row in a bracket game."""
    if team is None or not team.get("name") or team.get("name") == "TBD":
        return '<div class="bk-team bk-tbd"><span class="bk-seed"></span><span class="bk-name">TBD</span></div>'

    seed = team.get("seed")
    seed_html = f'<span class="bk-seed">({seed})</span>' if seed else '<span class="bk-seed"></span>'
    name = team.get("short_name") or team.get("name", "")
    score = team.get("score", "")
    score_html = f'<span class="bk-score">{score}</span>' if score else ""

    css_extra = ""
    if is_winner and state == "post":
        css_extra = " bk-winner"
    elif state == "in":
        css_extra = " bk-live"

    return f'<div class="bk-team{css_extra}">{seed_html}<span class="bk-name">{name}</span>{score_html}</div>'


def _game_html(game: dict, compact: bool = False) -> str:
    """Render a single bracket game (two team rows)."""
    home = game.get("home_team") or {}
    away = game.get("away_team") or {}
    state = game.get("state", "pre")

    home_winner = home.get("winner", False)
    away_winner = away.get("winner", False)

    # Top team = away (higher seed typically), bottom = home
    top = _team_cell(away, away_winner, state)
    bot = _team_cell(home, home_winner, state)

    # Scores
    home_score = game.get("home_score", "")
    away_score = game.get("away_score", "")

    if home_score:
        bot = bot.replace("</div>", f'<span class="bk-score">{home_score}</span></div>')
    if away_score:
        top = top.replace("</div>", f'<span class="bk-score">{away_score}</span></div>')

    status = ""
    if state == "in":
        status = '<div class="bk-status bk-status-live">LIVE</div>'
    elif state == "post":
        status = '<div class="bk-status">Final</div>'

    compact_cls = " bk-game-compact" if compact else ""
    return f'<div class="bk-game{compact_cls}">{top}{bot}{status}</div>'


def build_region_bracket(games: list[dict], region: str) -> str:
    """Build HTML for a single region's bracket."""
    rounds_map: dict[str, list[dict]] = {}
    for g in games:
        r = g.get("round", "")
        if r not in rounds_map:
            rounds_map[r] = []
        rounds_map[r].append(g)

    ordered_rounds = [r for r in _ROUND_ORDER if r in rounds_map]

    if not ordered_rounds:
        return ""

    html = f'<div class="bk-region"><div class="bk-region-title">{region} Region</div><div class="bk-rounds">'

    for rnd in ordered_rounds:
        round_games = rounds_map[rnd]
        compact = rnd in ("Sweet 16", "Elite 8")
        html += f'<div class="bk-round"><div class="bk-round-title">{rnd}</div>'
        for g in round_games:
            html += _game_html(g, compact=compact)
        html += '</div>'

    html += '</div></div>'
    return html


def build_final_four(games: list[dict]) -> str:
    """Build HTML for Final Four + Championship."""
    ff_games = [g for g in games if "Final Four" in g.get("round", "")]
    champ_games = [g for g in games if "Championship" in g.get("round", "") and "National" in g.get("round", "")]

    if not ff_games and not champ_games:
        return ""

    html = '<div class="bk-region"><div class="bk-region-title">Final Four & Championship</div><div class="bk-rounds">'

    if ff_games:
        html += '<div class="bk-round"><div class="bk-round-title">Final Four</div>'
        for g in ff_games:
            html += _game_html(g, compact=True)
        html += '</div>'

    if champ_games:
        html += '<div class="bk-round"><div class="bk-round-title">Championship</div>'
        for g in champ_games:
            html += _game_html(g, compact=True)
        html += '</div>'

    html += '</div></div>'
    return html


def render_full_bracket(all_games: list[dict]) -> str:
    """Render the complete NCAA tournament bracket as HTML."""
    ncaa_games = [g for g in all_games
                  if "NIT" not in g.get("round", "")
                  and "Crown" not in g.get("round", "")]

    # Group by region
    region_games: dict[str, list[dict]] = {}
    final_games: list[dict] = []

    for g in ncaa_games:
        region = g.get("region", "")
        rnd = g.get("round", "")
        if "Final Four" in rnd or "National Championship" in rnd:
            final_games.append(g)
        elif region:
            if region not in region_games:
                region_games[region] = []
            region_games[region].append(g)

    css = """
    <style>
    .bk-container { font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
    .bk-region { margin-bottom: 24px; }
    .bk-region-title {
        font-size: 1.1rem; font-weight: 700; color: #a5b4fc;
        padding: 8px 0; border-bottom: 2px solid rgba(100,100,140,0.3);
        margin-bottom: 12px;
    }
    .bk-rounds {
        display: flex; gap: 16px; overflow-x: auto; padding-bottom: 8px;
        align-items: flex-start;
    }
    .bk-round {
        min-width: 180px; flex-shrink: 0;
        display: flex; flex-direction: column; gap: 8px;
    }
    .bk-round-title {
        font-size: 0.75rem; font-weight: 600; color: #9ca3af;
        text-transform: uppercase; letter-spacing: 0.5px;
        padding-bottom: 4px; border-bottom: 1px solid rgba(100,100,140,0.2);
    }
    .bk-game {
        background: rgba(30,30,46,0.6);
        border: 1px solid rgba(100,100,140,0.25);
        border-radius: 6px; overflow: hidden;
        position: relative;
    }
    .bk-game-compact { min-width: 160px; }
    .bk-team {
        display: flex; align-items: center; gap: 6px;
        padding: 5px 8px; font-size: 0.8rem;
        border-bottom: 1px solid rgba(100,100,140,0.15);
    }
    .bk-team:last-of-type { border-bottom: none; }
    .bk-tbd { color: #6b7280; font-style: italic; }
    .bk-winner { background: rgba(16,185,129,0.15); font-weight: 600; }
    .bk-live { background: rgba(239,68,68,0.12); }
    .bk-seed { font-size: 0.7rem; color: #9ca3af; min-width: 22px; }
    .bk-name { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #e5e7eb; }
    .bk-score { font-weight: 700; font-size: 0.8rem; color: #e5e7eb; margin-left: auto; }
    .bk-status {
        text-align: center; font-size: 0.6rem; font-weight: 600;
        color: #6b7280; padding: 2px;
    }
    .bk-status-live { color: #ef4444; animation: blink 1.5s infinite; }
    @keyframes blink { 50% { opacity: 0.5; } }

    @media (max-width: 768px) {
        .bk-round { min-width: 150px; }
        .bk-team { font-size: 0.72rem; padding: 4px 6px; }
        .bk-seed { font-size: 0.65rem; }
    }
    </style>
    """

    html = css + '<div class="bk-container">'

    for region in _REGIONS:
        if region in region_games:
            html += build_region_bracket(region_games[region], region)

    html += build_final_four(final_games)
    html += '</div>'

    return html
