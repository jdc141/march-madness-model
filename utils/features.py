"""Derived matchup statistics and ML feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ML_FEATURE_NAMES = [
    "adj_em_diff",
    "adj_o_edge_a",
    "adj_o_edge_b",
    "tempo_diff",
    "seed_diff",
    "sos_diff",
    "luck_diff",
]

_D1_AVG_EFF = 100.0
_D1_AVG_TEMPO = 67.5


def _f(team: dict, key: str, default: float) -> float:
    """Safely extract a numeric value, coercing to float."""
    val = team.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


@dataclass
class MatchupStats:
    adj_em_diff: float
    adj_o_edge_a: float
    adj_o_edge_b: float
    tempo_projection: float
    seed_diff: float
    proj_score_a: float
    proj_score_b: float
    proj_margin: float
    proj_total: float
    upset_flag: bool
    upset_team: str | None


def compute_matchup_stats(
    team_a: dict[str, Any],
    team_b: dict[str, Any],
) -> MatchupStats:
    adj_em_a = _f(team_a, "adj_em", 0)
    adj_em_b = _f(team_b, "adj_em", 0)
    adj_o_a = _f(team_a, "adj_o", _D1_AVG_EFF)
    adj_d_a = _f(team_a, "adj_d", _D1_AVG_EFF)
    adj_o_b = _f(team_b, "adj_o", _D1_AVG_EFF)
    adj_d_b = _f(team_b, "adj_d", _D1_AVG_EFF)
    tempo_a = _f(team_a, "tempo", _D1_AVG_TEMPO)
    tempo_b = _f(team_b, "tempo", _D1_AVG_TEMPO)
    seed_a = _f(team_a, "seed", 8)
    seed_b = _f(team_b, "seed", 8)

    adj_em_diff = adj_em_a - adj_em_b
    adj_o_edge_a = adj_o_a - adj_d_b
    adj_o_edge_b = adj_o_b - adj_d_a

    tempo_proj = (tempo_a + tempo_b) / 2
    seed_diff = seed_b - seed_a

    exp_eff_a = 100 + (adj_o_a - 100) - (adj_d_b - 100)
    exp_eff_b = 100 + (adj_o_b - 100) - (adj_d_a - 100)

    score_a = exp_eff_a * tempo_proj / 100
    score_b = exp_eff_b * tempo_proj / 100
    margin = score_a - score_b + 0.12 * seed_diff
    total = score_a + score_b

    upset_flag = False
    upset_team = None
    if margin > 0 and seed_a > seed_b:
        upset_flag = True
        upset_team = team_a.get("team", "Team A")
    elif margin < 0 and seed_b > seed_a:
        upset_flag = True
        upset_team = team_b.get("team", "Team B")

    return MatchupStats(
        adj_em_diff=adj_em_diff,
        adj_o_edge_a=adj_o_edge_a,
        adj_o_edge_b=adj_o_edge_b,
        tempo_projection=tempo_proj,
        seed_diff=seed_diff,
        proj_score_a=score_a,
        proj_score_b=score_b,
        proj_margin=margin,
        proj_total=total,
        upset_flag=upset_flag,
        upset_team=upset_team,
    )


def build_ml_features(
    team_a: dict[str, Any],
    team_b: dict[str, Any],
) -> list[float]:
    """Build the feature vector consumed by the ML model."""
    return [
        _f(team_a, "adj_em", 0) - _f(team_b, "adj_em", 0),
        _f(team_a, "adj_o", _D1_AVG_EFF) - _f(team_b, "adj_d", _D1_AVG_EFF),
        _f(team_b, "adj_o", _D1_AVG_EFF) - _f(team_a, "adj_d", _D1_AVG_EFF),
        _f(team_a, "tempo", _D1_AVG_TEMPO) - _f(team_b, "tempo", _D1_AVG_TEMPO),
        _f(team_b, "seed", 8) - _f(team_a, "seed", 8),
        _f(team_a, "sos", 0) - _f(team_b, "sos", 0),
        _f(team_a, "luck", 0) - _f(team_b, "luck", 0),
    ]


def compute_market_edges(
    model_spread: float,
    model_total: float,
    market_spread: float | None,
    market_total: float | None,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {"spread_edge": None, "total_edge": None}
    if market_spread is not None:
        result["spread_edge"] = model_spread - market_spread
    if market_total is not None:
        result["total_edge"] = model_total - market_total
    return result
