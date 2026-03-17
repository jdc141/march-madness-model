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
    "seed_matchup",
    "tempo_mismatch",
    "off_def_asymmetry_a",
    "off_def_asymmetry_b",
    "efg_margin",
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

    exp_eff_a = adj_o_a * adj_d_b / _D1_AVG_EFF
    exp_eff_b = adj_o_b * adj_d_a / _D1_AVG_EFF

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
    adj_o_a = _f(team_a, "adj_o", _D1_AVG_EFF)
    adj_d_a = _f(team_a, "adj_d", _D1_AVG_EFF)
    adj_o_b = _f(team_b, "adj_o", _D1_AVG_EFF)
    adj_d_b = _f(team_b, "adj_d", _D1_AVG_EFF)
    seed_a = _f(team_a, "seed", 8)
    seed_b = _f(team_b, "seed", 8)
    off_efg_a = _f(team_a, "off_efg", 0)
    off_efg_b = _f(team_b, "off_efg", 0)
    def_efg_a = _f(team_a, "def_efg", 0)
    def_efg_b = _f(team_b, "def_efg", 0)
    tempo_a = _f(team_a, "tempo", _D1_AVG_TEMPO)
    tempo_b = _f(team_b, "tempo", _D1_AVG_TEMPO)

    return [
        # Core efficiency
        _f(team_a, "adj_em", 0) - _f(team_b, "adj_em", 0),
        adj_o_a - adj_d_b,
        adj_o_b - adj_d_a,
        tempo_a - tempo_b,
        seed_b - seed_a,
        # Four factors
        off_efg_a - off_efg_b,
        def_efg_a - def_efg_b,
        _f(team_a, "off_to", 0) - _f(team_b, "off_to", 0),
        _f(team_a, "off_orb", 0) - _f(team_b, "off_orb", 0),
        _f(team_a, "off_ftr", 0) - _f(team_b, "off_ftr", 0),
        _f(team_a, "def_to", 0) - _f(team_b, "def_to", 0),
        # Shooting
        _f(team_a, "fg3_pct", 0) - _f(team_b, "fg3_pct", 0),
        _f(team_a, "fg2_pct", 0) - _f(team_b, "fg2_pct", 0),
        _f(team_a, "ft_pct", 0) - _f(team_b, "ft_pct", 0),
        _f(team_a, "fg3_rate", 0) - _f(team_b, "fg3_rate", 0),
        _f(team_a, "ast_rate", 0) - _f(team_b, "ast_rate", 0),
        # Defense
        _f(team_a, "block_pct", 0) - _f(team_b, "block_pct", 0),
        _f(team_a, "stl_rate", 0) - _f(team_b, "stl_rate", 0),
        _f(team_a, "opp_fg3_pct", 0) - _f(team_b, "opp_fg3_pct", 0),
        _f(team_a, "opp_fg2_pct", 0) - _f(team_b, "opp_fg2_pct", 0),
        # Roster
        _f(team_a, "avg_hgt", 0) - _f(team_b, "avg_hgt", 0),
        _f(team_a, "experience", 0) - _f(team_b, "experience", 0),
        _f(team_a, "bench", 0) - _f(team_b, "bench", 0),
        _f(team_a, "continuity", 0) - _f(team_b, "continuity", 0),
        # Engineered
        seed_a * seed_b,
        abs(tempo_a - tempo_b),
        (adj_o_a - adj_d_b) - (adj_o_b - adj_d_a),
        (adj_o_b - adj_d_a) - (adj_o_a - adj_d_b),
        (off_efg_a - def_efg_b) - (off_efg_b - def_efg_a),
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
