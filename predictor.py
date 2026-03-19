"""Dual prediction engine: deterministic formula + optional ML model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from utils.features import build_ml_features

_MODEL_DIR = Path(__file__).parent / "models"
_MODEL_PATH = _MODEL_DIR / "model.pkl"
_CACHED_MODEL = None

_D1_AVG_EFF = 100.0
_D1_AVG_TEMPO = 67.5

# Tournament games tend to score ~6% lower than pure efficiency ratings predict.
# Elite defenses, slower pace in elimination games, and high-stakes execution
# all compress scoring relative to regular-season KenPom projections.
_TOURNAMENT_SCORE_FACTOR = 0.94


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FormulaPrediction:
    predicted_winner: str
    win_prob_a: float
    win_prob_b: float
    score_a: float
    score_b: float
    margin: float
    total: float
    fair_spread: float
    fair_ml_a: str
    fair_ml_b: str
    confidence: str


@dataclass
class MLPrediction:
    predicted_winner: str
    win_prob_a: float
    win_prob_b: float
    confidence: str


@dataclass
class MatchupPrediction:
    formula: FormulaPrediction
    ml: MLPrediction | None
    models_agree: bool
    consensus_winner: str | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(team: dict, key: str, default: float) -> float:
    """Safely extract a numeric value, coercing to float."""
    val = team.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _confidence_label(prob: float) -> str:
    p = max(prob, 1 - prob)
    if p >= 0.65:
        return "Strong Lean"
    if p >= 0.57:
        return "Solid"
    return "Lean"


def _to_moneyline(prob: float) -> str:
    if prob <= 0 or prob >= 1:
        return "N/A"
    if prob >= 0.5:
        return f"{-100 * prob / (1 - prob):+.0f}"
    return f"+{100 * (1 - prob) / prob:.0f}"


# ---------------------------------------------------------------------------
# Formula model
# ---------------------------------------------------------------------------

def predict_formula(
    team_a: dict[str, Any],
    team_b: dict[str, Any],
) -> FormulaPrediction:
    """Possession-based deterministic prediction using KenPom-style ratings."""
    adj_o_a = _f(team_a, "adj_o", _D1_AVG_EFF)
    adj_d_a = _f(team_a, "adj_d", _D1_AVG_EFF)
    adj_o_b = _f(team_b, "adj_o", _D1_AVG_EFF)
    adj_d_b = _f(team_b, "adj_d", _D1_AVG_EFF)
    tempo_a = _f(team_a, "tempo", _D1_AVG_TEMPO)
    tempo_b = _f(team_b, "tempo", _D1_AVG_TEMPO)
    seed_a = _f(team_a, "seed", 8)
    seed_b = _f(team_b, "seed", 8)

    exp_eff_a = adj_o_a * adj_d_b / _D1_AVG_EFF
    exp_eff_b = adj_o_b * adj_d_a / _D1_AVG_EFF

    poss = (tempo_a + tempo_b) / 2
    raw_score_a = exp_eff_a * poss / 100
    raw_score_b = exp_eff_b * poss / 100

    # Margin and win probability use raw scores (preserves accuracy of spread/ML)
    margin = (raw_score_a - raw_score_b) + 0.12 * (seed_b - seed_a)

    # Apply tournament calibration to displayed scores/total only.
    # Tournament games score ~6% lower than pure efficiency ratings predict
    # due to high-stakes defense, tighter game management, and elimination pressure.
    score_a = raw_score_a * _TOURNAMENT_SCORE_FACTOR
    score_b = raw_score_b * _TOURNAMENT_SCORE_FACTOR
    total = score_a + score_b

    win_prob_a = 1 / (1 + math.exp(-margin / 6.8))
    win_prob_b = 1 - win_prob_a

    name_a = team_a.get("team", "Team A")
    name_b = team_b.get("team", "Team B")
    winner = name_a if margin > 0 else name_b

    return FormulaPrediction(
        predicted_winner=winner,
        win_prob_a=win_prob_a,
        win_prob_b=win_prob_b,
        score_a=round(score_a, 1),
        score_b=round(score_b, 1),
        margin=round(margin, 1),
        total=round(total, 1),
        fair_spread=round(-margin, 1),
        fair_ml_a=_to_moneyline(win_prob_a),
        fair_ml_b=_to_moneyline(win_prob_b),
        confidence=_confidence_label(win_prob_a),
    )


# ---------------------------------------------------------------------------
# ML model
# ---------------------------------------------------------------------------

def _load_model():
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL
    if not _MODEL_PATH.exists():
        return None
    try:
        import joblib
        _CACHED_MODEL = joblib.load(_MODEL_PATH)
        return _CACHED_MODEL
    except Exception:
        return None


def predict_ml(
    team_a: dict[str, Any],
    team_b: dict[str, Any],
) -> MLPrediction | None:
    """ML-based prediction. Returns None if no model is available."""
    model = _load_model()
    if model is None:
        return None

    try:
        features = np.array([build_ml_features(team_a, team_b)])
        prob_a = model.predict_proba(features)[0][1]
        prob_b = 1 - prob_a
    except Exception:
        return None

    name_a = team_a.get("team", "Team A")
    name_b = team_b.get("team", "Team B")
    winner = name_a if prob_a > 0.5 else name_b

    return MLPrediction(
        predicted_winner=winner,
        win_prob_a=round(prob_a, 4),
        win_prob_b=round(prob_b, 4),
        confidence=_confidence_label(prob_a),
    )


# ---------------------------------------------------------------------------
# Combined prediction
# ---------------------------------------------------------------------------

def predict_matchup(
    team_a: dict[str, Any],
    team_b: dict[str, Any],
) -> MatchupPrediction:
    """Run both models and return a combined prediction."""
    formula = predict_formula(team_a, team_b)
    ml = predict_ml(team_a, team_b)

    if ml is not None:
        agree = formula.predicted_winner == ml.predicted_winner
        consensus = formula.predicted_winner if agree else None
    else:
        agree = True
        consensus = formula.predicted_winner

    return MatchupPrediction(
        formula=formula,
        ml=ml,
        models_agree=agree,
        consensus_winner=consensus,
    )
