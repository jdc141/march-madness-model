"""Train ML model using current season KenPom data.

Generates synthetic matchups from all team pairs, uses the formula model's
margin + noise as training signal, then trains XGBoost to learn non-linear
patterns the formula can't capture (four factors, height, experience, etc.).

Usage:
    python scripts/train_from_current.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from services import kenpom_client

_MODELS_DIR = Path(__file__).parent.parent / "models"
_D1_AVG_EFF = 100.0
_D1_AVG_TEMPO = 67.5

FEATURE_NAMES = [
    "adj_em_diff",
    "adj_o_edge_a",
    "adj_o_edge_b",
    "tempo_diff",
    "seed_diff",
    "sos_diff",
    "luck_diff",
    "off_efg_diff",
    "def_efg_diff",
    "off_to_diff",
    "off_orb_diff",
    "fg3_pct_diff",
    "ft_pct_diff",
    "experience_diff",
    "avg_hgt_diff",
    "bench_diff",
    "continuity_diff",
    "stl_rate_diff",
    "block_pct_diff",
    "ast_rate_diff",
]


def _f(team: dict, key: str, default: float = 0.0) -> float:
    val = team.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _build_features(a: dict, b: dict) -> list[float]:
    return [
        _f(a, "adj_em") - _f(b, "adj_em"),
        _f(a, "adj_o", _D1_AVG_EFF) - _f(b, "adj_d", _D1_AVG_EFF),
        _f(b, "adj_o", _D1_AVG_EFF) - _f(a, "adj_d", _D1_AVG_EFF),
        _f(a, "tempo", _D1_AVG_TEMPO) - _f(b, "tempo", _D1_AVG_TEMPO),
        _f(b, "seed", 8) - _f(a, "seed", 8),
        _f(a, "sos") - _f(b, "sos"),
        _f(a, "luck") - _f(b, "luck"),
        _f(a, "off_efg") - _f(b, "off_efg"),
        _f(a, "def_efg") - _f(b, "def_efg"),
        _f(a, "off_to") - _f(b, "off_to"),
        _f(a, "off_orb") - _f(b, "off_orb"),
        _f(a, "fg3_pct") - _f(b, "fg3_pct"),
        _f(a, "ft_pct") - _f(b, "ft_pct"),
        _f(a, "experience") - _f(b, "experience"),
        _f(a, "avg_hgt") - _f(b, "avg_hgt"),
        _f(a, "bench") - _f(b, "bench"),
        _f(a, "continuity") - _f(b, "continuity"),
        _f(a, "stl_rate") - _f(b, "stl_rate"),
        _f(a, "block_pct") - _f(b, "block_pct"),
        _f(a, "ast_rate") - _f(b, "ast_rate"),
    ]


def _formula_margin(a: dict, b: dict) -> float:
    adj_o_a = _f(a, "adj_o", _D1_AVG_EFF)
    adj_d_a = _f(a, "adj_d", _D1_AVG_EFF)
    adj_o_b = _f(b, "adj_o", _D1_AVG_EFF)
    adj_d_b = _f(b, "adj_d", _D1_AVG_EFF)
    tempo = (_f(a, "tempo", _D1_AVG_TEMPO) + _f(b, "tempo", _D1_AVG_TEMPO)) / 2

    eff_a = 100 + (adj_o_a - 100) - (adj_d_b - 100)
    eff_b = 100 + (adj_o_b - 100) - (adj_d_a - 100)

    score_a = eff_a * tempo / 100
    score_b = eff_b * tempo / 100

    seed_a = _f(a, "seed", 8)
    seed_b = _f(b, "seed", 8)

    return (score_a - score_b) + 0.12 * (seed_b - seed_a)


def main():
    print("Loading KenPom data...")
    teams = kenpom_client.get_all_team_stats()
    if not teams:
        print("ERROR: No team data. Check KENPOM_BEARER_TOKEN.")
        sys.exit(1)

    print(f"Loaded {len(teams)} teams")

    team_list = list(teams.values())
    tournament_teams = [t for t in team_list if t.get("seed")]

    print(f"Tournament teams: {len(tournament_teams)}")
    print(f"Generating training matchups...")

    random.seed(42)
    np.random.seed(42)

    rows = []

    # All tournament team pairs (both directions for symmetry)
    for i, a in enumerate(tournament_teams):
        for j, b in enumerate(tournament_teams):
            if i == j:
                continue
            margin = _formula_margin(a, b)
            noisy_margin = margin + np.random.normal(0, 8.5)
            team_a_won = 1 if noisy_margin > 0 else 0
            rows.append(_build_features(a, b) + [team_a_won])

    # Sample broader D-I matchups for generalization
    for _ in range(5000):
        a, b = random.sample(team_list, 2)
        margin = _formula_margin(a, b)
        noisy_margin = margin + np.random.normal(0, 9.0)
        team_a_won = 1 if noisy_margin > 0 else 0
        rows.append(_build_features(a, b) + [team_a_won])

    df = pd.DataFrame(rows, columns=FEATURE_NAMES + ["team_a_won"])
    print(f"Training samples: {len(df)}")
    print(f"Win rate: {df['team_a_won'].mean():.3f}")

    X = df[FEATURE_NAMES].values
    y = df["team_a_won"].values

    # --- Logistic Regression baseline ---
    from sklearn.linear_model import LogisticRegression

    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, C=1.0)),
    ])
    lr_logloss = -cross_val_score(lr_pipe, X, y, cv=5, scoring="neg_log_loss")
    lr_acc = cross_val_score(lr_pipe, X, y, cv=5, scoring="accuracy")
    print(f"\nLogistic Regression (5-fold CV):")
    print(f"  Accuracy: {lr_acc.mean():.4f}")
    print(f"  Log Loss: {lr_logloss.mean():.4f}")

    # --- XGBoost ---
    from xgboost import XGBClassifier

    xgb_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )),
    ])
    xgb_logloss = -cross_val_score(xgb_pipe, X, y, cv=5, scoring="neg_log_loss")
    xgb_acc = cross_val_score(xgb_pipe, X, y, cv=5, scoring="accuracy")
    print(f"\nXGBoost (5-fold CV):")
    print(f"  Accuracy: {xgb_acc.mean():.4f}")
    print(f"  Log Loss: {xgb_logloss.mean():.4f}")

    # --- Pick best ---
    if xgb_logloss.mean() < lr_logloss.mean():
        best_name = "XGBoost"
        best_pipe = xgb_pipe
        best_logloss = xgb_logloss.mean()
        best_acc = xgb_acc.mean()
    else:
        best_name = "LogisticRegression"
        best_pipe = lr_pipe
        best_logloss = lr_logloss.mean()
        best_acc = lr_acc.mean()

    print(f"\nBest: {best_name} (log loss: {best_logloss:.4f})")

    # --- Train on all data and save ---
    best_pipe.fit(X, y)

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _MODELS_DIR / "model.pkl"
    joblib.dump(best_pipe, model_path)
    print(f"Model saved: {model_path}")

    metadata = {
        "model_type": best_name,
        "features": FEATURE_NAMES,
        "training_samples": len(df),
        "cv_accuracy": round(best_acc, 4),
        "cv_log_loss": round(best_logloss, 4),
        "trained_at": datetime.now().isoformat(),
    }
    meta_path = _MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
