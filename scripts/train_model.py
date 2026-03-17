"""Train ML model on historical tournament matchup data with hyperparameter tuning.

Usage:
    python scripts/train_model.py

Reads data/training_data.csv, evaluates multiple model configurations, and
exports the best one to models/model.pkl.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

_DATA_DIR = Path(__file__).parent.parent / "data"
_MODELS_DIR = Path(__file__).parent.parent / "models"
_TRAINING_DATA = _DATA_DIR / "training_data.csv"

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    if not _TRAINING_DATA.exists():
        print(f"ERROR: Training data not found at {_TRAINING_DATA}")
        print("Run scripts/build_training_data.py first.")
        sys.exit(1)

    df = pd.read_csv(_TRAINING_DATA)
    feature_cols = [c for c in df.columns if c not in ("team_a_won", "season")]
    print(f"Loaded {len(df)} training samples, {len(feature_cols)} features")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Win rate: {df['team_a_won'].mean():.3f}")

    X = df[feature_cols].values
    y = df["team_a_won"].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []

    # --- Logistic Regression variants ---
    for C in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, C=C, solver="lbfgs")),
        ])
        acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        ll = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_log_loss")
        results.append({
            "name": f"LR(C={C})",
            "pipe": pipe,
            "acc": acc.mean(),
            "acc_std": acc.std(),
            "ll": ll.mean(),
            "ll_std": ll.std(),
        })

    # --- Gradient Boosting variants ---
    for n_est, depth, lr in [(100, 3, 0.1), (200, 4, 0.05), (300, 3, 0.05), (500, 4, 0.03)]:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("gb", GradientBoostingClassifier(
                n_estimators=n_est, max_depth=depth, learning_rate=lr,
                subsample=0.8, random_state=42,
            )),
        ])
        acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        ll = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_log_loss")
        results.append({
            "name": f"GB(n={n_est},d={depth},lr={lr})",
            "pipe": pipe,
            "acc": acc.mean(),
            "acc_std": acc.std(),
            "ll": ll.mean(),
            "ll_std": ll.std(),
        })

    # --- XGBoost variants ---
    try:
        from xgboost import XGBClassifier
        for n_est, depth, lr, sub, col in [
            (200, 4, 0.05, 0.8, 0.8),
            (300, 3, 0.05, 0.8, 0.7),
            (500, 4, 0.03, 0.7, 0.7),
            (300, 5, 0.05, 0.8, 0.8),
        ]:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("xgb", XGBClassifier(
                    n_estimators=n_est, max_depth=depth, learning_rate=lr,
                    subsample=sub, colsample_bytree=col,
                    use_label_encoder=False, eval_metric="logloss",
                    verbosity=0, random_state=42,
                )),
            ])
            acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
            ll = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_log_loss")
            results.append({
                "name": f"XGB(n={n_est},d={depth},lr={lr})",
                "pipe": pipe,
                "acc": acc.mean(),
                "acc_std": acc.std(),
                "ll": ll.mean(),
                "ll_std": ll.std(),
            })
    except ImportError:
        print("XGBoost not installed, skipping.")

    # --- Random Forest ---
    for n_est, depth in [(200, 6), (500, 8), (300, None)]:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=n_est, max_depth=depth, random_state=42,
            )),
        ])
        acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        ll = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_log_loss")
        results.append({
            "name": f"RF(n={n_est},d={depth})",
            "pipe": pipe,
            "acc": acc.mean(),
            "acc_std": acc.std(),
            "ll": ll.mean(),
            "ll_std": ll.std(),
        })

    # --- Print leaderboard ---
    results.sort(key=lambda r: r["ll"])
    print(f"\n{'Model':<35} {'Accuracy':>10} {'Log Loss':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<35} {r['acc']:.4f}±{r['acc_std']:.4f} {r['ll']:.4f}±{r['ll_std']:.4f}")

    best = results[0]
    print(f"\nBest: {best['name']} (log loss: {best['ll']:.4f}, accuracy: {best['acc']:.4f})")

    # --- Leave-one-season-out validation ---
    print(f"\n{'='*60}")
    print("Leave-One-Season-Out Validation (best model):")
    print(f"{'='*60}")
    seasons = sorted(df["season"].unique())
    loso_accs = []
    for held_out in seasons:
        train_mask = df["season"] != held_out
        test_mask = df["season"] == held_out
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]
        clone = type(best["pipe"])([
            (name, type(step)(**step.get_params()))
            for name, step in best["pipe"].steps
        ])
        clone.fit(X_tr, y_tr)
        acc = clone.score(X_te, y_te)
        loso_accs.append(acc)
        print(f"  {held_out}: {acc:.4f} ({test_mask.sum()} games)")
    print(f"  Average: {np.mean(loso_accs):.4f}")

    # --- Train on all data and save ---
    best["pipe"].fit(X, y)

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _MODELS_DIR / "model.pkl"
    joblib.dump(best["pipe"], model_path)
    print(f"\nModel saved: {model_path}")

    metadata = {
        "model_type": best["name"],
        "features": feature_cols,
        "training_samples": len(df),
        "cv_accuracy": round(best["acc"], 4),
        "cv_log_loss": round(best["ll"], 4),
        "loso_accuracy": round(np.mean(loso_accs), 4),
        "trained_at": datetime.now().isoformat(),
        "seasons": sorted(df["season"].unique().tolist()),
    }
    meta_path = _MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
