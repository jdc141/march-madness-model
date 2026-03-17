"""Train ML model on historical tournament matchup data.

Usage:
    python scripts/train_model.py

Reads data/training_data.csv (built by build_training_data.py), trains
logistic regression and XGBoost classifiers, evaluates both, and exports
the best model to models/model.pkl.
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
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

_DATA_DIR = Path(__file__).parent.parent / "data"
_MODELS_DIR = Path(__file__).parent.parent / "models"
_TRAINING_DATA = _DATA_DIR / "training_data.csv"

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.features import ML_FEATURE_NAMES


def main():
    if not _TRAINING_DATA.exists():
        print(f"ERROR: Training data not found at {_TRAINING_DATA}")
        print("Run scripts/build_training_data.py first.")
        sys.exit(1)

    df = pd.read_csv(_TRAINING_DATA)
    print(f"Loaded {len(df)} training samples")

    X = df[ML_FEATURE_NAMES].values
    y = df["team_a_won"].values

    # --- Logistic Regression ---
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, C=1.0)),
    ])

    lr_acc = cross_val_score(lr_pipe, X, y, cv=5, scoring="accuracy")
    lr_logloss = -cross_val_score(lr_pipe, X, y, cv=5, scoring="neg_log_loss")
    lr_brier = -cross_val_score(lr_pipe, X, y, cv=5, scoring="neg_brier_score")

    print(f"\nLogistic Regression (5-fold CV):")
    print(f"  Accuracy:  {lr_acc.mean():.4f} ± {lr_acc.std():.4f}")
    print(f"  Log Loss:  {lr_logloss.mean():.4f} ± {lr_logloss.std():.4f}")
    print(f"  Brier:     {lr_brier.mean():.4f} ± {lr_brier.std():.4f}")

    # --- XGBoost ---
    xgb_pipe = None
    xgb_logloss_mean = float("inf")
    try:
        from xgboost import XGBClassifier

        xgb_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )),
        ])

        xgb_acc = cross_val_score(xgb_pipe, X, y, cv=5, scoring="accuracy")
        xgb_logloss = -cross_val_score(xgb_pipe, X, y, cv=5, scoring="neg_log_loss")
        xgb_brier = -cross_val_score(xgb_pipe, X, y, cv=5, scoring="neg_brier_score")
        xgb_logloss_mean = xgb_logloss.mean()

        print(f"\nXGBoost (5-fold CV):")
        print(f"  Accuracy:  {xgb_acc.mean():.4f} ± {xgb_acc.std():.4f}")
        print(f"  Log Loss:  {xgb_logloss.mean():.4f} ± {xgb_logloss.std():.4f}")
        print(f"  Brier:     {xgb_brier.mean():.4f} ± {xgb_brier.std():.4f}")

    except ImportError:
        print("\nXGBoost not installed, using Logistic Regression only.")

    # --- Select best model ---
    if xgb_pipe is not None and xgb_logloss_mean < lr_logloss.mean():
        best_name = "XGBoost"
        best_pipe = xgb_pipe
        best_logloss = xgb_logloss_mean
        best_acc = xgb_acc.mean()
    else:
        best_name = "LogisticRegression"
        best_pipe = lr_pipe
        best_logloss = lr_logloss.mean()
        best_acc = lr_acc.mean()

    print(f"\nBest model: {best_name} (log loss: {best_logloss:.4f})")

    # --- Train on full data and export ---
    best_pipe.fit(X, y)

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _MODELS_DIR / "model.pkl"
    joblib.dump(best_pipe, model_path)
    print(f"Model saved: {model_path}")

    metadata = {
        "model_type": best_name,
        "features": ML_FEATURE_NAMES,
        "training_samples": len(df),
        "cv_accuracy": round(best_acc, 4),
        "cv_log_loss": round(best_logloss, 4),
        "trained_at": datetime.now().isoformat(),
        "seasons": sorted(df["season"].unique().tolist()),
    }
    meta_path = _MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {meta_path}")


if __name__ == "__main__":
    main()
