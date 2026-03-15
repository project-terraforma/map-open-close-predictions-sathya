"""
Step 5: Train and cross-validate an open/closed classifier.

Uses LightGBM with stratified k-fold CV. Stores per-business predicted
probabilities in the cv_scores schema.

Usage:
    python -m src.step5_cv             # train on all matched cities
    python -m src.step5_cv sf paris    # train on specific cities
"""

import logging
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import text

from src.config import engine
from src.step4_features import extract_features, FEATURE_COLS, CATEGORICAL_COLS, TARGET_COL

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

N_FOLDS = 5


def train_and_score(cities: list[str] | None = None):
    """Train a LightGBM classifier with stratified CV and return scores."""

    df = extract_features(cities)
    if df.empty:
        return None

    # Encode categoricals
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    feature_cols = FEATURE_COLS + [c + "_enc" for c in CATEGORICAL_COLS]
    X = df[feature_cols].values
    y = df[TARGET_COL].astype(int).values

    log.info("Training LightGBM classifier: %d samples, %d features, %d-fold CV",
             len(X), len(feature_cols), N_FOLDS)

    # Out-of-fold predictions
    oof_probs = np.zeros(len(X))
    oof_preds = np.zeros(len(X))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            verbose=-1,
        )

        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)

        oof_probs[val_idx] = probs
        oof_preds[val_idx] = preds

        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds)
        auc = roc_auc_score(y_val, probs)
        fold_metrics.append({"fold": fold + 1, "accuracy": acc, "f1": f1, "auc": auc})
        log.info("  Fold %d: acc=%.3f  f1=%.3f  auc=%.3f", fold + 1, acc, f1, auc)

    # Overall metrics
    overall_acc = accuracy_score(y, oof_preds)
    overall_f1 = f1_score(y, oof_preds)
    overall_auc = roc_auc_score(y, oof_probs)
    log.info("Overall CV: acc=%.3f  f1=%.3f  auc=%.3f", overall_acc, overall_f1, overall_auc)

    # Feature importance (from last fold model)
    log.info("Feature importance:")
    importances = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1])
    for feat, imp in importances:
        log.info("  %s: %d", feat, imp)

    # Store CV scores in database
    df["cv_prob_open"] = oof_probs
    _store_scores(df)

    return df, fold_metrics


def _store_scores(df: pd.DataFrame):
    """Write CV predicted probabilities to predictions.results."""
    log.info("Storing %d CV scores in database…", len(df))

    with engine.begin() as conn:
        # Clear existing predictions for these businesses
        overture_ids = df["overture_id"].tolist()
        conn.execute(text("DELETE FROM predictions.results WHERE overture_id = ANY(:ids)"),
                     {"ids": overture_ids})

        # Insert new scores
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO predictions.results
                    (overture_id, city, registry_pred_score, actual_open)
                VALUES (:ov_id, :city, :score, :actual)
            """), {
                "ov_id": row["overture_id"],
                "city": row["city"],
                "score": float(row["cv_prob_open"]),
                "actual": bool(row["is_open"]),
            })

    log.info("Done storing CV scores")


def run(cities: list[str] | None = None):
    result = train_and_score(cities)
    if result is None:
        log.error("No data to train on")


if __name__ == "__main__":
    args = sys.argv[1:] or None
    run(args)
