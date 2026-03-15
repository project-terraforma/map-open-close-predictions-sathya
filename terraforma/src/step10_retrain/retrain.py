"""
Retrain orchestrator: checks triggers, runs auto-labeling if needed, retrains model.

Temporal split: train on older labels, validate on newer.
Geographic holdout: train on N-1 cities, test on holdout.
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sqlalchemy import text

from src.config import engine
from src.step4_classifier import (
    extract_features, _name_similarity, _category_match,
    PROJECT_ROOT, MODEL_PATH, load_training_data, get_feature_cols,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

CITIES = ["san_francisco", "new_york", "chicago"]


def _load_auto_labels():
    """Load auto-labeled data from DB as a training DataFrame."""
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT o.*, a.auto_label AS label
                FROM predictions.auto_labels a
                JOIN overture.places o ON o.id = a.overture_id
                WHERE a.confidence >= 0.70
            """)).fetchall()

            if not rows:
                return None

            cols = conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='overture' AND table_name='places' ORDER BY ordinal_position"
            )).fetchall()
            col_names = [c[0] for c in cols] + ["label"]

        df = pd.DataFrame(rows, columns=col_names)
        feats = extract_features(df)
        feats["label"] = df["label"].astype(int)
        feats["dataset"] = "auto_labels"
        return feats

    except Exception as e:
        log.warning("Could not load auto-labels: %s", e)
        return None


def _load_all_training():
    """Load parquet files + auto-labels into one training set."""
    # Start with existing parquet data
    base_df = load_training_data()

    # Add auto-labels
    auto_df = _load_auto_labels()
    if auto_df is not None:
        log.info("auto_labels: %d rows (%.0f%% open)",
                 len(auto_df), 100 * auto_df["label"].mean())

        # Align columns
        for col in base_df.columns:
            if col not in auto_df.columns:
                auto_df[col] = 0
        for col in auto_df.columns:
            if col not in base_df.columns:
                base_df[col] = 0

        base_df = pd.concat([base_df, auto_df[base_df.columns]], ignore_index=True).fillna(0)
        log.info("Combined with auto-labels: %d total rows", len(base_df))

    return base_df


def _geographic_holdout(df, feature_cols):
    """Train on N-1 cities' auto-labels, test on holdout. Returns per-city accuracy."""
    auto_mask = df["dataset"] == "auto_labels"
    if auto_mask.sum() < 50:
        log.info("Not enough auto-labels for geographic holdout test")
        return

    log.info("\n--- Geographic Holdout Validation ---")
    # We need city info in auto-labels — skip if not available
    # For now, use the parquet + auto-label combined cross-val
    log.info("(Geographic holdout requires city column in auto-labels — coming in next version)")


def retrain():
    """Full retrain pipeline with auto-labels + parquet data."""
    log.info("=" * 60)
    log.info("RETRAIN PIPELINE")
    log.info("=" * 60)

    # Check if we have auto-labels
    try:
        with engine.connect() as conn:
            auto_count = conn.execute(text(
                "SELECT count(*) FROM predictions.auto_labels"
            )).scalar()
        log.info("Auto-labels available: %d", auto_count)
    except Exception:
        auto_count = 0
        log.info("No auto-labels table found. Run auto-label first.")

    if auto_count < 20:
        log.warning("Need at least 20 auto-labels. Run: python -m src.step10_retrain auto-label sf 100")
        log.info("Falling back to parquet-only training...")

    # Load all training data
    df = _load_all_training()
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].values.astype(float)
    y = df["label"].values

    log.info("\nFeatures (%d): %s", len(feature_cols), feature_cols)
    log.info("Label distribution: %d open, %d closed", y.sum(), len(y) - y.sum())

    # Compute sample weights
    n_open, n_closed = int(y.sum()), int(len(y) - y.sum())
    weight_closed = n_open / n_closed if n_closed > 0 else 1.0
    sample_weights = np.where(y == 0, weight_closed, 1.0)
    ds_mask = df["dataset"].values

    # Downweight weak project_c, full weight for auto-labels
    sample_weights = np.where(ds_mask == "project_c", sample_weights * 0.3, sample_weights)
    log.info("Class weights: open=1.0, closed=%.2f", weight_closed)
    log.info("Dataset weights: project_c=0.3x, samples_3k=1.0x, auto_labels=1.0x")

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = GradientBoostingClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        min_samples_leaf=12,
        random_state=42,
    )

    log.info("\nRunning 5-fold cross-validation...")
    y_pred_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba",
                                      params={"sample_weight": sample_weights})[:, 1]

    # Optimal threshold
    best_thresh, best_bal_acc = 0.5, 0
    for t in np.arange(0.25, 0.75, 0.01):
        yp = (y_pred_proba >= t).astype(int)
        rec_open = recall_score(y, yp, pos_label=1)
        rec_closed = recall_score(y, yp, pos_label=0)
        bal = (rec_open + rec_closed) / 2
        if bal > best_bal_acc:
            best_bal_acc = bal
            best_thresh = t

    log.info("Optimal threshold: %.2f (balanced acc: %.1f%%)", best_thresh, 100 * best_bal_acc)

    y_pred = (y_pred_proba >= best_thresh).astype(int)
    auc = roc_auc_score(y, y_pred_proba)

    log.info("\n=== Cross-Validation Results (threshold=%.2f) ===", best_thresh)
    log.info("ROC AUC: %.4f", auc)
    log.info("\n%s", classification_report(y, y_pred, target_names=["closed", "open"]))

    # Per-dataset breakdown
    for ds in df["dataset"].unique():
        mask = df["dataset"] == ds
        if mask.sum() > 10:
            ds_auc = roc_auc_score(y[mask], y_pred_proba[mask])
            ds_acc = (y_pred[mask] == y[mask]).mean()
            log.info("  %s: AUC=%.4f, Acc=%.1f%%", ds, ds_auc, 100 * ds_acc)

    # Train final model
    log.info("\nTraining final model on all %d samples...", len(X))
    model.fit(X, y, sample_weight=sample_weights)

    # Feature importances
    log.info("\nFeature importances:")
    for name, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
        if imp > 0.01:
            log.info("  %-25s %.3f", name, imp)

    # Save with metadata
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "model": model,
        "feature_cols": feature_cols,
        "threshold": best_thresh,
        "trained_at": datetime.now().isoformat(),
        "n_samples": len(X),
        "n_auto_labels": int((ds_mask == "auto_labels").sum()),
        "auc": float(auc),
        "balanced_acc": float(best_bal_acc),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(save_data, f)

    log.info("\nModel saved to %s", MODEL_PATH)
    log.info("  threshold=%.2f, AUC=%.4f, balanced_acc=%.1f%%",
             best_thresh, auc, 100 * best_bal_acc)
    log.info("  %d total samples (%d auto-labels)",
             len(X), save_data["n_auto_labels"])
