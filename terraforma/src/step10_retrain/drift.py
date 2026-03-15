"""
Distribution shift monitoring using KS tests.

Compares feature distributions between training data and current DB data
to detect when Overture adds new data providers or changes distributions.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sqlalchemy import text

from src.config import engine
from src.step4_classifier import extract_features, PROJECT_ROOT, MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Features worth monitoring for drift (continuous, not binary)
CONTINUOUS_FEATURES = [
    "confidence", "n_sources", "n_existence", "n_source_records",
    "name_len", "n_categories", "data_richness",
]

KS_THRESHOLD = 0.15  # KS statistic above this = drift
P_THRESHOLD = 0.01   # p-value below this = significant drift

CITY_MAP = {
    "sf": "san_francisco",
    "nyc": "new_york",
    "chicago": "chicago",
}


def _get_training_features():
    """Load training data features for comparison."""
    dfs = []
    p1 = PROJECT_ROOT / "project_c_samples.parquet"
    if p1.exists():
        df1 = pd.read_parquet(p1)
        dfs.append(extract_features(df1))

    p2 = PROJECT_ROOT / "samples_3k_project_c_updated.parquet"
    if p2.exists():
        df2 = pd.read_parquet(p2)
        dfs.append(extract_features(df2))

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True).fillna(0)


def _get_production_features(city: str, sample_size: int = 1000):
    """Sample current Overture features from DB for a city."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT o.*
            FROM overture.places o
            WHERE o.city = :city
            ORDER BY random()
            LIMIT :lim
        """), {"city": city, "lim": sample_size}).fetchall()

        if not rows:
            return None

        cols = conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='overture' AND table_name='places' ORDER BY ordinal_position"
        )).fetchall()
        col_names = [c[0] for c in cols]

    df = pd.DataFrame(rows, columns=col_names)
    return extract_features(df).fillna(0)


def check_drift(city_key: str):
    """Compare training vs production feature distributions."""
    if city_key not in CITY_MAP:
        log.error("Unknown city: %s", city_key)
        return

    db_city = CITY_MAP[city_key]

    log.info("=" * 60)
    log.info("DISTRIBUTION DRIFT CHECK: %s", db_city)
    log.info("=" * 60)

    train_feats = _get_training_features()
    if train_feats is None:
        log.error("No training data found")
        return

    prod_feats = _get_production_features(db_city)
    if prod_feats is None:
        log.error("No production data found for %s", db_city)
        return

    log.info("Training samples: %d | Production samples: %d", len(train_feats), len(prod_feats))
    log.info("")

    drifted = []
    for feat in CONTINUOUS_FEATURES:
        if feat not in train_feats.columns or feat not in prod_feats.columns:
            continue

        train_vals = train_feats[feat].values
        prod_vals = prod_feats[feat].values

        ks_stat, p_val = ks_2samp(train_vals, prod_vals)
        is_drift = ks_stat > KS_THRESHOLD and p_val < P_THRESHOLD

        status = "DRIFT" if is_drift else "OK"
        log.info("  %-20s KS=%.3f  p=%.4f  [%s]", feat, ks_stat, p_val, status)

        if is_drift:
            drifted.append(feat)
            log.info("    Train: mean=%.3f std=%.3f | Prod: mean=%.3f std=%.3f",
                     train_vals.mean(), train_vals.std(),
                     prod_vals.mean(), prod_vals.std())

    log.info("")
    if drifted:
        log.warning("DRIFT DETECTED in %d features: %s", len(drifted), drifted)
        log.warning("→ Recommend retraining: python -m src.step10_retrain retrain")
        return True
    else:
        log.info("No significant drift detected. Model is stable.")
        return False
