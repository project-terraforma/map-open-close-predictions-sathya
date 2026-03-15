"""Show status of auto-labels, model age, and label freshness."""

import logging
import pickle
from datetime import datetime
from pathlib import Path

from sqlalchemy import text
from src.config import engine
from src.step4_classifier import MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def show_status():
    log.info("=" * 60)
    log.info("RETRAIN PIPELINE STATUS")
    log.info("=" * 60)

    # Model info
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            saved = pickle.load(f)
        trained_at = saved.get("trained_at", "unknown")
        n_samples = saved.get("n_samples", "?")
        auc = saved.get("auc", "?")
        log.info("\nModel: %s", MODEL_PATH.name)
        log.info("  Trained: %s", trained_at)
        log.info("  Samples: %s", n_samples)
        log.info("  AUC: %s", auc)
    else:
        log.info("\nNo model found.")

    # Auto-labels
    try:
        with engine.connect() as conn:
            stats = conn.execute(text("""
                SELECT city, auto_label, count(*),
                       min(labeled_at), max(labeled_at),
                       count(*) FILTER (WHERE ttl_expires_at < NOW()) as expired
                FROM predictions.auto_labels
                GROUP BY city, auto_label
                ORDER BY city, auto_label
            """)).fetchall()

        if stats:
            log.info("\nAuto-labels:")
            for city, label, cnt, oldest, newest, expired in stats:
                lbl = "OPEN" if label else "CLOSED"
                log.info("  %s %s: %d (oldest: %s, expired: %d)",
                         city, lbl, cnt, oldest.strftime("%Y-%m-%d") if oldest else "?", expired)
        else:
            log.info("\nNo auto-labels yet.")
    except Exception:
        log.info("\nNo auto-labels table yet.")

    # Ground truth
    try:
        with engine.connect() as conn:
            gt = conn.execute(text("""
                SELECT city, is_open, count(*)
                FROM ground_truth.labels
                GROUP BY city, is_open
                ORDER BY city, is_open
            """)).fetchall()

        if gt:
            log.info("\nGround truth:")
            for city, is_open, cnt in gt:
                lbl = "OPEN" if is_open else "CLOSED"
                log.info("  %s %s: %d", city, lbl, cnt)
    except Exception:
        pass

    log.info("")
