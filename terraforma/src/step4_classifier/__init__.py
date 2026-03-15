"""
Step 4: Open/closed classifier.

Features: recency/staleness, brand, zombie_score, delta, identity changes, interactions.
Model: CatBoost + LightGBM ensemble.
Training data: Overture's pre-labeled samples (project_c_samples + samples_3k_project_c_updated).

Usage:
    python -m src.step4_classifier          # train + evaluate
    python -m src.step4_classifier predict   # score all DB matched records
"""

import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score, classification_report, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "open_closed_v1.pkl"

REFERENCE_DATE = datetime(2026, 2, 18)  # Feb release date


# ── JSON / parsing helpers ───────────────────────────────────────────

def _safe_json(val):
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, np.ndarray):
        return list(val)
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return None
    return val


def _len_json(val):
    v = _safe_json(val)
    return len(v) if isinstance(v, (list, np.ndarray)) else 0


def _has_value(val):
    if val is None:
        return 0
    if isinstance(val, float) and np.isnan(val):
        return 0
    if isinstance(val, str):
        return int(val.strip() not in ("", "[]", "None", "null"))
    if isinstance(val, (list, np.ndarray)):
        return int(len(val) > 0)
    return 1


def _get_name(names_raw):
    n = _safe_json(names_raw)
    if isinstance(n, dict):
        return (n.get("primary", "") or "").lower().strip()
    if isinstance(n, str):
        return n.lower().strip()
    return ""


def _category_primary(cat_raw):
    c = _safe_json(cat_raw)
    if isinstance(c, dict):
        return (c.get("primary", "") or "").lower().strip()
    if isinstance(c, str):
        return c.lower().strip()
    return "unknown"


def _has_brand(brand_raw):
    b = _safe_json(brand_raw)
    if not b or not isinstance(b, dict):
        return 0
    names = b.get("names", {})
    if isinstance(names, dict):
        return int(bool(names.get("primary", "")))
    return 0


def _source_names(sources_raw):
    s = _safe_json(sources_raw)
    if isinstance(s, (list, np.ndarray)):
        return [str(src.get("dataset", "")).lower() for src in s if isinstance(src, dict)]
    return []


def _count_sources(sources_raw):
    s = _safe_json(sources_raw)
    if s is None or not isinstance(s, (list, np.ndarray)):
        return 0
    return len(set(src.get("dataset", "") for src in s if isinstance(src, dict)))


def _total_records(sources_raw):
    s = _safe_json(sources_raw)
    if not isinstance(s, (list, np.ndarray)):
        return 0
    return len(s)


def _recency(sources_raw, reference_date=REFERENCE_DATE):
    """Return (min_days, max_days, avg_days) from source update_times."""
    s = _safe_json(sources_raw)
    if not isinstance(s, (list, np.ndarray)) or len(s) == 0:
        return (9999, 9999, 9999)
    days = []
    for item in s:
        if isinstance(item, dict):
            ut = item.get("update_time")
            if ut:
                try:
                    dt = datetime.strptime(str(ut).split("T")[0], "%Y-%m-%d")
                    days.append((reference_date - dt).days)
                except Exception:
                    pass
    if not days:
        return (9999, 9999, 9999)
    return (min(days), max(days), sum(days) / len(days))


def _website_domain(val):
    v = _safe_json(val)
    if isinstance(v, list) and v:
        url = str(v[0]).replace("http://", "").replace("https://", "").replace("www.", "")
        return url.split("/")[0].split("?")[0].lower()
    return None


def _address_key(val):
    v = _safe_json(val)
    if isinstance(v, list) and v:
        item = v[0]
        if isinstance(item, dict):
            return f"{item.get('locality', '')}_{item.get('postcode', '')}".lower()
    return None


def _has_platform(socials_raw, platform):
    s = _safe_json(socials_raw)
    if isinstance(s, list):
        return int(any(isinstance(x, str) and platform in x.lower() for x in s))
    return 0


def _congruence(websites_raw, socials_raw):
    """1 if website domain appears in any social URL."""
    w = _safe_json(websites_raw)
    s = _safe_json(socials_raw)
    if not w or not s or not isinstance(w, list) or not isinstance(s, list):
        return 0
    domain = str(w[0]).replace("http://", "").replace("https://", "").replace("www.", "")
    domain = domain.split("/")[0].split(".")[0].lower()
    return 1 if any(isinstance(x, str) and domain in x.lower() for x in s) else 0


# ── Feature extraction ───────────────────────────────────────────────

def extract_features(df):
    """Extract all features from a DataFrame with current + base columns.

    Expected columns: confidence, sources, names, categories, websites,
    phones, emails, socials, brand, addresses (and base_ prefixed versions).
    """
    feats = pd.DataFrame(index=df.index)

    # ── 1. Confidence (base only — leak-free per StatusNow V5) ──
    if "base_confidence" in df.columns:
        feats["base_conf"] = pd.to_numeric(df["base_confidence"], errors="coerce").fillna(0)
    elif "confidence" in df.columns:
        feats["base_conf"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0)
    else:
        feats["base_conf"] = 0
    feats["base_conf_sq"] = feats["base_conf"] ** 2

    # ── 2. Sources / recency ──
    src_col = "sources" if "sources" in df.columns else None
    base_src_col = "base_sources" if "base_sources" in df.columns else src_col

    if src_col and src_col in df.columns:
        feats["num_sources"] = df[src_col].apply(_count_sources)
        feats["log_num_sources"] = np.log1p(feats["num_sources"])
        feats["is_cross_verified"] = (feats["num_sources"] > 1).astype(int)

        snames = df[src_col].apply(_source_names)
        feats["source_has_msft"] = snames.apply(
            lambda x: 1 if any(s in x for s in ("microsoft", "msft")) else 0
        )
        feats["source_has_meta"] = snames.apply(
            lambda x: 1 if "meta" in x else 0
        )

        # Recency from source update_times
        rec = df[src_col].apply(_recency)
        feats["days_latest"] = rec.apply(lambda x: x[0])
        feats["days_oldest"] = rec.apply(lambda x: x[1])
        feats["days_avg"] = rec.apply(lambda x: x[2])
    else:
        feats["num_sources"] = 0
        feats["log_num_sources"] = 0
        feats["is_cross_verified"] = 0
        feats["source_has_msft"] = 0
        feats["source_has_meta"] = 0
        feats["days_latest"] = 9999
        feats["days_oldest"] = 9999
        feats["days_avg"] = 9999

    if base_src_col and base_src_col in df.columns:
        feats["base_num_sources"] = df[base_src_col].apply(_count_sources)
        feats["delta_sources"] = feats["num_sources"] - feats["base_num_sources"]
        feats["has_lost_sources"] = (feats["delta_sources"] < 0).astype(int)
    else:
        feats["base_num_sources"] = feats["num_sources"]
        feats["delta_sources"] = 0
        feats["has_lost_sources"] = 0

    # ── 3. Digital presence ──
    for field in ["websites", "socials", "phones"]:
        cur = field if field in df.columns else None
        base = f"base_{field}" if f"base_{field}" in df.columns else cur

        if cur and cur in df.columns:
            feats[f"num_{field}"] = df[cur].apply(_len_json)
            feats[f"has_{field.rstrip('s')}"] = (feats[f"num_{field}"] > 0).astype(int)
        else:
            feats[f"num_{field}"] = 0
            feats[f"has_{field.rstrip('s')}"] = 0

        if base and base in df.columns:
            base_count = df[base].apply(_len_json)
            feats[f"delta_{field}"] = feats[f"num_{field}"] - base_count
        else:
            feats[f"delta_{field}"] = 0

    # Email count
    if "emails" in df.columns:
        feats["num_emails"] = df["emails"].apply(_len_json)
    else:
        feats["num_emails"] = 0

    feats["contact_depth"] = feats["num_websites"] + feats["num_socials"] + feats["num_emails"]

    # Social platform flags
    soc_col = "socials" if "socials" in df.columns else None
    if soc_col:
        feats["has_facebook"] = df[soc_col].apply(lambda x: _has_platform(x, "facebook.com"))
        feats["has_instagram"] = df[soc_col].apply(lambda x: _has_platform(x, "instagram.com"))
        feats["has_yelp"] = df[soc_col].apply(lambda x: _has_platform(x, "yelp.com"))
    else:
        feats["has_facebook"] = 0
        feats["has_instagram"] = 0
        feats["has_yelp"] = 0

    feats["total_digital"] = (
        feats["has_website"] + feats["has_social"] + feats["has_phone"] +
        feats["has_facebook"] + feats["has_instagram"] + feats["has_yelp"]
    )

    # ── 4. Brand ──
    brand_col = "brand" if "brand" in df.columns else None
    feats["is_brand"] = df[brand_col].apply(_has_brand) if brand_col else 0

    # ── 5. Category ──
    cat_col = "categories" if "categories" in df.columns else None
    if cat_col and cat_col in df.columns:
        feats["category_primary_str"] = df[cat_col].apply(_category_primary)
    else:
        feats["category_primary_str"] = "unknown"
    feats["cat_is_unknown"] = (feats["category_primary_str"] == "unknown").astype(int)

    # ── 6. Delta features ──
    feats["has_lost_website"] = (feats["delta_websites"] < 0).astype(int)
    feats["has_gained_website"] = (feats["delta_websites"] > 0).astype(int)
    feats["has_lost_social"] = (feats["delta_socials"] < 0).astype(int)
    feats["has_gained_social"] = (feats["delta_socials"] > 0).astype(int)
    feats["has_lost_phone"] = (feats["delta_phones"] < 0).astype(int)
    feats["has_gained_phone"] = (feats["delta_phones"] > 0).astype(int)
    feats["delta_total"] = feats["delta_websites"] + feats["delta_socials"] + feats["delta_phones"]
    feats["has_any_loss"] = (
        (feats["delta_websites"] < 0) | (feats["delta_socials"] < 0) | (feats["delta_phones"] < 0)
    ).astype(int)
    feats["has_any_gain"] = (
        (feats["delta_websites"] > 0) | (feats["delta_socials"] > 0) | (feats["delta_phones"] > 0)
    ).astype(int)
    feats["num_loss_types"] = feats["has_lost_website"] + feats["has_lost_social"] + feats["has_lost_phone"]
    feats["num_gain_types"] = feats["has_gained_website"] + feats["has_gained_social"] + feats["has_gained_phone"]
    feats["has_complete_loss"] = (
        (feats["delta_websites"] < 0) & (feats["delta_socials"] < 0)
    ).astype(int)
    feats["contact_loss_severity"] = (
        feats["has_lost_website"] * 2 +
        feats["has_lost_social"] * 1.5 +
        feats["has_lost_phone"] * 1
    )

    # ── 7. Identity changes ──
    if "names" in df.columns and "base_names" in df.columns:
        curr_name = df["names"].apply(_get_name)
        base_name = df["base_names"].apply(_get_name)
        feats["name_changed"] = ((curr_name != base_name) & (curr_name != "") & (base_name != "")).astype(int)
        feats["name_length"] = curr_name.apply(len)
        feats["name_length_delta"] = curr_name.apply(len) - base_name.apply(len)
    else:
        name_col = "names" if "names" in df.columns else None
        feats["name_changed"] = 0
        feats["name_length"] = df[name_col].apply(lambda x: len(_get_name(x))) if name_col else 0
        feats["name_length_delta"] = 0

    if "categories" in df.columns and "base_categories" in df.columns:
        curr_cat = df["categories"].apply(_category_primary)
        base_cat = df["base_categories"].apply(_category_primary)
        feats["cat_changed"] = (
            (curr_cat != base_cat) & (curr_cat != "unknown") & (base_cat != "unknown")
        ).astype(int)
    else:
        feats["cat_changed"] = 0

    if "websites" in df.columns and "base_websites" in df.columns:
        curr_dom = df["websites"].apply(_website_domain)
        base_dom = df["base_websites"].apply(_website_domain)
        feats["website_domain_changed"] = (
            (curr_dom != base_dom) & curr_dom.notna() & base_dom.notna()
        ).astype(int)
    else:
        feats["website_domain_changed"] = 0

    if "addresses" in df.columns and "base_addresses" in df.columns:
        curr_addr = df["addresses"].apply(_address_key)
        base_addr = df["base_addresses"].apply(_address_key)
        feats["address_changed"] = (
            (curr_addr != base_addr) & curr_addr.notna() & base_addr.notna()
        ).astype(int)
    else:
        feats["address_changed"] = 0

    feats["identity_change_score"] = feats["name_changed"] + feats["cat_changed"] + feats["address_changed"]

    # ── 8. Recency / staleness ──
    days = feats["days_latest"].clip(upper=9999)
    feats["log_days"] = np.log1p(days)
    feats["is_stale_3mo"] = (days > 90).astype(int)
    feats["is_stale_6mo"] = (days > 180).astype(int)
    feats["is_stale_1yr"] = (days > 365).astype(int)
    feats["is_stale_2yr"] = (days > 730).astype(int)
    feats["recency_bucket"] = pd.cut(
        days, bins=[-1, 90, 365, 730, 99999], labels=[0, 1, 2, 3]
    ).astype(int)
    feats["recency_spread"] = (feats["days_oldest"] - feats["days_latest"]).clip(lower=0)

    feats["base_conf_x_stale"] = feats["base_conf"] * feats["is_stale_1yr"]

    # ── 9. Interaction features ──
    feats["zombie_score"] = feats["num_sources"] / (feats["days_avg"] + 1)
    feats["decay_velocity"] = feats["delta_total"] / (feats["days_avg"] + 1)
    feats["recency_x_loss"] = days * feats["has_any_loss"]
    feats["recency_x_social_loss"] = days * feats["has_lost_social"]
    feats["brand_x_stale"] = feats["is_brand"] * feats["is_stale_1yr"]
    feats["nonbrand_stale_risk"] = (1 - feats["is_brand"]) * feats["is_stale_6mo"]
    feats["brand_x_name_change"] = feats["is_brand"] * feats["name_changed"]
    feats["nonbrand_x_name_change"] = (1 - feats["is_brand"]) * feats["name_changed"]
    feats["source_loss_x_stale"] = feats["has_lost_sources"] * feats["is_stale_6mo"]
    feats["stale_x_loss_x_nonbrand"] = (
        feats["is_stale_6mo"] * feats["has_any_loss"] * (1 - feats["is_brand"])
    )
    feats["loss_x_low_conf"] = feats["has_any_loss"] * (feats["base_conf"] < 0.5).astype(int)
    feats["stale_x_low_conf"] = feats["is_stale_1yr"] * (feats["base_conf"] < 0.5).astype(int)
    feats["multi_signal_risk"] = (
        feats["has_any_loss"] + feats["is_stale_1yr"] + feats["name_changed"] + feats["cat_changed"]
    )

    # Digital congruence
    if "websites" in df.columns and "socials" in df.columns:
        feats["digital_congruence"] = [
            _congruence(w, s) for w, s in zip(df["websites"], df["socials"])
        ]
    else:
        feats["digital_congruence"] = 0

    return feats


# ── Feature columns (what goes into the model) ──────────────────────

NUMERIC_FEATURES = [
    # Confidence (current snapshot)
    "base_conf", "base_conf_sq",
    # Brand & sources
    "is_brand", "num_sources", "log_num_sources",
    "source_has_msft", "source_has_meta", "is_cross_verified",
    # Digital presence
    "has_website", "has_social", "has_phone", "contact_depth",
    "has_facebook", "has_instagram", "has_yelp",
    "total_digital", "num_websites", "num_socials",
    # Category
    "cat_is_unknown",
    # Recency / staleness (from source update_times — key signal)
    "log_days", "is_stale_3mo", "is_stale_6mo",
    "is_stale_1yr", "is_stale_2yr", "recency_bucket", "recency_spread",
    # Interactions
    "digital_congruence",
    "brand_x_stale", "nonbrand_stale_risk",
    "stale_x_low_conf",
    "name_length",
    "num_emails",
]


def get_feature_cols(df):
    """Get numeric feature columns present in the DataFrame."""
    return [c for c in NUMERIC_FEATURES if c in df.columns]


# ── Training data loading ────────────────────────────────────────────

def load_training_data():
    """Load Overture's pre-labeled training data.

    Uses samples_3k_project_c_updated.parquet — 3k places with real base/current
    delta snapshots and balanced labels (1809 open, 1191 closed).

    We do NOT mix in project_c_samples.parquet (3.4k single-snapshot, 91% open)
    because it teaches the model "no deltas = open" which breaks inference.
    """
    p_updated = PROJECT_ROOT / "samples_3k_project_c_updated.parquet"
    if not p_updated.exists():
        log.error("No training data! Need samples_3k_project_c_updated.parquet")
        sys.exit(1)

    df = pd.read_parquet(p_updated)
    df["label"] = df["label"].astype(int)
    log.info("Loaded %d rows from samples_3k_project_c_updated.parquet", len(df))

    feats = extract_features(df)
    feats["label"] = df["label"].values
    feats["city"] = "overture_labeled"

    # Fill NaN
    for col in feats.columns:
        if col not in ("label", "city", "category_primary_str"):
            feats[col] = pd.to_numeric(feats[col], errors="coerce").fillna(0)

    n_open = (feats["label"] == 1).sum()
    n_closed = (feats["label"] == 0).sum()
    log.info("  Open: %d | Closed: %d", n_open, n_closed)

    return feats


# ── Training ─────────────────────────────────────────────────────────

def train():
    """Train CatBoost + LightGBM ensemble with cross-validation."""
    from src.step4_classifier.ensemble_model import EnsembleClassifier

    df = load_training_data()
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].values.astype(float)
    y = df["label"].values

    log.info("Features (%d): %s", len(feature_cols), feature_cols)
    log.info("Label distribution: %d open, %d closed", y.sum(), len(y) - y.sum())

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    log.info("Running 5-fold cross-validation with CatBoost + LightGBM ensemble...")

    # We can't use cross_val_predict with custom ensemble easily,
    # so do manual CV loop
    y_pred_proba = np.zeros(len(y))

    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = EnsembleClassifier()
        model.fit(X_train, y_train)

        y_pred_proba[val_idx] = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, y_pred_proba[val_idx])
        log.info("  Fold %d: AUC=%.4f", fold_i + 1, fold_auc)

    # Find optimal threshold
    best_thresh, best_bal_acc = 0.5, 0
    for t in np.arange(0.25, 0.75, 0.01):
        yp = (y_pred_proba >= t).astype(int)
        bal = balanced_accuracy_score(y, yp)
        if bal > best_bal_acc:
            best_bal_acc = bal
            best_thresh = t

    log.info("Optimal threshold: %.2f (balanced acc: %.1f%%)", best_thresh, 100 * best_bal_acc)

    y_pred = (y_pred_proba >= best_thresh).astype(int)
    auc = roc_auc_score(y, y_pred_proba)

    log.info("\n=== Cross-Validation Results (threshold=%.2f) ===", best_thresh)
    log.info("ROC AUC: %.4f", auc)
    log.info("Balanced Accuracy: %.1f%%", 100 * best_bal_acc)
    log.info("\n%s", classification_report(y, y_pred, target_names=["closed", "open"]))

    # Per-city breakdown
    if "city" in df.columns:
        for city in sorted(df["city"].unique()):
            mask = df["city"] == city
            if mask.sum() > 50:
                city_auc = roc_auc_score(y[mask], y_pred_proba[mask])
                city_bal = balanced_accuracy_score(y[mask], y_pred[mask])
                log.info("  %s: AUC=%.4f, BalAcc=%.1f%%", city, city_auc, 100 * city_bal)

    # Geographic hold-out evaluation (train on N-1 cities, test on held-out city)
    if "city" in df.columns:
        cities = [c for c in df["city"].unique() if (df["city"] == c).sum() > 100]
        if len(cities) >= 2:
            log.info("\n=== Geographic Hold-Out Evaluation ===")
            for holdout_city in cities:
                train_mask = df["city"] != holdout_city
                test_mask = df["city"] == holdout_city

                X_tr, y_tr = X[train_mask], y[train_mask]
                X_te, y_te = X[test_mask], y[test_mask]

                if y_te.sum() == 0 or y_te.sum() == len(y_te):
                    continue  # skip if only one class

                geo_model = EnsembleClassifier()
                geo_model.fit(X_tr, y_tr)
                geo_proba = geo_model.predict_proba(X_te)[:, 1]
                geo_pred = (geo_proba >= best_thresh).astype(int)
                geo_auc = roc_auc_score(y_te, geo_proba)
                geo_bal = balanced_accuracy_score(y_te, geo_pred)
                geo_rec_closed = recall_score(y_te, geo_pred, pos_label=0)
                geo_rec_open = recall_score(y_te, geo_pred, pos_label=1)
                log.info("  Hold-out %s: AUC=%.4f, BalAcc=%.1f%%, ClosedRec=%.1f%%, OpenRec=%.1f%%",
                         holdout_city, geo_auc, 100 * geo_bal,
                         100 * geo_rec_closed, 100 * geo_rec_open)

    # Train final model on all data
    log.info("\nTraining final model on all %d samples...", len(X))
    final_model = EnsembleClassifier()
    final_model.fit(X, y)

    # Feature importance
    log.info("\nFeature importances:")
    importances = final_model.feature_importances_
    for name, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        if imp > 0.01:
            log.info("  %-30s %.3f", name, imp)

    # Save
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": final_model,
            "feature_cols": feature_cols,
            "threshold": best_thresh,
        }, f)
    log.info("\nModel saved to %s (threshold=%.2f)", MODEL_PATH, best_thresh)

    return final_model, feature_cols


def predict_db():
    """Score all matched records in the database."""
    from src.config import engine

    if not MODEL_PATH.exists():
        log.error("No trained model found. Run training first.")
        return

    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    feature_cols = saved["feature_cols"]

    log.info("Loading matched records from DB...")
    from sqlalchemy import text as sa_text
    with engine.connect() as conn:
        df = pd.read_sql(sa_text("""
            SELECT
                m.id AS match_id,
                m.overture_id,
                m.match_score,
                m.is_open AS registry_is_open,
                r.city,
                o.confidence,
                o.name_normalized,
                o.category,
                o.raw_json
            FROM overture.matched m
            JOIN registries.businesses r ON r.id = m.registry_id
            JOIN overture.places o       ON o.id = m.overture_id
        """), conn)
    log.info("Loaded %d records", len(df))

    if df.empty:
        return

    # Extract rich features from raw_json
    def _expand_json(row):
        rj = row.get("raw_json")
        if rj is None:
            return row
        if isinstance(rj, str):
            try:
                rj = json.loads(rj)
            except (json.JSONDecodeError, TypeError):
                return row
        for key in ("sources", "names", "categories", "websites", "socials",
                     "emails", "phones", "brand", "addresses"):
            if key not in row or row[key] is None:
                row[key] = rj.get(key)
            # Also set base_ columns (same as current for single-snapshot)
            row[f"base_{key}"] = row.get(key)
        row["base_confidence"] = row.get("confidence")
        return row

    df = df.apply(_expand_json, axis=1)
    feat_df = extract_features(df)

    for col in feature_cols:
        if col not in feat_df.columns:
            feat_df[col] = 0

    X = feat_df[feature_cols].values.astype(float)
    proba = model.predict_proba(X)[:, 1]
    threshold = saved.get("threshold", 0.5)
    pred = (proba >= threshold).astype(int)

    df["pred_open_prob"] = proba
    df["pred_open"] = pred

    log.info("\nPrediction summary:")
    log.info("  Predicted open:   %d (%.1f%%)", pred.sum(), 100 * pred.mean())
    log.info("  Predicted closed: %d", len(pred) - pred.sum())

    if "registry_is_open" in df.columns:
        agree = (df["pred_open"] == df["registry_is_open"].astype(int)).mean()
        log.info("  Agreement with registry: %.1f%%", 100 * agree)

    for city in sorted(df["city"].unique()):
        mask = df["city"] == city
        c_pred = df.loc[mask, "pred_open"]
        log.info("  %s: %d total, %d predicted open (%.0f%%)",
                 city, mask.sum(), c_pred.sum(), 100 * c_pred.mean())

    return df
