"""
Train XGBoost v2: uses pipeline signals (foursquare, website, text match)
in addition to Overture metadata.

Two training modes:
1. Overture-only features from yelp_training_data.json (6000+ samples)
2. Full pipeline features from test_data files (228 samples with all signals)

We train on the full-pipeline data since those features are what actually
discriminate open vs closed.

Usage: python scripts/train_xgboost_v2.py
"""

import json
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'src', 'data')
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'model')

# All features the model will use
FEATURES = [
    # Overture metadata (original)
    'overture_confidence',
    'source_age_days',
    'has_website',
    'has_phone',
    'has_brand',
    'address_complete',
    'category_closure_rate',
    'fields_populated',
    # Engineered from metadata
    'missing_signals',
    'data_completeness',
    'low_confidence',
    # Pipeline signals (NEW)
    'fsq_verified',        # 1 if foursquare verified
    'fsq_no_data',         # 1 if no foursquare match at all
    'website_alive',       # 1 if website is alive
    'website_dead',        # 1 if website is dead/parked
    'text_match',          # 1 if any text match (full or partial)
    'text_full_match',     # 1 if full text match
    'num_images',          # number of street-level images
    'image_age_days',      # age of best image in days
]


def extract_features(location):
    """Extract all features from a fully-processed location entry."""
    raw = location.get('overture_raw', {})
    vision = location.get('vision', {})
    layers = vision.get('layers', {})
    fsq = location.get('foursquare', {})
    ws = location.get('website_check', {})
    text_layer = layers.get('text', {})

    # Base Overture features
    confidence = float(raw.get('overture_confidence', 0))
    has_website_flag = float(raw.get('has_website', 0))
    has_phone = float(raw.get('has_phone', 0))
    has_brand = float(raw.get('has_brand', 0))
    fields = float(raw.get('fields_populated', 0))

    # Engineered
    missing_signals = (1 - has_website_flag) + (1 - has_phone) + (1 - has_brand)
    data_completeness = confidence * fields / 9.0
    low_confidence = 1.0 if confidence < 0.7 else 0.0

    # Pipeline signals
    fsq_status = fsq.get('status', 'no_data')
    ws_status = ws.get('status', 'no_url')
    text_verdict = text_layer.get('verdict', 'no_images')
    gallery = location.get('current_gallery', [])

    return {
        'overture_confidence': confidence,
        'source_age_days': float(raw.get('source_age_days', 0)),
        'has_website': has_website_flag,
        'has_phone': has_phone,
        'has_brand': has_brand,
        'address_complete': float(raw.get('address_complete', 0)),
        'category_closure_rate': float(raw.get('category_closure_rate', 0)),
        'fields_populated': fields,
        'missing_signals': missing_signals,
        'data_completeness': round(data_completeness, 4),
        'low_confidence': low_confidence,
        # Pipeline
        'fsq_verified': 1.0 if fsq_status == 'verified' else 0.0,
        'fsq_no_data': 1.0 if fsq_status == 'no_data' else 0.0,
        'website_alive': 1.0 if ws_status == 'alive' else 0.0,
        'website_dead': 1.0 if ws_status in ('dead', 'parked') else 0.0,
        'text_match': 1.0 if text_verdict in ('full_match', 'partial_match') else 0.0,
        'text_full_match': 1.0 if text_verdict == 'full_match' else 0.0,
        'num_images': float(len(gallery)),
        'image_age_days': float(raw.get('image_age_days', 0)),
    }


def load_pipeline_data():
    """Load all test data files with full pipeline signals."""
    files = [
        ('SF', os.path.join(DATA_DIR, 'test_data.json')),
        ('LA', os.path.join(DATA_DIR, 'test_data_la.json')),
        ('Chicago', os.path.join(DATA_DIR, 'test_data_chicago.json')),
    ]

    X, y, names, cities = [], [], [], []
    for city, path in files:
        if not os.path.exists(path):
            print(f"  Skipping {city}: {path} not found")
            continue
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        for loc in data:
            gt = loc.get('ground_truth')
            if gt not in ('open', 'closed'):
                continue
            feats = extract_features(loc)
            row = [feats[f] for f in FEATURES]
            X.append(row)
            y.append(1 if gt == 'open' else 0)
            names.append(loc.get('name', '?'))
            cities.append(city)

    return np.array(X), np.array(y), names, cities


def main():
    try:
        import xgboost as xgb
        from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
        from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
        from sklearn.calibration import CalibratedClassifierCV
    except ImportError:
        print("Need: pip install xgboost scikit-learn")
        return

    print("Loading pipeline data from test files...")
    X, y, names, cities = load_pipeline_data()
    n_open = int(sum(y))
    n_closed = len(y) - n_open
    print(f"  {len(X)} samples: {n_open} open, {n_closed} closed")
    print(f"  Features ({len(FEATURES)}): {FEATURES}")

    if len(X) < 20:
        print("Not enough data")
        return

    weight_ratio = n_open / max(n_closed, 1)
    print(f"  Class ratio: {weight_ratio:.1f}:1, scale_pos_weight={weight_ratio:.2f}")

    # --- Leave-One-City-Out CV (most honest evaluation) ---
    print("\n=== Leave-One-City-Out Cross-Validation ===")
    city_arr = np.array(cities)
    unique_cities = sorted(set(cities))

    all_preds = np.zeros(len(y))
    all_probs = np.zeros(len(y))

    for held_out in unique_cities:
        train_mask = city_arr != held_out
        test_mask = city_arr == held_out

        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=3,
            min_child_weight=3,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            gamma=0.2,
            scale_pos_weight=weight_ratio,
            eval_metric='logloss',
            random_state=42,
        )
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict(X[test_mask])
        probs = model.predict_proba(X[test_mask])[:, 1]
        all_preds[test_mask] = preds
        all_probs[test_mask] = probs

        tp = ((preds == 1) & (y[test_mask] == 1)).sum()
        tn = ((preds == 0) & (y[test_mask] == 0)).sum()
        fp = ((preds == 1) & (y[test_mask] == 0)).sum()
        fn = ((preds == 0) & (y[test_mask] == 1)).sum()
        acc = (tp + tn) / len(preds)
        print(f"  {held_out:8s}: acc={acc:.1%}  TP={tp} TN={tn} FP={fp} FN={fn}")

    overall_acc = accuracy_score(y, all_preds)
    bal_acc = balanced_accuracy_score(y, all_preds)
    print(f"\n  Overall LOCO-CV: acc={overall_acc:.1%}  balanced_acc={bal_acc:.1%}")
    print(classification_report(y, all_preds, target_names=['Closed', 'Open']))

    # --- Stratified 5-fold CV ---
    print("=== Stratified 5-Fold CV ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_preds = np.zeros(len(y))
    cv_probs = np.zeros(len(y))

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        m = xgb.XGBClassifier(
            n_estimators=150, max_depth=3, min_child_weight=3,
            learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=2.0, gamma=0.2,
            scale_pos_weight=weight_ratio, eval_metric='logloss', random_state=42,
        )
        m.fit(X[train_idx], y[train_idx])
        cv_preds[val_idx] = m.predict(X[val_idx])
        cv_probs[val_idx] = m.predict_proba(X[val_idx])[:, 1]

    cv_acc = accuracy_score(y, cv_preds)
    cv_bal = balanced_accuracy_score(y, cv_preds)
    print(f"  5-fold CV: acc={cv_acc:.1%}  balanced_acc={cv_bal:.1%}")
    print(classification_report(y, cv_preds, target_names=['Closed', 'Open']))

    # --- Train final model on ALL data ---
    print("=== Training Final Model ===")
    base_model = xgb.XGBClassifier(
        n_estimators=150, max_depth=3, min_child_weight=3,
        learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=2.0, gamma=0.2,
        scale_pos_weight=weight_ratio, eval_metric='logloss', random_state=42,
    )

    # Platt calibration
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model, method='sigmoid', cv=5,
    )
    calibrated_model.fit(X, y)

    y_prob_cal = calibrated_model.predict_proba(X)[:, 1]
    y_pred_cal = (y_prob_cal >= 0.5).astype(int)

    open_cal = y_prob_cal[y == 1]
    closed_cal = y_prob_cal[y == 0]
    print(f"  Calibrated probs - Open:   min={open_cal.min():.3f} median={np.median(open_cal):.3f} max={open_cal.max():.3f}")
    print(f"  Calibrated probs - Closed: min={closed_cal.min():.3f} median={np.median(closed_cal):.3f} max={closed_cal.max():.3f}")

    # Optimal threshold
    best_thresh, best_bal = 0.5, 0
    for thresh in np.arange(0.3, 0.7, 0.01):
        preds_t = (y_prob_cal >= thresh).astype(int)
        ba = balanced_accuracy_score(y, preds_t)
        if ba > best_bal:
            best_bal = ba
            best_thresh = thresh
    print(f"  Optimal threshold: {best_thresh:.2f} (balanced accuracy: {best_bal:.1%})")

    # Train raw model for saving
    base_model.fit(X, y)

    # Extract Platt params
    calibrator = calibrated_model.calibrated_classifiers_[0]
    platt_cal = calibrator.calibrators[0]
    platt_a = float(platt_cal.a_)
    platt_b = float(platt_cal.b_)
    print(f"  Platt scaling: A={platt_a:.4f}, B={platt_b:.4f}")

    # Feature importance
    importance = dict(zip(FEATURES, base_model.feature_importances_))
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
    print("\n  Feature importance:")
    for feat, imp in sorted_imp:
        bar = '#' * int(imp * 50)
        print(f"    {feat:25s} {imp:.3f} {bar}")

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'xgboost_model.json')
    base_model.save_model(model_path)
    print(f"\n  Model saved to {model_path}")

    meta = {
        'version': 2,
        'training_features': FEATURES,
        'n_training_samples': len(X),
        'n_open': n_open,
        'n_closed': n_closed,
        'optimal_threshold': round(float(best_thresh), 3),
        'loco_cv_accuracy': round(float(overall_acc), 4),
        'loco_cv_balanced_accuracy': round(float(bal_acc), 4),
        'cv5_accuracy': round(float(cv_acc), 4),
        'platt_a': platt_a,
        'platt_b': platt_b,
        'feature_importance': {k: round(float(v), 4) for k, v in sorted_imp},
    }
    meta_path = os.path.join(MODEL_DIR, 'xgb_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved to {meta_path}")

    # Show misclassifications from LOCO-CV
    print("\n=== LOCO-CV Misclassifications ===")
    for i in range(len(y)):
        if all_preds[i] != y[i]:
            pred_label = 'open' if all_preds[i] == 1 else 'closed'
            true_label = 'open' if y[i] == 1 else 'closed'
            kind = 'FP' if all_preds[i] == 1 else 'FN'
            print(f"  {kind} | {cities[i]:8s} | prob={all_probs[i]:.3f} | {names[i]} (truly {true_label})")


if __name__ == '__main__':
    main()
