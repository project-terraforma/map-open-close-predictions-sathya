"""
Train XGBoost model on Yelp-labeled Overture metadata to predict open/closed.

Key improvements:
- category_closure_rate: empirical closure rate per category (replaces useless hash)
- Platt scaling (sigmoid calibration) so outputs are meaningful probabilities
- scale_pos_weight for class imbalance

Training data: scripts/yelp_training_data.json (6000+ labeled businesses)
Test data: src/data/test_data.json (50 businesses — never trained on)

Usage: python scripts/train_xgboost.py
Output: model/xgboost_model.json, model/xgb_meta.json
"""

import json
import os
import numpy as np

# Base features extracted from Overture metadata
BASE_FEATURES = [
    'overture_confidence',
    'source_age_days',
    'has_website',
    'has_phone',
    'has_brand',
    'address_complete',
    'category_closure_rate',   # empirical closure rate for this category
    'fields_populated',
]

# Engineered features computed from base features
ENGINEERED_FEATURES = [
    'missing_signals',      # count of missing website + phone + brand (0-3)
    'data_completeness',    # confidence * fields_populated / 9 (composite quality score)
    'low_confidence',       # 1 if confidence < 0.7 (binary flag for weak data)
]

FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

TRAIN_PATH = os.path.join(os.path.dirname(__file__), 'yelp_training_data.json')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.json')


def compute_category_closure_rates(data):
    """Compute per-category closure rates from training data."""
    from collections import defaultdict
    counts = defaultdict(lambda: {'open': 0, 'closed': 0})
    for loc in data:
        gt = loc.get('ground_truth')
        cat = loc.get('category', 'Unknown')
        if gt in ('open', 'closed'):
            counts[cat][gt] += 1

    rates = {}
    total_closed = sum(c['closed'] for c in counts.values())
    total = sum(c['open'] + c['closed'] for c in counts.values())
    global_rate = total_closed / max(total, 1)

    for cat, c in counts.items():
        n = c['open'] + c['closed']
        if n >= 10:
            # Use actual rate for categories with enough samples
            rates[cat] = c['closed'] / n
        else:
            # Smooth toward global rate for small categories
            rates[cat] = (c['closed'] + global_rate * 5) / (n + 5)

    return rates, global_rate


def compute_engineered_features(features):
    """Compute engineered features from base features."""
    missing_signals = (
        (1 - features.get('has_website', 0)) +
        (1 - features.get('has_phone', 0)) +
        (1 - features.get('has_brand', 0))
    )
    confidence = features.get('overture_confidence', 0)
    fields = features.get('fields_populated', 0)
    data_completeness = confidence * fields / 9.0
    low_confidence = 1 if confidence < 0.7 else 0

    return {
        'missing_signals': missing_signals,
        'data_completeness': round(data_completeness, 4),
        'low_confidence': low_confidence,
    }


def load_training_data(path):
    """Load Yelp-labeled training data with category closure rates."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    # Compute category closure rates from the data itself
    closure_rates, global_rate = compute_category_closure_rates(data)

    X, y, names = [], [], []
    for loc in data:
        features = loc.get('features', {})
        gt = loc.get('ground_truth')
        if gt is None:
            continue

        # Replace category_encoded with category_closure_rate
        cat = loc.get('category', 'Unknown')
        features['category_closure_rate'] = closure_rates.get(cat, global_rate)

        # Compute engineered features
        eng = compute_engineered_features(features)

        # Build feature vector: base + engineered
        row = [float(features.get(f, 0)) for f in BASE_FEATURES]
        row += [float(eng.get(f, 0)) for f in ENGINEERED_FEATURES]

        label = 1 if gt == 'open' else 0
        X.append(row)
        y.append(label)
        names.append(loc.get('name', '?'))

    return np.array(X), np.array(y), names, closure_rates, global_rate


def main():
    try:
        import xgboost as xgb
    except ImportError:
        print("xgboost not installed. Run: pip install xgboost")
        return

    try:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
        from sklearn.calibration import CalibratedClassifierCV
    except ImportError:
        print("scikit-learn not installed. Run: pip install scikit-learn")
        return

    if not os.path.exists(TRAIN_PATH):
        print(f"Training data not found at {TRAIN_PATH}")
        return

    print("Loading training data...")
    X, y, names, closure_rates, global_rate = load_training_data(TRAIN_PATH)
    n_open = int(sum(y))
    n_closed = len(y) - n_open
    print(f"  {len(X)} samples: {n_open} open, {n_closed} closed")
    print(f"  Features ({len(FEATURES)}): {FEATURES}")
    print(f"  Class ratio: {n_open/max(n_closed,1):.1f}:1 (open:closed)")

    if len(X) < 20:
        print(f"\nERROR: Only {len(X)} samples.")
        return

    # Show category closure rates
    sorted_rates = sorted(closure_rates.items(), key=lambda x: -x[1])
    print(f"\n  Category closure rates (top 10):")
    for cat, rate in sorted_rates[:10]:
        print(f"    {cat:30s} {rate:.2%}")

    weight_ratio = n_open / max(n_closed, 1)
    print(f"\n  Using scale_pos_weight={weight_ratio:.2f}")

    # XGBoost config
    base_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        min_child_weight=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        gamma=0.2,
        scale_pos_weight=weight_ratio,
        eval_metric='logloss',
        random_state=42,
    )

    # Cross-validation on raw model
    n_splits = min(5, max(3, len(X) // 20))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    closed_recalls = []
    cv_probs_all = np.zeros(len(y))

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        m = xgb.XGBClassifier(**base_model.get_params())
        m.fit(X[train_idx], y[train_idx])
        preds = m.predict(X[val_idx])
        probs = m.predict_proba(X[val_idx])[:, 1]
        cv_probs_all[val_idx] = probs
        scores.append(float(accuracy_score(y[val_idx], preds)))
        val_closed_mask = y[val_idx] == 0
        if val_closed_mask.sum() > 0:
            closed_correct = ((preds == 0) & val_closed_mask).sum()
            closed_recalls.append(float(closed_correct / val_closed_mask.sum()))

    scores = np.array(scores)
    print(f"\n  {n_splits}-fold CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    print(f"  Per-fold: {[f'{s:.3f}' for s in scores]}")
    if closed_recalls:
        print(f"  Closed recall (avg): {np.mean(closed_recalls):.3f}")

    # Show raw probability distribution from CV
    open_probs = cv_probs_all[y == 1]
    closed_probs = cv_probs_all[y == 0]
    print(f"\n  Raw CV probabilities:")
    print(f"    Open:   min={open_probs.min():.3f}  median={np.median(open_probs):.3f}  max={open_probs.max():.3f}")
    print(f"    Closed: min={closed_probs.min():.3f}  median={np.median(closed_probs):.3f}  max={closed_probs.max():.3f}")

    # Train final model with Platt scaling (sigmoid calibration)
    print(f"\n  Training calibrated model (Platt scaling)...")
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method='sigmoid',
        cv=5,
    )
    calibrated_model.fit(X, y)

    # Get calibrated predictions
    y_prob_cal = calibrated_model.predict_proba(X)[:, 1]
    y_pred_cal = (y_prob_cal >= 0.5).astype(int)

    # Show calibrated probability distribution
    open_cal = y_prob_cal[y == 1]
    closed_cal = y_prob_cal[y == 0]
    print(f"\n  Calibrated probabilities:")
    print(f"    Open:   min={open_cal.min():.3f}  median={np.median(open_cal):.3f}  max={open_cal.max():.3f}")
    print(f"    Closed: min={closed_cal.min():.3f}  median={np.median(closed_cal):.3f}  max={closed_cal.max():.3f}")
    print(f"\n  Calibrated accuracy: {accuracy_score(y, y_pred_cal):.3f}")
    print(classification_report(y, y_pred_cal, target_names=['Closed', 'Open']))

    # Find optimal threshold
    best_thresh = 0.5
    best_bal_acc = 0
    for thresh in np.arange(0.3, 0.7, 0.01):
        preds_t = (y_prob_cal >= thresh).astype(int)
        bal_acc = balanced_accuracy_score(y, preds_t)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_thresh = thresh
    print(f"  Optimal threshold: {best_thresh:.2f} (balanced accuracy: {best_bal_acc:.3f})")

    y_pred_opt = (y_prob_cal >= best_thresh).astype(int)
    print(f"  At optimal threshold:")
    print(classification_report(y, y_pred_opt, target_names=['Closed', 'Open']))

    # Train raw model for saving (calibration params saved separately)
    base_model.fit(X, y)

    # Extract Platt scaling parameters from the calibrated model
    # CalibratedClassifierCV wraps with sigmoid: P(y=1|f) = 1 / (1 + exp(A*f + B))
    calibrator = calibrated_model.calibrated_classifiers_[0]
    # Get the calibrator for class 1
    platt_calibrator = calibrator.calibrators[0]
    platt_a = float(platt_calibrator.a_)
    platt_b = float(platt_calibrator.b_)
    print(f"\n  Platt scaling params: A={platt_a:.4f}, B={platt_b:.4f}")

    # Feature importance (from raw model)
    importance = dict(zip(FEATURES, base_model.feature_importances_))
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
    print("\n  Feature importance:")
    for feat, imp in sorted_imp:
        bar = '#' * int(imp * 50)
        print(f"    {feat:25s} {imp:.3f} {bar}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    base_model.save_model(MODEL_PATH)
    print(f"\n  Model saved to {MODEL_PATH}")

    # Save feature importance
    imp_path = os.path.join(MODEL_DIR, 'xgb_feature_importance.json')
    with open(imp_path, 'w') as f:
        json.dump([(k, float(v)) for k, v in sorted_imp], f, indent=2)

    # Save metadata including Platt params and closure rates
    meta_path = os.path.join(MODEL_DIR, 'xgb_meta.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'training_features': FEATURES,
            'base_features': BASE_FEATURES,
            'engineered_features': ENGINEERED_FEATURES,
            'n_training_samples': len(X),
            'n_open': n_open,
            'n_closed': n_closed,
            'optimal_threshold': round(float(best_thresh), 3),
            'cv_accuracy': round(float(scores.mean()), 4),
            'closed_recall': round(float(np.mean(closed_recalls)), 4) if closed_recalls else 0,
            'platt_a': platt_a,
            'platt_b': platt_b,
            'category_closure_rates': {k: round(v, 4) for k, v in closure_rates.items()},
            'global_closure_rate': round(global_rate, 4),
        }, f, indent=2)
    print(f"  Metadata saved to {meta_path}")


if __name__ == '__main__':
    main()
