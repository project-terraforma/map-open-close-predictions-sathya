"""
Train XGBoost model on Yelp-labeled Overture metadata to predict open/closed.

Uses 8 base features + 3 engineered features for better separation.
No SMOTE — uses scale_pos_weight for class imbalance instead.

Training data: scripts/yelp_training_data.json (6000+ labeled businesses)
Test data: src/data/test_data.json (50 businesses — never trained on)

Usage: python scripts/train_xgboost.py
Output: model/xgboost_model.json
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
    'category_encoded',
    'fields_populated',
]

# Engineered features computed from base features
ENGINEERED_FEATURES = [
    'missing_signals',      # count of missing website + phone + brand (0-3)
    'data_completeness',    # confidence * fields_populated / 9 (composite quality score)
    'low_confidence',       # 1 if confidence < 0.7 (binary flag for weak data)
]

FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

# Features available at inference (includes image-based features)
ALL_FEATURES = FEATURES + ['ocr_text_match', 'image_age_days', 'num_images']

TRAIN_PATH = os.path.join(os.path.dirname(__file__), 'yelp_training_data.json')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.json')


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
    """Load Yelp-labeled training data."""
    with open(path) as f:
        data = json.load(f)

    X, y, names = [], [], []
    for loc in data:
        features = loc.get('features', {})
        gt = loc.get('ground_truth')
        if gt is None:
            continue

        # Compute engineered features
        eng = compute_engineered_features(features)

        # Build feature vector: base + engineered
        row = [float(features.get(f, 0)) for f in BASE_FEATURES]
        row += [float(eng.get(f, 0)) for f in ENGINEERED_FEATURES]

        label = 1 if gt == 'open' else 0
        X.append(row)
        y.append(label)
        names.append(loc.get('name', '?'))

    return np.array(X), np.array(y), names


def main():
    try:
        import xgboost as xgb
    except ImportError:
        print("xgboost not installed. Run: pip install xgboost")
        return

    try:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import classification_report, accuracy_score
    except ImportError:
        print("scikit-learn not installed. Run: pip install scikit-learn")
        return

    if not os.path.exists(TRAIN_PATH):
        print(f"Training data not found at {TRAIN_PATH}")
        return

    print("Loading training data...")
    X, y, names = load_training_data(TRAIN_PATH)
    n_open = int(sum(y))
    n_closed = len(y) - n_open
    print(f"  {len(X)} samples: {n_open} open, {n_closed} closed")
    print(f"  Features ({len(FEATURES)}): {FEATURES}")
    print(f"  Class ratio: {n_open/max(n_closed,1):.1f}:1 (open:closed)")

    if len(X) < 20:
        print(f"\nERROR: Only {len(X)} samples.")
        return

    # Use scale_pos_weight instead of SMOTE for cleaner training signal
    weight_ratio = n_open / max(n_closed, 1)
    print(f"\n  Using scale_pos_weight={weight_ratio:.2f} (no SMOTE)")

    # XGBoost config — tuned for weak signal detection
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,            # shallower trees to avoid overfitting noise
        min_child_weight=5,     # more conservative splits
        learning_rate=0.05,     # slower learning for better generalization
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,          # more L1 regularization
        reg_lambda=2.0,         # more L2 regularization
        gamma=0.2,
        scale_pos_weight=weight_ratio,  # handle class imbalance natively
        eval_metric='logloss',
        random_state=42,
    )

    # Cross-validation
    n_splits = min(5, max(3, len(X) // 20))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    closed_recalls = []
    for train_idx, val_idx in cv.split(X, y):
        m = xgb.XGBClassifier(**model.get_params())
        m.fit(X[train_idx], y[train_idx])
        preds = m.predict(X[val_idx])
        scores.append(float(accuracy_score(y[val_idx], preds)))
        # Track closed recall specifically
        val_closed_mask = y[val_idx] == 0
        if val_closed_mask.sum() > 0:
            closed_correct = ((preds == 0) & val_closed_mask).sum()
            closed_recall = closed_correct / val_closed_mask.sum()
            closed_recalls.append(float(closed_recall))

    scores = np.array(scores)
    print(f"\n  {n_splits}-fold CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    print(f"  Per-fold: {[f'{s:.3f}' for s in scores]}")
    if closed_recalls:
        print(f"  Closed recall (avg): {np.mean(closed_recalls):.3f}")

    # Train final model on all data
    model.fit(X, y)

    # Training predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    print(f"\n  Training accuracy: {accuracy_score(y, y_pred):.3f}")
    print(f"\n  Classification report:")
    print(classification_report(y, y_pred, target_names=['Closed', 'Open']))

    # Find optimal threshold for balanced accuracy
    from sklearn.metrics import balanced_accuracy_score
    best_thresh = 0.5
    best_bal_acc = 0
    for thresh in np.arange(0.3, 0.7, 0.01):
        preds_t = (y_prob >= thresh).astype(int)
        bal_acc = balanced_accuracy_score(y, preds_t)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_thresh = thresh
    print(f"\n  Optimal threshold: {best_thresh:.2f} (balanced accuracy: {best_bal_acc:.3f})")

    # Show predictions at optimal threshold
    y_pred_opt = (y_prob >= best_thresh).astype(int)
    print(f"  At optimal threshold:")
    print(classification_report(y, y_pred_opt, target_names=['Closed', 'Open']))

    # Feature importance
    importance = dict(zip(FEATURES, model.feature_importances_))
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
    print("  Feature importance:")
    for feat, imp in sorted_imp:
        bar = '#' * int(imp * 50)
        print(f"    {feat:25s} {imp:.3f} {bar}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"\n  Model saved to {MODEL_PATH}")

    # Save feature importance
    imp_path = os.path.join(MODEL_DIR, 'xgb_feature_importance.json')
    with open(imp_path, 'w') as f:
        json.dump([(k, float(v)) for k, v in sorted_imp], f, indent=2)

    # Save metadata
    meta_path = os.path.join(MODEL_DIR, 'xgb_meta.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'training_features': FEATURES,
            'base_features': BASE_FEATURES,
            'engineered_features': ENGINEERED_FEATURES,
            'all_features': ALL_FEATURES,
            'n_training_samples': len(X),
            'n_open': n_open,
            'n_closed': n_closed,
            'optimal_threshold': round(float(best_thresh), 3),
            'cv_accuracy': round(float(scores.mean()), 4),
            'closed_recall': round(float(np.mean(closed_recalls)), 4) if closed_recalls else 0,
        }, f, indent=2)
    print(f"  Model metadata saved to {meta_path}")


if __name__ == '__main__':
    main()
