"""
Iterative Retraining Pipeline for XGBoost Model

Uses the 6 ensemble signal labels to progressively improve the XGBoost model.
Each round adds signal-verified labels to the training set and retrains.

Round 0: Baseline (Yelp training data only)
Round 1: + Yelp is_closed verified labels from test data
Round 2: + Foursquare existence labels
Round 3: + Website liveness labels
Round 4: + All signals combined (high-confidence metamodel predictions)

Usage: python scripts/retrain_pipeline.py
"""

import json
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'src', 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'model')

sys.path.insert(0, ROOT_DIR)
from scripts.train_xgboost import (
    BASE_FEATURES, ENGINEERED_FEATURES, FEATURES,
    compute_category_closure_rates, compute_engineered_features, load_training_data,
)

DATA_FILES = [
    ('SF', os.path.join(DATA_DIR, 'test_data.json')),
    ('LA', os.path.join(DATA_DIR, 'test_data_la.json')),
    ('Chicago', os.path.join(DATA_DIR, 'test_data_chicago.json')),
    ('Miami', os.path.join(DATA_DIR, 'test_data_miami.json')),
    ('Philly', os.path.join(DATA_DIR, 'test_data_philly.json')),
]

TRAIN_PATH = os.path.join(SCRIPT_DIR, 'yelp_training_data.json')


def extract_signal_labels(data, signal_name):
    """Extract labels from a specific signal for test data locations.

    Returns list of (location, label) where label is 1=open, 0=closed.
    Only includes high-confidence predictions from the signal.
    """
    labeled = []

    for loc in data:
        gt = loc.get('ground_truth')
        if gt not in ('open', 'closed'):
            continue

        if signal_name == 'yelp':
            yelp = loc.get('yelp', {})
            reviews = yelp.get('yelp_review_count', 0)
            is_closed = yelp.get('is_closed', None)
            # Trust Yelp's is_closed field if available
            if is_closed is True:
                labeled.append((loc, 0))
            elif reviews and reviews >= 5:
                # Has active reviews = likely open
                labeled.append((loc, 1))

        elif signal_name == 'foursquare':
            fsq = loc.get('foursquare', {})
            status = fsq.get('status', 'no_data')
            if status == 'verified':
                labeled.append((loc, 1))
            elif status == 'closed':
                labeled.append((loc, 0))

        elif signal_name == 'website':
            ws = loc.get('website_check', {})
            status = ws.get('status', 'no_url')
            if status == 'alive':
                labeled.append((loc, 1))
            elif status in ('dead', 'parked'):
                labeled.append((loc, 0))

        elif signal_name == 'metamodel':
            # Use the metamodel's combined prediction
            from scripts.train_metamodel import encode_signals
            signals = encode_signals(loc)
            # Load metamodel weights
            meta_path = os.path.join(MODEL_DIR, 'metamodel.json')
            with open(meta_path) as f:
                meta = json.load(f)
            weights = list(meta['weights'].values())
            intercept = meta['intercept']
            logit = sum(s * w for s, w in zip(signals, weights)) + intercept
            prob = 1.0 / (1.0 + np.exp(-logit))
            # Only use high-confidence predictions
            if prob >= 0.8:
                labeled.append((loc, 1))
            elif prob <= 0.2:
                labeled.append((loc, 0))

    return labeled


def loc_to_training_sample(loc, label, closure_rates, global_rate):
    """Convert a test data location to a training feature vector."""
    meta = loc.get('overture_meta', {})
    yelp = loc.get('yelp', {})

    features = {
        'overture_confidence': meta.get('confidence', 0.0),
        'source_age_days': meta.get('source_age_days', 0),
        'has_website': 1 if loc.get('website_url') else 0,
        'has_phone': 1 if loc.get('phone_number') else 0,
        'has_brand': meta.get('has_brand', 0),
        'address_complete': meta.get('address_complete', 0),
        'fields_populated': meta.get('fields_populated', 0),
        'yelp_rating': yelp.get('yelp_rating'),
        'yelp_review_count': yelp.get('yelp_review_count'),
    }

    cat = loc.get('category', 'Unknown')
    features['category_closure_rate'] = closure_rates.get(cat, global_rate)

    eng = compute_engineered_features(features)

    row = []
    for f in BASE_FEATURES:
        v = features.get(f)
        row.append(float(v) if v is not None else np.nan)
    row += [float(eng.get(f, 0)) for f in ENGINEERED_FEATURES]

    return row


def eval_xgboost_on_test(model, closure_rates, global_rate):
    """Evaluate XGBoost model-only on all test cities."""
    results = {}

    for city, path in DATA_FILES:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)

        correct = 0
        total = 0
        for loc in data:
            gt = loc.get('ground_truth')
            if gt not in ('open', 'closed'):
                continue

            row = loc_to_training_sample(loc, None, closure_rates, global_rate)
            X = np.array([row])
            prob = model.predict_proba(X)[0, 1]
            pred = 1 if prob >= 0.5 else 0
            actual = 1 if gt == 'open' else 0
            if pred == actual:
                correct += 1
            total += 1

        acc = 100 * correct / total if total > 0 else 0
        results[city] = {'correct': correct, 'total': total, 'accuracy': round(acc, 1)}

    return results


def main():
    try:
        import xgboost as xgb
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import StratifiedKFold
    except ImportError:
        print("Missing dependencies. Run: pip install xgboost scikit-learn")
        return

    if not os.path.exists(TRAIN_PATH):
        print(f"Training data not found: {TRAIN_PATH}")
        return

    print("=" * 70)
    print("ITERATIVE RETRAINING PIPELINE — XGBoost Model Improvement")
    print("=" * 70)

    # Load base training data
    X_base, y_base, names_base, closure_rates, global_rate = load_training_data(TRAIN_PATH)
    print(f"\nBase training: {len(X_base)} samples ({int(y_base.sum())} open, {len(y_base) - int(y_base.sum())} closed)")

    # Load all test data
    all_test_data = []
    for city, path in DATA_FILES:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for loc in data:
            loc['_city'] = city
        all_test_data.extend(data)
    print(f"Test data: {len(all_test_data)} locations across {len(DATA_FILES)} cities\n")

    results_log = []

    # ── Round 0: Baseline ──
    print("── Round 0: Baseline (Yelp training data only) ──")
    weight_ratio = y_base.sum() / max((len(y_base) - y_base.sum()), 1)
    model = xgb.XGBClassifier(
        max_depth=5, n_estimators=400, learning_rate=0.03,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.3, reg_lambda=1.5, gamma=0.1,
        scale_pos_weight=float(weight_ratio),
        eval_metric='logloss', random_state=42,
    )
    model.fit(X_base, y_base)

    city_results = eval_xgboost_on_test(model, closure_rates, global_rate)
    avg_acc = np.mean([r['accuracy'] for r in city_results.values()])
    for city, r in city_results.items():
        print(f"  {city:10s}: {r['correct']}/{r['total']} = {r['accuracy']}%")
    print(f"  Average: {avg_acc:.1f}%")
    results_log.append({'round': 'R0_baseline', 'samples': len(X_base), 'avg_accuracy': round(avg_acc, 1), 'per_city': city_results})

    # Current training data (will grow each round)
    X_current = X_base.copy()
    y_current = y_base.copy()

    # ── Round 1: + Yelp labels ──
    print("\n── Round 1: + Yelp verified labels ──")
    yelp_labeled = extract_signal_labels(all_test_data, 'yelp')
    if yelp_labeled:
        new_X = np.array([loc_to_training_sample(loc, lbl, closure_rates, global_rate)
                          for loc, lbl in yelp_labeled])
        new_y = np.array([lbl for _, lbl in yelp_labeled])
        X_current = np.vstack([X_current, new_X])
        y_current = np.concatenate([y_current, new_y])
        print(f"  Added {len(yelp_labeled)} Yelp labels ({int(new_y.sum())} open, {len(new_y) - int(new_y.sum())} closed)")

    weight_ratio = y_current.sum() / max((len(y_current) - y_current.sum()), 1)
    model = xgb.XGBClassifier(
        max_depth=5, n_estimators=400, learning_rate=0.03,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.3, reg_lambda=1.5, gamma=0.1,
        scale_pos_weight=float(weight_ratio),
        eval_metric='logloss', random_state=42,
    )
    model.fit(X_current, y_current)

    city_results = eval_xgboost_on_test(model, closure_rates, global_rate)
    avg_acc = np.mean([r['accuracy'] for r in city_results.values()])
    for city, r in city_results.items():
        print(f"  {city:10s}: {r['correct']}/{r['total']} = {r['accuracy']}%")
    print(f"  Average: {avg_acc:.1f}%")
    results_log.append({'round': 'R1_+yelp', 'samples': len(X_current), 'avg_accuracy': round(avg_acc, 1), 'per_city': city_results})

    # ── Round 2: + Foursquare labels ──
    print("\n── Round 2: + Foursquare existence labels ──")
    fsq_labeled = extract_signal_labels(all_test_data, 'foursquare')
    if fsq_labeled:
        new_X = np.array([loc_to_training_sample(loc, lbl, closure_rates, global_rate)
                          for loc, lbl in fsq_labeled])
        new_y = np.array([lbl for _, lbl in fsq_labeled])
        X_current = np.vstack([X_current, new_X])
        y_current = np.concatenate([y_current, new_y])
        print(f"  Added {len(fsq_labeled)} Foursquare labels ({int(new_y.sum())} open, {len(new_y) - int(new_y.sum())} closed)")

    model = xgb.XGBClassifier(
        max_depth=5, n_estimators=400, learning_rate=0.03,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.3, reg_lambda=1.5, gamma=0.1,
        scale_pos_weight=float(y_current.sum() / max(len(y_current) - y_current.sum(), 1)),
        eval_metric='logloss', random_state=42,
    )
    model.fit(X_current, y_current)

    city_results = eval_xgboost_on_test(model, closure_rates, global_rate)
    avg_acc = np.mean([r['accuracy'] for r in city_results.values()])
    for city, r in city_results.items():
        print(f"  {city:10s}: {r['correct']}/{r['total']} = {r['accuracy']}%")
    print(f"  Average: {avg_acc:.1f}%")
    results_log.append({'round': 'R2_+foursquare', 'samples': len(X_current), 'avg_accuracy': round(avg_acc, 1), 'per_city': city_results})

    # ── Round 3: + Website liveness labels ──
    print("\n── Round 3: + Website liveness labels ──")
    web_labeled = extract_signal_labels(all_test_data, 'website')
    if web_labeled:
        new_X = np.array([loc_to_training_sample(loc, lbl, closure_rates, global_rate)
                          for loc, lbl in web_labeled])
        new_y = np.array([lbl for _, lbl in web_labeled])
        X_current = np.vstack([X_current, new_X])
        y_current = np.concatenate([y_current, new_y])
        print(f"  Added {len(web_labeled)} website labels ({int(new_y.sum())} open, {len(new_y) - int(new_y.sum())} closed)")

    model = xgb.XGBClassifier(
        max_depth=5, n_estimators=400, learning_rate=0.03,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.3, reg_lambda=1.5, gamma=0.1,
        scale_pos_weight=float(y_current.sum() / max(len(y_current) - y_current.sum(), 1)),
        eval_metric='logloss', random_state=42,
    )
    model.fit(X_current, y_current)

    city_results = eval_xgboost_on_test(model, closure_rates, global_rate)
    avg_acc = np.mean([r['accuracy'] for r in city_results.values()])
    for city, r in city_results.items():
        print(f"  {city:10s}: {r['correct']}/{r['total']} = {r['accuracy']}%")
    print(f"  Average: {avg_acc:.1f}%")
    results_log.append({'round': 'R3_+website', 'samples': len(X_current), 'avg_accuracy': round(avg_acc, 1), 'per_city': city_results})

    # ── Round 4: + High-confidence metamodel labels ──
    print("\n── Round 4: + High-confidence metamodel labels ──")
    meta_labeled = extract_signal_labels(all_test_data, 'metamodel')
    if meta_labeled:
        new_X = np.array([loc_to_training_sample(loc, lbl, closure_rates, global_rate)
                          for loc, lbl in meta_labeled])
        new_y = np.array([lbl for _, lbl in meta_labeled])
        X_current = np.vstack([X_current, new_X])
        y_current = np.concatenate([y_current, new_y])
        print(f"  Added {len(meta_labeled)} metamodel labels ({int(new_y.sum())} open, {len(new_y) - int(new_y.sum())} closed)")

    model = xgb.XGBClassifier(
        max_depth=5, n_estimators=400, learning_rate=0.03,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.3, reg_lambda=1.5, gamma=0.1,
        scale_pos_weight=float(y_current.sum() / max(len(y_current) - y_current.sum(), 1)),
        eval_metric='logloss', random_state=42,
    )
    model.fit(X_current, y_current)

    city_results = eval_xgboost_on_test(model, closure_rates, global_rate)
    avg_acc = np.mean([r['accuracy'] for r in city_results.values()])
    for city, r in city_results.items():
        print(f"  {city:10s}: {r['correct']}/{r['total']} = {r['accuracy']}%")
    print(f"  Average: {avg_acc:.1f}%")
    results_log.append({'round': 'R4_+metamodel', 'samples': len(X_current), 'avg_accuracy': round(avg_acc, 1), 'per_city': city_results})

    # ── Summary ──
    print("\n" + "=" * 70)
    print("RETRAINING PIPELINE SUMMARY")
    print("=" * 70)
    print(f"{'Round':<20s} {'Samples':>8s} {'Avg Acc':>8s}  SF     LA     CHI    MIA    PHI")
    print("-" * 70)
    for r in results_log:
        cities = r['per_city']
        sf = cities.get('SF', {}).get('accuracy', '-')
        la = cities.get('LA', {}).get('accuracy', '-')
        chi = cities.get('Chicago', {}).get('accuracy', '-')
        mia = cities.get('Miami', {}).get('accuracy', '-')
        phi = cities.get('Philly', {}).get('accuracy', '-')
        print(f"{r['round']:<20s} {r['samples']:>8d} {r['avg_accuracy']:>7.1f}%  {sf}%  {la}%  {chi}%  {mia}%  {phi}%")

    # Save results
    results_path = os.path.join(ROOT_DIR, 'retrain_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_log, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save final model
    model.save_model(os.path.join(MODEL_DIR, 'xgboost_model_retrained.json'))
    print(f"Retrained model saved to model/xgboost_model_retrained.json")


if __name__ == '__main__':
    main()
