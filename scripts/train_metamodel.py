"""
Train a logistic regression metamodel over 6 signal scores.
Signals: Foursquare, Website, Text/OCR, XGBoost metadata, TomTom, Yelp reviews.
Evaluates per-city accuracy with Leave-One-City-Out CV.
"""
import json
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'src', 'data')

DATA_FILES = [
    ('SF', os.path.join(DATA_DIR, 'test_data.json')),
    ('LA', os.path.join(DATA_DIR, 'test_data_la.json')),
    ('Chicago', os.path.join(DATA_DIR, 'test_data_chicago.json')),
]


def encode_signals(loc):
    """Extract 4 signal scores from a fully-processed location."""
    vision = loc.get('vision', {})
    layers = vision.get('layers', {})

    # Signal 1: Foursquare (original encoding)
    fsq = loc.get('foursquare', {})
    fsq_status = fsq.get('status', 'no_data')
    if fsq_status == 'verified':
        fsq_score = 1.0
    elif fsq_status == 'mismatch':
        fsq_score = -0.5
    elif fsq_status == 'closed':
        fsq_score = -1.0
    else:  # no_data
        fsq_score = -0.3

    # Signal 2: Website
    ws = loc.get('website_check', {})
    ws_status = ws.get('status', 'no_url')
    if ws_status == 'alive':
        ws_score = 0.5
    elif ws_status == 'redirect':
        ws_score = -0.7
    elif ws_status == 'dead':
        ws_score = -0.3
    elif ws_status == 'parked':
        ws_score = -0.5
    else:  # no_url, ssl_error
        ws_score = 0.0

    # Signal 3: Text/OCR
    text_layer = layers.get('text', {})
    text_verdict = text_layer.get('verdict', 'no_images')
    closure_signals = text_layer.get('closure_signals', [])
    has_closure_text = len(closure_signals) > 0

    if has_closure_text:
        text_score = -1.0
    elif text_verdict == 'full_match':
        text_score = 1.0
    elif text_verdict == 'partial_match':
        text_score = 0.5
    elif text_verdict == 'no_match':
        text_score = -0.2
    else:  # no_images
        text_score = 0.0

    # Signal 4: XGBoost metadata
    xgb = layers.get('xgboost', {})
    xgb_score = xgb.get('score', 0.5)
    # Center around 0: positive = open, negative = closed
    xgb_centered = xgb_score - 0.5

    # Signal 5: TomTom directory (only trust strong matches)
    tt = loc.get('tomtom', {})
    tt_status = tt.get('status', 'no_data')
    tt_match = tt.get('match_score', 0.0)
    if tt_status == 'verified' and tt_match >= 0.8:
        tt_score = 1.0
    elif tt_status == 'verified' and tt_match >= 0.5:
        tt_score = 0.3
    else:  # no_data or weak match (< 0.5 is unreliable)
        tt_score = 0.0

    # Signal 6: Yelp review presence (log-scaled, centered)
    yelp = loc.get('yelp', {})
    yelp_reviews = yelp.get('yelp_review_count', 0)
    import math
    # Log scale: 0 reviews -> -1.0, 10 -> 0.0, 100 -> 0.5, 1000 -> 1.0
    if yelp_reviews > 0:
        yelp_score = min(1.0, (math.log10(yelp_reviews) - 1.0) / 2.0)
    else:
        yelp_score = -1.0

    # Interaction features
    fsq_verified_web_dead = 1.0 if (fsq_status == 'verified' and ws_status == 'dead') else 0.0
    fsq_no_data = 1.0 if fsq_status == 'no_data' else 0.0
    web_alive = 1.0 if ws_status == 'alive' else 0.0
    fsq_nodata_web_alive = fsq_no_data * web_alive
    tt_strong = tt_status == 'verified' and tt_match >= 0.8
    both_dirs_verified = 1.0 if (fsq_status == 'verified' and tt_strong) else 0.0
    both_dirs_missing = 1.0 if (fsq_status == 'no_data' and not tt_strong) else 0.0
    # Key interaction: Foursquare says open but no Yelp presence = ghost listing
    fsq_verified_no_yelp = 1.0 if (fsq_status == 'verified' and yelp_reviews == 0) else 0.0

    return [fsq_score, ws_score, text_score, xgb_centered, tt_score, yelp_score,
            fsq_verified_web_dead, fsq_no_data, fsq_nodata_web_alive,
            both_dirs_verified, both_dirs_missing, fsq_verified_no_yelp]


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

    # Load data
    X, y, names, cities = [], [], [], []
    for city, path in DATA_FILES:
        if not os.path.exists(path):
            print(f"  Skipping {city}: not found")
            continue
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        for loc in data:
            gt = loc.get('ground_truth')
            if gt not in ('open', 'closed'):
                continue
            signals = encode_signals(loc)
            X.append(signals)
            y.append(1 if gt == 'open' else 0)
            names.append(loc.get('name', '?'))
            cities.append(city)

    X = np.array(X)
    y = np.array(y)
    city_arr = np.array(cities)
    unique_cities = sorted(set(cities))

    print(f"Loaded {len(X)} samples: {int(y.sum())} open, {int((1-y).sum())} closed")
    print(f"Signals: Foursquare, Website, Text, XGBoost, TomTom, Yelp\n")

    # --- Leave-One-City-Out CV ---
    print("=== Leave-One-City-Out Cross-Validation ===")
    all_preds = np.zeros(len(y))
    all_probs = np.zeros(len(y))

    for held_out in unique_cities:
        train_mask = city_arr != held_out
        test_mask = city_arr == held_out

        model = LogisticRegression(C=1.0, max_iter=1000)
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
        print(f"  {held_out:10s}: {acc:.1%} accuracy  TP={tp} TN={tn} FP={fp} FN={fn}")

    overall_acc = accuracy_score(y, all_preds)
    bal_acc = balanced_accuracy_score(y, all_preds)
    print(f"\n  Overall LOCO-CV: {overall_acc:.1%} accuracy, {bal_acc:.1%} balanced accuracy")
    print(classification_report(y, all_preds, target_names=['Closed', 'Open']))

    # --- Train final model on all data ---
    print("=== Final Model (trained on all data) ===")
    final_model = LogisticRegression(C=1.0, max_iter=1000)
    final_model.fit(X, y)

    signal_names = ['Foursquare', 'Website', 'Text', 'XGBoost', 'TomTom', 'Yelp_reviews',
                    'FSQ_verified+Web_dead', 'FSQ_no_data', 'FSQ_nodata+Web_alive',
                    'Both_dirs_verified', 'Both_dirs_missing', 'FSQ_verified+No_yelp']
    print("  Learned weights:")
    for name, w in zip(signal_names, final_model.coef_[0]):
        print(f"    {name:25s}: {w:.4f}")
    print(f"    {'Intercept':25s}: {final_model.intercept_[0]:.4f}")

    # --- Compare: Current hand-tuned vs metamodel ---
    print("\n=== Comparison: Hand-tuned vs Metamodel ===")
    for held_out in unique_cities:
        test_mask = city_arr == held_out

        # Current system predictions
        current_correct = 0
        meta_correct = 0
        total = int(test_mask.sum())

        for i in np.where(test_mask)[0]:
            true_label = y[i]
            # Current prediction from data
            current_pred = 1 if all_probs[i] >= 0.5 else 0  # metamodel pred
            meta_correct += int(current_pred == true_label)

        print(f"  {held_out:10s}: Metamodel {meta_correct}/{total} = {meta_correct/total:.1%}")

    print(f"\n  Metamodel overall: {int(overall_acc * len(y))}/{len(y)} = {overall_acc:.1%}")

    # --- Show misclassifications ---
    print("\n=== LOCO-CV Misclassifications ===")
    for i in range(len(y)):
        if all_preds[i] != y[i]:
            pred_label = 'open' if all_preds[i] == 1 else 'closed'
            true_label = 'open' if y[i] == 1 else 'closed'
            kind = 'FP' if all_preds[i] == 1 else 'FN'
            sigs = X[i]
            print(f"  {kind} | {cities[i]:10s} | prob={all_probs[i]:.3f} | fsq={sigs[0]:+.1f} ws={sigs[1]:+.1f} txt={sigs[2]:+.1f} xgb={sigs[3]:+.2f} tt={sigs[4]:+.1f} yelp={sigs[5]:+.2f} | {names[i]} (truly {true_label})")

    # Save model weights for use in run_vision.py
    meta = {
        'weights': {name: round(float(w), 4) for name, w in zip(signal_names, final_model.coef_[0])},
        'intercept': round(float(final_model.intercept_[0]), 4),
        'loco_cv_accuracy': round(float(overall_acc), 4),
        'loco_cv_balanced_accuracy': round(float(bal_acc), 4),
    }
    meta_path = os.path.join(SCRIPT_DIR, '..', 'model', 'metamodel.json')
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metamodel weights saved to {meta_path}")


if __name__ == '__main__':
    main()
