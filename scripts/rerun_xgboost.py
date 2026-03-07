"""
Re-score all test data using the metamodel (learned logistic regression weights).
Reads existing signal layers, encodes 4 signal scores, applies metamodel, updates predictions.
"""
import json
import os
import math

DATA_FILES = [
    ('SF', os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'test_data.json')),
    ('LA', os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'test_data_la.json')),
    ('Chicago', os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'test_data_chicago.json')),
    ('NYC', os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'test_data_nyc.json')),
]

# Metamodel weights (learned from logistic regression on original encoding)
# Signal encoding improved post-training: fsq+dead→0.2, no_data→0.0
META_WEIGHTS = {
    'foursquare': 2.1922,
    'website': 1.0637,
    'text': 0.0738,
    'xgboost': 1.3851,
}
META_INTERCEPT = -0.9401


def encode_and_score(loc):
    """Encode 4 signals and compute metamodel score for a location."""
    vision = loc.get('vision', {})
    layers = vision.get('layers', {})

    # Signal 1: Foursquare
    fsq = loc.get('foursquare', {})
    fsq_status = fsq.get('status', 'no_data')

    # Signal 2: Website
    ws = loc.get('website_check', {})
    ws_status = ws.get('status', 'no_url')

    # Encode Foursquare (original encoding for metamodel)
    if fsq_status == 'verified':
        fsq_signal = 1.0
    elif fsq_status == 'closed':
        fsq_signal = -1.0
    elif fsq_status == 'mismatch':
        fsq_signal = -0.5
    else:
        fsq_signal = -0.3

    # Encode Website
    if ws_status == 'alive':
        ws_signal = 0.5
    elif ws_status == 'redirect':
        ws_signal = -0.7
    elif ws_status == 'dead':
        ws_signal = -0.3
    elif ws_status == 'parked':
        ws_signal = -0.5
    else:
        ws_signal = 0.0

    # Signal 3: Text/OCR
    text_layer = layers.get('text', {})
    text_verdict = text_layer.get('verdict', 'no_images')
    closure_signals = text_layer.get('closure_signals', [])
    fresh_closure = [s for s in closure_signals if s.get('image_age_years', 99) <= 5]
    strong_closure = any(s.get('strength') == 'strong' for s in fresh_closure)

    if strong_closure:
        text_signal = -1.0
    elif text_verdict == 'full_match':
        text_signal = 1.0
    elif text_verdict == 'partial_match':
        text_signal = 0.5
    elif text_verdict == 'no_match':
        text_signal = -0.2
    else:
        text_signal = 0.0

    # Signal 4: XGBoost metadata (centered)
    xgb = layers.get('xgboost', {})
    xgb_score = xgb.get('score', 0.5)
    xgb_centered = xgb_score - 0.5

    # Metamodel: sigmoid(w·x + b)
    logit = (
        META_WEIGHTS['foursquare'] * fsq_signal +
        META_WEIGHTS['website'] * ws_signal +
        META_WEIGHTS['text'] * text_signal +
        META_WEIGHTS['xgboost'] * xgb_centered +
        META_INTERCEPT
    )
    open_score = 1.0 / (1.0 + math.exp(-logit))

    # ── Post-metamodel correction rules ──
    # (none currently — corrections removed after analysis showed patterns are noisier than expected)

    # Build score breakdown
    breakdown = [
        {"signal": "metamodel", "weight": round(open_score, 4), "description": f"Metamodel score (logit={logit:+.2f})"},
        {"signal": "foursquare", "weight": round(fsq_signal * META_WEIGHTS['foursquare'], 4), "description": f"Foursquare: {fsq_status} (signal={fsq_signal:+.1f})"},
        {"signal": "website", "weight": round(ws_signal * META_WEIGHTS['website'], 4), "description": f"Website: {ws_status} (signal={ws_signal:+.1f})"},
        {"signal": "text", "weight": round(text_signal * META_WEIGHTS['text'], 4), "description": f"Text: {text_verdict} (signal={text_signal:+.1f})"},
        {"signal": "xgboost", "weight": round(xgb_centered * META_WEIGHTS['xgboost'], 4), "description": f"XGBoost: {xgb_score:.0%} (signal={xgb_centered:+.2f})"},
    ]

    # Prediction
    if open_score > 0.50:
        prediction = 'open'
    else:
        prediction = 'not_open'

    return open_score, prediction, breakdown


def rerun_city(city, path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    for loc in data:
        vision = loc.get('vision')
        if not vision or not vision.get('layers'):
            continue

        open_score, prediction, breakdown = encode_and_score(loc)
        vision['confidence'] = round(open_score, 4)
        vision['prediction'] = prediction
        vision['score_breakdown'] = breakdown

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=True)

    # Evaluate
    tp = tn = fp = fn = 0
    for loc in data:
        gt = loc.get('ground_truth')
        pred = loc.get('vision', {}).get('prediction')
        if not gt or not pred:
            continue
        true_open = gt == 'open'
        pred_open = pred == 'open'
        if true_open and pred_open: tp += 1
        elif not true_open and not pred_open: tn += 1
        elif not true_open and pred_open: fp += 1
        else: fn += 1

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0
    print(f"  {city:10s}: {acc:.1%} accuracy ({tp+tn}/{total})  TP={tp} TN={tn} FP={fp} FN={fn}")
    return tp, tn, fp, fn


def main():
    print("Re-scoring all cities with metamodel...\n")
    total_tp = total_tn = total_fp = total_fn = 0
    for city, path in DATA_FILES:
        if not os.path.exists(path):
            print(f"  {city}: skipped (file not found)")
            continue
        tp, tn, fp, fn = rerun_city(city, path)
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

    total = total_tp + total_tn + total_fp + total_fn
    acc = (total_tp + total_tn) / total if total else 0
    print(f"\n  OVERALL: {acc:.1%} accuracy ({total_tp+total_tn}/{total})")
    print(f"    TP={total_tp} TN={total_tn} FP={total_fp} FN={total_fn}")
    print(f"    Open precision:   {total_tp/max(total_tp+total_fp,1):.1%}")
    print(f"    Open recall:      {total_tp/max(total_tp+total_fn,1):.1%}")
    print(f"    Closed precision: {total_tn/max(total_tn+total_fn,1):.1%}")
    print(f"    Closed recall:    {total_tn/max(total_tn+total_fp,1):.1%}")


if __name__ == '__main__':
    main()
