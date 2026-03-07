"""
Re-score all test data using the metamodel (learned logistic regression weights).
Reads existing signal layers, encodes 6 signal scores, applies metamodel, updates predictions.
"""
import json
import os
import math

DATA_FILES = [
    ('SF', os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'test_data.json')),
    ('LA', os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'test_data_la.json')),
    ('Chicago', os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'test_data_chicago.json')),
    ('Miami', os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'test_data_miami.json')),
]

# Metamodel weights (learned from logistic regression with 6 signals + interactions)
META_WEIGHTS = {
    'foursquare': 1.8997,
    'website': 1.3448,
    'text': 0.1617,
    'xgboost': 0.6998,
    'tomtom': -0.0668,
    'yelp_reviews': 1.7241,
    'fsq_verified_web_dead': -0.9368,
    'fsq_no_data': -0.7736,
    'fsq_nodata_web_alive': -1.0269,
    'both_dirs_verified': 0.5090,
    'both_dirs_missing': -0.0729,
    'fsq_verified_no_yelp': -0.1488,
}
META_INTERCEPT = 0.6184


def encode_and_score(loc):
    """Encode 6 signals and compute metamodel score for a location."""
    vision = loc.get('vision', {})
    layers = vision.get('layers', {})

    # Signal 1: Foursquare
    fsq = loc.get('foursquare', {})
    fsq_status = fsq.get('status', 'no_data')

    # Signal 2: Website
    ws = loc.get('website_check', {})
    ws_status = ws.get('status', 'no_url')

    # Encode Foursquare
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

    # Signal 5: TomTom directory
    tt = loc.get('tomtom', {})
    tt_status = tt.get('status', 'no_data')
    tt_match = tt.get('match_score', 0.0)
    if tt_status == 'verified' and tt_match >= 0.8:
        tt_signal = 1.0
    elif tt_status == 'verified' and tt_match >= 0.5:
        tt_signal = 0.3
    else:
        tt_signal = 0.0

    # Signal 6: Yelp review count (log-scaled)
    yelp = loc.get('yelp', {})
    yelp_reviews = yelp.get('yelp_review_count', 0)
    if yelp_reviews > 0:
        yelp_signal = min(1.0, (math.log10(yelp_reviews) - 1.0) / 2.0)
    else:
        yelp_signal = -1.0

    # Interaction features
    fsq_verified_web_dead = 1.0 if (fsq_status == 'verified' and ws_status == 'dead') else 0.0
    fsq_no_data = 1.0 if fsq_status == 'no_data' else 0.0
    web_alive = 1.0 if ws_status == 'alive' else 0.0
    fsq_nodata_web_alive = fsq_no_data * web_alive
    tt_strong = tt_status == 'verified' and tt_match >= 0.8
    both_dirs_verified = 1.0 if (fsq_status == 'verified' and tt_strong) else 0.0
    both_dirs_missing = 1.0 if (fsq_status == 'no_data' and not tt_strong) else 0.0
    fsq_verified_no_yelp = 1.0 if (fsq_status == 'verified' and yelp_reviews == 0) else 0.0

    # Metamodel: sigmoid(w·x + b)
    logit = (
        META_WEIGHTS['foursquare'] * fsq_signal +
        META_WEIGHTS['website'] * ws_signal +
        META_WEIGHTS['text'] * text_signal +
        META_WEIGHTS['xgboost'] * xgb_centered +
        META_WEIGHTS['tomtom'] * tt_signal +
        META_WEIGHTS['yelp_reviews'] * yelp_signal +
        META_WEIGHTS['fsq_verified_web_dead'] * fsq_verified_web_dead +
        META_WEIGHTS['fsq_no_data'] * fsq_no_data +
        META_WEIGHTS['fsq_nodata_web_alive'] * fsq_nodata_web_alive +
        META_WEIGHTS['both_dirs_verified'] * both_dirs_verified +
        META_WEIGHTS['both_dirs_missing'] * both_dirs_missing +
        META_WEIGHTS['fsq_verified_no_yelp'] * fsq_verified_no_yelp +
        META_INTERCEPT
    )
    open_score = 1.0 / (1.0 + math.exp(-logit))

    # Build score breakdown
    breakdown = [
        {"signal": "metamodel", "weight": round(open_score, 4), "description": f"Metamodel score (logit={logit:+.2f})"},
        {"signal": "foursquare", "weight": round(fsq_signal * META_WEIGHTS['foursquare'], 4), "description": f"Foursquare: {fsq_status} (signal={fsq_signal:+.1f})"},
        {"signal": "website", "weight": round(ws_signal * META_WEIGHTS['website'], 4), "description": f"Website: {ws_status} (signal={ws_signal:+.1f})"},
        {"signal": "text", "weight": round(text_signal * META_WEIGHTS['text'], 4), "description": f"Text: {text_verdict} (signal={text_signal:+.1f})"},
        {"signal": "xgboost", "weight": round(xgb_centered * META_WEIGHTS['xgboost'], 4), "description": f"XGBoost: {xgb_score:.0%} (signal={xgb_centered:+.2f})"},
        {"signal": "tomtom", "weight": round(tt_signal * META_WEIGHTS['tomtom'], 4), "description": f"TomTom: {tt_status} match={tt_match:.1f} (signal={tt_signal:+.1f})"},
        {"signal": "yelp", "weight": round(yelp_signal * META_WEIGHTS['yelp_reviews'], 4), "description": f"Yelp: {yelp_reviews} reviews (signal={yelp_signal:+.2f})"},
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
