"""
Grid search over boost, boost_review_min, threshold, and new rules to maximize accuracy.
"""
import json
import math
import os

DATA_FILES = [
    ('SF', os.path.join('src', 'data', 'test_data.json')),
    ('LA', os.path.join('src', 'data', 'test_data_la.json')),
    ('Chicago', os.path.join('src', 'data', 'test_data_chicago.json')),
    ('Miami', os.path.join('src', 'data', 'test_data_miami.json')),
    ('Philly', os.path.join('src', 'data', 'test_data_philly.json')),
]

META_WEIGHTS = {
    'foursquare': 1.9779, 'website': 0.9863, 'text': 0.1037,
    'xgboost': 1.7269, 'tomtom': 0.0806, 'yelp_reviews': 0.7189,
    'fsq_verified_web_dead': -1.0508, 'fsq_no_data': -1.3316,
    'fsq_nodata_web_alive': -0.4890, 'both_dirs_verified': 0.7253,
    'both_dirs_missing': -0.0643, 'fsq_verified_no_yelp': -0.9879,
}
META_INTERCEPT = 0.2604

# Load all data once
all_locs = []
for city, path in DATA_FILES:
    with open(path) as f:
        data = json.load(f)
    for loc in data:
        if loc.get('ground_truth') and loc.get('vision', {}).get('layers'):
            all_locs.append((city, loc))


def encode_signals(loc):
    vision = loc.get('vision', {})
    layers = vision.get('layers', {})
    fsq = loc.get('foursquare', {})
    fsq_status = fsq.get('status', 'no_data')
    ws = loc.get('website_check', {})
    ws_status = ws.get('status', 'no_url')

    if fsq_status == 'verified': fsq_signal = 1.0
    elif fsq_status == 'closed': fsq_signal = -1.0
    elif fsq_status == 'mismatch': fsq_signal = -0.5
    else: fsq_signal = -0.3

    if ws_status == 'alive': ws_signal = 0.5
    elif ws_status == 'redirect': ws_signal = -0.7
    elif ws_status == 'dead': ws_signal = -0.3
    elif ws_status == 'parked': ws_signal = -0.5
    else: ws_signal = 0.0

    text_layer = layers.get('text', {})
    text_verdict = text_layer.get('verdict', 'no_images')
    closure_signals = text_layer.get('closure_signals', [])
    fresh_closure = [s for s in closure_signals if s.get('image_age_years', 99) <= 5]
    strong_closure = any(s.get('strength') == 'strong' for s in fresh_closure)
    if strong_closure: text_signal = -1.0
    elif text_verdict == 'full_match': text_signal = 1.0
    elif text_verdict == 'partial_match': text_signal = 0.5
    elif text_verdict == 'no_match': text_signal = -0.2
    else: text_signal = 0.0

    xgb = layers.get('xgboost', {})
    xgb_score = xgb.get('score', 0.5)
    xgb_centered = xgb_score - 0.5

    tt = loc.get('tomtom', {})
    tt_status = tt.get('status', 'no_data')
    tt_match = tt.get('match_score', 0.0)
    if tt_status == 'verified' and tt_match >= 0.8: tt_signal = 1.0
    elif tt_status == 'verified' and tt_match >= 0.5: tt_signal = 0.3
    else: tt_signal = 0.0

    yelp = loc.get('yelp', {})
    yelp_reviews = yelp.get('yelp_review_count', 0)
    if yelp_reviews > 0:
        yelp_signal = min(1.0, (math.log10(yelp_reviews) - 1.0) / 2.0)
    else:
        yelp_signal = -1.0

    fsq_verified_web_dead = 1.0 if (fsq_status == 'verified' and ws_status == 'dead') else 0.0
    fsq_no_data = 1.0 if fsq_status == 'no_data' else 0.0
    web_alive = 1.0 if ws_status == 'alive' else 0.0
    fsq_nodata_web_alive = fsq_no_data * web_alive
    tt_strong = tt_status == 'verified' and tt_match >= 0.8
    both_dirs_verified = 1.0 if (fsq_status == 'verified' and tt_strong) else 0.0
    both_dirs_missing = 1.0 if (fsq_status == 'no_data' and not tt_strong) else 0.0
    fsq_verified_no_yelp = 1.0 if (fsq_status == 'verified' and yelp_reviews == 0) else 0.0

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

    return {
        'logit': logit,
        'fsq_status': fsq_status, 'ws_status': ws_status,
        'yelp_closed': yelp.get('is_closed'), 'yelp_reviews': yelp_reviews,
        'tt_status': tt_status, 'tt_match': tt_match,
        'xgb_score': xgb_score, 'text_signal': text_signal,
    }


def evaluate(boost, boost_min_reviews, threshold,
             rescue_fsq_verified_dead=False,
             rescue_xgb_high=False, xgb_rescue_thresh=0.75,
             boost_mismatch_alive=0.0,
             yelp_open_boost=0.0, yelp_open_min_reviews=50):
    tp = tn = fp = fn = 0
    city_stats = {}

    for city, loc in all_locs:
        gt = loc['ground_truth']
        s = encode_signals(loc)
        logit = s['logit']

        # Soft boost: FSQ missing + Yelp open + reviews + corroboration
        yelp_not_closed = s['yelp_closed'] == False
        has_corroboration = s['ws_status'] == 'alive' or (s['tt_status'] == 'verified' and s['tt_match'] >= 0.5)
        if s['fsq_status'] == 'no_data' and yelp_not_closed and s['yelp_reviews'] >= boost_min_reviews and has_corroboration:
            logit += boost

        # Rule: Rescue fsq=verified + web=dead (reduce penalty when no yelp)
        if rescue_fsq_verified_dead and s['fsq_status'] == 'verified' and s['ws_status'] == 'dead':
            if s['yelp_closed'] is None and s['yelp_reviews'] == 0:
                logit += 0.8  # Partially undo the -1.05 and -0.99 penalties

        # Rule: High XGBoost + at least one positive signal
        if rescue_xgb_high and s['xgb_score'] >= xgb_rescue_thresh:
            if s['fsq_status'] == 'verified' or s['ws_status'] == 'alive':
                logit += 0.3

        # Rule: fsq=mismatch + web=alive boost
        if boost_mismatch_alive > 0 and s['fsq_status'] == 'mismatch' and s['ws_status'] == 'alive':
            logit += boost_mismatch_alive

        # Rule: Yelp explicitly open with decent reviews
        if yelp_open_boost > 0 and s['yelp_closed'] == False and s['yelp_reviews'] >= yelp_open_min_reviews:
            logit += yelp_open_boost

        open_score = 1.0 / (1.0 + math.exp(-logit))

        # Yelp closed override
        if s['yelp_closed'] == True:
            open_score = min(open_score, 0.15)

        pred = 'open' if open_score > threshold else 'not_open'
        true_open = gt == 'open'
        pred_open = pred == 'open'

        if true_open and pred_open: tp += 1
        elif not true_open and not pred_open: tn += 1
        elif not true_open and pred_open: fp += 1
        else: fn += 1

        if city not in city_stats:
            city_stats[city] = [0, 0, 0, 0]
        idx = 0 if (true_open and pred_open) else (1 if (not true_open and not pred_open) else (2 if (not true_open and pred_open) else 3))
        city_stats[city][idx] += 1

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0
    return acc, tp, tn, fp, fn, city_stats


# Grid search
best_acc = 0
best_config = None
results = []

for boost in [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    for boost_min_reviews in [5, 10, 20, 30, 50, 100]:
        for threshold in [0.30, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50]:
            for rescue_vd in [False, True]:
                for yelp_boost in [0.0, 0.3, 0.5]:
                    for yelp_min in [20, 50, 100]:
                        acc, tp, tn, fp, fn, cs = evaluate(
                            boost, boost_min_reviews, threshold,
                            rescue_fsq_verified_dead=rescue_vd,
                            yelp_open_boost=yelp_boost,
                            yelp_open_min_reviews=yelp_min,
                        )
                        if acc >= best_acc:
                            if acc > best_acc or fp < (best_config[5] if best_config else 999):
                                best_acc = acc
                                best_config = (boost, boost_min_reviews, threshold, rescue_vd, yelp_boost, fp, fn, cs, yelp_min)
                                if acc >= 0.895:
                                    results.append((acc, boost, boost_min_reviews, threshold, rescue_vd, yelp_boost, yelp_min, fp, fn, cs))

# Sort and show top results
results.sort(key=lambda x: (-x[0], x[7]))
print(f"Best accuracy: {best_acc:.1%}")
print(f"Best config: boost={best_config[0]}, min_reviews={best_config[1]}, threshold={best_config[2]}, rescue_vd={best_config[3]}, yelp_boost={best_config[4]}, yelp_min={best_config[8]}")
print(f"  FP={best_config[5]}, FN={best_config[6]}")
for city, stats in best_config[7].items():
    t = sum(stats)
    a = (stats[0] + stats[1]) / t if t else 0
    print(f"  {city:10s}: {a:.1%} ({stats[0]+stats[1]}/{t}) TP={stats[0]} TN={stats[1]} FP={stats[2]} FN={stats[3]}")

print(f"\nTop configs >= 89.5%:")
for r in results[:20]:
    acc, boost, bmr, thresh, rvd, yb, ym, fp, fn, cs = r
    min_city = min((sum(s[:2])/sum(s) if sum(s) else 0) for s in cs.values())
    print(f"  {acc:.1%} | boost={boost} min_rev={bmr} thresh={thresh} rescue_vd={rvd} yelp_boost={yb} yelp_min={ym} | FP={fp} FN={fn} | min_city={min_city:.1%}")
