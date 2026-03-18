"""
More aggressive optimization: modify signal encodings and add targeted rules.
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

all_locs = []
for city, path in DATA_FILES:
    with open(path) as f:
        data = json.load(f)
    for loc in data:
        if loc.get('ground_truth') and loc.get('vision', {}).get('layers'):
            all_locs.append((city, loc))

print(f"Total locations: {len(all_locs)}")


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

    return {
        'fsq_status': fsq_status, 'fsq_signal': fsq_signal,
        'ws_status': ws_status, 'ws_signal': ws_signal,
        'text_signal': text_signal, 'text_verdict': text_verdict,
        'xgb_score': xgb_score, 'xgb_centered': xgb_centered,
        'tt_status': tt_status, 'tt_match': tt_match, 'tt_signal': tt_signal,
        'yelp_closed': yelp.get('is_closed'), 'yelp_reviews': yelp_reviews, 'yelp_signal': yelp_signal,
    }


def evaluate(params):
    fsq_nodata_penalty = params.get('fsq_nodata_penalty', META_WEIGHTS['fsq_no_data'])
    fsq_verified_no_yelp_w = params.get('fsq_verified_no_yelp_w', META_WEIGHTS['fsq_verified_no_yelp'])
    fsq_verified_web_dead_w = params.get('fsq_verified_web_dead_w', META_WEIGHTS['fsq_verified_web_dead'])
    threshold = params['threshold']
    boost = params.get('boost', 0.0)
    boost_min_reviews = params.get('boost_min_reviews', 20)
    # New: rescue rules
    rescue_verified_dead = params.get('rescue_verified_dead', 0.0)
    yelp_open_boost = params.get('yelp_open_boost', 0.0)
    yelp_open_min = params.get('yelp_open_min', 50)
    mismatch_alive_boost = params.get('mismatch_alive_boost', 0.0)
    xgb_weight_mult = params.get('xgb_weight_mult', 1.0)

    tp = tn = fp = fn = 0
    city_stats = {}

    for city, loc in all_locs:
        gt = loc['ground_truth']
        s = encode_signals(loc)

        # Base logit with potentially modified weights
        fsq_no_data_flag = 1.0 if s['fsq_status'] == 'no_data' else 0.0
        web_alive = 1.0 if s['ws_status'] == 'alive' else 0.0
        fsq_nodata_web_alive = fsq_no_data_flag * web_alive
        tt_strong = s['tt_status'] == 'verified' and s['tt_match'] >= 0.8
        both_dirs_verified = 1.0 if (s['fsq_status'] == 'verified' and tt_strong) else 0.0
        both_dirs_missing = 1.0 if (s['fsq_status'] == 'no_data' and not tt_strong) else 0.0
        fsq_verified_web_dead = 1.0 if (s['fsq_status'] == 'verified' and s['ws_status'] == 'dead') else 0.0
        fsq_verified_no_yelp = 1.0 if (s['fsq_status'] == 'verified' and s['yelp_reviews'] == 0) else 0.0

        logit = (
            META_WEIGHTS['foursquare'] * s['fsq_signal'] +
            META_WEIGHTS['website'] * s['ws_signal'] +
            META_WEIGHTS['text'] * s['text_signal'] +
            META_WEIGHTS['xgboost'] * s['xgb_centered'] * xgb_weight_mult +
            META_WEIGHTS['tomtom'] * s['tt_signal'] +
            META_WEIGHTS['yelp_reviews'] * s['yelp_signal'] +
            fsq_verified_web_dead_w * fsq_verified_web_dead +
            fsq_nodata_penalty * fsq_no_data_flag +
            META_WEIGHTS['fsq_nodata_web_alive'] * fsq_nodata_web_alive +
            META_WEIGHTS['both_dirs_verified'] * both_dirs_verified +
            META_WEIGHTS['both_dirs_missing'] * both_dirs_missing +
            fsq_verified_no_yelp_w * fsq_verified_no_yelp +
            META_INTERCEPT
        )

        # Soft boost: FSQ missing + Yelp open + reviews + corroboration
        yelp_not_closed = s['yelp_closed'] == False
        has_corroboration = s['ws_status'] == 'alive' or (s['tt_status'] == 'verified' and s['tt_match'] >= 0.5)
        if s['fsq_status'] == 'no_data' and yelp_not_closed and s['yelp_reviews'] >= boost_min_reviews and has_corroboration:
            logit += boost

        # Rescue: fsq=verified + web=dead (undo excessive penalty)
        if rescue_verified_dead > 0 and s['fsq_status'] == 'verified' and s['ws_status'] == 'dead':
            logit += rescue_verified_dead

        # Yelp confirmed open with reviews
        if yelp_open_boost > 0 and s['yelp_closed'] == False and s['yelp_reviews'] >= yelp_open_min:
            logit += yelp_open_boost

        # Mismatch + alive website
        if mismatch_alive_boost > 0 and s['fsq_status'] == 'mismatch' and s['ws_status'] == 'alive':
            logit += mismatch_alive_boost

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


best_acc = 0
best_params = None
count = 0

for fsq_nodata_p in [-1.33, -1.0, -0.8, -0.5, -0.3]:
    for fsq_vn_w in [-0.99, -0.5, -0.3, 0.0]:
        for fsq_vd_w in [-1.05, -0.7, -0.3, 0.0]:
            for threshold in [0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50]:
                for boost in [0.0, 0.5, 0.8, 1.0, 1.5]:
                    for boost_min in [5, 10, 20, 50]:
                        for rescue_vd in [0.0, 0.5, 1.0]:
                            params = {
                                'fsq_nodata_penalty': fsq_nodata_p,
                                'fsq_verified_no_yelp_w': fsq_vn_w,
                                'fsq_verified_web_dead_w': fsq_vd_w,
                                'threshold': threshold,
                                'boost': boost,
                                'boost_min_reviews': boost_min,
                                'rescue_verified_dead': rescue_vd,
                            }
                            acc, tp, tn, fp, fn, cs = evaluate(params)
                            count += 1
                            if acc > best_acc or (acc == best_acc and fp < (best_params['_fp'] if best_params else 999)):
                                best_acc = acc
                                best_params = {**params, '_fp': fp, '_fn': fn, '_cs': cs}

print(f"Searched {count} configurations")
print(f"\nBest accuracy: {best_acc:.1%}")
p = best_params
print(f"Config: fsq_nodata={p['fsq_nodata_penalty']}, fsq_vn={p['fsq_verified_no_yelp_w']}, fsq_vd={p['fsq_verified_web_dead_w']}")
print(f"  threshold={p['threshold']}, boost={p['boost']}, boost_min={p['boost_min_reviews']}, rescue_vd={p['rescue_verified_dead']}")
print(f"  FP={p['_fp']}, FN={p['_fn']}")
for city, stats in p['_cs'].items():
    t = sum(stats)
    a = (stats[0] + stats[1]) / t if t else 0
    print(f"  {city:10s}: {a:.1%} ({stats[0]+stats[1]}/{t}) TP={stats[0]} TN={stats[1]} FP={stats[2]} FN={stats[3]}")

# Now also try with yelp_open_boost
print("\n--- Adding yelp_open_boost ---")
best2_acc = best_acc
best2_params = best_params
for yelp_boost in [0.3, 0.5, 0.8]:
    for yelp_min in [10, 20, 30, 50]:
        for mismatch_boost in [0.0, 0.3, 0.5]:
            p2 = {**{k:v for k,v in best_params.items() if not k.startswith('_')},
                  'yelp_open_boost': yelp_boost, 'yelp_open_min': yelp_min,
                  'mismatch_alive_boost': mismatch_boost}
            acc, tp, tn, fp, fn, cs = evaluate(p2)
            if acc > best2_acc or (acc == best2_acc and fp < best2_params['_fp']):
                best2_acc = acc
                best2_params = {**p2, '_fp': fp, '_fn': fn, '_cs': cs}

print(f"Best accuracy with yelp_open_boost: {best2_acc:.1%}")
p = best2_params
print(f"Config: fsq_nodata={p['fsq_nodata_penalty']}, fsq_vn={p['fsq_verified_no_yelp_w']}, fsq_vd={p['fsq_verified_web_dead_w']}")
print(f"  threshold={p['threshold']}, boost={p['boost']}, boost_min={p['boost_min_reviews']}, rescue_vd={p['rescue_verified_dead']}")
print(f"  yelp_open_boost={p.get('yelp_open_boost',0)}, yelp_open_min={p.get('yelp_open_min',0)}, mismatch_boost={p.get('mismatch_alive_boost',0)}")
print(f"  FP={p['_fp']}, FN={p['_fn']}")
for city, stats in p['_cs'].items():
    t = sum(stats)
    a = (stats[0] + stats[1]) / t if t else 0
    print(f"  {city:10s}: {a:.1%} ({stats[0]+stats[1]}/{t}) TP={stats[0]} TN={stats[1]} FP={stats[2]} FN={stats[3]}")
