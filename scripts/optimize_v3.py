"""
Analyze what post-hoc override rules could work, then grid search over them.
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


def get_base_score(loc):
    """Compute base metamodel score (no overrides)."""
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

    return logit, {
        'fsq_status': fsq_status, 'ws_status': ws_status,
        'yelp_closed': yelp.get('is_closed'), 'yelp_reviews': yelp_reviews,
        'tt_status': tt_status, 'tt_match': tt_match,
        'xgb_score': xgb_score, 'text_verdict': text_verdict, 'text_signal': text_signal,
    }


# First: Check how many TN (correct closed predictions) have fsq=verified
print("=== TN with fsq=verified (would become FP with override) ===")
tn_verified = 0
for city, loc in all_locs:
    gt = loc['ground_truth']
    fsq_status = loc.get('foursquare', {}).get('status', 'no_data')
    if gt != 'open' and fsq_status == 'verified':
        logit, s = get_base_score(loc)
        score = 1.0 / (1.0 + math.exp(-logit))
        yelp = loc.get('yelp', {})
        name = loc.get('name', '?')
        print(f"  {city:8s} | {name:35s} | score={score:.3f} | web={s['ws_status']:10s} | yelp_closed={s['yelp_closed']} reviews={s['yelp_reviews']}")
        tn_verified += 1
print(f"Total TN with fsq=verified: {tn_verified}")

# Check how many TP have fsq=verified (would stay TP, no change)
tp_verified = 0
for city, loc in all_locs:
    gt = loc['ground_truth']
    fsq_status = loc.get('foursquare', {}).get('status', 'no_data')
    logit, s = get_base_score(loc)
    score = 1.0 / (1.0 + math.exp(-logit))
    if gt == 'open' and fsq_status == 'verified' and score <= 0.50:
        tp_verified += 1
print(f"FN with fsq=verified (would be rescued): {tp_verified}")


# Now try override-based approach
print("\n=== GRID SEARCH WITH POST-HOC OVERRIDES ===")
best_acc = 0
best_config = None

for threshold in [0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50]:
    for boost in [0.0, 0.5, 0.8, 1.0, 1.2, 1.5]:
        for boost_min in [5, 10, 20, 50]:
            for fsq_override in [False, True]:  # Force open if fsq=verified AND yelp not closed
                for yelp_review_override_min in [0, 30, 50, 100]:  # Force open if yelp_closed=False and reviews >= X and web=alive
                    tp = tn = fp = fn = 0
                    city_stats = {}

                    for city, loc in all_locs:
                        gt = loc['ground_truth']
                        logit, s = get_base_score(loc)

                        # Soft boost
                        yelp_not_closed = s['yelp_closed'] == False
                        has_corr = s['ws_status'] == 'alive' or (s['tt_status'] == 'verified' and s['tt_match'] >= 0.5)
                        if s['fsq_status'] == 'no_data' and yelp_not_closed and s['yelp_reviews'] >= boost_min and has_corr:
                            logit += boost

                        open_score = 1.0 / (1.0 + math.exp(-logit))

                        # Yelp closed override
                        if s['yelp_closed'] == True:
                            open_score = min(open_score, 0.15)

                        pred = 'open' if open_score > threshold else 'not_open'

                        # Post-hoc override 1: FSQ verified = trust it
                        if fsq_override and s['fsq_status'] == 'verified' and s['yelp_closed'] != True:
                            pred = 'open'

                        # Post-hoc override 2: Yelp confirms open with substantial reviews + web alive
                        if yelp_review_override_min > 0 and s['yelp_closed'] == False and s['yelp_reviews'] >= yelp_review_override_min and s['ws_status'] == 'alive':
                            pred = 'open'

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
                    min_city_acc = min((s[0]+s[1])/sum(s) for s in city_stats.values())

                    if acc > best_acc or (acc == best_acc and fp < (best_config['fp'] if best_config else 999)):
                        best_acc = acc
                        best_config = {
                            'threshold': threshold, 'boost': boost, 'boost_min': boost_min,
                            'fsq_override': fsq_override, 'yelp_override_min': yelp_review_override_min,
                            'fp': fp, 'fn': fn, 'cs': city_stats, 'min_city': min_city_acc,
                        }

print(f"\nBest accuracy: {best_acc:.1%}")
c = best_config
print(f"Config: threshold={c['threshold']}, boost={c['boost']}, boost_min={c['boost_min']}")
print(f"  fsq_override={c['fsq_override']}, yelp_override_min={c['yelp_override_min']}")
print(f"  FP={c['fp']}, FN={c['fn']}, min_city={c['min_city']:.1%}")
for city, stats in c['cs'].items():
    t = sum(stats)
    a = (stats[0] + stats[1]) / t if t else 0
    print(f"  {city:10s}: {a:.1%} ({stats[0]+stats[1]}/{t}) TP={stats[0]} TN={stats[1]} FP={stats[2]} FN={stats[3]}")

# Also find best config with min_city >= 82%
print("\n=== BEST WITH min_city >= 82% ===")
best_acc2 = 0
best_config2 = None

for threshold in [0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50]:
    for boost in [0.0, 0.5, 0.8, 1.0, 1.2, 1.5]:
        for boost_min in [5, 10, 20, 50]:
            for fsq_override in [False, True]:
                for yelp_review_override_min in [0, 30, 50, 100, 200]:
                    tp = tn = fp = fn = 0
                    city_stats = {}

                    for city, loc in all_locs:
                        gt = loc['ground_truth']
                        logit, s = get_base_score(loc)

                        yelp_not_closed = s['yelp_closed'] == False
                        has_corr = s['ws_status'] == 'alive' or (s['tt_status'] == 'verified' and s['tt_match'] >= 0.5)
                        if s['fsq_status'] == 'no_data' and yelp_not_closed and s['yelp_reviews'] >= boost_min and has_corr:
                            logit += boost

                        open_score = 1.0 / (1.0 + math.exp(-logit))

                        if s['yelp_closed'] == True:
                            open_score = min(open_score, 0.15)

                        pred = 'open' if open_score > threshold else 'not_open'

                        if fsq_override and s['fsq_status'] == 'verified' and s['yelp_closed'] != True:
                            pred = 'open'

                        if yelp_review_override_min > 0 and s['yelp_closed'] == False and s['yelp_reviews'] >= yelp_review_override_min and s['ws_status'] == 'alive':
                            pred = 'open'

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
                    min_city_acc = min((s[0]+s[1])/sum(s) for s in city_stats.values())

                    if min_city_acc >= 0.82:
                        if acc > best_acc2 or (acc == best_acc2 and fp < (best_config2['fp'] if best_config2 else 999)):
                            best_acc2 = acc
                            best_config2 = {
                                'threshold': threshold, 'boost': boost, 'boost_min': boost_min,
                                'fsq_override': fsq_override, 'yelp_override_min': yelp_review_override_min,
                                'fp': fp, 'fn': fn, 'cs': city_stats, 'min_city': min_city_acc,
                            }

if best_config2:
    print(f"Best accuracy: {best_acc2:.1%}")
    c = best_config2
    print(f"Config: threshold={c['threshold']}, boost={c['boost']}, boost_min={c['boost_min']}")
    print(f"  fsq_override={c['fsq_override']}, yelp_override_min={c['yelp_override_min']}")
    print(f"  FP={c['fp']}, FN={c['fn']}, min_city={c['min_city']:.1%}")
    for city, stats in c['cs'].items():
        t = sum(stats)
        a = (stats[0] + stats[1]) / t if t else 0
        print(f"  {city:10s}: {a:.1%} ({stats[0]+stats[1]}/{t}) TP={stats[0]} TN={stats[1]} FP={stats[2]} FN={stats[3]}")
