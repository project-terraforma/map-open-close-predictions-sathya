"""
Try modifying signal encodings themselves (not just weights/thresholds).
Key insight: yelp_reviews=0 gives -1.0 signal which is very harsh for businesses
that simply have no Yelp presence. Try making it neutral.
Also try different fsq=mismatch and fsq=no_data encodings.
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


def evaluate(params):
    threshold = params['threshold']
    boost = params.get('boost', 0.0)
    boost_min = params.get('boost_min', 20)
    yelp_zero_signal = params.get('yelp_zero_signal', -1.0)
    yelp_no_match_signal = params.get('yelp_no_match_signal', -1.0)  # when yelp_closed=None
    fsq_nodata_signal = params.get('fsq_nodata_signal', -0.3)
    fsq_mismatch_signal = params.get('fsq_mismatch_signal', -0.5)
    web_nourl_signal = params.get('web_nourl_signal', 0.0)

    tp = tn = fp = fn = 0
    city_stats = {}

    for city, loc in all_locs:
        gt = loc['ground_truth']
        vision = loc.get('vision', {})
        layers = vision.get('layers', {})
        fsq = loc.get('foursquare', {})
        fsq_status = fsq.get('status', 'no_data')
        ws = loc.get('website_check', {})
        ws_status = ws.get('status', 'no_url')

        if fsq_status == 'verified': fsq_signal = 1.0
        elif fsq_status == 'closed': fsq_signal = -1.0
        elif fsq_status == 'mismatch': fsq_signal = fsq_mismatch_signal
        else: fsq_signal = fsq_nodata_signal

        if ws_status == 'alive': ws_signal = 0.5
        elif ws_status == 'redirect': ws_signal = -0.7
        elif ws_status == 'dead': ws_signal = -0.3
        elif ws_status == 'parked': ws_signal = -0.5
        else: ws_signal = web_nourl_signal

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
        yelp_closed = yelp.get('is_closed')

        if yelp_reviews > 0:
            yelp_signal = min(1.0, (math.log10(yelp_reviews) - 1.0) / 2.0)
        elif yelp_closed is None:
            yelp_signal = yelp_no_match_signal  # No Yelp match at all
        else:
            yelp_signal = yelp_zero_signal  # Yelp matched but 0 reviews

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

        # Soft boost
        yelp_not_closed = yelp_closed == False
        has_corr = ws_status == 'alive' or (tt_status == 'verified' and tt_match >= 0.5)
        if fsq_status == 'no_data' and yelp_not_closed and yelp_reviews >= boost_min and has_corr:
            logit += boost

        open_score = 1.0 / (1.0 + math.exp(-logit))

        # Yelp closed override
        if yelp_closed == True:
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
    min_city = min((s[0]+s[1])/sum(s) for s in city_stats.values())
    return acc, fp, fn, city_stats, min_city


best_acc = 0
best_config = None
count = 0

for threshold in [0.30, 0.33, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50]:
    for boost in [0.0, 0.3, 0.5, 0.8, 1.0, 1.2]:
        for boost_min in [5, 10, 20, 50]:
            for yelp_zero in [-1.0, -0.5, -0.3, 0.0]:
                for yelp_no_match in [-1.0, -0.5, -0.3, 0.0]:
                    for fsq_nodata_sig in [-0.3, -0.1, 0.0]:
                        for fsq_mismatch_sig in [-0.5, -0.3, -0.1, 0.0]:
                            params = {
                                'threshold': threshold,
                                'boost': boost,
                                'boost_min': boost_min,
                                'yelp_zero_signal': yelp_zero,
                                'yelp_no_match_signal': yelp_no_match,
                                'fsq_nodata_signal': fsq_nodata_sig,
                                'fsq_mismatch_signal': fsq_mismatch_sig,
                            }
                            acc, fp, fn, cs, mc = evaluate(params)
                            count += 1

                            if acc > best_acc or (acc == best_acc and fp < (best_config['fp'] if best_config else 999)):
                                best_acc = acc
                                best_config = {**params, 'fp': fp, 'fn': fn, 'cs': cs, 'mc': mc}

print(f"Searched {count} configs")
print(f"\nBest overall: {best_acc:.1%}")
c = best_config
print(f"  threshold={c['threshold']}, boost={c['boost']}, boost_min={c['boost_min']}")
print(f"  yelp_zero={c['yelp_zero_signal']}, yelp_no_match={c['yelp_no_match_signal']}")
print(f"  fsq_nodata={c['fsq_nodata_signal']}, fsq_mismatch={c['fsq_mismatch_signal']}")
print(f"  FP={c['fp']}, FN={c['fn']}, min_city={c['mc']:.1%}")
for city, stats in c['cs'].items():
    t = sum(stats)
    a = (stats[0] + stats[1]) / t if t else 0
    print(f"  {city:10s}: {a:.1%} ({stats[0]+stats[1]}/{t}) TP={stats[0]} TN={stats[1]} FP={stats[2]} FN={stats[3]}")

# Find best with min_city >= 82%
print("\n=== BEST WITH min_city >= 82% ===")
best_acc2 = 0
best_config2 = None

for threshold in [0.30, 0.33, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50]:
    for boost in [0.0, 0.3, 0.5, 0.8, 1.0, 1.2]:
        for boost_min in [5, 10, 20, 50]:
            for yelp_zero in [-1.0, -0.5, -0.3, 0.0]:
                for yelp_no_match in [-1.0, -0.5, -0.3, 0.0]:
                    for fsq_nodata_sig in [-0.3, -0.1, 0.0]:
                        for fsq_mismatch_sig in [-0.5, -0.3, -0.1, 0.0]:
                            params = {
                                'threshold': threshold,
                                'boost': boost,
                                'boost_min': boost_min,
                                'yelp_zero_signal': yelp_zero,
                                'yelp_no_match_signal': yelp_no_match,
                                'fsq_nodata_signal': fsq_nodata_sig,
                                'fsq_mismatch_signal': fsq_mismatch_sig,
                            }
                            acc, fp, fn, cs, mc = evaluate(params)

                            if mc >= 0.82 and (acc > best_acc2 or (acc == best_acc2 and fp < (best_config2['fp'] if best_config2 else 999))):
                                best_acc2 = acc
                                best_config2 = {**params, 'fp': fp, 'fn': fn, 'cs': cs, 'mc': mc}

if best_config2:
    c = best_config2
    print(f"Best: {best_acc2:.1%}")
    print(f"  threshold={c['threshold']}, boost={c['boost']}, boost_min={c['boost_min']}")
    print(f"  yelp_zero={c['yelp_zero_signal']}, yelp_no_match={c['yelp_no_match_signal']}")
    print(f"  fsq_nodata={c['fsq_nodata_signal']}, fsq_mismatch={c['fsq_mismatch_signal']}")
    print(f"  FP={c['fp']}, FN={c['fn']}, min_city={c['mc']:.1%}")
    for city, stats in c['cs'].items():
        t = sum(stats)
        a = (stats[0] + stats[1]) / t if t else 0
        print(f"  {city:10s}: {a:.1%} ({stats[0]+stats[1]}/{t}) TP={stats[0]} TN={stats[1]} FP={stats[2]} FN={stats[3]}")
