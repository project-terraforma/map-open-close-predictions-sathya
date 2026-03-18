import json

files = [
    ('SF', 'src/data/test_data.json'),
    ('LA', 'src/data/test_data_la.json'),
    ('Chicago', 'src/data/test_data_chicago.json'),
    ('Miami', 'src/data/test_data_miami.json'),
    ('Philly', 'src/data/test_data_philly.json'),
]

print('=== FALSE POSITIVES (predicted open, actually closed) ===')
fp_count = 0
for city, path in files:
    with open(path) as f:
        data = json.load(f)
    for loc in data:
        gt = loc.get('ground_truth')
        pred = loc.get('vision', {}).get('prediction')
        if not gt or not pred:
            continue
        if gt != 'open' and pred == 'open':
            fp_count += 1
            name = loc.get('name', '?')
            score = loc.get('vision', {}).get('confidence', 0)
            fsq = loc.get('foursquare', {}).get('status', 'no_data')
            ws = loc.get('website_check', {}).get('status', 'no_url')
            yelp = loc.get('yelp', {})
            yelp_closed = yelp.get('is_closed')
            yelp_reviews = yelp.get('yelp_review_count', 0)
            tt = loc.get('tomtom', {}).get('status', 'no_data')
            tt_match = loc.get('tomtom', {}).get('match_score', 0)
            xgb = loc.get('vision', {}).get('layers', {}).get('xgboost', {}).get('score', 0.5)
            print(f'{fp_count:2d} {city:8s} | {name:35s} | score={score:.3f} | fsq={fsq:10s} | web={ws:10s} | yelp_closed={str(yelp_closed):5s} reviews={yelp_reviews:3d} | tt={tt} match={tt_match:.1f} | xgb={xgb:.2f}')

print(f'\nTotal FP: {fp_count}')
print()
print('=== FALSE NEGATIVES (predicted not_open, actually open) ===')
fn_count = 0
for city, path in files:
    with open(path) as f:
        data = json.load(f)
    for loc in data:
        gt = loc.get('ground_truth')
        pred = loc.get('vision', {}).get('prediction')
        if not gt or not pred:
            continue
        if gt == 'open' and pred != 'open':
            fn_count += 1
            name = loc.get('name', '?')
            score = loc.get('vision', {}).get('confidence', 0)
            fsq = loc.get('foursquare', {}).get('status', 'no_data')
            ws = loc.get('website_check', {}).get('status', 'no_url')
            yelp = loc.get('yelp', {})
            yelp_closed = yelp.get('is_closed')
            yelp_reviews = yelp.get('yelp_review_count', 0)
            tt = loc.get('tomtom', {}).get('status', 'no_data')
            tt_match = loc.get('tomtom', {}).get('match_score', 0)
            xgb = loc.get('vision', {}).get('layers', {}).get('xgboost', {}).get('score', 0.5)
            print(f'{fn_count:2d} {city:8s} | {name:35s} | score={score:.3f} | fsq={fsq:10s} | web={ws:10s} | yelp_closed={str(yelp_closed):5s} reviews={yelp_reviews:3d} | tt={tt} match={tt_match:.1f} | xgb={xgb:.2f}')

print(f'\nTotal FN: {fn_count}')
