"""
Verify the 40 test businesses against Yelp to get real ground truth labels.
Replaces the Foursquare-based ground truth with Yelp's is_closed field.

Usage: python scripts/verify_test_yelp.py
Input:  src/data/test_data.json
Output: src/data/test_data.json (updated with Yelp ground truth)
"""

import os
import json
import re
import time
import urllib.request
import urllib.parse
import urllib.error

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'test_data.json')

ENV_PATH = os.path.join(os.path.dirname(__file__), '..', '.env')
YELP_API_KEY = None
if os.path.exists(ENV_PATH):
    with open(ENV_PATH) as f:
        for line in f:
            if line.startswith('YELP_API_KEY='):
                YELP_API_KEY = line.strip().split('=', 1)[1]

if not YELP_API_KEY:
    raise RuntimeError("YELP_API_KEY not found in .env")


def normalize(text):
    return re.sub(r'[^a-z0-9]', '', (text or '').lower())


def yelp_search(name, lat, lng):
    """Search Yelp for a business by name and coordinates."""
    params = urllib.parse.urlencode({
        'term': name,
        'latitude': lat,
        'longitude': lng,
        'radius': 500,
        'limit': 5,
    })
    url = f"https://api.yelp.com/v3/businesses/search?{params}"
    req = urllib.request.Request(url, headers={
        'Authorization': f'Bearer {YELP_API_KEY}',
        'Accept': 'application/json',
    })

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} for {name}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

    businesses = data.get('businesses', [])
    if not businesses:
        return None

    norm_name = normalize(name)
    best = None
    best_score = 0

    for biz in businesses:
        biz_name = normalize(biz.get('name', ''))
        if norm_name == biz_name:
            score = 1.0
        elif norm_name in biz_name or biz_name in norm_name:
            score = 0.8
        else:
            words_a = set(normalize(w) for w in name.split())
            words_b = set(normalize(w) for w in biz.get('name', '').split())
            overlap = len(words_a & words_b)
            score = overlap / max(len(words_a), 1) * 0.6

        if score > best_score:
            best_score = score
            best = biz

    if best and best_score >= 0.3:
        return {
            'yelp_name': best.get('name'),
            'is_closed': best.get('is_closed', False),
            'match_score': best_score,
            'yelp_rating': best.get('rating'),
            'yelp_review_count': best.get('review_count', 0),
            'yelp_url': best.get('url', ''),
        }
    return None


def main():
    with open(DATA_PATH, encoding='utf-8') as f:
        data = json.load(f)

    print(f"Verifying {len(data)} test businesses against Yelp...\n")

    for i, loc in enumerate(data):
        name = loc.get('name', '')
        coords = loc.get('location', [0, 0])
        lng, lat = coords[0], coords[1]

        result = yelp_search(name, lat, lng)
        time.sleep(0.3)

        if result:
            old_gt = loc.get('ground_truth', '?')
            new_gt = 'closed' if result['is_closed'] else 'open'
            changed = old_gt != new_gt

            loc['ground_truth'] = new_gt
            loc['yelp'] = result

            status = 'CHANGED' if changed else 'ok'
            print(f"  {i+1:2d}. {name:35s}  yelp={new_gt:6s}  old={old_gt:6s}  [{status}]  match={result['match_score']:.1f}  ({result['yelp_name']})")
        else:
            print(f"  {i+1:2d}. {name:35s}  NO YELP MATCH — keeping old ground truth")

    # Save updated data
    with open(DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Summary
    n_open = sum(1 for d in data if d.get('ground_truth') == 'open')
    n_closed = sum(1 for d in data if d.get('ground_truth') == 'closed')
    print(f"\n  Updated ground truth:")
    print(f"    Open:   {n_open}")
    print(f"    Closed: {n_closed}")
    print(f"  Saved to {DATA_PATH}")


if __name__ == '__main__':
    main()
