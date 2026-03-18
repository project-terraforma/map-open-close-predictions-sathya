"""
Find closed businesses in Miami from Overture fill candidates.
Checks Foursquare and Yelp to identify businesses that are closed.
Outputs candidates with ground_truth='closed' for manual review.

Usage: python scripts/find_closed_miami.py
"""
import json
import os
import sys
import re
import time
import random
import urllib.request
import urllib.parse
import urllib.error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')

ENV_PATH = os.path.join(PROJECT_DIR, '.env')
FOURSQUARE_API_KEY = None
YELP_API_KEY = None
if os.path.exists(ENV_PATH):
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line.startswith('FOURSQUARE_API_KEY='):
                FOURSQUARE_API_KEY = line.split('=', 1)[1]
            elif line.startswith('YELP_API_KEY='):
                YELP_API_KEY = line.split('=', 1)[1]

FILL_PATH = os.path.join(SCRIPT_DIR, 'overture_candidates_miami_fill.json')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'overture_candidates_miami_closed.json')


def normalize(text):
    return re.sub(r'[^a-z0-9]', '', (text or '').lower())


def name_match_score(query_name, result_name):
    norm_q = normalize(query_name)
    norm_r = normalize(result_name)
    if norm_q == norm_r:
        return 1.0
    if norm_q in norm_r or norm_r in norm_q:
        return 0.8
    words_q = set(normalize(w) for w in query_name.split() if len(w) > 1)
    words_r = set(normalize(w) for w in result_name.split() if len(w) > 1)
    if not words_q:
        return 0.0
    overlap = len(words_q & words_r)
    return overlap / max(len(words_q), 1) * 0.6


def check_foursquare(name, lat, lng):
    """Check Foursquare for business. Returns status."""
    params = urllib.parse.urlencode({
        'll': f'{lat},{lng}',
        'query': name,
        'radius': 200,
        'limit': 3,
    })
    url = f"https://api.foursquare.com/v3/places/search?{params}"
    req = urllib.request.Request(url, headers={
        'Authorization': FOURSQUARE_API_KEY,
        'Accept': 'application/json',
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return 'error'

    results = data.get('results', [])
    if not results:
        return 'not_found'

    for r in results:
        score = name_match_score(name, r.get('name', ''))
        if score >= 0.5:
            closed = r.get('closed_bucket', 'VeryLikelyOpen')
            if 'Closed' in closed or 'LikelyClosed' in closed:
                return 'closed'
            return 'open'
    return 'not_found'


def check_yelp(name, lat, lng):
    """Check Yelp for business. Returns (is_closed, review_count)."""
    params = urllib.parse.urlencode({
        'term': name,
        'latitude': lat,
        'longitude': lng,
        'limit': 3,
    })
    url = f"https://api.yelp.com/v3/businesses/search?{params}"
    req = urllib.request.Request(url, headers={
        'Authorization': f'Bearer {YELP_API_KEY}',
        'Accept': 'application/json',
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return None, 0

    for biz in data.get('businesses', []):
        score = name_match_score(name, biz.get('name', ''))
        if score >= 0.5:
            return biz.get('is_closed', False), biz.get('review_count', 0)
    return None, 0


def main():
    with open(FILL_PATH, encoding='utf-8') as f:
        candidates = json.load(f)

    # Shuffle to get variety
    random.seed(42)
    random.shuffle(candidates)

    closed_businesses = []
    checked = 0
    target = 46  # We want ~38 closed, check extra for margin

    print(f"Scanning {len(candidates)} candidates for closed businesses...")
    print(f"Target: {target} closed businesses\n")

    for loc in candidates:
        if len(closed_businesses) >= target:
            break

        name = loc.get('name', '')
        coords = loc.get('location', [0, 0])
        lng, lat = coords[0], coords[1]

        # Check Foursquare first (free, fast)
        fsq_status = check_foursquare(name, lat, lng)
        time.sleep(0.2)

        # Check Yelp
        yelp_closed, yelp_reviews = check_yelp(name, lat, lng)
        time.sleep(0.3)

        checked += 1

        # Determine if closed: Foursquare says closed OR Yelp says closed
        is_closed = False
        reason = ''
        if fsq_status == 'closed':
            is_closed = True
            reason = 'foursquare_closed'
        elif yelp_closed is True:
            is_closed = True
            reason = 'yelp_closed'
        elif fsq_status == 'not_found' and yelp_closed is None:
            # Not found on either — likely closed
            is_closed = True
            reason = 'not_found_anywhere'

        safe_name = name.encode('ascii', 'replace').decode()
        if is_closed:
            closed_businesses.append(loc)
            print(f"  CLOSED ({reason:20s}) | {safe_name}")
        else:
            if checked % 20 == 0:
                print(f"  ... checked {checked}, found {len(closed_businesses)} closed so far")

    print(f"\nChecked {checked} candidates, found {len(closed_businesses)} closed businesses")

    # Save
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(closed_businesses, f, indent=2, ensure_ascii=True)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
