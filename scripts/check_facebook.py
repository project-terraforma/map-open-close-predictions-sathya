"""
Check Facebook Places API for business verification.
Uses the Facebook Graph API to search for businesses by name + location.

Facebook statuses:
  - verified:  Found matching business page on Facebook
  - closed:    Found but marked as permanently closed
  - no_data:   Not found on Facebook

Usage: python scripts/check_facebook.py [--city sf|la|chicago|miami] [--input path]
"""

import json
import os
import sys
import re
import time
import urllib.request
import urllib.parse
import urllib.error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')

ENV_PATH = os.path.join(PROJECT_DIR, '.env')
FB_APP_ID = None
FB_APP_SECRET = None
if os.path.exists(ENV_PATH):
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line.startswith('FACEBOOK_APP_ID='):
                FB_APP_ID = line.split('=', 1)[1]
            elif line.startswith('FACEBOOK_APP_SECRET='):
                FB_APP_SECRET = line.split('=', 1)[1]

if not FB_APP_ID or not FB_APP_SECRET:
    raise RuntimeError("FACEBOOK_APP_ID and FACEBOOK_APP_SECRET not found in .env")

# App access token = app_id|app_secret
FB_ACCESS_TOKEN = f"{FB_APP_ID}|{FB_APP_SECRET}"

CITY_DATA = {
    'sf': os.path.join(PROJECT_DIR, 'src', 'data', 'test_data.json'),
    'la': os.path.join(PROJECT_DIR, 'src', 'data', 'test_data_la.json'),
    'chicago': os.path.join(PROJECT_DIR, 'src', 'data', 'test_data_chicago.json'),
    'miami': os.path.join(PROJECT_DIR, 'src', 'data', 'test_data_miami.json'),
}

# Parse args
city = None
input_path = None
for i, arg in enumerate(sys.argv):
    if arg == '--city' and i + 1 < len(sys.argv):
        city = sys.argv[i + 1]
    if arg == '--input' and i + 1 < len(sys.argv):
        input_path = sys.argv[i + 1]


def normalize(text):
    return re.sub(r'[^a-z0-9]', '', (text or '').lower())


def name_match_score(query_name, result_name):
    """Score how well two business names match (0-1)."""
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


def fb_places_search(name, lat, lng):
    """Search Facebook Places API for a business."""
    params = urllib.parse.urlencode({
        'type': 'place',
        'q': name,
        'center': f'{lat},{lng}',
        'distance': 500,
        'fields': 'name,location,phone,website,hours,permanently_closed,checkins,category_list,verification_status,is_verified',
        'limit': 5,
        'access_token': FB_ACCESS_TOKEN,
    })
    url = f"https://graph.facebook.com/v19.0/search?{params}"

    req = urllib.request.Request(url, headers={'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ''
        print(f"    HTTP {e.code}: {body[:200]}")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None

    results = data.get('data', [])
    if not results:
        return None

    best = None
    best_score = 0.0

    for r in results:
        r_name = r.get('name', '')
        score = name_match_score(name, r_name)
        if score > best_score:
            best_score = score
            best = r

    if best and best_score >= 0.3:
        is_closed = best.get('permanently_closed', False)
        checkins = best.get('checkins', 0)
        has_phone = bool(best.get('phone'))
        has_website = bool(best.get('website'))
        has_hours = bool(best.get('hours'))
        is_verified = best.get('is_verified', False)

        status = 'closed' if is_closed else 'verified'

        return {
            'status': status,
            'fb_name': best.get('name', ''),
            'match_score': round(best_score, 2),
            'checkins': checkins,
            'has_phone': has_phone,
            'has_website': has_website,
            'has_hours': has_hours,
            'is_verified': is_verified,
            'permanently_closed': is_closed,
        }
    return None


def process_file(path):
    """Process a single test data file."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        print("  Empty file, skipping")
        return

    found = 0
    not_found = 0
    closed_count = 0

    for i, loc in enumerate(data):
        name = loc.get('name', '')
        coords = loc.get('location', [0, 0])
        lng, lat = coords[0], coords[1]

        result = fb_places_search(name, lat, lng)
        time.sleep(0.3)  # Rate limit

        if result:
            found += 1
            if result['permanently_closed']:
                closed_count += 1
            loc['facebook'] = result
            print(f"  {i+1:3d}. {name:35s}  {result['status']:10s}  match={result['match_score']:.1f}  checkins={result['checkins']}  ({result['fb_name']})")
        else:
            not_found += 1
            loc['facebook'] = {'status': 'no_data'}
            print(f"  {i+1:3d}. {name:35s}  NOT FOUND")

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=True)

    print(f"\n  Results: {found} found ({closed_count} closed), {not_found} not found out of {len(data)}")
    return data


def main():
    if input_path:
        print(f"Processing {input_path}...")
        process_file(input_path)
    elif city:
        if city not in CITY_DATA:
            print(f"Unknown city: {city}")
            sys.exit(1)
        path = CITY_DATA[city]
        if not os.path.exists(path):
            print(f"File not found: {path}")
            sys.exit(1)
        print(f"Processing {city.upper()}...")
        process_file(path)
    else:
        # Process all cities
        for c, path in CITY_DATA.items():
            if not os.path.exists(path):
                print(f"  {c}: skipped (not found)")
                continue
            print(f"\n=== {c.upper()} ===")
            process_file(path)


if __name__ == '__main__':
    main()
