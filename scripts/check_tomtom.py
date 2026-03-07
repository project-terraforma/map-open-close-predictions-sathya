"""
Check TomTom Places API for business verification.
Searches for each business by name + coordinates and checks if it exists.

TomTom statuses:
  - verified:  Found matching business in TomTom
  - closed:    Found but marked as closed/permanently closed
  - mismatch:  Found but name doesn't match well
  - no_data:   Not found in TomTom

Usage: python scripts/check_tomtom.py [--city sf|la|chicago|miami] [--input path]
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
TOMTOM_API_KEY = None
if os.path.exists(ENV_PATH):
    with open(ENV_PATH) as f:
        for line in f:
            if line.startswith('TOMTOM_API_KEY='):
                TOMTOM_API_KEY = line.strip().split('=', 1)[1]

if not TOMTOM_API_KEY:
    raise RuntimeError("TOMTOM_API_KEY not found in .env")

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


def tomtom_search(name, lat, lng):
    """Search TomTom Places API for a business."""
    query = urllib.parse.quote(name)
    params = urllib.parse.urlencode({
        'key': TOMTOM_API_KEY,
        'lat': lat,
        'lon': lng,
        'radius': 500,
        'limit': 5,
        'categorySet': '7315',  # restaurant/eating/drinking + shops
    })
    url = f"https://api.tomtom.com/search/2/search/{query}.json?{params}"

    req = urllib.request.Request(url, headers={'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"    HTTP {e.code}")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None

    results = data.get('results', [])
    if not results:
        return None

    best = None
    best_score = 0.0

    for r in results:
        poi = r.get('poi', {})
        r_name = poi.get('name', '')
        score = name_match_score(name, r_name)

        if score > best_score:
            best_score = score
            best = r

    if best and best_score >= 0.3:
        poi = best.get('poi', {})
        addr = best.get('address', {})
        return {
            'tomtom_name': poi.get('name', ''),
            'match_score': round(best_score, 2),
            'categories': poi.get('categorySet', [{}])[0].get('name', '') if poi.get('categorySet') else '',
            'phone': poi.get('phone', ''),
            'url': poi.get('url', ''),
            'address': addr.get('freeformAddress', ''),
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

    for i, loc in enumerate(data):
        name = loc.get('name', '')
        coords = loc.get('location', [0, 0])
        lng, lat = coords[0], coords[1]

        result = tomtom_search(name, lat, lng)
        time.sleep(0.25)  # Rate limit: ~4 req/s

        if result:
            found += 1
            has_phone = bool(result.get('phone'))
            has_url = bool(result.get('url'))
            loc['tomtom'] = {
                'status': 'verified',
                'name': result['tomtom_name'],
                'match_score': result['match_score'],
                'has_phone': has_phone,
                'has_url': has_url,
                'phone': result.get('phone', ''),
                'url': result.get('url', ''),
            }
            print(f"  {i+1:3d}. {name:35s}  FOUND  match={result['match_score']:.1f}  ({result['tomtom_name']})")
        else:
            not_found += 1
            loc['tomtom'] = {'status': 'no_data'}
            print(f"  {i+1:3d}. {name:35s}  NOT FOUND")

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=True)

    print(f"\n  Results: {found} found, {not_found} not found out of {len(data)}")
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
