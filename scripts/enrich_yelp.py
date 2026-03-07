"""
Enrich existing Yelp data with detail endpoint fields: is_claimed, transactions, business_hours.
Uses business alias from existing yelp_url to call /v3/businesses/{alias} for each business.

Usage: python scripts/enrich_yelp.py [--city sf|la|chicago]
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
YELP_API_KEY = None
if os.path.exists(ENV_PATH):
    with open(ENV_PATH) as f:
        for line in f:
            if line.startswith('YELP_API_KEY='):
                YELP_API_KEY = line.strip().split('=', 1)[1]

if not YELP_API_KEY:
    raise RuntimeError("YELP_API_KEY not found in .env")

CITY_DATA = {
    'sf': os.path.join(PROJECT_DIR, 'src', 'data', 'test_data.json'),
    'la': os.path.join(PROJECT_DIR, 'src', 'data', 'test_data_la.json'),
    'chicago': os.path.join(PROJECT_DIR, 'src', 'data', 'test_data_chicago.json'),
    'miami': os.path.join(PROJECT_DIR, 'src', 'data', 'test_data_miami.json'),
}

# Parse args
city = None
for i, arg in enumerate(sys.argv):
    if arg == '--city' and i + 1 < len(sys.argv):
        city = sys.argv[i + 1]


def extract_alias(yelp_url):
    """Extract business alias from Yelp URL."""
    m = re.search(r'/biz/([^?]+)', yelp_url or '')
    return m.group(1) if m else None


def get_business_details(alias):
    """Get detailed business info from Yelp."""
    url = f"https://api.yelp.com/v3/businesses/{urllib.parse.quote(alias, safe='')}"
    req = urllib.request.Request(url, headers={
        'Authorization': f'Bearer {YELP_API_KEY}',
        'Accept': 'application/json',
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ''
        print(f"    HTTP {e.code}: {body[:150]}")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


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


def search_yelp(name, lat, lng):
    """Search Yelp for a business (fallback if no alias)."""
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
    except Exception as e:
        print(f"    Search error: {e}")
        return None

    best = None
    best_score = 0.0
    for biz in data.get('businesses', []):
        score = name_match_score(name, biz.get('name', ''))
        if score > best_score:
            best_score = score
            best = biz
    if best and best_score >= 0.3:
        return best.get('alias')
    return None


def process_file(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    enriched = 0
    skipped = 0
    no_yelp = 0

    for i, loc in enumerate(data):
        name = loc.get('name', '')
        yelp = loc.get('yelp', {})

        # Try to get alias from existing URL
        alias = extract_alias(yelp.get('yelp_url', ''))

        # If no alias, search for it
        if not alias and yelp.get('yelp_review_count', 0) == 0:
            coords = loc.get('location', [0, 0])
            lng, lat = coords[0], coords[1]
            alias = search_yelp(name, lat, lng)
            time.sleep(0.3)

        if not alias:
            no_yelp += 1
            print(f"  {i+1:3d}. {name:35s}  NO ALIAS")
            continue

        # Already enriched?
        if 'yelp_is_claimed' in yelp:
            skipped += 1
            print(f"  {i+1:3d}. {name:35s}  ALREADY ENRICHED")
            continue

        detail = get_business_details(alias)
        time.sleep(0.35)

        if not detail:
            no_yelp += 1
            print(f"  {i+1:3d}. {name:35s}  DETAIL FAILED")
            continue

        # Enrich yelp data
        yelp['yelp_is_claimed'] = detail.get('is_claimed', False)
        yelp['yelp_transactions'] = detail.get('transactions', [])
        hours = detail.get('hours', [{}])
        if hours:
            yelp['yelp_has_hours'] = bool(hours[0].get('open'))
        else:
            yelp['yelp_has_hours'] = False
        attrs = detail.get('attributes', {})
        yelp['yelp_temp_closed'] = attrs.get('business_temp_closed') or False

        # Update review count and rating if we got fresher data
        if detail.get('review_count'):
            yelp['yelp_review_count'] = detail['review_count']
        if detail.get('rating'):
            yelp['yelp_rating'] = detail['rating']
        if detail.get('is_closed') is not None:
            yelp['is_closed'] = detail['is_closed']

        loc['yelp'] = yelp
        enriched += 1

        claimed = "CLAIMED" if yelp['yelp_is_claimed'] else "unclaimed"
        txns = ','.join(yelp['yelp_transactions']) if yelp['yelp_transactions'] else 'none'
        hrs = "HAS_HOURS" if yelp['yelp_has_hours'] else 'no_hours'
        safe_name = name.encode('ascii', 'replace').decode()
        print(f"  {i+1:3d}. {safe_name:35s}  {claimed:10s}  txn={txns:20s}  {hrs}")

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=True)

    print(f"\n  Enriched: {enriched}, Skipped: {skipped}, No Yelp: {no_yelp}")


def main():
    if city:
        if city not in CITY_DATA:
            print(f"Unknown city: {city}")
            sys.exit(1)
        path = CITY_DATA[city]
        if not os.path.exists(path):
            print(f"File not found: {path}")
            sys.exit(1)
        print(f"Enriching {city.upper()}...")
        process_file(path)
    else:
        for c, path in CITY_DATA.items():
            if not os.path.exists(path):
                print(f"  {c}: skipped (not found)")
                continue
            print(f"\n=== {c.upper()} ===")
            process_file(path)


if __name__ == '__main__':
    main()
