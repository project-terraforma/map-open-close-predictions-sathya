"""
Find closed businesses in Philadelphia using Yelp search.
Searches across neighborhoods to find businesses marked is_closed=True.

Usage: python scripts/find_closed_philly.py
"""
import json
import os
import re
import time
import random
import urllib.request
import urllib.parse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'overture_candidates_philly_closed.json')

ENV_PATH = os.path.join(PROJECT_DIR, '.env')
YELP_API_KEY = None
if os.path.exists(ENV_PATH):
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line.startswith('YELP_API_KEY='):
                YELP_API_KEY = line.split('=', 1)[1]

# Philadelphia neighborhoods with coordinates
NEIGHBORHOODS = [
    (39.9526, -75.1652, 'Center City'),
    (39.9684, -75.1340, 'Fishtown'),
    (39.9480, -75.1720, 'Rittenhouse'),
    (39.9420, -75.1630, 'South Philly'),
    (39.9570, -75.1950, 'University City'),
    (39.9830, -75.1500, 'Kensington'),
    (39.9650, -75.1800, 'Brewerytown'),
    (39.9390, -75.1500, 'Passyunk'),
    (39.9560, -75.1430, 'Old City'),
    (39.9720, -75.1600, 'Northern Liberties'),
    (39.9350, -75.1780, 'Point Breeze'),
    (39.9600, -75.1700, 'Spring Garden'),
    (39.9480, -75.1500, 'Bella Vista'),
    (39.9700, -75.1450, 'Girard Ave'),
    (39.9550, -75.1550, 'Washington Square'),
]

CATEGORIES = [
    'restaurants', 'food', 'bars', 'coffee', 'shopping',
    'beautysvc', 'health', 'fitness',
]


def yelp_search(lat, lng, category, offset=0):
    params = urllib.parse.urlencode({
        'latitude': lat,
        'longitude': lng,
        'categories': category,
        'radius': 1000,
        'limit': 50,
        'offset': offset,
        'sort_by': 'distance',
    })
    url = f"https://api.yelp.com/v3/businesses/search?{params}"
    req = urllib.request.Request(url, headers={
        'Authorization': f'Bearer {YELP_API_KEY}',
        'Accept': 'application/json',
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ERROR: {e}")
        return {'businesses': []}


def main():
    closed_businesses = []
    seen_ids = set()
    target = 40

    print(f"Searching Yelp for closed businesses in Philadelphia (target: {target})...\n")

    for lat, lng, hood in NEIGHBORHOODS:
        if len(closed_businesses) >= target:
            break

        for cat in CATEGORIES:
            if len(closed_businesses) >= target:
                break

            data = yelp_search(lat, lng, cat)
            time.sleep(0.3)

            for biz in data.get('businesses', []):
                if biz['id'] in seen_ids:
                    continue
                seen_ids.add(biz['id'])

                if biz.get('is_closed', False):
                    loc = biz.get('coordinates', {})
                    addr_parts = biz.get('location', {})
                    address = addr_parts.get('address1', '') or ''
                    city = addr_parts.get('city', 'Philadelphia')

                    if city != 'Philadelphia':
                        continue

                    entry = {
                        'name': biz['name'],
                        'category': (biz.get('categories', [{}])[0].get('title', 'Restaurant')
                                    if biz.get('categories') else 'Restaurant'),
                        'address': f"{address}, Philadelphia" if address else "Philadelphia, PA",
                        'location': [loc.get('longitude', lng), loc.get('latitude', lat)],
                        'overture_meta': {
                            'confidence': 0.9,
                            'update_time': '2026-02-18',
                            'source': 'yelp_closed_search',
                            'brand': None,
                        },
                        'overture_raw': {
                            'overture_confidence': 0.9,
                            'source_age_days': 17,
                            'has_website': 0,
                            'has_phone': 1 if biz.get('phone') else 0,
                            'has_brand': 0,
                            'address_complete': 1 if address else 0,
                            'category_closure_rate': 0.15,
                            'fields_populated': 5,
                            'image_age_days': 0,
                            'num_images': 0,
                            'yelp_rating': biz.get('rating'),
                            'yelp_review_count': biz.get('review_count', 0),
                        },
                        'yelp_prefilled': {
                            'yelp_name': biz['name'],
                            'is_closed': True,
                            'match_score': 1.0,
                            'yelp_rating': biz.get('rating'),
                            'yelp_review_count': biz.get('review_count', 0),
                            'yelp_url': biz.get('url', ''),
                        },
                        'ground_truth': 'closed',
                    }
                    closed_businesses.append(entry)
                    safe_name = biz['name'].encode('ascii', 'replace').decode()
                    print(f"  CLOSED | {safe_name:40s} | {address} | {biz.get('review_count', 0)} reviews")

                    if len(closed_businesses) >= target:
                        break

    print(f"\nFound {len(closed_businesses)} closed businesses")

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(closed_businesses, f, indent=2, ensure_ascii=True)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
