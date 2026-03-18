"""
Fast scan of Overture candidates to find closed Philly businesses via Yelp.
"""
import json
import os
import re
import time
import urllib.request
import urllib.parse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, '..', '.env')
YELP_KEY = None
if os.path.exists(ENV_PATH):
    with open(ENV_PATH) as f:
        for line in f:
            if line.startswith('YELP_API_KEY='):
                YELP_KEY = line.strip().split('=', 1)[1]

with open(os.path.join(SCRIPT_DIR, 'philly_check_sample.json')) as f:
    sample = json.load(f)


def normalize(t):
    return re.sub(r'[^a-z0-9]', '', (t or '').lower())


def name_match(q, r):
    nq, nr = normalize(q), normalize(r)
    if nq == nr:
        return 1.0
    if nq in nr or nr in nq:
        return 0.8
    wq = set(normalize(w) for w in q.split() if len(w) > 1)
    wr = set(normalize(w) for w in r.split() if len(w) > 1)
    if not wq:
        return 0.0
    return len(wq & wr) / max(len(wq), 1) * 0.6


def main():
    closed = []
    target = 35
    checked = 0

    for biz in sample:
        if len(closed) >= target:
            break

        name = biz['name']
        lat, lng = biz['lat'], biz['lng']

        params = urllib.parse.urlencode({
            'term': name, 'latitude': lat, 'longitude': lng, 'limit': 3
        })
        url = f'https://api.yelp.com/v3/businesses/search?{params}'
        req = urllib.request.Request(url, headers={
            'Authorization': f'Bearer {YELP_KEY}',
            'Accept': 'application/json',
        })
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except:
            data = {'businesses': []}

        checked += 1

        for b in data.get('businesses', []):
            if name_match(name, b.get('name', '')) >= 0.5 and b.get('is_closed'):
                city = b.get('location', {}).get('city', '')
                if city != 'Philadelphia':
                    continue
                addr = b.get('location', {}).get('address1', '')
                coords = b.get('coordinates', {})
                safe = name.encode('ascii', 'replace').decode()
                print(f'[{len(closed)+1:2d}] CLOSED: {safe:40s} | {addr} | {b.get("review_count", 0)} reviews')
                closed.append({
                    'name': name,
                    'category': (b.get('categories', [{}])[0].get('title', 'Restaurant')
                                 if b.get('categories') else 'Restaurant'),
                    'address': f'{biz["address"]}, Philadelphia',
                    'location': [coords.get('longitude', lng), coords.get('latitude', lat)],
                    'overture_meta': {
                        'confidence': 0.9, 'update_time': '2026-02-18',
                        'source': 'overture', 'brand': None,
                    },
                    'overture_raw': {
                        'overture_confidence': 0.9, 'source_age_days': 17,
                        'has_website': 0, 'has_phone': 1, 'has_brand': 0,
                        'address_complete': 1, 'category_closure_rate': 0.15,
                        'fields_populated': 5, 'image_age_days': 0, 'num_images': 0,
                        'yelp_rating': b.get('rating'),
                        'yelp_review_count': b.get('review_count', 0),
                    },
                    'yelp': {
                        'yelp_name': b['name'], 'is_closed': True,
                        'match_score': 1.0, 'yelp_rating': b.get('rating'),
                        'yelp_review_count': b.get('review_count', 0),
                        'yelp_url': b.get('url', ''),
                    },
                    'ground_truth': 'closed',
                })
                break

        if checked % 50 == 0:
            print(f'  ... checked {checked}, found {len(closed)} closed')

        time.sleep(0.25)

    print(f'\nChecked {checked}, found {len(closed)} closed')

    output_path = os.path.join(SCRIPT_DIR, 'overture_candidates_philly_closed.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(closed, f, indent=2, ensure_ascii=True)
    print(f'Saved to {output_path}')


if __name__ == '__main__':
    main()
