"""
Fetch balanced open/closed training data from Yelp.

Strategy: Search Yelp across many SF neighborhoods and categories.
Yelp returns is_closed in search results, so we keep ALL closed businesses
we find and sample open ones to match.

This doesn't need Overture data — we query Yelp directly and extract
features from the Yelp response + synthetic Overture-like features.

Uses the same cache as fetch_yelp_training.py to avoid duplicate API calls.

Usage: python scripts/fetch_yelp_balanced.py
Output: scripts/yelp_training_data.json (overwritten with balanced data)
"""

import os
import json
import time
import re
import random
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'yelp_training_data.json')
CACHE_PATH = os.path.join(os.path.dirname(__file__), 'yelp_balanced_cache.json')

ENV_PATH = os.path.join(os.path.dirname(__file__), '..', '.env')
YELP_API_KEY = None
if os.path.exists(ENV_PATH):
    with open(ENV_PATH) as f:
        for line in f:
            if line.startswith('YELP_API_KEY='):
                YELP_API_KEY = line.strip().split('=', 1)[1]

if not YELP_API_KEY:
    raise RuntimeError("YELP_API_KEY not found in .env")

# SF neighborhoods with lat/lng centers — spread searches across the city
SF_LOCATIONS = [
    (37.7749, -122.4194, "Downtown"),
    (37.7849, -122.4094, "Financial District"),
    (37.7649, -122.4294, "Castro"),
    (37.7549, -122.4194, "Mission"),
    (37.7449, -122.4294, "Excelsior"),
    (37.7949, -122.3994, "North Beach"),
    (37.7849, -122.4394, "Western Addition"),
    (37.7649, -122.4594, "Sunset"),
    (37.7849, -122.4594, "Richmond"),
    (37.8049, -122.4194, "Fisherman's Wharf"),
    (37.7549, -122.3994, "Potrero Hill"),
    (37.7349, -122.4094, "Visitacion Valley"),
    (37.7749, -122.3894, "SoMa"),
    (37.7649, -122.4094, "Noe Valley"),
    (37.7949, -122.4294, "Pacific Heights"),
    (37.7449, -122.4694, "Parkside"),
    (37.7549, -122.4494, "Inner Sunset"),
    (37.7849, -122.4494, "Lone Mountain"),
    (37.7349, -122.3894, "Bayview"),
    (37.7949, -122.4494, "Presidio Heights"),
    # Bay Area expansion for more closed businesses
    (37.8044, -122.2712, "Oakland Downtown"),
    (37.8716, -122.2727, "Berkeley"),
    (37.6879, -122.4702, "Daly City"),
    (37.5585, -122.2711, "Hayward"),
    (37.3382, -121.8863, "San Jose"),
    (37.5485, -122.0574, "Fremont"),
    (37.4419, -122.1430, "Palo Alto"),
    (37.3861, -122.0839, "Sunnyvale"),
    (37.9715, -122.5311, "San Rafael"),
    (37.6547, -122.4077, "South SF"),
]

# Categories to search — high-turnover categories have more closures
YELP_CATEGORIES = [
    "restaurants", "food", "bars", "nightlife",
    "coffee", "bakeries", "pizza", "chinese",
    "mexican", "japanese", "thai", "vietnamese",
    "salons", "barbers", "beautysvc",
    "gyms", "yoga", "fitness",
    "drugstores", "dentists",
    "clothing", "jewelry", "flowers",
    "grocery", "convenience",
    "laundry", "dryclean",
    "icecream", "delis", "sandwiches",
    "burgers", "seafood", "italian",
    "korean", "indian", "mediterranean",
]

# Category encoding (matches training pipeline)
CATEGORY_MAP = {}
_cat_counter = 0

def encode_category(cat_str):
    global _cat_counter
    cat_lower = (cat_str or 'unknown').lower().replace('_', ' ').strip()
    if cat_lower not in CATEGORY_MAP:
        CATEGORY_MAP[cat_lower] = _cat_counter
        _cat_counter += 1
    return CATEGORY_MAP[cat_lower]


def yelp_search_page(term, lat, lng, offset=0, categories=None):
    """Search Yelp, returns raw list of businesses."""
    params = {
        'latitude': lat,
        'longitude': lng,
        'radius': 4000,
        'limit': 50,
        'offset': offset,
        'sort_by': 'distance',
    }
    if term:
        params['term'] = term
    if categories:
        params['categories'] = categories

    url = f"https://api.yelp.com/v3/businesses/search?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={
        'Authorization': f'Bearer {YELP_API_KEY}',
        'Accept': 'application/json',
    })

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            return 'RATE_LIMITED'
        print(f"  HTTP {e.code}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

    return data.get('businesses', [])


def biz_to_training(biz):
    """Convert a Yelp business result to a training record."""
    name = biz.get('name', '')
    coords = biz.get('coordinates', {})
    lat = coords.get('latitude', 0)
    lng = coords.get('longitude', 0)
    cats = biz.get('categories', [])
    cat_name = cats[0]['title'] if cats else 'Unknown'
    cat_alias = cats[0]['alias'] if cats else 'unknown'
    is_closed = biz.get('is_closed', False)

    addr_parts = []
    loc = biz.get('location', {})
    if loc.get('address1'):
        addr_parts.append(loc['address1'])
    if loc.get('city'):
        addr_parts.append(loc['city'])
    address = ', '.join(addr_parts)

    has_phone = bool(biz.get('phone'))
    has_url = bool(biz.get('url'))
    review_count = biz.get('review_count', 0)
    rating = biz.get('rating', 0)

    # Synthesize Overture-like features from Yelp data
    # These approximate what the Overture metadata features would look like
    features = {
        'overture_confidence': min(0.98, 0.5 + (review_count / 500) + (rating / 10)),
        'source_age_days': random.randint(30, 400),  # synthetic
        'has_website': int(has_url),
        'has_phone': int(has_phone),
        'has_brand': int(bool(biz.get('is_claimed', False))),  # is_claimed ≈ has_brand
        'address_complete': int(bool(loc.get('address1') and loc.get('city') and loc.get('zip_code'))),
        'category_encoded': encode_category(cat_alias),
        'fields_populated': sum([
            bool(name), bool(address), bool(has_phone), bool(has_url),
            bool(cats), bool(rating), bool(review_count), bool(loc.get('zip_code')),
        ]),
    }

    return {
        'name': name,
        'category': cat_name,
        'address': address,
        'location': [lng, lat],
        'features': features,
        'yelp': {
            'yelp_name': name,
            'is_closed': is_closed,
            'yelp_rating': rating,
            'yelp_review_count': review_count,
        },
        'ground_truth': 'closed' if is_closed else 'open',
    }


def main():
    # Load cache
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding='utf-8') as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached results")

    all_closed = []
    all_open = []

    # Load from cache first
    for key, biz in cache.items():
        if biz and isinstance(biz, dict):
            rec = biz_to_training(biz)
            if rec['ground_truth'] == 'closed':
                all_closed.append(rec)
            else:
                all_open.append(rec)

    print(f"From cache: {len(all_open)} open, {len(all_closed)} closed")

    # If we already have enough, skip API calls
    target_closed = 200
    target_open = 200
    need_more = len(all_closed) < target_closed

    if not need_more:
        print(f"Already have {len(all_closed)} closed businesses. Skipping API calls.")
    else:
        api_calls = 0
        MAX_CALLS = 470  # stay under 500/day

        print(f"\nSearching Yelp for closed businesses...")
        print(f"  Need: {target_closed - len(all_closed)} more closed")
        print(f"  Locations: {len(SF_LOCATIONS)}")
        print(f"  Categories: {len(YELP_CATEGORIES)}")

        # Randomize search order for variety
        search_combos = []
        for lat, lng, hood in SF_LOCATIONS:
            for cat in YELP_CATEGORIES:
                search_combos.append((lat, lng, hood, cat))
        random.seed(int(time.time()))
        random.shuffle(search_combos)

        seen_ids = set(cache.keys())

        for lat, lng, hood, cat in search_combos:
            if api_calls >= MAX_CALLS:
                print(f"\n  Hit {MAX_CALLS} API call limit. Run again tomorrow.")
                break

            if len(all_closed) >= target_closed:
                print(f"\n  Reached {target_closed} closed businesses!")
                break

            results = yelp_search_page(None, lat, lng, categories=cat)
            api_calls += 1
            time.sleep(0.22)

            if results == 'RATE_LIMITED':
                print("  RATE LIMITED. Run again tomorrow.")
                break

            if not results:
                continue

            new_closed = 0
            new_open = 0
            for biz in results:
                biz_id = biz.get('id', '')
                if biz_id in seen_ids:
                    continue
                seen_ids.add(biz_id)

                cache[biz_id] = biz
                rec = biz_to_training(biz)

                if rec['ground_truth'] == 'closed':
                    all_closed.append(rec)
                    new_closed += 1
                else:
                    all_open.append(rec)
                    new_open += 1

            if new_closed > 0:
                print(f"  [{api_calls:3d}] {hood:20s} {cat:15s} → +{new_closed} closed, +{new_open} open  (total: {len(all_closed)} closed, {len(all_open)} open)")

            # Save cache every 20 calls
            if api_calls % 20 == 0:
                with open(CACHE_PATH, 'w', encoding='utf-8') as f:
                    json.dump(cache, f, ensure_ascii=False)

        # Save cache
        with open(CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False)
        print(f"\n  Cache saved ({len(cache)} entries, {api_calls} API calls used)")

    # Build balanced dataset
    print(f"\n  Raw totals: {len(all_open)} open, {len(all_closed)} closed")

    # Deduplicate by name+location
    def dedup_key(r):
        return f"{r['name'].lower()}|{r['location'][1]:.4f}|{r['location'][0]:.4f}"

    seen = set()
    deduped_closed = []
    for r in all_closed:
        k = dedup_key(r)
        if k not in seen:
            seen.add(k)
            deduped_closed.append(r)

    deduped_open = []
    for r in all_open:
        k = dedup_key(r)
        if k not in seen:
            seen.add(k)
            deduped_open.append(r)

    print(f"  After dedup: {len(deduped_open)} open, {len(deduped_closed)} closed")

    # Balance: take all closed, sample equal number of open
    n_closed = len(deduped_closed)
    n_open_target = max(n_closed, min(len(deduped_open), target_open))

    random.seed(42)
    if len(deduped_open) > n_open_target:
        sampled_open = random.sample(deduped_open, n_open_target)
    else:
        sampled_open = deduped_open

    balanced = sampled_open + deduped_closed
    random.shuffle(balanced)

    print(f"\n  BALANCED DATASET:")
    print(f"    Open:   {len(sampled_open)}")
    print(f"    Closed: {n_closed}")
    print(f"    Total:  {len(balanced)}")
    print(f"    Ratio:  {n_closed / max(len(balanced), 1) * 100:.1f}% closed")

    # Save
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(balanced, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
