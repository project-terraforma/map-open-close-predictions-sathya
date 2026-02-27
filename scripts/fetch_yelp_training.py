"""
Fetch Yelp open/closed labels for Overture businesses to build XGBoost training data.

Uses Yelp Business Search API to match Overture businesses by name + coordinates,
then records Yelp's is_closed field as ground truth.

Yelp free tier: 500 requests/day. Script caches results so you can resume across days.

Usage: python scripts/fetch_yelp_training.py
Input:  pipeline/data/sf_places_large.parquet
Output: scripts/yelp_training_data.json
"""

import os
import json
import time
import re
import urllib.request
import urllib.parse
import urllib.error
import pandas as pd
import shapely.wkb
from datetime import datetime

PARQUET_PATH = os.path.join(os.path.dirname(__file__), '..', 'pipeline', 'data', 'sf_places_large.parquet')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'yelp_training_data.json')
CACHE_PATH = os.path.join(os.path.dirname(__file__), 'yelp_cache.json')

# Load API key from .env
ENV_PATH = os.path.join(os.path.dirname(__file__), '..', '.env')
YELP_API_KEY = None
if os.path.exists(ENV_PATH):
    with open(ENV_PATH) as f:
        for line in f:
            if line.startswith('YELP_API_KEY='):
                YELP_API_KEY = line.strip().split('=', 1)[1]

if not YELP_API_KEY:
    raise RuntimeError("YELP_API_KEY not found in .env")

# Target categories (same as select_businesses.py)
TARGET_CATEGORIES = [
    'restaurant', 'fast_food', 'cafe', 'bar', 'coffee_shop',
    'clothing_store', 'grocery_store', 'flowers_and_gifts_shop', 'jewelry_store',
    'hair_salon', 'beauty_salon', 'laundry_service',
    'gym', 'yoga_studio',
    'pharmacy', 'dentist',
    'bakery', 'pizza_restaurant', 'chinese_restaurant', 'mexican_restaurant',
    'ice_cream_shop', 'burger_restaurant', 'japanese_restaurant',
]

# Category encoding (matches select_businesses.py)
CATEGORY_MAP = {}
_cat_counter = 0

def encode_category(cat_str):
    global _cat_counter
    cat_lower = (cat_str or 'unknown').lower().replace('_', ' ').strip()
    if cat_lower not in CATEGORY_MAP:
        CATEGORY_MAP[cat_lower] = _cat_counter
        _cat_counter += 1
    return CATEGORY_MAP[cat_lower]


def normalize(text):
    """Normalize text for comparison."""
    return re.sub(r'[^a-z0-9]', '', (text or '').lower())


def yelp_search(name, lat, lng):
    """Search Yelp for a business by name and coordinates. Returns match dict or None."""
    params = urllib.parse.urlencode({
        'term': name,
        'latitude': lat,
        'longitude': lng,
        'radius': 150,  # meters
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
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print("  RATE LIMITED — stopping. Resume tomorrow.")
            return 'RATE_LIMITED'
        print(f"  HTTP {e.code} for {name}")
        return None
    except Exception as e:
        print(f"  Error for {name}: {e}")
        return None

    businesses = data.get('businesses', [])
    if not businesses:
        return None

    # Find best name match
    norm_name = normalize(name)
    best = None
    best_score = 0

    for biz in businesses:
        biz_name = normalize(biz.get('name', ''))
        # Simple overlap score
        if norm_name == biz_name:
            score = 1.0
        elif norm_name in biz_name or biz_name in norm_name:
            score = 0.8
        else:
            # Word overlap
            words_a = set(norm_name.split()) if ' ' in name else {norm_name}
            words_b = set(biz_name.split()) if ' ' in biz.get('name', '') else {biz_name}
            overlap = len(words_a & words_b)
            score = overlap / max(len(words_a), 1) * 0.6

        if score > best_score:
            best_score = score
            best = biz

    if best and best_score >= 0.4:
        return {
            'yelp_name': best.get('name'),
            'is_closed': best.get('is_closed', False),
            'match_score': best_score,
            'yelp_rating': best.get('rating'),
            'yelp_review_count': best.get('review_count', 0),
        }
    return None


def extract_features(row, name, category, confidence, update_time, brand):
    """Extract the XGBoost features from an Overture row."""
    source_age_days = 0
    if update_time:
        try:
            dt = datetime.strptime(update_time, '%Y-%m-%d')
            source_age_days = (datetime.now() - dt).days
        except:
            pass

    has_website = False
    try:
        websites = row.get('websites')
        if websites and len(websites) > 0:
            has_website = True
    except:
        pass

    has_phone = False
    try:
        phones = row.get('phones')
        if phones and len(phones) > 0:
            has_phone = True
    except:
        pass

    has_brand = bool(brand)

    address_complete = False
    try:
        addrs = row.get('addresses')
        if addrs and len(addrs) > 0:
            a = addrs[0]
            freeform = a.get('freeform') if hasattr(a, 'get') else ''
            locality = a.get('locality') if hasattr(a, 'get') else ''
            address_complete = bool(freeform and locality)
    except:
        pass

    fields_populated = 0
    for col in ['names', 'categories', 'confidence', 'websites', 'phones', 'brand', 'addresses', 'socials', 'emails']:
        try:
            val = row.get(col)
            if val is not None and val != '' and val != []:
                fields_populated += 1
        except:
            pass

    return {
        'overture_confidence': confidence or 0,
        'source_age_days': source_age_days,
        'has_website': int(has_website),
        'has_phone': int(has_phone),
        'has_brand': int(has_brand),
        'address_complete': int(address_complete),
        'category_encoded': encode_category(category),
        'fields_populated': fields_populated,
        'ocr_text_match': 0,    # not available for training data (no images)
        'image_age_days': 0,    # not available for training data
        'num_images': 0,        # not available for training data
    }


def main():
    # Load cache
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached Yelp lookups")

    print(f"Loading Overture places from {PARQUET_PATH}...")
    df = pd.read_parquet(PARQUET_PATH)
    df['geometry'] = df['geometry'].apply(lambda x: shapely.wkb.loads(x))
    print(f"  {len(df)} total places")

    # Filter to target categories
    candidates = []
    for idx, row in df.iterrows():
        try:
            name = row['names'].get('primary', '')
            if not name or len(name) < 2:
                continue
        except:
            continue

        try:
            lat = row['geometry'].y
            lng = row['geometry'].x
        except:
            continue

        category = 'Unknown'
        cat_match = False
        try:
            cats = row.get('categories')
            if cats and isinstance(cats, dict) and 'primary' in cats and cats['primary']:
                cat_raw = cats['primary']
                category = cat_raw.replace('_', ' ').title()
                cat_lower = cat_raw.lower()
                cat_match = cat_lower in TARGET_CATEGORIES or any(t in cat_lower for t in TARGET_CATEGORIES)
        except:
            pass

        if not cat_match:
            continue

        # Address
        address = ''
        try:
            addrs = row.get('addresses')
            if addrs and len(addrs) > 0:
                addr_obj = addrs[0]
                freeform = addr_obj.get('freeform') if hasattr(addr_obj, 'get') else ''
                if freeform:
                    address = freeform
        except:
            pass

        if not address:
            continue

        confidence = None
        try:
            confidence = row.get('confidence')
            if confidence is not None:
                confidence = round(float(confidence), 3)
        except:
            pass

        update_time = None
        try:
            sources = row.get('sources')
            if sources and len(sources) > 0:
                src = sources[0]
                ut = src.get('update_time', None) if hasattr(src, 'get') else None
                if ut:
                    update_time = ut.split('T')[0] if 'T' in str(ut) else str(ut)
        except:
            pass

        brand = ''
        try:
            brand = row.get('brand', {}).get('names', {}).get('primary', '') if isinstance(row.get('brand'), dict) else ''
        except:
            pass

        features = extract_features(row, name, category, confidence, update_time, brand)

        candidates.append({
            'name': name,
            'category': category,
            'address': address,
            'lat': lat,
            'lng': lng,
            'confidence': confidence,
            'update_time': update_time,
            'brand': brand,
            'features': features,
        })

    print(f"  {len(candidates)} candidates in target categories")

    # Sort by likelihood of being closed: lower confidence + older data first
    # This biases API calls toward businesses more likely to be closed
    import random
    random.seed(42)
    for c in candidates:
        conf = c['confidence'] or 0.5
        age = c['features']['source_age_days']
        fields = c['features']['fields_populated']
        # Lower score = more likely closed (low confidence, old, few fields)
        c['_priority'] = conf * 0.4 - (age / 2000) * 0.3 - ((8 - fields) / 8) * 0.3 + random.random() * 0.2
    candidates.sort(key=lambda x: x['_priority'])
    candidates = candidates[:5000]  # take more candidates

    # Build lookup of candidates by cache key for feature extraction
    cand_by_key = {}
    for cand in candidates:
        cache_key = f"{cand['name']}|{cand['lat']:.5f}|{cand['lng']:.5f}"
        cand_by_key[cache_key] = cand

    # Query Yelp for uncached candidates
    api_calls = 0
    MAX_CALLS = 480  # stay under 500/day limit

    for i, cand in enumerate(candidates):
        cache_key = f"{cand['name']}|{cand['lat']:.5f}|{cand['lng']:.5f}"

        if cache_key in cache:
            continue  # already cached

        if api_calls >= MAX_CALLS:
            print(f"\n  Reached {MAX_CALLS} API calls. Stopping to respect rate limit.")
            print(f"  Resume tomorrow — cached results will be reused.")
            break

        yelp = yelp_search(cand['name'], cand['lat'], cand['lng'])

        if yelp == 'RATE_LIMITED':
            break

        cache[cache_key] = yelp
        api_calls += 1

        # Rate limit: ~5 requests/second
        time.sleep(0.25)

        if api_calls % 50 == 0:
            print(f"  {api_calls} API calls...")
            # Save cache periodically
            with open(CACHE_PATH, 'w') as f:
                json.dump(cache, f)

    # Save cache
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f)
    print(f"\n  Cache saved ({len(cache)} entries, {api_calls} new API calls)")

    # Extract ALL results from cache (not just current batch)
    results = []
    for cache_key, yelp in cache.items():
        if not yelp or not isinstance(yelp, dict) or 'is_closed' not in yelp:
            continue

        cand = cand_by_key.get(cache_key)
        if not cand:
            # Try to find candidate by parsing cache key
            parts = cache_key.split('|')
            if len(parts) == 3:
                name, lat_s, lng_s = parts
                try:
                    lat, lng = float(lat_s), float(lng_s)
                except ValueError:
                    continue
                # Create a minimal candidate with synthetic features
                cat = 'Unknown'
                features = {
                    'overture_confidence': 0.7,
                    'source_age_days': 200,
                    'has_website': 1,
                    'has_phone': 1,
                    'has_brand': 0,
                    'address_complete': 1,
                    'category_encoded': encode_category(cat),
                    'fields_populated': 5,
                }
                cand = {
                    'name': name, 'category': cat, 'address': '',
                    'lat': lat, 'lng': lng, 'features': features,
                }
            else:
                continue

        results.append({
            'name': cand['name'],
            'category': cand['category'],
            'address': cand.get('address', ''),
            'location': [cand['lng'], cand['lat']],
            'features': cand['features'],
            'yelp': yelp,
            'ground_truth': 'closed' if yelp['is_closed'] else 'open',
        })

    # Save results
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    # Stats
    n_open = sum(1 for r in results if r['ground_truth'] == 'open')
    n_closed = sum(1 for r in results if r['ground_truth'] == 'closed')
    print(f"\n  RESULTS: {len(results)} labeled businesses")
    print(f"    Open:   {n_open}")
    print(f"    Closed: {n_closed}")
    print(f"    Closed rate: {n_closed / max(len(results), 1) * 100:.1f}%")
    print(f"\n  Saved to {OUTPUT_PATH}")

    if api_calls < len(candidates) - len([k for k in cache]):
        print(f"\n  NOTE: {len(candidates) - len(results)} businesses still unlabeled.")
        print(f"  Run again tomorrow to continue (cached results will be reused).")


if __name__ == '__main__':
    main()
