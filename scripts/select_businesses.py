"""
Select diverse businesses from Overture Maps parquet for open/closed prediction testing.
Not limited to fast food — samples restaurants, retail, services, etc.

Usage: python scripts/select_businesses.py [--city sf|la|chicago]
Output: scripts/overture_candidates_<city>.json
"""

import pandas as pd
import shapely.wkb
import json
import os
import sys
import random
from datetime import datetime

SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'pipeline', 'data')

CITY_CONFIG = {
    'sf': {
        'parquet': os.path.join(DATA_DIR, 'sf_places_large.parquet'),
        'default_city': 'San Francisco',
        'state': 'CA',
    },
    'la': {
        'parquet': os.path.join(DATA_DIR, 'la_places.parquet'),
        'default_city': 'Los Angeles',
        'state': 'CA',
    },
    'chicago': {
        'parquet': os.path.join(DATA_DIR, 'chicago_places.parquet'),
        'default_city': 'Chicago',
        'state': 'IL',
    },
}

# Parse --city arg
CITY_KEY = 'sf'
for i, arg in enumerate(sys.argv):
    if arg == '--city' and i + 1 < len(sys.argv):
        CITY_KEY = sys.argv[i + 1].lower()

if CITY_KEY not in CITY_CONFIG:
    print(f"Unknown city: {CITY_KEY}. Available: {', '.join(CITY_CONFIG.keys())}")
    sys.exit(1)

CITY = CITY_CONFIG[CITY_KEY]
PARQUET_PATH = CITY['parquet']
OUTPUT_PATH = os.path.join(SCRIPT_DIR, f'overture_candidates_{CITY_KEY}.json')

# Categories we want to sample from (diverse mix)
# These match the 'primary' field in Overture categories dict
TARGET_CATEGORIES = [
    'restaurant', 'fast_food', 'cafe', 'bar', 'coffee_shop',
    'clothing_store', 'grocery_store', 'flowers_and_gifts_shop', 'jewelry_store',
    'hair_salon', 'beauty_salon', 'laundry_service',
    'gym', 'yoga_studio',
    'pharmacy', 'dentist',
    'bakery', 'pizza_restaurant', 'chinese_restaurant', 'mexican_restaurant',
    'ice_cream_shop', 'burger_restaurant', 'japanese_restaurant',
]

INDOOR_KEYWORDS = [
    'mall', 'galleria', 'food court', 'plaza area', 'lower level',
    'level b', 'suite', 'ste ', 'spc ', 'unit ',
    'inside', 'interior', 'concourse', 'terminal', 'airport',
]

SKIP_NAMES = [
    'law office', 'house', 'entertainment', 'ronald mcdonald',
    'greenstein', 'muni', 'central subway', 'parking',
]

# Category encoding for XGBoost
CATEGORY_MAP = {}
_cat_counter = 0


def encode_category(cat_str):
    global _cat_counter
    cat_lower = (cat_str or 'unknown').lower().replace('_', ' ').strip()
    if cat_lower not in CATEGORY_MAP:
        CATEGORY_MAP[cat_lower] = _cat_counter
        _cat_counter += 1
    return CATEGORY_MAP[cat_lower]


def extract_features(row, name, address, category, confidence, update_time, brand):
    """Extract the 11 XGBoost features from an Overture row."""
    # source_age_days
    source_age_days = 0
    if update_time:
        try:
            dt = datetime.strptime(update_time, '%Y-%m-%d')
            source_age_days = (datetime.now() - dt).days
        except:
            pass

    # has_website
    has_website = False
    try:
        websites = row.get('websites')
        if websites and len(websites) > 0:
            has_website = True
    except:
        pass

    # has_phone
    has_phone = False
    try:
        phones = row.get('phones')
        if phones and len(phones) > 0:
            has_phone = True
    except:
        pass

    # has_brand
    has_brand = bool(brand)

    # address_complete
    address_complete = False
    try:
        addrs = row.get('addresses')
        if addrs and len(addrs) > 0:
            a = addrs[0]
            freeform = a.get('freeform') if hasattr(a, 'get') else a.get('freeform', '')
            locality = a.get('locality') if hasattr(a, 'get') else a.get('locality', '')
            postcode = a.get('postcode') if hasattr(a, 'get') else a.get('postcode', '')
            address_complete = bool(freeform and locality)
    except:
        pass

    # fields_populated — count non-null important fields
    fields_populated = 0
    for col in ['names', 'categories', 'confidence', 'websites', 'phones', 'brand', 'addresses', 'socials', 'emails']:
        try:
            val = row.get(col)
            if val is None:
                continue
            # Handle numpy arrays (val != [] throws ValueError on arrays)
            if hasattr(val, '__len__'):
                if len(val) > 0:
                    fields_populated += 1
            elif val != '' and val != 0:
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
        'ocr_text_match': 0,       # filled later by vision pipeline
        'image_age_days': 0,        # filled later by image fetch
        'num_images': 0,            # filled later by image fetch
    }


def main():
    print(f"Loading Overture places from {PARQUET_PATH}...")
    df = pd.read_parquet(PARQUET_PATH)
    df['geometry'] = df['geometry'].apply(lambda x: shapely.wkb.loads(x))
    print(f"  Loaded {len(df)} total places")

    candidates = []

    for idx, row in df.iterrows():
        # Extract name
        try:
            name = row['names'].get('primary', '')
            if not name or len(name) < 2:
                continue
        except:
            continue

        # Extract coordinates
        try:
            lat = row['geometry'].y
            lng = row['geometry'].x
        except:
            continue

        # Extract category — field is 'primary' in Overture schema
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

        # We want diverse businesses — check if category matches our targets
        if not cat_match:
            continue

        # Extract address
        address = f"{CITY['default_city']}, {CITY['state']}"
        try:
            addrs = row.get('addresses')
            if addrs is not None and len(addrs) > 0:
                addr_obj = addrs[0]
                freeform = addr_obj.get('freeform') if hasattr(addr_obj, 'get') else addr_obj['freeform']
                if freeform:
                    address = f"{freeform}, {CITY['default_city']}"
                else:
                    number = addr_obj.get('number', '') if hasattr(addr_obj, 'get') else ''
                    street = addr_obj.get('street', '') if hasattr(addr_obj, 'get') else ''
                    if number and street:
                        address = f"{number} {street}, {CITY['default_city']}"
                    elif street:
                        address = f"{street}, {CITY['default_city']}"
        except:
            pass

        # Skip bad addresses (only have city name, no street)
        if address == f"{CITY['default_city']}, {CITY['state']}":
            continue

        # Extract confidence
        confidence = None
        try:
            confidence = row.get('confidence')
            if confidence is not None:
                confidence = round(float(confidence), 3)
        except:
            pass

        # Extract source info
        update_time = None
        source_name = None
        try:
            sources = row.get('sources')
            if sources is not None and len(sources) > 0:
                src = sources[0]
                ut = src.get('update_time', None) if hasattr(src, 'get') else None
                if ut:
                    update_time = ut.split('T')[0] if 'T' in str(ut) else str(ut)
                dataset = src.get('dataset', None) if hasattr(src, 'get') else None
                if dataset:
                    source_name = dataset
        except:
            pass

        # Extract brand
        brand = ''
        try:
            brand = row.get('brand', {}).get('names', {}).get('primary', '') if isinstance(row.get('brand'), dict) else ''
        except:
            pass

        # Skip indoor/mall locations
        name_lower = name.lower()
        addr_lower = address.lower()
        if any(kw in addr_lower or kw in name_lower for kw in INDOOR_KEYWORDS):
            continue
        if any(kw in name_lower for kw in SKIP_NAMES):
            continue

        features = extract_features(row, name, address, category, confidence, update_time, brand)

        candidates.append({
            'name': name,
            'category': category,
            'address': address,
            'location': [lng, lat],
            'overture_meta': {
                'confidence': confidence,
                'update_time': update_time,
                'source': source_name,
                'brand': brand or None,
            },
            'overture_raw': features,
        })

    print(f"\n  {len(candidates)} candidates after filtering")

    # Sample ~80 diverse businesses across categories
    random.seed(42)
    by_category = {}
    for c in candidates:
        cat = c['category']
        by_category.setdefault(cat, []).append(c)

    print(f"\n  Categories found:")
    for cat, items in sorted(by_category.items(), key=lambda x: -len(x[1])):
        print(f"    {cat}: {len(items)}")

    # Sample proportionally, max 8 per category, target ~80 total
    selected = []
    target_total = 80
    n_cats = len(by_category)
    if n_cats == 0:
        print("  ERROR: No candidates found. Check category filter.")
        return
    per_cat = max(3, target_total // n_cats)

    for cat, items in by_category.items():
        n = min(len(items), per_cat)
        sampled = random.sample(items, n)
        selected.extend(sampled)

    # Shuffle and assign IDs
    random.shuffle(selected)
    if len(selected) > target_total:
        selected = selected[:target_total]

    for i, s in enumerate(selected):
        s['id'] = i + 1

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(selected, f, indent=2)

    print(f"\n  Selected {len(selected)} businesses -> {OUTPUT_PATH}")
    cats_selected = {}
    for s in selected:
        cats_selected[s['category']] = cats_selected.get(s['category'], 0) + 1
    for cat, count in sorted(cats_selected.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")


if __name__ == '__main__':
    main()
