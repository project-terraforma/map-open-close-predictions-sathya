"""
Fill each city's test data to exactly 50 open + 50 closed = 100 businesses.

Strategy:
1. Read existing test data and count open/closed
2. Select extra candidates from Overture parquet (excluding existing)
3. Run pipeline on new candidates (foursquare, images, website, vision, yelp)
4. Add new entries that fill the gap (need more of whichever group is smaller)
5. Trim excess from the over-represented group

Usage: python scripts/fill_50_50.py --city sf|la|chicago [--select-only]
"""

import json
import os
import sys
import random
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(PROJECT_DIR, 'pipeline', 'data')

CITY_CONFIG = {
    'sf': {
        'parquet': os.path.join(DATA_DIR, 'sf_places_large.parquet'),
        'test_data': os.path.join(PROJECT_DIR, 'src', 'data', 'test_data.json'),
        'default_city': 'San Francisco',
        'state': 'CA',
    },
    'la': {
        'parquet': os.path.join(DATA_DIR, 'la_places.parquet'),
        'test_data': os.path.join(PROJECT_DIR, 'src', 'data', 'test_data_la.json'),
        'default_city': 'Los Angeles',
        'state': 'CA',
    },
    'chicago': {
        'parquet': os.path.join(DATA_DIR, 'chicago_places.parquet'),
        'test_data': os.path.join(PROJECT_DIR, 'src', 'data', 'test_data_chicago.json'),
        'default_city': 'Chicago',
        'state': 'IL',
    },
    'miami': {
        'parquet': os.path.join(DATA_DIR, 'miami_places.parquet'),
        'test_data': os.path.join(PROJECT_DIR, 'src', 'data', 'test_data_miami.json'),
        'default_city': 'Miami',
        'state': 'FL',
    },
}

TARGET_PER_GROUP = 38

# Parse args
city = 'sf'
select_only = False
for i, arg in enumerate(sys.argv):
    if arg == '--city' and i + 1 < len(sys.argv):
        city = sys.argv[i + 1]
    if arg == '--select-only':
        select_only = True

if city not in CITY_CONFIG:
    print(f"Unknown city: {city}")
    sys.exit(1)

config = CITY_CONFIG[city]


def select_extra_candidates(existing_data, num_needed):
    """Select new candidates from Overture parquet, excluding existing businesses."""
    import pandas as pd
    import shapely.wkb
    from datetime import datetime

    existing_names = set(d['name'].lower() for d in existing_data)
    existing_locs = set()
    for d in existing_data:
        loc = d.get('location', [0, 0])
        existing_locs.add((round(loc[0], 4), round(loc[1], 4)))

    TARGET_CATEGORIES = [
        'restaurant', 'fast_food', 'cafe', 'bar', 'coffee_shop',
        'clothing_store', 'grocery_store', 'flowers_and_gifts_shop', 'jewelry_store',
        'hair_salon', 'beauty_salon', 'laundry_service',
        'gym', 'yoga_studio', 'pharmacy', 'dentist',
        'bakery', 'pizza_restaurant', 'chinese_restaurant', 'mexican_restaurant',
        'ice_cream_shop', 'burger_restaurant', 'japanese_restaurant',
        'sushi_restaurant', 'thai_restaurant', 'italian_restaurant',
        'nail_salon', 'dry_cleaning', 'pet_store', 'bookstore',
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

    df = pd.read_parquet(config['parquet'])
    df['geometry'] = df['geometry'].apply(lambda x: shapely.wkb.loads(x))
    print(f"  Loaded {len(df)} places from parquet")

    candidates = []
    for idx, row in df.iterrows():
        try:
            name = row['names'].get('primary', '')
            if not name or len(name) < 2:
                continue
        except:
            continue

        if name.lower() in existing_names:
            continue

        try:
            lat = row['geometry'].y
            lng = row['geometry'].x
        except:
            continue

        loc_key = (round(lng, 4), round(lat, 4))
        if loc_key in existing_locs:
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

        address = f"{config['default_city']}, {config['state']}"
        try:
            addrs = row.get('addresses')
            if addrs is not None and len(addrs) > 0:
                addr_obj = addrs[0]
                freeform = addr_obj.get('freeform') if hasattr(addr_obj, 'get') else addr_obj['freeform']
                if freeform:
                    address = f"{freeform}, {config['default_city']}"
        except:
            pass

        if address == f"{config['default_city']}, {config['state']}":
            continue

        name_lower = name.lower()
        addr_lower = address.lower()
        if any(kw in addr_lower or kw in name_lower for kw in INDOOR_KEYWORDS):
            continue
        if any(kw in name_lower for kw in SKIP_NAMES):
            continue

        # Extract metadata
        confidence = None
        try:
            confidence = row.get('confidence')
            if confidence is not None:
                confidence = round(float(confidence), 3)
        except:
            pass

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

        brand = ''
        try:
            brand = row.get('brand', {}).get('names', {}).get('primary', '') if isinstance(row.get('brand'), dict) else ''
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

        address_complete = False
        try:
            addrs2 = row.get('addresses')
            if addrs2 and len(addrs2) > 0:
                a = addrs2[0]
                fr = a.get('freeform') if hasattr(a, 'get') else ''
                lo = a.get('locality') if hasattr(a, 'get') else ''
                address_complete = bool(fr and lo)
        except:
            pass

        fields_populated = 0
        for col in ['names', 'categories', 'confidence', 'websites', 'phones', 'brand', 'addresses', 'socials', 'emails']:
            try:
                val = row.get(col)
                if val is None: continue
                if hasattr(val, '__len__'):
                    if len(val) > 0: fields_populated += 1
                elif val != '' and val != 0: fields_populated += 1
            except:
                pass

        from datetime import datetime as dt
        source_age_days = 0
        if update_time:
            try:
                d = dt.strptime(update_time, '%Y-%m-%d')
                source_age_days = (dt.now() - d).days
            except:
                pass

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
            'overture_raw': {
                'overture_confidence': confidence or 0,
                'source_age_days': source_age_days,
                'has_website': int(has_website),
                'has_phone': int(has_phone),
                'has_brand': int(bool(brand)),
                'address_complete': int(address_complete),
                'category_closure_rate': 0,
                'fields_populated': fields_populated,
            },
        })

    print(f"  Found {len(candidates)} new candidates")

    # Select more than needed (closed businesses are rarer — ~5% rate)
    target = min(len(candidates), num_needed * 15)
    random.seed(42)
    random.shuffle(candidates)
    selected = candidates[:target]

    max_id = max((d.get('id', 0) for d in existing_data), default=0)
    for i, s in enumerate(selected):
        s['id'] = max_id + 1 + i

    return selected


def run_pipeline(candidates_path):
    """Run the full pipeline on candidate businesses."""
    # Force UTF-8 output encoding for all subprocesses (Windows defaults to cp1252)
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}

    print("\n--- Running Foursquare lookup ---")
    subprocess.run(['node', os.path.join(SCRIPT_DIR, 'fetch_foursquare.mjs'), candidates_path], check=True)

    print("\n--- Fetching Mapillary images ---")
    subprocess.run(['node', os.path.join(SCRIPT_DIR, 'fetch_images.mjs'), candidates_path], check=True)

    print("\n--- Checking website liveness ---")
    subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, 'check_website_liveness.py'),
                    '--city', city, '--input', candidates_path], check=True, env=env)

    print("\n--- Running vision analysis ---")
    subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, 'run_vision.py'),
                    '--input', candidates_path], check=True, env=env)

    print("\n--- Verifying with Yelp ---")
    subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, 'verify_test_yelp.py'),
                    '--input', candidates_path], check=True, env=env)


def main():
    test_data_path = config['test_data']
    with open(test_data_path, encoding='utf-8') as f:
        existing = json.load(f)

    n_open = sum(1 for d in existing if d.get('ground_truth') == 'open')
    n_closed = sum(1 for d in existing if d.get('ground_truth') == 'closed')
    print(f"\n{city.upper()}: {len(existing)} total, {n_open} open, {n_closed} closed")
    print(f"Target: {TARGET_PER_GROUP} open + {TARGET_PER_GROUP} closed = {TARGET_PER_GROUP * 2}")

    need_open = max(0, TARGET_PER_GROUP - n_open)
    need_closed = max(0, TARGET_PER_GROUP - n_closed)
    total_needed = need_open + need_closed

    if total_needed == 0:
        print("Already at target! Trimming if needed...")
    else:
        print(f"Need {need_open} more open, {need_closed} more closed")
        print(f"Selecting ~{total_needed * 4} extra candidates (4x buffer for closed rarity)...")

        candidates = select_extra_candidates(existing, total_needed)

        # Save candidates
        candidates_path = os.path.join(SCRIPT_DIR, f'overture_candidates_{city}_fill.json')
        with open(candidates_path, 'w', encoding='utf-8') as f:
            json.dump(candidates, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(candidates)} candidates to {candidates_path}")

        if select_only:
            print("\n--select-only: Stopping after candidate selection.")
            print(f"Run pipeline manually, then re-run without --select-only to merge.")
            return

        # Run full pipeline
        run_pipeline(candidates_path)

        # Reload processed candidates
        with open(candidates_path, encoding='utf-8') as f:
            processed = json.load(f)

        # Separate by ground truth
        new_open = [d for d in processed if d.get('ground_truth') == 'open']
        new_closed = [d for d in processed if d.get('ground_truth') == 'closed']
        print(f"\nNew candidates after Yelp: {len(new_open)} open, {len(new_closed)} closed")

        # Add what we need
        random.shuffle(new_open)
        random.shuffle(new_closed)

        to_add = []
        if need_open > 0:
            to_add.extend(new_open[:need_open])
        if need_closed > 0:
            to_add.extend(new_closed[:need_closed])

        existing.extend(to_add)
        print(f"Added {len(to_add)} new entries")

    # Re-count
    n_open = sum(1 for d in existing if d.get('ground_truth') == 'open')
    n_closed = sum(1 for d in existing if d.get('ground_truth') == 'closed')
    print(f"After adding: {len(existing)} total, {n_open} open, {n_closed} closed")

    # Trim excess
    if n_open > TARGET_PER_GROUP:
        open_entries = [d for d in existing if d.get('ground_truth') == 'open']
        closed_entries = [d for d in existing if d.get('ground_truth') == 'closed']
        other = [d for d in existing if d.get('ground_truth') not in ('open', 'closed')]
        random.shuffle(open_entries)
        open_entries = open_entries[:TARGET_PER_GROUP]
        existing = open_entries + closed_entries + other
        print(f"Trimmed open to {TARGET_PER_GROUP}")

    if n_closed > TARGET_PER_GROUP:
        open_entries = [d for d in existing if d.get('ground_truth') == 'open']
        closed_entries = [d for d in existing if d.get('ground_truth') == 'closed']
        other = [d for d in existing if d.get('ground_truth') not in ('open', 'closed')]
        random.shuffle(closed_entries)
        closed_entries = closed_entries[:TARGET_PER_GROUP]
        existing = open_entries + closed_entries + other
        print(f"Trimmed closed to {TARGET_PER_GROUP}")

    # Reassign IDs
    for i, d in enumerate(existing):
        d['id'] = i + 1

    # Save
    with open(test_data_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    n_open = sum(1 for d in existing if d.get('ground_truth') == 'open')
    n_closed = sum(1 for d in existing if d.get('ground_truth') == 'closed')
    print(f"\nFINAL: {len(existing)} total, {n_open} open, {n_closed} closed")
    print(f"Saved to {test_data_path}")


if __name__ == '__main__':
    main()
