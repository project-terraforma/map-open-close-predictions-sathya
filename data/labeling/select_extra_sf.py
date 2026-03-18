"""Select 30 additional SF businesses not already in test_data.json."""
import pandas as pd
import shapely.wkb
import json
import os
import random
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'pipeline', 'data')
PARQUET_PATH = os.path.join(DATA_DIR, 'sf_places_large.parquet')

with open(os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'test_data.json'), encoding='utf-8') as f:
    existing = json.load(f)
existing_names = set(d['name'].lower() for d in existing)
print(f'Excluding {len(existing_names)} existing names')

TARGET_CATEGORIES = [
    'restaurant', 'fast_food', 'cafe', 'bar', 'coffee_shop',
    'clothing_store', 'grocery_store', 'flowers_and_gifts_shop', 'jewelry_store',
    'hair_salon', 'beauty_salon', 'laundry_service',
    'gym', 'yoga_studio', 'pharmacy', 'dentist',
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

CATEGORY_MAP = {}
_cat_counter = 0

def encode_category(cat_str):
    global _cat_counter
    cat_lower = (cat_str or 'unknown').lower().replace('_', ' ').strip()
    if cat_lower not in CATEGORY_MAP:
        CATEGORY_MAP[cat_lower] = _cat_counter
        _cat_counter += 1
    return CATEGORY_MAP[cat_lower]

df = pd.read_parquet(PARQUET_PATH)
df['geometry'] = df['geometry'].apply(lambda x: shapely.wkb.loads(x))
print(f'Loaded {len(df)} places')

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

    address = 'San Francisco, CA'
    try:
        addrs = row.get('addresses')
        if addrs is not None and len(addrs) > 0:
            addr_obj = addrs[0]
            freeform = addr_obj.get('freeform') if hasattr(addr_obj, 'get') else addr_obj['freeform']
            if freeform:
                address = f'{freeform}, San Francisco'
    except:
        pass

    if address == 'San Francisco, CA':
        continue

    name_lower = name.lower()
    addr_lower = address.lower()
    if any(kw in addr_lower or kw in name_lower for kw in INDOOR_KEYWORDS):
        continue
    if any(kw in name_lower for kw in SKIP_NAMES):
        continue

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

    source_age_days = 0
    if update_time:
        try:
            dt = datetime.strptime(update_time, '%Y-%m-%d')
            source_age_days = (datetime.now() - dt).days
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
            'category_encoded': encode_category(category),
            'fields_populated': fields_populated,
            'ocr_text_match': 0,
            'image_age_days': 0,
            'num_images': 0,
        },
    })

print(f'{len(candidates)} new candidates (excluding existing)')

random.seed(99)
by_category = {}
for c in candidates:
    by_category.setdefault(c['category'], []).append(c)

selected = []
n_cats = len(by_category)
per_cat = max(2, 30 // n_cats)
for cat, items in by_category.items():
    n = min(len(items), per_cat)
    sampled = random.sample(items, n)
    selected.extend(sampled)

random.shuffle(selected)
if len(selected) > 30:
    selected = selected[:30]

for i, s in enumerate(selected):
    s['id'] = 51 + i

OUTPUT = os.path.join(os.path.dirname(__file__), 'overture_candidates_sf_extra.json')
with open(OUTPUT, 'w') as f:
    json.dump(selected, f, indent=2)

print(f'Selected {len(selected)} extra SF businesses -> {OUTPUT}')
cats_selected = {}
for s in selected:
    cats_selected[s['category']] = cats_selected.get(s['category'], 0) + 1
for cat, count in sorted(cats_selected.items(), key=lambda x: -x[1]):
    print(f'  {cat}: {count}')
