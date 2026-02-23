"""
Extract fast food locations from Overture Maps parquet data.
Outputs overture_fastfood.json with metadata for the Mapillary fetch script.
"""

import pandas as pd
import shapely.wkb
import json
import os

PARQUET_PATH = os.path.join(os.path.dirname(__file__), '..', 'pipeline', 'data', 'sf_places_large.parquet')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'overture_fastfood.json')

CHAINS = [
    'McDonald', 'Burger King', 'Taco Bell', 'KFC',
    'Jack in the Box', 'Chick-fil-A', 'Popeyes', 'Subway',
    'Panda Express', 'Five Guys', 'Chipotle', 'In-N-Out',
    "Carl's Jr", 'Wingstop', 'Domino', 'Little Caesars',
    'Jollibee', 'Shake Shack', 'Del Taco', "Church's Chicken",
]

print(f"Loading Overture places from {PARQUET_PATH}...")
df = pd.read_parquet(PARQUET_PATH)
df['geometry'] = df['geometry'].apply(lambda x: shapely.wkb.loads(x))
print(f"  Loaded {len(df)} total places")

results = []

for idx, row in df.iterrows():
    # Extract name
    try:
        name = row['names'].get('primary', '')
        if not name:
            continue
    except:
        continue

    # Check if it's a fast food chain
    name_lower = name.lower()
    brand = ''
    try:
        brand = row.get('brand', {}).get('names', {}).get('primary', '') if isinstance(row.get('brand'), dict) else ''
    except:
        pass
    brand_lower = (brand or '').lower()

    is_chain = any(
        c.lower() in name_lower or c.lower() in brand_lower
        for c in CHAINS
    )
    if not is_chain:
        continue

    # Extract coordinates
    try:
        lat = row['geometry'].y
        lng = row['geometry'].x
    except:
        continue

    # Extract category
    category = 'Fast Food'
    try:
        cats = row.get('categories')
        if cats and isinstance(cats, dict) and 'main' in cats and cats['main']:
            category = cats['main'].replace('_', ' ').title()
    except:
        pass

    # Extract address
    address = 'San Francisco, CA'
    try:
        addrs = row.get('addresses')
        if addrs is not None and len(addrs) > 0:
            addr_obj = addrs[0]
            freeform = addr_obj.get('freeform') if hasattr(addr_obj, 'get') else addr_obj['freeform']
            if freeform:
                address = f"{freeform}, San Francisco"
            else:
                number = addr_obj.get('number', '') if hasattr(addr_obj, 'get') else addr_obj.get('number', '')
                street = addr_obj.get('street', '') if hasattr(addr_obj, 'get') else addr_obj.get('street', '')
                if number and street:
                    address = f"{number} {street}, San Francisco"
                elif street:
                    address = f"{street}, San Francisco"
    except:
        pass

    # Extract confidence
    confidence = None
    try:
        confidence = row.get('confidence')
        if confidence is not None:
            confidence = round(float(confidence), 3)
    except:
        pass

    # Extract source update_time
    update_time = None
    source_name = None
    try:
        sources = row.get('sources')
        if sources is not None and len(sources) > 0:
            src = sources[0]
            ut = src.get('update_time', None) if hasattr(src, 'get') else src.get('update_time', None)
            if ut:
                update_time = ut.split('T')[0] if 'T' in str(ut) else str(ut)
            dataset = src.get('dataset', None) if hasattr(src, 'get') else src.get('dataset', None)
            if dataset:
                source_name = dataset
    except:
        pass

    # Skip indoor/mall/food-court locations — not visible from street
    addr_lower = address.lower()
    name_lower_full = name.lower()
    INDOOR_KEYWORDS = [
        'mall', 'galleria', 'food court', 'plaza area', 'lower level',
        'level b', 'suite', 'ste ', 'spc ', '#4', 'unit ',
        'inside', 'interior', 'concourse', 'terminal', 'airport',
        'westlake', 'stonestown', 'metreon',
    ]
    # Also skip non-restaurant "McDonald" matches (law offices, houses, etc.)
    NON_RESTAURANT = [
        'law office', 'house', 'entertainment', 'ronald mcdonald',
        'greenstein', 'anna mcdonald', 'allied mcdonald', 'muni',
        'central subway',
    ]
    if any(kw in addr_lower or kw in name_lower_full for kw in INDOOR_KEYWORDS):
        continue
    if any(kw in name_lower_full for kw in NON_RESTAURANT):
        continue
    # Skip if address is just "San Francisco, CA" (no street = likely inside a complex)
    if address == 'San Francisco, CA':
        continue

    results.append({
        'id': len(results) + 1,
        'name': name,
        'category': category,
        'address': address,
        'location': [lng, lat],
        'overture_meta': {
            'confidence': confidence,
            'update_time': update_time,
            'source': source_name,
            'brand': brand or None,
        }
    })

with open(OUTPUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Extracted {len(results)} fast food locations to {OUTPUT_PATH}")
for chain in CHAINS:
    count = sum(1 for r in results if chain.lower() in r['name'].lower())
    if count > 0:
        print(f"  {chain}: {count}")
