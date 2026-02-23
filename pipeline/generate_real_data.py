"""
Generate Real Map Data: Overture Places + Mapillary Street-Level Imagery

Strategy: "Street Walk Gallery"
  1. Load ALL businesses from Overture Maps (51k+ for SF)
  2. Shuffle and iterate through them (any business type)
  3. For each business, query Mapillary for nearby 360° panoramas
  4. Split images into "before" pool (closest to Overture source date) and "after" pool (most recent)
  5. From each pool, greedily pick 4 images spaced ≥15m apart along the street
  6. Save incrementally to mock_data.json
"""

import pandas as pd
import requests
import json
import os
import math
import time
from datetime import datetime
from dotenv import load_dotenv
import shapely.wkb

load_dotenv()
MAPILLARY_ACCESS_TOKEN = os.getenv("MAPILLARY_ACCESS_TOKEN")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def haversine_m(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two lat/lng points."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def fetch_mapillary_images(lat, lng, radius_deg=0.0005):
    """Fetch Mapillary images within a bounding box around lat/lng.
    Tighter bbox (0.0005 ~ 55m) for images closer to the business."""
    bbox = f"{lng-radius_deg},{lat-radius_deg},{lng+radius_deg},{lat+radius_deg}"
    params = {
        "access_token": MAPILLARY_ACCESS_TOKEN,
        "fields": "id,thumb_2048_url,captured_at,is_pano,compass_angle,geometry",
        "bbox": bbox,
        "limit": 500  # More candidates for spacing selection
    }
    for attempt in range(3):
        try:
            r = requests.get("https://graph.mapillary.com/images", params=params, timeout=30)
            if r.status_code == 200:
                return r.json().get('data', [])
            elif r.status_code == 429:
                print("  Rate limited, waiting 5s...")
                time.sleep(5)
                return []
        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            print(f"  API error after 3 attempts: {e}")
    return []

def greedy_spaced_selection(images, min_spacing_m=15, max_count=4):
    """
    Greedily select up to max_count images that are each ≥min_spacing_m apart.
    Images should already be sorted by preference (e.g. distance to business).
    Returns list of selected images.
    """
    selected = []
    for img in images:
        try:
            img_lat = img['geometry']['coordinates'][1]
            img_lng = img['geometry']['coordinates'][0]
        except:
            continue
        
        # Check spacing against all already-selected images
        too_close = False
        for sel in selected:
            sel_lat = sel['geometry']['coordinates'][1]
            sel_lng = sel['geometry']['coordinates'][0]
            if haversine_m(img_lat, img_lng, sel_lat, sel_lng) < min_spacing_m:
                too_close = True
                break
        
        if not too_close:
            selected.append(img)
            if len(selected) >= max_count:
                break
    
    return selected

def find_street_walk(images, biz_lat, biz_lng, overture_timestamp_ms=None):
    """
    Find spatially-spaced before/after galleries for a business.
    
    Returns (before_gallery, after_gallery) where each is a list of
    {url, date, distance_m} dicts, or None if insufficient images.
    """
    # Filter: images within 40m with thumbnails (pano + regular)
    valid = []
    for img in images:
        if 'thumb_2048_url' not in img:
            continue
        try:
            img_lat = img['geometry']['coordinates'][1]
            img_lng = img['geometry']['coordinates'][0]
        except:
            continue
        
        dist = haversine_m(img_lat, img_lng, biz_lat, biz_lng)
        if dist > 40:
            continue
        
        img['_dist'] = dist
        img['_lat'] = img_lat
        img['_lng'] = img_lng
        valid.append(img)
    
    if len(valid) < 3:
        return None
    
    # Sort all by capture date
    valid.sort(key=lambda x: x['captured_at'])
    
    # Determine split point between "before" and "after"
    # If we have an Overture timestamp, use it; otherwise use median date
    if overture_timestamp_ms:
        # "Before" = images within ±90 days of the Overture date (time-aligned)
        # "After" = images well after the Overture date
        window_90d_ms = 90 * 24 * 3600 * 1000
        before_pool = [img for img in valid if abs(img['captured_at'] - overture_timestamp_ms) <= window_90d_ms]
        after_pool = [img for img in valid if img['captured_at'] > overture_timestamp_ms + window_90d_ms]
        
        # Sort before-pool by temporal closeness to Overture date (best match first),
        # then by distance to business
        before_pool.sort(key=lambda x: (abs(x['captured_at'] - overture_timestamp_ms), x['_dist']))
        
        # Fallback: if pools are too small, split by median
        if len(before_pool) < 2 or len(after_pool) < 2:
            mid = len(valid) // 2
            before_pool = valid[:mid]
            after_pool = valid[mid:]
    else:
        mid = len(valid) // 2
        before_pool = valid[:mid]
        after_pool = valid[mid:]
    
    # Need at least 2 images in each pool
    if len(before_pool) < 2 or len(after_pool) < 2:
        return None
    
    # Check time gap: oldest before vs newest after must be > 14 days
    oldest_before = before_pool[0]['captured_at']
    newest_after = after_pool[-1]['captured_at']
    days_gap = (newest_after - oldest_before) / (1000 * 60 * 60 * 24)
    if days_gap < 14:
        return None
    
    # Sort each pool by distance to business (closest first) for greedy selection
    # (before_pool already sorted by time-proximity if overture_timestamp was available)
    if not overture_timestamp_ms:
        before_pool.sort(key=lambda x: x['_dist'])
    after_pool.sort(key=lambda x: x['_dist'])
    
    # Greedy spaced selection: pick 4 images ≥8m apart (tighter spacing for closer images)
    before_selected = greedy_spaced_selection(before_pool, min_spacing_m=8, max_count=4)
    after_selected = greedy_spaced_selection(after_pool, min_spacing_m=8, max_count=4)
    
    # Need at least 2 in each
    if len(before_selected) < 2 or len(after_selected) < 2:
        return None
    
    # Build gallery entries
    def to_entry(img):
        return {
            "url": img['thumb_2048_url'],
            "date": datetime.fromtimestamp(img['captured_at'] / 1000).strftime('%Y-%m-%d'),
            "distance_m": round(img['_dist'])
        }
    
    before_gallery = [to_entry(img) for img in before_selected]
    after_gallery = [to_entry(img) for img in after_selected]
    
    return (before_gallery, after_gallery)

# ============================================================
# MAIN SCRIPT
# ============================================================

print("=" * 60)
print("GENERATE REAL DATA: Street Walk Gallery + Time-Aligned Images")
print("=" * 60)

# 1. Load Overture Places
print("\n[1/3] Loading Overture Places (SF full dataset)...")
places_df = pd.read_parquet("pipeline/data/sf_places_large.parquet")
places_df['geometry'] = places_df['geometry'].apply(lambda x: shapely.wkb.loads(x))
print(f"  Loaded {len(places_df)} places")

# 2. Shuffle and process
print(f"\n[2/3] Scanning for businesses with street-walk gallery coverage...")

output_data = []
TARGET_COUNT = 50
scanned = 0
api_calls = 0

# Shuffle to get geographic diversity
shuffled = places_df.sample(frac=1, random_state=42)

for idx, row in shuffled.iterrows():
    if len(output_data) >= TARGET_COUNT:
        print(f"\n🎉 Reached target of {TARGET_COUNT} locations!")
        break
    
    scanned += 1
    if scanned % 500 == 0:
        print(f"  Progress: scanned {scanned} / {len(shuffled)} | found {len(output_data)} matches | {api_calls} API calls")
    
    # Extract business info
    try:
        lat = row['geometry'].y
        lng = row['geometry'].x
        name = row['names'].get('primary', '')
        if not name or len(name) < 2:
            continue
    except:
        continue
    
    # Extract category
    category = "Business"
    try:
        cats = row.get('categories')
        if cats and isinstance(cats, dict):
            if 'main' in cats and cats['main']:
                category = cats['main'].replace('_', ' ').title()
    except:
        pass
    
    # Extract address
    address = "San Francisco, CA"
    try:
        addrs = row.get('addresses')
        if addrs is not None and len(addrs) > 0:
            addr_obj = addrs[0]
            try:
                freeform = addr_obj.get('freeform') if hasattr(addr_obj, 'get') else addr_obj['freeform']
                if freeform:
                    address = f"{freeform}, San Francisco"
                else:
                    number = addr_obj.get('number', '') if hasattr(addr_obj, 'get') else addr_obj['number']
                    street = addr_obj.get('street', '') if hasattr(addr_obj, 'get') else addr_obj['street']
                    if number and street:
                        address = f"{number} {street}, San Francisco"
                    elif street:
                        address = f"{street}, San Francisco"
            except:
                pass
    except:
        pass

    # Extract Overture source timestamp
    overture_ts_ms = None
    overture_date_str = None
    try:
        sources = row.get('sources')
        if sources is not None and len(sources) > 0:
            src = sources[0]
            record_id = src.get('recordId', '') if hasattr(src, 'get') else src['recordId']
            # Try to get the update_time
            update_time = src.get('update_time', None) if hasattr(src, 'get') else src.get('update_time', None)
            if update_time:
                dt = datetime.fromisoformat(update_time.replace('Z', '+00:00'))
                overture_ts_ms = int(dt.timestamp() * 1000)
                overture_date_str = dt.strftime('%Y-%m-%d')
    except:
        pass

    # Skip vague names
    if name.lower() in ['parking', 'atm', 'restroom', 'elevator']:
        continue
    
    # 3. Fetch Mapillary images near this business
    images = fetch_mapillary_images(lat, lng)
    api_calls += 1
    
    if len(images) < 4:
        continue
    
    # 4. Find street walk galleries
    result = find_street_walk(images, lat, lng, overture_ts_ms)
    if result is None:
        continue
    
    before_gallery, after_gallery = result
    
    entry = {
        "id": int(idx),
        "name": name,
        "category": category,
        "address": address,
        "location": [lng, lat],
        "overture_date": overture_date_str or "Unknown",
        "before_gallery": before_gallery,
        "after_gallery": after_gallery
    }
    output_data.append(entry)
    
    # Incremental save
    with open("src/data/mock_data.json", "w") as f:
        json.dump(output_data, f, indent=2)
    
    b_count = len(before_gallery)
    a_count = len(after_gallery)
    b_date = before_gallery[0]['date']
    a_date = after_gallery[-1]['date']
    print(f"  ✅ #{len(output_data):3d} | {name[:30]:30s} | {b_count}B/{a_count}A | {b_date} → {a_date}")

print(f"\n[3/3] Done! Generated {len(output_data)} locations")
print(f"  Total scanned: {scanned}")
print(f"  Total API calls: {api_calls}")
print(f"  Output: src/data/mock_data.json")
