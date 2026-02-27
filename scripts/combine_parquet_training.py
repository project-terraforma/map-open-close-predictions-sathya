"""
Combine parquet files with existing Yelp training data for XGBoost training.

Parquet files contain Overture places with open/closed labels.
Extracts the same 8 metadata features used by train_xgboost.py.

Usage: python scripts/combine_parquet_training.py
Output: scripts/yelp_training_data.json (overwritten with combined data)
"""

import json
import os
import hashlib
from datetime import datetime

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
TRAIN_PATH = os.path.join(SCRIPT_DIR, 'yelp_training_data.json')
OUTPUT_PATH = TRAIN_PATH  # overwrite with combined

# Category encoding (same as fetch_yelp_training.py)
def encode_category(cat_str):
    if not cat_str:
        return 0
    return int(hashlib.md5(cat_str.encode()).hexdigest()[:4], 16) % 500


def safe_json_parse(val):
    """Parse a value that might be a JSON string or already a dict/list/ndarray."""
    import numpy as np
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except:
            return val
    return val


def is_truthy(val):
    """Check if a value is truthy, handling numpy arrays safely."""
    if val is None:
        return False
    if hasattr(val, '__len__'):
        return len(val) > 0
    return bool(val)


def extract_source_age(sources_raw):
    """Get the oldest source age in days from sources field."""
    sources = safe_json_parse(sources_raw)
    if not is_truthy(sources):
        return 0

    oldest_days = 0
    now = datetime.now()

    if isinstance(sources, list):
        for s in sources:
            s = safe_json_parse(s)
            if isinstance(s, dict):
                ut = s.get('update_time', '')
                if ut:
                    try:
                        dt = datetime.fromisoformat(ut.replace('Z', '+00:00').replace('+00:00', ''))
                        days = (now - dt).days
                        if days > oldest_days:
                            oldest_days = days
                    except:
                        pass
    return oldest_days


def extract_features_from_row(row):
    """Extract the 8 XGBoost features from a parquet row."""
    # Confidence
    confidence = float(row.get('confidence', 0) or 0)

    # Source age
    source_age_days = extract_source_age(row.get('sources'))

    # Websites
    websites = safe_json_parse(row.get('websites'))
    has_website = 0
    if is_truthy(websites):
        if isinstance(websites, list) and len(websites) > 0:
            has_website = 1
        elif isinstance(websites, str) and websites.startswith('http'):
            has_website = 1

    # Phones
    phones = safe_json_parse(row.get('phones'))
    has_phone = 0
    if is_truthy(phones):
        if isinstance(phones, list) and len(phones) > 0:
            has_phone = 1
        elif isinstance(phones, str) and len(phones) > 3:
            has_phone = 1

    # Brand
    brand = safe_json_parse(row.get('brand'))
    has_brand = 0
    if is_truthy(brand):
        if isinstance(brand, dict):
            names = brand.get('names', {})
            if isinstance(names, dict) and names.get('primary'):
                has_brand = 1
            elif isinstance(names, str) and names:
                has_brand = 1
        elif isinstance(brand, str) and brand:
            has_brand = 1

    # Address completeness
    addresses = safe_json_parse(row.get('addresses'))
    address_complete = 0
    if is_truthy(addresses) and isinstance(addresses, list) and len(addresses) > 0:
        a = safe_json_parse(addresses[0])
        if isinstance(a, dict):
            freeform = a.get('freeform', '')
            locality = a.get('locality', '')
            if freeform and locality:
                address_complete = 1

    # Category
    categories = safe_json_parse(row.get('categories'))
    primary_cat = ''
    if isinstance(categories, dict):
        primary_cat = categories.get('primary', '') or ''
    elif isinstance(categories, str):
        try:
            c = json.loads(categories)
            primary_cat = c.get('primary', '') if isinstance(c, dict) else categories
        except:
            primary_cat = categories

    # Fields populated
    fields_populated = 0
    for col in ['names', 'categories', 'confidence', 'websites', 'phones', 'brand', 'addresses', 'socials', 'emails']:
        val = row.get(col)
        if val is not None:
            try:
                parsed = safe_json_parse(val)
                if parsed is not None:
                    if hasattr(parsed, '__len__'):
                        if len(parsed) > 0:
                            fields_populated += 1
                    elif parsed != '' and parsed != 0:
                        fields_populated += 1
            except:
                pass

    # Name
    names = safe_json_parse(row.get('names'))
    name = ''
    if isinstance(names, dict):
        name = names.get('primary', '') or ''
    elif isinstance(names, str):
        try:
            n = json.loads(names)
            name = n.get('primary', '') if isinstance(n, dict) else names
        except:
            name = names

    return {
        'name': name,
        'category': primary_cat,
        'features': {
            'overture_confidence': confidence,
            'source_age_days': source_age_days,
            'has_website': has_website,
            'has_phone': has_phone,
            'has_brand': has_brand,
            'address_complete': address_complete,
            'category_encoded': encode_category(primary_cat),
            'fields_populated': fields_populated,
        }
    }


def process_parquet(filepath, label_col, label_map):
    """Process a parquet file and return training entries."""
    print(f"\n  Processing: {os.path.basename(filepath)}")
    df = pd.read_parquet(filepath)
    print(f"    Rows: {len(df)}")

    if label_col not in df.columns:
        print(f"    WARNING: No '{label_col}' column, skipping")
        return []

    entries = []
    for _, row in df.iterrows():
        label_val = row[label_col]
        if pd.isna(label_val):
            continue

        gt = label_map(label_val)
        if gt is None:
            continue

        entry = extract_features_from_row(row)
        entry['ground_truth'] = gt
        entry['source'] = os.path.basename(filepath)
        entries.append(entry)

    n_open = sum(1 for e in entries if e['ground_truth'] == 'open')
    n_closed = sum(1 for e in entries if e['ground_truth'] == 'closed')
    print(f"    Extracted: {len(entries)} ({n_open} open, {n_closed} closed)")
    return entries


def main():
    print("Combining parquet training data with existing Yelp data...")

    # Load existing training data
    existing = []
    if os.path.exists(TRAIN_PATH):
        with open(TRAIN_PATH) as f:
            existing = json.load(f)
        labeled = [d for d in existing if d.get('ground_truth')]
        n_open = sum(1 for d in labeled if d['ground_truth'] == 'open')
        n_closed = sum(1 for d in labeled if d['ground_truth'] == 'closed')
        print(f"\n  Existing data: {len(labeled)} labeled ({n_open} open, {n_closed} closed)")

    # Process parquet files
    parquet_entries = []

    # project_c_samples: 'open' column, 1=open, 0=closed
    pc_path = os.path.join(PROJECT_DIR, 'project_c_samples (1).parquet')
    if os.path.exists(pc_path):
        entries = process_parquet(pc_path, 'open', lambda v: 'open' if v == 1 else 'closed')
        parquet_entries.extend(entries)

    # samples_3k_project_c_updated: 'label' column, 1.0=open, 0.0=closed
    s3k_path = os.path.join(PROJECT_DIR, 'samples_3k_project_c_updated (1).parquet')
    if os.path.exists(s3k_path):
        entries = process_parquet(s3k_path, 'label', lambda v: 'open' if v == 1.0 else 'closed')
        parquet_entries.extend(entries)

    # project_b_samples_2k: NO label column — skip
    pb_path = os.path.join(PROJECT_DIR, 'project_b_samples_2k.parquet')
    if os.path.exists(pb_path):
        print(f"\n  Skipping {os.path.basename(pb_path)}: no label column")

    # Deduplicate by ID-like features (name + confidence combo)
    seen = set()
    unique_parquet = []
    for e in parquet_entries:
        key = f"{e['name']}|{e['features']['overture_confidence']:.6f}"
        if key not in seen:
            seen.add(key)
            unique_parquet.append(e)

    n_dupes = len(parquet_entries) - len(unique_parquet)
    if n_dupes > 0:
        print(f"\n  Removed {n_dupes} duplicates between parquet files")

    # Combine: existing (Yelp-labeled) + parquet
    # Mark existing entries with source
    for e in existing:
        if 'source' not in e:
            e['source'] = 'yelp_api'

    combined = existing + unique_parquet
    labeled = [d for d in combined if d.get('ground_truth')]
    n_open = sum(1 for d in labeled if d['ground_truth'] == 'open')
    n_closed = sum(1 for d in labeled if d['ground_truth'] == 'closed')

    print(f"\n  COMBINED TOTAL: {len(labeled)} labeled samples")
    print(f"    Open:   {n_open}")
    print(f"    Closed: {n_closed}")
    print(f"    Ratio:  {n_open/max(n_closed,1):.1f}:1")

    # Save
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(combined, f, indent=1)
    print(f"\n  Saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
