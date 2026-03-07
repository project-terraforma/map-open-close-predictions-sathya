import pandas as pd
import numpy as np
import pickle
import sys

def get_region(lat, lon):
    if 37.0 <= lat <= 38.0 and -123.0 <= lon <= -121.5:
        return 'SF_Bay_Area'
    elif 33.5 <= lat <= 34.5 and -119.0 <= lon <= -117.5:
        return 'LA_Area'
    else:
        return 'Other'

try:
    df = pd.read_parquet('project_c_samples.parquet')
except Exception:
    df = pd.read_parquet('samples_3k_project_c_updated (1).parquet')

if 'latitude' in df.columns and 'longitude' in df.columns:
    df['region'] = df.apply(lambda row: get_region(row['latitude'], row['longitude']), axis=1)
elif 'geometry' in df.columns:
    def parse_wkb_point(wkb_bytes):
        import struct
        try:
            if wkb_bytes is None or len(wkb_bytes) < 21: return None, None
            byte_order = wkb_bytes[0]
            fmt = '<' if byte_order == 1 else '>'
            geom_type = struct.unpack(fmt + 'I', wkb_bytes[1:5])[0]
            if geom_type != 1: return None, None
            x = struct.unpack(fmt + 'd', wkb_bytes[5:13])[0]
            y = struct.unpack(fmt + 'd', wkb_bytes[13:21])[0]
            return y, x
        except: return None, None
    coords = df['geometry'].apply(parse_wkb_point)
    df['latitude'] = coords.apply(lambda c: c[0] if c else None)
    df['longitude'] = coords.apply(lambda c: c[1] if c else None)
    df['region'] = df.apply(lambda row: get_region(row['latitude'], row['longitude']) if pd.notna(row['latitude']) else 'Unknown', axis=1)
else:
    df['region'] = 'Unknown'

def has_data(x):
    if x is None: return 0
    if isinstance(x, (list, dict, str, np.ndarray)): return 1 if len(x) > 0 else 0
    return 0

if 'websites' in df.columns: df['has_website'] = df['websites'].apply(has_data)
if 'phones' in df.columns: df['has_phone'] = df['phones'].apply(has_data)
if 'sources' in df.columns: df['num_sources'] = df['sources'].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)

with open('tmp_results.txt', 'w', encoding='utf-8') as f:
    f.write("=== Region Data Sizes ===\n")
    f.write(str(df['region'].value_counts()) + "\n\n")

    target = 'label' if 'label' in df.columns else 'open'
    
    try:
        from feature_engineering import extract_features
        features_df = extract_features(df)
        with open('model/label_encoder.pkl', 'rb') as f: le = pickle.load(f)
        features_df['primary_category_encoded'] = le.transform(features_df['primary_category'].fillna('unknown'))
        with open('model/feature_names.json', 'r') as f:
            import json
            feature_names = json.load(f)
        for col in feature_names:
            if col in features_df.columns and features_df[col].dtype == bool:
                features_df[col] = features_df[col].astype(int)
        X = features_df[feature_names].fillna(0)
        with open('model/model.pkl', 'rb') as f: model = pickle.load(f)
        df['pred'] = model.predict(X)
        df['correct'] = (df['pred'] == df[target]).astype(int)
        
        f.write("=== Accuracy by Region ===\n")
        f.write(str(df.groupby('region')['correct'].mean().round(3) * 100) + "\n\n")
    except Exception as e:
        f.write(f"Could not run model predictions: {e}\n\n")
        
    f.write("=== Feature Completeness by Region ===\n")
    feats = [c for c in ['has_website', 'has_phone', 'num_sources', 'confidence'] if c in df.columns]
    f.write(str(df.groupby('region')[feats].mean().round(3)) + "\n")
