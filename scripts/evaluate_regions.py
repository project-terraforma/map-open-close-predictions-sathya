import pandas as pd
import numpy as np
import pickle
import os
import sys

# Ensure feature_engineering is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from feature_engineering import extract_features, get_feature_columns
from sklearn.metrics import accuracy_score

def evaluate():
    print("Loading dataset...")
    try:
        df = pd.read_parquet('samples_3k_project_c_updated.parquet')
        df = df.rename(columns={'label': 'open'})
    except Exception:
        df = pd.read_parquet('project_c_samples.parquet')
        if 'label' in df.columns: df = df.rename(columns={'label': 'open'})
    
    print("Extracting features (this might take a moment)...")
    feat_df = extract_features(df)
    
    print("Loading model...")
    model_dir = 'model'
    with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
        
    # Categorical encoding safe check
    cat_col = feat_df['primary_category'].fillna('unknown')
    
    # Map any unknown categories to a known one, or 'unknown' if available in classes
    safe_cats = []
    classes = set(le.classes_)
    default_cat = 'unknown' if 'unknown' in classes else le.classes_[0]
    for c in cat_col:
        safe_cats.append(c if c in classes else default_cat)
        
    feat_df['primary_category_encoded'] = le.transform(safe_cats)
    
    feature_names = get_feature_columns() + ['primary_category_encoded']
    
    for col in feature_names:
        if col in feat_df.columns and feat_df[col].dtype == bool:
            feat_df[col] = feat_df[col].astype(int)
            
    X = feat_df[feature_names].fillna(0)
    y_true = df['open'].values
    
    print("Predicting...")
    y_pred = model.predict(X)
    
    feat_df['true_label'] = y_true
    feat_df['pred_label'] = y_pred
    feat_df['correct'] = (feat_df['true_label'] == feat_df['pred_label'])
    
    # Target cities for evaluation
    target_cities = ['San Francisco', 'Los Angeles', 'Chicago', 'New York']
    
    print("\n--- Accuracy by City ---")
    results = []
    for city in target_cities:
        city_mask = feat_df['locality'].str.contains(city, case=False, na=False)
        city_df = feat_df[city_mask]
        
        if len(city_df) > 0:
            acc = city_df['correct'].mean() * 100
            results.append({'City': city, 'Accuracy': f"{acc:.1f}%", 'Samples': len(city_df)})
            print(f"{city}: {acc:.1f}% (n={len(city_df)})")
        else:
            print(f"{city}: No data found")
            
    # Also overall accuracy
    overall_acc = accuracy_score(y_true, y_pred) * 100
    print(f"\nOverall Accuracy: {overall_acc:.1f}%")

if __name__ == '__main__':
    evaluate()
