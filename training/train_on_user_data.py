"""
Train model on user's labeled parquet data.
This uses the actual features from the dataset with real open/closed labels.
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import os


def extract_overture_features(df):
    """Extract features from Overture-style parquet data."""
    features = pd.DataFrame()
    
    # Confidence score (key feature)
    features['confidence'] = df['confidence'].fillna(0.5).astype(float)
    
    # Count sources
    def count_sources(x):
        if x is None: return 0
        if isinstance(x, list): return len(x)
        return 1
    features['num_sources'] = df['sources'].apply(count_sources)
    
    # Has various attributes
    def has_data(x):
        if x is None: return 0
        if isinstance(x, list): return 1 if len(x) > 0 else 0
        if isinstance(x, dict): return 1 if len(x) > 0 else 0
        return 0
    
    features['has_websites'] = df['websites'].apply(has_data)
    features['has_phones'] = df['phones'].apply(has_data)
    features['has_socials'] = df['socials'].apply(lambda x: has_data(x) if 'socials' in df.columns else 0)
    features['has_emails'] = df['emails'].apply(lambda x: has_data(x) if 'emails' in df.columns else 0)
    features['has_brand'] = df['brand'].apply(has_data) if 'brand' in df.columns else 0
    
    # Address features
    def extract_address_features(addr):
        if addr is None or (isinstance(addr, list) and len(addr) == 0):
            return {'has_address': 0, 'has_locality': 0, 'has_postcode': 0}
        
        if isinstance(addr, list):
            addr = addr[0] if len(addr) > 0 else {}
        
        if isinstance(addr, dict):
            return {
                'has_address': 1,
                'has_locality': 1 if addr.get('locality') else 0,
                'has_postcode': 1 if addr.get('postcode') else 0
            }
        return {'has_address': 0, 'has_locality': 0, 'has_postcode': 0}
    
    addr_features = df['addresses'].apply(extract_address_features).apply(pd.Series)
    features = pd.concat([features, addr_features], axis=1)
    
    # Category features
    def get_primary_category(cat):
        if cat is None: return 'unknown'
        if isinstance(cat, dict): return cat.get('primary', 'unknown')
        if isinstance(cat, str): return cat
        return 'unknown'
    
    features['primary_category'] = df['categories'].apply(get_primary_category)
    
    # Name features
    def extract_name_features(names):
        if names is None:
            return {'name_length': 0, 'has_common': 0}
        if isinstance(names, dict):
            primary = names.get('primary', '')
            return {
                'name_length': len(primary) if primary else 0,
                'has_common': 1 if names.get('common') else 0
            }
        return {'name_length': 0, 'has_common': 0}
    
    name_features = df['names'].apply(extract_name_features).apply(pd.Series)
    features = pd.concat([features, name_features], axis=1)
    
    return features


def main():
    print("Loading user's labeled data...")
    
    # Load parquet files
    dfs = []
    if os.path.exists('samples_3k_project_c_updated.parquet'):
        df1 = pd.read_parquet('samples_3k_project_c_updated.parquet')
        print(f"Loaded samples_3k_project_c_updated.parquet: {len(df1)} rows")
        dfs.append(df1)
    
    if os.path.exists('project_c_samples.parquet'):
        df2 = pd.read_parquet('project_c_samples.parquet')
        print(f"Loaded project_c_samples.parquet: {len(df2)} rows")
        dfs.append(df2)
    
    if not dfs:
        print("ERROR: No labeled parquet files found!")
        return
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total samples: {len(df)}")
    
    # Check for label column
    if 'label' not in df.columns:
        print("ERROR: No 'label' column found!")
        return
    
    # Drop rows with missing labels
    df = df.dropna(subset=['label'])
    
    # Convert labels to binary (handle float type)
    df['is_open'] = df['label'].apply(lambda x: 1 if x == 1.0 else 0)
    print(f"\nLabel distribution:")
    print(df['is_open'].value_counts())
    
    # Extract features
    print("\nExtracting features...")
    features_df = extract_overture_features(df)
    
    # Encode categories
    label_encoder = LabelEncoder()
    features_df['primary_category_encoded'] = label_encoder.fit_transform(features_df['primary_category'].fillna('unknown'))
    
    # Drop non-numeric columns
    feature_names = [c for c in features_df.columns if c != 'primary_category']
    X = features_df[feature_names].fillna(0)
    y = df['is_open']
    
    print(f"\nFeatures: {feature_names}")
    print(f"X shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\nTraining set: {len(X_train)}")
    print(f"Test set: {len(X_test)}")
    
    # Train LightGBM with balanced class weights
    print("\nTraining LightGBM model...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        num_leaves=31,
        random_state=42,
        verbose=-1,
        class_weight='balanced'  # Prevent bias toward majority class
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=== Model Performance ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Closed', 'Open']))
    
    # Feature importance
    print("\nFeature Importance:")
    importance = sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
    for feat, imp in importance:
        print(f"  {feat}: {imp:.4f}")
    
    # Save model
    os.makedirs('model', exist_ok=True)
    
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('model/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    with open('model/feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    
    print(f"\n✅ Model saved to model/ directory!")
    print(f"Feature names: {feature_names}")


if __name__ == '__main__':
    main()
