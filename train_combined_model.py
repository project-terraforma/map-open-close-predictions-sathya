"""
Merge all datasets and retrain the model.
Datasets:
1. samples_3k_project_c_updated.parquet (New user file, label col)
2. project_c_samples.parquet (Original user file, open col)
3. overture_signal_labeled.parquet (My 100k synthetic signal-based labels)
"""
import pandas as pd
import numpy as np
import os
import pickle
import json
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from feature_engineering import extract_features

def merge_datasets():
    print("Merging datasets...")
    
    # 1. New Data
    df_new = pd.read_parquet('samples_3k_project_c_updated.parquet')
    if 'label' in df_new.columns:
        df_new = df_new.rename(columns={'label': 'open'})
    print(f"  New data: {len(df_new)} samples")
    
    # 2. Original Data
    df_orig = pd.read_parquet('project_c_samples.parquet')
    print(f"  Original data: {len(df_orig)} samples")
    
    # 3. Synthetic Data
    df_synth = pd.read_parquet('overture_signal_labeled.parquet')
    print(f"  Synthetic data: {len(df_synth)} samples")
    
    # Combine
    # Ensure they have common columns
    common_cols = list(set(df_new.columns) & set(df_orig.columns) & set(df_synth.columns))
    print(f"  Common columns: {len(common_cols)}")
    
    df_final = pd.concat([df_new, df_orig, df_synth], ignore_index=True)
    
    # Remove duplicates
    df_final = df_final.drop_duplicates(subset=['id'])
    
    print(f"  Total unique samples: {len(df_final)}")
    print(f"  Class balance: Open={df_final['open'].mean():.1%}")
    
    return df_final

def train_model(df):
    print("\nExtracting features...")
    # Extract features
    features_df = extract_features(df)
    
    # Prepare X and y
    y = df['open'].astype(int)
    
    # Feature selection
    feature_names = [
        'confidence',
        'num_sources', 
        'has_website', 'has_phone', 'has_socials', 'has_email',
        'has_brand', 'has_meta_source',
        'primary_category_encoded',
        'address_completeness'
    ]
    
    # Encode Categories using full dataset
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # Fit on all known categories including new ones
    all_cats = features_df['primary_category'].unique()
    le.fit(all_cats)
    features_df['primary_category_encoded'] = le.transform(features_df['primary_category'])
    
    X = features_df[feature_names].fillna(0)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)} samples...")
    
    # Train
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    # Save
    print("\nSaving model...")
    os.makedirs('model', exist_ok=True)
    pickle.dump(model, open('model/model.pkl', 'wb'))
    pickle.dump(le, open('model/label_encoder.pkl', 'wb'))
    with open('model/feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    
    print("Done!")

if __name__ == "__main__":
    df = merge_datasets()
    train_model(df)
