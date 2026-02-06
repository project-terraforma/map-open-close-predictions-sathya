"""
Train model on Yelp Dataset which has REAL is_open labels.
This is much better training data than our synthetic signal-based approach.
"""
import pandas as pd
import numpy as np
import json
import pickle
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

def load_yelp_data(filepath='yelp_academic_dataset_business.json'):
    """Load Yelp business data from JSON Lines file."""
    print(f"Loading Yelp data from {filepath}...")
    
    businesses = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            businesses.append(json.loads(line))
    
    df = pd.DataFrame(businesses)
    print(f"Loaded {len(df)} businesses")
    print(f"Columns: {list(df.columns)}")
    print(f"\nOpen/Closed distribution:")
    print(df['is_open'].value_counts())
    
    return df

def extract_yelp_features(df):
    """Extract features from Yelp data for training."""
    print("\nExtracting features...")
    
    features = pd.DataFrame()
    features['is_open'] = df['is_open']
    
    # Star rating and review count
    features['stars'] = df['stars'].fillna(0)
    features['review_count'] = df['review_count'].fillna(0)
    features['log_review_count'] = np.log1p(features['review_count'])
    
    # Has various attributes
    features['has_address'] = df['address'].notna() & (df['address'] != '')
    features['has_city'] = df['city'].notna() & (df['city'] != '')
    features['has_postal_code'] = df['postal_code'].notna() & (df['postal_code'] != '')
    
    # Category features
    def count_categories(cats):
        if pd.isna(cats) or cats is None:
            return 0
        return len(str(cats).split(','))
    
    features['num_categories'] = df['categories'].apply(count_categories)
    features['has_categories'] = features['num_categories'] > 0
    
    # State encoding
    le_state = LabelEncoder()
    states = df['state'].fillna('Unknown')
    features['state_encoded'] = le_state.fit_transform(states)
    
    # Primary category (first one)
    def get_primary_category(cats):
        if pd.isna(cats) or cats is None:
            return 'Unknown'
        parts = str(cats).split(',')
        return parts[0].strip() if parts else 'Unknown'
    
    primary_cats = df['categories'].apply(get_primary_category)
    le_cat = LabelEncoder()
    features['primary_category_encoded'] = le_cat.fit_transform(primary_cats)
    
    # Latitude/longitude features (some areas might have more closures)
    features['latitude'] = df['latitude'].fillna(0)
    features['longitude'] = df['longitude'].fillna(0)
    features['has_coords'] = (df['latitude'].notna()) & (df['longitude'].notna())
    
    # Attributes (if present)
    def parse_attributes(attrs):
        if pd.isna(attrs) or attrs is None:
            return 0
        if isinstance(attrs, dict):
            return len(attrs)
        return 0
    
    features['num_attributes'] = df['attributes'].apply(parse_attributes)
    features['has_attributes'] = features['num_attributes'] > 0
    
    # Hours (businesses with listed hours more likely to be open)
    def has_hours(hours):
        if pd.isna(hours) or hours is None:
            return False
        if isinstance(hours, dict):
            return len(hours) > 0
        return False
    
    features['has_hours'] = df['hours'].apply(has_hours)
    
    return features, le_state, le_cat

def train_model():
    """Train model on Yelp data."""
    # Load data
    df = load_yelp_data()
    
    # Extract features
    features_df, le_state, le_cat = extract_yelp_features(df)
    
    # Prepare X and y
    y = features_df['is_open'].astype(int)
    
    feature_cols = [
        'stars', 'review_count', 'log_review_count',
        'has_address', 'has_city', 'has_postal_code',
        'num_categories', 'has_categories',
        'state_encoded', 'primary_category_encoded',
        'latitude', 'longitude', 'has_coords',
        'num_attributes', 'has_attributes',
        'has_hours'
    ]
    
    X = features_df[feature_cols].astype(float)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train LightGBM
    print("\nTraining LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        random_state=42,
        class_weight='balanced',
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Closed', 'Open']))
    
    # Feature importance
    print("\nTop 10 Feature Importances:")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance.head(10).to_string(index=False))
    
    # Save model
    print("\nSaving model...")
    os.makedirs('model', exist_ok=True)
    pickle.dump(model, open('model/model.pkl', 'wb'))
    pickle.dump(le_cat, open('model/label_encoder.pkl', 'wb'))
    
    # Save feature names for inference
    with open('model/feature_names.json', 'w') as f:
        json.dump(feature_cols, f)
    
    # Save state encoder too
    pickle.dump(le_state, open('model/state_encoder.pkl', 'wb'))
    
    print("\n✅ Model saved to model/")
    print("   - model.pkl")
    print("   - label_encoder.pkl")
    print("   - state_encoder.pkl")
    print("   - feature_names.json")

if __name__ == "__main__":
    train_model()
