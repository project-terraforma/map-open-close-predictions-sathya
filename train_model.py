"""
Model Training Pipeline for Open/Closed Place Prediction

Trains a LightGBM classifier to predict whether a place is open (1) or closed (0).
Handles class imbalance and optimizes for the minority class (closed places).
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from feature_engineering import extract_features, get_feature_columns


def prepare_data(df: pd.DataFrame, features_df: pd.DataFrame, target_col: str = 'open'):
    """
    Prepare features and target for model training.
    
    Args:
        df: Original DataFrame with target column
        features_df: DataFrame with extracted features
        target_col: Name of target column
        
    Returns:
        X: Feature matrix
        y: Target vector
        label_encoder: Fitted encoder for primary_category
    """
    # Encode categorical feature: primary_category
    label_encoder = LabelEncoder()
    features_df['primary_category_encoded'] = label_encoder.fit_transform(
        features_df['primary_category'].fillna('unknown')
    )
    
    # Get feature columns
    feature_cols = get_feature_columns() + ['primary_category_encoded']
    
    # Convert boolean columns to int
    for col in feature_cols:
        if col in features_df.columns and features_df[col].dtype == bool:
            features_df[col] = features_df[col].astype(int)
    
    # Fill any remaining NaN values
    X = features_df[feature_cols].fillna(0)
    y = df[target_col].values
    
    return X, y, label_encoder


def train_model(X_train, y_train, X_val=None, y_val=None, params=None):
    """
    Train LightGBM classifier with class weight balancing.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        params: Custom LightGBM parameters
        
    Returns:
        Trained LightGBM model
    """
    # Calculate scale_pos_weight for class imbalance
    # (ratio of negative class to positive class)
    n_closed = (y_train == 0).sum()
    n_open = (y_train == 1).sum()
    scale_pos_weight = n_open / n_closed if n_closed > 0 else 1.0
    
    # For predicting closed places, we want to weight the closed class higher
    # LightGBM's is_unbalance or scale_pos_weight handles this
    
    default_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 6,  # Limit depth to reduce overfitting
        'learning_rate': 0.05,
        'n_estimators': 150,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        # Use moderate class weight instead of full is_unbalance
        # With 91% open / 9% closed, a ratio of 2-3 is more reasonable
        'scale_pos_weight': 3.0,  # Moderate boost for minority class
        'random_state': 42,
        'verbose': -1,
    }
    
    if params:
        default_params.update(params)
    
    model = lgb.LGBMClassifier(**default_params)
    
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
        )
    else:
        model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X, y, dataset_name='Test'):
    """
    Evaluate model and print metrics.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: True labels
        dataset_name: Name for logging
        
    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision_closed': precision_score(y, y_pred, pos_label=0),
        'recall_closed': recall_score(y, y_pred, pos_label=0),
        'f1_closed': f1_score(y, y_pred, pos_label=0),
        'precision_open': precision_score(y, y_pred, pos_label=1),
        'recall_open': recall_score(y, y_pred, pos_label=1),
        'f1_open': f1_score(y, y_pred, pos_label=1),
    }
    
    print(f"\n{'='*50}")
    print(f"{dataset_name} Set Evaluation")
    print('='*50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"\nClosed Places (class 0):")
    print(f"  Precision: {metrics['precision_closed']:.4f}")
    print(f"  Recall: {metrics['recall_closed']:.4f}")
    print(f"  F1-Score: {metrics['f1_closed']:.4f}")
    print(f"\nOpen Places (class 1):")
    print(f"  Precision: {metrics['precision_open']:.4f}")
    print(f"  Recall: {metrics['recall_open']:.4f}")
    print(f"  F1-Score: {metrics['f1_open']:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Closed (0)', 'Open (1)']))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(f"                 Predicted")
    print(f"                 Closed  Open")
    print(f"Actual Closed    {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"       Open      {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    return metrics


def get_feature_importance(model, feature_names):
    """Get feature importance as a DataFrame."""
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    return importance


def save_model(model, label_encoder, feature_names, output_dir='model'):
    """Save model and related artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save model
    model.booster_.save_model(str(output_path / 'model.lgb'))
    
    # Also save as pickle for easier loading
    with open(output_path / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save label encoder
    with open(output_path / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save feature names
    with open(output_path / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    
    print(f"\nModel saved to {output_path}")


def main():
    """Main training pipeline."""
    print("="*60)
    print("Open/Closed Place Prediction - Model Training")
    print("="*60)
    
    # Load data - use combined dataset with more samples
    print("\n1. Loading data...")
    df = pd.read_parquet('training_data_combined.parquet')
    print(f"   Loaded {len(df):,} samples")
    print(f"   Open: {(df['open'] == 1).sum():,} | Closed: {(df['open'] == 0).sum():,}")
    
    # Extract features
    print("\n2. Extracting features...")
    features_df = extract_features(df)
    print(f"   Extracted {len(features_df.columns)} features")
    
    # Prepare data
    print("\n3. Preparing data for training...")
    X, y, label_encoder = prepare_data(df, features_df)
    feature_names = X.columns.tolist()
    print(f"   Feature matrix shape: {X.shape}")
    
    # Train/test split with stratification
    print("\n4. Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"   Train class balance - Open: {(y_train == 1).sum()} | Closed: {(y_train == 0).sum()}")
    
    # Cross-validation
    print("\n5. Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_cv = lgb.LGBMClassifier(
        objective='binary',
        is_unbalance=True,
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )
    cv_scores = cross_val_score(model_cv, X_train, y_train, cv=cv, scoring='f1')
    print(f"   CV F1 Scores: {cv_scores}")
    print(f"   Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Train final model
    print("\n6. Training final model...")
    model = train_model(X_train, y_train, X_test, y_test)
    
    # Evaluate on test set
    print("\n7. Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test, 'Test')
    
    # Feature importance
    print("\n8. Feature Importance (Top 10):")
    importance = get_feature_importance(model, feature_names)
    print(importance.head(10).to_string(index=False))
    
    # Save model
    print("\n9. Saving model...")
    save_model(model, label_encoder, feature_names)
    
    # Save metrics
    with open('model/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    return model, metrics


if __name__ == '__main__':
    model, metrics = main()
