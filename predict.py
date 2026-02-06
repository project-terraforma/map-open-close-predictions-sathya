"""
Scalable Prediction Script for Open/Closed Place Prediction

Performs batch inference on new place data using the trained model.
Designed for memory-efficient processing of 100M+ places.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Optional, Iterator

from feature_engineering import extract_features_batch, get_feature_columns


def load_model(model_dir: str = 'model'):
    """Load trained model and artifacts."""
    model_path = Path(model_dir)
    
    with open(model_path / 'model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open(model_path / 'label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open(model_path / 'feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    return model, label_encoder, feature_names


def prepare_features(features_df: pd.DataFrame, label_encoder, feature_names: list) -> pd.DataFrame:
    """Prepare features for prediction."""
    # Handle primary_category encoding
    # Use transform with unknown handling
    features_df = features_df.copy()
    
    # Handle unseen categories
    known_categories = set(label_encoder.classes_)
    features_df['primary_category_clean'] = features_df['primary_category'].apply(
        lambda x: x if x in known_categories else 'unknown'
    )
    features_df['primary_category_encoded'] = label_encoder.transform(
        features_df['primary_category_clean']
    )
    
    # Convert boolean columns to int
    for col in feature_names:
        if col in features_df.columns and features_df[col].dtype == bool:
            features_df[col] = features_df[col].astype(int)
    
    # Select and order features
    X = features_df[feature_names].fillna(0)
    
    return X


def predict_batch(
    input_path: str,
    output_path: str,
    model_dir: str = 'model',
    batch_size: int = 100000,
    include_probabilities: bool = True
):
    """
    Perform batch predictions on a parquet file.
    
    Memory-efficient processing for large datasets (100M+ rows).
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to save predictions
        model_dir: Directory containing trained model
        batch_size: Number of rows to process at once
        include_probabilities: Whether to include probability scores
    """
    print(f"Loading model from {model_dir}...")
    model, label_encoder, feature_names = load_model(model_dir)
    
    print(f"Reading input file: {input_path}")
    
    # For very large files, use chunked reading
    # parquet_file = pq.ParquetFile(input_path)
    # for batch in parquet_file.iter_batches(batch_size=batch_size):
    #     df_chunk = batch.to_pandas()
    #     ...
    
    # For this example, process the full file in batches
    df = pd.read_parquet(input_path)
    total_rows = len(df)
    print(f"Total rows: {total_rows:,}")
    
    all_predictions = []
    
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx]
        
        # Extract features
        features_df = extract_features_batch(batch_df, batch_size=batch_size)
        
        # Prepare for prediction
        X = prepare_features(features_df, label_encoder, feature_names)
        
        # Predict
        predictions = model.predict(X)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'id': batch_df['id'].values,
            'predicted_open': predictions
        })
        
        if include_probabilities:
            probs = model.predict_proba(X)
            results['prob_closed'] = probs[:, 0]
            results['prob_open'] = probs[:, 1]
        
        all_predictions.append(results)
        
        print(f"Processed {end_idx:,} / {total_rows:,} rows")
    
    # Combine all results
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Save results
    predictions_df.to_parquet(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    
    # Summary
    print(f"\nPrediction Summary:")
    print(f"  Total predictions: {len(predictions_df):,}")
    print(f"  Predicted Open: {(predictions_df['predicted_open'] == 1).sum():,}")
    print(f"  Predicted Closed: {(predictions_df['predicted_open'] == 0).sum():,}")
    
    return predictions_df


def predict_single(place_data: dict, model_dir: str = 'model') -> dict:
    """
    Predict for a single place.
    
    Args:
        place_data: Dictionary with place attributes
        model_dir: Directory containing trained model
        
    Returns:
        Dictionary with prediction and probability
    """
    model, label_encoder, feature_names = load_model(model_dir)
    
    # Create single-row DataFrame
    df = pd.DataFrame([place_data])
    
    # Extract features
    from feature_engineering import extract_features
    features_df = extract_features(df)
    
    # Prepare
    X = prepare_features(features_df, label_encoder, feature_names)
    
    # Predict
    prediction = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    
    return {
        'predicted_open': int(prediction),
        'probability_closed': float(probs[0]),
        'probability_open': float(probs[1]),
        'status': 'open' if prediction == 1 else 'closed'
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict open/closed status for places')
    parser.add_argument('--input', '-i', default='project_c_samples.parquet',
                       help='Input parquet file')
    parser.add_argument('--output', '-o', default='predictions.parquet',
                       help='Output parquet file')
    parser.add_argument('--model-dir', '-m', default='model',
                       help='Model directory')
    parser.add_argument('--batch-size', '-b', type=int, default=100000,
                       help='Batch size for processing')
    
    args = parser.parse_args()
    
    predict_batch(
        input_path=args.input,
        output_path=args.output,
        model_dir=args.model_dir,
        batch_size=args.batch_size
    )
