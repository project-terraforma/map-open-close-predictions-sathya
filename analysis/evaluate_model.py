"""
Model Evaluation Script for Open/Closed Place Prediction

Generates detailed evaluation metrics, visualizations, and analysis.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

from feature_engineering import extract_features, get_feature_columns


def load_artifacts(model_dir: str = 'model'):
    """Load model and related artifacts."""
    model_path = Path(model_dir)
    
    with open(model_path / 'model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open(model_path / 'label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open(model_path / 'feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    return model, label_encoder, feature_names


def plot_confusion_matrix(y_true, y_pred, output_path: str = 'confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Closed', 'Open'],
                yticklabels=['Closed', 'Open'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def plot_roc_curve(y_true, y_prob, output_path: str = 'roc_curve.png'):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved ROC curve to {output_path}")
    
    return roc_auc


def plot_precision_recall_curve(y_true, y_prob, output_path: str = 'pr_curve.png'):
    """Plot and save precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved PR curve to {output_path}")
    
    return avg_precision


def plot_feature_importance(model, feature_names, output_path: str = 'feature_importance.png'):
    """Plot and save feature importance."""
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(importance['feature'], importance['importance'], color='steelblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved feature importance to {output_path}")
    
    return importance


def analyze_errors(df, y_true, y_pred, y_prob, features_df):
    """Analyze prediction errors."""
    errors_mask = y_true != y_pred
    
    # False positives (predicted open but actually closed)
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)
    
    print("\n" + "="*60)
    print("Error Analysis")
    print("="*60)
    
    print(f"\nTotal errors: {errors_mask.sum()} / {len(y_true)} ({errors_mask.sum()/len(y_true)*100:.1f}%)")
    print(f"  False Positives (predicted open, actually closed): {fp_mask.sum()}")
    print(f"  False Negatives (predicted closed, actually open): {fn_mask.sum()}")
    
    # Analyze feature differences
    print("\n--- Feature comparison: Correct vs Incorrect Predictions ---")
    for col in ['confidence', 'num_sources', 'has_website', 'has_phone', 'has_brand']:
        if col in features_df.columns:
            correct_mean = features_df.loc[~errors_mask, col].mean()
            error_mean = features_df.loc[errors_mask, col].mean()
            print(f"  {col}: Correct={correct_mean:.3f}, Errors={error_mean:.3f}")


def main():
    """Run full evaluation."""
    print("="*60)
    print("Open/Closed Place Prediction - Model Evaluation")
    print("="*60)
    
    # Create output directory
    output_dir = Path('evaluation')
    output_dir.mkdir(exist_ok=True)
    
    # Load model and data
    print("\n1. Loading model and data...")
    model, label_encoder, feature_names = load_artifacts()
    
    df = pd.read_parquet('project_c_samples.parquet')
    print(f"   Loaded {len(df):,} samples")
    
    # Extract features
    print("\n2. Extracting features...")
    features_df = extract_features(df)
    
    # Prepare features (same as training)
    features_df['primary_category_encoded'] = label_encoder.transform(
        features_df['primary_category'].fillna('unknown')
    )
    
    for col in feature_names:
        if col in features_df.columns and features_df[col].dtype == bool:
            features_df[col] = features_df[col].astype(int)
    
    X = features_df[feature_names].fillna(0)
    y_true = df['open'].values
    
    # Predictions
    print("\n3. Generating predictions...")
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    # Metrics
    print("\n4. Computing metrics...")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Closed (0)', 'Open (1)']))
    
    # Plots
    print("\n5. Generating visualizations...")
    plot_confusion_matrix(y_true, y_pred, str(output_dir / 'confusion_matrix.png'))
    roc_auc = plot_roc_curve(y_true, y_prob, str(output_dir / 'roc_curve.png'))
    avg_precision = plot_precision_recall_curve(y_true, y_prob, str(output_dir / 'pr_curve.png'))
    importance = plot_feature_importance(model, feature_names, str(output_dir / 'feature_importance.png'))
    
    # Error analysis
    print("\n6. Analyzing errors...")
    analyze_errors(df, y_true, y_pred, y_prob, features_df)
    
    # Save evaluation results
    eval_results = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_closed': float(precision_score(y_true, y_pred, pos_label=0)),
        'recall_closed': float(recall_score(y_true, y_pred, pos_label=0)),
        'f1_closed': float(f1_score(y_true, y_pred, pos_label=0)),
        'roc_auc': float(roc_auc),
        'average_precision': float(avg_precision),
        'total_samples': int(len(y_true)),
        'total_closed': int((y_true == 0).sum()),
        'total_open': int((y_true == 1).sum()),
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # Save feature importance
    importance.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print(f"Results saved to {output_dir}/")
    print("="*60)
    
    # Print top features
    print("\nTop 5 Most Important Features:")
    print(importance.tail(5).to_string(index=False))


if __name__ == '__main__':
    main()
