
import pandas as pd
import numpy as np

def analyze_data():
    # Load data
    df = pd.read_parquet('samples_3k_project_c_updated.parquet')
    
    # Helper to check if field has data
    def has_data(x):
        if x is None: return 0
        if isinstance(x, list): return 1 if len(x) > 0 else 0
        if isinstance(x, dict): return 1 if len(x) > 0 else 0
        if isinstance(x, str): return 1 if len(x) > 0 else 0
        return 0

    # Create feature columns for analysis
    df['has_website'] = df['websites'].apply(has_data)
    df['has_phone'] = df['phones'].apply(has_data)
    df['has_social'] = df['socials'].apply(lambda x: has_data(x) if 'socials' in df.columns else 0)
    df['has_address'] = df['addresses'].apply(has_data)
    df['confidence'] = df['confidence'].fillna(0.5)

    # Split into Open vs Closed
    open_df = df[df['label'] == 1.0]
    closed_df = df[df['label'] == 0.0]

    print("=== STATISTICS ===")
    print(f"Total Open: {len(open_df)}")
    print(f"Total Closed: {len(closed_df)}")
    print("\nFeature Presence (Percentage):")
    
    print(f"{'Feature':<15} | {'Open %':<10} | {'Closed %':<10} | {'Diff':<10}")
    print("-" * 55)
    
    features = ['has_website', 'has_phone', 'has_social', 'has_address']
    for feat in features:
        open_pct = open_df[feat].mean() * 100
        closed_pct = closed_df[feat].mean() * 100
        diff = open_pct - closed_pct
        print(f"{feat:<15} | {open_pct:<10.1f} | {closed_pct:<10.1f} | {diff:<10.1f}")

    print("\nConfidence Stats:")
    print(f"Open Mean Confidence: {open_df['confidence'].mean():.3f}")
    print(f"Closed Mean Confidence: {closed_df['confidence'].mean():.3f}")

if __name__ == "__main__":
    analyze_data()
