
import pandas as pd
import numpy as np

def analyze():
    print("Loading data...")
    df = pd.read_parquet('samples_3k_project_c_updated.parquet')
    
    # Define helpers
    def has_data(x):
        if x is None: return 0
        if isinstance(x, (list, dict, str)): return 1 if len(x) > 0 else 0
        return 0
        
    # Add computed cols
    df['has_web'] = df['websites'].apply(has_data)
    df['has_phone'] = df['phones'].apply(has_data) 
    df['has_addr'] = df['addresses'].apply(has_data)
    
    open_df = df[df['label'] == 1.0]
    closed_df = df[df['label'] == 0.0]
    
    print(f"\nStats (Open n={len(open_df)}, Closed n={len(closed_df)}):")
    print(f"{'Metric':<10} | {'Open':<6} | {'Closed':<6}")
    print("-" * 30)
    
    metrics = [
        ('Web %', 'has_web'),
        ('Phone %', 'has_phone'),
        ('Addr %', 'has_addr')
    ]
    
    for label, col in metrics:
        o = open_df[col].mean()
        c = closed_df[col].mean()
        print(f"{label:<10} | {o:.2f}   | {c:.2f}")
        
    # Confidence
    o_conf = open_df['confidence'].mean()
    c_conf = closed_df['confidence'].mean()
    print(f"{'Conf avg':<10} | {o_conf:.2f}   | {c_conf:.2f}")

if __name__ == "__main__":
    analyze()
