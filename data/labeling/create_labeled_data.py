"""
Create labeled training data with improved logic.

The label is based on BUSINESS SIGNALS that logically indicate a place is open:
- More contact info (website, phone, socials, email) → More likely OPEN
- More data sources → More likely OPEN  
- Has brand → More likely OPEN
- Complete address → More likely OPEN
- High confidence → More likely OPEN

A composite score determines the label, ensuring logical consistency.
"""
import duckdb
import pandas as pd
import numpy as np

def download_places(limit: int = 100000, output: str = "overture_large.parquet"):
    """Download places from latest Overture release."""
    print(f"Downloading {limit:,} places from Overture Maps...")
    
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    
    s3_path = "s3://overturemaps-us-west-2/release/2025-12-17.0/theme=places/type=place/*"
    
    query = f"""
    COPY (
        SELECT 
            id,
            sources,
            names,
            categories,
            confidence,
            websites,
            phones,
            socials,
            emails,
            brand,
            addresses
        FROM read_parquet('{s3_path}', filename=true, hive_partitioning=1)
        LIMIT {limit}
    ) TO '{output}' (FORMAT PARQUET);
    """
    
    try:
        con.execute(query)
        count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output}')").fetchone()[0]
        print(f"  ✅ Downloaded {count:,} records to {output}")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False
    finally:
        con.close()


def compute_openness_score(row):
    """
    Compute a score from 0-1 indicating how likely a place is OPEN.
    Based on logical business signals.
    """
    score = 0.0
    max_score = 0.0
    
    # Confidence score (weight: 3) - strongest signal
    max_score += 3.0
    if row.get('confidence') is not None:
        score += 3.0 * row['confidence']
    
    # Has website (weight: 2)
    max_score += 2.0
    websites = row.get('websites')
    if websites is not None and (isinstance(websites, list) and len(websites) > 0 or 
                                  isinstance(websites, np.ndarray) and len(websites) > 0):
        score += 2.0
    
    # Has phone (weight: 2)
    max_score += 2.0
    phones = row.get('phones')
    if phones is not None and (isinstance(phones, list) and len(phones) > 0 or
                                isinstance(phones, np.ndarray) and len(phones) > 0):
        score += 2.0
    
    # Has socials (weight: 1)
    max_score += 1.0
    socials = row.get('socials')
    if socials is not None and (isinstance(socials, list) and len(socials) > 0 or
                                 isinstance(socials, np.ndarray) and len(socials) > 0):
        score += 1.0
    
    # Has email (weight: 1)
    max_score += 1.0
    emails = row.get('emails')
    if emails is not None and (isinstance(emails, list) and len(emails) > 0 or
                                isinstance(emails, np.ndarray) and len(emails) > 0):
        score += 1.0
    
    # Number of sources (weight: up to 2)
    max_score += 2.0
    sources = row.get('sources')
    if sources is not None:
        if isinstance(sources, (list, np.ndarray)):
            num_sources = min(len(sources), 5)
            score += 2.0 * (num_sources / 5)
    
    # Has brand (weight: 1)
    max_score += 1.0
    brand = row.get('brand')
    if brand is not None and isinstance(brand, dict) and len(brand) > 0:
        score += 1.0
    
    # Has address (weight: 1)
    max_score += 1.0
    addresses = row.get('addresses')
    if addresses is not None and (isinstance(addresses, list) and len(addresses) > 0 or
                                   isinstance(addresses, np.ndarray) and len(addresses) > 0):
        score += 1.0
    
    return score / max_score if max_score > 0 else 0.5


def create_signal_based_labels(input_file: str, output_file: str):
    """Create labels based on composite business signals."""
    print("\nCreating labels based on business signals...")
    
    df = pd.read_parquet(input_file)
    print(f"  Loaded {len(df):,} records")
    
    # Compute openness score for each place
    df['openness_score'] = df.apply(compute_openness_score, axis=1)
    
    # Show distribution
    print(f"\n  Score distribution:")
    print(f"    Min: {df['openness_score'].min():.2f}")
    print(f"    Mean: {df['openness_score'].mean():.2f}")
    print(f"    Max: {df['openness_score'].max():.2f}")
    
    # Create labels using clear thresholds
    # High score (>= 0.6) → OPEN, Low score (< 0.4) → CLOSED
    # Skip ambiguous middle range for cleaner training data
    high_score = df[df['openness_score'] >= 0.6].copy()
    low_score = df[df['openness_score'] < 0.4].copy()
    
    print(f"\n  High score (≥0.6): {len(high_score):,} → OPEN")
    print(f"  Low score (<0.4): {len(low_score):,} → CLOSED")
    
    high_score['open'] = 1
    low_score['open'] = 0
    
    # Combine and shuffle
    labeled = pd.concat([high_score, low_score], ignore_index=True)
    labeled = labeled.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Remove the score column (don't want model to learn from it directly)
    labeled = labeled.drop(columns=['openness_score'])
    
    labeled.to_parquet(output_file, index=False)
    
    print(f"\n  ✅ Created {len(labeled):,} labeled samples")
    print(f"     Open: {(labeled['open'] == 1).sum():,} ({(labeled['open'] == 1).mean():.1%})")
    print(f"     Closed: {(labeled['open'] == 0).sum():,} ({(labeled['open'] == 0).mean():.1%})")
    print(f"  Saved to {output_file}")
    
    return labeled


def merge_with_original(new_file: str, original_file: str, output_file: str):
    """Merge new labeled data with original labeled data."""
    print("\nMerging with original labeled data...")
    
    new_df = pd.read_parquet(new_file)
    original_df = pd.read_parquet(original_file)
    
    # Make sure columns match
    common_cols = list(set(new_df.columns) & set(original_df.columns))
    
    combined = pd.concat([
        new_df[common_cols],
        original_df[common_cols]
    ], ignore_index=True)
    
    # Remove duplicates by ID
    combined = combined.drop_duplicates(subset=['id'], keep='first')
    
    combined.to_parquet(output_file, index=False)
    
    print(f"  ✅ Combined dataset: {len(combined):,} samples")
    print(f"     Open: {(combined['open'] == 1).sum():,}")
    print(f"     Closed: {(combined['open'] == 0).sum():,}")
    
    return combined


if __name__ == '__main__':
    print("=" * 60)
    print("Creating Training Data with Signal-Based Labels")
    print("=" * 60)
    
    # Download new data (or use existing)
    import os
    if not os.path.exists("overture_large.parquet"):
        download_places(100000, "overture_large.parquet")
    else:
        print("Using existing overture_large.parquet")
    
    # Create labels based on business signals
    create_signal_based_labels("overture_large.parquet", "overture_signal_labeled.parquet")
    
    # Merge with original data
    merge_with_original(
        "overture_signal_labeled.parquet",
        "project_c_samples.parquet",
        "training_data_combined.parquet"
    )
    
    print("\n" + "=" * 60)
    print("Done! Training data ready: training_data_combined.parquet")
    print("=" * 60)
