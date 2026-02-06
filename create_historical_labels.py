"""
Historical Comparison: Create Training Labels

Compares Overture Maps 2023 release to 2025 release to identify which places closed.
- Places in both releases → OPEN (1)
- Places only in 2023 (missing from 2025) → CLOSED (0)

This creates labeled training data automatically.
"""

import duckdb
import pandas as pd
import sys


def download_california_places(release: str, output: str, limit: int = 50000):
    """Download California places from a specific Overture release."""
    
    print(f"Downloading California places from release: {release}")
    
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    
    s3_path = f"s3://overturemaps-us-west-2/release/{release}/theme=places/type=place/*"
    
    # California bounding box
    query = f"""
    COPY (
        SELECT 
            id,
            sources,
            names,
            categories,
            confidence,
            websites,
            socials,
            emails,
            phones,
            brand,
            addresses
        FROM read_parquet('{s3_path}', filename=true, hive_partitioning=1)
        WHERE bbox.xmin >= -124.5 
          AND bbox.xmax <= -114.0
          AND bbox.ymin >= 32.5
          AND bbox.ymax <= 42.1
        LIMIT {limit}
    ) TO '{output}' (FORMAT PARQUET);
    """
    
    try:
        con.execute(query)
        result = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output}')").fetchone()
        print(f"✅ Downloaded {result[0]:,} places to {output}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        con.close()


def create_labeled_dataset(old_file: str, new_file: str, output: str):
    """
    Compare old and new datasets to create labels.
    
    - Places in both → open=1
    - Places only in old (missing from new) → open=0
    """
    
    print("\nCreating labeled dataset...")
    
    # Load both datasets
    old_df = pd.read_parquet(old_file)
    new_df = pd.read_parquet(new_file)
    
    print(f"Old dataset ({old_file}): {len(old_df):,} places")
    print(f"New dataset ({new_file}): {len(new_df):,} places")
    
    # Get IDs from new dataset
    new_ids = set(new_df['id'].tolist())
    
    # Label old places based on presence in new data
    old_df['open'] = old_df['id'].apply(lambda x: 1 if x in new_ids else 0)
    
    # Count
    open_count = (old_df['open'] == 1).sum()
    closed_count = (old_df['open'] == 0).sum()
    
    print(f"\nLabels created:")
    print(f"  Open (still exists in new data): {open_count:,}")
    print(f"  Closed (missing from new data): {closed_count:,}")
    print(f"  Closure rate: {closed_count / len(old_df):.1%}")
    
    # Save labeled dataset
    old_df.to_parquet(output, index=False)
    print(f"\n✅ Labeled dataset saved to {output}")
    
    return old_df


def main():
    # Configuration - Using stable releases that are available
    old_release = "2024-10-23.0"  # Oct 2024 stable release
    new_release = "2025-12-17.0"  # Dec 2025 stable release (14 months later)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    
    old_file = "california_old.parquet"
    new_file = "california_new.parquet"
    output_file = "california_labeled.parquet"
    
    print("=" * 60)
    print("Historical Comparison: Creating Training Labels")
    print("=" * 60)
    print(f"Comparing: {old_release} → {new_release}")
    print(f"Region: California")
    print(f"Limit: {limit:,} places per release")
    print("=" * 60)
    
    # Download old data
    print("\n[1/3] Downloading 2024 data...")
    if not download_california_places(old_release, old_file, limit):
        print("Trying 2024-04-16-beta.0 instead...")
        if not download_california_places("2024-04-16-beta.0", old_file, limit):
            print("Failed to download old data.")
            return
    
    # Download new data
    print("\n[2/3] Downloading 2025 data...")
    if not download_california_places(new_release, new_file, limit):
        print("Failed to download 2025 data")
        return
    
    # Create labels
    print("\n[3/3] Creating labeled dataset...")
    labeled_df = create_labeled_dataset(old_file, new_file, output_file)
    
    print("\n" + "=" * 60)
    print("Done! Use this data for training:")
    print(f"  python train_model.py --data {output_file}")
    print("=" * 60)


if __name__ == '__main__':
    main()
