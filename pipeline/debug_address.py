import pandas as pd
import numpy as np

print("Loading parquet...")
df = pd.read_parquet("pipeline/data/sf_places_large.parquet")

print(f"Columns: {df.columns}")
print("-" * 20)

# Check specific IDs
target_ids = [29972, 47508, 1812]
print(f"Checking IDs: {target_ids}")

for idx, row in df.iterrows():
    # Check by Index (which matches mock_data.json id)
    if idx in target_ids:
        print(f"Found Index {idx}:")
        print(f"  Name: {row.get('names', {}).get('primary')}")
        addrs = row['addresses']
        print(f"  Addresses: {addrs}")
        if addrs is not None and len(addrs) > 0:
             print(f"  First: {addrs[0]}")
             if isinstance(addrs[0], dict):
                 print(f"  Freeform: {addrs[0].get('freeform')}")
