import pandas as pd
import shapely.wkb

df = pd.read_parquet("pipeline/data/sf_places.parquet")
print("Columns:", df.columns)
print("\n--- Categories Sample ---")
# Drop NAs to see actual data
print(df['categories'].dropna().head(20).to_string())

print("\n--- Specific Row Drill-down ---")
sample = df['categories'].dropna().iloc[0]
print("Type:", type(sample))
print("Content:", sample)
