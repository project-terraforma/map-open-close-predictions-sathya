import overturemaps
import pandas as pd
import shapely.wkb

def ingest_sf_data():
    # Bounding Box for ALL of San Francisco
    # South: 37.70 (Daly City border)
    # North: 37.81 (Golden Gate)
    # West: -122.52 (Ocean Beach)
    # East: -122.36 (Bay Bridge)
    bbox = (-122.52, 37.70, -122.36, 37.81)
    
    print(f"Fetching Overture Places for bbox: {bbox}")
    
    table = overturemaps.record_batch_reader("place", bbox=bbox).read_all()
    df = table.to_pandas()
    
    print(f"Downloaded {len(df)} places.")
    
    # Save to parquet
    output_path = "pipeline/data/sf_places_large.parquet"
    df.to_parquet(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    ingest_sf_data()
