"""
Download historical business data from Los Angeles Open Data.
Source: https://data.lacity.org/Administration-Finance/Listing-of-Active-Businesses/6rrh-rzua
"""
import pandas as pd
import numpy as np
import requests
import os
from io import BytesIO

def download_and_process_la_data(limit=50000, output_file='la_historical_businesses.parquet'):
    """Download LA business data and filter for older businesses."""
    
    url = "https://data.lacity.org/api/views/6rrh-rzua/rows.csv?accessType=DOWNLOAD"
    print(f"Downloading data from {url}...")
    
    # Check if we already have the raw file to save bandwidth
    raw_file = 'la_active_businesses.csv'
    
    if os.path.exists(raw_file):
        print("Using existing CSV file...")
        # Read with pandas
        df = pd.read_csv(raw_file, nrows=limit)
    else:
        print("Streaming download...")
        # Stream download standard way
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            # Read first chunk to get the idea
            # Actually better to just use pandas read_csv with url
            df = pd.read_csv(url, nrows=limit)
    
    print(f"Raw data column names: {list(df.columns)}")
    print(f"Sample data:\n{df.head(2)}")
    
    # Expected columns: 
    # 'LOCATION ACCOUNT #', 'BUSINESS NAME', 'DBA NAME', 'STREET ADDRESS', 
    # 'CITY', 'ZIP CODE', 'LOCATION DESCRIPTION', 'MAILING ADDRESS', 
    # 'MAILING CITY', 'MAILING ZIP CODE', 'NAICS', 'PRIMARY NAICS DESCRIPTION', 
    # 'COUNCIL DISTRICT', 'LOCATION START DATE', 'LOCATION END DATE', 'LOCATION'
    
    # Filter for older businesses (Start Date < 2020)
    # The date format is usually MM/DD/YYYY
    if 'LOCATION START DATE' in df.columns:
        df['start_date'] = pd.to_datetime(df['LOCATION START DATE'], errors='coerce')
        
        # Filter: Started before 2020 (so > 5 years old)
        historical_df = df[df['start_date'].dt.year < 2020].copy()
        print(f"Filtered {len(historical_df)} businesses started before 2020")
    else:
        print("Warning: LOCATION START DATE not found, using all data")
        historical_df = df.copy()
    
    # Rename columns for app compatibility
    # Need: name, category, lat, lon
    
    # Coordinates are in 'LOCATION' column usually as (lat, lon) string or separate
    # Let's check the schema. Usually it's (lat, lon) in formatted string
    
    processed_df = pd.DataFrame()
    processed_df['name'] = historical_df['DBA NAME'].fillna(historical_df['BUSINESS NAME'])
    processed_df['category'] = historical_df['PRIMARY NAICS DESCRIPTION'].fillna('Unknown')
    processed_df['address'] = historical_df['STREET ADDRESS'] + ", " + historical_df['CITY'] + ", CA " + historical_df['ZIP CODE'].astype(str)
    processed_df['city'] = historical_df['CITY']
    
    # Extract coordinates
    # The 'LOCATION' column is typically `(34.123, -118.123)`
    if 'LOCATION' in historical_df.columns:
        # Extract using regex
        coords = historical_df['LOCATION'].astype(str).str.extract(r'\((?P<lat>[\d.-]+),\s*(?P<lon>[\d.-]+)\)')
        processed_df['latitude'] = pd.to_numeric(coords['lat'], errors='coerce')
        processed_df['longitude'] = pd.to_numeric(coords['lon'], errors='coerce')
    else:
        # Sometimes there are LAT/LON columns
        pass
        
    # Drop invalid coordinates
    processed_df = processed_df.dropna(subset=['latitude', 'longitude'])
    
    # Map NAICS descriptions to our categories for the model
    # 'restaurant', 'cafe', 'grocery_store', 'pharmacy', 'gas_station', 'bank', 'hotel', 'gym', 'bar', 'bakery'
    def map_category(desc):
        desc = str(desc).lower()
        if 'restaurant' in desc or 'food' in desc or 'eating' in desc: return 'restaurant'
        if 'coffee' in desc or 'cafe' in desc: return 'cafe'
        if 'grocery' in desc or 'market' in desc: return 'grocery_store'
        if 'pharmacy' in desc or 'drug' in desc: return 'pharmacy'
        if 'gas' in desc or 'fuel' in desc: return 'gas_station'
        if 'bank' in desc or 'financial' in desc: return 'bank'
        if 'hotel' in desc or 'motel' in desc: return 'hotel'
        if 'gym' in desc or 'fitness' in desc: return 'gym'
        if 'bar' in desc or 'lounge' in desc: return 'bar'
        if 'bakery' in desc or 'baked' in desc: return 'bakery'
        return 'other'

    processed_df['mapped_category'] = processed_df['category'].apply(map_category)
    
    # Keep only mapped categories for the demo consistency
    final_df = processed_df[processed_df['mapped_category'] != 'other'].copy()
    
    print(f"Final dataset has {len(final_df)} businesses with known categories and coordinates")
    
    final_df.to_parquet(output_file, index=False)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    download_and_process_la_data(limit=50000)
