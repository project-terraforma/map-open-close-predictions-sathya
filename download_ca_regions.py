"""
Download Overture data for ALL of California with proper geographic distribution.
Downloads separately from NorCal, Central, and SoCal to ensure even spread.
"""
import duckdb
import pandas as pd
import struct

def parse_wkb_point(wkb_bytes):
    """Parse WKB Point geometry to get lat/lon."""
    try:
        if wkb_bytes is None or len(wkb_bytes) < 21:
            return None, None
        byte_order = wkb_bytes[0]
        fmt = '<' if byte_order == 1 else '>'
        geom_type = struct.unpack(fmt + 'I', wkb_bytes[1:5])[0]
        if geom_type != 1:
            return None, None
        x = struct.unpack(fmt + 'd', wkb_bytes[5:13])[0]
        y = struct.unpack(fmt + 'd', wkb_bytes[13:21])[0]
        return y, x
    except:
        return None, None

def download_region(con, lat_min, lat_max, limit, region_name):
    """Download data for a specific latitude range."""
    print(f"  Downloading {region_name} (lat {lat_min}-{lat_max})...")
    
    query = f"""
    SELECT 
        id, names, categories, confidence, websites, phones, socials, 
        emails, brand, addresses, geometry, sources, bbox
    FROM read_parquet('s3://overturemaps-us-west-2/release/2025-12-17.0/theme=places/type=place/*', filename=true, hive_partitioning=1)
    WHERE bbox.xmin > -124.5 AND bbox.xmax < -114.0
      AND bbox.ymin > {lat_min} AND bbox.ymax < {lat_max}
    LIMIT {limit}
    """
    
    try:
        df = con.execute(query).fetchdf()
        print(f"    Got {len(df)} rows")
        return df
    except Exception as e:
        print(f"    Error: {e}")
        return pd.DataFrame()

def download_all_california(limit_per_region=7000, output='california_overture.parquet'):
    print("Downloading California places from ALL regions...")
    
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    
    # Define regions
    regions = [
        (37.5, 42.0, 'NorCal'),      # SF, Oakland, Sacramento, Redding
        (35.5, 37.5, 'Central'),      # Fresno, Bakersfield, Monterey
        (32.5, 35.5, 'SoCal'),        # LA, San Diego, Palm Springs
    ]
    
    all_dfs = []
    for lat_min, lat_max, name in regions:
        df = download_region(con, lat_min, lat_max, limit_per_region, name)
        if len(df) > 0:
            all_dfs.append(df)
    
    con.close()
    
    if not all_dfs:
        print("No data downloaded!")
        return
    
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal downloaded: {len(df)} rows")
    
    # Parse coordinates
    print("Parsing geometry...")
    def get_coords(row):
        lat, lon = parse_wkb_point(row['geometry'])
        if lat is None and 'bbox' in row and row['bbox'] is not None:
            try:
                b = row['bbox']
                lat = (b['ymin'] + b['ymax']) / 2
                lon = (b['xmin'] + b['xmax']) / 2
            except:
                pass
        return pd.Series({'latitude': lat, 'longitude': lon})
    
    coords = df.apply(get_coords, axis=1)
    df['latitude'] = coords['latitude']
    df['longitude'] = coords['longitude']
    
    df = df.dropna(subset=['latitude', 'longitude'])
    
    # Extract name/category
    def get_name(n):
        if n is None: return 'Unknown'
        if isinstance(n, dict): return n.get('primary', 'Unknown')
        return str(n)
    df['name'] = df['names'].apply(get_name)
    
    def get_cat(c):
        if c is None: return 'unknown'
        if isinstance(c, dict): return c.get('primary', 'unknown')
        return 'unknown'
    df['category'] = df['categories'].apply(get_cat)
    
    print(f"\nFinal stats:")
    print(f"  Total valid: {len(df)}")
    north = len(df[df['latitude'] > 37.5])
    central = len(df[(df['latitude'] > 35.5) & (df['latitude'] <= 37.5)])
    south = len(df[df['latitude'] <= 35.5])
    print(f"  NorCal: {north}, Central: {central}, SoCal: {south}")
    
    df.to_parquet(output, index=False)
    print(f"\n✅ Saved to {output}")

if __name__ == "__main__":
    download_all_california()
