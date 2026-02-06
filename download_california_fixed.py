"""
Download Overture Maps data with CORRECT coordinates for California.
Uses simpler approach - downloads raw data then filters in Python.
"""
import duckdb
import pandas as pd
import struct

def parse_wkb_point(wkb_bytes):
    """Parse WKB Point geometry to get lat/lon."""
    try:
        if wkb_bytes is None or len(wkb_bytes) < 21:
            return None, None
        # WKB Point format: byte order (1) + type (4) + x (8) + y (8)
        byte_order = wkb_bytes[0]
        fmt = '<' if byte_order == 1 else '>'
        geom_type = struct.unpack(fmt + 'I', wkb_bytes[1:5])[0]
        if geom_type != 1:  # Not a point
            return None, None
        x = struct.unpack(fmt + 'd', wkb_bytes[5:13])[0]  # longitude
        y = struct.unpack(fmt + 'd', wkb_bytes[13:21])[0]  # latitude
        return y, x  # lat, lon
    except:
        return None, None

def download_california_overture(limit=20000, output='california_overture.parquet'):
    print(f"Downloading {limit} California places from Overture...")
    
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    
    # Use bbox struct for filtering California
    # CA Bbox: lat 32.5-42.0, lon -124.5 to -114.1
    query = f"""
    SELECT 
        id,
        names,
        categories,
        confidence,
        websites,
        phones,
        socials,
        emails,
        brand,
        addresses,
        geometry,
        sources,
        bbox
    FROM read_parquet('s3://overturemaps-us-west-2/release/2025-12-17.0/theme=places/type=place/*', filename=true, hive_partitioning=1)
    WHERE bbox.xmin > -124.5 
      AND bbox.xmax < -114.0
      AND bbox.ymin > 32.5
      AND bbox.ymax < 42.1
    LIMIT {limit}
    """
    
    try:
        print("Running query (this may take a minute)...")
        df = con.execute(query).fetchdf()
        print(f"Downloaded {len(df)} rows")
        
        # Extract coordinates from WKB geometry
        print("Parsing geometry...")
        coords = df['geometry'].apply(lambda g: parse_wkb_point(g) if g is not None else (None, None))
        df['latitude'] = coords.apply(lambda x: x[0])
        df['longitude'] = coords.apply(lambda x: x[1])
        
        # Also try from bbox as backup
        if 'bbox' in df.columns:
            # bbox center as fallback
            def bbox_center(b):
                if b is None:
                    return None, None
                try:
                    lat = (b['ymin'] + b['ymax']) / 2
                    lon = (b['xmin'] + b['xmax']) / 2
                    return lat, lon
                except:
                    return None, None
            
            bbox_coords = df['bbox'].apply(bbox_center)
            # Use bbox center where geometry parsing failed
            for i in range(len(df)):
                if pd.isna(df.loc[i, 'latitude']) or pd.isna(df.loc[i, 'longitude']):
                    bc = bbox_coords.iloc[i]
                    df.loc[i, 'latitude'] = bc[0]
                    df.loc[i, 'longitude'] = bc[1]
        
        # Filter out invalid coordinates
        df = df.dropna(subset=['latitude', 'longitude'])
        
        # Verify coordinates are in California
        df = df[
            (df['latitude'] >= 32.5) & (df['latitude'] <= 42.1) &
            (df['longitude'] >= -124.5) & (df['longitude'] <= -114.0)
        ]
        
        print(f"Valid California rows: {len(df)}")
        print(f"Lat range: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
        print(f"Lon range: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")
        
        # Extract name and category
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
        
        # Save
        df.to_parquet(output, index=False)
        print(f"\n✅ Saved {len(df)} California places to {output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        con.close()

if __name__ == "__main__":
    download_california_overture(limit=20000)
