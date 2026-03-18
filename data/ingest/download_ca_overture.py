"""
Download REAL Overture Maps Places Data for California.
"""
import duckdb
import pandas as pd
import shapely.wkb

def download_california_real(limit=10000, output='california_real.parquet'):
    print(f"Downloading {limit} California places from Overture...")
    
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    
    # Simple query using bbox struct for filtering
    # CA Bbox ~ [-124.4, 32.5, -114.1, 42.0] (approx)
    
    query = f"""
    COPY (
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
            sources
        FROM read_parquet('s3://overturemaps-us-west-2/release/2025-12-17.0/theme=places/type=place/*', filename=true, hive_partitioning=1)
        WHERE bbox.xmin > -124.5 
          AND bbox.xmax < -114.0
          AND bbox.ymin > 32.5
          AND bbox.ymax < 42.0
        LIMIT {limit}
    ) TO '{output}' (FORMAT PARQUET);
    """
    
    try:
        con.execute(query)
        print("Download complete.")
        
        # Post-process to get lat/lon from WKB geometry
        print("Processing geometry...")
        df = pd.read_parquet(output)
        
        def parse_geom(wkb):
            try:
                g = shapely.wkb.loads(wkb)
                return g.y, g.x # lat, lon
            except:
                return None, None
                
        coords = df['geometry'].apply(parse_geom)
        df['latitude'] = coords.apply(lambda x: x[0])
        df['longitude'] = coords.apply(lambda x: x[1])
        
        # Clean up
        df = df.dropna(subset=['latitude', 'longitude'])
        
        # Extract primary name
        def get_name(n):
            try: return n['primary']
            except: return str(n)
        df['name'] = df['names'].apply(get_name)
        
        # Map categories
        def map_cat(c):
            try: return c['primary']
            except: return 'unknown'
        df['category'] = df['categories'].apply(map_cat)
        
        df.to_parquet(output, index=False)
        print(f"✅ Saved {len(df)} California places with real coordinates to {output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        con.close()

if __name__ == "__main__":
    download_california_real()
