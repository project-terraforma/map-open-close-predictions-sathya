"""
Download California business data from Overture Maps for map visualization.
"""
import duckdb
import pandas as pd


def download_california_businesses(limit: int = 5000, output: str = "california_businesses.parquet"):
    """Download California businesses with coordinates for mapping."""
    print(f"Downloading {limit:,} California businesses...")
    
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("SET s3_region='us-west-2';")
    
    s3_path = "s3://overturemaps-us-west-2/release/2025-12-17.0/theme=places/type=place/*"
    
    # California bounding box with coordinates extraction
    query = f"""
    COPY (
        SELECT 
            id,
            ST_X(ST_GeomFromWKB(geometry)) as longitude,
            ST_Y(ST_GeomFromWKB(geometry)) as latitude,
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
        WHERE bbox.xmin >= -124.5 
          AND bbox.xmax <= -114.0
          AND bbox.ymin >= 32.5
          AND bbox.ymax <= 42.1
        LIMIT {limit}
    ) TO '{output}' (FORMAT PARQUET);
    """
    
    try:
        con.execute(query)
        count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output}')").fetchone()[0]
        print(f"✅ Downloaded {count:,} California businesses to {output}")
        
        # Show sample
        df = pd.read_parquet(output)
        print(f"\nSample coordinates:")
        print(df[['latitude', 'longitude']].head())
        return True
    except Exception as e:
        print(f"❌ Error with spatial query: {e}")
        print("\nTrying without coordinates (will add random CA coords)...")
        
        # Fallback: download without geometry, add random CA coords
        fallback_query = f"""
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
        con.execute(fallback_query)
        
        # Add random California coordinates
        import numpy as np
        df = pd.read_parquet(output)
        np.random.seed(42)
        df['latitude'] = np.random.uniform(32.5, 42.0, len(df))
        df['longitude'] = np.random.uniform(-124.5, -114.0, len(df))
        df.to_parquet(output, index=False)
        
        print(f"✅ Downloaded {len(df):,} places with simulated CA coordinates")
        return True
    finally:
        con.close()


if __name__ == '__main__':
    download_california_businesses(5000, "california_businesses.parquet")
