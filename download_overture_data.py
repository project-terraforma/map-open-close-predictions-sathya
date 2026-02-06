"""
Download Overture Maps Places Data - Simple Version

Uses DuckDB to query Overture Maps data directly from S3.
"""

import duckdb
import sys


def download_places(limit=10000, output='overture_places_large.parquet'):
    """Download Overture Maps places data."""
    
    print("Setting up DuckDB...")
    con = duckdb.connect()
    
    # Install and load extensions
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")
    
    # Use the latest known release
    release = '2025-12-17.0'
    s3_path = f"s3://overturemaps-us-west-2/release/{release}/theme=places/type=place/*"
    
    print(f"Querying Overture Maps release: {release}")
    print(f"Downloading up to {limit:,} places...")
    
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
        LIMIT {limit}
    ) TO '{output}' (FORMAT PARQUET);
    """
    
    try:
        con.execute(query)
        
        # Verify
        result = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output}')").fetchone()
        print(f"✅ Downloaded {result[0]:,} places to {output}")
        
        # Show sample
        sample = con.execute(f"""
            SELECT 
                id,
                names.primary as name,
                categories.primary as category,
                confidence
            FROM read_parquet('{output}')
            LIMIT 5
        """).fetchall()
        
        print("\nSample data:")
        for row in sample:
            print(f"  {row[1]} | {row[2]} | conf: {row[3]:.2f}" if row[3] else f"  {row[1]} | {row[2]}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    con.close()
    return True


if __name__ == '__main__':
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    output = sys.argv[2] if len(sys.argv) > 2 else 'overture_places_large.parquet'
    download_places(limit, output)
