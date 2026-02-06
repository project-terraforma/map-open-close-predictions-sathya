"""Test S3 download"""
import duckdb

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("SET s3_region='us-west-2';")

# Test basic query
s3_path = "s3://overturemaps-us-west-2/release/2025-12-17.0/theme=places/type=place/*"

print("Testing S3 connection...")
try:
    # Simple download without geographic filter
    query = f"""
    COPY (
        SELECT 
            id,
            sources,
            names,
            categories,
            confidence,
            websites,
            phones,
            brand,
            addresses
        FROM read_parquet('{s3_path}', filename=true, hive_partitioning=1)
        LIMIT 10000
    ) TO 'test_overture.parquet' (FORMAT PARQUET);
    """
    con.execute(query)
    
    count = con.execute("SELECT COUNT(*) FROM read_parquet('test_overture.parquet')").fetchone()[0]
    print(f"Success! Downloaded {count} records")
except Exception as e:
    print(f"Error: {e}")

con.close()
