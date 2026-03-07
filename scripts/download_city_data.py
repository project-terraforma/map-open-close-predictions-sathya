"""
Download Overture Maps place data for a specific city.

Usage:
  python scripts/download_city_data.py la
  python scripts/download_city_data.py chicago
  python scripts/download_city_data.py --all

Cities: sf, la, chicago
"""

import sys
import os
import duckdb

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'pipeline', 'data')

CITIES = {
    'sf': {
        'name': 'San Francisco',
        'bbox': (-122.52, -122.35, 37.70, 37.83),
        'output': 'sf_places_large.parquet',
    },
    'la': {
        'name': 'Los Angeles',
        'bbox': (-118.70, -118.15, 33.85, 34.15),
        'output': 'la_places.parquet',
    },
    'chicago': {
        'name': 'Chicago',
        'bbox': (-87.85, -87.52, 41.65, 42.02),
        'output': 'chicago_places.parquet',
    },
    'miami': {
        'name': 'Miami',
        'bbox': (-80.22, -80.12, 25.74, 25.82),
        'output': 'miami_places.parquet',
    },
}

OVERTURE_PATH = "s3://overturemaps-us-west-2/release/2026-02-18.0/theme=places/type=place/*"


def download_city(city_key):
    city = CITIES[city_key]
    lon_min, lon_max, lat_min, lat_max = city['bbox']
    output_path = os.path.join(OUTPUT_DIR, city['output'])

    print(f"Downloading {city['name']} places from Overture Maps...")
    print(f"  Bbox: lon[{lon_min}, {lon_max}] lat[{lat_min}, {lat_max}]")

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")

    query = f"""
    COPY (
        SELECT *
        FROM read_parquet('{OVERTURE_PATH}', filename=true, hive_partitioning=1)
        WHERE bbox.xmin > {lon_min} AND bbox.xmax < {lon_max}
          AND bbox.ymin > {lat_min} AND bbox.ymax < {lat_max}
    ) TO '{output_path}' (FORMAT PARQUET)
    """

    try:
        con.execute(query)
        # Check how many rows
        count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path}')").fetchone()[0]
        print(f"  Downloaded {count} places to {output_path}")
    except Exception as e:
        print(f"  Error: {e}")
    finally:
        con.close()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <city>")
        print(f"Cities: {', '.join(CITIES.keys())}")
        print(f"Or: python {sys.argv[0]} --all")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    arg = sys.argv[1].lower()
    if arg == '--all':
        for key in CITIES:
            download_city(key)
    elif arg in CITIES:
        download_city(arg)
    else:
        print(f"Unknown city: {arg}")
        print(f"Available: {', '.join(CITIES.keys())}")
        sys.exit(1)


if __name__ == '__main__':
    main()
