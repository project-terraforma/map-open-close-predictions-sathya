"""
Verify database connection and schema setup.
Run: python -m src.db_init
"""

from src.config import check_connection, DATABASE_URL


def main():
    print(f"Connecting to: {DATABASE_URL.replace(DATABASE_URL.split(':')[2].split('@')[0], '***')}")
    try:
        schemas = check_connection()
        print(f"Connection OK. Found schemas: {schemas}")
        expected = ["cv_scores", "ground_truth", "overture", "predictions", "registries", "web_scores"]
        missing = set(expected) - set(schemas)
        if missing:
            print(f"WARNING: Missing schemas: {missing}")
            print("The schema.sql may not have run. Re-create the container:")
            print("  docker compose down -v && docker compose up -d")
        else:
            print("All 6 schemas present. Step 0 complete!")
    except Exception as e:
        print(f"Connection FAILED: {e}")
        print("Make sure PostgreSQL is running: docker compose up -d")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
