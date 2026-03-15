"""
Central configuration: loads .env and provides DB engine / session helpers.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# ── Load .env from project root ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── API Keys ─────────────────────────────────────────────────────────
MAPILLARY_TOKEN = os.getenv("MAPILLARY_ACCESS_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_STREETVIEW_KEY")
FOURSQUARE_API_KEY = os.getenv("FOURSQUARE_API_KEY")
YELP_API_KEY = os.getenv("YELP_API_KEY")
AZURE_MAPS_KEY = os.getenv("AZURE_MAPS_KEY")
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")
BRAVE_SEARCH_KEY = os.getenv("BRAVE_SEARCH_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Database ─────────────────────────────────────────────────────────
DB_USER = os.getenv("DB_USER", "terraforma")
DB_PASSWORD = os.getenv("DB_PASSWORD", "terraforma_dev")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "terraforma")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def get_session():
    """Return a new SQLAlchemy session."""
    return SessionLocal()


def check_connection():
    """Test that the database is reachable and the schema exists."""
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT schema_name FROM information_schema.schemata "
            "WHERE schema_name IN ('registries','overture','cv_scores','web_scores','ground_truth','predictions') "
            "ORDER BY schema_name"
        ))
        schemas = [row[0] for row in result]
        return schemas


# ── City bounding boxes (for Overture / API queries) ─────────────────
CITY_BBOXES = {
    "san_francisco": {"min_lon": -122.52, "max_lon": -122.35, "min_lat": 37.70, "max_lat": 37.82},
    "new_york":      {"min_lon": -74.05,  "max_lon": -73.90,  "min_lat": 40.68, "max_lat": 40.88},
    "chicago":       {"min_lon": -87.94,  "max_lon": -87.52,  "min_lat": 41.64, "max_lat": 42.02},
    "paris":         {"min_lon": 2.22,    "max_lon": 2.47,    "min_lat": 48.81, "max_lat": 48.90},
    "singapore":     {"min_lon": 103.60,  "max_lon": 104.05,  "min_lat": 1.22,  "max_lat": 1.47},
    "london":        {"min_lon": -0.35,   "max_lon": 0.05,    "min_lat": 51.40, "max_lat": 51.60},
    "mumbai":        {"min_lon": 72.77,   "max_lon": 72.98,   "min_lat": 18.89, "max_lat": 19.27},
    "philadelphia":  {"min_lon": -75.28,  "max_lon": -75.10,  "min_lat": 39.87, "max_lat": 40.02},
    "tucson":        {"min_lon": -111.10, "max_lon": -110.75, "min_lat": 32.05, "max_lat": 32.35},
    "tampa":         {"min_lon": -82.55,  "max_lon": -82.35,  "min_lat": 27.85, "max_lat": 28.10},
    "indianapolis":  {"min_lon": -86.30,  "max_lon": -86.05,  "min_lat": 39.65, "max_lat": 39.90},
    "nashville":     {"min_lon": -86.90,  "max_lon": -86.65,  "min_lat": 36.05, "max_lat": 36.25},
    "new_orleans":   {"min_lon": -90.15,  "max_lon": -89.95,  "min_lat": 29.90, "max_lat": 30.05},
    "saint_louis":   {"min_lon": -90.35,  "max_lon": -90.15,  "min_lat": 38.55, "max_lat": 38.70},
    "houston":       {"min_lon": -95.60,  "max_lon": -95.20,  "min_lat": 29.60, "max_lat": 29.90},
    "phoenix":       {"min_lon": -112.20, "max_lon": -111.90, "min_lat": 33.30, "max_lat": 33.60},
    "atlanta":       {"min_lon": -84.50,  "max_lon": -84.30,  "min_lat": 33.70, "max_lat": 33.85},
}

# US-only cities for the active pipeline
US_CITIES = ["san_francisco", "new_york", "chicago"]
