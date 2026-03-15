"""
Download business registry data from city open-data portals.
Supports: Socrata (SF, NYC), recherche-entreprises.api.gouv.fr (Paris),
          data.gov.sg (Singapore), FHRS (London).
"""

import csv
import io
import json
import logging
import time
import zipfile
from datetime import datetime
from pathlib import Path

import requests

from src.step1_registries.normalize import normalize_name, normalize_address

log = logging.getLogger(__name__)

PAGE_SIZE = 5000
MAX_RECORDS = 50_000  # cap per city to keep things manageable


def _paginate_socrata(base_url: str, params: dict, max_records: int = MAX_RECORDS):
    """Generic Socrata JSON API paginator."""
    offset = 0
    while offset < max_records:
        params.update({"$limit": PAGE_SIZE, "$offset": offset})
        resp = requests.get(base_url, params=params, timeout=60)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        yield from rows
        offset += len(rows)
        log.info("  fetched %d rows so far", offset)


# ── San Francisco ────────────────────────────────────────────────────

SF_URL = "https://data.sfgov.org/resource/g8m3-pdis.json"


def _sf_is_open(row: dict) -> bool:
    end = row.get("dba_end_date")
    if not end:
        return True
    try:
        return datetime.fromisoformat(end.replace("T", " ").split(".")[0]) > datetime.now()
    except ValueError:
        return True


def download_sf() -> list[dict]:
    log.info("Downloading San Francisco registry data …")
    params = {"$where": "city='San Francisco'", "$order": "uniqueid"}
    records = []
    for row in _paginate_socrata(SF_URL, params):
        loc = row.get("location", {})
        coords = loc.get("coordinates", [None, None]) if isinstance(loc, dict) else [None, None]
        records.append({
            "city": "san_francisco",
            "name": row.get("dba_name"),
            "name_normalized": normalize_name(row.get("dba_name")),
            "address": row.get("full_business_address"),
            "address_normalized": normalize_address(row.get("full_business_address")),
            "longitude": coords[0],
            "latitude": coords[1],
            "category": None,  # SF registry has no category
            "is_open": _sf_is_open(row),
            "raw_source": json.dumps(row),
        })
    log.info("SF: %d records", len(records))
    return records


# ── New York City ────────────────────────────────────────────────────

NYC_URL = "https://data.cityofnewyork.us/resource/w7w3-xahh.json"


def download_nyc() -> list[dict]:
    log.info("Downloading New York City registry data …")
    params = {"$order": "license_nbr"}
    records = []
    for row in _paginate_socrata(NYC_URL, params):
        if not row.get("business_name"):
            continue
        addr_parts = [
            row.get("address_building", ""),
            row.get("address_street_name", ""),
        ]
        address = " ".join(p for p in addr_parts if p).strip() or None

        lat = row.get("latitude")
        lon = row.get("longitude")

        records.append({
            "city": "new_york",
            "name": row.get("business_name"),
            "name_normalized": normalize_name(row.get("business_name")),
            "address": address,
            "address_normalized": normalize_address(address),
            "latitude": float(lat) if lat else None,
            "longitude": float(lon) if lon else None,
            "category": row.get("business_category"),
            "is_open": row.get("license_status", "").lower() == "active",
            "raw_source": json.dumps(row),
        })
    log.info("NYC: %d records", len(records))
    return records


# ── Paris (recherche-entreprises.api.gouv.fr — no auth needed) ────────

PARIS_URL = "https://recherche-entreprises.api.gouv.fr/search"


def download_paris() -> list[dict]:
    log.info("Downloading Paris registry data …")
    records = []
    page = 1
    while len(records) < MAX_RECORDS:
        resp = requests.get(PARIS_URL, params={
            "departement": "75",  # Paris département
            "per_page": 25,       # API max per page
            "page": page,
        }, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            break
        for row in results:
            name = row.get("nom_complet")
            if not name:
                continue
            siege = row.get("siege") or {}
            try:
                lat = float(siege.get("latitude"))
                lon = float(siege.get("longitude"))
            except (TypeError, ValueError):
                lat, lon = None, None
            is_open = row.get("etat_administratif") == "A"
            address = siege.get("adresse")
            if address and "[NON-DIFFUSIBLE]" in address:
                address = None
            records.append({
                "city": "paris",
                "name": name,
                "name_normalized": normalize_name(name),
                "address": address,
                "address_normalized": normalize_address(address),
                "latitude": lat,
                "longitude": lon,
                "category": row.get("activite_principale"),
                "is_open": is_open,
                "raw_source": json.dumps(row, default=str),
            })
        total_pages = data.get("total_pages", 1)
        log.info("  fetched page %d / %d (%d records)", page, total_pages, len(records))
        if page >= total_pages:
            break
        page += 1
        if page > 2000:  # safety cap (25 * 2000 = 50k)
            break
    log.info("Paris: %d records", len(records))
    return records


# ── Singapore (data.gov.sg bulk CSV download) ────────────────────────

SG_DATASET_ID = "d_8575e84912df3c28995b8e6e0e05205a"  # ACRA letter A
SG_INITIATE_URL = f"https://api-open.data.gov.sg/v1/public/api/datasets/{SG_DATASET_ID}/initiate-download"
SG_POLL_URL = f"https://api-open.data.gov.sg/v1/public/api/datasets/{SG_DATASET_ID}/poll-download"
SG_MAX = 10_000


def download_singapore() -> list[dict]:
    log.info("Downloading Singapore registry data (bulk CSV) …")

    # Step 1: initiate download to get S3 URL
    resp = requests.get(SG_INITIATE_URL, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    url = data.get("data", {}).get("url")

    # If not ready yet, poll
    if not url:
        for _ in range(10):
            time.sleep(5)
            resp = requests.get(SG_POLL_URL, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            url = data.get("data", {}).get("url")
            if url:
                break

    if not url:
        raise RuntimeError("Could not get Singapore CSV download URL")

    log.info("  downloading CSV from S3 …")
    csv_resp = requests.get(url, timeout=120)
    csv_resp.raise_for_status()

    reader = csv.DictReader(io.StringIO(csv_resp.text))
    records = []
    for row in reader:
        name = row.get("entity_name")
        if not name:
            continue
        street = row.get("reg_street_name")
        postal = row.get("reg_postal_code")
        address = f"{street}, {postal}" if street and postal else street or None
        status = (row.get("uen_status") or "").upper()
        records.append({
            "city": "singapore",
            "name": name,
            "name_normalized": normalize_name(name),
            "address": address,
            "address_normalized": normalize_address(address),
            "latitude": None,
            "longitude": None,
            "category": row.get("entity_type"),
            "is_open": status in ("L", "R"),
            "raw_source": json.dumps(row, default=str),
        })
        if len(records) >= SG_MAX:
            break
    log.info("Singapore: %d records", len(records))
    return records


# ── London (UK Food Hygiene Rating Scheme API — no auth) ─────────────

FHRS_URL = "https://api.ratings.food.gov.uk/Establishments"
FHRS_HEADERS = {"x-api-version": "2", "Accept": "application/json"}
# Largest London boroughs by establishment count
LONDON_BOROUGHS = [120, 93, 96, 115, 94, 117, 112, 106, 119, 91]
LONDON_MAX = 10_000


def download_london() -> list[dict]:
    log.info("Downloading London registry data (FHRS) …")
    records = []
    for la_id in LONDON_BOROUGHS:
        if len(records) >= LONDON_MAX:
            break
        page = 1
        while len(records) < LONDON_MAX:
            resp = requests.get(FHRS_URL, headers=FHRS_HEADERS, params={
                "localAuthorityId": la_id,
                "pageNumber": page,
                "pageSize": 5000,
            }, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            establishments = data.get("establishments", [])
            if not establishments:
                break
            for row in establishments:
                name = row.get("BusinessName")
                if not name:
                    continue
                addr_parts = [
                    row.get("AddressLine1", ""),
                    row.get("AddressLine2", ""),
                    row.get("AddressLine3", ""),
                    row.get("PostCode", ""),
                ]
                address = ", ".join(p for p in addr_parts if p).strip() or None
                geo = row.get("geocode", {})
                try:
                    lat = float(geo.get("latitude"))
                    lon = float(geo.get("longitude"))
                except (TypeError, ValueError):
                    lat, lon = None, None
                records.append({
                    "city": "london",
                    "name": name,
                    "name_normalized": normalize_name(name),
                    "address": address,
                    "address_normalized": normalize_address(address),
                    "latitude": lat,
                    "longitude": lon,
                    "category": row.get("BusinessType"),
                    "is_open": True,  # FHRS only lists active establishments
                    "raw_source": json.dumps(row, default=str),
                })
                if len(records) >= LONDON_MAX:
                    break
            log.info("  borough %d: %d records total", la_id, len(records))
            page += 1
    log.info("London: %d records", len(records))
    return records


# ── Mumbai (Kaggle Indian Companies CSV — local file) ────────────────

MUMBAI_ZIP = Path(__file__).resolve().parents[2] / "archive.zip"
MUMBAI_CSV_NAME = "registered_companies.csv"
MUMBAI_MAX = 10_000
MUMBAI_STATUS_MAP = {
    "ACTV": True, "STOF": False, "DISD": False, "ULQD": False,
    "AMAL": False, "NAEF": False, "CLLD": False, "CLLP": False,
    "D455": False, "UPSO": False, "LIQD": False,
}


def _parse_indian_date(d: str) -> datetime | None:
    try:
        return datetime.strptime(d.strip(), "%d-%m-%Y")
    except (ValueError, AttributeError):
        return None


def download_mumbai() -> list[dict]:
    log.info("Loading Mumbai registry data from %s …", MUMBAI_ZIP.name)
    if not MUMBAI_ZIP.exists():
        raise FileNotFoundError(f"Place the Kaggle archive at {MUMBAI_ZIP}")

    rows = []
    with zipfile.ZipFile(MUMBAI_ZIP) as z:
        with z.open(MUMBAI_CSV_NAME) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace"))
            for row in reader:
                roc = (row.get("REGISTRAR_OF_COMPANIES") or "").replace("\xa0", " ")
                if "MUMBAI" not in roc:
                    continue
                dt = _parse_indian_date(row.get("DATE_OF_REGISTRATION", ""))
                rows.append((dt, row))

    # Sort by most recent registration date first (None dates go last)
    rows.sort(key=lambda x: x[0] or datetime.min, reverse=True)

    records = []
    for dt, row in rows[:MUMBAI_MAX]:
        name = (row.get("COMPANY_NAME") or "").strip()
        if not name:
            continue
        address = (row.get("REGISTERED_OFFICE_ADDRESS") or "").strip() or None
        status = (row.get("COMPANY_STATUS") or "").strip()
        records.append({
            "city": "mumbai",
            "name": name,
            "name_normalized": normalize_name(name),
            "address": address,
            "address_normalized": normalize_address(address),
            "latitude": None,
            "longitude": None,
            "category": row.get("PRINCIPAL_BUSINESS_ACTIVITY_AS_PER_CIN"),
            "is_open": MUMBAI_STATUS_MAP.get(status, False),
            "raw_source": json.dumps(row, default=str),
        })
    log.info("Mumbai: %d records (most recent registrations)", len(records))
    return records


# ── Chicago (Socrata — data.cityofchicago.org Business Licenses) ───

CHI_URL = "https://data.cityofchicago.org/resource/r5kz-chrr.json"


def download_chicago() -> list[dict]:
    log.info("Downloading Chicago registry data …")
    params = {"$order": "id", "$where": "city='CHICAGO'"}
    records = []
    for row in _paginate_socrata(CHI_URL, params):
        name = row.get("doing_business_as_name") or row.get("legal_name")
        if not name:
            continue
        addr_parts = [
            row.get("address", ""),
            row.get("city", ""),
            row.get("state", ""),
            row.get("zip_code", ""),
        ]
        address = ", ".join(p for p in addr_parts if p).strip() or None

        lat = row.get("latitude")
        lon = row.get("longitude")
        status = (row.get("license_status") or "").upper()

        records.append({
            "city": "chicago",
            "name": name,
            "name_normalized": normalize_name(name),
            "address": address,
            "address_normalized": normalize_address(address),
            "latitude": float(lat) if lat else None,
            "longitude": float(lon) if lon else None,
            "category": row.get("license_description"),
            "is_open": status in ("AAI", "AAC"),
            "raw_source": json.dumps(row, default=str),
        })
    log.info("Chicago: %d records", len(records))
    return records


DOWNLOADERS = {
    "san_francisco": download_sf,
    "new_york": download_nyc,
    "chicago": download_chicago,
    "paris": download_paris,
    "singapore": download_singapore,
    "london": download_london,
    "mumbai": download_mumbai,
}
