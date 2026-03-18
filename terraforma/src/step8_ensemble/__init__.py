"""
Step 8: Ensemble scoring — combine 3 prediction signals into a single open/closed percentage.

Prediction signals:
  1. Trained Model       — GradientBoosting from Overture features (step4_classifier)
  2. OpenStreetMap       — Overpass POI lookup (found / disused / missing)
  3. Web Crawling        — website HEAD + Brave Search + Groq AI reasoning

Ground truth (for evaluation only, NOT used in prediction):
  - Google Places API   — business_status from step7

Each signal produces a 0-1 sub-score. The ensemble combines them with weights
into a final probability that the business is currently open.

Usage:
    python -m src.step8_ensemble sf        # score SF ground-truth businesses
    python -m src.step8_ensemble all       # all cities with ground truth
"""

import logging
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sqlalchemy import text

from src.config import engine, BRAVE_SEARCH_KEY, GROQ_API_KEY

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "open_closed_v1.pkl"

ALIASES = {
    "sf": "san_francisco", "san_francisco": "san_francisco",
    "nyc": "new_york", "new_york": "new_york",
    "chi": "chicago", "chicago": "chicago",
}

CITY_NAMES = {
    "san_francisco": "San Francisco",
    "new_york": "New York",
    "chicago": "Chicago",
}

# ── Signal weights ──────────────────────────────────────────────────
# Model: trained on Overture structural features, good baseline
# Web crawling: Brave + Groq AI reads real web content, strongest signal
# OSM: solid structural data but lower coverage
W_MODEL = 0.25
W_OSM   = 0.20
W_WEB   = 0.55


# ══════════════════════════════════════════════════════════════════════
# Signal 1: Trained Model (GradientBoosting from step4_classifier)
# ══════════════════════════════════════════════════════════════════════

def _load_model():
    """Load the trained model, feature columns, and threshold."""
    if not MODEL_PATH.exists():
        log.warning("No model at %s", MODEL_PATH)
        return None, None, None
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["feature_cols"], saved["threshold"]


_jan_cache = {}  # city -> {id -> row_dict}


def _load_jan_snapshot(city: str):
    """Load the Jan release parquet for a city into memory (cached)."""
    if city in _jan_cache:
        return _jan_cache[city]

    from pathlib import Path
    jan_path = Path(__file__).resolve().parent.parent.parent / f"cache_{city}_jan.parquet"
    if not jan_path.exists():
        _jan_cache[city] = {}
        return _jan_cache[city]

    import json as _json
    df = pd.read_parquet(jan_path)
    lookup = {}
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            val = row[col]
            # Convert to JSON string if needed (for consistency with DB raw_json)
            if col != "id" and col != "confidence":
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    if isinstance(val, str):
                        record[col] = val
                    else:
                        try:
                            record[col] = _json.dumps(val)
                        except (TypeError, ValueError):
                            record[col] = str(val)
                else:
                    record[col] = None
            else:
                record[col] = val
        lookup[row["id"]] = record

    _jan_cache[city] = lookup
    log.info("  Loaded Jan snapshot for %s: %d places", city, len(lookup))
    return lookup


def model_signal(overture_id: str, city: str = "san_francisco",
                 osm_result: dict = None, website_alive: bool = None) -> dict:
    """Run the CatBoost+LightGBM ensemble on a single Overture place.

    Uses current Overture features + cross-source features (OSM, website liveness).
    Cross-source features are passed in from the ensemble pipeline which computes
    them anyway — this way the model can learn from all signals together.
    """
    import json as _json
    from src.step4_classifier import extract_features

    model, feature_cols, threshold = _load_model()
    if model is None:
        return {"model_score": None, "model_pred_open": None}

    # Get current snapshot from DB
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT id, confidence, raw_json
            FROM overture.places WHERE id = :oid
        """), {"oid": overture_id}).fetchone()

    if not row:
        return {"model_score": None, "model_pred_open": None}

    # Parse current snapshot
    record = {"confidence": row[1], "base_confidence": row[1]}
    rj = row[2]
    if rj:
        if isinstance(rj, str):
            try:
                rj = _json.loads(rj)
            except (ValueError, TypeError):
                rj = {}
        if isinstance(rj, dict):
            for key in ("sources", "names", "categories", "websites", "socials",
                         "emails", "phones", "brand", "addresses"):
                val = rj.get(key)
                if val is not None:
                    serialized = _json.dumps(val) if not isinstance(val, str) else val
                    record[key] = serialized
                    record[f"base_{key}"] = serialized

    # Add cross-source features
    if website_alive is not None:
        record["website_alive"] = int(website_alive)
    if osm_result:
        record["osm_found"] = int(osm_result.get("osm_found", False))
        record["osm_disused"] = int(osm_result.get("osm_disused", False))
        record["osm_replaced"] = int(
            not osm_result.get("osm_found", False) and
            osm_result.get("osm_building", False) and
            osm_result.get("osm_businesses_here", 0) > 0
        )
        record["osm_building"] = int(osm_result.get("osm_building", False))
        record["osm_businesses_here"] = osm_result.get("osm_businesses_here", 0)

    df = pd.DataFrame([record])
    feat = extract_features(df)

    # Ensure all expected columns exist
    for col in feature_cols:
        if col not in feat.columns:
            feat[col] = 0

    X = feat[feature_cols].values.astype(float)
    proba = float(model.predict_proba(X)[:, 1][0])
    pred = proba >= threshold

    return {
        "model_score": round(proba, 4),
        "model_pred_open": bool(pred),
    }


# ══════════════════════════════════════════════════════════════════════
# Signal 2: OpenStreetMap (Overpass)
# ══════════════════════════════════════════════════════════════════════

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _name_overlap(a: str, b: str) -> bool:
    a, b = a.lower().strip(), b.lower().strip()
    if a in b or b in a:
        return True
    wa, wb = set(a.split()), set(b.split())
    if not wa or not wb:
        return False
    return len(wa & wb) / max(len(wa), len(wb)) >= 0.5


def osm_signal(name: str, lat: float, lon: float) -> dict:
    """Query OSM Overpass: find the building at this address and list all businesses inside.

    Strategy:
      1. Find the building (way/relation with building tag) nearest the coordinates
      2. Get all business POIs inside/near that building (shop, amenity, office, craft)
      3. Check if our target business is among them, replaced by something else, or absent
      4. Also check for disused:* tags (explicit closure markers)

    Returns 0-1 score + metadata about what was found.
    """
    # Query: find building + all named/tagged POIs within 50m
    query = f"""
    [out:json][timeout:15];
    (
      // Find the building at this location
      way["building"](around:35,{lat},{lon});
      relation["building"](around:35,{lat},{lon});
      // Find all business POIs nearby (inside or next to building)
      node["shop"](around:50,{lat},{lon});
      node["amenity"](around:50,{lat},{lon});
      node["office"](around:50,{lat},{lon});
      node["craft"](around:50,{lat},{lon});
      // Also catch named nodes/ways (some businesses only have name tag)
      node["name"](around:50,{lat},{lon});
      way["name"](around:50,{lat},{lon});
      // Catch explicitly closed/disused businesses
      node["disused:shop"](around:50,{lat},{lon});
      node["disused:amenity"](around:50,{lat},{lon});
      way["disused:shop"](around:50,{lat},{lon});
      way["disused:amenity"](around:50,{lat},{lon});
    );
    out body center;
    """

    data = None
    for server in OVERPASS_SERVERS:
        try:
            resp = requests.post(server, data={"data": query}, timeout=20)
            if resp.status_code == 429:
                time.sleep(2)
                continue
            if resp.status_code != 200:
                continue
            data = resp.json()
            break
        except KeyboardInterrupt:
            raise
        except Exception:
            continue

    if data is None:
        return {"osm_score": None, "osm_found": False, "osm_name": None,
                "osm_disused": False, "osm_building": False, "osm_businesses_here": 0}

    elements = data.get("elements", [])

    # Separate buildings from business POIs
    buildings = []
    pois = []
    for el in elements:
        tags = el.get("tags", {})
        if tags.get("building"):
            buildings.append(el)
        elif any(tags.get(k) for k in ("shop", "amenity", "office", "craft", "name",
                                        "disused:shop", "disused:amenity")):
            pois.append(el)

    has_building = len(buildings) > 0

    # Analyze all POIs: look for our business + count others in the building
    best_match = None
    best_dist = 999
    other_businesses = []
    disused_matches = []

    for el in pois:
        tags = el.get("tags", {})
        osm_name = tags.get("name", "") or tags.get("disused:name", "")
        is_disused = any(k.startswith("disused:") for k in tags)

        el_lat = el.get("lat") or el.get("center", {}).get("lat")
        el_lon = el.get("lon") or el.get("center", {}).get("lon")
        dist = _haversine(lat, lon, el_lat, el_lon) if (el_lat and el_lon) else 50

        if osm_name and _name_overlap(name, osm_name):
            # Found our business (or its disused version)
            if is_disused:
                disused_matches.append({"osm_name": osm_name, "dist": dist})
            elif dist < best_dist:
                best_match = {"osm_name": osm_name, "osm_disused": False, "dist": dist}
                best_dist = dist
        elif osm_name and dist < 50:
            # Different business at this location
            biz_type = (tags.get("shop") or tags.get("amenity") or
                        tags.get("office") or tags.get("craft") or "business")
            other_businesses.append({"name": osm_name, "type": biz_type, "dist": dist})

    n_businesses_here = len(other_businesses) + (1 if best_match else 0) + len(disused_matches)

    # Scoring logic — ONLY care about found=True cases:
    # 1. Found our business, active → boost (strong open signal)
    if best_match:
        score = 0.85 if best_match["dist"] < 25 else 0.70
        return {
            "osm_score": score,
            "osm_found": True,
            "osm_name": best_match["osm_name"],
            "osm_disused": False,
            "osm_building": has_building,
            "osm_businesses_here": n_businesses_here,
        }

    # 2. Found our business but marked disused → penalize (strong closed signal)
    if disused_matches:
        return {
            "osm_score": 0.10,
            "osm_found": True,
            "osm_name": disused_matches[0]["osm_name"],
            "osm_disused": True,
            "osm_building": has_building,
            "osm_businesses_here": n_businesses_here,
        }

    # 3. NOT found in OSM → completely neutral. Not being in OSM means nothing.
    #    OSM coverage is sparse — most real businesses aren't in it.
    return {"osm_score": None, "osm_found": False, "osm_name": None,
            "osm_disused": False, "osm_building": has_building,
            "osm_businesses_here": n_businesses_here}


# ══════════════════════════════════════════════════════════════════════
# Signal 3: Web Crawling (website HEAD + Brave Search + Groq AI)
# ══════════════════════════════════════════════════════════════════════

def _check_website(url: str) -> dict:
    if not url:
        return {"web_alive": None, "web_status_code": None}
    try:
        resp = requests.head(url, timeout=8, allow_redirects=True,
                             headers={"User-Agent": "TerraForma/1.0 (research)"})
        alive = resp.status_code < 400
        return {"web_alive": alive, "web_status_code": resp.status_code}
    except requests.exceptions.SSLError:
        try:
            resp = requests.head(url, timeout=8, allow_redirects=True, verify=False,
                                 headers={"User-Agent": "TerraForma/1.0 (research)"})
            return {"web_alive": resp.status_code < 400, "web_status_code": resp.status_code}
        except Exception:
            return {"web_alive": False, "web_status_code": 0}
    except Exception:
        return {"web_alive": False, "web_status_code": 0}


def _brave_search(name: str, city_label: str) -> list[dict]:
    """5 targeted Brave queries maximizing signal diversity.

    1. General listing — official site, Google Maps, directories
    2. Review platforms — Yelp, TripAdvisor (most reliable open/closed)
    3. Closure signals — explicit "permanently closed" mentions
    4. Social/directories — Facebook, Instagram, local listings
    5. News — recent articles about the business
    """
    if not BRAVE_SEARCH_KEY:
        return []

    def _search(q, count=3):
        try:
            resp = requests.get("https://api.search.brave.com/res/v1/web/search",
                                params={"q": q, "count": count},
                                headers={"X-Subscription-Token": BRAVE_SEARCH_KEY,
                                         "Accept": "application/json"},
                                timeout=10)
            return resp.json().get("web", {}).get("results", [])
        except Exception:
            return []

    from concurrent.futures import ThreadPoolExecutor

    q1 = f'"{name}" {city_label}'
    q2 = f'"{name}" {city_label} site:yelp.com OR site:tripadvisor.com OR site:google.com/maps'
    q3 = f'"{name}" {city_label} "permanently closed" OR "out of business" OR "shut down"'
    q4 = f'"{name}" {city_label} site:facebook.com OR site:instagram.com OR hours OR address'
    q5 = f'"{name}" {city_label} closed OR opening OR relocated OR new location'

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [
            pool.submit(_search, q1, 3),
            pool.submit(_search, q2, 3),
            pool.submit(_search, q3, 2),
            pool.submit(_search, q4, 2),
            pool.submit(_search, q5, 2),
        ]
        all_results = []
        for f in futures:
            all_results.extend(f.result())

    # Deduplicate by URL, prioritize review sites
    REVIEW_DOMAINS = ["yelp.com", "tripadvisor.com", "google.com/maps"]
    seen = set()
    review = []
    social = []
    others = []
    for r in all_results:
        url = r.get("url", "")
        if url in seen:
            continue
        seen.add(url)
        if any(d in url for d in REVIEW_DOMAINS):
            review.append(r)
        elif any(d in url for d in ["facebook.com", "instagram.com"]):
            social.append(r)
        else:
            others.append(r)

    return (review + social + others)[:10]


def _groq_judge(name: str, city: str, snippets: list[dict],
                website_alive: bool | None) -> dict:
    """Evidence-based reasoning via Groq LLM. Returns structured verdict."""
    if not GROQ_API_KEY:
        return {"ai_status": "UNKNOWN", "ai_confidence": 0,
                "ai_evidence": [], "ai_reasoning": ""}

    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    context = "\n".join(
        f"{i}. [{s.get('url', '')}]\n"
        f"   Title: {s.get('title', '')}\n"
        f"   Snippet: {s.get('description', '')}"
        for i, s in enumerate(snippets[:8], 1)
    )

    web_status = ("LIVE (responds with 200)" if website_alive is True
                  else "DOWN or EXPIRED" if website_alive is False
                  else "no website on file")

    from datetime import date
    today = date.today().isoformat()

    prompt = f"""Determine whether this business is currently operating or permanently closed.

BUSINESS: "{name}"
CITY: {city}
WEBSITE STATUS: {web_status}
DATE: {today}

SEARCH RESULTS:
{context if context.strip() else "(no search results found)"}

TASK: Extract evidence from EACH search result. Then classify.

STEP 1 — For each result, extract ONE of these signals:
  CLOSURE: "permanently closed", "out of business", "shut down", Yelp/TripAdvisor CLOSED label
  ACTIVE: recent reviews (last 12 months), current hours, active social media, booking available
  REPLACED: different business now at same address
  STALE: old directory listing, no dates, generic info (IGNORE these)
  IRRELEVANT: wrong business, different location (IGNORE these)

STEP 2 — Count your signals:
  How many CLOSURE signals? How many ACTIVE signals? How many REPLACED?

STEP 3 — Apply these rules IN ORDER:
  1. If Yelp/TripAdvisor/Google title contains "CLOSED" AND other sources also confirm closure → PERMANENTLY_CLOSED (confidence 85-95)
  2. If Yelp title says "CLOSED" but the business has a LIVE website with current content OR recent social media activity → check carefully. The Yelp CLOSED label might be for a different location or outdated. If other evidence shows activity, say OPEN (confidence 60-70).
  3. If 2+ independent sources say closed → PERMANENTLY_CLOSED (confidence 80-90)
  4. If a news article confirms closure → PERMANENTLY_CLOSED (confidence 75-90)
  5. If a different business is now at the same address → LIKELY_CLOSED (confidence 70-80)
  6. If recent reviews describe visits within 12 months → OPEN (confidence 75-90)
  7. If website has current content (menus, prices, dates from this year) → OPEN (confidence 60-75)
  8. If Yelp/TripAdvisor page exists with hours and NO closed label → OPEN (confidence 55-70)
  9. If only stale directory listings found → UNKNOWN (confidence 30-40)
  10. If no results found at all → UNKNOWN (confidence 20-30)

CRITICAL RULES:
- Do NOT count directory sites (MapQuest, Manta, YellowPages, Birdeye, Chamberofcommerce) as evidence of ANYTHING. They persist for years after closure.
- A live website ALONE is weak evidence. Many closed businesses keep websites up.
- A single Yelp "CLOSED" label is NOT conclusive — Yelp sometimes tags the wrong location or has stale data. Look for corroborating evidence before saying PERMANENTLY_CLOSED.
- Do NOT contradict yourself. If evidence is unreliable, do not use it.
- Only use facts from the search results. Do not hallucinate.

Return EXACTLY this JSON (no markdown, no explanation outside JSON):
{{"status": "OPEN|LIKELY_CLOSED|PERMANENTLY_CLOSED|UNKNOWN", "confidence": 0-100, "key_evidence": ["evidence1", "evidence2"], "reasoning_summary": "one sentence"}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.05, max_tokens=300,
        )
        import json
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        result = json.loads(raw)

        status = result.get("status", "UNKNOWN")
        reasoning = result.get("reasoning_summary", "")
        evidence = result.get("key_evidence", [])
        if reasoning:
            log.info("    AI: %s | %s", status, reasoning)
        if evidence:
            for e in evidence[:3]:
                log.info("      evidence: %s", e)

        return {
            "ai_status": status,
            "ai_confidence": int(result.get("confidence", 30)),
            "ai_evidence": evidence,
            "ai_reasoning": reasoning,
        }
    except Exception as e:
        log.warning("Groq failed for %s: %s", name, e)
        return {"ai_status": "UNKNOWN", "ai_confidence": 0,
                "ai_evidence": [], "ai_reasoning": str(e)}


def web_crawl_signal(name: str, city_label: str, website: str | None) -> dict:
    """Combine website check + Brave search + Groq AI into a single 0-1 score.

    Score mapping (direct from AI status — no manual tweaking):
      OPEN               → 0.50 + (confidence/100 * 0.45)  = 0.50 to 0.95
      UNKNOWN            → 0.50 (neutral)
      LIKELY_CLOSED      → 0.50 - (confidence/100 * 0.35)  = 0.15 to 0.50
      PERMANENTLY_CLOSED → 0.50 - (confidence/100 * 0.45)  = 0.05 to 0.50
    """
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        web_future = pool.submit(_check_website, website)
        brave_future = pool.submit(_brave_search, name, city_label)
        web = web_future.result()
        results = brave_future.result()

    n_results = len(results)

    # Run Groq AI — always, even with 0 results
    ai = _groq_judge(name, city_label, results, web["web_alive"])

    status = ai.get("ai_status", "UNKNOWN")
    conf = ai.get("ai_confidence", 0) or 0

    # Direct status → score mapping.
    # UNKNOWN leans slightly open (0.55) since most businesses ARE open.
    # This prevents thin-web-presence open businesses from falling below 60%.
    if status == "OPEN":
        score = 0.55 + (conf / 100) * 0.40
    elif status == "PERMANENTLY_CLOSED":
        score = 0.50 - (conf / 100) * 0.45
    elif status == "LIKELY_CLOSED":
        score = 0.50 - (conf / 100) * 0.35
    else:  # UNKNOWN — lean open, most businesses are open
        score = 0.55

    # Website check: tiny nudge only
    if web["web_alive"] is True:
        score += 0.02
    elif web["web_alive"] is False:
        score -= 0.02

    score = max(0.0, min(1.0, score))

    # Map back to ai_judgment for compatibility with ensemble combiner + feedback
    if status in ("PERMANENTLY_CLOSED", "LIKELY_CLOSED"):
        ai_judgment = "CLOSED"
    elif status == "OPEN":
        ai_judgment = "OPEN"
    else:
        ai_judgment = "UNCERTAIN"

    return {
        "web_score": round(score, 3),
        "web_alive": web["web_alive"],
        "web_status_code": web["web_status_code"],
        "brave_results": n_results,
        "ai_judgment": ai_judgment,
        "ai_confidence": conf,
    }


# ══════════════════════════════════════════════════════════════════════
# Ensemble combiner
# ══════════════════════════════════════════════════════════════════════

def combine_signals(model: dict, osm: dict, web: dict) -> dict:
    """Combine Model + OSM + Web Crawl into a single open probability percentage.

    Google is ground truth — it is NOT a prediction signal.
    Uses weighted average of available signals. If a signal is missing (None),
    its weight is redistributed proportionally to the remaining signals.
    """
    model_score = model.get("model_score")

    # If the model is uncertain (score between 0.40-0.60), it's basically
    # guessing — treat it as None so its weight goes to web/OSM instead.
    # Only let the model vote when it has a strong opinion.
    if model_score is not None and 0.40 <= model_score <= 0.60:
        model_score = None  # model is guessing, don't let it drag others down

    signals = {
        "model": (model_score,              W_MODEL),
        "osm":   (osm.get("osm_score"),    W_OSM),
        "web":   (web.get("web_score"),     W_WEB),
    }

    available = {k: (score, weight) for k, (score, weight) in signals.items()
                 if score is not None}

    if not available:
        return {"ensemble_score": None, "ensemble_pct": None, "ensemble_label": "unknown",
                "signals_used": 0}

    total_weight = sum(w for _, w in available.values())
    ensemble = sum(score * (w / total_weight) for score, w in available.values())
    ensemble = max(0.0, min(1.0, ensemble))

    pct = round(ensemble * 100, 1)

    if pct >= 60:
        label = "likely_open"
    elif pct >= 40:
        label = "uncertain"
    else:
        label = "likely_closed"

    return {
        "ensemble_score": round(ensemble, 4),
        "ensemble_pct": pct,
        "ensemble_label": label,
        "signals_used": len(available),
    }


# ══════════════════════════════════════════════════════════════════════
# DB storage
# ══════════════════════════════════════════════════════════════════════

ENSURE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions.ensemble (
    overture_id    TEXT PRIMARY KEY,
    city           TEXT NOT NULL,
    model_score    REAL,
    model_pred_open BOOLEAN,
    osm_score      REAL,
    osm_found      BOOLEAN,
    osm_name       TEXT,
    osm_disused    BOOLEAN,
    osm_building   BOOLEAN,
    osm_businesses_here INTEGER,
    web_score      REAL,
    web_alive      BOOLEAN,
    brave_results  INTEGER,
    ai_judgment    TEXT,
    ai_confidence  INTEGER,
    ensemble_score REAL,
    ensemble_pct   REAL,
    ensemble_label TEXT,
    signals_used   INTEGER,
    created_at     TIMESTAMP DEFAULT NOW()
);
"""


def _ensure_table():
    with engine.begin() as conn:
        conn.execute(text(ENSURE_TABLE_SQL))


def store_result(conn, r: dict):
    conn.execute(text("""
        INSERT INTO predictions.ensemble
            (overture_id, city, model_score, model_pred_open,
             osm_score, osm_found, osm_name, osm_disused, osm_building, osm_businesses_here,
             web_score, web_alive, brave_results, ai_judgment, ai_confidence,
             ensemble_score, ensemble_pct, ensemble_label, signals_used)
        VALUES
            (:overture_id, :city, :model_score, :model_pred_open,
             :osm_score, :osm_found, :osm_name, :osm_disused, :osm_building, :osm_businesses_here,
             :web_score, :web_alive, :brave_results, :ai_judgment, :ai_confidence,
             :ensemble_score, :ensemble_pct, :ensemble_label, :signals_used)
        ON CONFLICT (overture_id) DO UPDATE SET
            model_score = EXCLUDED.model_score,
            model_pred_open = EXCLUDED.model_pred_open,
            osm_score = EXCLUDED.osm_score,
            osm_found = EXCLUDED.osm_found,
            osm_name = EXCLUDED.osm_name,
            osm_disused = EXCLUDED.osm_disused,
            osm_building = EXCLUDED.osm_building,
            osm_businesses_here = EXCLUDED.osm_businesses_here,
            web_score = EXCLUDED.web_score,
            web_alive = EXCLUDED.web_alive,
            brave_results = EXCLUDED.brave_results,
            ai_judgment = EXCLUDED.ai_judgment,
            ai_confidence = EXCLUDED.ai_confidence,
            ensemble_score = EXCLUDED.ensemble_score,
            ensemble_pct = EXCLUDED.ensemble_pct,
            ensemble_label = EXCLUDED.ensemble_label,
            signals_used = EXCLUDED.signals_used,
            created_at = NOW()
    """), r)


# ══════════════════════════════════════════════════════════════════════
# Feedback loop: save high-confidence web results as training labels
# ══════════════════════════════════════════════════════════════════════

FEEDBACK_PATH = PROJECT_ROOT / "feedback_labels.parquet"

# Minimum confidence thresholds to accept a web crawl result as training data
FEEDBACK_MIN_CONFIDENCE = 85  # AI must be >= 85% confident (stricter = cleaner labels)
FEEDBACK_MIN_WEB_SCORE_OPEN = 0.85   # web score >= 0.85 to label as open
FEEDBACK_MAX_WEB_SCORE_CLOSED = 0.15  # web score <= 0.15 to label as closed


def _save_feedback_labels(results: list[dict], city: str):
    """Save high-confidence ensemble results as silver training labels.

    Only saves when the web crawling signal (Brave+Groq) is highly confident,
    so we don't feed noisy labels back into the model.

    Each saved record includes the Overture raw_json so we can extract features
    at training time without needing the DB.
    """
    import json as _json

    new_labels = []

    for r in results:
        web_score = r.get("web_score")
        ai_conf = r.get("ai_confidence", 0) or 0
        ai_judgment = r.get("ai_judgment", "UNCERTAIN")
        ov_id = r.get("overture_id")

        if web_score is None or ai_judgment == "UNCERTAIN" or ai_judgment == "SKIPPED":
            continue
        if ai_conf < FEEDBACK_MIN_CONFIDENCE:
            continue

        # Determine label from web signal
        if ai_judgment == "OPEN" and web_score >= FEEDBACK_MIN_WEB_SCORE_OPEN:
            label = 1  # open
        elif ai_judgment == "CLOSED" and web_score <= FEEDBACK_MAX_WEB_SCORE_CLOSED:
            label = 0  # closed
        else:
            continue  # not confident enough

        # Fetch Overture raw_json from DB to store alongside label
        try:
            with engine.connect() as conn:
                row = conn.execute(text("""
                    SELECT id, confidence, raw_json
                    FROM overture.places WHERE id = :oid
                """), {"oid": ov_id}).fetchone()

            if not row:
                continue

            rj = row[2]
            if isinstance(rj, str):
                try:
                    rj = _json.loads(rj)
                except (ValueError, TypeError):
                    rj = {}

            record = {
                "id": ov_id,
                "label": label,
                "city": city,
                "confidence": row[1],
                "web_score": web_score,
                "ai_judgment": ai_judgment,
                "ai_confidence": ai_conf,
                "osm_found": r.get("osm_found", False),
                "osm_disused": r.get("osm_disused", False),
                "osm_building": r.get("osm_building", False),
                "osm_businesses_here": r.get("osm_businesses_here", 0),
                "web_alive": r.get("web_alive"),
            }

            # Store Overture fields as JSON strings (same format as training parquet)
            if isinstance(rj, dict):
                for key in ("sources", "names", "categories", "websites", "socials",
                             "emails", "phones", "brand", "addresses"):
                    val = rj.get(key)
                    if val is not None:
                        record[key] = _json.dumps(val) if not isinstance(val, str) else val
                        record[f"base_{key}"] = record[key]
                    else:
                        record[key] = None
                        record[f"base_{key}"] = None

            record["base_confidence"] = row[1]
            new_labels.append(record)

        except Exception as e:
            log.warning("  Feedback: failed to fetch %s: %s", ov_id, e)
            continue

    if not new_labels:
        log.info("Feedback: no high-confidence labels to save this run")
        return

    new_df = pd.DataFrame(new_labels)

    # Merge with existing feedback file
    if FEEDBACK_PATH.exists():
        existing = pd.read_parquet(FEEDBACK_PATH)
        # Deduplicate by overture_id — newer results overwrite older ones
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["id"], keep="last")
    else:
        combined = new_df

    combined.to_parquet(FEEDBACK_PATH, index=False)

    n_open = (combined["label"] == 1).sum()
    n_closed = (combined["label"] == 0).sum()
    log.info("Feedback: saved %d new labels (%d total: %d open, %d closed) → %s",
             len(new_labels), len(combined), n_open, n_closed, FEEDBACK_PATH.name)


# ══════════════════════════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════════════════════════

def run_ensemble(city: str, skip_web: bool = False):
    """Run Model + OSM + Web Crawl on ground-truth businesses, combine, evaluate vs Google."""
    key = ALIASES.get(city, city)
    city_label = CITY_NAMES.get(key, key.replace("_", " ").title())

    log.info("=" * 60)
    log.info("ENSEMBLE SCORING: %s (%s)", key, city_label)
    if skip_web:
        log.info("Prediction signals: Model + OSM Overpass (web SKIPPED)")
    else:
        log.info("Prediction signals: Model + OSM Overpass + Web Crawling")
    log.info("Ground truth labels (from step7)")
    log.info("Weights: Model=%.0f%% OSM=%.0f%% Web=%.0f%%",
             W_MODEL * 100, W_OSM * 100, W_WEB * 100)
    log.info("=" * 60)

    _ensure_table()

    # Check if model is available
    has_model = MODEL_PATH.exists()
    if has_model:
        log.info("  Model loaded from %s", MODEL_PATH.name)
    else:
        log.warning("  No model found — will use OSM + Web only")

    # Get businesses to score — those with ground truth labels
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT g.overture_id, o.name, o.latitude, o.longitude,
                   o.website, g.is_open AS actual_open
            FROM ground_truth.labels g
            JOIN overture.places o ON o.id = g.overture_id
            WHERE g.city = :city
            ORDER BY o.name
        """), {"city": key}).fetchall()

    if not rows:
        log.warning("No ground truth data for %s. Run step7 first.", key)
        return []

    log.info("  %d businesses to score", len(rows))
    results = []

    for i, (ov_id, name, lat, lon, website, actual_open) in enumerate(rows):
        log.info("\n[%d/%d] %s", i + 1, len(rows), name)

        # Step 1: Get OSM + website check first (model needs these as features)
        from concurrent.futures import ThreadPoolExecutor
        from src.step8_ensemble import _check_website
        with ThreadPoolExecutor(max_workers=2) as pool:
            o_future = pool.submit(osm_signal, name, lat, lon)
            web_check_future = pool.submit(_check_website, website)
            o = o_future.result()
            web_check = web_check_future.result()

        # Step 2: Run model WITH cross-source features baked in
        if has_model:
            m = model_signal(ov_id, city,
                             osm_result=o,
                             website_alive=web_check.get("web_alive"))
        else:
            m = {"model_score": None, "model_pred_open": None}

        # Step 3: Web crawl signal (Brave + Groq) if enabled
        if skip_web:
            w = {"web_score": None, "web_alive": web_check.get("web_alive"),
                 "web_status_code": web_check.get("web_status_code"),
                 "brave_results": 0, "ai_judgment": "SKIPPED", "ai_confidence": 0}
        else:
            w = web_crawl_signal(name, city_label, website)

        if m["model_score"] is not None:
            log.info("  Model: %.3f (pred=%s)", m["model_score"],
                     "open" if m["model_pred_open"] else "closed")
        else:
            log.info("  Model: n/a")
        log.info("  OSM:   %.2f (found=%s, building=%s, businesses_here=%d, name=%s, disused=%s)",
                 o["osm_score"] or 0, o["osm_found"], o["osm_building"],
                 o["osm_businesses_here"], o["osm_name"], o["osm_disused"])
        if not skip_web:
            log.info("  Web:   %.2f (alive=%s, brave=%d, AI=%s %d%%)",
                     w["web_score"], w["web_alive"], w["brave_results"],
                     w["ai_judgment"], w["ai_confidence"])
        else:
            log.info("  Web:   SKIPPED")

        # Combine all 3 (no Google — that's ground truth)
        ens = combine_signals(m, o, w)
        predicted = "OPEN" if (ens["ensemble_pct"] or 0) >= 60 else "CLOSED"
        actual = "OPEN" if actual_open else "CLOSED"
        correct = predicted == actual
        log.info("  => ENSEMBLE: %.1f%% (%s) | Predicted: %s | Ground truth: %s %s",
                 ens["ensemble_pct"] or 0, ens["ensemble_label"],
                 predicted, actual, "+" if correct else "X")

        result = {
            "overture_id": ov_id,
            "city": key,
            "name": name,
            "actual_open": bool(actual_open),
            **m, **o, **w, **ens,
        }
        results.append(result)

        # Rate limit: only needed when calling Brave/Groq APIs
        if not skip_web:
            time.sleep(1.3)

    # Store all results
    log.info("\nStoring %d ensemble results...", len(results))
    with engine.begin() as conn:
        for r in results:
            store_result(conn, r)

    # ── Save high-confidence results as feedback labels for model retraining ──
    if not skip_web:
        _save_feedback_labels(results, key)

    # ── Accuracy vs ground truth ──
    log.info("\n" + "=" * 60)
    log.info("RESULTS FOR %s", key.upper())
    log.info("=" * 60)

    with engine.connect() as conn:
        gt_rows = conn.execute(text(
            "SELECT overture_id, is_open FROM ground_truth.labels WHERE city = :city"
        ), {"city": key}).fetchall()
    gt = {r[0]: r[1] for r in gt_rows}

    correct = 0
    total = 0
    for r in results:
        actual_open = gt.get(r["overture_id"])
        if actual_open is None or r["ensemble_pct"] is None:
            continue
        predicted_open = r["ensemble_pct"] >= 60
        actual_open = bool(actual_open)
        if predicted_open == actual_open:
            correct += 1
        total += 1

    if total > 0:
        acc = 100 * correct / total
        log.info("Ensemble accuracy vs Google: %d/%d = %.1f%%", correct, total, acc)

        # Per-signal accuracy
        for signal_name, score_key in [("Model", "model_score"), ("OSM", "osm_score"), ("Web", "web_score")]:
            sig_correct = 0
            sig_total = 0
            for r in results:
                actual_open = gt.get(r["overture_id"])
                if actual_open is None or r.get(score_key) is None:
                    continue
                pred = r[score_key] >= 0.5
                if pred == bool(actual_open):
                    sig_correct += 1
                sig_total += 1
            if sig_total > 0:
                log.info("  %s alone: %d/%d = %.1f%%", signal_name,
                         sig_correct, sig_total, 100 * sig_correct / sig_total)

    # Distribution
    likely_open = sum(1 for r in results if r.get("ensemble_label") == "likely_open")
    uncertain = sum(1 for r in results if r.get("ensemble_label") == "uncertain")
    likely_closed = sum(1 for r in results if r.get("ensemble_label") == "likely_closed")
    log.info("Distribution: %d likely_open, %d uncertain, %d likely_closed",
             likely_open, uncertain, likely_closed)

    # ── Labeled results table — shows every business with ground truth ──
    log.info("\n" + "=" * 60)
    log.info("LABELED RESULTS TABLE")
    log.info("=" * 60)
    log.info("%-4s %-35s %-8s %-8s %-6s %-7s %-5s %-10s %s",
             "#", "Business", "Actual", "Predict", "Ens%", "Web", "AI%", "AI_Judge", "Result")
    log.info("-" * 110)

    errors_open_as_closed = []  # false closed (actually open)
    errors_closed_as_open = []  # false open (actually closed)

    for i, r in enumerate(results, 1):
        actual_open = r.get("actual_open")
        ens_pct = r.get("ensemble_pct", 0) or 0
        predicted_open = ens_pct >= 60
        actual_str = "OPEN" if actual_open else "CLOSED"
        pred_str = "OPEN" if predicted_open else "CLOSED"
        web_score = r.get("web_score")
        web_str = f"{web_score:.2f}" if web_score is not None else "n/a"
        ai_conf = r.get("ai_confidence", 0) or 0
        ai_judge = r.get("ai_judgment", "n/a")
        match = "OK" if predicted_open == actual_open else "WRONG"
        name = r.get("name", "???")[:35]

        log.info("%-4d %-35s %-8s %-8s %-6.1f %-7s %-5d %-10s %s",
                 i, name, actual_str, pred_str, ens_pct, web_str,
                 ai_conf, ai_judge, match)

        if match == "WRONG":
            if actual_open and not predicted_open:
                errors_open_as_closed.append(r)
            else:
                errors_closed_as_open.append(r)

    # ── Error analysis ──
    if errors_open_as_closed or errors_closed_as_open:
        log.info("\n" + "-" * 60)
        log.info("ERROR ANALYSIS")
        log.info("-" * 60)

        if errors_closed_as_open:
            log.info("\n  FALSE OPEN (actually closed, we said open) — %d errors:",
                     len(errors_closed_as_open))
            for r in errors_closed_as_open:
                log.info("    - %s | web=%.2f ai=%s(%d%%) brave=%d alive=%s",
                         r.get("name", "???"),
                         r.get("web_score", 0),
                         r.get("ai_judgment", "?"),
                         r.get("ai_confidence", 0) or 0,
                         r.get("brave_results", 0),
                         r.get("web_alive"))

        if errors_open_as_closed:
            log.info("\n  FALSE CLOSED (actually open, we said closed) — %d errors:",
                     len(errors_open_as_closed))
            for r in errors_open_as_closed:
                log.info("    - %s | web=%.2f ai=%s(%d%%) brave=%d alive=%s",
                         r.get("name", "???"),
                         r.get("web_score", 0),
                         r.get("ai_judgment", "?"),
                         r.get("ai_confidence", 0) or 0,
                         r.get("brave_results", 0),
                         r.get("web_alive"))

        log.info("\n  PATTERN: %d false-open, %d false-closed",
                 len(errors_closed_as_open), len(errors_open_as_closed))

    return results


def run(args: list[str] | None = None):
    if not args:
        print("Usage: python -m src.step8_ensemble sf|nyc|chi|all [--no-web]")
        return

    skip_web = "--no-web" in args
    city_args = [a for a in args if not a.startswith("--")]

    if not city_args:
        print("Usage: python -m src.step8_ensemble sf|nyc|chi|all [--no-web]")
        return

    if city_args[0] == "all":
        with engine.connect() as conn:
            cities = conn.execute(text(
                "SELECT DISTINCT city FROM ground_truth.labels"
            )).fetchall()
            targets = [r[0] for r in cities]
    else:
        targets = [ALIASES.get(a, a) for a in city_args]

    for city in targets:
        run_ensemble(city, skip_web=skip_web)


if __name__ == "__main__":
    run(sys.argv[1:])
