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


def model_signal(overture_id: str, city: str = "san_francisco") -> dict:
    """Run the CatBoost+LightGBM ensemble on a single Overture place.

    Uses current Overture snapshot features only (no delta comparison).
    The model was trained on current-snapshot features: confidence, sources,
    digital presence, recency/staleness, brand, category.
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
                    record[f"base_{key}"] = serialized  # base = current (no old snapshot)

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

    # Scoring logic:
    # 1. Found our business, active → strong open signal
    if best_match:
        score = 0.9 if best_match["dist"] < 25 else 0.75
        return {
            "osm_score": score,
            "osm_found": True,
            "osm_name": best_match["osm_name"],
            "osm_disused": False,
            "osm_building": has_building,
            "osm_businesses_here": n_businesses_here,
        }

    # 2. Found our business but marked disused → strong closed signal
    if disused_matches:
        return {
            "osm_score": 0.08,
            "osm_found": True,
            "osm_name": disused_matches[0]["osm_name"],
            "osm_disused": True,
            "osm_building": has_building,
            "osm_businesses_here": n_businesses_here,
        }

    # 3. Building exists with OTHER businesses but not ours → likely replaced/closed
    #    This is the only "not found" case that's actually a negative signal
    if has_building and other_businesses:
        nearby = [b for b in other_businesses if b["dist"] < 15]
        if nearby:
            score = 0.30  # different business at exact same storefront → likely closed
            return {
                "osm_score": score,
                "osm_found": False,
                "osm_name": f"replaced: {nearby[0]['name']} ({nearby[0]['type']})",
                "osm_disused": False,
                "osm_building": True,
                "osm_businesses_here": n_businesses_here,
            }

    # 4. Building exists but no businesses listed → neutral (OSM just doesn't have detail)
    if has_building:
        return {
            "osm_score": 0.50,
            "osm_found": False,
            "osm_name": None,
            "osm_disused": False,
            "osm_building": True,
            "osm_businesses_here": 0,
        }

    # 5. No building, no POIs → neutral (not in OSM doesn't mean closed)
    #    IMPORTANT: must be 0.50 neutral — absence of OSM data is NOT evidence of closure
    return {"osm_score": 0.50, "osm_found": False, "osm_name": None,
            "osm_disused": False, "osm_building": False, "osm_businesses_here": 0}


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
    """Run targeted Brave searches to get 3-5 high-signal pages.

    Priority sources (in order):
      1. Google Maps listing
      2. Yelp page
      3. TripAdvisor page
      4. Official website / general result
      5. News or directory
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

    # Query 1: Find the business's own page (official website, Facebook, etc.)
    q1 = f'"{name}" {city_label}'
    # Query 2: Yelp + TripAdvisor (most reliable open/closed status sources)
    q2 = f'"{name}" {city_label} site:yelp.com OR site:tripadvisor.com'
    # Query 3: Address-specific search for open/closed status
    q3 = f'"{name}" {city_label} "permanently closed" OR "closed" OR "open" OR "hours"'

    # Run all 3 queries in parallel
    with ThreadPoolExecutor(max_workers=3) as pool:
        f1 = pool.submit(_search, q1, 3)
        f2 = pool.submit(_search, q2, 3)
        f3 = pool.submit(_search, q3, 3)
        all_results = f1.result() + f2.result() + f3.result()

    # Deduplicate by URL, prioritize review sites first
    PRIORITY_DOMAINS = ["yelp.com", "tripadvisor.com"]
    seen = set()
    priority = []
    others = []
    for r in all_results:
        url = r.get("url", "")
        if url in seen:
            continue
        seen.add(url)
        if any(d in url for d in PRIORITY_DOMAINS):
            priority.append(r)
        else:
            others.append(r)

    # Return priority sources first, then general, capped at 8 total
    return (priority + others)[:8]


def _groq_judge(name: str, city: str, snippets: list[dict],
                website_alive: bool | None) -> dict:
    """Send search snippets to Groq LLM for open/closed reasoning."""
    if not GROQ_API_KEY or not snippets:
        return {"ai_judgment": "UNCERTAIN", "ai_confidence": 0}

    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    # Label each source by type for the AI
    def _source_type(url: str) -> str:
        if "yelp.com" in url: return "Yelp"
        if "google.com/maps" in url: return "Google Maps"
        if "tripadvisor.com" in url: return "TripAdvisor"
        if "bbb.org" in url: return "BBB Directory"
        if "facebook.com" in url: return "Facebook"
        if "instagram.com" in url: return "Instagram"
        return "General Web"

    context = "\n".join(
        f"{i}. [{_source_type(s.get('url', ''))}] {s.get('title', '')}\n"
        f"   URL: {s.get('url', '')}\n"
        f"   Snippet: {s.get('description', '')}"
        for i, s in enumerate(snippets[:6], 1)
    )

    web_status = ("website is LIVE" if website_alive is True
                  else "website is DOWN/EXPIRED" if website_alive is False
                  else "no website listed")

    from datetime import date
    today = date.today().isoformat()

    prompt = f"""You are an AI system that determines whether a business is currently operating.

You will be given web search results, page snippets, and extracted text from multiple websites about a business. Your task is to analyze this evidence and determine the most likely operational status.

Business Name: "{name}"
Location: {city}
Official website status: {web_status}
Today's date: {today}

Web Evidence:
{context}

Instructions:
Analyze the evidence and determine whether the business is:
- OPEN (currently operating)
- CLOSED (permanently closed, shut down, or out of business)
- UNCERTAIN (not enough evidence to decide)

Evaluate all evidence carefully and weigh the strength of different signals.

Strong PERMANENT CLOSURE signals:
* Yelp, TripAdvisor, or Google explicitly says "Permanently Closed" or "CLOSED" in the title or snippet
* News articles confirming shutdown or closure
* Government business registries showing the business as inactive or dissolved
* Official announcements from the business about closing
* The business's website domain has expired or shows a parked/for-sale page

Moderate closure signals:
* Multiple directories indicating the business is closed
* Social media accounts inactive for 2+ years
* Recent reviews (past 12 months) mentioning "this place is closed" or "no longer open"
* A different business name now appears at the same address

Strong OPEN signals:
* Recent reviews (within last 12 months) describing an actual visit experience
* Listed business hours on Yelp/TripAdvisor WITHOUT a "closed" label
* Active website with current content, menus, prices, or booking functionality
* Reservation or online ordering links that are functional
* Recent social media posts showing business activity

Review signals (important):
User reviews from platforms like Yelp and TripAdvisor can indicate whether the business is operating.
Consider:
* Reviews describing a recent visit experience
* Reviews posted within the last 6-12 months
* Multiple recent reviews suggesting customers are still visiting
A recent review describing an actual visit is STRONG evidence the business is still open.

CRITICAL reasoning rules:
1. A Yelp or TripAdvisor PAGE existing does NOT mean the business is open — closed businesses keep their pages up for years with all their old reviews. You MUST look for explicit "Permanently Closed" or "CLOSED" labels in the title/snippet.
2. If a major review platform (Yelp, TripAdvisor) explicitly marks it "Permanently Closed" or "CLOSED" in the snippet title, trust it — this is very reliable.
3. Prefer recent information over older information. A 2025-2026 source beats a 2020 source.
4. Prefer multiple independent sources over a single source.
5. If the search results are about a DIFFERENT business with a similar name, classify as UNCERTAIN.
6. If there is insufficient evidence to decide, classify as UNCERTAIN. Do NOT guess.
7. An active website alone is NOT enough to say OPEN — many closed businesses keep websites running.

Respond with EXACTLY this JSON, nothing else:
{{"judgment": "OPEN" or "CLOSED" or "UNCERTAIN", "confidence": 0-100, "reasoning": "brief explanation citing the key evidence"}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=300,
        )
        import json
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        result = json.loads(raw)
        reasoning = result.get("reasoning", "")
        if reasoning:
            log.info("    AI reasoning: %s", reasoning)
        return {
            "ai_judgment": result.get("judgment", "UNCERTAIN"),
            "ai_confidence": int(result.get("confidence", 50)),
        }
    except Exception as e:
        log.warning("Groq failed for %s: %s", name, e)
        return {"ai_judgment": "UNCERTAIN", "ai_confidence": 0}


def web_crawl_signal(name: str, city_label: str, website: str | None) -> dict:
    """Combine website check + Brave search + Groq AI into a single 0-1 score."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        web_future = pool.submit(_check_website, website)
        brave_future = pool.submit(_brave_search, name, city_label)
        web = web_future.result()
        results = brave_future.result()
    ai = _groq_judge(name, city_label, results, web["web_alive"])

    score = 0.5  # neutral start

    judgment = ai.get("ai_judgment", "UNCERTAIN")
    conf = ai.get("ai_confidence", 0) or 0

    # AI judgment is the primary driver — but scale with confidence
    if judgment == "OPEN":
        score += 0.40 * (conf / 100)
    elif judgment == "CLOSED":
        score -= 0.45 * (conf / 100)
    # UNCERTAIN: stay near 0.5 (slight pull toward closed if confidence > 0)
    elif conf > 50:
        score -= 0.05

    # Website alive is a WEAK signal — closed businesses keep websites up
    if web["web_alive"] is True:
        score += 0.03  # tiny bump — many closed businesses still have live sites
    elif web["web_alive"] is False:
        score -= 0.08  # dead website is a moderate closed signal

    n_results = len(results)
    if n_results == 0:
        score -= 0.08  # no web presence at all

    score = max(0.0, min(1.0, score))

    return {
        "web_score": round(score, 3),
        "web_alive": web["web_alive"],
        "web_status_code": web["web_status_code"],
        "brave_results": n_results,
        "ai_judgment": ai["ai_judgment"],
        "ai_confidence": ai["ai_confidence"],
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
    signals = {
        "model": (model.get("model_score"), W_MODEL),
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

        # Run signals in parallel (skip web if --no-web flag)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as pool:
            m_future = pool.submit(model_signal, ov_id, city) if has_model else None
            o_future = pool.submit(osm_signal, name, lat, lon)
            if not skip_web:
                w_future = pool.submit(web_crawl_signal, name, city_label, website)

            m = m_future.result() if m_future else {"model_score": None, "model_pred_open": None}
            o = o_future.result()
            if skip_web:
                w = {"web_score": None, "web_alive": None, "web_status_code": None,
                     "brave_results": 0, "ai_judgment": "SKIPPED", "ai_confidence": 0}
            else:
                w = w_future.result()

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
