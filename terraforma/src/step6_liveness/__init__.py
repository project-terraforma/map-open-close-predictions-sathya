"""
Step 6b: Liveness scoring — website HEAD, Brave Search + Groq AI, OSM Overpass.

Checks three independent signals for each ground-truth business:
  1. Website HTTP HEAD check (from Overture website field)
  2. Brave Search API → snippets sent to Groq LLM for reasoning about open/closed
  3. OSM Overpass API — nearby POI match within 50 m

Groq (Llama 3.1 70B) reads the actual web snippets and reasons about whether
the business is open or closed, instead of dumb keyword matching.

Usage:
    python -m src.step6_liveness sf          # run for SF ground truth
    python -m src.step6_liveness all         # run for all cities
"""

import json
import logging
import math
import sys
import time

import requests
from groq import Groq
from sqlalchemy import text

from src.config import engine, BRAVE_SEARCH_KEY, GROQ_API_KEY

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ALIASES = {
    "sf": "san_francisco", "san_francisco": "san_francisco",
    "nyc": "new_york", "new_york": "new_york",
    "chi": "chicago", "chicago": "chicago",
    "paris": "paris", "london": "london", "ldn": "london",
    "sg": "singapore", "singapore": "singapore",
}

CITY_NAMES = {
    "san_francisco": "San Francisco",
    "new_york": "New York",
    "chicago": "Chicago",
    "paris": "Paris",
    "london": "London",
    "singapore": "Singapore",
}

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SOCIAL_DOMAINS = {"facebook.com", "instagram.com", "twitter.com", "x.com",
                  "linkedin.com", "tiktok.com", "youtube.com"}
DIRECTORY_DOMAINS = {"yelp.com", "tripadvisor.com", "tripadvisor.fr",
                     "foursquare.com", "yellowpages.com", "bbb.org",
                     "pagesjaunes.fr", "trustpilot.com", "google.com"}


# ── Signal 1: Website HEAD ──────────────────────────────────────────

def check_website(url: str) -> dict:
    """HTTP HEAD on the website URL. Returns alive/dead + status code."""
    if not url:
        return {"website_alive": None, "website_status_code": None}
    try:
        resp = requests.head(url, timeout=8, allow_redirects=True,
                             headers={"User-Agent": "TerraForma/1.0 (research)"})
        alive = resp.status_code < 400
        return {"website_alive": alive, "website_status_code": resp.status_code}
    except requests.exceptions.SSLError:
        try:
            resp = requests.head(url, timeout=8, allow_redirects=True, verify=False,
                                 headers={"User-Agent": "TerraForma/1.0 (research)"})
            alive = resp.status_code < 400
            return {"website_alive": alive, "website_status_code": resp.status_code}
        except Exception:
            return {"website_alive": False, "website_status_code": 0}
    except Exception:
        return {"website_alive": False, "website_status_code": 0}


# ── Signal 2: Brave Search + Groq AI ────────────────────────────────

def _brave_query(query: str) -> list[dict]:
    """Run a single Brave Search API call and return the result list."""
    try:
        resp = requests.get("https://api.search.brave.com/res/v1/web/search",
                            params={"q": query, "count": 10},
                            headers={"X-Subscription-Token": BRAVE_SEARCH_KEY,
                                     "Accept": "application/json"},
                            timeout=10)
        data = resp.json()
        return data.get("web", {}).get("results", [])
    except Exception as e:
        log.warning("Brave query failed for '%s': %s", query, e)
        return []


def _groq_judge(name: str, city: str, snippets: list[dict],
                website_alive: bool | None) -> dict:
    """Send Brave snippets to Groq LLM to reason about business status.

    Returns: {"ai_judgment": "OPEN"|"CLOSED"|"UNCERTAIN",
              "ai_confidence": 0-100,
              "ai_reason": "one line explanation"}
    """
    if not groq_client or not snippets:
        return {"ai_judgment": "UNCERTAIN", "ai_confidence": 0,
                "ai_reason": "no data"}

    # Build context from search results
    context_lines = []
    for i, s in enumerate(snippets[:10], 1):
        title = s.get("title", "")
        desc = s.get("description", "")
        url = s.get("url", "")
        context_lines.append(f"{i}. [{title}]({url})\n   {desc}")

    context = "\n".join(context_lines)

    # Website status context
    web_status = "no website listed"
    if website_alive is True:
        web_status = "website is LIVE and responding"
    elif website_alive is False:
        web_status = "website is DOWN / returns error"

    prompt = f"""You are analyzing web search results to determine if a business is currently open/operating or permanently closed.

Business: "{name}" in {city}
Website status: {web_status}

Web search results:
{context}

Based on these search results, is this business currently OPEN (still operating) or CLOSED (permanently shut down)?

IMPORTANT:
- "Closed on Sundays" or "closed for holidays" means the business IS open (just has hours)
- "Permanently closed" on Google/Yelp/TripAdvisor = actually closed
- A live website with current content = likely open
- Old/stale results with no recent activity = uncertain
- Reviews or listings that mention past tense ("was", "used to be") = likely closed

Respond with EXACTLY this JSON format, nothing else:
{{"judgment": "OPEN" or "CLOSED" or "UNCERTAIN", "confidence": 0-100, "reason": "one sentence explanation"}}"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()

        # Parse JSON from response
        # Handle cases where model wraps in markdown code blocks
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
        return {
            "ai_judgment": result.get("judgment", "UNCERTAIN"),
            "ai_confidence": int(result.get("confidence", 50)),
            "ai_reason": result.get("reason", "")[:200],
        }
    except Exception as e:
        log.warning("Groq judgment failed for %s: %s", name, e)
        return {"ai_judgment": "UNCERTAIN", "ai_confidence": 0,
                "ai_reason": f"error: {e}"}


def brave_search(name: str, city_label: str, website_alive: bool | None) -> dict:
    """Search Brave for the business, then use Groq AI to reason about status.

    1 Brave API call per business → snippets → Groq LLM judgment.
    """
    if not BRAVE_SEARCH_KEY:
        log.warning("BRAVE_SEARCH_KEY not set, skipping Brave")
        return {
            "brave_result_count": None, "brave_snippet_summary": None,
            "brave_has_social": None, "brave_has_directory": None,
            "ai_judgment": "UNCERTAIN", "ai_confidence": 0, "ai_reason": "no key",
        }

    # Single Brave query — general presence
    query = f'"{name}" "{city_label}"'
    results = _brave_query(query)

    result_count = len(results)
    has_social = False
    has_directory = False
    snippets = []

    for r in results:
        url = r.get("url", "")
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.replace("www.", "")
            if domain in SOCIAL_DOMAINS:
                has_social = True
            if domain in DIRECTORY_DOMAINS:
                has_directory = True
        except Exception:
            pass

        snippets.append(r.get("description", "")[:150])

    summary = " | ".join(snippets[:3]) if snippets else "no results"

    # Send to Groq for AI reasoning
    ai = _groq_judge(name, city_label, results, website_alive)
    log.info("    Groq: %s (%d%%) — %s",
             ai["ai_judgment"], ai["ai_confidence"], ai["ai_reason"])

    return {
        "brave_result_count": result_count,
        "brave_snippet_summary": summary[:500],
        "brave_has_social": has_social,
        "brave_has_directory": has_directory,
        **ai,
    }


# ── Signal 3: OSM Overpass ──────────────────────────────────────────

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def osm_check(name: str, lat: float, lon: float) -> dict:
    """Query OSM Overpass for POIs within 50m, check for name match."""
    query = f"""
    [out:json][timeout:10];
    (
      node["name"](around:50,{lat},{lon});
      way["name"](around:50,{lat},{lon});
    );
    out body;
    """

    data = None
    for server in OVERPASS_SERVERS:
        try:
            resp = requests.post(server, data={"data": query}, timeout=15)
            if resp.status_code == 429:
                log.warning("OSM rate limited on %s, trying next", server)
                time.sleep(2)
                continue
            if resp.status_code != 200:
                log.warning("OSM HTTP %d on %s", resp.status_code, server)
                continue
            data = resp.json()
            break
        except Exception as e:
            log.warning("OSM failed on %s for %s: %s", server, name, e)
            continue

    if data is None:
        log.warning("OSM: all servers failed for %s", name)
        return {"osm_found": False, "osm_name": None,
                "osm_distance_m": None, "osm_disused": False}

    elements = data.get("elements", [])
    name_lower = name.lower().strip()

    best_match = None
    best_dist = 999

    for el in elements:
        tags = el.get("tags", {})
        osm_name = tags.get("name", "")
        disused_name = tags.get("disused:name", "")
        is_disused = any(k.startswith("disused:") for k in tags)

        el_lat = el.get("lat") or el.get("center", {}).get("lat")
        el_lon = el.get("lon") or el.get("center", {}).get("lon")

        if el_lat and el_lon:
            dist = _haversine(lat, lon, el_lat, el_lon)
        else:
            dist = 50

        check_name = osm_name or disused_name
        if check_name and _name_match(name_lower, check_name.lower()):
            if dist < best_dist:
                best_match = {
                    "osm_found": True,
                    "osm_name": check_name,
                    "osm_distance_m": round(dist, 1),
                    "osm_disused": is_disused,
                }
                best_dist = dist

    if best_match:
        return best_match

    return {"osm_found": False, "osm_name": None,
            "osm_distance_m": None, "osm_disused": False}


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _name_match(a: str, b: str) -> bool:
    if a in b or b in a:
        return True
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
    return overlap >= 0.5


# ── Composite scoring ───────────────────────────────────────────────

def compute_liveness_score(signals: dict) -> tuple[float, str]:
    """
    Combine signals into 0-1 score and label (alive / uncertain / dead).

    Groq AI judgment is the primary signal — it actually reads and reasons
    about the web snippets. Website alive/dead and OSM are supporting signals.
    """
    score = 0.5  # neutral start

    # Groq AI judgment — primary signal (it actually understands context)
    judgment = signals.get("ai_judgment", "UNCERTAIN")
    ai_conf = signals.get("ai_confidence", 0) or 0

    if judgment == "OPEN":
        # Scale by AI confidence: high confidence = strong signal
        score += 0.25 * (ai_conf / 100)
    elif judgment == "CLOSED":
        score -= 0.30 * (ai_conf / 100)
    # UNCERTAIN: no change from AI

    # Website alive/dead — strong supporting signal
    if signals.get("website_alive") is True:
        score += 0.15
    elif signals.get("website_alive") is False:
        score -= 0.12

    # Brave result count — mild signal
    rc = signals.get("brave_result_count")
    if rc is not None:
        if rc == 0:
            score -= 0.08  # no web presence at all
        elif rc >= 5:
            score += 0.03

    # OSM — independent confirmation
    if signals.get("osm_found"):
        if signals.get("osm_disused"):
            score -= 0.15
        else:
            score += 0.08

    # Clamp
    score = max(0.0, min(1.0, score))

    # Label
    if score >= 0.55:
        label = "alive"
    elif score >= 0.40:
        label = "uncertain"
    else:
        label = "dead"

    return round(score, 3), label


# ── Main runner ─────────────────────────────────────────────────────

def run_liveness(city: str):
    """Run liveness checks for all ground truth businesses in a city."""
    key = ALIASES.get(city, city)
    city_label = CITY_NAMES.get(key, key.replace("_", " ").title())

    log.info("Running liveness checks for %s (%s)", key, city_label)

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT g.overture_id, o.name, o.latitude, o.longitude,
                   o.website, o.phone, g.is_open, g.business_status
            FROM ground_truth.labels g
            JOIN overture.places o ON o.id = g.overture_id
            WHERE g.city = :city
            ORDER BY g.is_open, o.name
        """), {"city": key}).fetchall()

    log.info("  %d businesses to check", len(rows))
    results = []

    for i, (ov_id, name, lat, lon, website, phone, is_open, biz_status) in enumerate(rows):
        log.info("  [%d/%d] %s", i + 1, len(rows), name)

        # Signal 1: Website
        web = check_website(website)
        log.info("    Website: %s (code=%s)",
                 "alive" if web["website_alive"] else "dead/none",
                 web["website_status_code"])

        # Signal 2: Brave + Groq AI
        brave = brave_search(name, city_label, web["website_alive"])
        log.info("    Brave: %d results, social=%s, dir=%s",
                 brave.get("brave_result_count", 0) or 0,
                 brave.get("brave_has_social"),
                 brave.get("brave_has_directory"))

        # Signal 3: OSM
        osm = osm_check(name, lat, lon)
        log.info("    OSM: found=%s name=%s dist=%s disused=%s",
                 osm["osm_found"], osm["osm_name"],
                 osm["osm_distance_m"], osm["osm_disused"])

        # Combine all signals
        all_signals = {**web, **brave, **osm, "website_url": website}
        score, label = compute_liveness_score(all_signals)
        log.info("    => Liveness: %.3f (%s) | Google: %s", score, label, biz_status)

        all_signals["overture_id"] = ov_id
        all_signals["city"] = key
        all_signals["liveness_score"] = score
        all_signals["liveness_label"] = label
        results.append(all_signals)

        # Rate limit: Brave 1 req/sec, Groq ~30 req/min on free tier
        time.sleep(1.5)

    # Store results
    log.info("Storing %d liveness results...", len(results))
    with engine.begin() as conn:
        for r in results:
            conn.execute(text(
                "DELETE FROM web_scores.liveness WHERE overture_id = :overture_id"
            ), {"overture_id": r["overture_id"]})
            conn.execute(text("""
                INSERT INTO web_scores.liveness
                    (overture_id, city, website_url, website_alive, website_status_code,
                     brave_result_count, brave_snippet_summary,
                     brave_has_social, brave_has_directory,
                     osm_found, osm_name, osm_distance_m, osm_disused,
                     liveness_score, liveness_label)
                VALUES
                    (:overture_id, :city, :website_url, :website_alive, :website_status_code,
                     :brave_result_count, :brave_snippet_summary,
                     :brave_has_social, :brave_has_directory,
                     :osm_found, :osm_name, :osm_distance_m, :osm_disused,
                     :liveness_score, :liveness_label)
            """), {
                "overture_id": r["overture_id"],
                "city": r["city"],
                "website_url": r.get("website_url"),
                "website_alive": r.get("website_alive"),
                "website_status_code": r.get("website_status_code"),
                "brave_result_count": r.get("brave_result_count"),
                "brave_snippet_summary": r.get("brave_snippet_summary"),
                "brave_has_social": r.get("brave_has_social"),
                "brave_has_directory": r.get("brave_has_directory"),
                "osm_found": r.get("osm_found"),
                "osm_name": r.get("osm_name"),
                "osm_distance_m": r.get("osm_distance_m"),
                "osm_disused": r.get("osm_disused"),
                "liveness_score": r["liveness_score"],
                "liveness_label": r["liveness_label"],
            })

    # Summary
    alive_count = sum(1 for r in results if r["liveness_label"] == "alive")
    dead_count = sum(1 for r in results if r["liveness_label"] == "dead")
    uncertain_count = sum(1 for r in results if r["liveness_label"] == "uncertain")
    log.info("Liveness summary for %s:", key)
    log.info("  alive=%d  uncertain=%d  dead=%d", alive_count, uncertain_count, dead_count)
    log.info("  avg score=%.3f", sum(r["liveness_score"] for r in results) / len(results))

    # Accuracy vs Google ground truth
    with engine.connect() as conn:
        gt_rows = conn.execute(text("""
            SELECT overture_id, is_open FROM ground_truth.labels WHERE city = :city
        """), {"city": key}).fetchall()
    gt = {r[0]: r[1] for r in gt_rows}

    correct = 0
    for r in results:
        google_open = gt.get(r["overture_id"])
        if google_open is None:
            continue
        if (r["liveness_label"] == "alive" and google_open) or \
           (r["liveness_label"] == "dead" and not google_open):
            correct += 1
    non_uncertain = [r for r in results if r["liveness_label"] != "uncertain"]
    if non_uncertain:
        log.info("  Accuracy (excl uncertain): %d/%d = %.1f%%",
                 correct, len(non_uncertain), 100 * correct / len(non_uncertain))

    return results


def run(args: list[str] | None = None):
    if not args:
        print("Usage: python -m src.step6_liveness sf|nyc|paris|london|all")
        return

    if args[0] == "all":
        with engine.connect() as conn:
            cities = conn.execute(text(
                "SELECT DISTINCT city FROM ground_truth.labels"
            )).fetchall()
            targets = [r[0] for r in cities]
    else:
        targets = [ALIASES.get(a, a) for a in args]

    for city in targets:
        run_liveness(city)


if __name__ == "__main__":
    run(sys.argv[1:])
