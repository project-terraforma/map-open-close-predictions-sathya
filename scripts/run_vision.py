"""
Batch vision analysis: run OCR + XGBoost on Mapillary images for each location.
Downloads images, runs the model, and writes predictions back to test_data.json.

Multi-signal prediction system:
  Signal 1 (Foursquare): Directory cross-reference — verified/mismatch/no_data
  Signal 2 (Text):       OCR text matching — visual signal from street imagery
  Signal 3 (Closure):    OCR closure text detection — "for lease", "closed", etc.
  Signal 4 (Website):    Website liveness — alive/dead/redirect/parked
  Signal 5 (XGBoost):    Metadata model — calibrated Overture feature signal

Usage: python scripts/run_vision.py
"""

import os
import sys
import json
import tempfile
import requests
import time
import re
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from ocr_model import detect_text, get_ocr, normalize, get_brand_search_names
from check_website_liveness import check_website

# Accept --input arg for different city data files
_default_data = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'test_data.json')
DATA_PATH = _default_data
for _i, _arg in enumerate(sys.argv):
    if _arg == '--input' and _i + 1 < len(sys.argv):
        DATA_PATH = sys.argv[_i + 1]

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'xgboost_model.json')

# Closure-indicating text patterns (detected by OCR on storefront)
CLOSURE_KEYWORDS = [
    'for lease', 'for rent', 'for sale', 'space available', 'available now',
    'now leasing', 'now available', 'retail for lease', 'office for lease',
    'permanently closed', 'closed permanently', 'going out of business',
    'store closing', 'closing sale', 'everything must go', 'final sale',
    'coming soon', 'opening soon', 'under construction', 'under renovation',
    'moved to', 'we have moved', 'relocated', 'new location',
    'vacant', 'vacate', 'emptied',
    'rent me', 'lease me',
]

# Google Places API key (free tier: 100k requests/month)
GOOGLE_PLACES_API_KEY = os.environ.get('GOOGLE_PLACES_API_KEY', '')

# Must match the features used during training (10 base + 9 engineered)
BASE_FEATURES = [
    'overture_confidence', 'source_age_days', 'has_website', 'has_phone',
    'has_brand', 'address_complete', 'category_closure_rate', 'fields_populated',
    'yelp_rating', 'yelp_review_count',
]
ENGINEERED_FEATURES = [
    'missing_signals', 'data_completeness', 'low_confidence',
    'no_web_no_phone', 'confidence_x_fields', 'confidence_x_website',
    'old_source', 'high_quality', 'sparse_record',
]
FEATURES = BASE_FEATURES + ENGINEERED_FEATURES


def download_image(url, suffix=".jpg"):
    """Download image to temp file, return path."""
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        tmp = tempfile.mktemp(suffix=suffix)
        with open(tmp, 'wb') as f:
            f.write(resp.content)
        return tmp
    except Exception as e:
        print(f"      DL fail: {e}", flush=True)
        return None


def _longest_common_subsequence(a, b):
    """Length of longest common subsequence between two strings."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _char_overlap(a, b):
    """Fraction of characters in 'a' that appear in 'b' (order-independent)."""
    if not a:
        return 0.0
    from collections import Counter
    ca = Counter(a)
    cb = Counter(b)
    overlap = sum(min(ca[c], cb[c]) for c in ca)
    return overlap / len(a)


def detect_closure_text(detected_text):
    """Check if OCR text contains closure-indicating keywords.
    Returns: (keyword_matched, strength) or (None, None)
    strength: 'strong' for definitive closure signals, 'weak' for ambiguous ones
    """
    norm = normalize(detected_text)
    if not norm or len(norm) < 3:
        return None, None

    strong_signals = [
        'for lease', 'for rent', 'space available', 'available now',
        'now leasing', 'retail for lease', 'office for lease',
        'permanently closed', 'closed permanently', 'going out of business',
        'store closing', 'closing sale', 'everything must go', 'final sale',
        'vacant', 'rent me', 'lease me',
    ]
    weak_signals = [
        'coming soon', 'opening soon', 'under construction', 'under renovation',
        'moved to', 'we have moved', 'relocated', 'new location',
    ]

    for kw in strong_signals:
        if kw in norm:
            return kw, 'strong'
    for kw in weak_signals:
        if kw in norm:
            return kw, 'weak'
    return None, None


def lookup_google_places(name, lat, lng):
    """Layer 3: Google Places API — check business_status.
    Returns dict with status, detail, place info."""
    if not GOOGLE_PLACES_API_KEY:
        return {"status": "no_api_key", "detail": "Google Places API key not configured", "business_status": None}

    try:
        # Use Nearby Search with keyword
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{lat},{lng}",
            "radius": 100,
            "keyword": name,
            "key": GOOGLE_PLACES_API_KEY,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "OK" or not data.get("results"):
            return {"status": "not_found", "detail": f"No Google Places results for '{name}'", "business_status": None}

        # Find best name match
        best = None
        for place in data["results"]:
            pname = normalize(place.get("name", ""))
            nname = normalize(name)
            if nname in pname or pname in nname or (len(nname) >= 4 and nname[:4] in pname):
                best = place
                break

        if not best:
            best = data["results"][0]  # use closest result

        biz_status = best.get("business_status", "UNKNOWN")
        place_name = best.get("name", "?")

        return {
            "status": "found",
            "detail": f"Google Places: '{place_name}' is {biz_status}",
            "business_status": biz_status,
            "place_name": place_name,
            "place_id": best.get("place_id"),
        }
    except Exception as e:
        return {"status": "error", "detail": f"Google Places error: {e}", "business_status": None}


def fuzzy_text_match(detected_text, search_names):
    """Check if detected OCR text matches any of the business search names.
    Returns: ('full', matched_name) | ('partial', matched_name) | (None, None)
    """
    norm_text = normalize(detected_text)
    if not norm_text or len(norm_text) < 2:
        return None, None

    for search_name in search_names:
        if not search_name:
            continue

        # Level 1: Full name match (exact or substring)
        if search_name in norm_text or norm_text in search_name:
            return "full", search_name

        # Level 2: Word-level matching
        search_words = search_name.split()
        text_words = norm_text.split()

        for sw in search_words:
            if len(sw) < 3:
                continue
            for tw in text_words:
                if len(tw) < 3:
                    continue
                if sw == tw:
                    return "partial", search_name
                if len(sw) >= 4 and len(tw) >= 4:
                    if sw in tw or tw in sw:
                        return "partial", search_name
                if len(sw) >= 5 and len(tw) >= 5 and abs(len(sw) - len(tw)) <= 1:
                    diffs = sum(1 for a, b in zip(sw, tw) if a != b)
                    if diffs <= 1:
                        return "partial", search_name

        # Level 3: Garbled OCR detection (LCS + char overlap)
        # Only apply to multi-word search names — single words are too prone to false positives
        if len(search_words) >= 2:
            name_nospace = search_name.replace(" ", "")
            text_nospace = norm_text.replace(" ", "")

            if len(text_nospace) >= 4 and len(name_nospace) >= 4:
                overlap = _char_overlap(name_nospace, text_nospace)
                lcs_len = _longest_common_subsequence(name_nospace, text_nospace)
                lcs_ratio = lcs_len / len(name_nospace)

                if overlap >= 0.55 and lcs_ratio >= 0.50:
                    return "partial", search_name

                for sw in search_words:
                    if len(sw) < 4:
                        continue
                    for tw in text_words:
                        if len(tw) < 3:
                            continue
                        if len(sw) >= 6 and len(tw) >= 4:
                            lcs = _longest_common_subsequence(sw, tw)
                            if lcs >= len(sw) * 0.6 and lcs >= 3:
                                return "partial", search_name

    return None, None


def compute_engineered_features(raw):
    """Compute engineered features from base features."""
    has_web = raw.get('has_website', 0)
    has_phone = raw.get('has_phone', 0)
    has_brand = raw.get('has_brand', 0)
    missing = (1 - has_web) + (1 - has_phone) + (1 - has_brand)
    conf = raw.get('overture_confidence', 0)
    fp = raw.get('fields_populated', 0)
    age = raw.get('source_age_days', 0)
    return {
        'missing_signals': missing,
        'data_completeness': round(conf * fp / 9.0, 4),
        'low_confidence': 1 if conf < 0.7 else 0,
        'no_web_no_phone': 1 if (has_web == 0 and has_phone == 0) else 0,
        'confidence_x_fields': round(conf * fp, 4),
        'confidence_x_website': round(conf * has_web, 4),
        'old_source': 1 if age > 400 else 0,
        'high_quality': 1 if (conf >= 0.95 and fp >= 7) else 0,
        'sparse_record': 1 if fp <= 4 else 0,
    }


def predict_xgboost(location):
    """XGBoost metadata model prediction with Platt-calibrated probabilities.
    Returns dict with verdict, score (calibrated), detail, feature_contributions."""
    try:
        import xgboost as xgb
        import numpy as np
    except ImportError:
        return {
            "verdict": "inconclusive",
            "score": 0.5,
            "confidence": 0.5,
            "detail": "XGBoost not installed",
            "feature_contributions": {},
        }

    if not os.path.exists(MODEL_PATH):
        return {
            "verdict": "inconclusive",
            "score": 0.5,
            "confidence": 0.5,
            "detail": "No trained model found",
            "feature_contributions": {},
        }

    # Load metadata (threshold, Platt params, closure rates)
    meta_path = MODEL_PATH.replace('xgboost_model.json', 'xgb_meta.json')
    threshold = 0.69
    platt_a, platt_b = None, None
    closure_rates = {}
    global_closure_rate = 0.22
    try:
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
            threshold = meta.get('optimal_threshold', 0.69)
            platt_a = meta.get('platt_a')
            platt_b = meta.get('platt_b')
            closure_rates = meta.get('category_closure_rates', {})
            global_closure_rate = meta.get('global_closure_rate', 0.22)
    except:
        pass

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    raw = location.get('overture_raw', {})

    # Look up category closure rate
    category = location.get('category', 'Unknown')
    raw['category_closure_rate'] = closure_rates.get(category, global_closure_rate)

    # Yelp rating/review count — NaN if not available (XGBoost handles natively)
    raw['yelp_rating'] = None      # not available at prediction time
    raw['yelp_review_count'] = None

    eng = compute_engineered_features(raw)

    # Build feature vector — use np.nan for None values
    features = []
    for f in BASE_FEATURES:
        v = raw.get(f)
        features.append(float(v) if v is not None else np.nan)
    features += [float(eng.get(f, 0)) for f in ENGINEERED_FEATURES]

    prob = model.predict_proba([np.array(features)])[0]
    raw_open_prob = float(prob[1])

    # Apply Platt scaling if params available
    if platt_a is not None and platt_b is not None:
        # Platt: P(y=1|f) = 1 / (1 + exp(A*f + B))
        open_prob = 1.0 / (1.0 + np.exp(platt_a * raw_open_prob + platt_b))
        open_prob = float(open_prob)
    else:
        open_prob = raw_open_prob

    confidence = max(open_prob, 1 - open_prob)

    if open_prob >= threshold:
        verdict = "supports_open"
    elif open_prob < (1 - threshold):
        verdict = "supports_closed"
    else:
        verdict = "inconclusive"

    all_features = {f: round(v, 4) for f, v in zip(FEATURES, features)}

    return {
        "verdict": verdict,
        "score": round(open_prob, 4),
        "confidence": round(confidence, 4),
        "raw_score": round(raw_open_prob, 4),
        "threshold": threshold,
        "detail": f"XGBoost: {open_prob:.0%} open (calibrated from {raw_open_prob:.0%} raw)",
        "feature_contributions": all_features,
    }


def analyze_location(location):
    """Run vision on ALL images, produce 2-layer output (Text + XGBoost)."""
    name = location["name"]
    gallery = location.get("current_gallery", [])
    search_names = get_brand_search_names(name)

    if not gallery:
        # Even with no images, use Foursquare + website + metadata to make a call
        fsq = location.get("foursquare", {})
        fsq_status = fsq.get("status", "no_data")
        score = 0.50
        evidence = []
        if fsq_status == "verified":
            score += 0.30
            evidence.append("Foursquare: verified")
        elif fsq_status == "closed":
            score -= 0.35
            evidence.append("Foursquare: closed")
        elif fsq_status == "mismatch":
            score -= 0.25
            evidence.append(f"Foursquare: mismatch")

        # Website liveness check
        website_url = location.get("website_url")
        if website_url:
            ws_status, ws_code, ws_detail = check_website(website_url)
        else:
            ws_status, ws_code, ws_detail = 'no_url', None, 'No URL in Overture'
        website_layer = {"status": ws_status, "url": website_url, "status_code": ws_code, "detail": ws_detail}

        xgb_layer = predict_xgboost(location)

        if ws_status == 'redirect':
            score -= 0.20
            evidence.append(f"Website: redirected")
        elif ws_status == 'dead':
            score -= 0.10
            evidence.append(f"Website: dead")
        elif ws_status == 'parked':
            score -= 0.15
            evidence.append(f"Website: parked")
        elif ws_status == 'alive':
            score += 0.05
            evidence.append(f"Website: alive")

        evidence.append("No street-level images")
        score = max(0.0, min(1.0, score))
        pred = "open" if score > 0.50 else "not_open"
        if fsq_status == "no_data" and ws_status == "no_url":
            pred = "unknown"  # truly no info at all
        return {
            "prediction": pred,
            "confidence": round(score, 4),
            "primary_layer": "foursquare" if fsq_status != "no_data" else "website" if ws_status not in ("no_url",) else "xgboost",
            "evidence": " | ".join(evidence),
            "layers": {
                "text": {"verdict": "no_images", "score": 0.0, "best_image_url": None, "best_image_idx": None,
                         "matched_text": None, "match_strength": None, "ocr_texts": [], "ocr_texts_detail": [],
                         "detail": "No images to analyze"},
                "xgboost": xgb_layer,
                "website": website_layer,
                "foursquare": {"status": fsq_status, "detail": fsq.get("detail", ""), "match": fsq.get("match")},
            },
            "ocr_texts": [],
            "images_analyzed": 0,
            "best_image_idx": None,
            "best_image_url": None,
            "text_match": False,
            "per_image": [],
        }

    sorted_imgs = sorted(gallery, key=lambda x: x.get("distance_m", 999))

    # ── Phase 1: Analyze every image with OCR ──
    all_ocr_texts = []
    per_image_results = []
    images_analyzed = 0
    closure_signals = []  # Track closure-indicating text found across all images

    best_text_img = {"idx": 0, "url": None, "strength": None, "matched_text": None, "texts": [], "age_years": 0.0, "age_factor": 1.0}

    for img_idx, img_info in enumerate(sorted_imgs):
        url = img_info.get("url")

        # Calculate image age in years
        img_date_str = img_info.get("date", "")
        img_age_years = 0.0
        try:
            if img_date_str:
                img_date = datetime.strptime(img_date_str, "%Y-%m-%d")
                img_age_years = (datetime.now() - img_date).days / 365.25
        except (ValueError, TypeError):
            img_age_years = 10.0  # assume old if we can't parse

        # Pre-compute age factor (used in evidence scoring below)
        if img_age_years <= 2:
            age_factor = 1.0
        elif img_age_years <= 5:
            age_factor = max(0.4, 1.0 - (img_age_years - 2) * 0.2)
        else:
            age_factor = max(0.1, 0.4 - (img_age_years - 5) * 0.06)

        img_result = {
            "idx": img_idx,
            "url": url,
            "distance_m": img_info.get("distance_m", 999),
            "image_age_years": round(img_age_years, 1),
            "texts": [],
            "text_match": None,
            "text_match_strength": None,
            "text_matched_name": None,
            "evidence_score": 0.0,
        }

        if not url:
            per_image_results.append(img_result)
            continue

        tmp_path = download_image(url)
        if not tmp_path:
            per_image_results.append(img_result)
            continue

        try:
            # OCR
            texts = detect_text(tmp_path)
            images_analyzed += 1

            img_text_strings = [t["text"] for t in texts]
            img_result["texts"] = img_text_strings
            img_result["texts_with_bbox"] = [
                {"text": t["text"], "confidence": t["confidence"], "bbox_pct": t.get("bbox_pct")}
                for t in texts
            ]
            all_ocr_texts.extend(img_text_strings)

            # Text match
            best_match_strength = None
            best_matched_name = None
            for t in texts:
                strength, matched_name = fuzzy_text_match(t["text"], search_names)
                if strength == "full":
                    best_match_strength = "full"
                    best_matched_name = matched_name
                    break
                elif strength == "partial" and best_match_strength is None:
                    best_match_strength = "partial"
                    best_matched_name = matched_name

            img_result["text_match_strength"] = best_match_strength
            img_result["text_matched_name"] = best_matched_name
            if best_match_strength:
                img_result["text_match"] = True

            # Check for closure-indicating text (for lease, closed, etc.)
            for t in texts:
                closure_kw, closure_strength = detect_closure_text(t["text"])
                if closure_kw:
                    closure_signals.append({
                        "keyword": closure_kw,
                        "strength": closure_strength,
                        "text": t["text"],
                        "image_idx": img_idx,
                        "image_age_years": img_age_years,
                        "age_factor": age_factor,
                    })

            # Track best text image
            strength_rank = {"full": 2, "partial": 1}
            current_rank = strength_rank.get(best_match_strength, 0)
            best_rank = strength_rank.get(best_text_img["strength"], 0)
            if current_rank > best_rank:
                best_text_img = {"idx": img_idx, "url": url, "strength": best_match_strength,
                                 "matched_text": best_matched_name, "texts": img_text_strings,
                                 "age_years": img_age_years, "age_factor": age_factor}

            # Evidence score (text-only, no CLIP)
            evidence_score = 0.0
            if best_match_strength == "full":
                evidence_score += 0.7
            elif best_match_strength == "partial":
                evidence_score += 0.3
            dist = img_info.get("distance_m", 100)
            evidence_score += max(0, (60 - dist) / 600)

            # Image age penalty: old images are unreliable evidence
            evidence_score *= age_factor
            img_result["age_factor"] = round(age_factor, 2)
            img_result["evidence_score"] = round(evidence_score, 4)

            print(f"      img{img_idx}: OCR={len(texts)} texts, "
                  f"text={'FULL' if best_match_strength == 'full' else 'PART' if best_match_strength == 'partial' else 'none'}, "
                  f"score={evidence_score:.2f}, age={img_age_years:.0f}yr (x{age_factor:.1f})", flush=True)

        except Exception as e:
            print(f"      img{img_idx}: ERROR {e}", flush=True)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        per_image_results.append(img_result)

    # ── Phase 2: Aggregate ──
    best_img = max(per_image_results, key=lambda x: x["evidence_score"])
    best_img_idx = best_img["idx"]
    best_img_url = best_img["url"]

    any_text_full = any(r["text_match_strength"] == "full" for r in per_image_results)
    any_text_partial = any(r["text_match_strength"] == "partial" for r in per_image_results)
    text_found = any_text_full or any_text_partial
    text_strength = "full" if any_text_full else ("partial" if any_text_partial else None)

    # ── Phase 3: Per-layer sub-verdicts ──

    # Layer 1: Text (discounted by image age)
    best_age = best_text_img.get("age_years", 0.0)
    best_age_factor = best_text_img.get("age_factor", 1.0)
    age_warning = ""
    if best_age > 3:
        age_warning = f" (image is {best_age:.0f} years old — low reliability)"

    if text_strength == "full":
        text_verdict = "full_match"
        text_score = 0.85 * best_age_factor
        text_detail = f"Text '{name}' found in images{age_warning}"
    elif text_strength == "partial":
        text_verdict = "partial_match"
        text_score = 0.35 * best_age_factor
        text_detail = f"Partial text match for '{name}'{age_warning}"
    else:
        text_verdict = "no_match"
        text_score = 0.0
        text_detail = f"No '{name}' text detected"

    matched_text_display = best_text_img["matched_text"] or name if text_found else None
    text_layer = {
        "verdict": text_verdict,
        "score": round(text_score, 4),
        "best_image_url": best_text_img["url"],
        "best_image_idx": best_text_img["idx"] if best_text_img["url"] else None,
        "matched_text": matched_text_display,
        "match_strength": text_strength,
        "ocr_texts": list(dict.fromkeys(all_ocr_texts)),
        "ocr_texts_detail": [
            {
                "text": t["text"],
                "confidence": t["confidence"],
                "bbox_pct": t.get("bbox_pct"),
                "image_url": r["url"],
                "image_idx": r["idx"],
                "is_match": fuzzy_text_match(t["text"], search_names)[0] is not None,
                "match_strength": fuzzy_text_match(t["text"], search_names)[0],
            }
            for r in per_image_results if r["url"]
            for t in r.get("texts_with_bbox", [])
        ],
        "detail": text_detail,
        "image_age_years": round(best_age, 1),
        "image_age_factor": round(best_age_factor, 2),
    }

    # Layer 2: Google Places API (if configured)
    [lng, lat] = location.get("location", [0, 0])
    google_layer = lookup_google_places(name, lat, lng)

    # Layer 3: Website liveness check
    website_url = location.get("website_url")
    if website_url:
        ws_status, ws_code, ws_detail = check_website(website_url)
    else:
        ws_status, ws_code, ws_detail = 'no_url', None, 'No URL in Overture'
    website_layer = {
        "status": ws_status,
        "url": website_url,
        "status_code": ws_code,
        "detail": ws_detail,
    }

    # Layer 4: Foursquare (already fetched, just format)
    fsq_raw = location.get("foursquare", {})
    fsq_layer = {
        "status": fsq_raw.get("status", "no_data"),
        "detail": fsq_raw.get("detail", "No Foursquare data"),
        "match": fsq_raw.get("match"),
    }

    # Layer 5: XGBoost metadata model
    xgb_layer = predict_xgboost(location)

    # ── Phase 4: Encode signal scores for metamodel ──
    # Each signal is encoded as a single numeric value, then combined via
    # learned logistic regression weights (trained with LOCO-CV, 82.5% accuracy).
    import math

    # Metamodel weights (learned from logistic regression with 6 signals + interactions)
    META_WEIGHTS = {
        'foursquare': 1.8997,
        'website': 1.3448,
        'text': 0.1617,
        'xgboost': 0.6998,
        'tomtom': -0.0668,
        'yelp_reviews': 1.7241,
        'fsq_verified_web_dead': -0.9368,
        'fsq_no_data': -0.7736,
        'fsq_nodata_web_alive': -1.0269,
        'both_dirs_verified': 0.5090,
        'both_dirs_missing': -0.0729,
        'fsq_verified_no_yelp': -0.1488,
    }
    META_INTERCEPT = 0.6184

    evidence_parts = []
    primary_layer = "xgboost"
    score_breakdown = []

    age_note = f" ({best_age:.0f}yr old)" if best_age > 3 else ""
    xgb_score = xgb_layer["score"]

    # Closure text on recent images
    fresh_closure_signals = [s for s in closure_signals if s["image_age_years"] <= 5]
    strong_closure_text = any(s["strength"] == "strong" for s in fresh_closure_signals)

    # Foursquare status
    fsq = location.get("foursquare", {})
    fsq_status = fsq.get("status", "no_data")
    fsq_verified = fsq_status == "verified"
    fsq_mismatch = fsq_status == "mismatch"
    fsq_closed = fsq_status == "closed"

    # Google Places status
    google_closed = google_layer.get("business_status") in ("CLOSED_PERMANENTLY", "CLOSED_TEMPORARILY")
    google_open = google_layer.get("business_status") == "OPERATIONAL"

    # ── Phase 5: Encode 4 signal scores ──

    # Signal 1: Foursquare
    if fsq_verified:
        fsq_signal = 1.0
        primary_layer = "foursquare"
        evidence_parts.append("Foursquare: verified")
    elif fsq_closed:
        fsq_signal = -1.0
        primary_layer = "foursquare"
        evidence_parts.append("Foursquare: closed")
    elif fsq_mismatch:
        fsq_match = fsq.get("match", {})
        other_name = fsq_match.get("name", "?") if fsq_match else "?"
        fsq_signal = -0.5
        primary_layer = "foursquare"
        evidence_parts.append(f"Foursquare: mismatch (\"{other_name}\")")
    else:
        fsq_signal = -0.3
        evidence_parts.append("Foursquare: no data")

    score_breakdown.append({"signal": "foursquare", "weight": round(fsq_signal * META_WEIGHTS['foursquare'], 4), "description": f"Foursquare: {fsq_status} (signal={fsq_signal:+.1f})"})

    # Google (not in metamodel, but still log as evidence)
    if google_open:
        evidence_parts.append("Google: operational")
    elif google_closed:
        evidence_parts.append("Google: closed")

    # Signal 2: Website
    if ws_status == 'alive':
        ws_signal = 0.5
        evidence_parts.append("Website: alive")
    elif ws_status == 'redirect':
        ws_signal = -0.7
        evidence_parts.append(f"Website: redirected ({ws_detail})")
    elif ws_status == 'dead':
        ws_signal = -0.3
        evidence_parts.append(f"Website: dead ({ws_detail})")
    elif ws_status == 'parked':
        ws_signal = -0.5
        evidence_parts.append("Website: parked domain")
    else:
        ws_signal = 0.0
        if ws_status == 'ssl_error':
            evidence_parts.append("Website: SSL error")

    score_breakdown.append({"signal": "website", "weight": round(ws_signal * META_WEIGHTS['website'], 4), "description": f"Website: {ws_status} (signal={ws_signal:+.1f})"})

    # Signal 3: Text/OCR
    if strong_closure_text:
        best_closure = fresh_closure_signals[0]
        text_signal = -1.0
        primary_layer = "text"
        evidence_parts.append(f"Closure text: \"{best_closure['text']}\"")
    elif text_found:
        if text_strength == "full":
            text_signal = 1.0
        else:
            text_signal = 0.5
        if primary_layer == "xgboost":
            primary_layer = "text"
        evidence_parts.append(f"Text: {text_strength} match{age_note}")
    elif best_age <= 3 and images_analyzed >= 2:
        text_signal = -0.2
        evidence_parts.append(f"No text match ({images_analyzed} recent images)")
    elif images_analyzed == 0:
        text_signal = 0.0
        evidence_parts.append("No images available")
    else:
        text_signal = 0.0

    score_breakdown.append({"signal": "text", "weight": round(text_signal * META_WEIGHTS['text'], 4), "description": f"Text: signal={text_signal:+.1f}"})

    # Signal 4: XGBoost metadata (centered around 0)
    xgb_centered = xgb_score - 0.5
    evidence_parts.append(f"Metadata: {xgb_score:.0%} open")

    score_breakdown.append({"signal": "xgboost", "weight": round(xgb_centered * META_WEIGHTS['xgboost'], 4), "description": f"Metadata: {xgb_score:.0%} open (signal={xgb_centered:+.2f})"})

    # Signal 5: TomTom directory
    tt = location.get("tomtom", {})
    tt_status = tt.get("status", "no_data")
    tt_match = tt.get("match_score", 0.0)
    if tt_status == 'verified' and tt_match >= 0.8:
        tt_signal = 1.0
        evidence_parts.append(f"TomTom: verified (match={tt_match:.1f})")
    elif tt_status == 'verified' and tt_match >= 0.5:
        tt_signal = 0.3
        evidence_parts.append(f"TomTom: weak match ({tt_match:.1f})")
    else:
        tt_signal = 0.0
        evidence_parts.append("TomTom: no data")

    score_breakdown.append({"signal": "tomtom", "weight": round(tt_signal * META_WEIGHTS['tomtom'], 4), "description": f"TomTom: {tt_status} match={tt_match:.1f} (signal={tt_signal:+.1f})"})

    # Signal 6: Yelp review count (log-scaled)
    yelp_data = location.get("yelp", {})
    yelp_reviews = yelp_data.get("yelp_review_count", 0)
    if yelp_reviews > 0:
        yelp_signal = min(1.0, (math.log10(yelp_reviews) - 1.0) / 2.0)
        evidence_parts.append(f"Yelp: {yelp_reviews} reviews")
    else:
        yelp_signal = -1.0
        evidence_parts.append("Yelp: no reviews")

    score_breakdown.append({"signal": "yelp", "weight": round(yelp_signal * META_WEIGHTS['yelp_reviews'], 4), "description": f"Yelp: {yelp_reviews} reviews (signal={yelp_signal:+.2f})"})

    # Interaction features
    fsq_verified_web_dead = 1.0 if (fsq_status == 'verified' and ws_status == 'dead') else 0.0
    fsq_no_data_flag = 1.0 if fsq_status == 'no_data' else 0.0
    fsq_nodata_web_alive = fsq_no_data_flag * (1.0 if ws_status == 'alive' else 0.0)
    tt_strong = tt_status == 'verified' and tt_match >= 0.8
    both_dirs_verified = 1.0 if (fsq_status == 'verified' and tt_strong) else 0.0
    both_dirs_missing = 1.0 if (fsq_status == 'no_data' and not tt_strong) else 0.0
    fsq_verified_no_yelp = 1.0 if (fsq_status == 'verified' and yelp_reviews == 0) else 0.0

    # ── Metamodel: sigmoid(w·x + b) ──
    logit = (
        META_WEIGHTS['foursquare'] * fsq_signal +
        META_WEIGHTS['website'] * ws_signal +
        META_WEIGHTS['text'] * text_signal +
        META_WEIGHTS['xgboost'] * xgb_centered +
        META_WEIGHTS['tomtom'] * tt_signal +
        META_WEIGHTS['yelp_reviews'] * yelp_signal +
        META_WEIGHTS['fsq_verified_web_dead'] * fsq_verified_web_dead +
        META_WEIGHTS['fsq_no_data'] * fsq_no_data_flag +
        META_WEIGHTS['fsq_nodata_web_alive'] * fsq_nodata_web_alive +
        META_WEIGHTS['both_dirs_verified'] * both_dirs_verified +
        META_WEIGHTS['both_dirs_missing'] * both_dirs_missing +
        META_WEIGHTS['fsq_verified_no_yelp'] * fsq_verified_no_yelp +
        META_INTERCEPT
    )
    open_score = 1.0 / (1.0 + math.exp(-logit))

    score_breakdown.insert(0, {"signal": "metamodel", "weight": round(open_score, 4), "description": f"Metamodel score (logit={logit:+.2f})"})

    # Binary prediction
    if images_analyzed == 0 and fsq_status == "no_data" and not google_open and not google_closed:
        prediction = "unknown"
    elif open_score > 0.50:
        prediction = "open"
    else:
        prediction = "not_open"

    confidence = round(open_score, 4)
    evidence_parts.append(f"Score: {open_score:.2f}")
    if closure_signals and not strong_closure_text:
        evidence_parts.append(f"Closure hint: \"{closure_signals[0]['keyword']}\"")

    unique_texts = list(dict.fromkeys(all_ocr_texts))
    evidence_str = " | ".join(evidence_parts)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "score_breakdown": score_breakdown,
        "primary_layer": primary_layer,
        "evidence": evidence_str,
        "detail": evidence_str,
        "layers": {
            "text": text_layer,
            "xgboost": xgb_layer,
            "google": google_layer,
            "website": website_layer,
            "foursquare": fsq_layer,
        },
        "closure_signals": closure_signals,
        "ocr_texts": unique_texts,
        "images_analyzed": images_analyzed,
        "best_image_idx": best_img_idx,
        "best_image_url": best_img_url,
        "text_match": text_found,
        "per_image": [
            {
                "url": r["url"],
                "idx": r["idx"],
                "texts_found": len(r["texts"]),
                "texts_with_bbox": r.get("texts_with_bbox", []),
                "text_match": r["text_match_strength"],
                "evidence_score": r["evidence_score"],
            }
            for r in per_image_results if r["url"]
        ],
    }


def main():
    print("=" * 70, flush=True)
    print("TERRAFORMA Vision Analysis — 2-Layer Prediction", flush=True)
    print("Layer 1: Text (OCR) | Layer 2: Metadata (XGBoost)", flush=True)
    print("=" * 70, flush=True)

    print(f"\nLoading data from {DATA_PATH}...", flush=True)
    with open(DATA_PATH, encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} locations\n", flush=True)

    # Pre-load OCR model
    print("Pre-loading models...", flush=True)
    get_ocr()
    print("Models ready!\n", flush=True)

    counts = {"open": 0, "not_open": 0, "unknown": 0}
    primary_counts = {"text": 0, "xgboost": 0, "google": 0, "foursquare": 0}
    start_time = time.time()

    for i, location in enumerate(data):
        name = location["name"]
        n_images = len(location.get("current_gallery", []))
        print(f"\n[{i+1}/{len(data)}] {name} ({n_images} imgs)", flush=True)

        result = analyze_location(location)
        location["vision"] = result

        tag = {"open": "OPEN", "not_open": "CLOSED", "uncertain": "UNSURE", "unknown": "??"}
        icon = tag.get(result["prediction"], "??")
        flags = []
        fsq_s = result.get("layers", {}).get("foursquare", {}).get("status", "")
        if fsq_s and fsq_s != "no_data":
            flags.append(f"FSQ:{fsq_s[:4].upper()}")
        if result["text_match"]:
            flags.append("TEXT")
        if result.get("closure_signals"):
            flags.append("CLOSURE")
        google_status = result.get("layers", {}).get("google", {}).get("business_status")
        if google_status:
            flags.append(f"G:{google_status[:4]}")
        ws_s = result.get("layers", {}).get("website", {}).get("status", "")
        if ws_s and ws_s not in ("no_url",):
            flags.append(f"WEB:{ws_s[:5].upper()}")
        flags.append(f"XGB:{result['layers']['xgboost']['score']:.0%}")
        flag_str = f" [{'+'.join(flags)}]" if flags else ""
        gt = location.get("ground_truth", "?")
        print(f"    => [{icon}] gt={gt}{flag_str} primary={result['primary_layer']} | {result['evidence']}", flush=True)

        counts[result["prediction"]] = counts.get(result["prediction"], 0) + 1
        primary_counts[result["primary_layer"]] = primary_counts.get(result["primary_layer"], 0) + 1

        # Save every 10 locations
        if (i + 1) % 10 == 0:
            with open(DATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            elapsed = time.time() - start_time
            print(f"    [Saved checkpoint at {i+1}/{len(data)}, {elapsed:.0f}s elapsed]", flush=True)

    elapsed = time.time() - start_time

    # Final save
    with open(DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}", flush=True)
    print(f"DONE in {elapsed:.0f}s ({elapsed/len(data):.1f}s per location)", flush=True)
    print(f"\nPredictions:", flush=True)
    for k, v in counts.items():
        print(f"  {k:12s}: {v}", flush=True)
    print(f"\nPrimary layer:", flush=True)
    for k, v in primary_counts.items():
        print(f"  {k:12s}: {v}", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"Results saved to {DATA_PATH}", flush=True)


if __name__ == "__main__":
    main()
