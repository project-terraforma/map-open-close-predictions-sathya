"""
Batch vision analysis: run OCR + CLIP on Mapillary images for each location.
Downloads images, runs the model, and writes predictions back to mock_data.json.

Three-layer prediction system:
  Layer 1 (Logo):  CLIP brand detection — primary visual signal
  Layer 2 (Text):  OCR text matching — confirms/boosts logo
  Layer 3 (Data):  Foursquare + Overture cross-referencing — external verification

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
from ocr_model import detect_text, detect_brands, get_ocr, get_clip, normalize, get_brand_search_names

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'mock_data.json')

# Layer weights
LOGO_WEIGHT = 0.40
TEXT_WEIGHT = 0.25
DATA_WEIGHT = 0.35


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


def fuzzy_text_match(detected_text, search_names):
    """Check if detected OCR text matches any of the brand search names.
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


def clip_matches_business(brands, search_names):
    """Check if CLIP detected the correct brand.
    Returns (matched, score, matched_brand_name)."""
    for b in brands[:5]:
        norm_brand = normalize(b["name"])
        for search_name in search_names:
            if not search_name:
                continue
            if norm_brand == search_name or search_name in norm_brand or norm_brand in search_name:
                return True, b["score"], b["name"]
    return False, 0.0, None


# ── Layer 3: Data verification ──

def _names_match(a, b):
    """Check if two business names refer to the same chain."""
    na = normalize(a)
    nb = normalize(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    # Check first significant word
    wa = [w for w in na.split() if len(w) >= 3]
    wb = [w for w in nb.split() if len(w) >= 3]
    if wa and wb and wa[0] == wb[0]:
        return True
    return False


def _address_similarity(addr1, addr2):
    """Word overlap similarity between two addresses. Returns 0-1."""
    w1 = set(normalize(addr1).split())
    w2 = set(normalize(addr2).split())
    # Remove very common words
    stop = {'st', 'ave', 'rd', 'dr', 'blvd', 'ln', 'ct', 'pl', 'way', 'san', 'francisco', 'ca', 'usa'}
    w1 = w1 - stop
    w2 = w2 - stop
    if not w1 or not w2:
        return 0.0
    overlap = len(w1 & w2)
    return overlap / max(len(w1), len(w2))


def analyze_data_layer(location):
    """Layer 3: Data cross-referencing using Foursquare + Overture metadata.
    Returns a sub-verdict dict."""
    fs = location.get("foursquare", {})
    overture = location.get("overture_meta", {})
    name = location["name"]
    address = location.get("address", "")

    score = 0.0
    signals = []

    # Signal 1: Foursquare verification status (weight 0.35)
    fs_status = fs.get("status", "no_data")
    fs_match = fs.get("match") or {}

    if fs_status == "verified" and not fs_match.get("closed", False):
        score += 0.35
        signals.append({"signal": "Foursquare status", "value": "Verified open", "contribution": 0.35})
    elif fs_status == "closed" or fs_match.get("closed", True) is True:
        if fs_status == "verified" and fs_match.get("closed", False):
            score -= 0.40
            signals.append({"signal": "Foursquare status", "value": "Confirmed closed", "contribution": -0.40})
        elif fs_status == "closed":
            score -= 0.40
            signals.append({"signal": "Foursquare status", "value": "Closed", "contribution": -0.40})
        else:
            signals.append({"signal": "Foursquare status", "value": "No data", "contribution": 0.0})
    elif fs_status == "mismatch":
        score -= 0.20
        signals.append({"signal": "Foursquare status", "value": "Name mismatch", "contribution": -0.20})
    else:
        signals.append({"signal": "Foursquare status", "value": "No data", "contribution": 0.0})

    # Signal 2: Overture confidence (weight 0.25)
    overture_conf = overture.get("confidence")
    if overture_conf is not None:
        contrib = (overture_conf - 0.5) * 0.5  # maps 0.5-1.0 → 0.0-0.25
        contrib = max(-0.15, min(0.25, contrib))
        score += contrib
        signals.append({"signal": "Overture confidence", "value": f"{overture_conf:.0%}", "contribution": round(contrib, 4)})
    else:
        signals.append({"signal": "Overture confidence", "value": "N/A", "contribution": 0.0})

    # Signal 3: Brand name consistency (weight 0.15)
    overture_brand = overture.get("brand")
    fs_name = fs_match.get("name", "")
    brand_consistent = False
    brand_contrib = 0.0

    if overture_brand and fs_name:
        if _names_match(name, overture_brand) and _names_match(name, fs_name):
            brand_consistent = True
            brand_contrib = 0.15
        elif _names_match(name, overture_brand) or _names_match(name, fs_name):
            brand_consistent = True
            brand_contrib = 0.08
    elif overture_brand and _names_match(name, overture_brand):
        brand_consistent = True
        brand_contrib = 0.10
    elif fs_name and _names_match(name, fs_name):
        brand_consistent = True
        brand_contrib = 0.10

    score += brand_contrib
    signals.append({"signal": "Brand consistency", "value": "Match" if brand_consistent else "No match", "contribution": round(brand_contrib, 4)})

    # Signal 4: Address consistency (weight 0.15)
    fs_address = fs_match.get("address", "")
    addr_score = 0.0
    if fs_address and address:
        addr_score = _address_similarity(address, fs_address)
        addr_contrib = addr_score * 0.15
        score += addr_contrib
        signals.append({"signal": "Address match", "value": f"{addr_score:.0%}", "contribution": round(addr_contrib, 4)})
    else:
        signals.append({"signal": "Address match", "value": "N/A", "contribution": 0.0})

    # Signal 5: Data recency (weight 0.10)
    update_time = overture.get("update_time")
    recency_contrib = 0.0
    if update_time:
        try:
            update_date = datetime.strptime(update_time, "%Y-%m-%d")
            age_days = (datetime.now() - update_date).days
            if age_days < 365:
                recency_contrib = 0.10
            elif age_days < 730:
                recency_contrib = 0.05
            elif age_days < 1825:  # 5 years
                recency_contrib = 0.0
            else:
                recency_contrib = -0.05
            score += recency_contrib
        except ValueError:
            pass
    signals.append({"signal": "Data recency", "value": update_time or "N/A", "contribution": round(recency_contrib, 4)})

    # Clamp
    score = max(-1.0, min(1.0, score))

    # Sub-verdict
    if score >= 0.30:
        verdict = "supports_open"
    elif score <= -0.15:
        verdict = "supports_closed"
    else:
        verdict = "inconclusive"

    # Human-readable detail
    if verdict == "supports_open":
        detail_parts = []
        if fs_status == "verified":
            detail_parts.append("Foursquare verified open")
        if overture_conf and overture_conf > 0.8:
            detail_parts.append(f"Overture confidence {overture_conf:.0%}")
        if brand_consistent:
            detail_parts.append("Brand name matches")
        detail = " | ".join(detail_parts) if detail_parts else f"Data score: {score:.2f}"
    elif verdict == "supports_closed":
        if fs_match.get("closed"):
            detail = "Foursquare confirms business is closed"
        elif fs_status == "mismatch":
            detail = "Foursquare found a different business at this location"
        else:
            detail = f"Data signals suggest closed (score: {score:.2f})"
    else:
        detail = f"Insufficient data to confirm status (score: {score:.2f})"

    return {
        "verdict": verdict,
        "score": round(score, 4),
        "signals": signals,
        "fsq_status": fs_status,
        "fsq_closed": fs_match.get("closed"),
        "overture_confidence": overture_conf,
        "brand_consistent": brand_consistent,
        "address_match_score": round(addr_score, 2) if fs_address and address else None,
        "detail": detail,
    }


def analyze_location(location):
    """Run vision on ALL images, track per-image evidence, produce 3-layer output."""
    name = location["name"]
    gallery = location.get("current_gallery", [])
    search_names = get_brand_search_names(name)

    # Layer 3 always runs (no images needed)
    data_layer = analyze_data_layer(location)

    if not gallery:
        return {
            "prediction": "unknown",
            "confidence": 0.0,
            "primary_layer": "data",
            "evidence": "No street-level images available",
            "layers": {
                "logo": {"verdict": "no_images", "score": 0.0, "best_image_url": None, "best_image_idx": None, "clip_brand": None, "clip_score": 0.0, "detail": "No images to analyze"},
                "text": {"verdict": "no_images", "score": 0.0, "best_image_url": None, "best_image_idx": None, "matched_text": None, "match_strength": None, "ocr_texts": [], "detail": "No images to analyze"},
                "data": data_layer,
            },
            "ocr_texts": [],
            "clip_brand": None,
            "clip_score": 0.0,
            "images_analyzed": 0,
            "best_image_idx": None,
            "best_image_url": None,
            "text_match": False,
            "logo_match": False,
            "logo_match_score": None,
            "per_image": [],
        }

    sorted_imgs = sorted(gallery, key=lambda x: x.get("distance_m", 999))

    # ── Phase 1: Analyze every image ──
    all_ocr_texts = []
    per_image_results = []
    images_analyzed = 0

    # Track best images per layer
    best_logo_img = {"idx": 0, "url": None, "score": 0.0, "brand": None, "crop_region": None}
    best_text_img = {"idx": 0, "url": None, "strength": None, "matched_text": None, "texts": []}

    for img_idx, img_info in enumerate(sorted_imgs):
        url = img_info.get("url")
        img_result = {
            "idx": img_idx,
            "url": url,
            "distance_m": img_info.get("distance_m", 999),
            "texts": [],
            "text_match": None,
            "text_match_strength": None,
            "text_matched_name": None,
            "brands": [],
            "clip_match": False,
            "clip_match_score": 0.0,
            "clip_match_brand": None,
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

            # Track best text image
            strength_rank = {"full": 2, "partial": 1}
            current_rank = strength_rank.get(best_match_strength, 0)
            best_rank = strength_rank.get(best_text_img["strength"], 0)
            if current_rank > best_rank:
                best_text_img = {"idx": img_idx, "url": url, "strength": best_match_strength, "matched_text": best_matched_name, "texts": img_text_strings}

            # CLIP
            brands = detect_brands(tmp_path)
            img_result["brands"] = [
                {"name": b["name"], "score": b["score"], "best_crop_region": b.get("best_crop_region")}
                for b in brands[:3]
            ]

            matched, score, matched_brand = clip_matches_business(brands, search_names)
            img_result["clip_match"] = matched
            img_result["clip_match_score"] = score
            img_result["clip_match_brand"] = matched_brand

            # Track best logo image
            if matched and score > best_logo_img["score"]:
                matching_brand_obj = next((b for b in brands if b["name"] == matched_brand), None)
                best_logo_img = {
                    "idx": img_idx, "url": url, "score": score, "brand": matched_brand,
                    "crop_region": matching_brand_obj.get("best_crop_region") if matching_brand_obj else None,
                }

            # Composite evidence score
            evidence_score = 0.0
            if matched:
                evidence_score += score + 0.4
            if best_match_strength == "full" and matched:
                evidence_score += 0.3
            elif best_match_strength == "full":
                evidence_score += 0.5
            elif best_match_strength == "partial" and matched:
                evidence_score += 0.15
            elif best_match_strength == "partial":
                evidence_score += 0.1
            dist = img_info.get("distance_m", 100)
            evidence_score += max(0, (60 - dist) / 600)
            img_result["evidence_score"] = round(evidence_score, 4)

            print(f"      img{img_idx}: OCR={len(texts)} texts, "
                  f"text={'FULL' if best_match_strength == 'full' else 'PART' if best_match_strength == 'partial' else 'none'}, "
                  f"CLIP={matched_brand or (brands[0]['name'] if brands else '?')}({score:.0%}), "
                  f"score={evidence_score:.2f}", flush=True)

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

    best_clip_score = max((r["clip_match_score"] for r in per_image_results), default=0.0)
    logo_found = best_clip_score > 0.0
    logo_strong = best_clip_score >= 0.25

    top_clip_brand = None
    top_clip_score = 0.0
    for r in per_image_results:
        for b in r.get("brands", []):
            if b["score"] > top_clip_score:
                top_clip_brand = b["name"]
                top_clip_score = b["score"]

    # ── Phase 3: Per-layer sub-verdicts ──

    # Layer 1: Logo
    if logo_found and logo_strong:
        logo_verdict = "detected"
        logo_score = min(best_clip_score * 2.5, 1.0)  # normalize to 0-1
        logo_detail = f"Logo match: {best_logo_img['brand']} at {best_clip_score:.1%}"
    elif logo_found:
        logo_verdict = "weak"
        logo_score = best_clip_score * 2.0
        logo_detail = f"Weak logo match: {best_logo_img['brand']} at {best_clip_score:.1%}"
    else:
        logo_verdict = "not_detected"
        logo_score = 0.0
        logo_detail = f"No logo detected (top: {top_clip_brand} at {top_clip_score:.1%})" if top_clip_brand else "No logo detected"

    logo_layer = {
        "verdict": logo_verdict,
        "score": round(logo_score, 4),
        "best_image_url": best_logo_img["url"],
        "best_image_idx": best_logo_img["idx"] if best_logo_img["url"] else None,
        "clip_brand": best_logo_img["brand"] or top_clip_brand,
        "clip_score": round(best_clip_score, 4),
        "best_crop_region": best_logo_img.get("crop_region"),
        "detail": logo_detail,
    }

    # Layer 2: Text
    if text_strength == "full":
        text_verdict = "full_match"
        text_score = 0.85
        text_detail = f"Text '{name}' clearly found in images"
    elif text_strength == "partial":
        text_verdict = "partial_match"
        text_score = 0.35
        text_detail = f"Partial text match for '{name}'"
    else:
        text_verdict = "no_match"
        text_score = 0.0
        text_detail = f"No '{name}' text detected"

    # Collect all matched texts for display
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
            }
            for r in per_image_results if r["url"]
            for t in r.get("texts_with_bbox", [])
        ],
        "detail": text_detail,
    }

    # Layer 3: Data (already computed)
    data_score_normalized = max(0, (data_layer["score"] + 0.4) / 0.9)  # normalize -0.4..0.5 → 0..1
    data_score_normalized = min(1.0, data_score_normalized)

    # ── Phase 4: Determine primary_layer (semantic, not weighted) ──
    # Primary layer = which layer provides the most interesting/distinguishing evidence
    # Visual evidence (logo/text) is always more interesting to show than data
    if logo_found:
        primary_layer = "logo"
    elif text_found:
        primary_layer = "text"
    else:
        primary_layer = "data"

    # ── Phase 5: Make prediction using all 3 layers ──
    evidence_parts = []

    # Override 1: Foursquare says closed → override to not_open
    if data_layer["verdict"] == "supports_closed" and data_layer.get("fsq_closed") is True:
        prediction = "not_open"
        confidence = 0.0
        primary_layer = "data"  # data overrides visual here
        evidence_parts.append("Foursquare confirms business is closed")
        if logo_found:
            evidence_parts.append(f"(logo still visible: {best_clip_score:.1%})")
    # Logo + full text = strongest
    elif logo_found and text_found and text_strength == "full":
        prediction = "open"
        confidence = 0.95
        evidence_parts.append(f"Logo match: {best_clip_score:.1%}")
        evidence_parts.append(f"Text '{name}' confirmed")
        if data_layer["verdict"] == "supports_open":
            evidence_parts.append("Data also confirms")
    # Strong logo alone
    elif logo_found and logo_strong:
        prediction = "open"
        confidence = max(best_clip_score, 0.70)
        evidence_parts.append(f"Logo match: {best_clip_score:.1%}")
        if data_layer["verdict"] == "supports_open":
            confidence = min(confidence + 0.1, 0.95)
            evidence_parts.append("Data confirms")
    # Logo + partial text
    elif logo_found and text_found:
        prediction = "open"
        confidence = max(best_clip_score, 0.65)
        evidence_parts.append(f"Logo match: {best_clip_score:.1%}")
        evidence_parts.append(f"Partial text match")
        if data_layer["verdict"] == "supports_open":
            confidence = min(confidence + 0.1, 0.90)
            evidence_parts.append("Data confirms")
    # Weak logo + data supports open
    elif logo_found and data_layer["verdict"] == "supports_open":
        prediction = "open"
        confidence = max(best_clip_score, 0.60)
        evidence_parts.append(f"Weak logo match: {best_clip_score:.1%}")
        evidence_parts.append("Data verification confirms")
    # Weak logo alone
    elif logo_found:
        prediction = "uncertain"
        confidence = best_clip_score
        evidence_parts.append(f"Weak logo match: {best_clip_score:.1%} — needs confirmation")
    # Full text + data supports open
    elif text_found and text_strength == "full" and data_layer["verdict"] == "supports_open":
        prediction = "open"
        confidence = 0.75
        evidence_parts.append(f"Text '{name}' found in images")
        evidence_parts.append("Data verification confirms")
    # Full text alone
    elif text_found and text_strength == "full":
        prediction = "open"
        confidence = 0.70
        evidence_parts.append(f"Text '{name}' found in images (no logo)")
    # Partial text + data supports open → open with lower confidence
    elif text_found and text_strength == "partial" and data_layer["verdict"] == "supports_open":
        prediction = "open"
        confidence = 0.55
        evidence_parts.append(f"Partial text match for '{name}'")
        evidence_parts.append("Data verification confirms")
    # Data strongly supports open (no visual evidence at all)
    elif data_layer["verdict"] == "supports_open" and data_layer["score"] >= 0.40:
        prediction = "open"
        confidence = 0.50
        evidence_parts.append("Data strongly supports open (no visual confirmation)")
        primary_layer = "data"
    # Data supports open (weaker)
    elif data_layer["verdict"] == "supports_open":
        prediction = "uncertain"
        confidence = 0.40
        evidence_parts.append("Data suggests open but no visual confirmation")
        primary_layer = "data"
    # Partial text alone
    elif text_found and text_strength == "partial":
        prediction = "not_open"
        confidence = 0.0
        evidence_parts.append(f"Only partial text match for '{name}' — no logo to confirm")
    # Nothing
    else:
        prediction = "not_open"
        confidence = 0.0
        if top_clip_brand:
            evidence_parts.append(f"No '{name}' text or logo found (top CLIP: {top_clip_brand} at {top_clip_score:.1%})")
        else:
            evidence_parts.append(f"No '{name}' text or logo detected in any image")

    unique_texts = list(dict.fromkeys(all_ocr_texts))
    evidence_str = " | ".join(evidence_parts)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "primary_layer": primary_layer,
        "evidence": evidence_str,
        "detail": evidence_str,
        "layers": {
            "logo": logo_layer,
            "text": text_layer,
            "data": data_layer,
        },
        # Backward-compatible flat fields
        "ocr_texts": unique_texts,
        "clip_brand": top_clip_brand,
        "clip_score": round(top_clip_score, 4),
        "images_analyzed": images_analyzed,
        "best_image_idx": best_img_idx,
        "best_image_url": best_img_url,
        "text_match": text_found,
        "logo_match": logo_found,
        "logo_match_score": round(best_clip_score, 4) if logo_found else None,
        "per_image": [
            {
                "url": r["url"],
                "idx": r["idx"],
                "texts_found": len(r["texts"]),
                "texts_with_bbox": r.get("texts_with_bbox", []),
                "text_match": r["text_match_strength"],
                "clip_brand": r["clip_match_brand"] or (r["brands"][0]["name"] if r["brands"] else None),
                "clip_score": r["clip_match_score"] if r["clip_match"] else (r["brands"][0]["score"] if r["brands"] else 0),
                "clip_crop_region": r["brands"][0].get("best_crop_region") if r.get("brands") else None,
                "evidence_score": r["evidence_score"],
            }
            for r in per_image_results if r["url"]
        ],
    }


def main():
    print("=" * 70, flush=True)
    print("TERRAFORMA Vision Analysis — 3-Layer Prediction", flush=True)
    print("Layer 1: Logo (CLIP) | Layer 2: Text (OCR) | Layer 3: Data (FSQ+Overture)", flush=True)
    print("=" * 70, flush=True)

    print(f"\nLoading data from {DATA_PATH}...", flush=True)
    with open(DATA_PATH, encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} locations\n", flush=True)

    # Pre-load models
    print("Pre-loading models...", flush=True)
    get_ocr()
    get_clip()
    print("Models ready!\n", flush=True)

    counts = {"open": 0, "not_open": 0, "uncertain": 0, "unknown": 0}
    primary_counts = {"logo": 0, "text": 0, "data": 0}
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
        if result["text_match"]:
            flags.append("TEXT")
        if result["logo_match"]:
            score_str = f"{result['logo_match_score']:.0%}" if result["logo_match_score"] else "?"
            flags.append(f"LOGO({score_str})")
        flags.append(f"DATA:{result['layers']['data']['verdict'][:4].upper()}")
        flag_str = f" [{'+'.join(flags)}]" if flags else ""
        print(f"    => [{icon}]{flag_str} primary={result['primary_layer']} | {result['evidence']}", flush=True)

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
