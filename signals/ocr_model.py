"""
Vision-based business detection using RapidOCR text reading.
Identifies text visible in street-level images.

Improvements:
  - Multi-pass OCR: original, enhanced contrast, CLAHE, grayscale
  - Lower OCR thresholds to catch faint/distant signs
"""

import re
import os
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

# ── Singletons ──
_ocr_engine = None


def get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        print("Loading RapidOCR engine...", flush=True)
        _ocr_engine = RapidOCR()
        print("RapidOCR ready.", flush=True)
    return _ocr_engine


def _preprocess_variants(image_path):
    """Generate multiple preprocessed versions of the image for OCR.
    Returns (list of (path, should_cleanup) tuples, scale_factor)."""
    img = cv2.imread(image_path)
    if img is None:
        return [(image_path, False)], 1.0

    h, w = img.shape[:2]
    variants = []
    base = os.path.splitext(image_path)[0]
    scale = 1.0

    # Always upscale small images
    if max(h, w) < 1200:
        scale = 1200 / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Variant 1: Sharpened original
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(img, -1, kernel)
    p1 = base + "_v1_sharp.jpg"
    cv2.imwrite(p1, sharp)
    variants.append((p1, True))

    # Variant 2: CLAHE (adaptive histogram equalization) - great for signs in shadow
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    p2 = base + "_v2_clahe.jpg"
    cv2.imwrite(p2, enhanced)
    variants.append((p2, True))

    # Variant 3: High contrast grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.filter2D(gray, -1, kernel)
    p3 = base + "_v3_gray.jpg"
    cv2.imwrite(p3, gray)
    variants.append((p3, True))

    return variants, scale


def normalize(text):
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[''`\u2019\u2018]", "", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_text(image_path):
    """Run OCR on multiple preprocessed variants and merge results.
    Returns all unique detected text fragments with confidence and bbox_pct."""
    engine = get_ocr()
    variants, scale = _preprocess_variants(image_path)

    # Get original image dimensions for bbox normalization
    orig_img = cv2.imread(image_path)
    if orig_img is not None:
        orig_h, orig_w = orig_img.shape[:2]
    else:
        orig_h, orig_w = 1, 1  # fallback

    seen_texts = {}  # normalized text -> best result

    for var_path, should_cleanup in variants:
        try:
            result, _ = engine(
                var_path,
                det_db_thresh=0.15,      # lower = catch fainter text
                det_db_box_thresh=0.2,   # lower = catch smaller boxes
            )
            if result:
                for item in result:
                    bbox, text, conf = item
                    norm = normalize(text)
                    if len(norm) < 2:
                        continue  # skip single-char noise
                    # Keep the highest-confidence version of each text
                    if norm not in seen_texts or conf > seen_texts[norm]["confidence"]:
                        # Convert 4-corner bbox to percentage-based rect
                        pts = np.array(bbox)
                        x_min, y_min = pts.min(axis=0)
                        x_max, y_max = pts.max(axis=0)
                        bbox_pct = [
                            round(float(x_min / (orig_w * scale)), 4),
                            round(float(y_min / (orig_h * scale)), 4),
                            round(float((x_max - x_min) / (orig_w * scale)), 4),
                            round(float((y_max - y_min) / (orig_h * scale)), 4),
                        ]
                        seen_texts[norm] = {
                            "text": text,
                            "confidence": round(float(conf), 3),
                            "bbox": bbox,
                            "bbox_pct": bbox_pct,
                        }
        except Exception as e:
            pass  # skip failed variants
        finally:
            if should_cleanup and os.path.exists(var_path):
                os.unlink(var_path)

    return list(seen_texts.values())


def get_brand_search_names(business_name):
    """Get the list of search names for a business.
    Returns the full normalized name plus significant sub-phrases and words
    so partial sign text (e.g. just 'Al Masri' from 'Al-Masri Egyptian Restaurant')
    can still match."""
    norm = normalize(business_name)
    if not norm:
        return []

    names = [norm]
    words = norm.split()

    # Common suffixes/generic words to strip for sub-phrase generation
    generic = {
        'restaurant', 'cafe', 'coffee', 'bar', 'grill', 'kitchen', 'bakery',
        'salon', 'spa', 'studio', 'shop', 'store', 'market', 'deli',
        'pizza', 'pizzeria', 'taqueria', 'bistro', 'lounge', 'pub',
        'inc', 'llc', 'corp', 'the', 'and', 'of', 'at', 'by', 'on',
        'beauty', 'barber', 'dental', 'fitness', 'gym', 'yoga',
        'cleaners', 'cleaning', 'pharmacy', 'supply', 'supplies',
        'express', 'boutique', 'design', 'center', 'club',
    }

    # Add the name without trailing generic words (e.g. "al masri" from "al masri egyptian restaurant")
    core_words = []
    for w in words:
        if w in generic and len(core_words) > 0:
            break
        core_words.append(w)
    if len(core_words) >= 2 and core_words != words:
        names.append(' '.join(core_words))

    # Add individual significant words (5+ chars, not generic)
    for w in words:
        if len(w) >= 5 and w not in generic:
            names.append(w)

    # Add consecutive 2-word pairs (catches things like "peri peri", "dim sum")
    for i in range(len(words) - 1):
        pair = f"{words[i]} {words[i+1]}"
        if len(pair) >= 6 and words[i] not in generic and words[i+1] not in generic:
            names.append(pair)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return unique


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ocr_model.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    texts = detect_text(image_path)

    print(f"\n=== OCR Detected Text ===")
    if texts:
        for t in texts:
            print(f"  '{t['text']}' (confidence: {t['confidence']}, bbox: {t['bbox_pct']})")
    else:
        print("  (none)")
