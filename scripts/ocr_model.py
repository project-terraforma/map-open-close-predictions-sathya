"""
Vision-based business detection.
Combines RapidOCR (text reading) + CLIP (logo/brand recognition) to identify
what businesses are visible in a street-level image.

Improvements:
  - Multi-pass OCR: original, enhanced contrast, CLAHE, grayscale
  - Lower OCR thresholds to catch faint/distant signs
  - More CLIP crops (15 total) for partial logo detection
  - Brand name aliases for better matching
"""

import re
import os
import cv2
import numpy as np
import torch
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
import open_clip

# ── Known fast food / restaurant chains to scan for ──
KNOWN_CHAINS = [
    "McDonald's", "Burger King", "Wendy's", "Taco Bell", "Subway",
    "Chick-fil-A", "KFC", "Popeyes", "Five Guys", "In-N-Out Burger",
    "Jack in the Box", "Carl's Jr", "Hardee's", "Sonic Drive-In",
    "Arby's", "Panda Express", "Chipotle", "Domino's", "Pizza Hut",
    "Papa John's", "Little Caesars", "Wingstop", "Raising Cane's",
    "Whataburger", "Culver's", "Shake Shack", "El Pollo Loco",
    "Church's Chicken", "Zaxby's", "Starbucks", "Dunkin'",
    "Jamba Juice", "Baskin-Robbins", "Dairy Queen",
    "Krispy Kreme", "Tim Hortons", "Panera Bread",
    "Jimmy John's", "Jersey Mike's", "Firehouse Subs",
    "Denny's", "IHOP", "Waffle House", "Cracker Barrel",
    "Applebee's", "Chili's", "Olive Garden", "Red Lobster",
    "Buffalo Wild Wings", "Hooters", "TGI Friday's",
    "Del Taco", "Jollibee",
]

# ── Name aliases: maps normalized business names to the matching chain names ──
# This handles cases like "SUBWAY" vs "Subway", "KFC" vs "Kentucky Fried Chicken"
BRAND_ALIASES = {
    "mcdonalds": ["mcdonalds"],
    "burger king": ["burger king"],
    "taco bell": ["taco bell"],
    "taco bell express": ["taco bell"],
    "taco bell mobile delivery charter oak cloudkitchens": ["taco bell"],
    "taco bell kfc": ["taco bell", "kfc"],
    "subway": ["subway"],
    "kfc": ["kfc"],
    "popeyes": ["popeyes"],
    "popeyes louisiana kitchen": ["popeyes"],
    "dominos pizza": ["dominos"],
    "dominos": ["dominos"],
    "chipotle mexican grill": ["chipotle"],
    "chipotle": ["chipotle"],
    "jack in the box": ["jack in the box"],
    "carls jr": ["carls jr"],
    "carls jr green burrito": ["carls jr"],
    "little caesars": ["little caesars"],
    "little caesars pizza": ["little caesars"],
    "panda express": ["panda express"],
    "wingstop": ["wingstop"],
    "five guys": ["five guys"],
    "in n out burger": ["in n out burger"],
    "shake shack": ["shake shack"],
    "del taco": ["del taco"],
    "jollibee": ["jollibee"],
    "jollibee nha kauswagan": ["jollibee"],
    "san francisco": [],  # not a chain
}

# ── Singletons ──
_ocr_engine = None
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_chain_text_features = None


def get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        print("Loading RapidOCR engine...", flush=True)
        _ocr_engine = RapidOCR()
        print("RapidOCR ready.", flush=True)
    return _ocr_engine


def get_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer, _chain_text_features
    if _clip_model is None:
        print("Loading CLIP model (ViT-B-16 datacomp)...", flush=True)
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='datacomp_xl_s13b_b90k'
        )
        _clip_tokenizer = open_clip.get_tokenizer('ViT-B-16')
        _clip_model.eval()

        # Pre-encode with multiple prompt templates for better matching
        templates = [
            "a photo of a {} restaurant",
            "a {} sign on a building",
            "a {} logo",
            "a {} storefront",
            "a {} fast food restaurant sign",
            "the {} brand logo on a building",
            "a street photo showing a {} restaurant",
            "a {} drive through",
        ]
        all_prompts = []
        for name in KNOWN_CHAINS:
            all_prompts.extend([t.format(name) for t in templates])

        tokens = _clip_tokenizer(all_prompts)
        with torch.no_grad():
            all_features = _clip_model.encode_text(tokens)
            all_features /= all_features.norm(dim=-1, keepdim=True)
            n_templates = len(templates)
            _chain_text_features = torch.stack([
                all_features[i * n_templates:(i + 1) * n_templates].mean(dim=0)
                for i in range(len(KNOWN_CHAINS))
            ])
            _chain_text_features /= _chain_text_features.norm(dim=-1, keepdim=True)
        print("CLIP ready.", flush=True)
    return _clip_model, _clip_preprocess, _clip_tokenizer, _chain_text_features


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


def _get_crops(img):
    """Generate 15 crops to catch logos and signs at any position."""
    w, h = img.size
    crops = [img]  # 1. full image

    # 2. Center crop (60%)
    cx, cy = w // 2, h // 2
    cw, ch = int(w * 0.6), int(h * 0.6)
    crops.append(img.crop((cx - cw // 2, cy - ch // 2, cx + cw // 2, cy + ch // 2)))

    # 3-4. Upper/lower half
    crops.append(img.crop((0, 0, w, h // 2)))
    crops.append(img.crop((0, h // 2, w, h)))

    # 5-6. Left/right 2/3
    third = w // 3
    crops.append(img.crop((0, 0, third * 2, h)))
    crops.append(img.crop((third, 0, w, h)))

    # 7-10. Four quadrants
    crops.append(img.crop((0, 0, w // 2, h // 2)))
    crops.append(img.crop((w // 2, 0, w, h // 2)))
    crops.append(img.crop((0, h // 2, w // 2, h)))
    crops.append(img.crop((w // 2, h // 2, w, h)))

    # 11-13. Vertical thirds (signs often in center third)
    crops.append(img.crop((0, 0, third, h)))
    crops.append(img.crop((third, 0, third * 2, h)))
    crops.append(img.crop((third * 2, 0, w, h)))

    # 14. Center 80% (removes borders/sky)
    margin_x, margin_y = int(w * 0.1), int(h * 0.1)
    crops.append(img.crop((margin_x, margin_y, w - margin_x, h - margin_y)))

    # 15. Upper center (where most storefront signs are)
    crops.append(img.crop((w // 4, 0, w * 3 // 4, h // 2)))

    return crops


def _get_crop_regions():
    """Return the 15 crop regions as percentage rects [x, y, w, h] in 0-1 range.
    Matches the crops generated by _get_crops() in the same order."""
    return [
        [0.0, 0.0, 1.0, 1.0],       # 0: full image
        [0.2, 0.2, 0.6, 0.6],       # 1: center 60%
        [0.0, 0.0, 1.0, 0.5],       # 2: upper half
        [0.0, 0.5, 1.0, 0.5],       # 3: lower half
        [0.0, 0.0, 0.667, 1.0],     # 4: left 2/3
        [0.333, 0.0, 0.667, 1.0],   # 5: right 2/3
        [0.0, 0.0, 0.5, 0.5],       # 6: top-left quadrant
        [0.5, 0.0, 0.5, 0.5],       # 7: top-right quadrant
        [0.0, 0.5, 0.5, 0.5],       # 8: bottom-left quadrant
        [0.5, 0.5, 0.5, 0.5],       # 9: bottom-right quadrant
        [0.0, 0.0, 0.333, 1.0],     # 10: left third
        [0.333, 0.0, 0.333, 1.0],   # 11: center third
        [0.667, 0.0, 0.333, 1.0],   # 12: right third
        [0.1, 0.1, 0.8, 0.8],       # 13: center 80%
        [0.25, 0.0, 0.5, 0.5],      # 14: upper center
    ]


def detect_brands(image_path, top_k=5):
    """Use CLIP with multi-crop to identify brands/logos.
    Returns brands with best_crop_region showing where the logo was detected."""
    model, preprocess, tokenizer, chain_features = get_clip()

    img = Image.open(image_path).convert("RGB")
    crops = _get_crops(img)
    crop_regions = _get_crop_regions()

    img_tensors = torch.stack([preprocess(c) for c in crops])

    with torch.no_grad():
        image_features = model.encode_image(img_tensors)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        all_similarities = image_features @ chain_features.T
        similarities, best_crop_indices = all_similarities.max(dim=0)

    top_indices = similarities.topk(top_k).indices.tolist()
    brands = []
    for idx in top_indices:
        score = similarities[idx].item()
        crop_idx = best_crop_indices[idx].item()
        brands.append({
            "name": KNOWN_CHAINS[idx],
            "score": round(score, 4),
            "best_crop_idx": crop_idx,
            "best_crop_region": crop_regions[crop_idx],
        })
    return brands


def get_brand_search_names(business_name):
    """Get the list of chain names to search for given a business name.
    Handles aliases like 'Taco Bell Express' -> search for 'Taco Bell'."""
    norm = normalize(business_name)
    if norm in BRAND_ALIASES:
        return BRAND_ALIASES[norm]
    # Fallback: just use the normalized name
    return [norm]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ocr_model.py <image_path> [business_name]")
        sys.exit(1)

    image_path = sys.argv[1]
    business_name = sys.argv[2] if len(sys.argv) > 2 else None

    texts = detect_text(image_path)
    brands = detect_brands(image_path)

    print(f"\n=== OCR Detected Text ===")
    if texts:
        for t in texts:
            print(f"  '{t['text']}' (confidence: {t['confidence']})")
    else:
        print("  (none)")

    print(f"\n=== CLIP Brand Recognition ===")
    for b in brands:
        bar = "#" * int(b["score"] * 100)
        print(f"  {b['name']:25s} {b['score']:.4f} {bar}")
