"""
Patch existing mock_data.json with spatial bounding box data.
Re-downloads images and runs OCR + CLIP to extract bbox_pct and crop_region,
then merges into existing data WITHOUT re-running scoring/Foursquare/Overture.
"""

import json
import os
import sys
import tempfile
import urllib.request

# Add scripts dir to path for ocr_model import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_model import detect_text, detect_brands, normalize, get_brand_search_names

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "data", "mock_data.json")

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')


def download_image(url, tmp_dir):
    """Download image to temp file, return path or None."""
    try:
        tmp_path = os.path.join(tmp_dir, f"img_{hash(url) & 0xFFFFFFFF}.jpg")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            with open(tmp_path, "wb") as f:
                f.write(resp.read())
        return tmp_path
    except Exception as e:
        print(f"    Download failed: {e}", flush=True)
        return None


def patch_location(loc, tmp_dir):
    """Patch a single location's vision data with spatial info."""
    vision = loc.get("vision")
    if not vision:
        return False

    per_image = vision.get("per_image", [])
    if not per_image:
        return False

    name = loc.get("name", "")
    search_names = get_brand_search_names(name)

    patched_any = False
    all_texts_detail = []

    for img_i, pi in enumerate(per_image):
        url = pi.get("url")
        if not url:
            continue

        # Download image
        tmp_path = download_image(url, tmp_dir)
        if not tmp_path:
            continue

        try:
            # Run OCR with bbox
            texts = detect_text(tmp_path)
            pi["texts_with_bbox"] = [
                {"text": t["text"], "confidence": t["confidence"], "bbox_pct": t.get("bbox_pct")}
                for t in texts
            ]

            # Collect for layer-level detail
            img_idx = pi.get("idx", img_i)
            for t in texts:
                all_texts_detail.append({
                    "text": t["text"],
                    "confidence": t["confidence"],
                    "bbox_pct": t.get("bbox_pct"),
                    "image_url": url,
                    "image_idx": img_idx,
                })

            # Run CLIP with crop regions
            brands = detect_brands(tmp_path)
            if brands:
                pi["clip_crop_region"] = brands[0].get("best_crop_region")
                # Check if the top brand matches this location
                for b in brands:
                    norm_brand = normalize(b["name"])
                    for sn in search_names:
                        if sn in norm_brand or norm_brand in sn:
                            pi["clip_crop_region"] = b.get("best_crop_region")
                            break

            patched_any = True
            print(f"      img{img_i}: OCR={len(texts)} texts, CLIP top={brands[0]['name'] if brands else '?'}", flush=True)
        except Exception as e:
            print(f"      img{img_i}: ERROR {e}", flush=True)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # Patch layer-level data
    if patched_any:
        layers = vision.get("layers", {})

        # Patch text layer with ocr_texts_detail
        if "text" in layers:
            layers["text"]["ocr_texts_detail"] = all_texts_detail

        # Patch logo layer with best_crop_region
        if "logo" in layers:
            best_logo_idx = layers["logo"].get("best_image_idx")
            if best_logo_idx is not None:
                for pi in per_image:
                    if pi.get("idx") == best_logo_idx and pi.get("clip_crop_region"):
                        layers["logo"]["best_crop_region"] = pi["clip_crop_region"]
                        break
            # Fallback: use first image with a crop region
            if not layers["logo"].get("best_crop_region"):
                for pi in per_image:
                    if pi.get("clip_crop_region"):
                        layers["logo"]["best_crop_region"] = pi["clip_crop_region"]
                        break

    return patched_any


def main():
    print("=" * 60, flush=True)
    print("TERRAFORMA - Patch Bounding Box Data", flush=True)
    print("=" * 60, flush=True)

    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} locations\n", flush=True)

    # Pre-load models once
    print("Pre-loading models...", flush=True)
    from ocr_model import get_ocr, get_clip
    get_ocr()
    get_clip()
    print("Models ready.\n", flush=True)

    patched = 0
    skipped = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, loc in enumerate(data):
            name = loc.get("name", "Unknown")
            per_image = loc.get("vision", {}).get("per_image", [])

            if not per_image:
                print(f"[{i+1}/{len(data)}] {name} - no per_image data, skipping", flush=True)
                skipped += 1
                continue

            # Check if already patched
            has_bbox = any(pi.get("texts_with_bbox") for pi in per_image)
            has_crop = any(pi.get("clip_crop_region") for pi in per_image)
            if has_bbox and has_crop:
                print(f"[{i+1}/{len(data)}] {name} - already patched, skipping", flush=True)
                skipped += 1
                continue

            print(f"[{i+1}/{len(data)}] {name} - patching {len(per_image)} images...", flush=True)

            if patch_location(loc, tmp_dir):
                patched += 1
                n_texts = len(loc["vision"].get("layers", {}).get("text", {}).get("ocr_texts_detail", []))
                has_logo_region = bool(loc["vision"].get("layers", {}).get("logo", {}).get("best_crop_region"))
                print(f"    OK {n_texts} texts with bbox, logo region: {has_logo_region}", flush=True)
            else:
                skipped += 1

            # Save periodically every 10 locations
            if (i + 1) % 10 == 0:
                print(f"  [checkpoint] Saving progress ({i+1}/{len(data)})...", flush=True)
                with open(DATA_PATH, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)

    # Final save
    print(f"\nSaving to {DATA_PATH}...", flush=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Done! Patched {patched}, skipped {skipped}", flush=True)


if __name__ == "__main__":
    main()
