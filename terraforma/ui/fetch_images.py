"""
Pre-fetch Mapillary street-level images for ground truth businesses.
Embeds image URLs directly into ground_truth_map.html so no client-side API calls needed.

Usage:
    python ui/fetch_images.py
"""

import json
import math
import re
import requests
import time
from pathlib import Path

MAPILLARY_TOKEN = "MLY|25120378337641785|21245babc5f6905ed3da857aba13bc87"

HTML_PATH = Path(__file__).parent / "ground_truth_map.html"

MAX_DIST = 30  # hard cap: only images within 30m


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing(lat1, lon1, lat2, lon2):
    """Bearing in degrees from point 1 to point 2."""
    dLon = math.radians(lon2 - lon1)
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    x = math.sin(dLon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dLon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def angle_diff(a, b):
    """Smallest angle between two bearings (0-180)."""
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def get_bbox(lat, lon, radius_m):
    d_lat = radius_m / 111320
    d_lon = radius_m / (111320 * math.cos(math.radians(lat)))
    return f"{lon - d_lon},{lat - d_lat},{lon + d_lon},{lat + d_lat}"


def mapillary_images(lat, lon, limit=5):
    """Fetch nearest Mapillary images, preferring ones facing the business."""
    all_images = {}

    # Start very tight and expand — the API returns better results with small bboxes
    for r in [10, 15, 20, 30, 50, 75, 100]:
        try:
            bbox = get_bbox(lat, lon, r)
            resp = requests.get(
                "https://graph.mapillary.com/images",
                params={
                    "fields": "id,captured_at,thumb_1024_url,compass_angle,geometry",
                    "bbox": bbox,
                    "limit": 50,
                },
                headers={"Authorization": f"OAuth {MAPILLARY_TOKEN}"},
                timeout=10,
            )
            if resp.status_code == 200:
                for img in resp.json().get("data", []):
                    coords = img["geometry"]["coordinates"]
                    img["_dist"] = haversine(lat, lon, coords[1], coords[0])
                    if img["id"] not in all_images:
                        all_images[img["id"]] = img
                # Stop once we have enough close + facing images
                good = [i for i in all_images.values()
                        if i["_dist"] <= MAX_DIST and _faces_target(i, lat, lon)]
                if len(good) >= 3:
                    print(f"    Found {len(good)} good images at r={r}m, stopping")
                    break
            else:
                print(f"    Mapillary HTTP {resp.status_code} (r={r}m)")
        except Exception as e:
            print(f"    Mapillary error (r={r}m): {e}")
        time.sleep(0.2)

    # Score: prefer close + facing the business
    candidates = []
    for img in all_images.values():
        if img["_dist"] > MAX_DIST:
            continue
        facing = _faces_target(img, lat, lon)
        # Score: lower is better. Facing images get a bonus (subtract 15m from effective dist)
        score = img["_dist"] - (15 if facing else 0)
        candidates.append((score, img, facing))

    candidates.sort(key=lambda x: x[0])

    # If no images within MAX_DIST, fall back to closest regardless
    if not candidates:
        fallback = sorted(all_images.values(), key=lambda x: x["_dist"])
        if fallback:
            print(f"    No images within {MAX_DIST}m, using closest at {fallback[0]['_dist']:.0f}m")
        return fallback[:limit]

    return [c[1] for c in candidates[:limit]]


def _faces_target(img, target_lat, target_lon):
    """Check if the camera is roughly facing toward the target (within 60°)."""
    compass = img.get("compass_angle")
    if compass is None:
        return False
    coords = img["geometry"]["coordinates"]
    img_lat, img_lon = coords[1], coords[0]
    # Bearing from image location to the business
    to_target = bearing(img_lat, img_lon, target_lat, target_lon)
    return angle_diff(compass, to_target) < 60


def fetch_for_business(biz):
    """Fetch street view data for a single business."""
    name = biz["name"]
    lat, lon = biz["latitude"], biz["longitude"]

    biz["mapillary_link"] = f"https://www.mapillary.com/app/?lat={lat}&lng={lon}&z=18"

    print(f"  [{name}] Mapillary search (max {MAX_DIST}m, prefer facing)...")
    images = mapillary_images(lat, lon, 5)

    if images:
        dist = round(images[0]["_dist"])
        total = len(images)
        facing = sum(1 for i in images if _faces_target(i, lat, lon))
        print(f"    -> {total} image(s), closest {dist}m | {facing} facing business")
        biz["sv_images"] = []
        for img in images:
            biz["sv_images"].append({
                "id": img["id"],
                "thumb_url": img.get("thumb_1024_url", ""),
                "captured_at": img.get("captured_at"),
                "dist_m": round(img["_dist"]),
            })
        biz["sv_source"] = "overture"
    else:
        print(f"    -> No images found")
        biz["sv_images"] = []
        biz["sv_source"] = None

    time.sleep(0.3)


def main():
    html = HTML_PATH.read_text(encoding="utf-8")

    match = re.search(r"const data = (\[.*?\]);", html, re.DOTALL)
    if not match:
        print("ERROR: Could not find data array in HTML")
        return

    data = json.loads(match.group(1))
    print(f"Found {len(data)} businesses\n")

    for i, biz in enumerate(data):
        print(f"[{i+1}/{len(data)}] {biz['name']}")
        fetch_for_business(biz)
        print()

    with_images = sum(1 for b in data if b.get("sv_images"))
    print(f"\nDone: {with_images}/{len(data)} businesses have street-level images")

    new_data_str = json.dumps(data)
    new_html = html[:match.start(1)] + new_data_str + html[match.end(1):]
    HTML_PATH.write_text(new_html, encoding="utf-8")
    print(f"Updated {HTML_PATH}")


if __name__ == "__main__":
    main()
