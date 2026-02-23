import requests
import json
from datetime import datetime, timedelta
import os
import pandas as pd
from shapely.geometry import Point
from dotenv import load_dotenv

load_dotenv()

MAPILLARY_ACCESS_TOKEN = os.getenv("MAPILLARY_ACCESS_TOKEN", "YOUR_MAPILLARY_ACCESS_TOKEN")

def fetch_mapillary_sequences(bbox, start_date=None, end_date=None):
    """
    Fetches Mapillary image metadata within the bbox and date range.
    """
    print(f"Fetching Mapillary images for bbox: {bbox}")
    
    if MAPILLARY_ACCESS_TOKEN == "YOUR_MAPILLARY_ACCESS_TOKEN":
        print("WARNING: MAPILLARY_ACCESS_TOKEN not set. Skipping API call.")
        return []

    # Mapillary API endpoint
    url = "https://graph.mapillary.com/images"
    params = {
        "access_token": MAPILLARY_ACCESS_TOKEN,
        "fields": "id,computed_geometry,captured_at,sequence,compass_angle,thumb_2048_url",
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "limit": 2000 
    }
    
    if start_date:
        params["start_captured_at"] = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    if end_date:
        params["end_captured_at"] = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    print(f"Requesting: {url} with params: {params}")

    images = []
    while True:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error fetching Mapillary data: {response.text}")
            break
            
        data = response.json()
        images.extend(data.get("data", []))
        
        # Pagination
        if "paging" in data and "next" in data["paging"]:
            url = data["paging"]["next"]
            params = {} # params are encoded in the next url
        else:
            break
            
    print(f"Fetched {len(images)} images.")
    return images

def apply_golden_rule(images):
    """
    Filters sequences to find pairs separated by 30-45 days for the same approximate location.
    
    This is a simplified heuristic:
    1. Group images by Sequence ID (assuming sequences are single-day).
    2. Check spatial overlap between sequences.
    3. If overlap, check time difference.
    """
    # Convert to DataFrame for easier handling
    if not images:
        return []

    df = pd.DataFrame(images)
    df['captured_at'] = pd.to_datetime(df['captured_at'], unit='ms')
    
    # Simple logic: Find images within 10 meters of each other but 30-45 days apart
    # This is O(N^2) effectively, so we need spatial indexing (S2 or H3 or simple grid)
    # For demo, we might just look for sequences in the same H3 cell
    
    import h3
    
    # Add H3 index (resolution 10, approx 66m edge)
    def get_h3(row):
        # coordinates are [lon, lat] in mapillary
        geom = row.get('computed_geometry')
        if not isinstance(geom, dict) or 'coordinates' not in geom:
            return None
        coords = geom['coordinates']
        try:
            return h3.latlng_to_cell(coords[1], coords[0], 10)
        except AttributeError:
             return h3.geo_to_h3(coords[1], coords[0], 10)
        except Exception as e:
            print(f"Error in get_h3: {e}, coords: {coords}")
            return None
    
    df['h3_index'] = df.apply(get_h3, axis=1)
    
    pairs = []
    
    # Group by H3 cell
    for h3_idx, group in df.groupby('h3_index'):
        # Sort by time
        group = group.sort_values('captured_at')
        
        # Iterate to find pairs
        # Depending on volume, this might be slow. Optimization: Compare days.
        
        # Just finding valid pairs for demonstration
        dates = group['captured_at'].dt.date.unique()
        
        for date1 in dates:
            for date2 in dates:
                if date1 >= date2:
                    continue
                    
                diff = (date2 - date1).days
                if 30 <= diff <= 45:
                    # Found a potential date pair in this cell
                    # Now match specific images/sequences
                    img1 = group[group['captured_at'].dt.date == date1].iloc[0] # Pick representative
                    img2 = group[group['captured_at'].dt.date == date2].iloc[0]
                    
                    pairs.append({
                        "before_id": img1['id'],
                        "after_id": img2['id'],
                        "before_date": str(date1),
                        "after_date": str(date2),
                        "h3_index": h3_idx,
                        "location": img1['computed_geometry']['coordinates']
                    })
                    
    print(f"Found {len(pairs)} pairs satisfying the Golden Rule.")
    return pairs

if __name__ == "__main__":
    # San Francisco Bounding Box
    sf_bbox = [-122.5155, 37.7038, -122.3556, 37.8125]
    
    # Run
    pass
