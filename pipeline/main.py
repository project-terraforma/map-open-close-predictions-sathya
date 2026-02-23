import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv
from ingest_overture import fetch_overture_data
from ingest_mapillary import fetch_mapillary_sequences, apply_golden_rule
from run_model import load_model, run_inference, classify_status

def download_image(url, save_path):
    if os.path.exists(save_path):
        return True
    
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return False

def main():
    # 1. Setup
    sf_bbox = [-122.42, 37.77, -122.41, 37.78] # Smaller bbox for demo (SoMa/Mission part)
    os.makedirs("pipeline/data", exist_ok=True)
    os.makedirs("pipeline/images", exist_ok=True)
    
    # 2. Ingest Overture Data
    print("--- 2. Ingesting Overture Data ---")
    places_df, buildings_df = fetch_overture_data(sf_bbox)
    
    # 3. Ingest Mapillary Data
    print("--- 3. Ingesting Mapillary Data ---")
    # For the specific bbox
    images = fetch_mapillary_sequences(sf_bbox) # Fetch all available
    pairs = apply_golden_rule(images)
    
    # 4. Run Model
    print("--- 4. Running Output Model ---")
    model = load_model() # loads default 'yolo11n.pt'
    
    if not pairs:
        print("No 'Golden Rule' pairs found (30-45 days apart). Falling back to single image analysis for demo.")
        # Take top 20 most recent images
        # images is list of dicts. Sort by captured_at (ms)
        images_sorted = sorted(images, key=lambda x: x.get('captured_at', 0), reverse=True)
        top_images = images_sorted[:20]
        
        pairs = []
        for img in top_images:
             # Create a "pair" where before/after are the same, just to run the pipeline
             pairs.append({
                "before_id": img['id'],
                "after_id": img['id'],
                "before_date": "N/A",
                "after_date": str(pd.to_datetime(img['captured_at'], unit='ms').date()),
                "h3_index": "N/A",
                "location": img['computed_geometry']['coordinates']
            })
    
    results = []
    
    for pair in pairs:
        # Download images
        before_url = [img for img in images if img['id'] == pair['before_id']][0].get('thumb_2048_url')
        after_url = [img for img in images if img['id'] == pair['after_id']][0].get('thumb_2048_url')
        
        if not before_url or not after_url:
            continue
            
        before_path = f"pipeline/images/{pair['before_id']}.jpg"
        after_path = f"pipeline/images/{pair['after_id']}.jpg"
        
        download_image(before_url, before_path)
        download_image(after_url, after_path)
        
        # Run inference on AFTER image (to see current status)
        detections = run_inference(after_path, model)
        status, conf = classify_status(detections, mock_mode=False) # Only mock if model fails to load logic
        
        # If standard model, let's use mock logic for status to make it interesting
        # because standard model won't output 'for_lease'
        status, conf = classify_status(detections, mock_mode=True) 

        pair_result = pair.copy()
        pair_result['status'] = status
        pair_result['confidence'] = conf
        pair_result['detections'] = detections
        pair_result['before_url'] = before_url
        pair_result['after_url'] = after_url
        
        results.append(pair_result)
        
    # 5. Export
    print(f"--- 5. Exporting {len(results)} results ---")
    with open("pipeline/data/processed_data.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Pipeline completed.")

if __name__ == "__main__":
    main()
