import json
import random

# Center on SoMa, San Francisco
base_lat = 37.7749
base_lng = -122.4194

# Generate 100 points
data = []
for i in range(100):
    # Random offset within ~1km
    lat = base_lat + random.uniform(-0.01, 0.01)
    lng = base_lng + random.uniform(-0.01, 0.01)
    
    status = "Open" if random.random() > 0.2 else "Closed"
    
    # Use real placeholder images from Unsplash or similar if possible, 
    # but strictly use the existing pattern for now or placeholder URLs.
    # We will use reliable placeholder image services for demo.
    
    # Randomly pick from a set of realistic looking "storefront" images
    # These are just placeholders for the demo
    before_img = f"https://picsum.photos/seed/{i}_before/400/300"
    after_img = f"https://picsum.photos/seed/{i}_after/400/300"

    item = {
        "id": f"poi_{i}",
        "location": [lng, lat], # GeoJSON style [lng, lat]
        "status": status,
        "confidence": round(random.uniform(0.7, 0.99), 2),
        "before_date": "2023-01-15",
        "after_date": "2023-03-20",
        "before_url": before_img,
        "after_url": after_img,
        "name": f"Business #{i+1}",
        "address": f"{random.randint(100, 999)} Mission St"
    }
    data.append(item)

with open('src/data/mock_data.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Generated {len(data)} mock data points.")
