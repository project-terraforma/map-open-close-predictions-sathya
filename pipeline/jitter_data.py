import json
import random

with open('src/data/mock_data.json', 'r') as f:
    data = json.load(f)

for i, item in enumerate(data):
    # Add small random jitter to coordinates (~5-10 meters)
    # 0.0001 degrees is roughly 11 meters
    lng_jitter = random.uniform(-0.0002, 0.0002)
    lat_jitter = random.uniform(-0.0002, 0.0002)
    
    item['location'] = [
        item['location'][0] + lng_jitter,
        item['location'][1] + lat_jitter
    ]
    
    # Force the 5th item to be Closed if it isn't, just to be sure
    if i == 4:
        item['status'] = 'Closed'
        item['confidence'] = 0.95

with open('src/data/mock_data.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Jittered {len(data)} locations.")
