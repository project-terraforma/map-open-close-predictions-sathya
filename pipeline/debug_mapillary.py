import requests
import os
from dotenv import load_dotenv

load_dotenv()

def debug():
    token = os.getenv("MAPILLARY_ACCESS_TOKEN")
    print(f"Token: {token[:10]}...") 
    
    bbox = "-122.42,37.77,-122.41,37.78"
    url = "https://graph.mapillary.com/images"
    params = {
        "access_token": token,
        "fields": "id,computed_geometry,captured_at,thumb_2048_url",
        "bbox": bbox,
        "limit": 5
    }
    
    print(f"Requesting {url} with bbox {bbox}")
    resp = requests.get(url, params=params)
    print(f"Status: {resp.status_code}")
    try:
        data = resp.json()
        print(f"Data keys: {data.keys()}")
        if "data" in data:
            print(f"Count: {len(data['data'])}")
            print(f"Sample: {data['data'][0] if data['data'] else 'None'}")
        else:
            print("No 'data' key in response.")
            print(data)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        print(resp.text)

if __name__ == "__main__":
    debug()
