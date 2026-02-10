"""
Open/Closed Place Prediction Demo App with Map Visualization

Two tabs:
1. California Map - View predictions on a map with Google verification
2. Interactive Predictor - Test individual predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import folium
from streamlit_folium import st_folium
from feature_engineering import extract_features
import requests
import concurrent.futures


# Worldwide cities across all continents
WORLD_CITIES = [
    # North America
    {"name": "New York", "lat": 40.7128, "lon": -74.0060},
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
    {"name": "Toronto", "lat": 43.6532, "lon": -79.3832},
    {"name": "Mexico City", "lat": 19.4326, "lon": -99.1332},
    # South America
    {"name": "São Paulo", "lat": -23.5505, "lon": -46.6333},
    {"name": "Buenos Aires", "lat": -34.6037, "lon": -58.3816},
    {"name": "Bogotá", "lat": 4.7110, "lon": -74.0721},
    {"name": "Lima", "lat": -12.0464, "lon": -77.0428},
    {"name": "Santiago", "lat": -33.4489, "lon": -70.6693},
    # Europe
    {"name": "London", "lat": 51.5074, "lon": -0.1278},
    {"name": "Paris", "lat": 48.8566, "lon": 2.3522},
    {"name": "Berlin", "lat": 52.5200, "lon": 13.4050},
    {"name": "Rome", "lat": 41.9028, "lon": 12.4964},
    {"name": "Madrid", "lat": 40.4168, "lon": -3.7038},
    {"name": "Amsterdam", "lat": 52.3676, "lon": 4.9041},
    {"name": "Stockholm", "lat": 59.3293, "lon": 18.0686},
    {"name": "Prague", "lat": 50.0755, "lon": 14.4378},
    # Asia
    {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
    {"name": "Shanghai", "lat": 31.2304, "lon": 121.4737},
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    {"name": "Seoul", "lat": 37.5665, "lon": 126.9780},
    {"name": "Bangkok", "lat": 13.7563, "lon": 100.5018},
    {"name": "Singapore", "lat": 1.3521, "lon": 103.8198},
    {"name": "Dubai", "lat": 25.2048, "lon": 55.2708},
    {"name": "Istanbul", "lat": 41.0082, "lon": 28.9784},
    {"name": "Jakarta", "lat": -6.2088, "lon": 106.8456},
    {"name": "Kuala Lumpur", "lat": 3.1390, "lon": 101.6869},
    # Africa
    {"name": "Cairo", "lat": 30.0444, "lon": 31.2357},
    {"name": "Lagos", "lat": 6.5244, "lon": 3.3792},
    {"name": "Nairobi", "lat": -1.2921, "lon": 36.8219},
    {"name": "Cape Town", "lat": -33.9249, "lon": 18.4241},
    {"name": "Casablanca", "lat": 33.5731, "lon": -7.5898},
    # Oceania
    {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
    {"name": "Melbourne", "lat": -37.8136, "lon": 144.9631},
    {"name": "Auckland", "lat": -36.8485, "lon": 174.7633},
]


# Page config
st.set_page_config(
    page_title="Open/Closed Place Prediction",
    page_icon="🗺️",
    layout="wide"
)


@st.cache_resource
def load_model():
    """Load trained model and artifacts."""
    model = pickle.load(open('model/model.pkl', 'rb'))
    label_encoder = pickle.load(open('model/label_encoder.pkl', 'rb'))
    feature_names = json.load(open('model/feature_names.json'))
    return model, label_encoder, feature_names


@st.cache_data
def load_california_data():
    """Load California business data for mapping."""
    if os.path.exists('california_businesses.parquet'):
        df = pd.read_parquet('california_businesses.parquet')
        return df
    return None


@st.cache_data
def load_historical_data():
    """Load historical LA business data."""
    if os.path.exists('la_historical_businesses.parquet'):
        return pd.read_parquet('la_historical_businesses.parquet')
    return None


@st.cache_data
def load_overture_ca_data():
    """Load Overture data with coordinates for map visualization.
    Note: User's labeled files don't have lat/lon, so we use downloaded Overture data.
    """
    # Use downloaded Overture data (has coordinates)
    if os.path.exists('us_overture.parquet'):
        return pd.read_parquet('us_overture.parquet')
    if os.path.exists('california_overture.parquet'):
        return pd.read_parquet('california_overture.parquet')
    return None


@st.cache_data
def load_user_labeled_data():
    """Load user's labeled parquet files (for Google Places lookup)."""
    dfs = []
    
    if os.path.exists('samples_3k_project_c_updated.parquet'):
        df = pd.read_parquet('samples_3k_project_c_updated.parquet')
        df['source_file'] = '3k_updated'
        dfs.append(df)
    
    if os.path.exists('project_c_samples.parquet'):
        df = pd.read_parquet('project_c_samples.parquet')
        df['source_file'] = 'project_c'
        dfs.append(df)
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None


def generate_realistic_coordinates(n=1):
    """Generate coordinates near California cities."""
    coords = []
    for _ in range(n):
        city = np.random.choice(CALIFORNIA_CITIES)
        # Add small random offset (within ~5 miles)
        lat = city['lat'] + np.random.uniform(-0.05, 0.05)
        lon = city['lon'] + np.random.uniform(-0.05, 0.05)
        coords.append({'lat': lat, 'lon': lon, 'city': city['name']})
    return coords


def check_google_places(name, lat, lon, api_key):
    """Check if a place is open using Google Places API."""
    try:
        # Search for the place
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{lat},{lon}",
            "radius": 100,
            "keyword": name,
            "key": api_key
        }
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        if data.get('results'):
            place = data['results'][0]
            # Check business status
            business_status = place.get('business_status', 'UNKNOWN')
            is_open = business_status == 'OPERATIONAL'
            return {
                'found': True,
                'actual_open': is_open,
                'business_status': business_status,
                'google_name': place.get('name', name),
                'rating': place.get('rating'),
                'total_ratings': place.get('user_ratings_total', 0)
            }
        return {'found': False, 'actual_open': None, 'business_status': 'NOT_FOUND'}
    except Exception as e:
        return {'found': False, 'actual_open': None, 'business_status': f'ERROR: {str(e)}'}


def search_google_places_text(name, address, api_key):
    """Search Google Places by business name and address to get coordinates."""
    try:
        # Use Text Search API
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        query = f"{name} {address}" if address else name
        params = {
            "query": query,
            "key": api_key
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get('results'):
            place = data['results'][0]
            location = place.get('geometry', {}).get('location', {})
            business_status = place.get('business_status', 'UNKNOWN')
            
            return {
                'found': True,
                'lat': location.get('lat'),
                'lon': location.get('lng'),
                'google_name': place.get('name'),
                'formatted_address': place.get('formatted_address'),
                'business_status': business_status,
                'is_open': business_status == 'OPERATIONAL',
                'rating': place.get('rating'),
                'total_ratings': place.get('user_ratings_total', 0),
                'place_id': place.get('place_id')
            }
        return {'found': False, 'lat': None, 'lon': None}
    except Exception as e:
        return {'found': False, 'lat': None, 'lon': None, 'error': str(e)}


def predict_place_features(features_dict, model, label_encoder, feature_names):
    """Make a prediction for a single place using a confidence-based heuristic.
    Returns balanced predictions based on data quality signals.
    """
    # Extract features
    confidence = features_dict.get('confidence', 0.5)
    if confidence is None:
        confidence = 0.5
    confidence = float(confidence)
    
    # Count sources
    sources = features_dict.get('sources', [])
    num_sources = len(sources) if isinstance(sources, list) else 1
    
    # Has website/phone/etc
    def has_data(x):
        if x is None: return 0
        if isinstance(x, list): return 1 if len(x) > 0 else 0
        if isinstance(x, dict): return 1 if len(x) > 0 else 0
        return 0
    
    has_websites = has_data(features_dict.get('websites'))
    has_phones = has_data(features_dict.get('phones'))
    has_address = 1 if features_dict.get('addresses') and len(features_dict.get('addresses', [])) > 0 else 0
    has_hours = features_dict.get('has_hours', None)  # None = unknown, True/False = known
    
    # Calculate Score (0-100)
    score = 50  # Start neutral
    
    # 1. Confidence Impact (Centered around 0.75, reduced weight)
    conf_impact = (confidence - 0.75) * 120
    score += conf_impact
    
    # 2. Proof of Life
    if has_websites: score += 8
    else: score -= 8
        
    if has_phones: score += 8
    else: score -= 8
        
    if has_address: score += 3
    
    # 3. Opening Hours - strongest signal of active operation
    if has_hours is not None:
        if has_hours:
            score += 15  # Has hours listed = strong open signal
        else:
            score -= 35  # NO hours listed = very strong closed signal
    
    # 4. Review count signal — few/no reviews suggests inactive
    total_ratings = features_dict.get('total_ratings', None)
    if total_ratings is not None:
        if total_ratings == 0:
            score -= 10
        elif total_ratings < 5:
            score -= 5
    
    # 5. Source Bonus (reduced)
    if num_sources > 2: score += 5
    
    # Clamp
    score = np.clip(score, 0, 100)
    prob_open = score / 100.0
    
    # Determine prediction
    prediction = 1 if score > 62 else 0
    probabilities = np.array([1 - prob_open, prob_open])
    
    # Explanation (for debugging/user info)
    reasons = []
    if confidence > 0.8: reasons.append("High Confidence")
    elif confidence < 0.7: reasons.append("Low Confidence")
    
    if has_websites: reasons.append("Has Website")
    else: reasons.append("Missing Website")
    
    if has_phones: reasons.append("Has Phone")
    else: reasons.append("Missing Phone")
    
    if has_hours is not None:
        if has_hours: reasons.append("Has Hours")
        else: reasons.append("No Hours Listed")
    
    if total_ratings is not None and total_ratings < 5:
        reasons.append("Few Reviews")
    
    return prediction, probabilities, ", ".join(reasons)


def generate_random_business():
    """Generate a random business with realistic attributes."""
    categories = ['restaurant', 'cafe', 'grocery_store', 'pharmacy', 
                  'gas_station', 'bank', 'hotel', 'gym', 'bar', 'bakery']
    
    business_names = {
        'restaurant': ['The Golden Spoon', 'Mama\'s Kitchen', 'Urban Eats', 'Sunset Grill', 'The Local Spot'],
        'cafe': ['Morning Brew', 'The Coffee House', 'Bean There', 'Espresso Lane', 'Cafe Mocha'],
        'grocery_store': ['Fresh Mart', 'City Market', 'Daily Grocers', 'Green Valley Foods', 'Corner Store'],
        'pharmacy': ['HealthPlus', 'Care Pharmacy', 'MediCare Rx', 'Wellness Drugs', 'Family Pharmacy'],
        'gas_station': ['Quick Stop Gas', 'Fuel Up', 'Highway Fuel', 'City Gas', 'Express Fuel'],
        'bank': ['First National', 'Community Bank', 'Trust Savings', 'Metro Bank', 'Pacific Credit Union'],
        'hotel': ['Comfort Inn', 'City Lodge', 'Sunset Suites', 'Plaza Hotel', 'Garden Inn'],
        'gym': ['FitLife', 'PowerHouse Gym', 'Iron Works', 'Flex Fitness', '24/7 Gym'],
        'bar': ['The Tipsy Owl', 'Night Cap', 'Cheers Bar', 'The Pub', 'Moonlight Tavern'],
        'bakery': ['Sweet Delights', 'Golden Crust', 'The Bread Box', 'Sugar & Spice', 'Fresh Baked']
    }
    
    category = np.random.choice(categories)
    name = np.random.choice(business_names[category])
    coord = generate_realistic_coordinates(1)[0]
    
    # Random attributes
    has_website = np.random.random() > 0.3
    has_phone = np.random.random() > 0.2
    has_social = np.random.random() > 0.5
    has_email = np.random.random() > 0.6
    has_brand = np.random.random() > 0.7
    num_sources = np.random.randint(1, 6)
    confidence = np.random.uniform(0.3, 0.98)
    
    features = {
        'id': f'demo_{np.random.randint(10000)}',
        'sources': [{'dataset': 'meta' if np.random.random() > 0.5 else 'other'}] * num_sources,
        'confidence': confidence,
        'websites': ['https://example.com'] if has_website else None,
        'phones': ['+1-555-1234'] if has_phone else None,
        'socials': ['https://facebook.com/example'] if has_social else None,
        'emails': ['contact@example.com'] if has_email else None,
        'brand': {'names': {'primary': 'Brand'}} if has_brand else None,
        'categories': {'primary': category},
        'addresses': [{'freeform': f'123 Main St, {coord["city"]}, CA', 'locality': coord['city'], 'country': 'US'}],
        'names': {'primary': name},
        'version': num_sources,
    }
    
    return {
        'name': name,
        'category': category,
        'lat': coord['lat'],
        'lon': coord['lon'],
        'city': coord['city'],
        'features': features,
        'has_website': has_website,
        'has_phone': has_phone,
        'has_social': has_social,
        'has_email': has_email,
        'has_brand': has_brand,
        'num_sources': num_sources,
        'confidence': confidence,
    }


def search_google_nearby(lat, lon, api_key, place_type='restaurant', radius=2000):
    """Search for real businesses near a location using Google Places API."""
    try:
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{lat},{lon}",
            "radius": radius,
            "type": place_type,
            "key": api_key
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        results = []
        for place in data.get('results', []):
            loc = place.get('geometry', {}).get('location', {})
            business_status = place.get('business_status', 'UNKNOWN')
            is_open = business_status == 'OPERATIONAL'
            
            results.append({
                'name': place.get('name', 'Unknown'),
                'category': place_type,
                'lat': loc.get('lat'),
                'lon': loc.get('lng'),
                'business_status': business_status,
                'is_open': is_open,
                'rating': place.get('rating'),
                'total_ratings': place.get('user_ratings_total', 0),
                'vicinity': place.get('vicinity', ''),
                'place_id': place.get('place_id'),
            })
        return results
    except Exception as e:
        return []


def get_place_details(place_id, api_key):
    """Fetch website, phone, and opening hours from Google Place Details API."""
    try:
        url = "https://maps.googleapis.com/maps/api/place/details/json"
        params = {
            "place_id": place_id,
            "fields": "website,formatted_phone_number,opening_hours",
            "key": api_key
        }
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        result = data.get('result', {})
        return {
            'website': result.get('website'),
            'phone': result.get('formatted_phone_number'),
            'has_hours': 'opening_hours' in result,
        }
    except Exception:
        return {'website': None, 'phone': None, 'has_hours': False}


def get_worldwide_businesses(api_key, n=100, target_status=None):
    """Fetch businesses from cities worldwide using Google Places API.
    
    Args:
        api_key: Google API key
        n: target number of businesses
        target_status: 'open', 'closed', or None for both
    """
    place_types = ['restaurant', 'cafe', 'store', 'pharmacy', 'bank', 
                   'gym', 'hotel', 'gas_station', 'bakery', 'bar']
    
    # Shuffle cities for variety
    cities = WORLD_CITIES.copy()
    np.random.shuffle(cities)
    
    businesses = []
    
    # When looking for closed businesses, search harder (more types, bigger radius)
    if target_status == 'closed':
        types_per_city = 3  # Search 3 random types per city (was ALL 10)
        search_radius = 10000  # 10km
        max_per_search = 20
    else:
        types_per_city = 1  # 1 random type
        search_radius = 5000
        max_per_search = max(5, n // len(cities) + 1)
    
    def build_business(p, city, details):
        """Convert a Google Places result into a business dict using real details."""
        if p['lat'] is None or p['lon'] is None:
            return None
        
        # Use REAL website/phone data from Place Details
        has_website = details.get('website') is not None
        has_phone = details.get('phone') is not None
        has_reviews = (p.get('total_ratings', 0) or 0) > 0
        
        # More realistic confidence: based on actual data signals
        num_sources = 1
        if has_website: num_sources += 1
        if has_phone: num_sources += 1
        if has_reviews: num_sources += 1
        
        confidence = min(0.4 + (0.15 if has_website else 0) + 
                       (0.15 if has_phone else 0) + 
                       (0.1 if has_reviews else 0) +
                       (0.05 * min(num_sources, 3)), 0.95)
        
        
        
        features = {
            'id': p.get('place_id', f'gp_{np.random.randint(100000)}'),
            'sources': [{'dataset': 'google'}] * num_sources,
            'confidence': confidence,
            'websites': [details['website']] if has_website else None,
            'phones': [details['phone']] if has_phone else None,
            'socials': None,
            'emails': None,
            'brand': None,
            'categories': {'primary': p['category']},
            'addresses': [{'freeform': p.get('vicinity', ''), 'locality': city['name']}],
            'names': {'primary': p['name']},
            'version': num_sources,
            'has_hours': details.get('has_hours', False),
            'total_ratings': p.get('total_ratings', 0),
        }
        
        return {
            'name': p['name'],
            'category': p['category'],
            'lat': p['lat'],
            'lon': p['lon'],
            'city': city['name'],
            'features': features,
            'prob_open': 0,
            'prediction': 0,
            'is_real': True,
            'source': 'Google Places',
            'google': {
                'found': True,
                'actual_open': p['is_open'],
                'business_status': p['business_status'],
                'google_name': p['name'],
                'rating': p.get('rating'),
                'total_ratings': p.get('total_ratings', 0),
            }
        }
    
    def fetch_city(city):
        """Search for businesses in a single city."""
        city_results = []
        seen_ids = set()
        candidates = []
        
        # Pick random types for this city
        city_types = list(np.random.choice(place_types, size=min(types_per_city, len(place_types)), replace=False))
        
        for ptype in city_types:
            places = search_google_nearby(
                city['lat'], city['lon'], api_key, 
                place_type=ptype, radius=search_radius
            )
            for p in places[:max_per_search]:
                pid = p.get('place_id', '')
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                
                # For closed search: pre-filter to only non-OPERATIONAL businesses
                # This avoids expensive Place Details calls on businesses we'll discard
                if target_status == 'closed' and p.get('business_status') == 'OPERATIONAL':
                    continue
                
                candidates.append(p)
        
        # Fetch real website/phone/hours details in parallel
        def get_details_for(p):
            pid = p.get('place_id', '')
            if pid:
                return p, get_place_details(pid, api_key)
            return p, {'website': None, 'phone': None, 'has_hours': False}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as detail_executor:
            detail_results = list(detail_executor.map(get_details_for, candidates))
        
        for p, details in detail_results:
            b = build_business(p, city, details)
            if b:
                city_results.append(b)
        return city_results
    
    # Fetch from cities in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        city_results = list(executor.map(fetch_city, cities))
    
    for cr in city_results:
        businesses.extend(cr)
    
    # Filter by target status if specified
    if target_status == 'open':
        businesses = [b for b in businesses if b['google']['actual_open']]
    elif target_status == 'closed':
        businesses = [b for b in businesses if not b['google']['actual_open']]
    
    # Shuffle and limit
    np.random.shuffle(businesses)
    return businesses[:n]


def render_map_tab():
    """Render the worldwide map tab."""
    model, label_encoder, feature_names = load_model()
    
    st.subheader("🗺️ Worldwide Business Predictions")
    st.markdown("*Search for real businesses around the world and predict their status*")
    
    # Google API Key (Hardcoded for demo)
    api_key = "AIzaSyAinBHpEi7ysEdtcGtIeE2Ys1e38Z42Q3o"
    
    # Initialize session state
    if 'map_businesses' not in st.session_state:
        st.session_state.map_businesses = []
    if 'map_mode' not in st.session_state:
        st.session_state.map_mode = None
    
    # Show current mode indicator
    if st.session_state.map_mode == 'open':
        st.success("🟢 **Currently showing: Actually OPEN businesses** — Model is predicting whether it thinks each is open or closed")
    elif st.session_state.map_mode == 'closed':
        st.error("🔴 **Currently showing: Actually CLOSED businesses** — Model is predicting whether it thinks each is open or closed")
    
    # Buttons
    st.markdown("### Generate Businesses by Actual Status")
    st.caption("These search Google Places in cities worldwide to find truly open or closed businesses")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        if st.button("✅ Generate 100 Actually OPEN"):
            st.session_state.map_businesses = []
            st.session_state.map_mode = 'open'
            
            with st.spinner("Searching for OPEN businesses worldwide..."):
                open_businesses = get_worldwide_businesses(api_key, n=100, target_status='open')
                
                for b in open_businesses:
                    pred, probs, reason = predict_place_features(b['features'], model, label_encoder, feature_names)
                    b['prediction'] = pred
                    b['prob_open'] = probs[1]
                    b['reason'] = reason
                
                if open_businesses:
                    st.session_state.map_businesses = open_businesses
                    st.toast(f"Found {len(open_businesses)} Actually OPEN businesses worldwide!")
                    st.rerun()
                else:
                    st.error("Couldn't find enough open businesses")
    
    with col2:
        if st.button("❌ Generate 100 Actually CLOSED", type="secondary"):
            st.session_state.map_businesses = []
            st.session_state.map_mode = 'closed'
            
            with st.spinner("Searching for CLOSED businesses worldwide..."):
                closed_businesses = get_worldwide_businesses(api_key, n=100, target_status='closed')
                
                for b in closed_businesses:
                    pred, probs, reason = predict_place_features(b['features'], model, label_encoder, feature_names)
                    b['prediction'] = pred
                    b['prob_open'] = probs[1]
                    b['reason'] = reason
                
                if closed_businesses:
                    st.session_state.map_businesses = closed_businesses
                    st.toast(f"Found {len(closed_businesses)} Actually CLOSED businesses worldwide!")
                    st.rerun()
                else:
                    st.error("Couldn't find enough closed businesses")
    
    with col3:
        if st.button("🗑️ Clear"):
            st.session_state.map_businesses = []
            st.session_state.map_mode = None
            st.rerun()
    
    # Stats
    # Initialize filtered businesses
    businesses_to_display = st.session_state.map_businesses.copy() if st.session_state.map_businesses else []
    
    if st.session_state.map_businesses:
        businesses = st.session_state.map_businesses
        total = len(businesses)
        pred_open = sum(1 for b in businesses if b['prediction'] == 1)
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Businesses", total)
        col2.metric("Predicted Open", pred_open)
        col3.metric("Predicted Closed", total - pred_open)
        
        # Calculate accuracy from Google verification
        verified = [b for b in businesses if b.get('google', {}).get('found')]
        if verified:
            actually_open = [b for b in verified if b['google']['actual_open']]
            actually_closed = [b for b in verified if not b['google']['actual_open']]
            
            # Calculate accuracy
            correct = sum(1 for b in verified if b['prediction'] == (1 if b['google']['actual_open'] else 0))
            accuracy = (correct / len(verified)) * 100
            
            st.markdown("### Google Verification Results")
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Verified", len(verified))
            mcol2.metric("Actually Open", len(actually_open))
            mcol3.metric("Actually Closed", len(actually_closed))
            mcol4.metric("Model Accuracy", f"{accuracy:.1f}%")
    
    m = folium.Map(
        location=[20, 0],
        zoom_start=2,
        tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        attr='Google'
    )
    
    # Add markers for each business
    for business in businesses_to_display:
        # Determine color
        if business['prediction'] == 1:
            color = 'green'
            status = '✅ OPEN'
        else:
            color = 'red'
            status = '❌ CLOSED'
        
        # Build popup content
        prob_open = business['prob_open']
        if business['prediction'] == 1:
            conf_text = f"{prob_open:.0%} open"
        else:
            conf_text = f"{(1 - prob_open):.0%} closed"
        
        popup_html = f"""
        <div style="width: 250px; font-family: Arial, sans-serif;">
            <h3 style="margin: 0; color: #333;">{business['name']}</h3>
            <p style="color: #666; margin: 5px 0;"><b>Category:</b> {business['category']}</p>
            <p style="color: #666; margin: 5px 0;"><b>City:</b> {business['city']}</p>
            <hr style="margin: 10px 0;">
            <p><b>🤖 Model Prediction:</b> <span style="color: {color};">{status}</span></p>
            <p><b>Confidence:</b> {conf_text}</p>
            <p style="font-size: 12px; color: #555;"><i>Why? {business.get('reason', '')}</i></p>
        """
        
        if business.get('google') and business['google'].get('found'):
            g = business['google']
            actual_status = '✅ OPEN' if g['actual_open'] else '❌ CLOSED'
            actual_color = 'green' if g['actual_open'] else 'red'
            popup_html += f"""
            <hr style="margin: 10px 0;">
            <p><b>🔍 Google Actual:</b> <span style="color: {actual_color};">{actual_status}</span></p>
            <p><b>Status:</b> {g['business_status']}</p>
            """
            if g.get('rating'):
                popup_html += f"<p><b>Rating:</b> ⭐ {g['rating']} ({g['total_ratings']} reviews)</p>"
        elif business.get('google'):
            popup_html += """
            <hr style="margin: 10px 0;">
            <p style="color: #999;"><i>Not found on Google</i></p>
            """
        
        popup_html += "</div>"
        
        folium.CircleMarker(
            location=[business['lat'], business['lon']],
            radius=10,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(m)
    
    # Display map
    st_folium(m, width=None, height=500)
    
    # Business list
    if st.session_state.map_businesses:
        st.markdown("---")
        st.subheader("📋 Generated Businesses")
        
        data = []
        for b in st.session_state.map_businesses:
            row = {
                'Name': b['name'],
                'Category': b['category'],
                'City': b['city'],
                'Prediction': '✅ Open' if b['prediction'] == 1 else '❌ Closed',
                'Confidence': f"{b['prob_open']:.0%} open" if b['prediction'] == 1 else f"{(1 - b['prob_open']):.0%} closed",
            }
            if b.get('google') and b['google'].get('found'):
                row['Actual (Google)'] = '✅ Open' if b['google']['actual_open'] else '❌ Closed'
            else:
                row['Actual (Google)'] = 'N/A'
            data.append(row)
        
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)


def render_predictor_tab():
    """Render the interactive predictor tab."""
    model, label_encoder, feature_names = load_model()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🏪 Describe the Place")
        
        categories = ['restaurant', 'cafe', 'grocery_store', 'pharmacy', 
                     'gas_station', 'bank', 'hotel', 'gym', 'bar', 'bakery']
        category = st.selectbox("What type of business?", categories)
        
        st.markdown("**What contact info is available online?**")
        c1, c2, c3, c4 = st.columns(4)
        has_website = c1.checkbox("Website", value=True)
        has_phone = c2.checkbox("Phone", value=True)
        has_social = c3.checkbox("Social Media")
        has_email = c4.checkbox("Email")
        
        st.markdown("**How well-documented is this place?**")
        num_sources = st.slider("Number of databases listing it", 1, 5, 2)
        has_meta = st.checkbox("Listed on Facebook/Instagram")
        
        st.markdown("**Other signals**")
        c1, c2 = st.columns(2)
        has_brand = c1.checkbox("Known chain/franchise")
        has_address = c2.checkbox("Has complete address", value=True)
        
        if st.button("🔮 PREDICT", type="primary", use_container_width=True):
            sources = [{'dataset': 'meta' if has_meta else 'other'}] * num_sources
            confidence = min(0.5 + 0.1 * num_sources + (0.1 if has_website else 0) + (0.1 if has_phone else 0), 0.98)
            
            features = {
                'id': 'demo',
                'sources': sources,
                'confidence': confidence,
                'websites': ['https://example.com'] if has_website else None,
                'phones': ['+1-555-1234'] if has_phone else None,
                'socials': ['https://facebook.com/example'] if has_social else None,
                'emails': ['contact@example.com'] if has_email else None,
                'brand': {'names': {'primary': 'Brand'}} if has_brand else None,
                'categories': {'primary': category},
                'addresses': [{'freeform': '123 Main St'}] if has_address else [],
                'names': {'primary': 'Demo Business'},
                'version': num_sources,
            }
            
            prediction, probs = predict_place_features(features, model, label_encoder, feature_names)
            st.session_state['prediction'] = prediction
            st.session_state['probs'] = probs
    
    with col2:
        st.subheader("🎯 Prediction")
        
        if 'prediction' in st.session_state:
            prediction = st.session_state['prediction']
            probs = st.session_state['probs']
            
            if prediction == 1:
                st.success("## ✅ OPEN")
            else:
                st.error("## ❌ CLOSED")
            
            st.markdown("### Model Confidence")
            c1, c2 = st.columns(2)
            c1.metric("Open", f"{probs[1]:.0%}")
            c2.metric("Closed", f"{probs[0]:.0%}")
            st.progress(probs[1])
        else:
            st.info("Configure the place details and click **PREDICT**")


def main():
    st.title("🗺️ Open/Closed Place Prediction")
    st.markdown("*Predicting which businesses around the world are still open today*")
    
    tab1, tab2 = st.tabs(["🌍 Worldwide Map", "🔮 Interactive Predictor"])
    
    with tab1:
        render_map_tab()
    
    with tab2:
        render_predictor_tab()


if __name__ == '__main__':
    main()
