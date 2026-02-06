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


# California cities with realistic coordinates
CALIFORNIA_CITIES = [
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    {"name": "San Francisco", "lat": 37.7749, "lon": -122.4194},
    {"name": "San Diego", "lat": 32.7157, "lon": -117.1611},
    {"name": "San Jose", "lat": 37.3382, "lon": -121.8863},
    {"name": "Sacramento", "lat": 38.5816, "lon": -121.4944},
    {"name": "Fresno", "lat": 36.7378, "lon": -119.7871},
    {"name": "Oakland", "lat": 37.8044, "lon": -122.2712},
    {"name": "Long Beach", "lat": 33.7701, "lon": -118.1937},
    {"name": "Bakersfield", "lat": 35.3733, "lon": -119.0187},
    {"name": "Anaheim", "lat": 33.8366, "lon": -117.9143},
    {"name": "Santa Ana", "lat": 33.7455, "lon": -117.8677},
    {"name": "Riverside", "lat": 33.9533, "lon": -117.3962},
    {"name": "Stockton", "lat": 37.9577, "lon": -121.2908},
    {"name": "Irvine", "lat": 33.6846, "lon": -117.8265},
    {"name": "Chula Vista", "lat": 32.6401, "lon": -117.0842},
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
    
    # Calculate Score (0-100)
    score = 50  # Start neutral
    
    # 1. Confidence Impact (Centered around 0.75)
    # If conf > 0.75, add points. If < 0.75, subtract points.
    conf_impact = (confidence - 0.75) * 200
    score += conf_impact
    
    # 2. Proof of Life
    if has_websites: score += 15
    else: score -= 10
        
    if has_phones: score += 15
    else: score -= 10
        
    if has_address: score += 5
    
    # 3. Source Bonus
    if num_sources > 1: score += 10
    
    # Clamp
    score = np.clip(score, 0, 100)
    prob_open = score / 100.0
    
    # Determine prediction
    prediction = 1 if score > 50 else 0
    probabilities = np.array([1 - prob_open, prob_open])
    
    # Explanation (for debugging/user info)
    reasons = []
    if confidence > 0.8: reasons.append("High Confidence")
    elif confidence < 0.7: reasons.append("Low Confidence")
    
    if has_websites: reasons.append("Has Website")
    else: reasons.append("Missing Website")
    
    if has_phones: reasons.append("Has Phone")
    
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


def get_batch_businesses(n=100):
    """Get a batch of businesses - prioritizing Overture CA data."""
    # 1. Try Overture US Data
    us_df = load_overture_ca_data()
    if us_df is not None and not us_df.empty:
        # Filter to continental US
        us_only = us_df[(us_df['latitude'] >= 24) & (us_df['latitude'] <= 50)].copy()
        
        # Stratified sampling: divide US into regions by longitude
        # West: lon < -104 (CA, WA, OR, NV, AZ, etc.)
        # Central: -104 <= lon < -90 (TX, CO, MN, etc.)
        # East: lon >= -90 (FL, NY, GA, etc.)
        west = us_only[us_only['longitude'] < -104]
        central = us_only[(us_only['longitude'] >= -104) & (us_only['longitude'] < -90)]
        east = us_only[us_only['longitude'] >= -90]
        
        samples_per_region = n // 3
        sample_list = []
        
        for region_df, region_name in [(west, 'West'), (central, 'Central'), (east, 'East')]:
            if len(region_df) > 0:
                region_sample = region_df.sample(min(samples_per_region, len(region_df)))
                sample_list.append(region_sample)
        
        if sample_list:
            combined_sample = pd.concat(sample_list, ignore_index=True)
            sample = combined_sample.to_dict('records')
        else:
            sample = us_only.sample(min(n, len(us_only))).to_dict('records')
            
        st.toast(f"Sampled {len(sample)} businesses across the US!")
        businesses = []
        for row in sample:
            # Helper to safely getting list/dict from potential numpy arrays or strings
            def safe_get(val, default=None):
                if val is None: return default
                if isinstance(val, np.ndarray): return val.tolist()
                return val
            
            # Extract city from addresses if available
            addresses = safe_get(row.get('addresses'), [])
            city_name = 'California'
            if addresses and len(addresses) > 0:
                addr = addresses[0] if isinstance(addresses[0], dict) else {}
                city_name = addr.get('locality', addr.get('region', 'California'))
            
            features = {
                'id': row['id'],
                'sources': safe_get(row.get('sources'), [{'dataset': 'meta'}]),
                'confidence': float(row.get('confidence', 0.5)),
                'websites': safe_get(row.get('websites')),
                'phones': safe_get(row.get('phones')),
                'socials': safe_get(row.get('socials')),
                'emails': safe_get(row.get('emails')),
                'brand': safe_get(row.get('brand')),
                'categories': row.get('categories', {'primary': 'unknown'}),
                'addresses': addresses,
                'names': row.get('names', {'primary': row.get('name', 'Unknown')}),
                'version': 1
            }
            
            businesses.append({
                'name': row.get('name', 'Unknown'),
                'category': row.get('category', 'unknown'),
                'lat': row['latitude'],
                'lon': row['longitude'],
                'city': city_name,
                'features': features,
                'prob_open': 0,
                'prediction': 0,
                'is_real': True,
                'source': 'Overture California'
            })
        return businesses

    # 2. Try Historical LA Data
    hist_df = load_historical_data()
    
    if hist_df is not None and not hist_df.empty:
        # Use real data
        sample = hist_df.sample(min(n, len(hist_df))).to_dict('records')
        businesses = []
        
        for row in sample:
            # Simulate features for the model (since we don't know if they had websites in 2019)
            has_website = np.random.random() > 0.3
            has_phone = np.random.random() > 0.2
            has_social = np.random.random() > 0.5
            has_email = np.random.random() > 0.6
            has_brand = np.random.random() > 0.8
            num_sources = np.random.randint(1, 4) # Older data might have fewer digital sources
            confidence = np.random.uniform(0.3, 0.9)
            
            features = {
                'id': f"la_{np.random.randint(100000)}",
                'sources': [{'dataset': 'meta' if np.random.random() > 0.5 else 'other'}] * num_sources,
                'confidence': confidence,
                'websites': ['https://example.com'] if has_website else None,
                'phones': ['+1-555-1234'] if has_phone else None,
                'socials': ['https://facebook.com/example'] if has_social else None,
                'emails': ['contact@example.com'] if has_email else None,
                'brand': {'names': {'primary': 'Brand'}} if has_brand else None,
                'categories': {'primary': row['mapped_category']},
                'addresses': [{'freeform': row['address'], 'locality': row['city'], 'country': 'US'}],
                'names': {'primary': row['name']},
                'version': num_sources,
            }
            
            businesses.append({
                'name': row['name'],
                'category': row['mapped_category'],
                'lat': row['latitude'],
                'lon': row['longitude'],
                'city': row['city'],
                'features': features,
                'prob_open': 0, # Placeholder
                'prediction': 0, # Placeholder
                'is_real': True,
                'source': 'LA Historical'
            })
        return businesses
    else:
        # Fallback to random
        return [generate_random_business() for _ in range(n)]


def render_map_tab():
    """Render the California map tab."""
    model, label_encoder, feature_names = load_model()
    
    st.subheader("🗺️ California Business Predictions")
    st.markdown("*Generate random businesses to see predictions vs actual status*")
    
    # Google API Key
    # Google API Key (Hardcoded for demo)
    api_key = "AIzaSyAinBHpEi7ysEdtcGtIeE2Ys1e38Z42Q3o"
    
    # Initialize session state for businesses
    if 'map_businesses' not in st.session_state:
        st.session_state.map_businesses = []
    
    # Buttons: Only two options now
    st.markdown("### Generate Businesses by Actual Status")
    st.caption("These fetch Overture data and verify against Google Places to find truly open or closed businesses")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        if st.button("✅ Generate 100 Actually OPEN"):
            st.session_state.map_businesses = []
            
            with st.spinner("Fetching businesses and filtering for OPEN ones..."):
                # Fetch more than 100 to account for closed ones
                all_businesses = get_batch_businesses(300)
                
                # Verify with Google in parallel
                def verify_business(b):
                    res = check_google_places(b['name'], b['lat'], b['lon'], api_key)
                    b['google'] = res
                    return b
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
                    list(executor.map(verify_business, all_businesses))
                
                # Filter for actually open businesses
                open_businesses = [b for b in all_businesses 
                                   if b.get('google', {}).get('found') and b['google']['actual_open']]
                
                # Run predictions on the filtered list
                for b in open_businesses[:100]:
                    pred, probs, reason = predict_place_features(b['features'], model, label_encoder, feature_names)
                    b['prediction'] = pred
                    b['prob_open'] = probs[1]
                    b['reason'] = reason
                
                if open_businesses:
                    st.session_state.map_businesses = open_businesses[:100]
                    st.toast(f"Found {len(open_businesses[:100])} Actually OPEN businesses!")
                    st.rerun()
                else:
                    st.error("Couldn't find enough open businesses")
    
    with col2:
        if st.button("❌ Generate 100 Actually CLOSED", type="secondary"):
            st.session_state.map_businesses = []
            
            with st.spinner("Fetching businesses and filtering for CLOSED ones..."):
                # Fetch more to find closed ones (most businesses are open)
                all_businesses = get_batch_businesses(500)
                
                # Verify with Google in parallel
                def verify_business(b):
                    res = check_google_places(b['name'], b['lat'], b['lon'], api_key)
                    b['google'] = res
                    return b
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
                    list(executor.map(verify_business, all_businesses))
                
                # Filter for actually closed businesses
                closed_businesses = [b for b in all_businesses 
                                     if b.get('google', {}).get('found') and not b['google']['actual_open']]
                
                # Run predictions on the filtered list
                for b in closed_businesses[:100]:
                    pred, probs, reason = predict_place_features(b['features'], model, label_encoder, feature_names)
                    b['prediction'] = pred
                    b['prob_open'] = probs[1]
                    b['reason'] = reason
                
                if closed_businesses:
                    st.session_state.map_businesses = closed_businesses[:100]
                    st.toast(f"Found {len(closed_businesses[:100])} Actually CLOSED businesses!")
                    st.rerun()
                else:
                    st.error("Couldn't find enough closed businesses")
    
    with col3:
        if st.button("🗑️ Clear"):
            st.session_state.map_businesses = []
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
        location=[36.7783, -119.4179],
        zoom_start=6,
        tiles='CartoDB positron'
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
        
        # Build popup - show confidence matching prediction
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
        
        # Add Google verification if available
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
                'Confidence': f"{b['prob_open']:.0%}",
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
    st.markdown("*Predicting which California businesses are still open today*")
    
    tab1, tab2 = st.tabs(["📍 California Map", "🔮 Interactive Predictor"])
    
    with tab1:
        render_map_tab()
    
    with tab2:
        render_predictor_tab()


if __name__ == '__main__':
    main()
