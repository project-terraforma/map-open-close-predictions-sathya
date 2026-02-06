# Test that logic makes sense - more features = more likely open
import pickle
import json
import pandas as pd
from feature_engineering import extract_features

model = pickle.load(open('model/model.pkl', 'rb'))
le = pickle.load(open('model/label_encoder.pkl', 'rb'))
fn = json.load(open('model/feature_names.json'))

def test_place(name, **kwargs):
    """Test a place configuration."""
    base = {
        'id': 'test',
        'sources': kwargs.get('sources', [{'dataset': 'meta'}]),
        'confidence': kwargs.get('confidence', 0.5),
        'websites': kwargs.get('websites', None),
        'phones': kwargs.get('phones', None),
        'socials': kwargs.get('socials', None),
        'emails': kwargs.get('emails', None),
        'brand': kwargs.get('brand', None),
        'categories': {'primary': 'grocery_store'},
        'addresses': kwargs.get('addresses', []),
        'names': {'primary': 'Test'},
        'version': 1,
    }
    
    df = pd.DataFrame([base])
    f = extract_features(df)
    f['primary_category_encoded'] = le.transform(['grocery_store'])[0]
    for col in fn:
        if col in f.columns and f[col].dtype == bool:
            f[col] = f[col].astype(int)
    X = f[fn].fillna(0)
    probs = model.predict_proba(X)[0]
    pred = "OPEN" if probs[1] > 0.5 else "CLOSED"
    print(f"{name}: {pred} ({probs[1]:.0%} open)")

print("Testing logical consistency:")
print("-" * 40)

# Minimal place - should be more likely closed
test_place("Minimal (nothing)", 
           sources=[{'dataset': 'other'}],
           confidence=0.3)

# Add website
test_place("+ Website",
           sources=[{'dataset': 'other'}],
           confidence=0.3,
           websites=['https://example.com'])

# Add phone
test_place("+ Website + Phone",
           sources=[{'dataset': 'other'}],
           confidence=0.3,
           websites=['https://example.com'],
           phones=['+1-555-1234'])

# Add social
test_place("+ Website + Phone + Social",
           sources=[{'dataset': 'other'}],
           confidence=0.3,
           websites=['https://example.com'],
           phones=['+1-555-1234'],
           socials=['https://fb.com'])

# Add email
test_place("+ All contact info",
           sources=[{'dataset': 'other'}],
           confidence=0.3,
           websites=['https://example.com'],
           phones=['+1-555-1234'],
           socials=['https://fb.com'],
           emails=['test@example.com'])

# Full package
test_place("MAXED OUT",
           sources=[{'dataset': 'meta'}] * 5,
           confidence=0.98,
           websites=['https://example.com'],
           phones=['+1-555-1234'],
           socials=['https://fb.com'],
           emails=['test@example.com'],
           brand={'names': {'primary': 'Safeway'}},
           addresses=[{'freeform': '123 Main', 'locality': 'LA'}])
