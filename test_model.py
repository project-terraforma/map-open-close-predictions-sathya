# Test new model predictions
import pickle
import json
import pandas as pd
from feature_engineering import extract_features

model = pickle.load(open('model/model.pkl', 'rb'))
le = pickle.load(open('model/label_encoder.pkl', 'rb'))
feature_names = json.load(open('model/feature_names.json'))

# Test maxed out place
place = {
    'id': 'test',
    'sources': [{'dataset': 'meta'}] * 5,
    'confidence': 0.98,
    'websites': ['https://example.com'],
    'phones': ['+1-555-1234'],
    'socials': ['https://fb.com'],
    'emails': ['test@example.com'],
    'brand': {'names': {'primary': 'Starbucks'}},
    'categories': {'primary': 'restaurant'},
    'addresses': [{'freeform': '123 Main St', 'locality': 'LA', 'country': 'US'}],
    'names': {'primary': 'Test Restaurant'},
    'version': 5,
}

df = pd.DataFrame([place])
features = extract_features(df)

# Encode category
if 'restaurant' in le.classes_:
    features['primary_category_encoded'] = le.transform(['restaurant'])[0]
else:
    features['primary_category_encoded'] = 0

for col in feature_names:
    if col in features.columns and features[col].dtype == bool:
        features[col] = features[col].astype(int)

X = features[feature_names].fillna(0)
pred = model.predict(X)[0]
probs = model.predict_proba(X)[0]

print(f"\nMaxed-out Restaurant:")
print(f"  Prediction: {'OPEN' if pred == 1 else 'CLOSED'}")
print(f"  Prob Open: {probs[1]:.1%}")
print(f"  Prob Closed: {probs[0]:.1%}")
