# Test grocery store prediction
import pickle
import json
import pandas as pd
from feature_engineering import extract_features

model = pickle.load(open('model/model.pkl', 'rb'))
le = pickle.load(open('model/label_encoder.pkl', 'rb'))
fn = json.load(open('model/feature_names.json'))

place = {
    'id': 'test',
    'sources': [{'dataset': 'meta'}] * 5,
    'confidence': 0.98,
    'websites': ['https://example.com'],
    'phones': ['+1-555-1234'],
    'socials': ['https://fb.com'],
    'emails': ['test@example.com'],
    'brand': {'names': {'primary': 'Safeway'}},
    'categories': {'primary': 'grocery_store'},
    'addresses': [{'freeform': '123 Main', 'locality': 'LA', 'country': 'US'}],
    'names': {'primary': 'Safeway'},
    'version': 5,
}

df = pd.DataFrame([place])
f = extract_features(df)

# Check if grocery_store is in label encoder
print(f"grocery_store in classes: {'grocery_store' in le.classes_}")
print(f"Sample classes: {list(le.classes_[:10])}")

if 'grocery_store' in le.classes_:
    f['primary_category_encoded'] = le.transform(['grocery_store'])[0]
else:
    print("grocery_store NOT found, using fallback")
    f['primary_category_encoded'] = 0

for col in fn:
    if col in f.columns and f[col].dtype == bool:
        f[col] = f[col].astype(int)

X = f[fn].fillna(0)
pred = model.predict(X)[0]
probs = model.predict_proba(X)[0]

print(f"\nGrocery Store maxed:")
print(f"  Prediction: {'OPEN' if pred == 1 else 'CLOSED'}")
print(f"  Prob Open: {probs[1]:.1%}")
print(f"  Prob Closed: {probs[0]:.1%}")
