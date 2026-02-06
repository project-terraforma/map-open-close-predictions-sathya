"""
Create a simple balanced heuristic model that doesn't heavily favor one class.
"""

import pickle
import json
import numpy as np
import os


class BalancedHeuristicModel:
    """A simple model that gives more balanced predictions based on confidence."""
    
    def __init__(self):
        self.classes_ = np.array([0, 1])
    
    def predict_proba(self, X):
        """Return probabilities based on confidence and other features."""
        probs = []
        for _, row in X.iterrows():
            # Base probability on confidence (higher confidence = more likely open)
            confidence = row.get('confidence', 0.5)
            num_sources = row.get('num_sources', 1)
            has_websites = row.get('has_websites', 0)
            has_phones = row.get('has_phones', 0)
            has_address = row.get('has_address', 0)
            
            # Start with confidence as base probability of being open
            prob_open = float(confidence) * 0.6  # Scale confidence
            
            # Add bonuses for having more data
            prob_open += 0.1 if has_websites else 0
            prob_open += 0.1 if has_phones else 0
            prob_open += 0.05 if has_address else 0
            prob_open += 0.05 * min(num_sources, 3) / 3  # Bonus for multiple sources
            
            # Clamp to valid range with some margin
            prob_open = np.clip(prob_open, 0.15, 0.90)
            
            probs.append([1 - prob_open, prob_open])
        
        return np.array(probs)
    
    def predict(self, X):
        """Return predictions (0 or 1)."""
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)


def main():
    print("Creating balanced heuristic model...")
    
    model = BalancedHeuristicModel()
    
    # Save model
    os.makedirs('model', exist_ok=True)
    
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Keep existing label encoder and feature names
    feature_names = [
        'confidence', 'num_sources', 'has_websites', 'has_phones',
        'has_socials', 'has_emails', 'has_brand', 'has_address',
        'has_locality', 'has_postcode', 'name_length', 'has_common',
        'primary_category_encoded'
    ]
    
    with open('model/feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    
    print("✅ Balanced heuristic model saved!")
    
    # Test
    import pandas as pd
    test_high = pd.DataFrame([{
        'confidence': 0.95, 'num_sources': 3, 'has_websites': 1, 'has_phones': 1,
        'has_socials': 0, 'has_emails': 0, 'has_brand': 0, 'has_address': 1,
        'has_locality': 1, 'has_postcode': 1, 'name_length': 15, 'has_common': 0,
        'primary_category_encoded': 0
    }])
    
    test_low = pd.DataFrame([{
        'confidence': 0.3, 'num_sources': 1, 'has_websites': 0, 'has_phones': 0,
        'has_socials': 0, 'has_emails': 0, 'has_brand': 0, 'has_address': 0,
        'has_locality': 0, 'has_postcode': 0, 'name_length': 5, 'has_common': 0,
        'primary_category_encoded': 0
    }])
    
    print("\nTest - High confidence business:")
    print(f"  Prediction: {model.predict(test_high)[0]}")
    print(f"  Probabilities: {model.predict_proba(test_high)[0]}")
    
    print("\nTest - Low confidence business:")
    print(f"  Prediction: {model.predict(test_low)[0]}")
    print(f"  Probabilities: {model.predict_proba(test_low)[0]}")


if __name__ == '__main__':
    main()
