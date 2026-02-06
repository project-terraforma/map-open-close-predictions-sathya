"""
Feature Engineering Pipeline for Open/Closed Place Prediction

Extracts meaningful features from Overture Maps place data to predict
whether a place is open (1) or closed (0).

Designed for scalability to 100M+ places with efficient batch processing.
"""

import pandas as pd
import numpy as np
from typing import Optional


def is_valid_array(val):
    """Check if value is a valid list-like with content."""
    if val is None:
        return False
    if isinstance(val, np.ndarray):
        return len(val) > 0
    if isinstance(val, (list, tuple)):
        return len(val) > 0
    return False


def to_list(val):
    """Convert numpy array or list-like to list."""
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return list(val)
    return []


def extract_source_features(sources) -> dict:
    """Extract features from the sources column."""
    if not is_valid_array(sources):
        return {
            'num_sources': 0,
            'has_meta_source': False,
            'has_facebook_source': False,
        }
    
    sources_list = to_list(sources)
    num_sources = len(sources_list)
    source_datasets = [s.get('dataset', '').lower() for s in sources_list if isinstance(s, dict)]
    
    return {
        'num_sources': num_sources,
        'has_meta_source': any('meta' in ds or 'facebook' in ds for ds in source_datasets),
        'has_facebook_source': any('facebook' in ds for ds in source_datasets),
    }


def extract_contact_features(row: pd.Series) -> dict:
    """Extract presence flags for contact information."""
    websites = row.get('websites')
    phones = row.get('phones')
    socials = row.get('socials')
    emails = row.get('emails')
    
    has_website = is_valid_array(websites)
    has_phone = is_valid_array(phones)
    has_socials = is_valid_array(socials)
    has_email = is_valid_array(emails)
    
    return {
        'has_website': has_website,
        'has_phone': has_phone,
        'has_socials': has_socials,
        'has_email': has_email,
        'num_websites': len(to_list(websites)),
        'num_phones': len(to_list(phones)),
        'num_socials': len(to_list(socials)),
        'contact_completeness': sum([has_website, has_phone, has_socials, has_email]) / 4.0,
    }


def extract_brand_features(brand) -> dict:
    """Extract features from the brand column."""
    if not brand or not isinstance(brand, dict):
        return {
            'has_brand': False,
            'has_brand_wikidata': False,
        }
    
    return {
        'has_brand': True,
        'has_brand_wikidata': bool(brand.get('wikidata')),
    }


def extract_category_features(categories) -> dict:
    """Extract features from the categories column."""
    if categories is None or not isinstance(categories, dict):
        return {
            'primary_category': 'unknown',
            'num_alternate_categories': 0,
            'has_alternate_categories': False,
        }
    
    primary = categories.get('primary', 'unknown')
    alternate = categories.get('alternate', [])
    alternate_list = to_list(alternate) if is_valid_array(alternate) else []
    
    return {
        'primary_category': primary if primary else 'unknown',
        'num_alternate_categories': len(alternate_list),
        'has_alternate_categories': len(alternate_list) > 0,
    }


def extract_address_features(addresses) -> dict:
    """Extract features from the addresses column."""
    if not is_valid_array(addresses):
        return {
            'has_address': False,
            'address_completeness': 0.0,
            'has_postal_code': False,
            'has_country': False,
        }
    
    # Take first address
    addresses_list = to_list(addresses)
    addr = addresses_list[0] if isinstance(addresses_list[0], dict) else {}
    
    # Count filled address fields
    address_fields = ['freeform', 'locality', 'postcode', 'region', 'country']
    filled_fields = sum(1 for f in address_fields if addr.get(f))
    
    return {
        'has_address': True,
        'address_completeness': filled_fields / len(address_fields),
        'has_postal_code': bool(addr.get('postcode')),
        'has_country': bool(addr.get('country')),
    }


def extract_name_features(names) -> dict:
    """Extract features from the names column."""
    if not names or not isinstance(names, dict):
        return {
            'has_name': False,
            'num_name_variants': 0,
        }
    
    primary = names.get('primary')
    common = names.get('common', {})
    
    num_variants = 0
    if isinstance(common, dict):
        num_variants = len(common)
    
    return {
        'has_name': bool(primary),
        'num_name_variants': num_variants,
    }


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all features from a DataFrame of place records.
    
    Args:
        df: DataFrame with columns from Overture Maps place schema
        
    Returns:
        DataFrame with extracted features ready for ML model
    """
    features = []
    
    for idx, row in df.iterrows():
        feat = {'id': row['id']}
        
        # Source features
        feat.update(extract_source_features(row.get('sources')))
        
        # Contact features
        feat.update(extract_contact_features(row))
        
        # Brand features
        feat.update(extract_brand_features(row.get('brand')))
        
        # Category features
        feat.update(extract_category_features(row.get('categories')))
        
        # Address features
        feat.update(extract_address_features(row.get('addresses')))
        
        # Name features
        feat.update(extract_name_features(row.get('names')))
        
        # Direct numeric features
        feat['confidence'] = row.get('confidence', 0.0)
        feat['version'] = row.get('version', 0)
        
        features.append(feat)
    
    return pd.DataFrame(features)


def extract_features_batch(df: pd.DataFrame, batch_size: int = 10000) -> pd.DataFrame:
    """
    Extract features in batches for memory-efficient processing of large datasets.
    
    Designed for 100M+ scale processing.
    
    Args:
        df: Input DataFrame
        batch_size: Number of rows to process at once
        
    Returns:
        DataFrame with all extracted features
    """
    all_features = []
    
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        batch_features = extract_features(batch_df)
        all_features.append(batch_features)
        
        if start_idx % (batch_size * 10) == 0:
            print(f"Processed {end_idx:,} / {len(df):,} rows")
    
    return pd.concat(all_features, ignore_index=True)


def get_feature_columns() -> list:
    """Return list of numeric feature columns used by the model."""
    return [
        # Source features
        'num_sources',
        'has_meta_source',
        'has_facebook_source',
        
        # Contact features
        'has_website',
        'has_phone',
        'has_socials',
        'has_email',
        'num_websites',
        'num_phones',
        'num_socials',
        'contact_completeness',
        
        # Brand features
        'has_brand',
        'has_brand_wikidata',
        
        # Category features (primary_category needs encoding)
        'num_alternate_categories',
        'has_alternate_categories',
        
        # Address features
        'has_address',
        'address_completeness',
        'has_postal_code',
        'has_country',
        
        # Name features
        'has_name',
        'num_name_variants',
        
        # Direct features
        'confidence',
        'version',
    ]


if __name__ == '__main__':
    # Test feature extraction
    import pandas as pd
    
    df = pd.read_parquet('project_c_samples.parquet')
    print(f"Loaded {len(df)} rows")
    
    features_df = extract_features(df)
    print(f"\nExtracted {len(features_df.columns)} features:")
    print(features_df.columns.tolist())
    print(f"\nSample features:")
    print(features_df.head())
