"""
Check website liveness for businesses.
Extracts URLs from Overture parquet and tests if they're alive.

Website states:
  - alive:    HTTP 200, real content
  - dead:     Connection refused, DNS failure, timeout
  - parked:   Domain for sale / expired domain parking page
  - redirect: Redirects to unrelated domain

Usage: python scripts/check_website_liveness.py
"""

import json
import os
import sys
import re
import time
import requests
from urllib.parse import urlparse

import pandas as pd

PARQUET_PATH = os.path.join(os.path.dirname(__file__), '..', 'pipeline', 'data', 'sf_places_large.parquet')
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'test_data.json')
TRAIN_DATA_PATH = os.path.join(os.path.dirname(__file__), 'yelp_training_data.json')

# Known domain parking / expired domain indicators
PARKED_INDICATORS = [
    'domain is for sale', 'buy this domain', 'domain parking',
    'this domain', 'domain has expired', 'expired domain',
    'godaddy', 'afternic', 'sedo.com', 'hugedomains',
    'parked free', 'dan.com', 'undeveloped.com',
    'this site can\'t be reached', 'domain not found',
    'this webpage is not available',
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}


def extract_url_from_parquet(business_name, location, df):
    """Find a business in the parquet and extract its website URL and phone."""
    # Match by name
    name_lower = business_name.lower()
    matches = df[df['names'].apply(
        lambda x: name_lower in str(x).lower() if x is not None else False
    )]

    if len(matches) == 0:
        return None, None

    # If multiple matches, pick closest by location
    if len(matches) > 1 and location:
        lng, lat = location
        def dist(row):
            try:
                geom = row.get('geometry')
                if hasattr(geom, 'x'):
                    return ((geom.x - lng)**2 + (geom.y - lat)**2)
            except:
                pass
            return 999
        matches = matches.copy()
        matches['_dist'] = matches.apply(dist, axis=1)
        matches = matches.sort_values('_dist')

    row = matches.iloc[0]
    websites = row.get('websites')
    phones = row.get('phones')

    url = None
    phone = None
    if websites is not None and len(websites) > 0:
        url = websites[0]
    if phones is not None and len(phones) > 0:
        phone = phones[0]

    return url, phone


def check_website(url, timeout=10):
    """Check if a website URL is alive, dead, or parked.
    Returns: (status, status_code, detail)
    """
    if not url:
        return 'no_url', None, 'No URL available'

    # Normalize URL
    if not url.startswith('http'):
        url = 'http://' + url

    try:
        resp = requests.get(url, timeout=timeout, headers=HEADERS, allow_redirects=True)
        status_code = resp.status_code

        if status_code >= 500:
            return 'error', status_code, f'Server error {status_code}'

        if status_code == 404:
            return 'dead', status_code, 'Page not found (404)'

        if status_code >= 400:
            return 'dead', status_code, f'Client error {status_code}'

        # Check for parked/expired domain
        body = resp.text[:5000].lower()
        for indicator in PARKED_INDICATORS:
            if indicator in body:
                return 'parked', status_code, f'Domain appears parked: "{indicator}"'

        # Check if redirected to a totally different domain
        final_domain = urlparse(resp.url).netloc.lower()
        orig_domain = urlparse(url).netloc.lower()
        # Strip www
        final_domain = final_domain.replace('www.', '')
        orig_domain = orig_domain.replace('www.', '')
        if final_domain != orig_domain:
            # Check if it's a social media redirect (Instagram, Facebook etc — still alive)
            social_domains = ['instagram.com', 'facebook.com', 'yelp.com', 'google.com']
            if any(s in final_domain for s in social_domains):
                return 'alive', status_code, f'Redirects to social media ({final_domain})'
            return 'redirect', status_code, f'Redirects to different domain: {final_domain}'

        return 'alive', status_code, f'Website responding (HTTP {status_code})'

    except requests.exceptions.SSLError:
        # SSL errors often mean the domain exists but cert is expired
        try:
            resp = requests.get(url.replace('https://', 'http://'), timeout=timeout,
                              headers=HEADERS, allow_redirects=True, verify=False)
            if resp.status_code < 400:
                return 'alive', resp.status_code, 'SSL error but HTTP works'
        except:
            pass
        return 'ssl_error', None, 'SSL certificate error'

    except requests.exceptions.ConnectionError as e:
        err = str(e).lower()
        if 'name or service not known' in err or 'nodename nor servname' in err or 'getaddrinfo failed' in err:
            return 'dead', None, 'DNS resolution failed — domain does not exist'
        return 'dead', None, f'Connection refused'

    except requests.exceptions.Timeout:
        return 'dead', None, 'Connection timed out'

    except Exception as e:
        return 'dead', None, f'Error: {str(e)[:80]}'


def main():
    print("Loading parquet data...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  {len(df)} places loaded")

    print("\nLoading test data...")
    with open(TEST_DATA_PATH, encoding='utf-8') as f:
        test_data = json.load(f)

    print(f"\n{'='*70}")
    print("CHECKING TEST DATA WEBSITES")
    print(f"{'='*70}")

    results = []
    for i, biz in enumerate(test_data):
        name = biz['name']
        location = biz.get('location')
        gt = biz.get('ground_truth', '?')

        url, phone = extract_url_from_parquet(name, location, df)

        print(f"\n[{i+1}/{len(test_data)}] {name}")
        print(f"  URL: {url or 'none'}")
        print(f"  Phone: {phone or 'none'}")

        if url:
            status, code, detail = check_website(url)
            print(f"  Status: {status} ({detail})")
        else:
            status, code, detail = 'no_url', None, 'No URL in Overture'

        results.append({
            'name': name,
            'url': url,
            'phone': phone,
            'website_status': status,
            'status_code': code,
            'detail': detail,
            'ground_truth': gt,
        })

        # Rate limit
        if url:
            time.sleep(0.5)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    statuses = {}
    for r in results:
        s = r['website_status']
        statuses[s] = statuses.get(s, 0) + 1
    for s, count in sorted(statuses.items(), key=lambda x: -x[1]):
        print(f"  {s:15s}: {count}")

    # Cross-reference with ground truth
    print(f"\n  Website status vs ground truth:")
    for status in ['alive', 'dead', 'parked', 'redirect', 'ssl_error', 'error', 'no_url']:
        open_count = sum(1 for r in results if r['website_status'] == status and r['ground_truth'] == 'open')
        closed_count = sum(1 for r in results if r['website_status'] == status and r['ground_truth'] == 'closed')
        total = open_count + closed_count
        if total > 0:
            print(f"    {status:15s}: {open_count} open, {closed_count} closed (closed rate: {closed_count/total:.0%})")

    # Update test data with website info
    for biz, result in zip(test_data, results):
        biz['website_url'] = result['url']
        biz['phone_number'] = result['phone']
        biz['website_check'] = {
            'status': result['website_status'],
            'status_code': result['status_code'],
            'detail': result['detail'],
        }

    with open(TEST_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Updated {TEST_DATA_PATH}")


if __name__ == '__main__':
    main()
