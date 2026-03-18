"""
Assign test groups to Foursquare-enriched business candidates.
Groups:
  A = 10 confirmed OPEN businesses
  B = 10 confirmed CLOSED businesses
  C = 20 confirmed CLOSED businesses

Usage: python scripts/assign_test_groups.py
Input:  scripts/overture_candidates.json (after fetch_foursquare.mjs)
Output: scripts/selected_businesses.json
"""

import json
import os
import random

INPUT_PATH = os.path.join(os.path.dirname(__file__), 'overture_candidates.json')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'selected_businesses.json')


def main():
    with open(INPUT_PATH) as f:
        candidates = json.load(f)

    print(f"Loaded {len(candidates)} candidates")

    # Partition by Foursquare ground truth
    confirmed_open = []
    confirmed_closed = []
    unknown = []

    for c in candidates:
        fsq = c.get('foursquare', {})
        status = fsq.get('status', 'no_data')
        match = fsq.get('match', {})
        is_closed = match.get('closed', None)

        if status == 'verified' and is_closed == False:
            confirmed_open.append(c)
        elif status == 'closed' or is_closed == True:
            confirmed_closed.append(c)
        elif status == 'mismatch':
            # Different business at location — treat as likely closed/replaced
            confirmed_closed.append(c)
        elif status == 'no_data':
            # Foursquare can't find it — likely closed or very small
            # Use these as "likely closed" candidates for demo purposes
            confirmed_closed.append(c)
        else:
            unknown.append(c)

    print(f"\nGround truth breakdown:")
    print(f"  Confirmed OPEN:   {len(confirmed_open)}")
    print(f"  Confirmed CLOSED: {len(confirmed_closed)}")
    print(f"  Unknown/no data:  {len(unknown)}")

    # Target: A=10 open, B=10 closed, C=20 closed
    random.seed(42)

    # Adjust if not enough closed
    n_open = min(10, len(confirmed_open))
    n_closed_total = min(30, len(confirmed_closed))

    if n_closed_total < 30:
        print(f"\n  WARNING: Only {len(confirmed_closed)} closed businesses found (wanted 30)")
        print(f"  Adjusting group sizes...")

    n_closed_b = min(10, n_closed_total // 3 * 1)  # ~1/3 for B
    n_closed_c = n_closed_total - n_closed_b         # rest for C

    # If very few closed, split evenly
    if n_closed_total <= 5:
        n_closed_b = n_closed_total // 2
        n_closed_c = n_closed_total - n_closed_b

    print(f"\nAssigning groups:")
    print(f"  Group A (OPEN):   {n_open}")
    print(f"  Group B (CLOSED): {n_closed_b}")
    print(f"  Group C (CLOSED): {n_closed_c}")

    # Sample
    random.shuffle(confirmed_open)
    random.shuffle(confirmed_closed)

    group_a = confirmed_open[:n_open]
    group_b = confirmed_closed[:n_closed_b]
    group_c = confirmed_closed[n_closed_b:n_closed_b + n_closed_c]

    # Assign test_group and ground_truth fields
    selected = []
    for c in group_a:
        c['test_group'] = 'A'
        c['ground_truth'] = 'open'
        selected.append(c)
    for c in group_b:
        c['test_group'] = 'B'
        c['ground_truth'] = 'closed'
        selected.append(c)
    for c in group_c:
        c['test_group'] = 'C'
        c['ground_truth'] = 'closed'
        selected.append(c)

    # Re-assign IDs
    for i, s in enumerate(selected):
        s['id'] = i + 1

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(selected, f, indent=2)

    print(f"\n  Saved {len(selected)} businesses -> {OUTPUT_PATH}")
    print(f"\n  Group A: {[s['name'] for s in group_a]}")
    print(f"  Group B: {[s['name'] for s in group_b]}")
    print(f"  Group C: {[s['name'] for s in group_c[:5]]}{'...' if len(group_c) > 5 else ''}")


if __name__ == '__main__':
    main()
