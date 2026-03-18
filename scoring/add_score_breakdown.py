"""
Add score_breakdown to existing test data by replaying the additive scoring logic
from the signals already stored in each entry's vision.layers.

Usage: python scripts/add_score_breakdown.py [--input path]
Default: processes all 3 city test data files.
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(__file__)
DEFAULT_FILES = [
    os.path.join(SCRIPT_DIR, '..', 'src', 'data', 'test_data.json'),
    os.path.join(SCRIPT_DIR, '..', 'src', 'data', 'test_data_la.json'),
    os.path.join(SCRIPT_DIR, '..', 'src', 'data', 'test_data_chicago.json'),
]

# Parse --input arg
input_files = []
for i, arg in enumerate(sys.argv):
    if arg == '--input' and i + 1 < len(sys.argv):
        input_files.append(sys.argv[i + 1])

if not input_files:
    input_files = DEFAULT_FILES


def compute_score_breakdown(entry):
    """Replay the additive scoring logic from stored signals."""
    v = entry.get('vision', {})
    if not v:
        return None

    layers = v.get('layers', {})
    breakdown = [{"signal": "base", "weight": 0.50, "description": "Base score"}]
    open_score = 0.50

    # Foursquare
    fsq = layers.get('foursquare', {})
    fsq_status = fsq.get('status', 'no_data') if fsq else 'no_data'

    if fsq_status == 'verified':
        open_score += 0.30
        breakdown.append({"signal": "foursquare", "weight": 0.30, "description": "Foursquare: verified"})
    elif fsq_status == 'closed':
        open_score -= 0.35
        breakdown.append({"signal": "foursquare", "weight": -0.35, "description": "Foursquare: closed"})
    elif fsq_status == 'mismatch':
        fsq_match = fsq.get('match', {})
        other_name = fsq_match.get('name', '?') if fsq_match else '?'
        open_score -= 0.25
        breakdown.append({"signal": "foursquare", "weight": -0.25, "description": f"Foursquare: mismatch (\"{other_name}\")"})

    # Google
    google = layers.get('google', {})
    biz_status = google.get('business_status') if google else None

    if biz_status == 'OPERATIONAL':
        open_score += 0.20
        breakdown.append({"signal": "google", "weight": 0.20, "description": "Google: operational"})
    elif biz_status in ('CLOSED_PERMANENTLY', 'CLOSED_TEMPORARILY'):
        open_score -= 0.30
        breakdown.append({"signal": "google", "weight": -0.30, "description": "Google: closed"})

    # Website
    website = layers.get('website', {})
    ws_status = website.get('status') if website else None

    if ws_status == 'redirect':
        open_score -= 0.20
        breakdown.append({"signal": "website", "weight": -0.20, "description": "Website: redirected"})
    elif ws_status == 'dead':
        open_score -= 0.10
        breakdown.append({"signal": "website", "weight": -0.10, "description": "Website: dead"})
    elif ws_status == 'parked':
        open_score -= 0.15
        breakdown.append({"signal": "website", "weight": -0.15, "description": "Website: parked"})

    # Closure text
    closure_signals = v.get('closure_signals', [])
    fresh_closure = [s for s in closure_signals if s.get('image_age_years', 99) <= 5]
    strong_closure = any(s.get('strength') == 'strong' for s in fresh_closure)

    if strong_closure:
        best = fresh_closure[0]
        open_score -= 0.30
        breakdown.append({"signal": "closure_text", "weight": -0.30, "description": f"Closure text: \"{best.get('text', '?')}\""})

    # Text detection
    text_layer = layers.get('text', {})
    if text_layer:
        verdict = text_layer.get('verdict', 'no_match')
        image_age = text_layer.get('image_age_years', 99)
        images_analyzed = v.get('images_analyzed', 0)
        match_strength = text_layer.get('match_strength')

        text_found = verdict in ('full_match', 'partial_match')

        if text_found:
            if image_age <= 2:
                age_discount = 1.0
            elif image_age <= 4:
                age_discount = 0.5
            elif image_age <= 5:
                age_discount = 0.2
            else:
                age_discount = 0.0

            strength = match_strength or ('full' if verdict == 'full_match' else 'partial')
            if strength == 'full':
                text_boost = 0.20 * age_discount
            else:
                text_boost = 0.10 * age_discount

            if text_boost != 0:
                open_score += text_boost
                age_note = f" ({image_age:.0f}yr)" if image_age > 3 else ""
                breakdown.append({"signal": "text", "weight": round(text_boost, 4), "description": f"Text: {strength} match{age_note}"})
        else:
            if image_age <= 3 and images_analyzed >= 2:
                open_score -= 0.05
                breakdown.append({"signal": "text", "weight": -0.05, "description": f"No text match ({images_analyzed} recent images)"})

    # XGBoost nudge
    xgb = layers.get('xgboost', {})
    xgb_score = xgb.get('score', 0.5) if xgb else 0.5

    if xgb_score < 0.40:
        xgb_nudge = (xgb_score - 0.5) * 0.20
        open_score += xgb_nudge
        breakdown.append({"signal": "xgboost", "weight": round(xgb_nudge, 4), "description": f"Metadata: {xgb_score:.0%} open (nudge)"})

    open_score = max(0.0, min(1.0, open_score))
    return breakdown


def main():
    for filepath in input_files:
        if not os.path.exists(filepath):
            print(f"Skipping {filepath} (not found)")
            continue

        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)

        updated = 0
        mismatches = 0
        for entry in data:
            v = entry.get('vision', {})
            if not v:
                continue

            breakdown = compute_score_breakdown(entry)
            if breakdown:
                v['score_breakdown'] = breakdown

                # Sanity check: sum of weights should be close to confidence
                total = sum(item['weight'] for item in breakdown)
                total = max(0.0, min(1.0, total))
                conf = v.get('confidence', 0)
                if abs(total - conf) > 0.02:
                    mismatches += 1
                    print(f"  MISMATCH: {entry.get('name', '?')}: reconstructed={total:.4f} vs stored={conf}")
                updated += 1

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"{os.path.basename(filepath)}: {updated} entries updated, {mismatches} mismatches")


if __name__ == '__main__':
    main()
