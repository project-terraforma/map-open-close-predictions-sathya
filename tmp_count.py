import json
from collections import Counter

cities = {
    'SF': 'src/data/test_data.json',
    'LA': 'src/data/test_data_la.json',
    'Chicago': 'src/data/test_data_chicago.json'
}

fps = []
fns = []

for city, path in cities.items():
    with open(path) as f:
        data = json.load(f)
    for entry in data:
        gt = entry.get('ground_truth', 'unknown')
        pred = entry['vision']['prediction']
        conf = entry['vision']['confidence']
        layers = entry['vision']['layers']
        fsq = layers.get('foursquare', {}).get('status', 'N/A')
        web = layers.get('website', {}).get('status', 'N/A')
        text = layers.get('text', {}).get('verdict', 'N/A')
        xgb = layers.get('xgboost', {}).get('score', 0)

        rec = {'city': city, 'name': entry.get('name','?'), 'conf': conf, 'fsq': fsq, 'web': web, 'text': text, 'xgb': xgb}

        if pred == 'open' and gt == 'closed':
            fps.append(rec)
        elif pred == 'not_open' and gt == 'open':
            fns.append(rec)

# === FP PATTERN ANALYSIS ===
print('=' * 80)
print('FALSE POSITIVE PATTERN ANALYSIS (predicted OPEN, actually CLOSED)')
print('=' * 80)

fp_fsq = Counter(e['fsq'] for e in fps)
print('\nFoursquare status distribution:')
for k, v in fp_fsq.most_common():
    print(f'  {k}: {v}/{len(fps)} ({v/len(fps)*100:.0f}%)')

fp_web = Counter(e['web'] for e in fps)
print('\nWebsite status distribution:')
for k, v in fp_web.most_common():
    print(f'  {k}: {v}/{len(fps)} ({v/len(fps)*100:.0f}%)')

fp_text = Counter(e['text'] for e in fps)
print('\nText verdict distribution:')
for k, v in fp_text.most_common():
    print(f'  {k}: {v}/{len(fps)} ({v/len(fps)*100:.0f}%)')

fp_xgb_high = sum(1 for e in fps if e['xgb'] > 0.5)
fp_xgb_low = sum(1 for e in fps if e['xgb'] <= 0.5)
print(f'\nXGBoost score: >0.5 (looks open): {fp_xgb_high}, <=0.5 (looks closed): {fp_xgb_low}')
print(f'  Mean XGBoost score: {sum(e["xgb"] for e in fps)/len(fps):.3f}')

# Compound patterns
print('\nCompound patterns:')
fp_fsq_verified_web_dead = sum(1 for e in fps if e['fsq']=='verified' and e['web']=='dead')
print(f'  fsq=verified + web=dead: {fp_fsq_verified_web_dead}/{len(fps)} ({fp_fsq_verified_web_dead/len(fps)*100:.0f}%)')
fp_fsq_verified_web_alive = sum(1 for e in fps if e['fsq']=='verified' and e['web']=='alive')
print(f'  fsq=verified + web=alive: {fp_fsq_verified_web_alive}/{len(fps)} ({fp_fsq_verified_web_alive/len(fps)*100:.0f}%)')
fp_text_match_web_dead = sum(1 for e in fps if e['text']=='full_match' and e['web']=='dead')
print(f'  text=full_match + web=dead: {fp_text_match_web_dead}/{len(fps)}')

# By city
print('\nPer-city FP breakdown:')
for c in ['SF','LA','Chicago']:
    city_fps = [e for e in fps if e['city']==c]
    print(f'  {c}: {len(city_fps)} FPs')
    for e in city_fps:
        print(f'    {e["name"]}: conf={e["conf"]:.3f} fsq={e["fsq"]} web={e["web"]} text={e["text"]} xgb={e["xgb"]:.4f}')

# === FN PATTERN ANALYSIS ===
print()
print('=' * 80)
print('FALSE NEGATIVE PATTERN ANALYSIS (predicted NOT_OPEN, actually OPEN)')
print('=' * 80)

fn_fsq = Counter(e['fsq'] for e in fns)
print('\nFoursquare status distribution:')
for k, v in fn_fsq.most_common():
    print(f'  {k}: {v}/{len(fns)} ({v/len(fns)*100:.0f}%)')

fn_web = Counter(e['web'] for e in fns)
print('\nWebsite status distribution:')
for k, v in fn_web.most_common():
    print(f'  {k}: {v}/{len(fns)} ({v/len(fns)*100:.0f}%)')

fn_text = Counter(e['text'] for e in fns)
print('\nText verdict distribution:')
for k, v in fn_text.most_common():
    print(f'  {k}: {v}/{len(fns)} ({v/len(fns)*100:.0f}%)')

fn_xgb_high = sum(1 for e in fns if e['xgb'] > 0.5)
fn_xgb_low = sum(1 for e in fns if e['xgb'] <= 0.5)
print(f'\nXGBoost score: >0.5 (looks open): {fn_xgb_high}, <=0.5 (looks closed): {fn_xgb_low}')
print(f'  Mean XGBoost score: {sum(e["xgb"] for e in fns)/len(fns):.3f}')

# Compound patterns
print('\nCompound patterns:')
fn_nodata_dead = sum(1 for e in fns if e['fsq']=='no_data' and e['web']=='dead')
print(f'  fsq=no_data + web=dead: {fn_nodata_dead}/{len(fns)} ({fn_nodata_dead/len(fns)*100:.0f}%)')
fn_nodata_nourl = sum(1 for e in fns if e['fsq']=='no_data' and e['web']=='no_url')
print(f'  fsq=no_data + web=no_url: {fn_nodata_nourl}/{len(fns)} ({fn_nodata_nourl/len(fns)*100:.0f}%)')
fn_nodata_alive = sum(1 for e in fns if e['fsq']=='no_data' and e['web']=='alive')
print(f'  fsq=no_data + web=alive: {fn_nodata_alive}/{len(fns)} ({fn_nodata_alive/len(fns)*100:.0f}%)')
fn_mismatch = sum(1 for e in fns if e['fsq']=='mismatch')
print(f'  fsq=mismatch (any web): {fn_mismatch}/{len(fns)} ({fn_mismatch/len(fns)*100:.0f}%)')
fn_no_images = sum(1 for e in fns if e['text']=='no_images')
print(f'  text=no_images: {fn_no_images}/{len(fns)} ({fn_no_images/len(fns)*100:.0f}%)')
fn_xgb_agrees_open = sum(1 for e in fns if e['xgb'] > 0.6)
print(f'  XGBoost >0.6 (visual looks open): {fn_xgb_agrees_open}/{len(fns)} ({fn_xgb_agrees_open/len(fns)*100:.0f}%)')

# By city
print('\nPer-city FN breakdown:')
for c in ['SF','LA','Chicago']:
    city_fns = [e for e in fns if e['city']==c]
    print(f'  {c}: {len(city_fns)} FNs')
    for e in city_fns:
        print(f'    {e["name"]}: conf={e["conf"]:.3f} fsq={e["fsq"]} web={e["web"]} text={e["text"]} xgb={e["xgb"]:.4f}')

# === OVERALL SUMMARY ===
print()
print('=' * 80)
print('ACCURACY SUMMARY')
print('=' * 80)
for city, path in cities.items():
    with open(path) as f:
        data = json.load(f)
    total = len(data)
    city_fp = len([e for e in fps if e['city']==city])
    city_fn = len([e for e in fns if e['city']==city])
    correct = total - city_fp - city_fn
    print(f'  {city:>8}: {correct}/{total} correct ({correct/total*100:.1f}%)  |  {city_fp} FP, {city_fn} FN')
total_all = 76*3
total_correct = total_all - len(fps) - len(fns)
print(f'  {"OVERALL":>8}: {total_correct}/{total_all} correct ({total_correct/total_all*100:.1f}%)  |  {len(fps)} FP, {len(fns)} FN')

print()
print('=' * 80)
print('IMPACT ANALYSIS - Signal improvements ranked by error reduction')
print('=' * 80)

# FP fixes
print()
print('--- Fixing FALSE POSITIVES (17 total) ---')
print(f'  1. Make web=dead OVERRIDE fsq=verified -> fixes {fp_fsq_verified_web_dead}/17 FPs (88%)')
print(f'     [The dominant FP pattern: foursquare says verified but website is dead]')
print(f'  2. Only 2/17 FPs have web=alive; these are genuinely hard cases')

# FN fixes
fn_fixable_by_xgb = sum(1 for e in fns if e['xgb'] > 0.6 and e['fsq'] == 'no_data')
fn_fixable_by_web_alive = sum(1 for e in fns if e['web'] == 'alive' and e['fsq'] == 'no_data')
fn_fixable_by_either = sum(1 for e in fns if (e['xgb'] > 0.6 or e['web'] == 'alive') and e['fsq'] == 'no_data')

print()
print('--- Fixing FALSE NEGATIVES (23 total) ---')
print(f'  1. Trust XGBoost>0.6 when fsq=no_data -> fixes {fn_fixable_by_xgb}/23 FNs')
print(f'  2. Trust web=alive when fsq=no_data -> fixes {fn_fixable_by_web_alive}/23 FNs')
print(f'  3. Trust EITHER XGBoost>0.6 OR web=alive when fsq=no_data -> fixes {fn_fixable_by_either}/23 FNs')
print(f'  4. The core issue: fsq=no_data is treated as strong negative evidence.')
print(f'     It should be NEUTRAL (absence of data, not evidence of closure).')
