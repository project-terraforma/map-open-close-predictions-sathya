"""
Generate TerraForma ensemble map.

Pin colors = ensemble prediction (green=open, red=closed).
Stats panel shows accuracy vs Google ground truth.
Filter by city and correctness.

Usage:
    python -m ui.generate_map
"""

import json
from pathlib import Path

from sqlalchemy import text
from src.config import engine

TARGET_PER_CITY = 20
DEMO_CITIES = ("san_francisco", "new_york", "chicago")


def fetch_data():
    """Pull ground truth + ensemble scores in one query."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                g.city,
                g.overture_id,
                o.name,
                o.latitude,
                o.longitude,
                o.address,
                g.business_status,
                g.is_open AS google_open,
                e.ensemble_pct,
                e.ensemble_label,
                e.model_score,
                e.osm_score,
                e.osm_found,
                e.osm_name,
                e.osm_disused,
                e.osm_building,
                e.osm_businesses_here,
                e.web_score,
                e.web_alive,
                e.brave_results,
                e.ai_judgment,
                e.ai_confidence,
                e.signals_used
            FROM ground_truth.labels g
            JOIN overture.places o ON o.id = g.overture_id
            LEFT JOIN predictions.ensemble e ON e.overture_id = g.overture_id
            WHERE g.city IN :cities
            ORDER BY g.city, o.name
        """), {"cities": DEMO_CITIES}).fetchall()
    return rows


def balance_cities(rows):
    """Pick up to TARGET_PER_CITY per city, balanced open/closed."""
    by_city = {}
    for r in rows:
        city = r[0]
        if city not in by_city:
            by_city[city] = {"open": [], "closed": []}
        bucket = "open" if r[7] else "closed"
        by_city[city][bucket].append(r)

    result = {}
    for city, buckets in by_city.items():
        closed = buckets["closed"]
        opened = buckets["open"]
        n_closed = min(len(closed), TARGET_PER_CITY // 2)
        n_open = min(len(opened), TARGET_PER_CITY - n_closed)
        picked = closed[:n_closed] + opened[:n_open]
        result[city] = picked
        print(f"{city}: {n_open} open + {n_closed} closed = {len(picked)}")
    return result


def row_to_dict(r):
    addr = r[5]
    if isinstance(addr, dict):
        addr = addr.get("freeform", "")
    elif addr is None:
        addr = ""

    ens_pct = float(r[8]) if r[8] is not None else None
    predicted_open = ens_pct >= 60 if ens_pct is not None else None
    google_open = bool(r[7])
    correct = predicted_open == google_open if predicted_open is not None else None

    return {
        "city": r[0],
        "overture_id": r[1],
        "name": r[2],
        "lat": float(r[3]),
        "lon": float(r[4]),
        "address": str(addr),
        "google_status": r[6] or "UNKNOWN",
        "google_open": google_open,
        "ens_pct": ens_pct,
        "ens_label": r[9] or "unknown",
        "predicted_open": predicted_open,
        "correct": correct,
        "model_score": float(r[10]) if r[10] is not None else None,
        "osm_score": float(r[11]) if r[11] is not None else None,
        "osm_found": bool(r[12]) if r[12] is not None else False,
        "osm_name": r[13],
        "osm_disused": bool(r[14]) if r[14] is not None else False,
        "osm_building": bool(r[15]) if r[15] is not None else False,
        "osm_biz_count": r[16] or 0,
        "web_score": float(r[17]) if r[17] is not None else None,
        "web_alive": r[18],
        "brave_results": r[19] or 0,
        "ai_judgment": r[20],
        "ai_confidence": r[21] or 0,
        "signals_used": r[22] or 0,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TerraForma — Ensemble Predictions</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
        #map { width: 100%; height: 100vh; }

        /* ── Top bar: city + filter toggles ── */
        .top-bar {
            position: absolute; top: 10px; left: 60px; z-index: 1000;
            display: flex; gap: 8px; align-items: center;
        }
        .btn-group {
            background: white; padding: 6px 8px; border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15); display: flex; gap: 4px;
        }
        .btn {
            padding: 6px 14px; border: none; border-radius: 8px;
            background: #f1f5f9; cursor: pointer; font-size: 12px; font-weight: 600;
            color: #475569; transition: all 0.15s;
        }
        .btn:hover { background: #e2e8f0; }
        .btn.active { background: #1e40af; color: white; }
        .btn.active-green { background: #16a34a; color: white; }
        .btn.active-red { background: #dc2626; color: white; }

        /* ── Stats panel ── */
        .stats {
            position: absolute; top: 10px; right: 10px; z-index: 1000;
            background: white; padding: 16px 20px; border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.15); min-width: 200px; font-size: 13px;
            text-align: center;
        }
        .stats h3 { font-size: 15px; margin-bottom: 6px; color: #1e293b; }
        .accuracy-big {
            font-size: 36px; font-weight: 800;
            margin: 4px 0 2px; line-height: 1;
        }
        .accuracy-sub { font-size: 11px; color: #64748b; margin-bottom: 10px; }
        .gt-filters { display: flex; gap: 4px; justify-content: center; }
        .gt-btn {
            padding: 6px 12px; border: none; border-radius: 8px;
            background: #f1f5f9; cursor: pointer; font-size: 11px; font-weight: 600;
            color: #475569; transition: all 0.15s;
        }
        .gt-btn:hover { background: #e2e8f0; }
        .gt-btn.active { background: #1e40af; color: white; }
        .gt-btn.active-green { background: #16a34a; color: white; }
        .gt-btn.active-red { background: #dc2626; color: white; }

        /* ── Legend ── */
        .legend {
            background: white; padding: 10px 14px; border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15); font-size: 12px; line-height: 1.8;
        }
        .legend-item { display: flex; align-items: center; gap: 8px; }
        .legend-dot {
            width: 12px; height: 12px; border-radius: 50%;
            border: 2px solid white; box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }

        /* ── Popup ── */
        .popup { max-width: 300px; font-size: 12px; }
        .popup h3 { font-size: 15px; margin-bottom: 6px; color: #1e293b; }
        .popup .addr { color: #64748b; margin-bottom: 8px; }

        .ens-box {
            padding: 10px; border-radius: 10px; margin-bottom: 8px;
        }
        .ens-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
        .ens-pct { font-size: 22px; font-weight: 800; }
        .ens-verdict { font-size: 12px; font-weight: 600; }
        .ens-bar { height: 6px; border-radius: 3px; background: #e2e8f0; overflow: hidden; }
        .ens-fill { height: 100%; border-radius: 3px; }

        .signals { display: flex; gap: 6px; margin-top: 8px; }
        .sig {
            flex: 1; text-align: center; padding: 6px 4px;
            border-radius: 8px; background: #f8fafc;
        }
        .sig-pct { font-size: 14px; font-weight: 700; display: block; }
        .sig-name { font-size: 9px; color: #64748b; font-weight: 600; text-transform: uppercase; }
        .sig-detail { font-size: 9px; color: #94a3b8; margin-top: 2px; }

        .truth-row {
            margin-top: 8px; padding: 6px 10px; border-radius: 8px;
            font-size: 11px; display: flex; justify-content: space-between; align-items: center;
        }
        .truth-open { background: #f0fdf4; color: #166534; }
        .truth-closed { background: #fef2f2; color: #991b1b; }
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="top-bar" id="topBar"></div>
    <div class="stats" id="stats"></div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        const allData = __DATA__;

        const CENTERS = {
            san_francisco: [37.77, -122.42],
            new_york: [40.75, -73.99],
            chicago: [41.88, -87.63],
        };
        const LABELS = {
            san_francisco: "San Francisco",
            new_york: "New York",
            chicago: "Chicago",
        };

        const map = L.map('map').setView([37.77, -122.42], 4);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap contributors', maxZoom: 19,
        }).addTo(map);

        let markers = [];
        let activeCity = 'all';
        let activeFilter = 'all';  // all | correct | wrong

        // ── Pin color = ensemble prediction ──
        function pinColor(d) {
            if (d.ens_pct === null) return '#9ca3af';  // no prediction yet
            return d.predicted_open ? '#22c55e' : '#ef4444';
        }

        function pinBorder(d) {
            if (d.correct === null) return '#fff';
            return d.correct ? '#fff' : '#000';  // black border = wrong prediction
        }

        // ── Popup ──
        function pctColor(pct) {
            if (pct >= 60) return '#16a34a';
            if (pct >= 40) return '#ca8a04';
            return '#dc2626';
        }

        function renderPopup(d) {
            const hasPrediction = d.ens_pct !== null;
            let ensHtml = '';

            if (hasPrediction) {
                const pct = d.ens_pct.toFixed(1);
                const color = pctColor(d.ens_pct);
                const bgColor = d.predicted_open ? '#f0fdf4' : '#fef2f2';
                const verdict = d.predicted_open ? 'OPEN' : 'CLOSED';

                // Build the weighted math line
                // Weights: Model=25%, OSM=20%, Web=55% (redistributed if missing)
                const signals = [];
                if (d.model_score !== null) signals.push({ name: 'Model', score: d.model_score, baseW: 0.25 });
                if (d.osm_score !== null) signals.push({ name: 'OSM', score: d.osm_score, baseW: 0.20 });
                if (d.web_score !== null) signals.push({ name: 'Web', score: d.web_score, baseW: 0.55 });

                // Redistribute weights to match what ensemble actually did
                const totalW = signals.reduce((s, x) => s + x.baseW, 0);
                signals.forEach(s => s.w = s.baseW / totalW);

                // Build math string: "(82% x 0.25) + (50% x 0.20) + (90% x 0.55) = 78.0%"
                const mathParts = signals.map(s =>
                    `(${(s.score * 100).toFixed(0)}% x ${s.w.toFixed(2)})`
                );
                const mathStr = mathParts.join(' + ') + ` = <b>${pct}%</b>`;

                // Signal detail lines
                let osmDetail = '';
                if (d.osm_found && d.osm_disused) osmDetail = 'disused';
                else if (d.osm_found) osmDetail = 'found on OSM';
                else if (d.osm_name && d.osm_name.startsWith('replaced:')) osmDetail = 'replaced by ' + d.osm_name.substring(10);
                else if (d.osm_building) osmDetail = 'building found, no listing';
                else osmDetail = 'not on OSM';

                let webDetail = '';
                if (d.ai_judgment && d.ai_judgment !== 'UNCERTAIN') {
                    webDetail = 'AI says ' + d.ai_judgment.toLowerCase() + ' (' + d.ai_confidence + '% conf)';
                } else if (d.web_alive === true) webDetail = 'website alive';
                else if (d.web_alive === false) webDetail = 'website down';
                else webDetail = 'no website found';

                ensHtml = `
                    <div class="ens-box" style="background:${bgColor}">
                        <div class="ens-header">
                            <span class="ens-pct" style="color:${color}">${pct}%</span>
                            <span class="ens-verdict" style="color:${color}">${verdict}</span>
                        </div>
                        <div class="ens-bar">
                            <div class="ens-fill" style="width:${pct}%;background:${color}"></div>
                        </div>
                        <div style="font-size:10px;color:#64748b;margin-top:8px;line-height:1.5;font-family:monospace">
                            ${mathStr}
                        </div>
                        <div style="margin-top:8px;font-size:11px;line-height:1.7">
                            ${signals.map(s => {
                                const icon = s.score >= 0.5 ? '&#9679;' : '&#9675;';
                                const c = s.score >= 0.5 ? '#16a34a' : '#dc2626';
                                let detail = '';
                                if (s.name === 'OSM') detail = osmDetail;
                                else if (s.name === 'Web') detail = webDetail;
                                else detail = s.score >= 0.5 ? 'predicts open' : 'predicts closed';
                                return `<div><span style="color:${c}">${icon}</span> <b>${s.name}</b> ${(s.score*100).toFixed(0)}% — ${detail}</div>`;
                            }).join('')}
                        </div>
                    </div>
                `;
            } else {
                ensHtml = '<div class="ens-box" style="background:#f8fafc;text-align:center;color:#94a3b8">No prediction yet</div>';
            }

            const truthClass = d.google_open ? 'truth-open' : 'truth-closed';
            const truthLabel = d.google_open ? 'OPEN' : 'CLOSED';
            const matchText = d.correct === true ? '&#10003;' : (d.correct === false ? '&#10007;' : '');
            const matchColor = d.correct === true ? '#16a34a' : '#dc2626';

            return `
                <div class="popup">
                    <h3>${d.name}</h3>
                    <div class="addr">${d.address || 'No address'}</div>
                    ${ensHtml}
                    <div class="truth-row ${truthClass}">
                        <span>Google: <b>${truthLabel}</b></span>
                        <b style="color:${matchColor}">${matchText}</b>
                    </div>
                </div>
            `;
        }

        // ── Stats panel — accuracy + ground truth filter ──
        let activeGT = 'all';  // all | open | closed

        function updateStats(unfilteredData) {
            const hasEns = unfilteredData.filter(d => d.ens_pct !== null);
            const correct = hasEns.filter(d => d.correct === true).length;
            const acc = hasEns.length > 0 ? (100 * correct / hasEns.length) : 0;
            const accColor = acc >= 80 ? '#16a34a' : (acc >= 60 ? '#ca8a04' : '#dc2626');

            const gtOpen = unfilteredData.filter(d => d.google_open).length;
            const gtClosed = unfilteredData.length - gtOpen;

            const cityName = activeCity === 'all' ? 'All Cities' : (LABELS[activeCity] || activeCity);

            let html = `<h3>${cityName}</h3>`;
            if (hasEns.length > 0) {
                html += `
                    <div class="accuracy-big" style="color:${accColor}">${acc.toFixed(0)}%</div>
                    <div class="accuracy-sub">accuracy vs Google (${correct}/${hasEns.length})</div>
                `;
            } else {
                html += `<div class="accuracy-sub" style="margin:8px 0">No predictions yet</div>`;
            }

            html += `
                <div style="border-top:1px solid #e2e8f0;margin:10px 0 8px"></div>
                <div style="font-size:11px;color:#64748b;margin-bottom:6px">Ground Truth Filter</div>
                <div class="gt-filters">
                    <button class="gt-btn ${activeGT === 'all' ? 'active' : ''}" onclick="setGT('all')">All (${unfilteredData.length})</button>
                    <button class="gt-btn ${activeGT === 'open' ? 'active-green' : ''}" onclick="setGT('open')">Open (${gtOpen})</button>
                    <button class="gt-btn ${activeGT === 'closed' ? 'active-red' : ''}" onclick="setGT('closed')">Closed (${gtClosed})</button>
                </div>
            `;

            document.getElementById('stats').innerHTML = html;
        }

        window.setGT = function(val) {
            activeGT = val;
            render();
        };

        // ── Render markers ──
        function render() {
            markers.forEach(m => map.removeLayer(m));
            markers = [];

            let data = activeCity === 'all'
                ? allData
                : allData.filter(d => d.city === activeCity);

            // For stats, use city-filtered but NOT gt-filtered data
            const statsData = data;

            // Apply ground truth filter
            if (activeGT === 'open') data = data.filter(d => d.google_open);
            if (activeGT === 'closed') data = data.filter(d => !d.google_open);

            // Apply correct/wrong filter
            if (activeFilter === 'correct') data = data.filter(d => d.correct === true);
            if (activeFilter === 'wrong') data = data.filter(d => d.correct === false);

            data.forEach(d => {
                const m = L.circleMarker([d.lat, d.lon], {
                    radius: 9,
                    fillColor: pinColor(d),
                    color: pinBorder(d),
                    weight: d.correct === false ? 3 : 2,
                    opacity: 1,
                    fillOpacity: 0.9,
                }).addTo(map);
                m.bindPopup(renderPopup(d), { maxWidth: 340 });
                markers.push(m);
            });

            if (data.length > 0 && activeCity !== 'all' && CENTERS[activeCity]) {
                map.setView(CENTERS[activeCity], 13);
            } else if (data.length > 0) {
                const group = L.featureGroup(markers);
                map.fitBounds(group.getBounds().pad(0.1));
            }

            updateStats(statsData);

            // Update button states
            document.querySelectorAll('.city-btn').forEach(b =>
                b.classList.toggle('active', b.dataset.city === activeCity));
            document.querySelectorAll('.filter-btn').forEach(b => {
                b.classList.remove('active', 'active-green', 'active-red');
                if (b.dataset.filter === activeFilter) {
                    if (activeFilter === 'correct') b.classList.add('active-green');
                    else if (activeFilter === 'wrong') b.classList.add('active-red');
                    else b.classList.add('active');
                }
            });
        }

        // ── Build top bar ──
        const topBar = document.getElementById('topBar');

        // City buttons
        const cityGroup = document.createElement('div');
        cityGroup.className = 'btn-group';

        const cities = [...new Set(allData.map(d => d.city))];
        [{ key: 'all', label: 'All' }, ...cities.map(c => ({ key: c, label: LABELS[c] || c }))].forEach(c => {
            const btn = document.createElement('button');
            btn.className = 'btn city-btn';
            const count = c.key === 'all' ? allData.length : allData.filter(d => d.city === c.key).length;
            btn.textContent = `${c.label} (${count})`;
            btn.dataset.city = c.key;
            btn.onclick = () => { activeCity = c.key; activeGT = 'all'; render(); };
            cityGroup.appendChild(btn);
        });
        topBar.appendChild(cityGroup);

        // Filter buttons (correct/wrong)
        const filterGroup = document.createElement('div');
        filterGroup.className = 'btn-group';
        [
            { key: 'all', label: 'All' },
            { key: 'correct', label: 'Correct' },
            { key: 'wrong', label: 'Wrong' },
        ].forEach(f => {
            const btn = document.createElement('button');
            btn.className = 'btn filter-btn';
            btn.textContent = f.label;
            btn.dataset.filter = f.key;
            btn.onclick = () => { activeFilter = f.key; render(); };
            filterGroup.appendChild(btn);
        });
        topBar.appendChild(filterGroup);

        // Legend
        const legend = L.control({ position: 'bottomleft' });
        legend.onAdd = () => {
            const div = L.DomUtil.create('div', 'legend');
            div.innerHTML = `
                <div class="legend-item"><div class="legend-dot" style="background:#22c55e"></div> Predicted Open</div>
                <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div> Predicted Closed</div>
                <div style="margin-top:4px;font-size:11px;color:#64748b">Black border = wrong prediction</div>
            `;
            return div;
        };
        legend.addTo(map);

        render();
    </script>
</body>
</html>"""


def generate():
    rows = fetch_data()
    cities = balance_cities(rows)

    all_rows = []
    for city, city_rows in cities.items():
        all_rows.extend(city_rows)

    all_data = [row_to_dict(r) for r in all_rows]

    scored = sum(1 for d in all_data if d["ens_pct"] is not None)
    print(f"\nEnsemble scores: {scored}/{len(all_data)} businesses")

    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(all_data, indent=2))

    out = "ui/ground_truth_map.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out} with {len(all_data)} businesses across {len(cities)} cities")


if __name__ == "__main__":
    generate()
