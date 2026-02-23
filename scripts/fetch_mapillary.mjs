import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_PATH = path.join(__dirname, '..', 'src', 'data', 'mock_data.json');
const ACCESS_TOKEN = 'MLY|25120378337641785|21245babc5f6905ed3da857aba13bc87';

const THREE_YEARS_MS = 3 * 365.25 * 24 * 60 * 60 * 1000;
const ONE_YEAR_MS = 365.25 * 24 * 60 * 60 * 1000;

// ============================================================
// Step 1: Fetch real fast food locations from OpenStreetMap
// ============================================================

async function fetchFastFoodLocations() {
    console.log('Fetching real fast food locations from OpenStreetMap...');
    const query = `
[out:json][timeout:30];
area["name"="San Francisco"]["admin_level"="6"]->.sf;
node["amenity"="fast_food"]["name"](area.sf);
out body;
`;
    let data;
    for (let attempt = 0; attempt < 5; attempt++) {
        const res = await fetch('https://overpass-api.de/api/interpreter', {
            method: 'POST',
            body: 'data=' + encodeURIComponent(query),
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        });
        const text = await res.text();
        try {
            data = JSON.parse(text);
            break;
        } catch {
            console.log(`  Overpass attempt ${attempt + 1} failed, retrying in 15s...`);
            await new Promise((r) => setTimeout(r, 15000));
        }
    }
    if (!data) throw new Error('Overpass API failed after 5 attempts');

    const chains = [
        'McDonald', 'Burger King', 'Taco Bell', 'KFC',
        'Jack in the Box', 'Chick-fil-A', 'Popeyes', 'Subway',
        'Panda Express', 'Five Guys', 'Chipotle', 'In-N-Out',
        "Carl's Jr", 'Wingstop', 'Domino', 'Little Caesars',
        'Jollibee', 'Shake Shack', 'Del Taco', "Church's Chicken",
    ];

    const locations = data.elements
        .filter((e) =>
            chains.some(
                (c) =>
                    e.tags.name?.toLowerCase().includes(c.toLowerCase()) ||
                    e.tags.brand?.toLowerCase().includes(c.toLowerCase())
            )
        )
        .map((e, i) => ({
            id: i + 1,
            name: e.tags.name,
            category: 'Fast Food',
            address: [e.tags['addr:housenumber'], e.tags['addr:street']].filter(Boolean).join(' ') + ', San Francisco',
            location: [e.lon, e.lat],
        }));

    console.log(`Found ${locations.length} verified fast food locations from OSM.\n`);
    return locations;
}

// ============================================================
// GEOMETRY HELPERS
// ============================================================

function haversineDistance(lat1, lon1, lat2, lon2) {
    const R = 6371000;
    const toRad = (deg) => (deg * Math.PI) / 180;
    const dLat = toRad(lat2 - lat1);
    const dLon = toRad(lon2 - lon1);
    const a =
        Math.sin(dLat / 2) ** 2 +
        Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function bearing(lat1, lon1, lat2, lon2) {
    const toRad = (deg) => (deg * Math.PI) / 180;
    const toDeg = (rad) => (rad * 180) / Math.PI;
    const dLon = toRad(lon2 - lon1);
    const y = Math.sin(dLon) * Math.cos(toRad(lat2));
    const x =
        Math.cos(toRad(lat1)) * Math.sin(toRad(lat2)) -
        Math.sin(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.cos(dLon);
    return (toDeg(Math.atan2(y, x)) + 360) % 360;
}

function angleDiff(a, b) {
    const diff = Math.abs(a - b) % 360;
    return diff > 180 ? 360 - diff : diff;
}

// ============================================================
// MAPILLARY API — get as many images as possible near a point
// ============================================================

async function fetchImages(lng, lat, radiusMeters) {
    const delta = radiusMeters / 111000;
    const bbox = `${lng - delta},${lat - delta},${lng + delta},${lat + delta}`;
    const fields = 'id,thumb_2048_url,captured_at,geometry,compass_angle,computed_compass_angle,is_pano';
    const url = `https://graph.mapillary.com/images?access_token=${ACCESS_TOKEN}&fields=${fields}&bbox=${bbox}&limit=200`;

    const res = await fetch(url);
    if (!res.ok) {
        console.error(`  API error ${res.status}: ${await res.text()}`);
        return [];
    }
    const json = await res.json();
    return json.data || [];
}

// ============================================================
// SCORING — strict: close + directly facing the building
// ============================================================

function scoreImage(img, poiLat, poiLng) {
    const [imgLng, imgLat] = img.geometry.coordinates;
    const dist = haversineDistance(poiLat, poiLng, imgLat, imgLng);

    // Use computed compass angle if available (more accurate), fall back to raw
    const camAngle = img.computed_compass_angle ?? img.compass_angle;
    const bearingToPoi = bearing(imgLat, imgLng, poiLat, poiLng);
    const facingError = angleDiff(camAngle, bearingToPoi);

    // Approach direction: where the camera is relative to the business
    const approachBearing = (bearingToPoi + 180) % 360;
    const quadrant = Math.floor(approachBearing / 90) % 4;

    return {
        score: facingError * 3 + dist, // facing matters even more
        dist: Math.round(dist),
        facingError: Math.round(facingError),
        quadrant,
    };
}

function formatImage(img, poiLat, poiLng) {
    const [imgLng, imgLat] = img.geometry.coordinates;
    const dist = Math.round(haversineDistance(poiLat, poiLng, imgLat, imgLng));
    const date = new Date(img.captured_at).toISOString().split('T')[0];
    return {
        url: img.thumb_2048_url,
        date,
        distance_m: dist,
    };
}

// Pick diverse approach directions, up to maxPerQuadrant from each of 4 quadrants
function pickDiverseImages(candidates, maxPerQuadrant = 2) {
    const buckets = [[], [], [], []];
    for (const c of candidates) buckets[c.quadrant].push(c);
    const picked = [];
    for (const bucket of buckets) picked.push(...bucket.slice(0, maxPerQuadrant));
    return picked;
}

// Find the best before/after anchor pair ~3 years apart
function findBestTimePair(scored) {
    let bestPair = null;
    let bestQuality = Infinity;

    for (const early of scored) {
        for (const late of scored) {
            if (late.img.captured_at <= early.img.captured_at) continue;
            const gap = late.img.captured_at - early.img.captured_at;
            const gapDiff = Math.abs(gap - THREE_YEARS_MS);
            const quality = gapDiff / THREE_YEARS_MS + (early.score + late.score) / 500;
            if (quality < bestQuality) {
                bestQuality = quality;
                bestPair = { early, late, gapYears: gap / (365.25 * 24 * 60 * 60 * 1000) };
            }
        }
    }
    return bestPair;
}

// ============================================================
// MAIN
// ============================================================

async function main() {
    const FAST_FOOD = await fetchFastFoodLocations();
    const updatedData = [];
    let skipped = 0;

    for (let i = 0; i < FAST_FOOD.length; i++) {
        const poi = FAST_FOOD[i];
        const [lng, lat] = poi.location;
        console.log(`[${i + 1}/${FAST_FOOD.length}] ${poi.name} - ${poi.address}`);

        // Pull lots of images from a tight area
        let allImages = [];
        for (const radius of [25, 40, 60]) {
            allImages = await fetchImages(lng, lat, radius);
            if (allImages.length >= 30) break;
            console.log(`  ${allImages.length} imgs at ${radius}m, widening...`);
            await new Promise((r) => setTimeout(r, 100));
        }

        if (allImages.length === 0) {
            console.log(`  SKIP: no images found`);
            skipped++;
            continue;
        }

        // Score all images, strict filtering:
        //   - Must have compass angle
        //   - Facing within 45 degrees of the building (tighter than before)
        //   - Within 50m of the building
        //   - Prefer non-pano (sidewalk) images
        const scored = allImages
            .filter((img) => (img.computed_compass_angle ?? img.compass_angle) != null)
            .map((img) => {
                const s = scoreImage(img, lat, lng);
                const panoPenalty = img.is_pano ? 40 : 0;
                return { img, ...s, score: s.score + panoPenalty };
            })
            .filter((s) => s.facingError <= 45 && s.dist <= 50)
            .sort((a, b) => a.score - b.score);

        if (scored.length < 2) {
            console.log(`  SKIP: only ${scored.length} good images (need 2+)`);
            skipped++;
            continue;
        }

        // Find best time pair ~3 years apart
        const pair = findBestTimePair(scored);
        if (!pair || pair.gapYears < 1.5) {
            console.log(`  SKIP: gap ${pair ? pair.gapYears.toFixed(1) + 'y' : 'none'} (need 1.5y+)`);
            skipped++;
            continue;
        }

        const beforeAnchor = pair.early.img.captured_at;
        const afterAnchor = pair.late.img.captured_at;

        // Gather candidates within 1 year of each anchor, then pick diverse angles
        const beforeCandidates = scored.filter((s) => Math.abs(s.img.captured_at - beforeAnchor) <= ONE_YEAR_MS);
        const afterCandidates = scored.filter((s) => Math.abs(s.img.captured_at - afterAnchor) <= ONE_YEAR_MS);

        const beforePicked = pickDiverseImages(beforeCandidates, 3); // up to 3 per quadrant = max 12
        const afterPicked = pickDiverseImages(afterCandidates, 3);

        const beforeImages = beforePicked.map((s) => formatImage(s.img, lat, lng));
        const afterImages = afterPicked.map((s) => formatImage(s.img, lat, lng));

        if (beforeImages.length === 0 || afterImages.length === 0) {
            console.log(`  SKIP: empty gallery`);
            skipped++;
            continue;
        }

        const bDirs = new Set(beforePicked.map((s) => s.quadrant)).size;
        const aDirs = new Set(afterPicked.map((s) => s.quadrant)).size;
        const bestDist = scored[0].dist;
        const bestAngle = scored[0].facingError;

        console.log(`  ${allImages.length} total -> ${scored.length} good (<=45deg, <=50m)`);
        console.log(`  Best: ${bestDist}m, ${bestAngle}deg | Gap: ${pair.gapYears.toFixed(1)}y`);
        console.log(`  Before: ${beforeImages[0].date} (${beforeImages.length} imgs, ${bDirs} dirs) | After: ${afterImages[0].date} (${afterImages.length} imgs, ${aDirs} dirs)`);

        updatedData.push({
            ...poi,
            overture_date: new Date().toISOString().split('T')[0],
            before_gallery: beforeImages,
            after_gallery: afterImages,
        });

        await new Promise((r) => setTimeout(r, 200));
    }

    fs.writeFileSync(DATA_PATH, JSON.stringify(updatedData, null, 2));

    // Summary stats
    const gaps = updatedData.map((d) => {
        const b = new Date(d.before_gallery[0].date).getTime();
        const a = new Date(d.after_gallery[0].date).getTime();
        return (a - b) / (365.25 * 24 * 60 * 60 * 1000);
    });
    const totalImgs = updatedData.reduce((t, d) => t + d.before_gallery.length + d.after_gallery.length, 0);
    const avgImgs = (totalImgs / updatedData.length).toFixed(1);
    const avgGap = (gaps.reduce((a, b) => a + b, 0) / gaps.length).toFixed(1);

    console.log(`\n========================================`);
    console.log(`DONE: ${updatedData.length} locations, ${skipped} skipped`);
    console.log(`Images: ${totalImgs} total, ~${avgImgs} per location`);
    console.log(`Time gaps: avg ${avgGap}y, min ${Math.min(...gaps).toFixed(1)}y, max ${Math.max(...gaps).toFixed(1)}y`);
    console.log(`========================================`);
}

main().catch(console.error);
