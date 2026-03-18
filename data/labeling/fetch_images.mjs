/**
 * Fetch Mapillary street-level images for selected businesses.
 * Simpler than fetch_mapillary.mjs — just gets current best images, no before/after pairs.
 *
 * Usage: node scripts/fetch_images.mjs
 * Input:  scripts/selected_businesses.json
 * Output: src/data/test_data.json
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Accept --input and --output args for different city files
let inputFile = path.join(__dirname, 'selected_businesses.json');
let outputFile = path.join(__dirname, '..', 'src', 'data', 'test_data.json');
const inputIdx = process.argv.indexOf('--input');
if (inputIdx !== -1 && process.argv[inputIdx + 1]) {
    inputFile = path.resolve(process.argv[inputIdx + 1]);
}
const outputIdx = process.argv.indexOf('--output');
if (outputIdx !== -1 && process.argv[outputIdx + 1]) {
    outputFile = path.resolve(process.argv[outputIdx + 1]);
}
const INPUT_PATH = inputFile;
const OUTPUT_PATH = outputFile;
const ACCESS_TOKEN = 'MLY|25120378337641785|21245babc5f6905ed3da857aba13bc87';

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
// MAPILLARY API
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
// SCORING
// ============================================================

function scoreImage(img, poiLat, poiLng) {
    const [imgLng, imgLat] = img.geometry.coordinates;
    const dist = haversineDistance(poiLat, poiLng, imgLat, imgLng);
    const camAngle = img.computed_compass_angle ?? img.compass_angle;
    const bearingToPoi = bearing(imgLat, imgLng, poiLat, poiLng);
    const facingError = angleDiff(camAngle, bearingToPoi);
    const approachBearing = (bearingToPoi + 180) % 360;
    const quadrant = Math.floor(approachBearing / 90) % 4;

    return {
        score: facingError * 3 + dist,
        dist: Math.round(dist),
        facingError: Math.round(facingError),
        quadrant,
    };
}

function pickDiverseImages(candidates, maxPerQuadrant = 2) {
    const buckets = [[], [], [], []];
    for (const c of candidates) buckets[c.quadrant].push(c);
    const picked = [];
    for (const bucket of buckets) picked.push(...bucket.slice(0, maxPerQuadrant));
    return picked;
}

// ============================================================
// MAIN
// ============================================================

async function main() {
    console.log('Loading selected businesses...');
    const locations = JSON.parse(fs.readFileSync(INPUT_PATH, 'utf-8'));
    console.log(`${locations.length} locations to fetch images for\n`);

    let success = 0, skipped = 0;

    for (let i = 0; i < locations.length; i++) {
        const poi = locations[i];
        const [lng, lat] = poi.location;
        console.log(`[${i + 1}/${locations.length}] ${poi.name} - ${poi.address}`);

        // Fetch images at increasing radii
        let allImages = [];
        for (const radius of [25, 40, 60, 80]) {
            allImages = await fetchImages(lng, lat, radius);
            if (allImages.length >= 10) break;
            console.log(`  ${allImages.length} imgs at ${radius}m, widening...`);
            await new Promise((r) => setTimeout(r, 100));
        }

        if (allImages.length === 0) {
            console.log(`  SKIP: no images found`);
            poi.current_gallery = [];
            skipped++;
            continue;
        }

        // Score and filter: facing within 60deg, within 60m
        const scored = allImages
            .filter((img) => (img.computed_compass_angle ?? img.compass_angle) != null)
            .map((img) => {
                const s = scoreImage(img, lat, lng);
                const panoPenalty = img.is_pano ? 40 : 0;
                return { img, ...s, score: s.score + panoPenalty };
            })
            .filter((s) => s.facingError <= 60 && s.dist <= 60)
            .sort((a, b) => a.score - b.score);

        if (scored.length === 0) {
            console.log(`  SKIP: no well-facing images`);
            poi.current_gallery = [];
            skipped++;
            continue;
        }

        // Pick diverse angles, up to 8 images
        const picked = pickDiverseImages(scored, 2).slice(0, 8);

        poi.current_gallery = picked.map((s) => {
            const [imgLng, imgLat] = s.img.geometry.coordinates;
            const dist = Math.round(haversineDistance(lat, lng, imgLat, imgLng));
            const date = new Date(s.img.captured_at).toISOString().split('T')[0];
            return {
                url: s.img.thumb_2048_url,
                date,
                distance_m: dist,
                group: 'sequential',
            };
        });

        // Update overture_raw with image metadata
        if (poi.overture_raw) {
            const dates = picked.map((s) => s.img.captured_at);
            const avgAge = dates.reduce((sum, d) => sum + (Date.now() - d), 0) / dates.length;
            poi.overture_raw.image_age_days = Math.round(avgAge / (24 * 60 * 60 * 1000));
            poi.overture_raw.num_images = picked.length;
        }

        const dirs = new Set(picked.map((s) => s.quadrant)).size;
        console.log(`  ${allImages.length} total -> ${scored.length} good -> ${picked.length} picked (${dirs} dirs)`);
        success++;

        await new Promise((r) => setTimeout(r, 200));
    }

    // Write output
    fs.writeFileSync(OUTPUT_PATH, JSON.stringify(locations, null, 2));

    console.log(`\n========================================`);
    console.log(`DONE: ${success} locations with images, ${skipped} skipped`);
    console.log(`Output: ${OUTPUT_PATH}`);
    console.log(`========================================`);
}

main().catch(console.error);
