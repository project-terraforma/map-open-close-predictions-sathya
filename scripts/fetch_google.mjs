import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_PATH = path.join(__dirname, '..', 'src', 'data', 'mock_data.json');
const IMG_DIR = path.join(__dirname, '..', 'public', 'streetview');
const API_KEY = 'AIzaSyA73WD0wnYgO4mSBAPd-z_sLC0Fn2TzYMY';

const THREE_YEARS_MS = 3 * 365.25 * 24 * 60 * 60 * 1000;

// ============================================================
// Step 1: Fetch fast food locations from OpenStreetMap
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
// GEOMETRY
// ============================================================

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

// ============================================================
// GOOGLE STREET VIEW API
// ============================================================

// Get metadata: actual camera position + date for a location
async function getStreetViewMeta(lat, lng) {
    const url = `https://maps.googleapis.com/maps/api/streetview/metadata?location=${lat},${lng}&source=outdoor&key=${API_KEY}`;
    const res = await fetch(url);
    const data = await res.json();
    if (data.status !== 'OK') return null;
    return data; // { date, location: {lat, lng}, pano_id, status }
}

// Download a Street View image and save locally
async function downloadStreetViewImage(lat, lng, heading, fov, filename) {
    const url = `https://maps.googleapis.com/maps/api/streetview?size=640x640&location=${lat},${lng}&heading=${heading}&pitch=5&fov=${fov}&source=outdoor&key=${API_KEY}`;
    const res = await fetch(url);

    if (!res.ok) return null;

    const buffer = Buffer.from(await res.arrayBuffer());

    // Check if Google returned a "no imagery" placeholder (very small file)
    if (buffer.length < 5000) return null;

    const filepath = path.join(IMG_DIR, filename);
    fs.writeFileSync(filepath, buffer);
    return `/streetview/${filename}`;
}

// ============================================================
// MAIN
// ============================================================

async function main() {
    // Create output directory
    if (!fs.existsSync(IMG_DIR)) fs.mkdirSync(IMG_DIR, { recursive: true });

    const FAST_FOOD = await fetchFastFoodLocations();
    const updatedData = [];
    let skipped = 0;
    let totalImages = 0;

    for (let i = 0; i < FAST_FOOD.length; i++) {
        const poi = FAST_FOOD[i];
        const [lng, lat] = poi.location;
        console.log(`[${i + 1}/${FAST_FOOD.length}] ${poi.name} - ${poi.address}`);

        // Step 1: Get street view metadata to find the actual camera position
        const meta = await getStreetViewMeta(lat, lng);
        if (!meta) {
            console.log(`  SKIP: no street view coverage`);
            skipped++;
            await new Promise((r) => setTimeout(r, 100));
            continue;
        }

        const camLat = meta.location.lat;
        const camLng = meta.location.lng;
        const captureDate = meta.date || 'unknown'; // "YYYY-MM" format

        // Step 2: Calculate heading from camera directly at the business
        const headingToBusiness = bearing(camLat, camLng, lat, lng);

        console.log(`  Camera at ${camLat.toFixed(5)}, ${camLng.toFixed(5)} | Date: ${captureDate}`);
        console.log(`  Heading to business: ${headingToBusiness.toFixed(0)}deg`);

        // Step 3: Download images from multiple angles
        // - Direct view (heading straight at the building)
        // - Left offset (-40deg) for wider context
        // - Right offset (+40deg) for wider context
        // - Wide FOV shot for full building context
        const slug = poi.name.toLowerCase().replace(/[^a-z0-9]/g, '-').replace(/-+/g, '-');
        const angles = [
            { suffix: 'front', heading: headingToBusiness, fov: 80 },
            { suffix: 'left', heading: (headingToBusiness - 40 + 360) % 360, fov: 80 },
            { suffix: 'right', heading: (headingToBusiness + 40) % 360, fov: 80 },
            { suffix: 'wide', heading: headingToBusiness, fov: 120 },
        ];

        const gallery = [];
        for (const angle of angles) {
            const filename = `${slug}-${poi.id}-${angle.suffix}.jpg`;
            const localPath = await downloadStreetViewImage(camLat, camLng, angle.heading, angle.fov, filename);
            if (localPath) {
                gallery.push({
                    url: localPath,
                    date: captureDate,
                    distance_m: 0,
                    view: angle.suffix,
                });
                totalImages++;
            }
            await new Promise((r) => setTimeout(r, 50));
        }

        if (gallery.length === 0) {
            console.log(`  SKIP: all image downloads failed`);
            skipped++;
            continue;
        }

        console.log(`  Downloaded ${gallery.length} views: ${gallery.map((g) => g.view).join(', ')}`);

        // Split into before/after galleries
        // Google only gives us the latest pano, so we put the direct views in "after"
        // and offset views in "before" (different angles of the same time)
        // For a real before/after, you'd combine with Mapillary historical data
        const midIdx = Math.max(1, Math.floor(gallery.length / 2));

        updatedData.push({
            ...poi,
            overture_date: new Date().toISOString().split('T')[0],
            before_gallery: gallery.slice(0, midIdx),
            after_gallery: gallery.slice(midIdx),
        });

        await new Promise((r) => setTimeout(r, 200));
    }

    fs.writeFileSync(DATA_PATH, JSON.stringify(updatedData, null, 2));

    console.log(`\n========================================`);
    console.log(`DONE: ${updatedData.length} locations, ${skipped} skipped`);
    console.log(`Images: ${totalImages} total downloaded to public/streetview/`);
    console.log(`========================================`);
}

main().catch(console.error);
