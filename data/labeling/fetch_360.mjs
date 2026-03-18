import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_PATH = path.join(__dirname, '..', 'src', 'data', 'mock_data.json');
const OVERTURE_PATH = path.join(__dirname, 'overture_fastfood.json');
const ACCESS_TOKEN = 'MLY|25120378337641785|21245babc5f6905ed3da857aba13bc87';

// ── Minimum image requirements ──
const MIN_TOTAL_IMAGES = 6;        // minimum images per location
const MIN_PER_SIDE = 3;            // minimum per side for intersections
const MAX_PER_SIDE = 5;            // max per side for intersections
const MAX_MIDBLOCK = 10;           // max for mid-block sequential

// ============================================================
// Step 1: Load locations
// ============================================================

function loadLocations() {
    // If mock_data.json exists with galleries, load it to preserve existing data
    // Otherwise fall back to overture
    if (fs.existsSync(DATA_PATH)) {
        const raw = fs.readFileSync(DATA_PATH, 'utf-8');
        const data = JSON.parse(raw);
        if (data.length > 0 && data[0].current_gallery) {
            console.log(`Loading existing ${data.length} locations from mock_data.json`);
            return { locations: data, isExisting: true };
        }
    }
    console.log('Loading Overture fast food locations...');
    const raw = fs.readFileSync(OVERTURE_PATH, 'utf-8');
    const locations = JSON.parse(raw);
    console.log(`Loaded ${locations.length} locations from Overture.\n`);
    return { locations, isExisting: false };
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

function reducedAngleDiff(a, b) {
    let diff = Math.abs(a - b) % 180;
    return diff > 90 ? 180 - diff : diff;
}

// ============================================================
// MAPILLARY API
// ============================================================

async function fetchImages(lng, lat, radiusMeters) {
    const delta = radiusMeters / 111000;
    const bbox = `${lng - delta},${lat - delta},${lng + delta},${lat + delta}`;
    const fields = 'id,thumb_2048_url,captured_at,geometry,compass_angle,computed_compass_angle,is_pano,sequence';
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
// TEMPORAL CLUSTERING
// ============================================================

function clusterByTime(scoredImages, maxGapDays = 365) {
    if (scoredImages.length === 0) return [];
    const sorted = [...scoredImages].sort((a, b) => a.img.captured_at - b.img.captured_at);

    const clusters = [];
    let current = [sorted[0]];

    for (let i = 1; i < sorted.length; i++) {
        const gapMs = sorted[i].img.captured_at - sorted[i - 1].img.captured_at;
        const gapDays = gapMs / (1000 * 60 * 60 * 24);
        if (gapDays > maxGapDays) {
            clusters.push(current);
            current = [];
        }
        current.push(sorted[i]);
    }
    clusters.push(current);
    return clusters;
}

function computeDateRange(images) {
    if (images.length < 2) return 0;
    const dates = images.map((s) => s.img.captured_at);
    return Math.round((Math.max(...dates) - Math.min(...dates)) / (1000 * 60 * 60 * 24));
}

function selectTemporalCluster(deduped) {
    const clusters = clusterByTime(deduped);
    if (clusters.length <= 1) {
        return { pool: deduped, flag: null, dateRangeDays: computeDateRange(deduped) };
    }

    // Prefer the most recent cluster
    const recentCluster = clusters[clusters.length - 1];

    if (recentCluster.length >= MIN_TOTAL_IMAGES) {
        return { pool: recentCluster, flag: 'recent_cluster', dateRangeDays: computeDateRange(recentCluster) };
    }

    // Merge backwards until we have enough
    let merged = [...recentCluster];
    let flag = 'recent_cluster';
    for (let c = clusters.length - 2; c >= 0 && merged.length < MIN_TOTAL_IMAGES; c--) {
        merged = [...clusters[c], ...merged];
        flag = 'widened';
    }

    return { pool: merged, flag, dateRangeDays: computeDateRange(merged) };
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

    const approachBearing = bearing(poiLat, poiLng, imgLat, imgLng);

    return {
        score: facingError * 5 + dist,
        dist: Math.round(dist),
        facingError: Math.round(facingError),
        approachBearing,
    };
}

// ============================================================
// DEDUPLICATION
// ============================================================

function deduplicateByPosition(scored, minDist = 5) {
    const kept = [];
    for (const s of scored) {
        const [sLng, sLat] = s.img.geometry.coordinates;
        const isDupe = kept.some((k) => {
            const [kLng, kLat] = k.img.geometry.coordinates;
            return haversineDistance(sLat, sLng, kLat, kLng) < minDist;
        });
        if (!isDupe) kept.push(s);
    }
    return kept;
}

// ============================================================
// LOCATION TYPE
// ============================================================

function detectLocationType(scoredImages) {
    if (scoredImages.length < 3) {
        return { isIntersection: false, directions: [0] };
    }

    const reduced = scoredImages.map((s) => s.approachBearing % 180);
    const bins = new Array(12).fill(0);
    reduced.forEach((b) => bins[Math.floor(b / 15) % 12]++);

    const peaks = bins
        .map((count, idx) => ({ idx, count, angle: idx * 15 + 7.5 }))
        .filter((b) => b.count >= 2)
        .sort((a, b) => b.count - a.count);

    if (peaks.length < 2) {
        return { isIntersection: false, directions: [peaks[0]?.angle || 0] };
    }

    const diff = Math.abs(peaks[0].angle - peaks[1].angle);
    const isPerp = diff >= 50 && diff <= 130;

    return {
        isIntersection: isPerp,
        directions: peaks.slice(0, 2).map((p) => p.angle),
    };
}

// ============================================================
// SELECTION — enforce minimums
// ============================================================

function selectMidBlockImages(scoredImages, poiLat, poiLng, locationType, maxImages = MAX_MIDBLOCK) {
    const roadDirDeg = locationType.directions[0] || 0;
    const roadDirRad = (roadDirDeg * Math.PI) / 180;

    const withPos = scoredImages.map((s) => {
        const approachRad = (s.approachBearing * Math.PI) / 180;
        const dist = s.dist;
        const roadPos = dist * Math.cos(approachRad - roadDirRad);
        return { ...s, roadPos, group: 'sequential' };
    });

    withPos.sort((a, b) => a.roadPos - b.roadPos);

    if (withPos.length <= maxImages) return withPos;

    const step = (withPos.length - 1) / (maxImages - 1);
    const result = [];
    for (let i = 0; i < maxImages; i++) {
        result.push(withPos[Math.round(i * step)]);
    }
    return result;
}

function selectIntersectionImages(scoredImages, locationType) {
    const [dir1, dir2] = locationType.directions;

    const sideA = [];
    const sideB = [];

    for (const s of scoredImages) {
        const reduced = s.approachBearing % 180;
        const d1 = reducedAngleDiff(reduced, dir1);
        const d2 = reducedAngleDiff(reduced, dir2);

        if (d1 <= d2) {
            sideA.push({ ...s, group: 'side_a' });
        } else {
            sideB.push({ ...s, group: 'side_b' });
        }
    }

    sideA.sort((a, b) => a.score - b.score);
    sideB.sort((a, b) => a.score - b.score);

    // Enforce minimum per side: at least MIN_PER_SIDE from each, up to MAX_PER_SIDE
    let selectedA = sideA.slice(0, MAX_PER_SIDE);
    let selectedB = sideB.slice(0, MAX_PER_SIDE);

    // If one side is short, take extra from the other to meet total minimum
    const total = selectedA.length + selectedB.length;
    if (total < MIN_TOTAL_IMAGES) {
        const needed = MIN_TOTAL_IMAGES - total;
        if (selectedA.length < MIN_PER_SIDE && sideB.length > MAX_PER_SIDE) {
            selectedB = sideB.slice(0, MAX_PER_SIDE + needed);
        } else if (selectedB.length < MIN_PER_SIDE && sideA.length > MAX_PER_SIDE) {
            selectedA = sideA.slice(0, MAX_PER_SIDE + needed);
        }
    }

    return [...selectedA, ...selectedB];
}

// ============================================================
// FORMAT
// ============================================================

function formatImage(scored, poiLat, poiLng) {
    const img = scored.img;
    const [imgLng, imgLat] = img.geometry.coordinates;
    const dist = Math.round(haversineDistance(poiLat, poiLng, imgLat, imgLng));
    const date = new Date(img.captured_at).toISOString().split('T')[0];
    return {
        url: img.thumb_2048_url,
        date,
        captured_at: img.captured_at,
        distance_m: dist,
        group: scored.group || 'sequential',
    };
}

// ============================================================
// FETCH + SELECT FOR ONE LOCATION (with progressive relaxation)
// ============================================================

async function fetchAndSelectForLocation(poi) {
    const [lng, lat] = poi.location;

    // Progressive widening: try increasing radii
    let allImages = [];
    for (const radius of [40, 60, 90, 120, 160]) {
        allImages = await fetchImages(lng, lat, radius);
        if (allImages.length >= 30) break;
        console.log(`  ${allImages.length} imgs at ${radius}m, widening...`);
        await new Promise((r) => setTimeout(r, 100));
    }

    if (allImages.length === 0) {
        return null;
    }

    // Try with strict filters first, then progressively relax
    const filterConfigs = [
        { facingMax: 60, distMax: 60, dedupDist: 5, label: 'strict' },
        { facingMax: 75, distMax: 80, dedupDist: 4, label: 'relaxed' },
        { facingMax: 90, distMax: 100, dedupDist: 3, label: 'wide' },
    ];

    for (const config of filterConfigs) {
        const scored = allImages
            .filter((img) => !img.is_pano)
            .filter((img) => (img.computed_compass_angle ?? img.compass_angle) != null)
            .map((img) => ({ img, ...scoreImage(img, lat, lng) }))
            .filter((s) => s.facingError <= config.facingMax && s.dist <= config.distMax)
            .sort((a, b) => a.score - b.score);

        const dedupedRaw = deduplicateByPosition(scored, config.dedupDist);

        if (dedupedRaw.length < MIN_TOTAL_IMAGES) {
            if (config.label !== 'wide') {
                console.log(`  Only ${dedupedRaw.length} imgs with ${config.label} filter, relaxing...`);
                continue;
            }
        }

        if (dedupedRaw.length < 3) {
            console.log(`  Only ${dedupedRaw.length} usable imgs even with ${config.label} filter`);
            return null;
        }

        // Temporal clustering: prefer images from the same time period
        const { pool: deduped, flag: temporalFlag, dateRangeDays } = selectTemporalCluster(dedupedRaw);
        if (temporalFlag) {
            console.log(`  Temporal: ${dedupedRaw.length} -> ${deduped.length} imgs (${temporalFlag}, ${dateRangeDays}d range)`);
        }

        if (deduped.length < 3) {
            // If temporal filtering left too few, use all
            console.log(`  Temporal filtering too aggressive, using all ${dedupedRaw.length} imgs`);
        }

        const finalPool = deduped.length >= 3 ? deduped : dedupedRaw;
        const locationType = detectLocationType(finalPool);
        let selected;

        const temporalMeta = { flag: temporalFlag || null, date_range_days: dateRangeDays || 0 };

        if (locationType.isIntersection) {
            selected = selectIntersectionImages(finalPool, locationType);
            const aCount = selected.filter((s) => s.group === 'side_a').length;
            const bCount = selected.filter((s) => s.group === 'side_b').length;

            if (aCount < MIN_PER_SIDE || bCount < MIN_PER_SIDE) {
                if (finalPool.length >= MIN_TOTAL_IMAGES) {
                    selected = selectMidBlockImages(finalPool, lat, lng, locationType, Math.min(finalPool.length, MAX_MIDBLOCK));
                    console.log(`  INTERSECTION->MID-BLOCK (side_a=${aCount}, side_b=${bCount}, need ${MIN_PER_SIDE} each) [${config.label}]`);
                    return {
                        locationType: 'mid-block',
                        gallery: selected.map((s) => formatImage(s, lat, lng)),
                        temporal: temporalMeta,
                        stats: { total: allImages.length, scored: scored.length, deduped: finalPool.length, final: selected.length, filter: config.label },
                    };
                }
            }

            console.log(`  INTERSECTION: ${aCount} side-A + ${bCount} side-B [${config.label}]`);
            return {
                locationType: 'intersection',
                gallery: selected.map((s) => formatImage(s, lat, lng)),
                temporal: temporalMeta,
                stats: { total: allImages.length, scored: scored.length, deduped: finalPool.length, final: selected.length, filter: config.label },
            };
        } else {
            selected = selectMidBlockImages(finalPool, lat, lng, locationType, Math.min(finalPool.length, MAX_MIDBLOCK));
            console.log(`  MID-BLOCK: ${selected.length} sequential images [${config.label}]`);
            return {
                locationType: 'mid-block',
                gallery: selected.map((s) => formatImage(s, lat, lng)),
                temporal: temporalMeta,
                stats: { total: allImages.length, scored: scored.length, deduped: finalPool.length, final: selected.length, filter: config.label },
            };
        }
    }

    return null;
}

// ============================================================
// MAIN — refetch only locations that need more images
// ============================================================

async function main() {
    const { locations, isExisting } = loadLocations();
    let refetched = 0;
    let skipped = 0;
    let alreadyGood = 0;

    for (let i = 0; i < locations.length; i++) {
        const poi = locations[i];
        const gallery = poi.current_gallery || [];
        const sideACount = gallery.filter((g) => g.group === 'side_a').length;
        const sideBCount = gallery.filter((g) => g.group === 'side_b').length;
        const isIntersection = poi.location_type === 'intersection';

        // Check if this location needs refetching
        const needsRefetch =
            gallery.length < MIN_TOTAL_IMAGES ||
            (isIntersection && (sideACount < MIN_PER_SIDE || sideBCount < MIN_PER_SIDE)) ||
            !isExisting;

        if (!needsRefetch) {
            alreadyGood++;
            continue;
        }

        console.log(`[${i + 1}/${locations.length}] ${poi.name} - ${poi.address} (currently ${gallery.length} imgs, a=${sideACount} b=${sideBCount})`);

        const result = await fetchAndSelectForLocation(poi);

        if (!result) {
            console.log(`  SKIP: not enough images found`);
            skipped++;
            continue;
        }

        // Update location
        poi.location_type = result.locationType;
        poi.current_gallery = result.gallery;
        poi.temporal = result.temporal || { flag: null, date_range_days: 0 };

        // Clear old vision data since images changed
        delete poi.vision;

        const s = result.stats;
        console.log(`  ${s.total} total -> ${s.scored} scored -> ${s.deduped} deduped -> ${s.final} final [${s.filter}]`);

        refetched++;
        await new Promise((r) => setTimeout(r, 200));
    }

    // Save
    fs.writeFileSync(DATA_PATH, JSON.stringify(locations, null, 2));

    // Summary
    const allGalleries = locations.filter((l) => l.current_gallery?.length > 0);
    const totalImgs = allGalleries.reduce((t, d) => t + d.current_gallery.length, 0);
    const gallerySizes = allGalleries.map((d) => d.current_gallery.length);
    const minSize = Math.min(...gallerySizes);
    const maxSize = Math.max(...gallerySizes);
    const avgSize = (totalImgs / allGalleries.length).toFixed(1);

    const underMin = allGalleries.filter((l) => l.current_gallery.length < MIN_TOTAL_IMAGES).length;
    const intersections = allGalleries.filter((l) => l.location_type === 'intersection');
    const lowSides = intersections.filter((l) => {
        const a = l.current_gallery.filter((g) => g.group === 'side_a').length;
        const b = l.current_gallery.filter((g) => g.group === 'side_b').length;
        return a < MIN_PER_SIDE || b < MIN_PER_SIDE;
    });

    console.log(`\n========================================`);
    console.log(`DONE: ${allGalleries.length} locations with galleries`);
    console.log(`  Refetched: ${refetched}, Already good: ${alreadyGood}, Skipped: ${skipped}`);
    console.log(`  Images: ${totalImgs} total, min=${minSize}, max=${maxSize}, avg=${avgSize}`);
    console.log(`  Under minimum (${MIN_TOTAL_IMAGES}): ${underMin} locations`);
    console.log(`  Intersections with low sides: ${lowSides.length}`);
    console.log(`========================================`);
}

main().catch(console.error);
