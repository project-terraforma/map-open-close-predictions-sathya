/**
 * Cross-reference Overture fast food locations with Foursquare Places API.
 * Uses v2 Venues API with Client ID + Client Secret (free tier).
 *
 * Usage: node scripts/fetch_foursquare.mjs
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OVERTURE_PATH = path.join(__dirname, 'overture_candidates.json');
const OUTPUT_PATH = path.join(__dirname, 'overture_candidates.json'); // enrich in place

// Credentials from Foursquare Developer Console (visible in project settings)
const CLIENT_ID = 'AKAFWGBHXBQ335KTR5AJJVOEUPDEAOCUKX4OYBTI2YJETQLW';
const CLIENT_SECRET = 'GKYFIXVL4OVV0AA1Z2LEOAA0ZICIDL35WPLOI33ATWK4UELV';
const API_VERSION = '20240101';

// ============================================================
// FOURSQUARE v2 — Venue Search
// ============================================================

async function searchFoursquare(lat, lng, query) {
    const params = new URLSearchParams({
        ll: `${lat},${lng}`,
        query,
        radius: 100,
        limit: '5',
        client_id: CLIENT_ID,
        client_secret: CLIENT_SECRET,
        v: API_VERSION,
    });

    const url = `https://api.foursquare.com/v2/venues/search?${params}`;
    const res = await fetch(url);

    if (!res.ok) {
        const text = await res.text();
        console.error(`  Foursquare API error ${res.status}: ${text.slice(0, 200)}`);
        return [];
    }

    const json = await res.json();
    if (json.meta?.code !== 200) {
        console.error(`  Foursquare error: ${json.meta?.errorDetail || 'unknown'}`);
        return [];
    }

    return json.response?.venues || [];
}

// ============================================================
// MATCHING — does Foursquare agree with Overture?
// ============================================================

function normalizeChain(name) {
    return name
        .toLowerCase()
        .replace(/['']/g, '')
        .replace(/[^a-z0-9 ]/g, '')
        .replace(/\s+/g, ' ')
        .trim();
}

function chainsMatch(overtureName, fsqName) {
    const on = normalizeChain(overtureName);
    const fn = normalizeChain(fsqName);

    // Direct name overlap
    if (on.includes(fn) || fn.includes(on)) return true;

    // Check first word (brand name) — "McDonalds" vs "McDonald's"
    const oFirst = on.split(' ')[0];
    const fFirst = fn.split(' ')[0];
    if (oFirst.length >= 3 && fFirst.length >= 3) {
        if (oFirst.includes(fFirst) || fFirst.includes(oFirst)) return true;
    }

    return false;
}

function classifyMatch(poi, venues) {
    if (venues.length === 0) {
        return { status: 'no_data', detail: 'Foursquare has no results nearby', fsq: null };
    }

    // Find best match by name
    for (const venue of venues) {
        if (chainsMatch(poi.name, venue.name)) {
            // Check if venue is marked as closed
            const isClosed = venue.closed === true;
            if (isClosed) {
                return {
                    status: 'closed',
                    detail: `Foursquare reports "${venue.name}" as closed`,
                    fsq: formatVenue(venue),
                };
            }
            return {
                status: 'verified',
                detail: `Foursquare confirms "${venue.name}" at this location`,
                fsq: formatVenue(venue),
            };
        }
    }

    // No name match — different business may be there
    const topResult = venues[0];
    return {
        status: 'mismatch',
        detail: `Foursquare shows "${topResult.name}" instead`,
        fsq: formatVenue(topResult),
    };
}

function formatVenue(venue) {
    return {
        fsq_id: venue.id,
        name: venue.name,
        category: venue.categories?.[0]?.name || null,
        closed: venue.closed || false,
        website: venue.url || null,
        phone: venue.contact?.formattedPhone || venue.contact?.phone || null,
        address: venue.location?.formattedAddress?.join(', ') || null,
    };
}

// ============================================================
// MAIN
// ============================================================

async function main() {
    console.log('Loading Overture locations...');
    // Re-extract clean data first (remove any previous foursquare fields)
    const locations = JSON.parse(fs.readFileSync(OVERTURE_PATH, 'utf-8'))
        .map(({ foursquare, ...rest }) => rest);
    console.log(`${locations.length} locations to cross-reference\n`);

    let verified = 0, closed = 0, mismatch = 0, noData = 0;

    for (let i = 0; i < locations.length; i++) {
        const poi = locations[i];
        const [lng, lat] = poi.location;
        console.log(`[${i + 1}/${locations.length}] ${poi.name} - ${poi.address}`);

        const venues = await searchFoursquare(lat, lng, poi.name);
        const match = classifyMatch(poi, venues);

        poi.foursquare = {
            status: match.status,
            detail: match.detail,
            ...(match.fsq && { match: match.fsq }),
        };

        const statusIcon = { verified: 'OK', closed: 'CLOSED', mismatch: 'DIFF', no_data: '??' }[match.status];
        console.log(`  [${statusIcon}] ${match.detail}`);

        if (match.status === 'verified') verified++;
        else if (match.status === 'closed') closed++;
        else if (match.status === 'mismatch') mismatch++;
        else noData++;

        // Respect rate limits
        await new Promise((r) => setTimeout(r, 100));
    }

    fs.writeFileSync(OUTPUT_PATH, JSON.stringify(locations, null, 2));

    console.log(`\n========================================`);
    console.log(`DONE: ${locations.length} locations cross-referenced`);
    console.log(`  Verified: ${verified}`);
    console.log(`  Closed:   ${closed}`);
    console.log(`  Mismatch: ${mismatch}`);
    console.log(`  No data:  ${noData}`);
    console.log(`========================================`);
    console.log(`\nEnriched data saved to ${OUTPUT_PATH}`);
    console.log(`Next: run "node scripts/fetch_360.mjs" to rebuild with Foursquare data`);
}

main().catch(console.error);
