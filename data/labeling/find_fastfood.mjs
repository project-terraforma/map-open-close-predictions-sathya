// Query Overpass API for real fast food restaurant locations in San Francisco
const query = `
[out:json][timeout:30];
area["name"="San Francisco"]["admin_level"="6"]->.sf;
node["amenity"="fast_food"]["name"](area.sf);
out body;
`;

const res = await fetch('https://overpass-api.de/api/interpreter', {
    method: 'POST',
    body: 'data=' + encodeURIComponent(query),
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
});

const data = await res.json();
const places = data.elements.map(e => ({
    name: e.tags.name,
    brand: e.tags.brand || '',
    lat: e.lat,
    lon: e.lon,
    addr: [e.tags['addr:housenumber'], e.tags['addr:street']].filter(Boolean).join(' ') || 'San Francisco',
    cuisine: e.tags.cuisine || ''
}));

const chains = [
    'McDonald', 'Burger King', 'Wendy', 'Taco Bell', 'KFC',
    'Jack in the Box', 'Chick-fil-A', 'Popeyes', 'Subway',
    'Panda Express', 'Five Guys', 'Chipotle', 'In-N-Out',
    "Carl's Jr", 'Wingstop', 'Pizza Hut', "Church's Chicken",
    'Jollibee', 'Shake Shack', 'Del Taco', "Arby's", 'Domino',
    'Little Caesars', 'El Pollo Loco'
];

const fastFood = places.filter(p =>
    chains.some(c =>
        p.name.toLowerCase().includes(c.toLowerCase()) ||
        p.brand.toLowerCase().includes(c.toLowerCase())
    )
);

console.log('Total fast food nodes in SF:', places.length);
console.log('Known chains found:', fastFood.length);
console.log(JSON.stringify(fastFood, null, 2));
