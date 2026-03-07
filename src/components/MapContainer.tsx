import { useState, useEffect, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { MapContainer as LeafletMap, TileLayer, Marker, useMap } from 'react-leaflet';
import L from 'leaflet';
import { renderToStaticMarkup } from 'react-dom/server';
import testDataSf from '../data/test_data.json';
import testDataLa from '../data/test_data_la.json';
import testDataChicago from '../data/test_data_chicago.json';
import testDataMiami from '../data/test_data_miami.json';

// ============================================================
// CITY CONFIG
// ============================================================
const CITIES: Record<string, { label: string; center: [number, number]; zoom: number; data: any[] }> = {
    sf: { label: 'San Francisco', center: [37.7749, -122.4194], zoom: 12, data: testDataSf as any[] },
    la: { label: 'Los Angeles', center: [34.0522, -118.2437], zoom: 11, data: testDataLa as any[] },
    chicago: { label: 'Chicago', center: [41.8781, -87.6298], zoom: 12, data: testDataChicago as any[] },
    miami: { label: 'Miami', center: [25.7617, -80.1918], zoom: 12, data: testDataMiami as any[] },
};

function MapFlyTo({ center, zoom }: { center: [number, number]; zoom: number }) {
    const map = useMap();
    useEffect(() => {
        map.flyTo(center, zoom, { duration: 1.5 });
    }, [center, zoom, map]);
    return null;
}
import {
    MapPin, X, Calendar, Navigation, Eye, ShieldCheck,
    Clock, Database, Crosshair, ArrowRight, CheckCircle,
    AlertTriangle, XCircle, HelpCircle, Globe, Phone, Type,
    Layers, ImageIcon, ChevronLeft, ChevronRight,
    ScanEye, ExternalLink, BarChart3
} from 'lucide-react';

// ============================================================
// PIN ICONS
// ============================================================
const createCustomIcon = (color: string) => {
    const iconHtml = renderToStaticMarkup(
        <div style={{ color, filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.4))' }}>
            <MapPin size={30} fill={color} strokeWidth={1.5} className="text-white" />
        </div>
    );
    return L.divIcon({
        html: iconHtml,
        className: 'custom-pin-icon',
        iconSize: [30, 30],
        iconAnchor: [15, 30],
        popupAnchor: [0, -30]
    });
};

const pinOpen = createCustomIcon('#22c55e');
const pinNotOpen = createCustomIcon('#ef4444');
const pinUncertain = createCustomIcon('#f59e0b');
const pinUnknown = createCustomIcon('#6b7280');

function getPinIcon(poi: any) {
    const p = poi.vision?.prediction;
    if (p === 'open') return pinOpen;
    if (p === 'not_open') return pinNotOpen;
    if (p === 'uncertain') return pinUncertain;
    return pinUnknown;
}

// ============================================================
// GALLERY
// ============================================================
interface GalleryImage {
    url: string;
    date: string;
    distance_m: number;
    group?: string;
}

function GalleryStrip({ images, label, accent }: { images: GalleryImage[]; label: string; accent: string }) {
    return (
        <div className="panel-content-enter">
            <div className="flex items-center justify-center w-full mt-2">
                <div style={{ backgroundColor: '#9333ea', borderColor: '#6b21a8' }} className="w-full py-2.5 rounded-lg border shadow-sm flex items-center justify-center">
                    <span style={{ fontWeight: 900 }} className="text-[12px] text-white font-mono tracking-wide uppercase text-center block w-full">
                        {images.length} images around the business
                    </span>
                </div>
            </div>
            <div className="gallery-scroll">
                {images.map((img, i) => (
                    <div key={i} className="gallery-item" onClick={() => window.open(img.url, '_blank')}>
                        <img src={img.url} alt={`${label} ${i + 1}`} loading="lazy" />
                        <div className="gallery-badge">
                            <span><Calendar size={9} className="inline mr-0.5" />{img.date}</span>
                            <span><Navigation size={9} className="inline mr-0.5" />{img.distance_m}m</span>
                        </div>
                    </div>
                ))}
            </div>
            <p className="text-[10px] text-slate-600 mt-1.5 flex items-center justify-center gap-1 text-center">
                <ExternalLink size={9} /> Click to view full size
            </p>
        </div>
    );
}

// ============================================================
// SMALL COMPONENTS
// ============================================================
function PredictionPill({ prediction }: { prediction: string }) {
    const config: Record<string, { label: string; bg: string; text: string; border: string; icon: any }> = {
        open: { label: 'OPEN', bg: 'rgba(34,197,94,0.12)', text: '#4ade80', border: 'rgba(34,197,94,0.25)', icon: CheckCircle },
        not_open: { label: 'NOT OPEN', bg: 'rgba(239,68,68,0.12)', text: '#f87171', border: 'rgba(239,68,68,0.25)', icon: XCircle },
        uncertain: { label: 'UNCERTAIN', bg: 'rgba(245,158,11,0.12)', text: '#fbbf24', border: 'rgba(245,158,11,0.25)', icon: AlertTriangle },
    };
    const c = config[prediction] || { label: 'UNKNOWN', bg: 'rgba(100,116,139,0.12)', text: '#94a3b8', border: 'rgba(100,116,139,0.25)', icon: HelpCircle };
    const Icon = c.icon;
    return (
        <span
            className="inline-flex items-center gap-1 text-[10px] font-bold px-2.5 py-1 rounded-lg tracking-wide font-mono"
            style={{ backgroundColor: c.bg, color: c.text, border: `1px solid ${c.border}` }}
        >
            <Icon size={10} /> {c.label}
        </span>
    );
}

function ScoreBar({ score, max = 1.0, color }: { score: number; max?: number; color: string }) {
    const pct = Math.min((score / max) * 100, 100);
    return (
        <div className="w-full h-1 bg-white/[0.04] rounded-full overflow-hidden">
            <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, backgroundColor: color }} />
        </div>
    );
}

function GroundTruthBadge({ poi }: { poi: any }) {
    const gt = poi.ground_truth;
    const prediction = poi.vision?.prediction;
    if (!gt) return null;

    const predIsOpen = prediction === 'open';
    const gtIsOpen = gt === 'open';
    const isUncertain = prediction === 'uncertain' || prediction === 'unknown';
    const isCorrect = predIsOpen === gtIsOpen || (prediction === 'not_open' && !gtIsOpen);

    // If prediction is uncertain/unknown, show that instead of wrong
    if (isUncertain) {
        return (
            <span className="inline-flex items-center gap-1 text-[9px] font-bold px-1.5 py-0.5 rounded bg-yellow-500/15 text-yellow-400 ring-1 ring-yellow-500/20">
                <AlertTriangle size={8} />
                UNSURE
            </span>
        );
    }

    return (
        <span className={`inline-flex items-center gap-1 text-[9px] font-bold px-1.5 py-0.5 rounded ${isCorrect
            ? 'bg-green-500/15 text-green-400 ring-1 ring-green-500/20'
            : 'bg-red-500/15 text-red-400 ring-1 ring-red-500/20'
            }`}>
            {isCorrect ? <CheckCircle size={8} /> : <XCircle size={8} />}
            {isCorrect ? 'CORRECT' : 'WRONG'}
        </span>
    );
}

function VerificationBadge({ status }: { status: string | undefined }) {
    if (!status) return null;
    const config: Record<string, { icon: any; label: string; color: string }> = {
        verified: { icon: CheckCircle, label: 'Verified', color: '#10b981' },
        closed: { icon: XCircle, label: 'Closed', color: '#ef4444' },
        mismatch: { icon: AlertTriangle, label: 'Mismatch', color: '#f59e0b' },
        no_data: { icon: HelpCircle, label: 'Unverified', color: '#64748b' },
    };
    const c = config[status] || config.no_data;
    const Icon = c.icon;
    return (
        <span
            className="inline-flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-md"
            style={{ backgroundColor: `${c.color}15`, color: c.color, border: `1px solid ${c.color}25` }}
        >
            <Icon size={9} /> {c.label}
        </span>
    );
}

// ============================================================
// LAYER INDICATOR
// ============================================================
const LAYER_CONFIG: Record<string, { label: string; icon: any; color: string }> = {
    text: { label: 'TEXT', icon: Type, color: '#10b981' },
    xgboost: { label: 'METADATA', icon: Database, color: '#3b82f6' },
};

function LayerIndicator({ primaryLayer }: { primaryLayer: string }) {
    return (
        <div className="flex gap-1.5">
            {Object.entries(LAYER_CONFIG).map(([key, cfg]) => {
                const isPrimary = key === primaryLayer;
                const Icon = cfg.icon;
                return (
                    <div
                        key={key}
                        className={`flex items-center gap-1 px-2 py-1 rounded-md text-[9px] font-bold tracking-wide transition-all ${isPrimary ? 'ring-1' : 'opacity-40'}`}
                        style={{
                            backgroundColor: `${cfg.color}${isPrimary ? '18' : '08'}`,
                            color: cfg.color,
                            ...(isPrimary ? { boxShadow: `0 0 0 1px ${cfg.color}40` } : {}),
                        }}
                    >
                        <Icon size={10} />
                        {cfg.label}
                        {isPrimary && <span className="ml-0.5 text-[8px] opacity-80">PRIMARY</span>}
                    </div>
                );
            })}
        </div>
    );
}

// ============================================================
// LAYER EVIDENCE CARDS
// ============================================================

function TextEvidenceCard({ layer, isPrimary, activeTextId, onToggleText }: { layer: any; isPrimary: boolean; activeTextId?: string | null; onToggleText?: (id: string | null, detail: any) => void }) {
    if (!layer) return null;
    const textsDetail: any[] = layer.ocr_texts_detail || [];
    const hasDetail = textsDetail.length > 0;

    const allTexts = hasDetail
        ? textsDetail.map((d: any, i: number) => ({ id: `text-${i}`, text: d.text, hasBbox: !!d.bbox_pct, detail: d, imageUrl: d.image_url }))
        : (layer.ocr_texts || []).map((t: string, i: number) => ({ id: `text-${i}`, text: t, hasBbox: false, detail: null, imageUrl: null }));

    const activeItem = activeTextId ? allTexts.find((t: any) => t.id === activeTextId) : null;
    const displayImageUrl = activeItem?.imageUrl || layer.best_image_url;
    const hasActiveAnnotation = !!activeItem?.detail?.bbox_pct;

    return (
        <div className="rounded-xl overflow-hidden" style={{ background: isPrimary ? 'rgba(16,185,129,0.06)' : 'rgba(255,255,255,0.02)', border: `1px solid ${isPrimary ? '#10b98140' : 'rgba(255,255,255,0.05)'}` }}>
            <div className="p-3.5">
                <div className="flex items-center gap-2 mb-2">
                    <Type size={12} className="text-emerald-400" />
                    <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-widest">Layer 1: Text Detection</span>
                    <span className={`ml-auto text-[9px] font-bold px-1.5 py-0.5 rounded ${layer.verdict === 'full_match' ? 'bg-green-500/10 text-green-400' :
                        layer.verdict === 'partial_match' ? 'bg-amber-500/10 text-amber-400' :
                            'bg-slate-500/10 text-slate-500'
                        }`}>
                        {layer.verdict === 'full_match' ? 'FULL MATCH' : layer.verdict === 'partial_match' ? 'PARTIAL' : 'NO MATCH'}
                    </span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed mb-2">{layer.detail}</p>

                {layer.matched_text && (
                    <div className="mb-2">
                        <span className="text-[10px] text-slate-600 mr-2">Matched:</span>
                        <span className="text-[11px] font-semibold text-emerald-300 font-mono">"{layer.matched_text}"</span>
                    </div>
                )}

                {displayImageUrl && (layer.matched_text || hasActiveAnnotation) ? (
                    <div className="mt-2">
                        <div className="relative rounded-lg overflow-hidden ring-1 ring-emerald-500/20 transition-all">
                            <img src={displayImageUrl} className={`w-full object-cover transition-all duration-300 ${hasActiveAnnotation ? 'h-52' : 'h-32'}`} alt="Text evidence" loading="lazy" />
                            {hasActiveAnnotation && activeItem?.detail?.bbox_pct && (
                                <div
                                    style={{
                                        position: 'absolute',
                                        left: `${activeItem.detail.bbox_pct[0] * 100}%`,
                                        top: `${activeItem.detail.bbox_pct[1] * 100}%`,
                                        width: `${activeItem.detail.bbox_pct[2] * 100}%`,
                                        height: `${activeItem.detail.bbox_pct[3] * 100}%`,
                                        border: '2px solid #10b981',
                                        backgroundColor: 'rgba(16,185,129,0.15)',
                                        borderRadius: '3px',
                                        pointerEvents: 'none',
                                    }}
                                >
                                    <span style={{
                                        position: 'absolute', top: '-16px', left: 0,
                                        fontSize: '9px', backgroundColor: '#10b981', color: 'white',
                                        padding: '1px 5px', borderRadius: '3px', whiteSpace: 'nowrap',
                                    }}>
                                        "{activeItem.detail.text}"
                                    </span>
                                </div>
                            )}
                        </div>
                    </div>
                ) : !layer.matched_text && (
                    <div className="mt-2 py-4 rounded-lg flex items-center justify-center" style={{ background: 'rgba(255,255,255,0.03)' }}>
                        <span className="text-slate-600 text-[10px] font-medium">
                            {layer.verdict === 'no_images' ? 'No images to analyze' : 'No text match found'}
                        </span>
                    </div>
                )}

                {allTexts.length > 0 && (
                    <div className="mt-2">
                        {hasDetail && (
                            <p className="text-[9px] text-slate-600 mb-1">Click any text to show where it was read</p>
                        )}
                        <div className="flex flex-wrap gap-1">
                            {allTexts.map((item: any) => {
                                const isActive = activeTextId === item.id;
                                return (
                                    <button
                                        key={item.id}
                                        onClick={() => {
                                            if (!item.hasBbox || !onToggleText) return;
                                            onToggleText(isActive ? null : item.id, item.detail);
                                        }}
                                        className={`text-[9px] px-1.5 py-0.5 rounded font-mono transition-all ${isActive
                                            ? 'bg-emerald-500/20 text-emerald-300 ring-1 ring-emerald-500/40'
                                            : item.hasBbox
                                                ? 'text-slate-400 hover:bg-emerald-500/10 hover:text-emerald-300 cursor-pointer'
                                                : 'text-slate-400 cursor-default'
                                            }`}
                                        style={!isActive ? { backgroundColor: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.06)' } : {}}
                                        title={item.hasBbox ? (isActive ? 'Click to hide' : 'Click to show on image') : 'No position data'}
                                    >
                                        {item.text}
                                    </button>
                                );
                            })}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

function XGBoostEvidenceCard({ layer, isPrimary }: { layer: any; isPrimary: boolean }) {
    if (!layer) return null;
    const score = layer.score || 0;

    return (
        <div className="rounded-xl overflow-hidden" style={{ background: isPrimary ? 'rgba(59,130,246,0.06)' : 'rgba(255,255,255,0.02)', border: `1px solid ${isPrimary ? '#3b82f640' : 'rgba(255,255,255,0.05)'}` }}>
            <div className="p-3.5">
                <div className="flex items-center gap-2 mb-2">
                    <Database size={12} className="text-blue-400" />
                    <span className="text-[10px] font-bold text-blue-400 uppercase tracking-widest">Layer 2: Metadata Model</span>
                    <span className={`ml-auto text-[9px] font-bold px-1.5 py-0.5 rounded ${layer.verdict === 'supports_open' ? 'bg-green-500/10 text-green-400' :
                        layer.verdict === 'supports_closed' ? 'bg-red-500/10 text-red-400' :
                            'bg-slate-500/10 text-slate-500'
                        }`}>
                        {layer.verdict === 'supports_open' ? 'OPEN' : layer.verdict === 'supports_closed' ? 'CLOSED' : 'INCONCLUSIVE'}
                        {' '}({(score * 100).toFixed(0)}%)
                    </span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed mb-2">{layer.detail}</p>

                <ScoreBar score={score} max={1.0} color="#3b82f6" />

                {/* Feature contributions */}
                {layer.feature_contributions && Object.keys(layer.feature_contributions).length > 0 && (
                    <div className="mt-3">
                        <p className="text-[9px] text-slate-600 uppercase tracking-widest mb-1.5 font-semibold">Feature Values</p>
                        <div className="space-y-1">
                            {Object.entries(layer.feature_contributions).map(([key, val]: [string, any]) => (
                                <div key={key} className="flex items-center justify-between text-[10px]">
                                    <span className="text-slate-500">{key.replace(/_/g, ' ')}</span>
                                    <span className="text-slate-300 font-mono">{typeof val === 'number' ? val.toFixed(2) : String(val)}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

// ============================================================
// TEST GROUP TOGGLE BAR
// ============================================================
const TEST_GROUPS = [
    { group: null, label: 'All', color: '#94a3b8' },
    { group: 'open', label: 'Actually Open', color: '#22c55e' },
    { group: 'closed', label: 'Actually Closed', color: '#ef4444' },
];

function TestGroupBar({ activeGroup, onSetGroup, data }: { activeGroup: string | null; onSetGroup: (g: string | null) => void; data: any[] }) {
    return (
        <div className="absolute top-4 right-4 z-[1000] flex items-center gap-1.5 bg-slate-900/90 backdrop-blur-md rounded-xl px-2 py-1.5 ring-1 ring-white/10">
            {TEST_GROUPS.map(btn => {
                const count = btn.group ? data.filter((p: any) => p.ground_truth === btn.group).length : data.length;
                const isActive = activeGroup === btn.group;
                return (
                    <button
                        key={btn.group || 'all'}
                        onClick={() => onSetGroup(btn.group)}
                        className={`px-2.5 py-1 rounded-lg text-[10px] font-bold tracking-wide transition-all ${isActive
                            ? 'ring-1 shadow-lg'
                            : 'opacity-60 hover:opacity-100'
                            }`}
                        style={{
                            backgroundColor: isActive ? `${btn.color}20` : 'transparent',
                            color: btn.color,
                            ...(isActive ? { boxShadow: `0 0 0 1px ${btn.color}50` } : {}),
                        }}
                    >
                        {btn.label} <span className="font-mono opacity-70">({count})</span>
                    </button>
                );
            })}
        </div>
    );
}

function AccuracySummary({ data, activeGroup }: { data: any[]; activeGroup: string | null }) {
    const filtered = activeGroup ? data.filter((p: any) => p.test_group === activeGroup) : data;
    const withPredictions = filtered.filter((p: any) => p.vision?.prediction && p.ground_truth);

    if (withPredictions.length === 0) return null;

    const correct = withPredictions.filter((p: any) => {
        const pred = p.vision.prediction;
        const gt = p.ground_truth;
        return (pred === 'open' && gt === 'open') || (pred === 'not_open' && gt === 'closed');
    }).length;

    const accuracy = ((correct / withPredictions.length) * 100).toFixed(0);

    return (
        <div className="absolute bottom-4 left-4 z-[1000] bg-slate-900/90 backdrop-blur-md rounded-xl px-4 py-3 ring-1 ring-white/10">
            <div className="flex items-center gap-3">
                <BarChart3 size={16} className="text-blue-400" />
                <div>
                    <p className="text-[11px] text-slate-400">
                        {activeGroup ? `Group ${activeGroup}` : 'All Groups'} Accuracy
                    </p>
                    <p className="text-[18px] font-bold font-mono text-white">
                        {correct}/{withPredictions.length}
                        <span className="text-[13px] text-slate-400 ml-1.5">({accuracy}%)</span>
                    </p>
                </div>
            </div>
        </div>
    );
}

// ============================================================
// MAIN
// ============================================================
export default function MapContainer() {
    const [selectedPoi, setSelectedPoi] = useState<any | null>(null);
    const [activeTextId, setActiveTextId] = useState<string | null>(null);
    const [activeGroup, setActiveGroup] = useState<string | null>(null);
    const [activeCity, setActiveCity] = useState<string>('sf');
    const [activeLayer, setActiveLayer] = useState<string | null>(null);
    const [textGalleryIdx, setTextGalleryIdx] = useState(0);
    const [highlightedTextId, setHighlightedTextId] = useState<string | null>(null);

    const cityConfig = CITIES[activeCity];
    const allData = cityConfig.data;
    const filteredData = useMemo(() =>
        activeGroup ? allData.filter((p: any) => p.ground_truth === activeGroup) : allData,
        [activeGroup, allData]
    );

    useEffect(() => {
        const handleOpen = (e: any) => { if (e.detail) setSelectedPoi(e.detail); };
        window.addEventListener('open-location-panel', handleOpen);
        return () => window.removeEventListener('open-location-panel', handleOpen);
    }, []);

    // Clear annotations when switching POIs
    useEffect(() => {
        setActiveTextId(null);
        setActiveLayer(null);
        setTextGalleryIdx(0);
    }, [selectedPoi?.id]);

    const openPanel = (poi: any) => {
        setSelectedPoi(poi);
        setTextGalleryIdx(0);
        setHighlightedTextId(null);
    };

    // City selector portal
    const cityPortal = createPortal(
        <div style={{ position: 'fixed', top: 16, right: 16, zIndex: 99999, pointerEvents: 'auto' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: 'rgba(15,23,42,0.95)', borderRadius: 12, padding: '6px 8px', boxShadow: '0 4px 24px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.1)' }}>
                {Object.entries(CITIES).map(([key, city]) => {
                    const isActive = activeCity === key;
                    return (
                        <button
                            key={key}
                            onClick={() => { setActiveCity(key); setSelectedPoi(null); setActiveGroup(null); }}
                            style={{
                                padding: '5px 12px',
                                borderRadius: 8,
                                fontSize: 11,
                                fontWeight: 700,
                                letterSpacing: '0.03em',
                                border: 'none',
                                cursor: 'pointer',
                                transition: 'all 0.2s',
                                backgroundColor: isActive ? 'rgba(96,165,250,0.2)' : 'transparent',
                                color: isActive ? '#60a5fa' : '#94a3b8',
                                boxShadow: isActive ? '0 0 0 1px rgba(96,165,250,0.5)' : 'none',
                                fontFamily: 'Inter, system-ui, sans-serif',
                            }}
                        >
                            {city.label} <span style={{ fontSize: 9, opacity: 0.7, marginLeft: 4 }}>{city.data.length}</span>
                        </button>
                    );
                })}
            </div>
        </div>,
        document.body
    );

    // Portal-rendered toggle bar — renders into document.body to escape Leaflet's DOM/z-index
    const togglePortal = createPortal(
        <div style={{ position: 'fixed', top: 16, left: selectedPoi ? 432 : 16, zIndex: 99999, pointerEvents: 'auto', transition: 'left 0.5s ease-out' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: 'rgba(15,23,42,0.95)', borderRadius: 12, padding: '6px 8px', boxShadow: '0 4px 24px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.1)' }}>
                {TEST_GROUPS.map(btn => {
                    const hasGt = allData.some((p: any) => p.ground_truth);
                    const count = btn.group ? allData.filter((p: any) => p.ground_truth === btn.group).length : allData.length;
                    const isActive = activeGroup === btn.group;
                    const isDisabled = btn.group !== null && !hasGt;
                    return (
                        <button
                            key={btn.group || 'all'}
                            onClick={() => !isDisabled && setActiveGroup(btn.group)}
                            style={{
                                padding: '4px 10px',
                                borderRadius: 8,
                                fontSize: 10,
                                fontWeight: 700,
                                letterSpacing: '0.05em',
                                border: 'none',
                                cursor: isDisabled ? 'not-allowed' : 'pointer',
                                transition: 'all 0.2s',
                                backgroundColor: isActive ? `${btn.color}33` : 'transparent',
                                color: btn.color,
                                opacity: isDisabled ? 0.25 : isActive ? 1 : 0.6,
                                boxShadow: isActive ? `0 0 0 1px ${btn.color}80` : 'none',
                                fontFamily: 'Inter, system-ui, sans-serif',
                            }}
                            title={isDisabled ? 'No ground truth labels for this city' : ''}
                        >
                            {btn.label} ({count})
                        </button>
                    );
                })}
            </div>
        </div>,
        document.body
    );

    const accuracyPortal = createPortal(
        <div style={{ position: 'fixed', top: 72, right: 16, zIndex: 99999, pointerEvents: 'auto' }}>
            {(() => {
                const withPredictions = allData.filter((p: any) => p.vision?.prediction && p.ground_truth);
                if (withPredictions.length === 0) return null;

                // Split by ground truth
                const openGt = withPredictions.filter((p: any) => p.ground_truth === 'open');
                const closedGt = withPredictions.filter((p: any) => p.ground_truth === 'closed');
                const openCorrect = openGt.filter((p: any) => p.vision.prediction === 'open').length;
                const closedCorrect = closedGt.filter((p: any) => p.vision.prediction === 'not_open').length;
                const openPct = openGt.length > 0 ? Math.round((openCorrect / openGt.length) * 100) : 0;
                const closedPct = closedGt.length > 0 ? Math.round((closedCorrect / closedGt.length) * 100) : 0;
                const totalCorrect = openCorrect + closedCorrect;
                const totalGt = openGt.length + closedGt.length;
                const totalPct = totalGt > 0 ? Math.round((totalCorrect / totalGt) * 100) : 0;

                const pctColor = (pct: number) => pct >= 80 ? '#4ade80' : pct >= 60 ? '#fbbf24' : '#f87171';

                // If a filter is active, show only that group
                const showOpen = !activeGroup || activeGroup === 'open';
                const showClosed = !activeGroup || activeGroup === 'closed';

                return (
                    <div style={{ background: 'rgba(15,23,42,0.95)', borderRadius: 12, padding: '12px 16px', boxShadow: '0 4px 24px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.1)', display: 'flex', flexDirection: 'column', gap: 6 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2 }}>
                            <BarChart3 size={14} style={{ color: '#60a5fa' }} />
                            <span style={{ fontSize: 10, color: '#64748b', fontWeight: 600, letterSpacing: '0.05em', textTransform: 'uppercase' }}>Accuracy</span>
                        </div>
                        {showOpen && (
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                                <span style={{ fontSize: 11, color: '#94a3b8' }}>Actually Open</span>
                                <span style={{ fontSize: 14, fontWeight: 700, fontFamily: 'JetBrains Mono, monospace', color: 'white' }}>
                                    <span style={{ color: '#22c55e' }}>{openCorrect}</span>
                                    <span style={{ color: '#475569' }}>/{openGt.length}</span>
                                    <span style={{ fontSize: 11, color: pctColor(openPct), marginLeft: 6 }}>({openPct}%)</span>
                                </span>
                            </div>
                        )}
                        {showClosed && (
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                                <span style={{ fontSize: 11, color: '#94a3b8' }}>Actually Closed</span>
                                <span style={{ fontSize: 14, fontWeight: 700, fontFamily: 'JetBrains Mono, monospace', color: 'white' }}>
                                    <span style={{ color: '#ef4444' }}>{closedCorrect}</span>
                                    <span style={{ color: '#475569' }}>/{closedGt.length}</span>
                                    <span style={{ fontSize: 11, color: pctColor(closedPct), marginLeft: 6 }}>({closedPct}%)</span>
                                </span>
                            </div>
                        )}
                        {!activeGroup && (
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: 6, marginTop: 2 }}>
                                <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600 }}>Overall</span>
                                <span style={{ fontSize: 14, fontWeight: 700, fontFamily: 'JetBrains Mono, monospace', color: 'white' }}>
                                    {totalCorrect}<span style={{ color: '#475569' }}>/{totalGt}</span>
                                    <span style={{ fontSize: 11, color: pctColor(totalPct), marginLeft: 6 }}>({totalPct}%)</span>
                                </span>
                            </div>
                        )}
                    </div>
                );
            })()}
        </div>,
        document.body
    );

    return (
        <div className="relative w-full h-full flex overflow-hidden">
            {cityPortal}
            {togglePortal}
            {accuracyPortal}

            {/* ── DETAIL PANEL (left side) ── */}
            {selectedPoi && (() => {
                const v = selectedPoi.vision;
                const layers = v?.layers;
                const xgbScore = layers?.xgboost?.score || 0;
                const textVerdict = layers?.text?.verdict;

                const textsDetail: any[] = layers?.text?.ocr_texts_detail || [];
                const allTexts = textsDetail.length > 0
                    ? textsDetail.map((d: any, i: number) => ({ id: `text-${i}`, text: d.text, hasBbox: !!d.bbox_pct, detail: d, imageUrl: d.image_url, isMatch: d.is_match || false }))
                    : (layers?.text?.ocr_texts || []).map((t: string, i: number) => ({ id: `text-${i}`, text: t, hasBbox: false, detail: null, imageUrl: null, isMatch: false }));

                // Group texts by image for overlay rendering
                const textsByImage: Record<string, typeof allTexts> = {};
                for (const t of allTexts) {
                    if (t.imageUrl && t.hasBbox) {
                        if (!textsByImage[t.imageUrl]) textsByImage[t.imageUrl] = [];
                        textsByImage[t.imageUrl].push(t);
                    }
                }

                // Pick the display image: actively clicked text's image, or best match image
                const displayImageUrl = (activeTextId ? allTexts.find((t: any) => t.id === activeTextId)?.imageUrl : null)
                    || layers?.text?.best_image_url
                    || (Object.keys(textsByImage)[0] ?? null);
                const textsOnDisplayImage = displayImageUrl ? (textsByImage[displayImageUrl] || []) : [];

                const predColor = v?.prediction === 'open' ? '#22c55e' : v?.prediction === 'not_open' ? '#ef4444' : v?.prediction === 'uncertain' ? '#f59e0b' : '#6b7280';

                return createPortal(
                    <div
                        className="panel-enter"
                        style={{
                            position: 'fixed',
                            top: 16,
                            left: 16,
                            maxHeight: 'calc(100vh - 32px)',
                            height: 'fit-content',
                            width: 400,
                            zIndex: 99999,
                            overflowY: 'auto',
                            background: 'rgba(15,23,42,0.95)',
                            backdropFilter: 'blur(16px)',
                            borderRadius: 20,
                            boxShadow: `0 20px 60px rgba(0,0,0,0.8), 0 0 40px rgba(255,255,255,0.05)`,
                            pointerEvents: 'auto',
                            border: '1px solid rgba(255,255,255,0.1)',
                        }}
                    >
                        {/* ── Close button ── */}
                        <button
                            onClick={() => setSelectedPoi(null)}
                            style={{
                                position: 'absolute',
                                top: 16,
                                right: 16,
                                zIndex: 50,
                                background: 'none',
                                border: 'none',
                                padding: 0,
                                margin: 0,
                                cursor: 'pointer',
                                color: 'white',
                                lineHeight: 1,
                            }}
                        >
                            <X size={18} />
                        </button>

                        {/* ── Header ── */}
                        <div className="px-8 pt-7 pb-5 text-center">
                            <h2 className="text-[22px] font-bold text-white leading-snug pr-6">{selectedPoi.name}</h2>
                            <div className="flex items-center justify-center gap-2 mt-0">
                                <span
                                    className="text-[11px] font-semibold uppercase tracking-widest px-5 py-1.5 rounded-full"
                                    style={{ color: '#60a5fa', border: '1px solid rgba(59,130,246,0.3)' }}
                                >
                                    &nbsp;&nbsp;{selectedPoi.category}&nbsp;&nbsp;
                                </span>
                            </div>
                            <div className="flex items-center justify-center text-[11px] text-slate-500 mt-1.5 gap-1.5">
                                <MapPin size={11} className="shrink-0 text-slate-600" />
                                <span>&nbsp;{selectedPoi.address}</span>
                            </div>
                            <div className="flex items-center justify-center text-[11px] text-slate-500 font-mono mt-1 gap-1.5">
                                <Navigation size={11} className="shrink-0 text-slate-600" />
                                <span>&nbsp;{selectedPoi.location[1].toFixed(5)}, {selectedPoi.location[0].toFixed(5)}</span>
                            </div>
                        </div>

                        {/* ── Divider Removed ── */}

                        {/* ── Content ── */}
                        <div className="px-8 pb-8 pt-6 space-y-5">

                            {/* ── CONFIDENCE HERO + SCORE MATH ── */}
                            {v?.confidence != null && (() => {
                                const rawScore = v.confidence;
                                const isOpen = v.prediction === 'open';
                                const directedConf = isOpen ? rawScore : (1 - rawScore);
                                const pct = Math.round(directedConf * 100);
                                const color = isOpen ? '#4ade80' : '#f87171';
                                const label = isOpen ? 'LIKELY OPEN' : 'LIKELY CLOSED';
                                const breakdown: { signal: string; weight: number; description: string }[] = v.score_breakdown || [];
                                return (
                                    <div className="text-center py-4">
                                        <div className="text-[44px] font-black font-mono leading-none tracking-tighter" style={{ color }}>{pct}%</div>
                                        <div className="text-[10px] tracking-[0.25em] mt-2 uppercase" style={{ color, opacity: 0.8, fontWeight: 900 }}>{label}</div>
                                        <div className="text-[10px] mt-1 text-slate-500">
                                            {isOpen
                                                ? <>{Math.round((1 - rawScore) * 100)}% chance closed</>
                                                : <>{Math.round(rawScore * 100)}% chance open</>
                                            }
                                        </div>
                                        <div className="w-3/4 mx-auto h-1 rounded-full overflow-hidden mt-4" style={{ background: 'rgba(255,255,255,0.06)' }}>
                                            <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, backgroundColor: color }} />
                                        </div>

                                        {/* Score breakdown table — metamodel signals */}
                                        {breakdown.length > 0 && (() => {
                                            const metaRow = breakdown.find(b => b.signal === 'metamodel');
                                            const signalRows = breakdown.filter(b => b.signal !== 'metamodel');
                                            // Parse logit from metamodel description
                                            const logitMatch = metaRow?.description?.match(/logit=([+-]?\d+\.?\d*)/);
                                            const logit = logitMatch ? parseFloat(logitMatch[1]) : null;
                                            return (
                                            <div className="mt-5 mx-auto" style={{ maxWidth: 300 }}>
                                                <div style={{ height: 1, background: 'rgba(255,255,255,0.06)' }} className="mb-3" />
                                                {/* Signal header */}
                                                <div className="flex items-center justify-between text-[8px] uppercase tracking-widest text-slate-600 mb-2">
                                                    <span>Signal</span>
                                                    <div className="flex gap-6">
                                                        <span style={{ width: 40, textAlign: 'right' }}>Value</span>
                                                        <span style={{ width: 50, textAlign: 'right' }}>Weight</span>
                                                        <span style={{ width: 50, textAlign: 'right' }}>Contrib</span>
                                                    </div>
                                                </div>
                                                {signalRows.map((item, i) => {
                                                    // Extract signal value from description
                                                    const sigMatch = item.description.match(/signal=([+-]?\d+\.?\d*)/);
                                                    const sigVal = sigMatch ? parseFloat(sigMatch[1]) : 0;
                                                    const metaWeight = item.signal === 'foursquare' ? 1.98 : item.signal === 'website' ? 0.99 : item.signal === 'text' ? 0.10 : item.signal === 'xgboost' ? 1.73 : item.signal === 'tomtom' ? 0.08 : item.signal === 'yelp' ? 0.72 : 0;
                                                    const contrib = item.weight;
                                                    const contribColor = contrib > 0.01 ? '#4ade80' : contrib < -0.01 ? '#f87171' : '#94a3b8';
                                                    const sigColor = sigVal > 0.01 ? '#4ade80' : sigVal < -0.01 ? '#f87171' : '#94a3b8';
                                                    // Label
                                                    const label = item.signal === 'foursquare' ? 'Foursquare' : item.signal === 'website' ? 'Website' : item.signal === 'text' ? 'Text/OCR' : item.signal === 'xgboost' ? 'XGBoost' : item.signal === 'tomtom' ? 'TomTom' : item.signal === 'yelp' ? 'Yelp' : item.signal;
                                                    // Status text from description
                                                    const statusMatch = item.description.match(/:\s*(\w+)/);
                                                    const status = statusMatch ? statusMatch[1] : '';
                                                    return (
                                                        <div key={i} className="flex items-center justify-between text-[10px] py-0.5">
                                                            <span className="text-slate-400 text-left">
                                                                {label} <span className="text-slate-600 text-[8px]">({status})</span>
                                                            </span>
                                                            <div className="flex gap-6">
                                                                <span className="font-mono font-semibold" style={{ color: sigColor, width: 40, textAlign: 'right' }}>{sigVal > 0 ? '+' : ''}{sigVal.toFixed(1)}</span>
                                                                <span className="font-mono text-slate-500" style={{ width: 50, textAlign: 'right' }}>{'\u00D7'}{metaWeight.toFixed(2)}</span>
                                                                <span className="font-mono font-semibold" style={{ color: contribColor, width: 50, textAlign: 'right' }}>{contrib > 0 ? '+' : ''}{contrib.toFixed(2)}</span>
                                                            </div>
                                                        </div>
                                                    );
                                                })}
                                                {/* Intercept */}
                                                <div className="flex items-center justify-between text-[10px] py-0.5">
                                                    <span className="text-slate-500 text-left">Intercept (bias)</span>
                                                    <div className="flex gap-6">
                                                        <span style={{ width: 40 }}></span>
                                                        <span style={{ width: 50 }}></span>
                                                        <span className="font-mono font-semibold text-slate-400" style={{ width: 50, textAlign: 'right' }}>+0.26</span>
                                                    </div>
                                                </div>
                                                <div style={{ height: 1, background: 'rgba(255,255,255,0.06)' }} className="my-2" />
                                                {/* Logit sum */}
                                                {logit != null && (
                                                    <div className="flex items-center justify-between text-[10px] py-0.5">
                                                        <span className="text-slate-400">Logit sum</span>
                                                        <span className="font-mono font-semibold" style={{ color: logit > 0 ? '#4ade80' : '#f87171' }}>{logit > 0 ? '+' : ''}{logit.toFixed(2)}</span>
                                                    </div>
                                                )}
                                                {/* Final probability */}
                                                <div className="flex items-center justify-between text-[11px] py-0.5">
                                                    <span className="text-white font-semibold">sigmoid({logit != null ? logit.toFixed(2) : '?'}) =</span>
                                                    <span className="font-mono font-bold" style={{ color }}>{rawScore.toFixed(2)}</span>
                                                </div>
                                            </div>
                                            );
                                        })()}
                                    </div>
                                );
                            })()}

                            {/* ── Ground Truth ── */}
                            {selectedPoi.ground_truth && (
                                <div className="space-y-2 text-center">
                                    <div className="flex items-center justify-center gap-1.5">
                                        <ShieldCheck size={13} className="text-slate-500" />
                                        <span className="text-[11px] text-slate-400 font-medium tracking-wide">&nbsp;GROUND TRUTH:&nbsp;</span>
                                        <span className={`text-[12px] font-bold font-mono ${selectedPoi.ground_truth === 'open' ? 'text-green-400' : 'text-red-400'}`}>
                                            {selectedPoi.ground_truth === 'open' ? 'OPEN' : 'CLOSED'}
                                        </span>
                                    </div>
                                    {selectedPoi.vision?.prediction && (() => {
                                        const isCorrect = (selectedPoi.vision.prediction === 'open' && selectedPoi.ground_truth === 'open') ||
                                            (selectedPoi.vision.prediction === 'not_open' && selectedPoi.ground_truth === 'closed');
                                        return (
                                            <div className="flex items-center justify-center pt-1.5 pb-1">
                                                <span className={`inline-flex items-center gap-1 text-[11px] font-bold px-2 py-0.5 rounded-full ${isCorrect ? 'bg-green-500/15 text-green-400' : 'bg-red-500/15 text-red-400'}`}>
                                                    {isCorrect ? <CheckCircle size={10} /> : <XCircle size={10} />}
                                                    <span>&nbsp;{isCorrect ? 'PREDICTION CORRECT' : 'PREDICTION INCORRECT'}</span>
                                                </span>
                                            </div>
                                        );
                                    })()}
                                    {selectedPoi.yelp && (
                                        <p className="text-[9px] text-slate-600">
                                            Source: Yelp — {selectedPoi.yelp.yelp_name}
                                            {selectedPoi.yelp.yelp_rating && ` (${selectedPoi.yelp.yelp_rating}★)`}
                                        </p>
                                    )}
                                </div>
                            )}

                            {/* ── Divider Removed ── */}

                            {/* ── Signal Icons ── */}
                            <div className="flex items-start justify-center gap-6 h-16">
                                {(() => {
                                    const isActive = activeLayer === 'website';
                                    return (
                                        <button
                                            onClick={() => setActiveLayer(isActive ? null : 'website')}
                                            style={{ background: 'transparent', border: 'none', padding: '8px 8px 4px 8px', cursor: 'pointer', opacity: isActive ? 1 : 0.6, transition: 'all 0.2s', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}
                                            title="Website"
                                        >
                                            <Globe size={24} style={{ color: '#3b82f6' }} />
                                            <div style={{ width: 20, height: 2, backgroundColor: isActive ? 'white' : 'transparent', borderRadius: 2, transition: 'background-color 0.2s' }} />
                                        </button>
                                    );
                                })()}
                                {(() => {
                                    const isActive = activeLayer === 'text';
                                    return (
                                        <button
                                            onClick={() => setActiveLayer(isActive ? null : 'text')}
                                            style={{ background: 'transparent', border: 'none', padding: '8px 8px 4px 8px', cursor: 'pointer', opacity: isActive ? 1 : 0.6, transition: 'all 0.2s', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}
                                            title="Text Detection & Images"
                                        >
                                            <Type size={24} style={{ color: '#ef4444' }} />
                                            <div style={{ width: 20, height: 2, backgroundColor: isActive ? 'white' : 'transparent', borderRadius: 2, transition: 'background-color 0.2s' }} />
                                        </button>
                                    );
                                })()}
                                {(() => {
                                    const isActive = activeLayer === 'xgboost';
                                    return (
                                        <button
                                            onClick={() => setActiveLayer(isActive ? null : 'xgboost')}
                                            style={{ background: 'transparent', border: 'none', padding: '8px 8px 4px 8px', cursor: 'pointer', opacity: isActive ? 1 : 0.6, transition: 'all 0.2s', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}
                                            title="Metadata Model"
                                        >
                                            <Database size={24} style={{ color: '#f97316' }} />
                                            <div style={{ width: 20, height: 2, backgroundColor: isActive ? 'white' : 'transparent', borderRadius: 2, transition: 'background-color 0.2s' }} />
                                        </button>
                                    );
                                })()}
                            </div>

                            {/* ── Expanded Layer Content ── */}
                            {activeLayer === 'website' && (
                                <div className="panel-content-enter pt-4 flex flex-col items-center text-center">
                                    {layers?.website ? (
                                        <>
                                            <div className="w-full">
                                                <div style={{ backgroundColor: layers.website.status === 'alive' ? '#059669' : layers.website.status === 'dead' ? '#dc2626' : '#d97706', borderColor: layers.website.status === 'alive' ? '#065f46' : layers.website.status === 'dead' ? '#991b1b' : '#92400e' }} className="w-full py-2.5 rounded-lg shadow-sm border flex items-center justify-center">
                                                    <span style={{ fontWeight: 900 }} className="text-[12px] text-white font-mono tracking-wide uppercase text-center block w-full">
                                                        {layers.website.status === 'alive' ? 'ALIVE' : layers.website.status === 'dead' ? 'DEAD' : layers.website.status === 'redirect' ? 'REDIRECTED' : layers.website.status === 'parked' ? 'PARKED' : layers.website.status.toUpperCase()}
                                                    </span>
                                                </div>
                                            </div>
                                            <div className="flex flex-col items-center gap-0 w-full pt-3">
                                                {layers.website.url && (
                                                    <a href={layers.website.url} target="_blank" rel="noopener noreferrer"
                                                        className="text-[11px] text-slate-400 hover:text-slate-300 flex items-center justify-center gap-1.5 break-all font-medium text-center">
                                                        <ExternalLink size={10} className="shrink-0" />
                                                        {layers.website.url}
                                                    </a>
                                                )}
                                                <div className="text-[11px] text-slate-400/80 leading-tight text-center m-0 p-0">{layers.website.detail}</div>
                                            </div>
                                        </>
                                    ) : (
                                        <div className="w-full">
                                            <div style={{ backgroundColor: '#64748b', borderColor: '#475569' }} className="w-full py-2.5 rounded-lg shadow-sm border flex items-center justify-center">
                                                <span style={{ fontWeight: 900 }} className="text-[12px] text-white font-mono tracking-wide uppercase text-center block w-full">
                                                    NO WEBSITE
                                                </span>
                                            </div>
                                            <p className="text-[11px] text-slate-500 mt-3">No website on record for this business</p>
                                        </div>
                                    )}
                                </div>
                            )}

                            {activeLayer === 'text' && (() => {
                                const gallery: any[] = selectedPoi.current_gallery || [];
                                const safeIdx = Math.min(textGalleryIdx, Math.max(0, gallery.length - 1));
                                const currentImg = gallery[safeIdx];
                                const currentImgUrl = currentImg?.url;
                                // Texts detected on this specific image
                                const textsForCurrentImg = currentImgUrl ? allTexts.filter((t: any) => t.imageUrl === currentImgUrl) : [];
                                // The highlighted text (clicked chip) — show its bbox
                                const highlightedItem = highlightedTextId ? textsForCurrentImg.find((t: any) => t.id === highlightedTextId) : null;
                                return (
                                <div className="panel-content-enter pt-4 flex flex-col items-center text-center">
                                    {/* Verdict badge */}
                                    {layers?.text && (
                                        <div className="w-full">
                                            <div style={{ backgroundColor: textVerdict === 'full_match' ? '#059669' : textVerdict === 'partial_match' ? '#d97706' : textVerdict === 'no_match' ? '#dc2626' : '#64748b', borderColor: textVerdict === 'full_match' ? '#065f46' : textVerdict === 'partial_match' ? '#92400e' : textVerdict === 'no_match' ? '#991b1b' : '#475569' }} className="w-full py-2.5 rounded-lg shadow-sm border flex items-center justify-center">
                                                <span style={{ fontWeight: 900 }} className="text-[12px] text-white font-mono tracking-wide uppercase text-center block w-full">
                                                    {textVerdict === 'full_match' ? 'FULL MATCH' : textVerdict === 'partial_match' ? 'PARTIAL' : textVerdict === 'no_images' ? 'NO IMAGES' : 'NO MATCH'}
                                                </span>
                                            </div>
                                            {layers.text.matched_text && (
                                                <div className="text-center mt-2">
                                                    <span className="text-[10px] text-slate-500 mr-2">Matched:</span>
                                                    <span className="text-[12px] font-bold text-red-400 font-mono">"{layers.text.matched_text}"</span>
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    {/* Image browser */}
                                    {gallery.length > 0 ? (
                                        <div className="w-full mt-3">
                                            {/* Navigation row: arrow — image — arrow */}
                                            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                                                {gallery.length > 1 ? (
                                                    <button onClick={() => { setTextGalleryIdx(safeIdx > 0 ? safeIdx - 1 : gallery.length - 1); setHighlightedTextId(null); }}
                                                        style={{ background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, width: 32, height: 32, display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', flexShrink: 0 }}>
                                                        <ChevronLeft size={18} color="white" />
                                                    </button>
                                                ) : <div style={{ width: 32, flexShrink: 0 }} />}
                                                <div className="relative rounded-lg overflow-hidden shadow-lg" style={{ flex: 1, minWidth: 0 }}>
                                                    <img src={currentImgUrl} style={{ width: '100%', height: 180, objectFit: 'cover', display: 'block' }} alt={`Street view ${safeIdx + 1}`} loading="lazy" />
                                                    {/* Highlighted text bbox overlay (clicked chip) */}
                                                    {highlightedItem?.detail?.bbox_pct && (() => {
                                                        const bbox = highlightedItem.detail.bbox_pct;
                                                        const color = highlightedItem.isMatch ? '#ef4444' : '#3b82f6';
                                                        return (
                                                            <div style={{
                                                                position: 'absolute', left: `${bbox[0] * 100}%`, top: `${bbox[1] * 100}%`, width: `${bbox[2] * 100}%`, height: `${bbox[3] * 100}%`,
                                                                border: `2px solid ${color}`, backgroundColor: `${color}33`,
                                                                borderRadius: '2px', zIndex: 2, pointerEvents: 'none',
                                                            }}>
                                                                <span style={{ position: 'absolute', top: '-20px', left: '50%', transform: 'translateX(-50%)', fontSize: '10px', fontWeight: 700, backgroundColor: color, color: 'white', padding: '2px 8px', borderRadius: '4px', whiteSpace: 'nowrap', lineHeight: '14px' }}>
                                                                    {highlightedItem.text}
                                                                </span>
                                                            </div>
                                                        );
                                                    })()}
                                                    {/* Counter badge */}
                                                    <div style={{ position: 'absolute', bottom: 6, right: 6, background: 'rgba(0,0,0,0.7)', borderRadius: 8, padding: '2px 8px', zIndex: 5 }}>
                                                        <span className="text-[10px] text-white font-mono font-semibold">{safeIdx + 1}/{gallery.length}</span>
                                                    </div>
                                                </div>
                                                {gallery.length > 1 ? (
                                                    <button onClick={() => { setTextGalleryIdx(safeIdx < gallery.length - 1 ? safeIdx + 1 : 0); setHighlightedTextId(null); }}
                                                        style={{ background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, width: 32, height: 32, display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', flexShrink: 0 }}>
                                                        <ChevronRight size={18} color="white" />
                                                    </button>
                                                ) : <div style={{ width: 32, flexShrink: 0 }} />}
                                            </div>
                                            {/* Image metadata */}
                                            {currentImg && (
                                                <div className="flex items-center justify-center gap-3 mt-1.5">
                                                    <span className="text-[9px] text-slate-500 flex items-center gap-1"><Calendar size={9} />{currentImg.date}</span>
                                                    <span className="text-[9px] text-slate-500 flex items-center gap-1"><Navigation size={9} />{currentImg.distance_m}m away</span>
                                                </div>
                                            )}
                                            {/* All OCR texts detected in this image — click to highlight on image */}
                                            {textsForCurrentImg.length > 0 ? (
                                                <div className="mt-2">
                                                    <p className="text-[8px] text-slate-600 uppercase tracking-widest mb-1.5">Detected text — click to locate</p>
                                                    <div className="flex flex-wrap justify-center gap-1.5">
                                                        {textsForCurrentImg.map((item: any) => {
                                                            const isHighlighted = highlightedTextId === item.id;
                                                            const isMatch = item.isMatch;
                                                            return (
                                                                <button key={item.id}
                                                                    onClick={() => setHighlightedTextId(isHighlighted ? null : item.id)}
                                                                    style={{
                                                                        cursor: item.detail?.bbox_pct ? 'pointer' : 'default',
                                                                        background: isHighlighted ? (isMatch ? 'rgba(239,68,68,0.3)' : 'rgba(59,130,246,0.3)') : (isMatch ? 'rgba(239,68,68,0.15)' : 'rgba(255,255,255,0.04)'),
                                                                        border: isHighlighted ? `1.5px solid ${isMatch ? '#ef4444' : '#3b82f6'}` : (isMatch ? '1px solid rgba(239,68,68,0.4)' : '1px solid rgba(255,255,255,0.06)'),
                                                                        borderRadius: 4, padding: '2px 6px', fontSize: '9px', fontFamily: 'monospace',
                                                                        color: isMatch ? '#fca5a5' : '#94a3b8',
                                                                        fontWeight: isMatch || isHighlighted ? 600 : 400,
                                                                        transition: 'all 0.15s',
                                                                    }}>
                                                                    {isMatch ? '★ ' : ''}{item.text}
                                                                </button>
                                                            );
                                                        })}
                                                    </div>
                                                </div>
                                            ) : (
                                                <p className="text-[9px] text-slate-600 mt-2">No text detected in this image</p>
                                            )}
                                        </div>
                                    ) : (
                                        <div className="w-full mt-3 py-6 rounded-lg flex flex-col items-center justify-center" style={{ background: 'rgba(255,255,255,0.03)' }}>
                                            <span className="text-slate-600 text-[11px] font-medium">No street-level images available</span>
                                        </div>
                                    )}

                                    {/* Summary */}
                                    <p className="text-[10px] text-slate-500 pt-2 text-center">
                                        {v?.images_analyzed || 0} images analyzed · {allTexts.filter((t: any) => t.isMatch).length} text matches
                                    </p>
                                </div>
                                );
                            })()}

                            {activeLayer === 'xgboost' && layers?.xgboost && (
                                <div className="panel-content-enter pt-4 flex flex-col items-center text-center">
                                    <div className="w-full">
                                        <div style={{ backgroundColor: xgbScore > 0.6 ? '#059669' : xgbScore < 0.4 ? '#dc2626' : '#d97706', borderColor: xgbScore > 0.6 ? '#065f46' : xgbScore < 0.4 ? '#991b1b' : '#92400e' }} className="w-full py-2.5 rounded-lg shadow-sm border flex items-center justify-center">
                                            <span style={{ fontWeight: 900 }} className="text-[12px] text-white font-mono tracking-wide uppercase text-center block w-full">
                                                {(xgbScore * 100).toFixed(0)}% OPEN
                                            </span>
                                        </div>
                                    </div>
                                    <div className="w-full mt-3">
                                        <ScoreBar score={xgbScore} max={1.0} color="#f97316" />
                                    </div>
                                    <div className="text-[11px] text-slate-400/80 leading-tight text-center mt-2 m-0 p-0">{layers.xgboost.detail}</div>
                                    {layers.xgboost.feature_contributions && Object.keys(layers.xgboost.feature_contributions).length > 0 && (
                                        <div className="space-y-1.5 pt-2 w-full flex flex-col items-center" style={{ borderTop: '1px solid rgba(249,115,22,0.1)' }}>
                                            <p className="text-[9px] text-orange-400/80 uppercase tracking-widest font-semibold text-center pb-1">Features</p>
                                            <div className="w-full max-w-[200px]">
                                                {Object.entries(layers.xgboost.feature_contributions).map(([key, val]: [string, any]) => (
                                                    <div key={key} className="flex items-center justify-between text-[10px]">
                                                        <span className="text-slate-500">{key.replace(/_/g, ' ')}</span>
                                                        <span className="text-slate-300 font-mono">{typeof val === 'number' ? val.toFixed(2) : String(val)}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* ── Footer ── */}
                            <div className="pt-4 text-center" style={{ borderTop: '1px solid rgba(255,255,255,0.04)' }}>
                                <p className="text-[7px] text-slate-700 uppercase tracking-[0.25em]">Overture Maps · Yelp · Mapillary · Foursquare · OCR + XGBoost</p>
                            </div>
                        </div>
                    </div>,
                    document.body
                );
            })()}



            {/* Map */}
            <div className="relative h-full flex-1 min-w-0 transition-all duration-500 ease-out">
                <LeafletMap center={cityConfig.center} zoom={cityConfig.zoom} className="w-full h-full">
                    <MapFlyTo center={cityConfig.center} zoom={cityConfig.zoom} />
                    <TileLayer
                        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                    />
                    {filteredData.map((poi: any) => (
                        <Marker
                            key={poi.id}
                            position={[poi.location[1], poi.location[0]]}
                            icon={getPinIcon(poi)}
                            eventHandlers={{ click: () => openPanel(poi) }}
                        />
                    ))}
                </LeafletMap>
            </div>
        </div>
    );
}
