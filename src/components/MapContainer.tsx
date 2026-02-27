import { useState, useEffect, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { MapContainer as LeafletMap, TileLayer, Marker } from 'react-leaflet';
import L from 'leaflet';
import { renderToStaticMarkup } from 'react-dom/server';
import testData from '../data/test_data.json';
import {
    MapPin, X, Calendar, Navigation, Eye, ShieldCheck,
    Clock, Database, Crosshair, ArrowRight, CheckCircle,
    AlertTriangle, XCircle, HelpCircle, Globe, Phone, Type,
    Layers, ImageIcon,
    ScanEye, ExternalLink, BarChart3,
    ChevronDown, ChevronRight
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
            <div className="flex items-center gap-2 mb-3">
                <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: accent }} />
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest">{label}</span>
                <span className="text-[11px] text-slate-600 ml-auto font-mono">{images.length} views</span>
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
            <p className="text-[10px] text-slate-600 mt-1.5 flex items-center gap-1">
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
        <span className={`inline-flex items-center gap-1 text-[9px] font-bold px-1.5 py-0.5 rounded ${
            isCorrect
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
                    <span className={`ml-auto text-[9px] font-bold px-1.5 py-0.5 rounded ${
                        layer.verdict === 'full_match' ? 'bg-green-500/10 text-green-400' :
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

                {displayImageUrl && (
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
                                        className={`text-[9px] px-1.5 py-0.5 rounded font-mono transition-all ${
                                            isActive
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
                    <span className={`ml-auto text-[9px] font-bold px-1.5 py-0.5 rounded ${
                        layer.verdict === 'supports_open' ? 'bg-green-500/10 text-green-400' :
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
    { group: null, label: 'ALL', color: '#94a3b8' },
    { group: 'open', label: 'Open', color: '#22c55e' },
    { group: 'closed', label: 'Closed', color: '#ef4444' },
];

function TestGroupBar({ activeGroup, onSetGroup, data }: { activeGroup: string | null; onSetGroup: (g: string | null) => void; data: any[] }) {
    return (
        <div className="absolute top-4 right-4 z-[1000] flex items-center gap-1.5 bg-slate-900/90 backdrop-blur-md rounded-xl px-2 py-1.5 ring-1 ring-white/10">
            {TEST_GROUPS.map(btn => {
                const count = btn.group ? data.filter((p: any) => p.test_group === btn.group).length : data.length;
                const isActive = activeGroup === btn.group;
                return (
                    <button
                        key={btn.group || 'all'}
                        onClick={() => onSetGroup(btn.group)}
                        className={`px-2.5 py-1 rounded-lg text-[10px] font-bold tracking-wide transition-all ${
                            isActive
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
    const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({});

    const toggleSection = (key: string) => {
        setExpandedSections(prev => ({ ...prev, [key]: !prev[key] }));
    };

    const allData = testData as any[];
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
    }, [selectedPoi?.id]);

    const openPanel = (poi: any) => {
        setSelectedPoi(poi);
    };

    // Portal-rendered toggle bar — renders into document.body to escape Leaflet's DOM/z-index
    const togglePortal = createPortal(
        <div style={{ position: 'fixed', top: 16, right: selectedPoi ? 460 : 16, zIndex: 99999, transition: 'right 0.5s ease-out', pointerEvents: 'auto' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: 'rgba(15,23,42,0.95)', borderRadius: 12, padding: '6px 8px', boxShadow: '0 4px 24px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.1)' }}>
                {TEST_GROUPS.map(btn => {
                    const count = btn.group ? allData.filter((p: any) => p.ground_truth === btn.group).length : allData.length;
                    const isActive = activeGroup === btn.group;
                    return (
                        <button
                            key={btn.group || 'all'}
                            onClick={() => setActiveGroup(btn.group)}
                            style={{
                                padding: '4px 10px',
                                borderRadius: 8,
                                fontSize: 10,
                                fontWeight: 700,
                                letterSpacing: '0.05em',
                                border: 'none',
                                cursor: 'pointer',
                                transition: 'all 0.2s',
                                backgroundColor: isActive ? `${btn.color}33` : 'transparent',
                                color: btn.color,
                                opacity: isActive ? 1 : 0.6,
                                boxShadow: isActive ? `0 0 0 1px ${btn.color}80` : 'none',
                                fontFamily: 'Inter, system-ui, sans-serif',
                            }}
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
        <div style={{ position: 'fixed', bottom: 16, left: 16, zIndex: 99999, pointerEvents: 'auto' }}>
            {(() => {
                const filtered = activeGroup ? allData.filter((p: any) => p.ground_truth === activeGroup) : allData;
                const withPredictions = filtered.filter((p: any) => p.vision?.prediction && p.ground_truth);
                if (withPredictions.length === 0) return null;
                const correct = withPredictions.filter((p: any) => {
                    const pred = p.vision.prediction;
                    const gt = p.ground_truth;
                    return (pred === 'open' && gt === 'open') || (pred === 'not_open' && gt === 'closed');
                }).length;
                const accuracy = ((correct / withPredictions.length) * 100).toFixed(0);
                return (
                    <div style={{ background: 'rgba(15,23,42,0.95)', borderRadius: 12, padding: '12px 16px', boxShadow: '0 4px 24px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.1)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                            <BarChart3 size={16} style={{ color: '#60a5fa' }} />
                            <div>
                                <p style={{ fontSize: 11, color: '#94a3b8', margin: 0 }}>
                                    {activeGroup ? `${activeGroup.charAt(0).toUpperCase() + activeGroup.slice(1)} Businesses` : 'All'} Accuracy
                                </p>
                                <p style={{ fontSize: 18, fontWeight: 700, color: 'white', margin: 0, fontFamily: 'JetBrains Mono, monospace' }}>
                                    {correct}/{withPredictions.length}
                                    <span style={{ fontSize: 13, color: '#94a3b8', marginLeft: 6 }}>({accuracy}%)</span>
                                </p>
                            </div>
                        </div>
                    </div>
                );
            })()}
        </div>,
        document.body
    );

    return (
        <div className="relative w-full h-full flex overflow-hidden">
            {togglePortal}
            {accuracyPortal}

            {/* Map */}
            <div className={`relative h-full transition-all duration-500 ease-out ${selectedPoi ? 'w-[calc(100%-440px)]' : 'w-full'}`}>
                <LeafletMap center={[37.7749, -122.4194]} zoom={12} className="w-full h-full">
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

            {/* ── DETAIL PANEL ── */}
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

                return (
                    <div
                        className="w-[440px] h-full z-[1000] overflow-y-auto absolute right-0 top-0 panel-enter"
                        style={{
                            background: 'linear-gradient(180deg, #0c1021 0%, #080d1a 100%)',
                            borderLeft: '1px solid rgba(255,255,255,0.05)',
                            boxShadow: '-20px 0 60px rgba(0,0,0,0.5)',
                        }}
                    >
                        {/* ── Header ── */}
                        <div
                            className="sticky top-0 z-20 px-5 pt-5 pb-4"
                            style={{ background: 'linear-gradient(180deg, #0c1021 85%, transparent)', backdropFilter: 'blur(16px)' }}
                        >
                            <div className="flex items-start justify-between gap-3">
                                <div className="flex-1 min-w-0">
                                    <h2 className="text-lg font-bold text-white tracking-tight truncate leading-tight">{selectedPoi.name}</h2>
                                    <div className="flex items-center gap-1.5 mt-2 flex-wrap">
                                        <span
                                            className="text-[9px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-md"
                                            style={{ backgroundColor: 'rgba(59,130,246,0.1)', color: '#60a5fa', border: '1px solid rgba(59,130,246,0.2)' }}
                                        >
                                            {selectedPoi.category}
                                        </span>
                                        {v?.prediction && <PredictionPill prediction={v.prediction} />}
                                    </div>
                                </div>
                                <button
                                    onClick={() => setSelectedPoi(null)}
                                    className="text-slate-600 hover:text-white p-1.5 hover:bg-white/[0.06] rounded-lg transition-all"
                                >
                                    <X size={18} />
                                </button>
                            </div>
                            <div className="flex items-start text-xs text-slate-500 mt-2.5 gap-1.5">
                                <MapPin size={12} className="mt-0.5 shrink-0 text-slate-600" />
                                <span className="leading-relaxed">{selectedPoi.address}</span>
                            </div>
                        </div>

                        {/* ── Content ── */}
                        <div className="px-5 pb-8 space-y-3">

                            {/* ── CONFIDENCE HERO ── */}
                            {v?.confidence != null && (() => {
                                const rawScore = v.confidence;
                                const isOpen = v.prediction === 'open';
                                const directedConf = isOpen ? rawScore : (1 - rawScore);
                                const pct = Math.round(directedConf * 100);
                                const color = isOpen ? '#4ade80' : '#f87171';
                                const bgColor = isOpen ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)';
                                const borderColor = isOpen ? 'rgba(34,197,94,0.2)' : 'rgba(239,68,68,0.2)';
                                const label = isOpen ? 'LIKELY OPEN' : 'LIKELY CLOSED';
                                return (
                                    <div className="rounded-xl p-4 text-center" style={{ background: bgColor, border: `1px solid ${borderColor}` }}>
                                        <div className="text-[36px] font-black font-mono leading-none" style={{ color }}>{pct}%</div>
                                        <div className="text-[11px] font-bold tracking-[0.15em] mt-1.5" style={{ color }}>{label}</div>
                                        <div className="w-full h-1.5 bg-white/[0.06] rounded-full overflow-hidden mt-3">
                                            <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, backgroundColor: color }} />
                                        </div>
                                    </div>
                                );
                            })()}

                            {/* ── SECTION 1: Ground Truth (always visible) ── */}
                            {selectedPoi.ground_truth && (
                                <div className="rounded-xl p-3.5" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-2">
                                            <ShieldCheck size={14} className="text-slate-400" />
                                            <span className="text-[11px] text-slate-400">Ground Truth</span>
                                        </div>
                                        <span className={`text-[12px] font-bold font-mono ${selectedPoi.ground_truth === 'open' ? 'text-green-400' : 'text-red-400'}`}>
                                            {selectedPoi.ground_truth === 'open' ? 'OPEN' : 'CLOSED'}
                                        </span>
                                    </div>
                                    {/* Model accuracy vs ground truth */}
                                    {selectedPoi.vision?.prediction && (
                                        <div className="flex items-center justify-between mt-2 pt-2" style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                                            <span className="text-[10px] text-slate-500">Model predicted: <span className="font-mono font-bold text-slate-400">{selectedPoi.vision.prediction.toUpperCase()}</span></span>
                                            <GroundTruthBadge poi={selectedPoi} />
                                        </div>
                                    )}
                                    {selectedPoi.yelp && (
                                        <p className="text-[9px] text-slate-600 mt-1.5">
                                            Source: Yelp — {selectedPoi.yelp.yelp_name}
                                            {selectedPoi.yelp.yelp_rating && ` (${selectedPoi.yelp.yelp_rating}★)`}
                                        </p>
                                    )}
                                </div>
                            )}

                            {/* ── SECTION 2: Metadata Model (collapsible) ── */}
                            {layers?.xgboost && (
                                <div className="rounded-xl overflow-hidden" style={{ background: 'rgba(59,130,246,0.04)', border: '1px solid rgba(59,130,246,0.15)' }}>
                                    <button
                                        onClick={() => toggleSection('xgboost')}
                                        className="w-full flex items-center gap-2 p-3.5 hover:bg-white/[0.02] transition-colors"
                                    >
                                        <Database size={14} className="text-blue-400" />
                                        <span className="text-[11px] font-semibold text-blue-400">Metadata Model</span>
                                        <span className={`ml-auto text-[10px] font-bold font-mono px-2 py-0.5 rounded ${
                                            xgbScore > 0.6 ? 'bg-green-500/10 text-green-400' :
                                            xgbScore < 0.4 ? 'bg-red-500/10 text-red-400' :
                                            'bg-slate-500/10 text-slate-400'
                                        }`}>
                                            {(xgbScore * 100).toFixed(0)}% open
                                        </span>
                                        {expandedSections.xgboost
                                            ? <ChevronDown size={14} className="text-slate-500 ml-1" />
                                            : <ChevronRight size={14} className="text-slate-500 ml-1" />
                                        }
                                    </button>
                                    {expandedSections.xgboost && (
                                        <div className="px-3.5 pb-3.5 pt-0 space-y-2.5">
                                            <ScoreBar score={xgbScore} max={1.0} color="#3b82f6" />
                                            <p className="text-[10px] text-slate-500 leading-relaxed">{layers.xgboost.detail}</p>
                                            {layers.xgboost.feature_contributions && Object.keys(layers.xgboost.feature_contributions).length > 0 && (
                                                <div className="space-y-1 pt-1">
                                                    <p className="text-[9px] text-slate-600 uppercase tracking-widest font-semibold">Features</p>
                                                    {Object.entries(layers.xgboost.feature_contributions).map(([key, val]: [string, any]) => (
                                                        <div key={key} className="flex items-center justify-between text-[10px]">
                                                            <span className="text-slate-500">{key.replace(/_/g, ' ')}</span>
                                                            <span className="text-slate-300 font-mono">{typeof val === 'number' ? val.toFixed(2) : String(val)}</span>
                                                        </div>
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* ── SECTION 3: Website Liveness (collapsible) ── */}
                            {layers?.website && layers.website.status !== 'no_url' && (
                                <div className="rounded-xl overflow-hidden" style={{
                                    background: layers.website.status === 'alive' ? 'rgba(34,197,94,0.04)' :
                                               layers.website.status === 'dead' ? 'rgba(239,68,68,0.04)' :
                                               'rgba(245,158,11,0.04)',
                                    border: `1px solid ${layers.website.status === 'alive' ? 'rgba(34,197,94,0.15)' :
                                                         layers.website.status === 'dead' ? 'rgba(239,68,68,0.15)' :
                                                         'rgba(245,158,11,0.15)'}`,
                                }}>
                                    <button
                                        onClick={() => toggleSection('website')}
                                        className="w-full flex items-center gap-2 p-3.5 hover:bg-white/[0.02] transition-colors"
                                    >
                                        <Globe size={14} className={
                                            layers.website.status === 'alive' ? 'text-green-400' :
                                            layers.website.status === 'dead' ? 'text-red-400' :
                                            'text-amber-400'
                                        } />
                                        <span className={`text-[11px] font-semibold ${
                                            layers.website.status === 'alive' ? 'text-green-400' :
                                            layers.website.status === 'dead' ? 'text-red-400' :
                                            'text-amber-400'
                                        }`}>Website</span>
                                        <span className={`ml-auto text-[10px] font-bold px-2 py-0.5 rounded ${
                                            layers.website.status === 'alive' ? 'bg-green-500/10 text-green-400' :
                                            layers.website.status === 'dead' ? 'bg-red-500/10 text-red-400' :
                                            layers.website.status === 'redirect' ? 'bg-red-500/10 text-red-400' :
                                            'bg-amber-500/10 text-amber-400'
                                        }`}>
                                            {layers.website.status === 'alive' ? 'ALIVE' :
                                             layers.website.status === 'dead' ? 'DEAD' :
                                             layers.website.status === 'redirect' ? 'REDIRECTED' :
                                             layers.website.status === 'parked' ? 'PARKED' :
                                             layers.website.status.toUpperCase()}
                                        </span>
                                        {expandedSections.website
                                            ? <ChevronDown size={14} className="text-slate-500 ml-1" />
                                            : <ChevronRight size={14} className="text-slate-500 ml-1" />
                                        }
                                    </button>
                                    {expandedSections.website && (
                                        <div className="px-3.5 pb-3.5 pt-0 space-y-2">
                                            {layers.website.url && (
                                                <a
                                                    href={layers.website.url}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className="text-[10px] text-blue-400 hover:text-blue-300 flex items-center gap-1 break-all"
                                                >
                                                    <ExternalLink size={9} className="shrink-0" />
                                                    {layers.website.url}
                                                </a>
                                            )}
                                            <p className="text-[10px] text-slate-500 leading-relaxed">{layers.website.detail}</p>
                                            {layers.website.status_code && (
                                                <span className="text-[9px] font-mono text-slate-600">HTTP {layers.website.status_code}</span>
                                            )}
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* ── SECTION 4: Text Detection (collapsible) ── */}
                            {layers?.text && (
                                <div className="rounded-xl overflow-hidden" style={{ background: 'rgba(16,185,129,0.04)', border: '1px solid rgba(16,185,129,0.15)' }}>
                                    <button
                                        onClick={() => toggleSection('text')}
                                        className="w-full flex items-center gap-2 p-3.5 hover:bg-white/[0.02] transition-colors"
                                    >
                                        <Type size={14} className="text-emerald-400" />
                                        <span className="text-[11px] font-semibold text-emerald-400">Text Detection</span>
                                        <span className={`ml-auto text-[10px] font-bold px-2 py-0.5 rounded ${
                                            textVerdict === 'full_match' ? 'bg-green-500/10 text-green-400' :
                                            textVerdict === 'partial_match' ? 'bg-amber-500/10 text-amber-400' :
                                            'bg-slate-500/10 text-slate-500'
                                        }`}>
                                            {textVerdict === 'full_match' ? 'FULL MATCH' : textVerdict === 'partial_match' ? 'PARTIAL' : textVerdict === 'no_images' ? 'NO IMAGES' : 'NO MATCH'}
                                        </span>
                                        {expandedSections.text
                                            ? <ChevronDown size={14} className="text-slate-500 ml-1" />
                                            : <ChevronRight size={14} className="text-slate-500 ml-1" />
                                        }
                                    </button>
                                    {expandedSections.text && (
                                        <div className="px-3.5 pb-3.5 pt-0 space-y-2.5">
                                            <p className="text-[10px] text-slate-500 leading-relaxed">{layers.text.detail}</p>

                                            {/* Image age warning */}
                                            {layers.text.image_age_years > 3 && (
                                                <div className="flex items-center gap-1.5 px-2 py-1.5 rounded-md" style={{
                                                    backgroundColor: layers.text.image_age_years > 5 ? 'rgba(239,68,68,0.08)' : 'rgba(245,158,11,0.08)',
                                                    border: `1px solid ${layers.text.image_age_years > 5 ? 'rgba(239,68,68,0.2)' : 'rgba(245,158,11,0.2)'}`,
                                                }}>
                                                    <Clock size={10} className={layers.text.image_age_years > 5 ? 'text-red-400' : 'text-amber-400'} />
                                                    <span className={`text-[9px] font-semibold ${layers.text.image_age_years > 5 ? 'text-red-400' : 'text-amber-400'}`}>
                                                        Image is {layers.text.image_age_years.toFixed(0)} years old
                                                        {layers.text.image_age_years > 5 ? ' — low reliability' : ' — reduced confidence'}
                                                    </span>
                                                    <span className="ml-auto text-[9px] font-mono text-slate-500">
                                                        {layers.text.image_age_factor !== undefined ? `${(layers.text.image_age_factor * 100).toFixed(0)}% weight` : ''}
                                                    </span>
                                                </div>
                                            )}

                                            {layers.text.matched_text && (
                                                <div>
                                                    <span className="text-[10px] text-slate-600 mr-2">Matched:</span>
                                                    <span className="text-[11px] font-semibold text-emerald-300 font-mono">"{layers.text.matched_text}"</span>
                                                </div>
                                            )}

                                            {displayImageUrl && (
                                                <div className="relative rounded-lg overflow-hidden ring-1 ring-emerald-500/20">
                                                    <img src={displayImageUrl} className="w-full object-cover h-52" alt="Text evidence" loading="lazy" />
                                                    {/* All bounding boxes on this image */}
                                                    {textsOnDisplayImage.map((item: any) => {
                                                        const bbox = item.detail?.bbox_pct;
                                                        if (!bbox) return null;
                                                        const isActive = activeTextId === item.id;
                                                        const isMatch = item.isMatch;
                                                        return (
                                                            <div
                                                                key={item.id}
                                                                onClick={(e) => { e.stopPropagation(); setActiveTextId(isActive ? null : item.id); }}
                                                                style={{
                                                                    position: 'absolute',
                                                                    left: `${bbox[0] * 100}%`,
                                                                    top: `${bbox[1] * 100}%`,
                                                                    width: `${bbox[2] * 100}%`,
                                                                    height: `${bbox[3] * 100}%`,
                                                                    border: isActive ? '2px solid #fbbf24' : isMatch ? '2px solid #10b981' : '1px solid rgba(148,163,184,0.4)',
                                                                    backgroundColor: isActive ? 'rgba(251,191,36,0.25)' : isMatch ? 'rgba(16,185,129,0.15)' : 'rgba(148,163,184,0.08)',
                                                                    borderRadius: '2px',
                                                                    cursor: 'pointer',
                                                                    zIndex: isActive ? 3 : isMatch ? 2 : 1,
                                                                    transition: 'all 0.15s ease',
                                                                }}
                                                            >
                                                                {(isActive || isMatch) && (
                                                                    <span style={{
                                                                        position: 'absolute', top: '-15px', left: 0,
                                                                        fontSize: '8px',
                                                                        backgroundColor: isActive ? '#f59e0b' : '#10b981',
                                                                        color: 'white',
                                                                        padding: '1px 4px', borderRadius: '2px', whiteSpace: 'nowrap',
                                                                        lineHeight: '12px',
                                                                    }}>
                                                                        {item.text}
                                                                    </span>
                                                                )}
                                                            </div>
                                                        );
                                                    })}
                                                </div>
                                            )}

                                            {allTexts.length > 0 && (
                                                <div>
                                                    <p className="text-[9px] text-slate-600 mb-1">Click text to locate on image</p>
                                                    <div className="flex flex-wrap gap-1">
                                                        {allTexts.map((item: any) => {
                                                            const isActive = activeTextId === item.id;
                                                            return (
                                                                <button
                                                                    key={item.id}
                                                                    onClick={(e) => {
                                                                        e.stopPropagation();
                                                                        if (!item.hasBbox) return;
                                                                        setActiveTextId(isActive ? null : item.id);
                                                                    }}
                                                                    className={`text-[9px] px-1.5 py-0.5 rounded font-mono transition-all ${
                                                                        isActive
                                                                            ? 'bg-amber-500/20 text-amber-300 ring-1 ring-amber-500/40'
                                                                            : item.isMatch
                                                                                ? 'bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30'
                                                                                : item.hasBbox
                                                                                    ? 'text-slate-400 hover:bg-white/[0.06] cursor-pointer'
                                                                                    : 'text-slate-500 cursor-default'
                                                                    }`}
                                                                    style={!isActive && !item.isMatch ? { backgroundColor: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.06)' } : {}}
                                                                >
                                                                    {item.isMatch ? '● ' : ''}{item.text}
                                                                </button>
                                                            );
                                                        })}
                                                    </div>
                                                </div>
                                            )}

                                            <p className="text-[9px] text-slate-600">
                                                {v?.images_analyzed || 0} images analyzed · {textsOnDisplayImage.filter((t: any) => t.isMatch).length} matches found
                                            </p>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* ── SECTION 4: Street View (collapsible) ── */}
                            {selectedPoi.current_gallery?.length > 0 && (
                                <div className="rounded-xl overflow-hidden" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}>
                                    <button
                                        onClick={() => toggleSection('gallery')}
                                        className="w-full flex items-center gap-2 p-3.5 hover:bg-white/[0.02] transition-colors"
                                    >
                                        <ImageIcon size={14} className="text-slate-400" />
                                        <span className="text-[11px] font-semibold text-slate-400">Street View</span>
                                        <span className="ml-auto text-[10px] text-slate-500 font-mono">{selectedPoi.current_gallery.length} images</span>
                                        {expandedSections.gallery
                                            ? <ChevronDown size={14} className="text-slate-500 ml-1" />
                                            : <ChevronRight size={14} className="text-slate-500 ml-1" />
                                        }
                                    </button>
                                    {expandedSections.gallery && (
                                        <div className="px-3.5 pb-3.5 pt-0">
                                            <GalleryStrip images={selectedPoi.current_gallery} label="Street View" accent="#3b82f6" />
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* FOOTER */}
                            <div className="pt-3 border-t border-white/[0.04] text-center">
                                <p className="text-[10px] text-slate-700 font-mono">{selectedPoi.location[1].toFixed(5)}, {selectedPoi.location[0].toFixed(5)}</p>
                                <p className="text-[8px] text-slate-800 mt-1 uppercase tracking-[0.2em]">Overture Maps · Yelp · Mapillary · Foursquare · OCR + XGBoost</p>
                            </div>
                        </div>
                    </div>
                );
            })()}
        </div>
    );
}
