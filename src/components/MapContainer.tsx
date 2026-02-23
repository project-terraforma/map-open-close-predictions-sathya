import { useState, useEffect } from 'react';
import { MapContainer as LeafletMap, TileLayer, Marker } from 'react-leaflet';
import L from 'leaflet';
import { renderToStaticMarkup } from 'react-dom/server';
import mockData from '../data/mock_data.json';
import {
    MapPin, X, Calendar, Navigation, Eye, ShieldCheck,
    Clock, Database, Crosshair, ArrowRight, CheckCircle,
    AlertTriangle, XCircle, HelpCircle, Globe, Phone, Type,
    Layers, ImageIcon,
    ScanEye, ExternalLink
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

function ScoreBar({ score, max = 0.4, color }: { score: number; max?: number; color: string }) {
    const pct = Math.min((score / max) * 100, 100);
    return (
        <div className="w-full h-1 bg-white/[0.04] rounded-full overflow-hidden">
            <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, backgroundColor: color }} />
        </div>
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
    logo: { label: 'LOGO', icon: Eye, color: '#3b82f6' },
    text: { label: 'TEXT', icon: Type, color: '#10b981' },
    data: { label: 'DATA', icon: Database, color: '#f59e0b' },
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
function LogoEvidenceCard({ layer, isPrimary, showRegion, onToggleRegion, perImage }: { layer: any; accent?: string; isPrimary: boolean; showRegion?: boolean; onToggleRegion?: () => void; perImage?: any[] }) {
    if (!layer) return null;
    const hasCropRegion = layer.best_crop_region && layer.best_image_url;

    return (
        <div className="rounded-xl overflow-hidden" style={{ background: isPrimary ? 'rgba(59,130,246,0.06)' : 'rgba(255,255,255,0.02)', border: `1px solid ${isPrimary ? '#3b82f640' : 'rgba(255,255,255,0.05)'}` }}>
            <div className="p-3.5">
                <div className="flex items-center gap-2 mb-2">
                    <Eye size={12} className="text-blue-400" />
                    <span className="text-[10px] font-bold text-blue-400 uppercase tracking-widest">Layer 1: Logo Detection</span>
                    <span className={`ml-auto text-[9px] font-bold px-1.5 py-0.5 rounded ${
                        layer.verdict === 'detected' ? 'bg-green-500/10 text-green-400' :
                        layer.verdict === 'weak' ? 'bg-amber-500/10 text-amber-400' :
                        'bg-slate-500/10 text-slate-500'
                    }`}>
                        {layer.verdict === 'detected' ? 'DETECTED' : layer.verdict === 'weak' ? 'WEAK' : 'NOT FOUND'}
                    </span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed mb-2">
                    {/* Replace raw CLIP % with normalized score in detail text */}
                    {layer.detail?.replace(/at \d+\.\d+%/, `at ${((layer.score || 0) * 100).toFixed(0)}%`) || ''}
                </p>

                {layer.clip_brand && (
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-[12px] font-semibold text-white">{layer.clip_brand}</span>
                        <span className="text-[11px] font-bold font-mono px-2 py-0.5 rounded-md bg-blue-500/10 text-blue-400">
                            {((layer.score || 0) * 100).toFixed(0)}%
                        </span>
                    </div>
                )}
                {layer.score > 0 && (
                    <ScoreBar score={layer.score} max={1.0} color="#3b82f6" />
                )}

                {/* Evidence image with toggleable region overlay */}
                {layer.best_image_url && (
                    <div className="mt-3">
                        <div className="relative rounded-lg overflow-hidden ring-1 ring-blue-500/20 transition-all">
                            <img src={layer.best_image_url} className={`w-full object-cover transition-all duration-300 ${showRegion ? 'h-52' : 'h-32'}`} alt="Logo evidence" loading="lazy" />
                            {showRegion && layer.best_crop_region && (
                                <div
                                    style={{
                                        position: 'absolute',
                                        left: `${layer.best_crop_region[0] * 100}%`,
                                        top: `${layer.best_crop_region[1] * 100}%`,
                                        width: `${layer.best_crop_region[2] * 100}%`,
                                        height: `${layer.best_crop_region[3] * 100}%`,
                                        border: '2px solid #3b82f6',
                                        backgroundColor: 'rgba(59,130,246,0.15)',
                                        borderRadius: '3px',
                                        pointerEvents: 'none',
                                    }}
                                >
                                    <span style={{
                                        position: 'absolute', top: '-16px', left: 0,
                                        fontSize: '9px', backgroundColor: '#3b82f6', color: 'white',
                                        padding: '1px 5px', borderRadius: '3px', whiteSpace: 'nowrap',
                                    }}>
                                        {layer.clip_brand} detected here
                                    </span>
                                </div>
                            )}
                        </div>
                        {hasCropRegion && (
                            <button
                                onClick={onToggleRegion}
                                className={`mt-1.5 text-[9px] font-mono px-2 py-1 rounded transition-all ${
                                    showRegion
                                        ? 'bg-blue-500/20 text-blue-300 ring-1 ring-blue-500/30'
                                        : 'bg-white/5 text-slate-500 hover:text-blue-300 hover:bg-blue-500/10'
                                }`}
                            >
                                {showRegion ? 'Hide region' : 'Show detection region'}
                            </button>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

function TextEvidenceCard({ layer, isPrimary, activeTextId, onToggleText, perImage }: { layer: any; accent?: string; isPrimary: boolean; activeTextId?: string | null; onToggleText?: (id: string | null, detail: any) => void; perImage?: any[] }) {
    if (!layer) return null;
    const textsDetail: any[] = layer.ocr_texts_detail || [];
    const hasDetail = textsDetail.length > 0;

    // Build flat list with ids
    const allTexts = hasDetail
        ? textsDetail.map((d: any, i: number) => ({ id: `text-${i}`, text: d.text, hasBbox: !!d.bbox_pct, detail: d, imageUrl: d.image_url }))
        : (layer.ocr_texts || []).map((t: string, i: number) => ({ id: `text-${i}`, text: t, hasBbox: false, detail: null, imageUrl: null }));

    // Find active text detail
    const activeItem = activeTextId ? allTexts.find((t: any) => t.id === activeTextId) : null;
    const displayImageUrl = activeItem?.imageUrl || layer.best_image_url;
    const hasActiveAnnotation = !!activeItem?.detail?.bbox_pct;

    return (
        <div className="rounded-xl overflow-hidden" style={{ background: isPrimary ? 'rgba(16,185,129,0.06)' : 'rgba(255,255,255,0.02)', border: `1px solid ${isPrimary ? '#10b98140' : 'rgba(255,255,255,0.05)'}` }}>
            <div className="p-3.5">
                <div className="flex items-center gap-2 mb-2">
                    <Type size={12} className="text-emerald-400" />
                    <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-widest">Layer 2: Text Detection</span>
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

                {/* Evidence image — shows active text's image or best_image_url */}
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

                {/* All OCR text tags — click to toggle bbox on image */}
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

function DataEvidenceCard({ layer, isPrimary, foursquare }: { layer: any; isPrimary: boolean; foursquare: any; overture?: any }) {
    if (!layer) return null;
    const borderColor = isPrimary ? '#f59e0b' : 'rgba(255,255,255,0.05)';
    return (
        <div className="rounded-xl overflow-hidden" style={{ background: isPrimary ? 'rgba(245,158,11,0.06)' : 'rgba(255,255,255,0.02)', border: `1px solid ${borderColor}${isPrimary ? '40' : ''}` }}>
            <div className="p-3.5">
                <div className="flex items-center gap-2 mb-2">
                    <Database size={12} className="text-amber-400" />
                    <span className="text-[10px] font-bold text-amber-400 uppercase tracking-widest">Layer 3: Data Verification</span>
                    <span className={`ml-auto text-[9px] font-bold px-1.5 py-0.5 rounded ${
                        layer.verdict === 'supports_open' ? 'bg-green-500/10 text-green-400' :
                        layer.verdict === 'supports_closed' ? 'bg-red-500/10 text-red-400' :
                        'bg-slate-500/10 text-slate-500'
                    }`}>
                        {layer.verdict === 'supports_open' ? 'SUPPORTS OPEN' : layer.verdict === 'supports_closed' ? 'SUPPORTS CLOSED' : 'INCONCLUSIVE'}
                    </span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed mb-3">{layer.detail}</p>

                {/* Signals breakdown */}
                {layer.signals && (
                    <div className="space-y-1.5">
                        {layer.signals.map((sig: any, i: number) => (
                            <div key={i} className="flex items-center justify-between text-[10px]">
                                <span className="text-slate-500">{sig.signal}</span>
                                <div className="flex items-center gap-2">
                                    <span className="text-slate-400 font-mono">{sig.value}</span>
                                    <span className={`font-bold font-mono min-w-[40px] text-right ${
                                        sig.contribution > 0 ? 'text-green-400' : sig.contribution < 0 ? 'text-red-400' : 'text-slate-600'
                                    }`}>
                                        {sig.contribution > 0 ? '+' : ''}{sig.contribution.toFixed(2)}
                                    </span>
                                </div>
                            </div>
                        ))}
                        <div className="flex items-center justify-between text-[10px] pt-1.5 border-t border-white/[0.04]">
                            <span className="text-slate-400 font-semibold">Total Score</span>
                            <span className={`font-bold font-mono ${layer.score > 0 ? 'text-green-400' : layer.score < 0 ? 'text-red-400' : 'text-slate-400'}`}>
                                {layer.score > 0 ? '+' : ''}{layer.score.toFixed(2)}
                            </span>
                        </div>
                    </div>
                )}

                {/* Foursquare details when primary */}
                {isPrimary && foursquare?.match && (
                    <div className="mt-3 pt-3 border-t border-white/[0.04]">
                        <div className="flex items-center gap-2 mb-2">
                            <ShieldCheck size={11} className="text-amber-400" />
                            <span className="text-[10px] font-semibold text-slate-400">Foursquare</span>
                            <VerificationBadge status={foursquare.status} />
                        </div>
                        <div className="grid grid-cols-2 gap-y-1.5 gap-x-3 text-[10px]">
                            {foursquare.match.name && (
                                <div className="text-slate-500">Name <span className="text-slate-300">{foursquare.match.name}</span></div>
                            )}
                            {foursquare.match.category && (
                                <div className="text-slate-500">Type <span className="text-slate-300">{foursquare.match.category}</span></div>
                            )}
                            {foursquare.match.website && (
                                <div className="col-span-2 flex items-center gap-1 text-slate-500">
                                    <Globe size={9} className="text-slate-600 shrink-0" />
                                    <a href={foursquare.match.website} target="_blank" rel="noopener noreferrer"
                                       className="text-blue-400 hover:text-blue-300 truncate transition-colors text-[10px]">
                                        {foursquare.match.website.replace(/https?:\/\/(www\.)?/, '')}
                                    </a>
                                </div>
                            )}
                            {foursquare.match.phone && (
                                <div className="flex items-center gap-1 text-slate-500">
                                    <Phone size={9} className="text-slate-600 shrink-0" />
                                    <span className="text-slate-300 font-mono">{foursquare.match.phone}</span>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

// ============================================================
// MAIN
// ============================================================
export default function MapContainer() {
    const [selectedPoi, setSelectedPoi] = useState<any | null>(null);
    const [showLogoRegion, setShowLogoRegion] = useState(false);
    const [activeTextId, setActiveTextId] = useState<string | null>(null);

    useEffect(() => {
        const handleOpen = (e: any) => { if (e.detail) setSelectedPoi(e.detail); };
        window.addEventListener('open-location-panel', handleOpen);
        return () => window.removeEventListener('open-location-panel', handleOpen);
    }, []);

    // Clear annotations when switching POIs
    useEffect(() => {
        setShowLogoRegion(false);
        setActiveTextId(null);
    }, [selectedPoi?.id]);

    const openPanel = (poi: any) => {
        setSelectedPoi(poi);
    };

    return (
        <div className="relative w-full h-full flex overflow-hidden">
            {/* Map */}
            <div className={`relative h-full transition-all duration-500 ease-out ${selectedPoi ? 'w-[calc(100%-440px)]' : 'w-full'}`}>
                <LeafletMap center={[37.7749, -122.4194]} zoom={12} className="w-full h-full">
                    <TileLayer
                        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                    />
                    {mockData.map((poi: any) => (
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
                const isOpen = v?.prediction === 'open';
                const isNotOpen = v?.prediction === 'not_open';
                const isUncertain = v?.prediction === 'uncertain';
                const accent = isOpen ? '#22c55e' : isNotOpen ? '#ef4444' : isUncertain ? '#f59e0b' : '#64748b';
                // Override primary_layer: logo only if score >= 0.70
                let primaryLayer = v?.primary_layer || 'logo';
                if (primaryLayer === 'logo' && (v?.layers?.logo?.score || 0) < 0.70) {
                    primaryLayer = v?.layers?.text?.verdict !== 'no_match' ? 'text' : 'data';
                }
                const layers = v?.layers;

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
                                        <span className={`text-[9px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-md flex items-center gap-1 ${
                                            selectedPoi.location_type === 'intersection'
                                                ? 'bg-purple-500/10 text-purple-400 border border-purple-500/20'
                                                : 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20'
                                        }`}>
                                            {selectedPoi.location_type === 'intersection'
                                                ? <><Crosshair size={9} /> Intersection</>
                                                : <><ArrowRight size={9} /> Mid-block</>
                                            }
                                        </span>
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
                        <div className="px-5 pb-8 space-y-4">

                            {/* PREDICTION OVERVIEW */}
                            {v && (
                                <div className="rounded-xl overflow-hidden" style={{ background: `${accent}08`, border: `1px solid ${accent}18` }}>
                                    <div className="p-4">
                                        <div className="flex items-center justify-between mb-3">
                                            <div className="flex items-center gap-2.5">
                                                <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ backgroundColor: `${accent}12` }}>
                                                    <ScanEye size={14} style={{ color: accent }} />
                                                </div>
                                                <div>
                                                    <h4 className="text-[13px] font-bold text-white tracking-tight">Prediction</h4>
                                                    <p className="text-[9px] text-slate-600 uppercase tracking-widest font-mono">3-Layer Analysis</p>
                                                </div>
                                            </div>
                                            <PredictionPill prediction={v.prediction} />
                                        </div>

                                        {/* Layer indicator */}
                                        {v.primary_layer && (
                                            <div className="mb-3">
                                                <LayerIndicator primaryLayer={primaryLayer} />
                                            </div>
                                        )}

                                        <p className="text-[11px] text-slate-400 leading-relaxed mb-2.5">
                                            {(v.evidence || v.detail || '').replace(/Logo match: \d+\.\d+%/, `Logo match: ${((layers?.logo?.score || 0) * 100).toFixed(0)}%`)}
                                        </p>

                                        <div className="flex items-center gap-1.5 text-[11px] text-slate-500">
                                            <ImageIcon size={11} />
                                            <span><strong className="text-slate-300 font-mono">{v.images_analyzed}</strong> images analyzed</span>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* TEMPORAL WARNING */}
                            {selectedPoi.temporal?.flag === 'widened' && (
                                <div className="flex items-center gap-1.5 text-[10px] text-amber-400 bg-amber-500/10 px-3 py-2 rounded-lg border border-amber-500/20">
                                    <AlertTriangle size={11} className="shrink-0" />
                                    <span>Images span {selectedPoi.temporal.date_range_days} days — mixed time periods may affect accuracy</span>
                                </div>
                            )}

                            {/* ALL LAYERS — always Logo → Text → Data order, primary one highlighted */}
                            {layers && (
                                <div className="space-y-2.5">
                                    <LogoEvidenceCard layer={layers.logo} isPrimary={primaryLayer === 'logo'} showRegion={showLogoRegion} onToggleRegion={() => setShowLogoRegion(r => !r)} perImage={v?.per_image} />
                                    <TextEvidenceCard layer={layers.text} isPrimary={primaryLayer === 'text'} activeTextId={activeTextId} onToggleText={(id) => setActiveTextId(id)} perImage={v?.per_image} />
                                    <DataEvidenceCard layer={layers.data} isPrimary={primaryLayer === 'data'} foursquare={selectedPoi.foursquare} overture={selectedPoi.overture_meta} />
                                </div>
                            )}

                            {/* OVERTURE MAPS (when data is not primary) */}
                            {primaryLayer !== 'data' && (
                                <div className="rounded-xl p-4" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
                                    <div className="flex items-center gap-2.5 mb-3">
                                        <div className="w-7 h-7 rounded-lg bg-blue-500/10 flex items-center justify-center">
                                            <Layers size={13} className="text-blue-400" />
                                        </div>
                                        <h4 className="text-[13px] font-bold text-white tracking-tight">Overture Maps</h4>
                                    </div>
                                    <div className="grid grid-cols-2 gap-y-2.5 gap-x-4 text-[11px]">
                                        {selectedPoi.overture_meta?.update_time && (
                                            <div className="flex items-center gap-1.5 text-slate-500">
                                                <Clock size={11} className="text-slate-600 shrink-0" />
                                                <span>Updated <span className="text-slate-300 font-mono">{selectedPoi.overture_meta.update_time}</span></span>
                                            </div>
                                        )}
                                        {selectedPoi.overture_meta?.source && (
                                            <div className="flex items-center gap-1.5 text-slate-500">
                                                <ShieldCheck size={11} className="text-slate-600 shrink-0" />
                                                <span>Source <span className="text-slate-300">{selectedPoi.overture_meta.source}</span></span>
                                            </div>
                                        )}
                                        {selectedPoi.overture_meta?.confidence != null && (
                                            <div className="flex items-center gap-1.5 text-slate-500">
                                                <CheckCircle size={11} className="text-slate-600 shrink-0" />
                                                <span>Confidence <span className="text-slate-300 font-mono">{(selectedPoi.overture_meta.confidence * 100).toFixed(0)}%</span></span>
                                            </div>
                                        )}
                                        {selectedPoi.overture_meta?.brand && (
                                            <div className="flex items-center gap-1.5 text-slate-500">
                                                <Layers size={11} className="text-slate-600 shrink-0" />
                                                <span>Brand <span className="text-slate-300 font-medium">{selectedPoi.overture_meta.brand}</span></span>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* FOURSQUARE (when data is not primary, since primary already shows it) */}
                            {primaryLayer !== 'data' && selectedPoi.foursquare && (() => {
                                const fs = selectedPoi.foursquare;
                                const colors: Record<string, string> = { verified: '#10b981', closed: '#ef4444', mismatch: '#f59e0b' };
                                const c = colors[fs.status] || '#64748b';
                                return (
                                    <div className="rounded-xl p-4" style={{ background: `${c}06`, border: `1px solid ${c}15` }}>
                                        <div className="flex items-center gap-2.5 mb-3">
                                            <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ backgroundColor: `${c}12` }}>
                                                <ShieldCheck size={13} style={{ color: c }} />
                                            </div>
                                            <h4 className="text-[13px] font-bold text-white tracking-tight">Foursquare</h4>
                                            <VerificationBadge status={fs.status} />
                                        </div>
                                        <p className="text-[11px] text-slate-400 leading-relaxed mb-2.5">{fs.detail}</p>
                                        {fs.match && (
                                            <div className="grid grid-cols-2 gap-y-2 gap-x-4 text-[11px]">
                                                {fs.match.chain && (
                                                    <div className="text-slate-500">Chain <span className="text-slate-300 font-medium">{fs.match.chain}</span></div>
                                                )}
                                                {fs.match.category && (
                                                    <div className="text-slate-500">Type <span className="text-slate-300">{fs.match.category}</span></div>
                                                )}
                                                {fs.match.website && (
                                                    <div className="flex items-center gap-1 text-slate-500">
                                                        <Globe size={10} className="text-slate-600 shrink-0" />
                                                        <a href={fs.match.website} target="_blank" rel="noopener noreferrer"
                                                           className="text-blue-400 hover:text-blue-300 truncate transition-colors">
                                                            {fs.match.website.replace(/https?:\/\/(www\.)?/, '')}
                                                        </a>
                                                    </div>
                                                )}
                                                {fs.match.phone && (
                                                    <div className="flex items-center gap-1 text-slate-500">
                                                        <Phone size={10} className="text-slate-600 shrink-0" />
                                                        <span className="text-slate-300 font-mono">{fs.match.phone}</span>
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                );
                            })()}

                            {/* STREET VIEW */}
                            {selectedPoi.current_gallery?.length > 0 && (
                                selectedPoi.location_type === 'intersection' ? (
                                    <>
                                        {(() => {
                                            const a = selectedPoi.current_gallery.filter((i: any) => i.group === 'side_a');
                                            const b = selectedPoi.current_gallery.filter((i: any) => i.group === 'side_b');
                                            return (
                                                <>
                                                    {a.length > 0 && <GalleryStrip images={a} label="Street Side A" accent="#8b5cf6" />}
                                                    {b.length > 0 && <GalleryStrip images={b} label="Street Side B" accent="#06b6d4" />}
                                                </>
                                            );
                                        })()}
                                    </>
                                ) : (
                                    <GalleryStrip images={selectedPoi.current_gallery} label="Street View" accent="#3b82f6" />
                                )
                            )}

                            {/* FOOTER */}
                            <div className="pt-4 border-t border-white/[0.04]">
                                <div className="grid grid-cols-2 gap-2.5">
                                    <div className="rounded-lg p-3" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)' }}>
                                        <p className="text-[9px] text-slate-600 uppercase tracking-widest mb-0.5">Overture Updated</p>
                                        <p className="text-slate-300 text-[12px] font-semibold font-mono">{selectedPoi.overture_meta?.update_time || '—'}</p>
                                    </div>
                                    <div className="rounded-lg p-3" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)' }}>
                                        <p className="text-[9px] text-slate-600 uppercase tracking-widest mb-0.5">Street Views</p>
                                        <p className="text-slate-300 text-[12px] font-semibold font-mono">{selectedPoi.current_gallery?.length || 0} images</p>
                                    </div>
                                </div>
                                <div className="mt-3 text-center">
                                    <p className="text-[10px] text-slate-700 font-mono">{selectedPoi.location[1].toFixed(5)}, {selectedPoi.location[0].toFixed(5)}</p>
                                    <p className="text-[8px] text-slate-800 mt-1 uppercase tracking-[0.2em]">Overture Maps · Foursquare · Mapillary · OCR + CLIP</p>
                                </div>
                            </div>
                        </div>
                    </div>
                );
            })()}
        </div>
    );
}
