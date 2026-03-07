'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { API_BASE } from '@/lib/constants';
import { useNotification } from '@/hooks/useNotification';
import { getErrorMessage, formatErrorToast } from '@/utils/errorMessages';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ToleranciaData {
    perfil_resultado: string;
    volatilidad_maxima: number;   // fraction: 0.08, 0.12 …
    puntaje_total: number;
    fecha_completado: string;
}

interface SuenosData {
    retorno_minimo_requerido: number;  // fraction: 0.05, 0.12 …
    nivel: string;
    anos_horizonte: number;
    meta_tipo: string;
    moneda: string;
}

interface HistorialRow {
    fecha: string;
    perfil_resultado: string;
    retorno_minimo: number;
    volatilidad_maxima: number;
}

interface Props {
    tolerancia: ToleranciaData | null;
    suenos: SuenosData | null;
    historial: HistorialRow[];
}

// ---------------------------------------------------------------------------
// Static metadata
// ---------------------------------------------------------------------------

const PERFIL_META: Record<string, {
    icon: string;
    label: string;
    color: string;
    bg: string;
    description: string;
}> = {
    conservador: {
        icon: '🌿',
        label: 'Inversionista Conservador',
        color: '#14B8A6',
        bg: 'rgba(20,184,166,0.12)',
        description:
            'Priorizas la preservación de tu capital por encima del crecimiento. Tu portafolio ideal se basa en renta fija y activos de bajo riesgo que te den tranquilidad y estabilidad.',
    },
    moderado: {
        icon: '🌱',
        label: 'Inversionista Moderado',
        color: '#2563EB',
        bg: 'rgba(37,99,235,0.12)',
        description:
            'Buscas un balance entre seguridad y crecimiento. Aceptas algo de volatilidad a cambio de mejores rendimientos. Un portafolio mixto con predominancia de renta fija es tu punto óptimo.',
    },
    balanceado: {
        icon: '⚖️',
        label: 'Inversionista Balanceado',
        color: '#8B5CF6',
        bg: 'rgba(139,92,246,0.12)',
        description:
            'Aceptas volatilidad moderada a cambio de un mayor potencial de rendimiento. Una mezcla equilibrada entre renta fija y variable refleja bien tus objetivos financieros.',
    },
    crecimiento: {
        icon: '🚀',
        label: 'Inversionista de Crecimiento',
        color: '#F59E0B',
        bg: 'rgba(245,158,11,0.12)',
        description:
            'Tu objetivo principal es el crecimiento del capital a largo plazo. Tienes mayor tolerancia al riesgo y un portafolio con alta exposición a renta variable encaja con tus metas.',
    },
    agresivo: {
        icon: '🔥',
        label: 'Inversionista Agresivo',
        color: '#EF4444',
        bg: 'rgba(239,68,68,0.12)',
        description:
            'Tienes alta tolerancia al riesgo y buscas maximizar el rendimiento. Un portafolio concentrado en renta variable y activos de alto crecimiento es tu terreno natural.',
    },
};

const META_TIPO_LABELS: Record<string, string> = {
    casa: 'Compra de casa',
    retiro: 'Retiro / Jubilación',
    educacion: 'Educación',
    viaje: 'Viaje',
    libertad: 'Libertad financiera',
    otra: 'Meta personal',
};

function getPerfilMeta(perfil: string) {
    return PERFIL_META[perfil?.toLowerCase()] ?? PERFIL_META['moderado'];
}

function pct(fraction: number, decimals = 1) {
    return `${(fraction * 100).toFixed(decimals)}%`;
}

function formatDate(iso: string) {
    return new Date(iso).toLocaleDateString('es-MX', {
        day: '2-digit', month: 'short', year: 'numeric',
    });
}

// ---------------------------------------------------------------------------
// Tooltip
// ---------------------------------------------------------------------------

function Tooltip({ text, children }: { text: string; children: React.ReactNode }) {
    const [show, setShow] = useState(false);
    return (
        <span
            style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}
            onMouseEnter={() => setShow(true)}
            onMouseLeave={() => setShow(false)}
        >
            {children}
            {show && (
                <span style={{
                    position: 'absolute', bottom: '130%', left: '50%', transform: 'translateX(-50%)',
                    background: 'rgba(11,17,32,0.97)', border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '8px', padding: '7px 12px', fontSize: '0.75rem', color: '#94A3B8',
                    whiteSpace: 'nowrap', zIndex: 50, pointerEvents: 'none',
                    boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
                }}>
                    {text}
                </span>
            )}
        </span>
    );
}

// ---------------------------------------------------------------------------
// Skeletons
// ---------------------------------------------------------------------------

function SkeletonBlock({ h = 20, w = '100%', radius = 8 }: { h?: number; w?: number | string; radius?: number }) {
    return (
        <div style={{
            height: h, width: w, borderRadius: radius,
            background: 'rgba(255,255,255,0.06)',
            animation: 'pulse 1.5s ease-in-out infinite',
        }} />
    );
}

// ---------------------------------------------------------------------------
// Missing test banner
// ---------------------------------------------------------------------------

function MissingTestBanner({ test, href, label }: { test: string; href: string; label: string }) {
    return (
        <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px',
            padding: '14px 18px', borderRadius: '12px', flexWrap: 'wrap',
            background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.3)',
        }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span style={{ fontSize: '1.2rem' }}>⚠️</span>
                <div>
                    <p style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: '2px' }}>
                        Falta completar: <span style={{ color: '#F59E0B' }}>{label}</span>
                    </p>
                    <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                        Necesitamos esta información para calcular tu perfil completo.
                    </p>
                </div>
            </div>
            <Link
                href={href}
                style={{
                    padding: '8px 18px', borderRadius: '8px', fontSize: '0.82rem', fontWeight: 600,
                    background: 'rgba(245,158,11,0.2)', color: '#F59E0B',
                    border: '1px solid rgba(245,158,11,0.4)', textDecoration: 'none',
                    whiteSpace: 'nowrap', transition: 'all 0.2s ease',
                }}
            >
                Completar ahora →
            </Link>
        </div>
    );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

const PAGE_SIZE = 10;

export default function PerfilInversor({ tolerancia, suenos, historial }: Props) {
    const router = useRouter();
    const [recalcLoading, setRecalcLoading] = useState(false);
    const [recalcError, setRecalcError] = useState<string | null>(null);
    const [recalcOk, setRecalcOk] = useState(false);
    const [histPage, setHistPage] = useState(0);
    const notify = useNotification();

    const missingTolerancia = !tolerancia;
    const missingSuenos = !suenos;
    const bothComplete = !missingTolerancia && !missingSuenos;

    const meta = tolerancia ? getPerfilMeta(tolerancia.perfil_resultado) : null;

    const totalHistPages = Math.ceil(historial.length / PAGE_SIZE);
    const histPage_rows = historial.slice(histPage * PAGE_SIZE, (histPage + 1) * PAGE_SIZE);

    const yearMeta = suenos
        ? new Date().getFullYear() + suenos.anos_horizonte
        : null;

    async function handleRecalcular() {
        setRecalcLoading(true);
        setRecalcError(null);
        setRecalcOk(false);
        try {
            const res = await fetch(`${API_BASE}/api/ml/recalcular-perfil`, { method: 'POST' });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            setRecalcOk(true);
            notify.success('Perfil recalculado exitosamente.');
            router.refresh();
        } catch (err) {
            const em = getErrorMessage('PROFILE_RECALC_FAILED');
            setRecalcError(em.message);
            notify.error(formatErrorToast(em));
        } finally {
            setRecalcLoading(false);
        }
    }

    return (
        <div className="max-w-[860px] mx-auto">
            {/* Page header */}
            <div className="mb-6">
                <h2 className="text-xl font-bold" style={{ fontFamily: 'var(--font-display)' }}>
                    Mi Perfil de Inversionista
                </h2>
                <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
                    Calculado a partir de tus tests de sueños y tolerancia al riesgo.
                </p>
            </div>

            {/* Missing tests banners */}
            {(missingTolerancia || missingSuenos) && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginBottom: '24px' }}>
                    {missingSuenos && (
                        <MissingTestBanner
                            test="suenos"
                            href="/dashboard/tests?test=suenos"
                            label="Test de Sueños"
                        />
                    )}
                    {missingTolerancia && (
                        <MissingTestBanner
                            test="tolerancia"
                            href="/dashboard/tests?test=tolerancia"
                            label="Test de Tolerancia al Riesgo"
                        />
                    )}
                    {!bothComplete && (
                        <div className="glass-panel p-8 text-center" style={{ marginTop: '8px' }}>
                            <div style={{ fontSize: '2.5rem', marginBottom: '12px' }}>🧩</div>
                            <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.1rem', fontWeight: 700, marginBottom: '6px' }}>
                                Completa ambos tests para ver tu perfil
                            </h3>
                            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                                Tu perfil combina tu situación financiera, experiencia inversora y metas personales.
                            </p>
                        </div>
                    )}
                </div>
            )}

            {/* Full profile — only when both tests are done */}
            {bothComplete && meta && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

                    {/* ── Main profile card ─────────────────────────── */}
                    <div
                        className="glass-panel p-6"
                        style={{ borderTop: `3px solid ${meta.color}` }}
                    >
                        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px', flexWrap: 'wrap' }}>
                            <div style={{
                                width: '64px', height: '64px', borderRadius: '16px', flexShrink: 0,
                                background: meta.bg, display: 'flex', alignItems: 'center',
                                justifyContent: 'center', fontSize: '2rem',
                                border: `1px solid ${meta.color}30`,
                            }}>
                                {meta.icon}
                            </div>
                            <div style={{ flex: 1, minWidth: '200px' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap', marginBottom: '6px' }}>
                                    <h3 style={{
                                        fontFamily: 'var(--font-display)', fontSize: '1.35rem',
                                        fontWeight: 800, color: meta.color,
                                    }}>
                                        {meta.label}
                                    </h3>
                                    <span style={{
                                        padding: '3px 12px', borderRadius: '99px', fontSize: '0.72rem',
                                        fontWeight: 700, background: meta.bg, color: meta.color,
                                        border: `1px solid ${meta.color}40`, textTransform: 'capitalize',
                                    }}>
                                        {tolerancia.perfil_resultado}
                                    </span>
                                </div>
                                <p style={{ color: 'var(--text-muted)', fontSize: '0.86rem', lineHeight: 1.6 }}>
                                    {meta.description}
                                </p>
                                <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '10px' }}>
                                    Puntaje de riesgo: <span style={{ fontFamily: 'var(--font-mono)', color: meta.color, fontWeight: 700 }}>{tolerancia.puntaje_total}/20</span>
                                    {' · '}
                                    Calculado el {formatDate(tolerancia.fecha_completado)}
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* ── Key metrics ────────────────────────────────── */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px' }}>
                        {/* Retorno mínimo */}
                        <div className="glass-panel p-5">
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
                                <span style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>
                                    Retorno mínimo requerido
                                </span>
                                <Tooltip text="Lo calculamos a partir de tu meta y horizonte de inversión">
                                    <span style={{
                                        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                                        width: '16px', height: '16px', borderRadius: '50%', cursor: 'default',
                                        background: 'rgba(255,255,255,0.07)', color: 'var(--text-muted)',
                                        fontSize: '0.65rem', fontWeight: 700,
                                    }}>?</span>
                                </Tooltip>
                            </div>
                            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '2rem', fontWeight: 800, color: '#14B8A6' }}>
                                {pct(suenos.retorno_minimo_requerido)}
                            </div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                                anual para alcanzar tu meta
                            </div>
                        </div>

                        {/* Volatilidad máxima */}
                        <div className="glass-panel p-5">
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
                                <span style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>
                                    Volatilidad máxima
                                </span>
                                <Tooltip text="Es el nivel de riesgo que tu perfil puede tolerar">
                                    <span style={{
                                        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                                        width: '16px', height: '16px', borderRadius: '50%', cursor: 'default',
                                        background: 'rgba(255,255,255,0.07)', color: 'var(--text-muted)',
                                        fontSize: '0.65rem', fontWeight: 700,
                                    }}>?</span>
                                </Tooltip>
                            </div>
                            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '2rem', fontWeight: 800, color: meta.color }}>
                                {pct(tolerancia.volatilidad_maxima)}
                            </div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                                volatilidad anual tolerable
                            </div>
                        </div>
                    </div>

                    {/* ── Additional details ─────────────────────────── */}
                    <div className="glass-panel p-5">
                        <h4 style={{ fontFamily: 'var(--font-display)', fontSize: '0.95rem', fontWeight: 700, marginBottom: '16px' }}>
                            Detalles de tu perfil
                        </h4>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '12px' }}>
                            {[
                                {
                                    label: 'Horizonte de inversión',
                                    value: `${suenos.anos_horizonte} años`,
                                    sub: yearMeta ? `hasta ${yearMeta}` : undefined,
                                    icon: '📅',
                                },
                                {
                                    label: 'Meta principal',
                                    value: META_TIPO_LABELS[suenos.meta_tipo] ?? suenos.meta_tipo,
                                    icon: '🎯',
                                },
                                {
                                    label: 'Moneda de referencia',
                                    value: suenos.moneda,
                                    icon: '💱',
                                },
                                {
                                    label: 'Nivel de sueños',
                                    value: suenos.nivel.charAt(0).toUpperCase() + suenos.nivel.slice(1),
                                    icon: '✨',
                                },
                            ].map((item) => (
                                <div key={item.label} style={{
                                    padding: '12px', borderRadius: '10px',
                                    background: 'rgba(255,255,255,0.03)',
                                    border: '1px solid rgba(255,255,255,0.06)',
                                }}>
                                    <div style={{ fontSize: '1.1rem', marginBottom: '6px' }}>{item.icon}</div>
                                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '2px' }}>
                                        {item.label}
                                    </div>
                                    <div style={{ fontSize: '0.9rem', fontWeight: 600 }}>{item.value}</div>
                                    {item.sub && (
                                        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '1px' }}>
                                            {item.sub}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* ── History table ──────────────────────────────── */}
                    <div className="glass-panel overflow-hidden">
                        <div style={{ padding: '16px 20px 12px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--border-light)' }}>
                            <h4 style={{ fontFamily: 'var(--font-display)', fontSize: '0.95rem', fontWeight: 700 }}>
                                Historial de perfiles
                            </h4>
                            {historial.length > 0 && (
                                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                                    {historial.length} registro{historial.length !== 1 ? 's' : ''}
                                </span>
                            )}
                        </div>

                        {historial.length === 0 ? (
                            <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                                Sin historial aún. El historial se genera cuando recalculas tu perfil.
                            </div>
                        ) : (
                            <>
                                <div style={{ overflowX: 'auto' }}>
                                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.84rem' }}>
                                        <thead>
                                            <tr style={{ borderBottom: '1px solid var(--border-light)' }}>
                                                {['Fecha', 'Clasificación', 'Retorno Mín.', 'Volatilidad Máx.'].map((h) => (
                                                    <th key={h} style={{
                                                        padding: '10px 20px', textAlign: 'left',
                                                        fontSize: '0.7rem', fontWeight: 700,
                                                        textTransform: 'uppercase', letterSpacing: '0.05em',
                                                        color: 'var(--text-muted)',
                                                    }}>
                                                        {h}
                                                    </th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {histPage_rows.map((row, i) => {
                                                const m = getPerfilMeta(row.perfil_resultado);
                                                return (
                                                    <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                                                        <td style={{ padding: '12px 20px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                                                            {formatDate(row.fecha)}
                                                        </td>
                                                        <td style={{ padding: '12px 20px' }}>
                                                            <span style={{
                                                                padding: '3px 10px', borderRadius: '99px', fontSize: '0.72rem',
                                                                fontWeight: 600, background: m.bg, color: m.color,
                                                                border: `1px solid ${m.color}30`, textTransform: 'capitalize',
                                                            }}>
                                                                {m.icon} {row.perfil_resultado}
                                                            </span>
                                                        </td>
                                                        <td style={{ padding: '12px 20px', fontFamily: 'var(--font-mono)', color: '#14B8A6' }}>
                                                            {pct(row.retorno_minimo)}
                                                        </td>
                                                        <td style={{ padding: '12px 20px', fontFamily: 'var(--font-mono)', color: m.color }}>
                                                            {pct(row.volatilidad_maxima)}
                                                        </td>
                                                    </tr>
                                                );
                                            })}
                                        </tbody>
                                    </table>
                                </div>

                                {/* Pagination */}
                                {totalHistPages > 1 && (
                                    <div style={{
                                        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                                        padding: '12px 20px', borderTop: '1px solid var(--border-light)',
                                    }}>
                                        <button
                                            onClick={() => setHistPage((p) => Math.max(0, p - 1))}
                                            disabled={histPage === 0}
                                            style={{
                                                padding: '6px 14px', borderRadius: '8px', fontSize: '0.8rem', fontWeight: 500,
                                                border: '1px solid var(--border-light)', cursor: histPage === 0 ? 'not-allowed' : 'pointer',
                                                background: 'rgba(255,255,255,0.04)', color: histPage === 0 ? 'var(--text-muted)' : 'var(--text-main)',
                                                transition: 'all 0.15s ease',
                                            }}
                                        >
                                            ← Anterior
                                        </button>
                                        <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                                            Página {histPage + 1} de {totalHistPages}
                                        </span>
                                        <button
                                            onClick={() => setHistPage((p) => Math.min(totalHistPages - 1, p + 1))}
                                            disabled={histPage === totalHistPages - 1}
                                            style={{
                                                padding: '6px 14px', borderRadius: '8px', fontSize: '0.8rem', fontWeight: 500,
                                                border: '1px solid var(--border-light)',
                                                cursor: histPage === totalHistPages - 1 ? 'not-allowed' : 'pointer',
                                                background: 'rgba(255,255,255,0.04)',
                                                color: histPage === totalHistPages - 1 ? 'var(--text-muted)' : 'var(--text-main)',
                                                transition: 'all 0.15s ease',
                                            }}
                                        >
                                            Siguiente →
                                        </button>
                                    </div>
                                )}
                            </>
                        )}
                    </div>

                    {/* ── Action buttons ─────────────────────────────── */}
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
                        <Link
                            href="/dashboard/tests?test=suenos"
                            style={{
                                flex: '1 1 160px', padding: '11px 18px', borderRadius: '10px',
                                fontSize: '0.84rem', fontWeight: 600, textAlign: 'center', textDecoration: 'none',
                                background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-light)',
                                color: 'var(--text-main)', transition: 'all 0.2s ease',
                            }}
                        >
                            ✏️ Actualizar Test de Sueños
                        </Link>
                        <Link
                            href="/dashboard/tests?test=tolerancia"
                            style={{
                                flex: '1 1 160px', padding: '11px 18px', borderRadius: '10px',
                                fontSize: '0.84rem', fontWeight: 600, textAlign: 'center', textDecoration: 'none',
                                background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-light)',
                                color: 'var(--text-main)', transition: 'all 0.2s ease',
                            }}
                        >
                            ✏️ Actualizar Test de Riesgo
                        </Link>
                        <button
                            onClick={handleRecalcular}
                            disabled={recalcLoading}
                            style={{
                                flex: '1 1 160px', padding: '11px 18px', borderRadius: '10px',
                                fontSize: '0.84rem', fontWeight: 600, border: 'none', cursor: recalcLoading ? 'wait' : 'pointer',
                                background: recalcLoading ? 'rgba(37,99,235,0.2)' : 'linear-gradient(135deg, #2563EB, #1D4ED8)',
                                color: 'white', transition: 'all 0.2s ease',
                                boxShadow: recalcLoading ? 'none' : '0 4px 16px rgba(37,99,235,0.3)',
                            }}
                        >
                            {recalcLoading ? '⏳ Recalculando...' : '🔄 Recalcular perfil completo'}
                        </button>
                    </div>

                    {/* Feedback messages */}
                    {recalcOk && (
                        <div style={{ padding: '12px 16px', borderRadius: '10px', background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)', fontSize: '0.84rem', color: '#6ee7b7' }}>
                            ✅ Perfil recalculado exitosamente.
                        </div>
                    )}
                    {recalcError && (
                        <div style={{ padding: '12px 16px', borderRadius: '10px', background: 'rgba(239,68,68,0.1)', border: '1px solid var(--danger)', fontSize: '0.84rem', color: '#fca5a5' }}>
                            {recalcError}
                        </div>
                    )}

                </div>
            )}

            {/* keyframe animation for skeletons */}
            <style>{`@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }`}</style>
        </div>
    );
}
