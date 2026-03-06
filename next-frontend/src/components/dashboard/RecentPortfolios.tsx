'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Briefcase, TrendingUp, Info, ExternalLink } from 'lucide-react';
import type { Portafolio } from '@/lib/types';
import SkeletonCard from './SkeletonCard';

interface Props {
    portfolios: Portafolio[];
}

function formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return 'Hoy';
    if (diffDays === 1) return 'Ayer';
    if (diffDays < 7) return `Hace ${diffDays} días`;
    if (diffDays < 30) return `Hace ${Math.floor(diffDays / 7)} sem.`;
    return date.toLocaleDateString('es-MX', { day: 'numeric', month: 'short', year: 'numeric' });
}

function SharpeTooltip() {
    const [visible, setVisible] = useState(false);
    return (
        <span
            className="relative inline-flex items-center cursor-help ml-1"
            onMouseEnter={() => setVisible(true)}
            onMouseLeave={() => setVisible(false)}
        >
            <Info className="w-3.5 h-3.5" style={{ color: 'var(--text-muted)' }} />
            {visible && (
                <span
                    className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 text-xs rounded-lg whitespace-nowrap z-50"
                    style={{
                        background: 'rgba(15,23,42,0.95)',
                        border: '1px solid var(--border-light)',
                        color: 'var(--text-main)',
                        boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
                    }}
                >
                    Retorno ajustado por riesgo. Mayor = mejor.
                </span>
            )}
        </span>
    );
}

export default function RecentPortfolios({ portfolios }: Props) {
    if (!portfolios || portfolios.length === 0) {
        return (
            <div className="glass-panel p-5">
                <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                        style={{ background: 'rgba(20,184,166,0.12)' }}>
                        <Briefcase className="w-5 h-5" style={{ color: 'var(--accent-secondary)' }} />
                    </div>
                    <h3 className="text-sm font-semibold" style={{ color: 'var(--text-main)' }}>
                        Portafolios Recientes
                    </h3>
                </div>
                <div
                    className="text-center py-8 rounded-xl"
                    style={{ background: 'rgba(15,23,42,0.3)', border: '1px dashed var(--border-light)' }}
                >
                    <Briefcase className="w-8 h-8 mx-auto mb-2" style={{ color: 'var(--text-muted)' }} />
                    <p className="text-sm font-medium mb-1" style={{ color: 'var(--text-muted)' }}>
                        Sin portafolios guardados
                    </p>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        Crea tu primer portafolio desde el optimizador.
                    </p>
                    <Link
                        href="/dashboard/optimizar"
                        className="inline-flex items-center gap-1 mt-3 text-xs font-semibold"
                        style={{ color: 'var(--accent-primary)' }}
                    >
                        Ir al Optimizador <ExternalLink className="w-3 h-3" />
                    </Link>
                </div>
            </div>
        );
    }

    return (
        <div className="glass-panel p-5">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                        style={{ background: 'rgba(20,184,166,0.12)' }}>
                        <Briefcase className="w-5 h-5" style={{ color: 'var(--accent-secondary)' }} />
                    </div>
                    <h3 className="text-sm font-semibold" style={{ color: 'var(--text-main)' }}>
                        Portafolios Recientes
                    </h3>
                </div>
                <span className="text-xs px-2 py-1 rounded-full" style={{
                    background: 'rgba(20,184,166,0.1)',
                    color: 'var(--accent-secondary)',
                    border: '1px solid rgba(20,184,166,0.2)',
                }}>
                    {portfolios.length}
                </span>
            </div>

            <div className="flex flex-col gap-3">
                {portfolios.map((p) => (
                    <div
                        key={p.id}
                        className="p-4 rounded-xl transition-colors"
                        style={{
                            background: 'rgba(15,23,42,0.4)',
                            border: '1px solid var(--border-light)',
                        }}
                    >
                        <div className="flex items-start justify-between mb-3">
                            <div>
                                <p className="text-sm font-semibold" style={{ color: 'var(--text-main)' }}>
                                    {p.nombre}
                                </p>
                                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                                    {formatDate(p.created_at)}
                                </p>
                            </div>
                            <Link
                                href={`/dashboard/portafolio/${p.id}`}
                                className="text-xs font-medium px-2.5 py-1 rounded-lg transition-colors"
                                style={{
                                    background: 'rgba(37,99,235,0.1)',
                                    color: 'var(--accent-primary)',
                                    border: '1px solid rgba(37,99,235,0.2)',
                                }}
                            >
                                Ver detalle
                            </Link>
                        </div>

                        <div className="grid grid-cols-3 gap-3">
                            <div>
                                <p className="text-xs flex items-center" style={{ color: 'var(--text-muted)' }}>
                                    Sharpe <SharpeTooltip />
                                </p>
                                <p className="text-sm font-bold" style={{
                                    fontFamily: 'var(--font-mono)',
                                    color: (p.sharpe_ratio ?? 0) >= 1 ? 'var(--success)' : 'var(--text-main)',
                                }}>
                                    {p.sharpe_ratio != null ? p.sharpe_ratio.toFixed(2) : '—'}
                                </p>
                            </div>
                            <div>
                                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Rend. Esperado</p>
                                <p className="text-sm font-bold" style={{
                                    fontFamily: 'var(--font-mono)',
                                    color: (p.rendimiento_pct ?? 0) > 0 ? 'var(--success)' : 'var(--danger)',
                                }}>
                                    {p.rendimiento_pct != null ? `${p.rendimiento_pct.toFixed(1)}%` : '—'}
                                </p>
                            </div>
                            <div>
                                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Rend. Real</p>
                                <p className="text-sm font-bold" style={{
                                    fontFamily: 'var(--font-mono)',
                                    color: 'var(--text-muted)',
                                }}>
                                    —
                                </p>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
