'use client';

import Link from 'next/link';
import { Star, ExternalLink } from 'lucide-react';
import type { TickerFavorito } from '@/lib/types';

interface Props {
    favorites: TickerFavorito[];
}

export default function FavoriteTickers({ favorites }: Props) {
    if (!favorites || favorites.length === 0) {
        return (
            <div className="glass-panel p-5">
                <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                        style={{ background: 'rgba(245,158,11,0.12)' }}>
                        <Star className="w-5 h-5" style={{ color: 'var(--warning)' }} />
                    </div>
                    <h3 className="text-sm font-semibold" style={{ color: 'var(--text-main)' }}>
                        Tickers Favoritos
                    </h3>
                </div>
                <div
                    className="text-center py-8 rounded-xl"
                    style={{ background: 'rgba(15,23,42,0.3)', border: '1px dashed var(--border-light)' }}
                >
                    <Star className="w-8 h-8 mx-auto mb-2" style={{ color: 'var(--text-muted)' }} />
                    <p className="text-sm font-medium mb-1" style={{ color: 'var(--text-muted)' }}>
                        Sin tickers favoritos
                    </p>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        Agrega tickers favoritos desde el optimizador para acceso rápido.
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
                        style={{ background: 'rgba(245,158,11,0.12)' }}>
                        <Star className="w-5 h-5" style={{ color: 'var(--warning)' }} />
                    </div>
                    <h3 className="text-sm font-semibold" style={{ color: 'var(--text-main)' }}>
                        Tickers Favoritos
                    </h3>
                </div>
                <span className="text-xs px-2 py-1 rounded-full" style={{
                    background: 'rgba(245,158,11,0.1)',
                    color: 'var(--warning)',
                    border: '1px solid rgba(245,158,11,0.2)',
                }}>
                    {favorites.length}
                </span>
            </div>

            <div className="overflow-x-auto">
                <table className="w-full text-sm">
                    <thead>
                        <tr style={{ borderBottom: '1px solid var(--border-light)' }}>
                            <th className="pb-2 text-left text-xs uppercase tracking-widest font-semibold"
                                style={{ color: 'var(--text-muted)' }}>Ticker</th>
                            <th className="pb-2 text-left text-xs uppercase tracking-widest font-semibold"
                                style={{ color: 'var(--text-muted)' }}>Empresa</th>
                            <th className="pb-2 text-right text-xs uppercase tracking-widest font-semibold"
                                style={{ color: 'var(--text-muted)' }}>Precio</th>
                            <th className="pb-2 text-right text-xs uppercase tracking-widest font-semibold"
                                style={{ color: 'var(--text-muted)' }}>Cambio</th>
                        </tr>
                    </thead>
                    <tbody>
                        {favorites.map((fav) => (
                            <tr
                                key={fav.id}
                                className="transition-colors"
                                style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}
                            >
                                <td className="py-2.5">
                                    <span
                                        className="text-sm font-bold"
                                        style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-primary)' }}
                                    >
                                        {fav.ticker}
                                    </span>
                                </td>
                                <td className="py-2.5">
                                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                                        {fav.nombre || fav.ticker}
                                    </span>
                                </td>
                                <td className="py-2.5 text-right">
                                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                                        —
                                    </span>
                                </td>
                                <td className="py-2.5 text-right">
                                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                                        —
                                    </span>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <p className="text-xs mt-3 text-center" style={{ color: 'var(--text-muted)' }}>
                Precios en tiempo real disponibles próximamente.
            </p>
        </div>
    );
}
