import type { Metadata } from 'next';
import { redirect, notFound } from 'next/navigation';
import Link from 'next/link';
import { createClient } from '@/utils/supabase/server';
import { ArrowLeft, TrendingUp } from 'lucide-react';
import VersionHistorial from './VersionHistorial';

export const metadata: Metadata = {
    title: 'Detalle de Estrategia — Kaudal',
    description: 'Historial de versiones y metricas de tu estrategia.',
};

interface PageProps {
    params: Promise<{ id: string }>;
}

export default async function EstrategiaDetailPage({ params }: PageProps) {
    const { id } = await params;
    const supabase = await createClient();
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) redirect('/login');

    // 1. Fetch the strategy (verify ownership)
    const { data: estrategia } = await supabase
        .from('estrategias')
        .select('id, nombre, tipo, parametros, created_at')
        .eq('id', id)
        .eq('user_id', user.id)
        .single();

    if (!estrategia) notFound();

    // 2. Fetch all portfolio versions linked to this strategy
    const { data: portafolios } = await supabase
        .from('portafolios')
        .select('id, nombre, rendimiento_pct, volatilidad_pct, sharpe_ratio, allocation, created_at')
        .eq('estrategia_id', id)
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });

    const versions = (portafolios ?? []).map((p) => ({
        id: p.id as string,
        nombre: p.nombre as string,
        rendimiento_pct: p.rendimiento_pct as number | null,
        volatilidad_pct: p.volatilidad_pct as number | null,
        sharpe_ratio: p.sharpe_ratio as number | null,
        allocation: (p.allocation ?? []) as { ticker: string; weight_pct: number; shares: number }[],
        created_at: p.created_at as string,
    }));

    const tipoLabel = estrategia.tipo === 'markowitz' ? 'Markowitz' : 'HRP';
    const tickers = (estrategia.parametros as Record<string, unknown>)?.tickers as string[] | undefined;
    const fecha = new Date(estrategia.created_at).toLocaleDateString('es-MX', {
        day: 'numeric', month: 'short', year: 'numeric',
    });

    return (
        <div className="max-w-[1200px] mx-auto">
            {/* Back link */}
            <Link
                href="/dashboard/estrategias"
                className="inline-flex items-center gap-1.5 text-sm mb-5 transition-colors hover:opacity-80"
                style={{ color: 'var(--text-muted)' }}
            >
                <ArrowLeft className="w-4 h-4" />
                Volver a Estrategias
            </Link>

            {/* Header */}
            <div className="glass-panel p-6 mb-6" style={{ borderTop: '4px solid var(--accent-primary)' }}>
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                        <div
                            className="w-12 h-12 rounded-xl flex items-center justify-center"
                            style={{ background: 'rgba(37,99,235,0.12)' }}
                        >
                            <TrendingUp className="w-6 h-6" style={{ color: 'var(--accent-primary)' }} />
                        </div>
                        <div>
                            <h2
                                className="text-xl font-bold"
                                style={{ fontFamily: 'var(--font-display)', color: 'var(--text-main)' }}
                            >
                                {estrategia.nombre}
                            </h2>
                            <p className="text-sm flex items-center gap-2 mt-0.5" style={{ color: 'var(--text-muted)' }}>
                                <span
                                    className="px-2 py-0.5 rounded text-[10px] font-medium"
                                    style={{
                                        background: 'rgba(37,99,235,0.15)',
                                        color: 'var(--accent-primary)',
                                    }}
                                >
                                    {tipoLabel}
                                </span>
                                Creada {fecha}
                            </p>
                        </div>
                    </div>

                    {tickers && tickers.length > 0 && (
                        <div className="flex flex-wrap gap-1.5">
                            {tickers.map((t) => (
                                <span
                                    key={t}
                                    className="px-2 py-1 rounded-md text-xs font-medium"
                                    style={{
                                        background: 'rgba(37,99,235,0.1)',
                                        border: '1px solid rgba(37,99,235,0.2)',
                                        color: 'var(--accent-primary)',
                                    }}
                                >
                                    {t}
                                </span>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            {/* Version History */}
            <div className="mb-6">
                <h3
                    className="text-sm font-semibold mb-4 flex items-center gap-2"
                    style={{ color: 'var(--text-main)' }}
                >
                    Historial de Versiones
                    <span
                        className="text-xs font-normal px-2 py-0.5 rounded-full"
                        style={{
                            background: 'rgba(37,99,235,0.1)',
                            color: 'var(--accent-primary)',
                        }}
                    >
                        {versions.length}
                    </span>
                </h3>

                <VersionHistorial
                    versions={versions}
                    estrategiaTipo={estrategia.tipo}
                />
            </div>
        </div>
    );
}
