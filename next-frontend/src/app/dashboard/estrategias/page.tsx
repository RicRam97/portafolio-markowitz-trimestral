import type { Metadata } from 'next';
import { redirect } from 'next/navigation';
import Link from 'next/link';
import { createClient } from '@/utils/supabase/server';
import type { PlanTier } from '@/lib/types';
import { PLAN_LIMITS, PLAN_LABELS, PLAN_UPGRADE_TARGET } from '@/lib/constants';
import { Layers, ArrowUpRight } from 'lucide-react';
import EstrategiaDetail from './EstrategiaDetail';
import EstrategiasEmptyState from './EstrategiasEmptyState';

export const metadata: Metadata = {
    title: 'Estrategias — Kaudal',
    description: 'Consulta y administra tus estrategias de inversion.',
};

export default async function EstrategiasPage() {
    const supabase = await createClient();
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) redirect('/login');

    const [profileRes, estrategiasRes] = await Promise.all([
        supabase.from('profiles').select('plan').eq('id', user.id).single(),
        supabase
            .from('estrategias')
            .select(`
                id,
                nombre,
                tipo,
                parametros,
                created_at,
                portafolios (
                    presupuesto,
                    rendimiento_pct,
                    volatilidad_pct,
                    sharpe_ratio,
                    allocation,
                    metricas
                )
            `)
            .eq('user_id', user.id)
            .order('created_at', { ascending: false }),
    ]);

    const plan = (profileRes.data?.plan || 'basico') as PlanTier;
    const limit = PLAN_LIMITS[plan];
    const estrategias = estrategiasRes.data ?? [];
    const count = estrategias.length;

    const ratio = limit > 0 ? count / limit : 1;
    const percentage = Math.min(ratio * 100, 100);
    const barColor =
        ratio >= 1 ? 'var(--danger)' : ratio >= 0.7 ? 'var(--warning)' : 'var(--success)';
    const atLimit = count >= limit;
    const upgradeTo = PLAN_UPGRADE_TARGET[plan];

    return (
        <div className="max-w-[1200px] mx-auto">
            {/* Header */}
            <div className="mb-6">
                <h2 className="text-xl font-bold" style={{ fontFamily: 'var(--font-display)' }}>
                    Mis Estrategias
                </h2>
                <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
                    Consulta tus estrategias guardadas, metricas y fronteras eficientes.
                </p>
            </div>

            {/* Progress Bar */}
            <div className="glass-panel p-5 mb-6">
                <div className="flex items-center gap-3 mb-4">
                    <div
                        className="w-10 h-10 rounded-xl flex items-center justify-center"
                        style={{ background: 'rgba(37,99,235,0.12)' }}
                    >
                        <Layers className="w-5 h-5" style={{ color: 'var(--accent-primary)' }} />
                    </div>
                    <div>
                        <h3 className="text-sm font-semibold" style={{ color: 'var(--text-main)' }}>
                            Uso de Estrategias
                        </h3>
                        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                            Plan {PLAN_LABELS[plan]}
                        </p>
                    </div>
                </div>

                <div className="mb-3">
                    <div className="flex items-baseline justify-between mb-2">
                        <span
                            className="text-2xl font-bold"
                            style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-main)' }}
                        >
                            {count}
                            <span className="text-sm font-normal" style={{ color: 'var(--text-muted)' }}>
                                {' '}de {limit}
                            </span>
                        </span>
                        <span className="text-xs font-medium" style={{ color: barColor }}>
                            {Math.round(percentage)}%
                        </span>
                    </div>

                    <div className="progress-bar-track">
                        <div
                            className="progress-bar-fill"
                            style={{ width: `${percentage}%`, backgroundColor: barColor }}
                        />
                    </div>
                </div>

                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {atLimit
                        ? 'Has alcanzado el limite de tu plan actual.'
                        : `Puedes crear ${limit - count} estrategia${limit - count !== 1 ? 's' : ''} mas.`}
                </p>

                {atLimit && upgradeTo && (
                    <Link
                        href="/planes"
                        className="mt-4 flex items-center justify-between w-full px-4 py-3 rounded-xl text-sm font-semibold transition-all hover:scale-[1.02]"
                        style={{
                            background: 'linear-gradient(135deg, rgba(37,99,235,0.15), rgba(20,184,166,0.15))',
                            border: '1px solid rgba(37,99,235,0.3)',
                            color: 'var(--accent-primary)',
                        }}
                    >
                        <span>Actualiza a {PLAN_LABELS[upgradeTo]} — Hasta {PLAN_LIMITS[upgradeTo]} estrategias</span>
                        <ArrowUpRight className="w-4 h-4" />
                    </Link>
                )}
            </div>

            {/* Strategies list */}
            {estrategias.length > 0 ? (
                <div className="flex flex-col gap-4">
                    {estrategias.map((e) => (
                        <EstrategiaDetail key={e.id} estrategia={e} />
                    ))}
                </div>
            ) : (
                <EstrategiasEmptyState />
            )}
        </div>
    );
}
