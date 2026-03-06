'use client';

import Link from 'next/link';
import { Layers, ArrowUpRight } from 'lucide-react';
import type { PlanTier } from '@/lib/types';
import { PLAN_LIMITS, PLAN_LABELS, PLAN_UPGRADE_TARGET } from '@/lib/constants';

interface Props {
    count: number;
    plan: PlanTier;
}

export default function StrategyCounter({ count, plan }: Props) {
    const limit = PLAN_LIMITS[plan];
    const ratio = limit > 0 ? count / limit : 1;
    const percentage = Math.min(ratio * 100, 100);

    const barColor =
        ratio >= 1
            ? 'var(--danger)'
            : ratio >= 0.7
                ? 'var(--warning)'
                : 'var(--success)';

    const atLimit = count >= limit;
    const upgradeTo = PLAN_UPGRADE_TARGET[plan];

    return (
        <div className="glass-panel p-5">
            <div className="flex items-center gap-3 mb-4">
                <div
                    className="w-10 h-10 rounded-xl flex items-center justify-center"
                    style={{ background: 'rgba(37,99,235,0.12)' }}
                >
                    <Layers className="w-5 h-5" style={{ color: 'var(--accent-primary)' }} />
                </div>
                <div>
                    <h3 className="text-sm font-semibold" style={{ color: 'var(--text-main)' }}>
                        Estrategias
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
                        style={{
                            width: `${percentage}%`,
                            backgroundColor: barColor,
                        }}
                    />
                </div>
            </div>

            <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
                {atLimit
                    ? 'Has alcanzado el límite de tu plan actual.'
                    : `Puedes crear ${limit - count} estrategia${limit - count !== 1 ? 's' : ''} más.`
                }
            </p>

            {atLimit && upgradeTo && (
                <Link
                    href="/planes"
                    className="flex items-center justify-between w-full px-4 py-3 rounded-xl text-sm font-semibold transition-all hover:scale-[1.02]"
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
    );
}
