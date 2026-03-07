'use client';

import { Lock } from 'lucide-react';
import type { OptimizerModel, PlanTier } from '@/lib/types';

const MODELS: {
    value: OptimizerModel;
    label: string;
    description: string;
    minTier: PlanTier;
}[] = [
    {
        value: 'markowitz',
        label: 'Max Sharpe (Markowitz)',
        description: 'Portafolio optimo por frontera eficiente',
        minTier: 'basico',
    },
    {
        value: 'hrp',
        label: 'Paridad de Riesgo (HRP)',
        description: 'Diversificacion jerarquica, robusto en crisis',
        minTier: 'pro',
    },
    {
        value: 'montecarlo',
        label: 'Monte Carlo',
        description: 'Simulacion de 10,000+ portafolios aleatorios',
        minTier: 'ultra',
    },
];

const TIER_RANK: Record<PlanTier, number> = { basico: 0, pro: 1, ultra: 2 };

const TIER_LABEL: Record<PlanTier, string> = {
    basico: 'Basico',
    pro: 'Pro',
    ultra: 'Ultra',
};

interface ModelSelectorProps {
    value: OptimizerModel;
    onChange: (model: OptimizerModel) => void;
    userPlan: PlanTier;
    disabled?: boolean;
}

export default function ModelSelector({ value, onChange, userPlan, disabled }: ModelSelectorProps) {
    const userRank = TIER_RANK[userPlan];

    return (
        <div>
            <label className="text-xs font-bold uppercase tracking-widest mb-2 block" style={{ color: 'var(--text-muted)' }}>
                Modelo de Optimizacion
            </label>
            <div className="flex flex-col gap-2">
                {MODELS.map((model) => {
                    const locked = TIER_RANK[model.minTier] > userRank;
                    const selected = value === model.value;

                    return (
                        <button
                            key={model.value}
                            type="button"
                            disabled={locked || disabled}
                            onClick={() => onChange(model.value)}
                            className="text-left px-3 py-2.5 rounded-lg text-sm transition-all"
                            style={{
                                background: selected
                                    ? 'rgba(37,99,235,0.15)'
                                    : 'rgba(15,23,42,0.6)',
                                border: selected
                                    ? '1px solid rgba(37,99,235,0.4)'
                                    : '1px solid var(--border-light)',
                                color: locked ? 'var(--text-muted)' : 'var(--text-main)',
                                opacity: locked ? 0.5 : 1,
                                cursor: locked ? 'not-allowed' : 'pointer',
                            }}
                        >
                            <div className="flex items-center justify-between">
                                <span className="font-semibold text-xs">{model.label}</span>
                                {locked && (
                                    <span className="flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded-full"
                                        style={{ background: 'rgba(245,158,11,0.1)', color: 'var(--warning)', border: '1px solid rgba(245,158,11,0.25)' }}>
                                        <Lock size={10} />
                                        {TIER_LABEL[model.minTier]}+
                                    </span>
                                )}
                            </div>
                            <p className="text-[11px] mt-0.5" style={{ color: 'var(--text-muted)' }}>
                                {model.description}
                            </p>
                        </button>
                    );
                })}
            </div>
        </div>
    );
}
