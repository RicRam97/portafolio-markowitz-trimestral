'use client';

import { useState, useRef, useEffect } from 'react';
import { ChevronDown, Lock } from 'lucide-react';
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
    const [open, setOpen] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);
    const userRank = TIER_RANK[userPlan];
    const current = MODELS.find((m) => m.value === value);

    // Close dropdown on outside click
    useEffect(() => {
        if (!open) return;
        const handler = (e: MouseEvent) => {
            if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
                setOpen(false);
            }
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, [open]);

    return (
        <div ref={containerRef} className="relative">
            <span className="text-xs font-bold uppercase tracking-widest mb-2 block" style={{ color: 'var(--text-muted)' }}>
                Modelo
            </span>

            {/* Trigger */}
            <button
                type="button"
                disabled={disabled}
                onClick={() => setOpen((v) => !v)}
                className="flex items-center justify-between gap-3 px-3 py-2.5 rounded-lg text-sm w-full min-w-[220px] transition-colors"
                style={{
                    background: 'rgba(15,23,42,0.6)',
                    border: `1px solid ${open ? 'rgba(37,99,235,0.4)' : 'var(--border-light)'}`,
                    color: 'var(--text-main)',
                }}
            >
                <span className="font-semibold text-xs truncate">{current?.label}</span>
                <ChevronDown
                    size={14}
                    className="flex-shrink-0 transition-transform"
                    style={{
                        color: 'var(--text-muted)',
                        transform: open ? 'rotate(180deg)' : 'rotate(0deg)',
                    }}
                />
            </button>

            {/* Dropdown menu */}
            {open && (
                <div
                    className="absolute z-40 left-0 right-0 mt-1 rounded-lg py-1 overflow-hidden"
                    style={{
                        background: 'var(--bg-card, rgba(15,23,42,0.97))',
                        border: '1px solid var(--border-light)',
                        boxShadow: '0 12px 32px rgba(0,0,0,0.4)',
                    }}
                >
                    {MODELS.map((model) => {
                        const locked = TIER_RANK[model.minTier] > userRank;
                        const selected = value === model.value;

                        return (
                            <button
                                key={model.value}
                                type="button"
                                disabled={locked || disabled}
                                onClick={() => {
                                    onChange(model.value);
                                    setOpen(false);
                                }}
                                className="w-full text-left px-3 py-2.5 text-sm transition-colors"
                                style={{
                                    background: selected ? 'rgba(37,99,235,0.12)' : 'transparent',
                                    color: locked ? 'var(--text-muted)' : 'var(--text-main)',
                                    opacity: locked ? 0.5 : 1,
                                    cursor: locked ? 'not-allowed' : 'pointer',
                                }}
                                onMouseEnter={(e) => {
                                    if (!locked && !selected) {
                                        (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.05)';
                                    }
                                }}
                                onMouseLeave={(e) => {
                                    (e.currentTarget as HTMLElement).style.background = selected ? 'rgba(37,99,235,0.12)' : 'transparent';
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
            )}
        </div>
    );
}
