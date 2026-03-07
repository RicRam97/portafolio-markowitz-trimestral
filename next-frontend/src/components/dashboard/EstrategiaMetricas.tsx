'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    TrendingUp,
    Activity,
    BarChart3,
    ShieldCheck,
    AlertTriangle,
    ChevronDown,
} from 'lucide-react';
import EducationalTooltip from '@/components/EducationalTooltip';

interface Props {
    rendimiento_pct: number | null;
    volatilidad_pct: number | null;
    sharpe_ratio: number | null;
    metricas: {
        max_drawdown?: number;
    } | null;
    modoExperto: boolean;
}

function riskLevel(vol: number) {
    if (vol <= 12) return { label: 'Bajo', color: '#10b981', bg: 'rgba(16,185,129,0.12)' };
    if (vol <= 25) return { label: 'Medio', color: '#f59e0b', bg: 'rgba(245,158,11,0.12)' };
    return { label: 'Alto', color: '#ef4444', bg: 'rgba(239,68,68,0.12)' };
}

function rebalanceStatus(vol: number, sharpe: number | null) {
    if (sharpe != null && sharpe >= 0.5 && vol <= 25)
        return { balanced: true, label: 'Balanceado' };
    return { balanced: false, label: 'Requiere ajuste' };
}

const fmt = (n: number, d = 2) => n.toFixed(d);

// Reference volatility values for comparative bar chart
const VOL_BENCHMARKS = [
    { label: 'S&P 500', value: 15 },
    { label: 'Bonos US', value: 5 },
];

export default function EstrategiaMetricas({
    rendimiento_pct,
    volatilidad_pct,
    sharpe_ratio,
    metricas,
    modoExperto,
}: Props) {
    const [expanded, setExpanded] = useState(false);
    const showAdvanced = modoExperto || expanded;

    const risk = volatilidad_pct != null ? riskLevel(volatilidad_pct) : null;
    const rebalance =
        volatilidad_pct != null
            ? rebalanceStatus(volatilidad_pct, sharpe_ratio)
            : null;

    const maxDrawdown = metricas?.max_drawdown ?? null;

    return (
        <div className="flex flex-col gap-4">
            {/* ── BASIC METRICS (always visible) ── */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {/* Retorno anual esperado */}
                <div
                    className="glass-panel p-4"
                    style={{ borderLeft: '3px solid var(--success)' }}
                >
                    <div className="flex items-center gap-2 mb-2">
                        <TrendingUp
                            className="w-4 h-4"
                            style={{ color: 'var(--success)' }}
                        />
                        <span
                            className="text-[10px] uppercase tracking-widest"
                            style={{ color: 'var(--text-muted)' }}
                        >
                            Retorno Anual Esperado
                        </span>
                        <EducationalTooltip termKey="retorno_esperado" />
                    </div>
                    <span
                        className="text-3xl font-bold"
                        style={{
                            fontFamily: 'var(--font-mono)',
                            color: 'var(--text-main)',
                        }}
                    >
                        {rendimiento_pct != null ? `${fmt(rendimiento_pct)}%` : '--'}
                    </span>
                </div>

                {/* Nivel de riesgo */}
                <div
                    className="glass-panel p-4"
                    style={{
                        borderLeft: `3px solid ${risk?.color ?? 'var(--text-muted)'}`,
                    }}
                >
                    <div className="flex items-center gap-2 mb-2">
                        <Activity
                            className="w-4 h-4"
                            style={{ color: risk?.color ?? 'var(--text-muted)' }}
                        />
                        <span
                            className="text-[10px] uppercase tracking-widest"
                            style={{ color: 'var(--text-muted)' }}
                        >
                            Nivel de Riesgo
                        </span>
                        <EducationalTooltip termKey="volatilidad" />
                    </div>
                    <div className="flex items-center gap-2">
                        <span
                            className="text-xl font-bold"
                            style={{
                                fontFamily: 'var(--font-mono)',
                                color: 'var(--text-main)',
                            }}
                        >
                            {volatilidad_pct != null ? `${fmt(volatilidad_pct)}%` : '--'}
                        </span>
                        {risk && (
                            <span
                                className="text-xs font-semibold px-2 py-0.5 rounded-full"
                                style={{
                                    color: risk.color,
                                    background: risk.bg,
                                }}
                            >
                                {risk.label}
                            </span>
                        )}
                    </div>
                </div>

                {/* Estado de rebalanceo */}
                <div
                    className="glass-panel p-4"
                    style={{
                        borderLeft: `3px solid ${rebalance?.balanced ? 'var(--success)' : 'var(--warning)'}`,
                    }}
                >
                    <div className="flex items-center gap-2 mb-2">
                        {rebalance?.balanced ? (
                            <ShieldCheck
                                className="w-4 h-4"
                                style={{ color: 'var(--success)' }}
                            />
                        ) : (
                            <AlertTriangle
                                className="w-4 h-4"
                                style={{ color: 'var(--warning)' }}
                            />
                        )}
                        <span
                            className="text-[10px] uppercase tracking-widest"
                            style={{ color: 'var(--text-muted)' }}
                        >
                            Rebalanceo
                        </span>
                        <EducationalTooltip termKey="rebalanceo" />
                    </div>
                    <span
                        className="text-lg font-bold"
                        style={{ color: 'var(--text-main)' }}
                    >
                        {rebalance
                            ? rebalance.balanced
                                ? '\u2713 Balanceado'
                                : '\u26A0\uFE0F Requiere ajuste'
                            : '--'}
                    </span>
                </div>
            </div>

            {/* ── EXPAND BUTTON (hidden in expert mode) ── */}
            {!modoExperto && (
                <button
                    onClick={() => setExpanded((v) => !v)}
                    className="flex items-center gap-2 self-start px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                    style={{
                        background: 'rgba(37,99,235,0.1)',
                        color: 'var(--accent-primary)',
                        border: '1px solid rgba(37,99,235,0.2)',
                    }}
                >
                    <BarChart3 className="w-4 h-4" />
                    {expanded ? 'Ocultar metricas avanzadas' : 'Ver metricas avanzadas'}
                    <motion.span
                        animate={{ rotate: expanded ? 180 : 0 }}
                        transition={{ duration: 0.25 }}
                        className="inline-flex"
                    >
                        <ChevronDown className="w-4 h-4" />
                    </motion.span>
                </button>
            )}

            {/* ── ADVANCED METRICS ── */}
            <AnimatePresence initial={false}>
                {showAdvanced && (
                    <motion.div
                        key="advanced"
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.35, ease: [0.4, 0, 0.2, 1] }}
                        className="overflow-hidden"
                    >
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
                            {/* Sharpe Ratio with tooltip */}
                            <div
                                className="glass-panel p-4 relative group"
                                style={{ borderLeft: '3px solid var(--accent-primary)' }}
                            >
                                <div className="flex items-center gap-2 mb-2">
                                    <BarChart3
                                        className="w-4 h-4"
                                        style={{ color: 'var(--accent-primary)' }}
                                    />
                                    <span
                                        className="text-[10px] uppercase tracking-widest"
                                        style={{ color: 'var(--text-muted)' }}
                                    >
                                        Ratio de Sharpe
                                    </span>
                                    <EducationalTooltip termKey="sharpe_ratio" />
                                </div>
                                <div className="flex items-baseline gap-2">
                                    <span
                                        className="text-2xl font-bold"
                                        style={{
                                            fontFamily: 'var(--font-mono)',
                                            color: 'var(--text-main)',
                                        }}
                                    >
                                        {sharpe_ratio != null ? fmt(sharpe_ratio) : '--'}
                                    </span>
                                    {sharpe_ratio != null && (
                                        <span
                                            className="text-[10px] px-1.5 py-0.5 rounded font-medium"
                                            style={{
                                                color:
                                                    sharpe_ratio >= 1
                                                        ? '#10b981'
                                                        : sharpe_ratio >= 0.5
                                                          ? '#3b82f6'
                                                          : '#f59e0b',
                                                background:
                                                    sharpe_ratio >= 1
                                                        ? 'rgba(16,185,129,0.12)'
                                                        : sharpe_ratio >= 0.5
                                                          ? 'rgba(59,130,246,0.12)'
                                                          : 'rgba(245,158,11,0.12)',
                                            }}
                                        >
                                            {sharpe_ratio >= 1
                                                ? 'Excelente'
                                                : sharpe_ratio >= 0.5
                                                  ? 'Bueno'
                                                  : 'Bajo'}
                                        </span>
                                    )}
                                </div>
                            </div>

                            {/* Volatilidad con barras comparativas */}
                            <div
                                className="glass-panel p-4"
                                style={{ borderLeft: '3px solid var(--warning)' }}
                            >
                                <div className="flex items-center gap-2 mb-2">
                                    <Activity
                                        className="w-4 h-4"
                                        style={{ color: 'var(--warning)' }}
                                    />
                                    <span
                                        className="text-[10px] uppercase tracking-widest"
                                        style={{ color: 'var(--text-muted)' }}
                                    >
                                        Volatilidad Anual
                                    </span>
                                    <EducationalTooltip termKey="volatilidad" />
                                </div>

                                <div className="flex flex-col gap-2">
                                    {/* Current portfolio bar */}
                                    <VolBar
                                        label="Tu portafolio"
                                        value={volatilidad_pct}
                                        color="var(--accent-primary)"
                                        maxVal={40}
                                    />
                                    {VOL_BENCHMARKS.map((b) => (
                                        <VolBar
                                            key={b.label}
                                            label={b.label}
                                            value={b.value}
                                            color="var(--text-muted)"
                                            maxVal={40}
                                        />
                                    ))}
                                </div>
                            </div>

                            {/* Drawdown maximo */}
                            <div
                                className="glass-panel p-4"
                                style={{ borderLeft: '3px solid var(--danger)' }}
                            >
                                <div className="flex items-center gap-2 mb-2">
                                    <TrendingUp
                                        className="w-4 h-4 rotate-180"
                                        style={{ color: 'var(--danger)' }}
                                    />
                                    <span
                                        className="text-[10px] uppercase tracking-widest"
                                        style={{ color: 'var(--text-muted)' }}
                                    >
                                        Drawdown Maximo
                                    </span>
                                </div>
                                <span
                                    className="text-2xl font-bold"
                                    style={{
                                        fontFamily: 'var(--font-mono)',
                                        color: 'var(--danger)',
                                    }}
                                >
                                    {maxDrawdown != null
                                        ? `${fmt(Math.abs(maxDrawdown))}%`
                                        : '--'}
                                </span>
                                <p
                                    className="text-[10px] mt-1"
                                    style={{ color: 'var(--text-muted)' }}
                                >
                                    Peor caida historica del portafolio
                                </p>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

function VolBar({
    label,
    value,
    color,
    maxVal,
}: {
    label: string;
    value: number | null;
    color: string;
    maxVal: number;
}) {
    const pct = value != null ? Math.min((value / maxVal) * 100, 100) : 0;
    return (
        <div className="flex items-center gap-2">
            <span
                className="text-[10px] w-20 truncate text-right"
                style={{ color: 'var(--text-muted)' }}
            >
                {label}
            </span>
            <div
                className="flex-1 h-3 rounded-full overflow-hidden"
                role="progressbar"
                aria-label={`Volatilidad de ${label}`}
                aria-valuenow={value ?? 0}
                aria-valuemin={0}
                aria-valuemax={maxVal}
                style={{ background: 'rgba(100,116,139,0.15)' }}
            >
                <motion.div
                    className="h-full rounded-full"
                    style={{ background: color }}
                    initial={{ width: 0 }}
                    animate={{ width: `${pct}%` }}
                    transition={{ duration: 0.6, ease: 'easeOut' }}
                />
            </div>
            <span
                className="text-[10px] w-10 font-medium"
                style={{
                    fontFamily: 'var(--font-mono)',
                    color: 'var(--text-muted)',
                }}
            >
                {value != null ? `${fmt(value, 1)}%` : '--'}
            </span>
        </div>
    );
}
