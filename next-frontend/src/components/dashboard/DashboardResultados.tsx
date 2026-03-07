'use client';

import { useState, useEffect, useMemo, useId } from 'react';
import { motion } from 'framer-motion';
import * as Tooltip from '@radix-ui/react-tooltip';
import {
    PieChart, Pie, Cell, Legend, ResponsiveContainer,
    ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
    Tooltip as RechartsTooltip,
} from 'recharts';
import type {
    OptimizationResult, OptimizerModel, PlanTier,
} from '@/lib/types';
import { API_BASE } from '@/lib/constants';
import EducationalTooltip from '@/components/EducationalTooltip';

/* ─── Palette ──────────────────────────────────────────── */
const COLORS = [
    '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
    '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
];

/* ─── Types ────────────────────────────────────────────── */
interface TickerQuoteData {
    ticker: string;
    name: string;
    price: number;
    change_pct: number;
    error?: boolean;
}

interface ProfileLimits {
    volatilidad_maxima?: number;
    retorno_minimo?: number;
}

interface Allocation {
    ticker: string;
    peso_teorico: number;
    peso_real: number;
    acciones: number;
    inversion: number;
    comision: number;
    precio_compra: number;
    name: string;
}

interface AllocationSummary {
    asignacion: Allocation[];
    efectivo_restante: number;
    inversion_total: number;
    comisiones_totales: number;
    porcentaje_invertido: number;
    desviacion_maxima_peso: number;
}

const BROKER_COMMISSION_DEFAULT = 0.0025; // 0.25%

interface DashboardResultadosProps {
    result: OptimizationResult;
    budget: number;
    model: OptimizerModel;
    userPlan: PlanTier;
    perfil?: ProfileLimits;
    comisionBroker?: number;
    onRecalculate: () => void;
    onSave?: () => void;
    onCompare?: () => void;
}

/* ─── Helpers ──────────────────────────────────────────── */
const fmt = (n: number, d = 2) => n.toFixed(d);
const fmtUSD = (n: number) => n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

function sharpeLabel(s: number) {
    if (s >= 1) return { text: 'Excelente', color: '#10b981' };
    if (s >= 0.5) return { text: 'Bueno', color: '#3b82f6' };
    return { text: 'Bajo', color: '#f59e0b' };
}

const fadeUp = {
    hidden: { opacity: 0, y: 20 },
    visible: (i: number) => ({
        opacity: 1,
        y: 0,
        transition: { delay: i * 0.08, duration: 0.4, ease: 'easeOut' as const },
    }),
};

/* ─── Allocation algorithm (mirrors spec) ──────────────── */
function calcularAccionesYEfectivo(
    pesosOptimos: Record<string, number>,
    preciosActuales: Record<string, number>,
    presupuestoTotal: number,
    comisionBroker: number,
    names: Record<string, string>,
    maxWeight: number = 0.25,
): AllocationSummary {
    let presupuestoDisponible = presupuestoTotal;
    const asignacion: Allocation[] = [];

    // Sort by weight descending
    const sorted = Object.entries(pesosOptimos).sort(([, a], [, b]) => b - a);

    for (const [ticker, peso] of sorted) {
        const precio = preciosActuales[ticker] ?? 0;
        if (precio <= 0) {
            // No price available — show theoretical allocation only
            asignacion.push({
                ticker,
                peso_teorico: peso,
                peso_real: peso,
                acciones: 0,
                inversion: peso * presupuestoTotal,
                comision: 0,
                precio_compra: 0,
                name: names[ticker] ?? ticker,
            });
            continue;
        }

        const inversionObjetivo = peso * presupuestoTotal;
        const numAcciones = Math.floor(inversionObjetivo / precio);
        const inversionReal = numAcciones * precio;
        const comision = inversionReal * comisionBroker;
        const costoTotal = inversionReal + comision;
        const pesoReal = inversionReal / presupuestoTotal;

        asignacion.push({
            ticker,
            peso_teorico: peso,
            peso_real: pesoReal,
            acciones: numAcciones,
            inversion: inversionReal,
            comision,
            precio_compra: precio,
            name: names[ticker] ?? ticker,
        });

        presupuestoDisponible -= costoTotal;
    }

    // Optimizar efectivo restante si > 5% del presupuesto
    const umbral = 0.05 * presupuestoTotal;
    while (presupuestoDisponible > umbral) {
        let mejorIdx = -1;
        let mejorCosto = 0;

        for (let i = 0; i < asignacion.length; i++) {
            const a = asignacion[i];
            if (a.precio_compra <= 0) continue;
            const costoAccion = a.precio_compra * (1 + comisionBroker);
            if (costoAccion > presupuestoDisponible) continue;
            const nuevoPeso = (a.inversion + a.precio_compra) / presupuestoTotal;
            if (nuevoPeso > maxWeight) continue;
            if (a.precio_compra > mejorCosto) {
                mejorCosto = a.precio_compra;
                mejorIdx = i;
            }
        }

        if (mejorIdx < 0) break;

        const best = asignacion[mejorIdx];
        const comisionExtra = best.precio_compra * comisionBroker;
        best.acciones += 1;
        best.inversion += best.precio_compra;
        best.comision += comisionExtra;
        best.peso_real = best.inversion / presupuestoTotal;
        presupuestoDisponible -= best.precio_compra + comisionExtra;
    }

    const inversionTotal = asignacion.reduce((s, a) => s + a.inversion, 0);
    const comisionesTotales = asignacion.reduce((s, a) => s + a.comision, 0);
    const porcentajeInvertido = presupuestoTotal > 0 ? (inversionTotal / presupuestoTotal) * 100 : 0;
    const desviaciones = asignacion
        .filter((a) => a.precio_compra > 0)
        .map((a) => Math.abs(a.peso_real - a.peso_teorico));
    const desviacionMaxima = desviaciones.length > 0 ? Math.max(...desviaciones) : 0;

    return {
        asignacion,
        efectivo_restante: presupuestoDisponible,
        inversion_total: inversionTotal,
        comisiones_totales: comisionesTotales,
        porcentaje_invertido: porcentajeInvertido,
        desviacion_maxima_peso: desviacionMaxima,
    };
}

/* ─── Tooltip wrapper ──────────────────────────────────── */
function InfoTooltip({ text, children }: { text: string; children: React.ReactNode }) {
    const tooltipId = useId();
    return (
        <Tooltip.Provider delayDuration={200}>
            <Tooltip.Root>
                <Tooltip.Trigger asChild>
                    <span aria-describedby={tooltipId}>{children}</span>
                </Tooltip.Trigger>
                <Tooltip.Portal>
                    <Tooltip.Content
                        id={tooltipId}
                        role="tooltip"
                        side="top"
                        sideOffset={6}
                        className="z-50 max-w-[220px] rounded-lg px-3 py-2 text-xs leading-relaxed shadow-xl"
                        style={{
                            background: 'rgba(15,23,42,0.95)',
                            border: '1px solid var(--border-light)',
                            color: 'var(--text-main)',
                        }}
                    >
                        {text}
                        <Tooltip.Arrow style={{ fill: 'rgba(15,23,42,0.95)' }} />
                    </Tooltip.Content>
                </Tooltip.Portal>
            </Tooltip.Root>
        </Tooltip.Provider>
    );
}

/* ═══════════════════════════════════════════════════════════
   MAIN COMPONENT
   ═══════════════════════════════════════════════════════════ */
export default function DashboardResultados({
    result,
    budget,
    model,
    userPlan,
    perfil,
    comisionBroker = BROKER_COMMISSION_DEFAULT,
    onRecalculate,
    onSave,
    onCompare,
}: DashboardResultadosProps) {
    const opt = result.portafolio_optimo;
    const weights = opt.weights;
    const tickerList = Object.keys(weights).sort((a, b) => weights[b] - weights[a]);

    /* ─── Backend allocation vs client-side fallback ─────────── */
    const backendAlloc = result.asignacion_real;
    const hasBackendAlloc = !!backendAlloc && Object.keys(backendAlloc.asignacion).length > 0;

    // Only fetch prices client-side if backend didn't provide allocation
    const [quotes, setQuotes] = useState<Record<string, TickerQuoteData>>({});
    const [pricesLoading, setPricesLoading] = useState(!hasBackendAlloc);
    const [riskAccepted, setRiskAccepted] = useState(false);

    const incompatible = result.compatible_con_perfil === false;
    const advertencias = result.advertencias ?? [];

    useEffect(() => {
        if (hasBackendAlloc) {
            setPricesLoading(false);
            return;
        }

        let cancelled = false;
        setPricesLoading(true);

        const fetchBatch = async () => {
            try {
                const params = new URLSearchParams({ tickers: tickerList.join(',') });
                const res = await fetch(`${API_BASE}/precio-actual?${params}`);
                if (!res.ok) throw new Error();
                const data = await res.json();
                if (!cancelled) {
                    setQuotes(data.prices ?? {});
                }
            } catch {
                if (!cancelled) setQuotes({});
            } finally {
                if (!cancelled) setPricesLoading(false);
            }
        };

        fetchBatch();
        return () => { cancelled = true; };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [result, hasBackendAlloc]);

    /* ─── Derived data ─────────────────────────────────────── */
    const retPct = opt.expected_return * 100;
    const volPct = opt.volatility * 100;
    const sr = opt.sharpe_ratio;
    const srInfo = sharpeLabel(sr);

    // Build allocation summary: prefer backend, fallback to client
    const alloc: AllocationSummary = useMemo(() => {
        if (hasBackendAlloc && backendAlloc) {
            // Convert backend dict to sorted array
            const sorted = Object.entries(backendAlloc.asignacion)
                .sort(([, a], [, b]) => b.peso_teorico - a.peso_teorico);
            return {
                asignacion: sorted.map(([ticker, a]) => ({
                    ticker,
                    peso_teorico: a.peso_teorico,
                    peso_real: a.peso_real,
                    acciones: a.acciones,
                    inversion: a.inversion,
                    comision: a.comision,
                    precio_compra: a.precio_compra,
                    name: quotes[ticker]?.name ?? ticker,
                })),
                efectivo_restante: backendAlloc.efectivo_restante,
                inversion_total: backendAlloc.inversion_total,
                comisiones_totales: backendAlloc.comisiones_totales,
                porcentaje_invertido: backendAlloc.porcentaje_invertido,
                desviacion_maxima_peso: backendAlloc.desviacion_maxima_peso,
            };
        }

        // Client-side fallback
        const priceMap: Record<string, number> = {};
        const nameMap: Record<string, string> = {};
        for (const t of tickerList) {
            const q = quotes[t];
            priceMap[t] = q?.price ?? 0;
            nameMap[t] = q?.name ?? t;
        }
        return calcularAccionesYEfectivo(weights, priceMap, budget, comisionBroker, nameMap);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [hasBackendAlloc, pricesLoading, budget, comisionBroker]);

    const hasPrices = hasBackendAlloc || tickerList.some((t) => (quotes[t]?.price ?? 0) > 0);
    const cashPct = budget > 0 ? (alloc.efectivo_restante / budget) * 100 : 0;
    const backendWarning = backendAlloc?.warning;

    /* ─── Pie data ─────────────────────────────────────────── */
    const pieData = tickerList.map((t, i) => ({
        name: t,
        value: parseFloat((weights[t] * 100).toFixed(2)),
        fill: COLORS[i % COLORS.length],
    }));

    /* ─── Frontier data ────────────────────────────────────── */
    const frontierData = (result.frontera_eficiente ?? []).map((p) => ({
        x: p.volatility * 100,
        y: p.expected_return * 100,
    }));
    const optimalPoint = [{ x: volPct, y: retPct }];

    return (
        <div className="flex flex-col gap-5">
            {/* ════════════════════════════════════════════════════
               1. METRICAS PRINCIPALES
               ════════════════════════════════════════════════════ */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                {([
                    {
                        label: 'Rendimiento Esperado',
                        sub: 'Anual',
                        value: `${fmt(retPct)}%`,
                        cls: 'positive',
                        tip: 'Retorno anualizado estimado basado en datos historicos de 3 anios.',
                        tooltipKey: 'retorno_esperado',
                    },
                    {
                        label: 'Volatilidad (Riesgo)',
                        value: `${fmt(volPct)}%`,
                        cls: 'neutral',
                        tip: 'Desviacion estandar anualizada de los retornos. Mide la fluctuacion esperada.',
                        tooltipKey: 'volatilidad',
                    },
                    {
                        label: 'Ratio de Sharpe',
                        value: fmt(sr),
                        cls: 'highlight',
                        tip: 'Retorno por unidad de riesgo. Mayor a 1 es excelente.',
                        badge: srInfo,
                        tooltipKey: 'sharpe_ratio',
                    },
                    {
                        label: 'Efectivo Restante',
                        value: pricesLoading ? '...' : `$${fmtUSD(alloc.efectivo_restante)}`,
                        cls: 'highlight',
                        tip: `Capital no asignado tras comprar acciones enteras${comisionBroker > 0 ? ` (incluye ${(comisionBroker * 100).toFixed(2)}% comision broker)` : ''}.`,
                        extra: pricesLoading ? undefined : `${fmt(cashPct, 1)}% del presupuesto`,
                    },
                ] as Array<{
                    label: string;
                    sub?: string;
                    value: string;
                    cls: string;
                    tip: string;
                    badge?: { text: string; color: string };
                    extra?: string;
                    tooltipKey?: 'retorno_esperado' | 'volatilidad' | 'sharpe_ratio';
                }>).map((m, i) => (
                    <motion.div
                        key={m.label}
                        custom={i}
                        initial="hidden"
                        animate="visible"
                        variants={fadeUp}
                    >
                        <InfoTooltip text={m.tip}>
                            <div className="glass-panel p-4 cursor-help hover:brightness-110 transition">
                                <h3
                                    className="text-xs uppercase tracking-widest mb-1 leading-tight flex items-center gap-1"
                                    style={{ color: 'var(--text-muted)' }}
                                >
                                    {m.label}
                                    {m.sub && (
                                        <span
                                            className="text-[10px] px-1.5 py-0.5 rounded"
                                            style={{
                                                background: 'rgba(37,99,235,0.15)',
                                                color: 'var(--accent-primary)',
                                            }}
                                        >
                                            {m.sub}
                                        </span>
                                    )}
                                    {m.tooltipKey && (
                                        <span className="ml-auto"><EducationalTooltip termKey={m.tooltipKey} /></span>
                                    )}
                                </h3>
                                <div className={`metric-value ${m.cls}`}>
                                    {m.value}
                                    {'badge' in m && m.badge && (
                                        <span
                                            className="text-[10px] ml-2 px-1.5 py-0.5 rounded"
                                            style={{
                                                color: m.badge.color,
                                                background: `${m.badge.color}15`,
                                            }}
                                        >
                                            {m.badge.text}
                                        </span>
                                    )}
                                </div>
                                {'extra' in m && m.extra && (
                                    <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>
                                        {m.extra}
                                    </p>
                                )}
                            </div>
                        </InfoTooltip>
                    </motion.div>
                ))}
            </div>

            {/* Summary bar (only when prices loaded) */}
            {!pricesLoading && hasPrices && (
                <motion.div
                    className="glass-panel px-5 py-3 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.25 }}
                    style={{ borderLeft: '3px solid var(--accent-primary)' }}
                >
                    <span>
                        <b style={{ color: 'var(--text-main)' }}>{fmt(alloc.porcentaje_invertido, 1)}%</b>
                        <span style={{ color: 'var(--text-muted)' }}> invertido</span>
                    </span>
                    <span>
                        <b style={{ color: 'var(--text-main)' }}>${fmtUSD(alloc.comisiones_totales)}</b>
                        <span style={{ color: 'var(--text-muted)' }}> comisiones ({(comisionBroker * 100).toFixed(2)}%)</span>
                    </span>
                    <span>
                        <b style={{ color: 'var(--text-main)' }}>{fmt(alloc.desviacion_maxima_peso * 100, 2)}%</b>
                        <span style={{ color: 'var(--text-muted)' }}> desviacion max. peso</span>
                    </span>
                </motion.div>
            )}

            {/* Weight deviation warning (>5%) */}
            {!pricesLoading && (backendWarning || alloc.desviacion_maxima_peso > 0.05) && (
                <motion.div
                    className="glass-panel p-4 text-sm flex items-start gap-3"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.3 }}
                    style={{ borderLeft: '3px solid var(--warning)' }}
                >
                    <span className="text-lg leading-none flex-shrink-0">{'\u26A0\uFE0F'}</span>
                    <p style={{ color: 'var(--warning)' }}>
                        {backendWarning ?? `Los pesos reales se desvian hasta ${fmt(alloc.desviacion_maxima_peso * 100, 2)}% de los optimos debido al redondeo de acciones enteras. Considera aumentar el presupuesto para mayor precision.`}
                    </p>
                </motion.div>
            )}

            {/* ════════════════════════════════════════════════════
               2. GRAFICA CIRCULAR
               ════════════════════════════════════════════════════ */}
            <motion.div
                className="glass-panel p-5"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3, duration: 0.4 }}
            >
                <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
                    Distribucion de Pesos
                    <EducationalTooltip termKey="peso_activo" />
                </h3>
                <div className="flex flex-col lg:flex-row items-center gap-6">
                    <div className="w-full lg:w-1/2" style={{ height: 300 }} role="img" aria-label="Gráfico de composición del portafolio">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={pieData}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={110}
                                    paddingAngle={2}
                                    dataKey="value"
                                    label={({ name, value }) => `${name} ${value}%`}
                                    labelLine={false}
                                >
                                    {pieData.map((entry, idx) => (
                                        <Cell key={entry.name} fill={COLORS[idx % COLORS.length]} />
                                    ))}
                                </Pie>
                                <RechartsTooltip
                                    contentStyle={{
                                        background: 'rgba(15,23,42,0.95)',
                                        border: '1px solid rgba(255,255,255,0.1)',
                                        borderRadius: 8,
                                        color: '#fff',
                                        fontSize: 12,
                                    }}
                                    formatter={(value) => [`${value}%`, 'Peso']}
                                />
                                <Legend
                                    layout="vertical"
                                    align="right"
                                    verticalAlign="middle"
                                    wrapperStyle={{ fontSize: 12 }}
                                    className="hidden lg:block"
                                />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                    {/* Mobile legend */}
                    <div className="flex flex-wrap gap-2 lg:hidden justify-center">
                        {pieData.map((d, i) => (
                            <span
                                key={d.name}
                                className="flex items-center gap-1.5 text-xs px-2 py-1 rounded-md"
                                style={{
                                    background: `${COLORS[i % COLORS.length]}15`,
                                    border: `1px solid ${COLORS[i % COLORS.length]}40`,
                                }}
                            >
                                <span
                                    className="w-2.5 h-2.5 rounded-full"
                                    style={{ background: COLORS[i % COLORS.length] }}
                                />
                                {d.name} {d.value}%
                            </span>
                        ))}
                    </div>
                </div>
            </motion.div>

            {/* ════════════════════════════════════════════════════
               3. TABLA DETALLADA
               ════════════════════════════════════════════════════ */}
            <motion.div
                className="glass-panel overflow-hidden"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4, duration: 0.4 }}
            >
                <h3 className="text-sm font-semibold p-5 pb-3">
                    Asignacion Detallada del Portafolio
                </h3>
                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr style={{ borderBottom: '1px solid var(--border-light)' }}>
                                <th className="px-4 py-3 text-left text-xs uppercase tracking-widest font-semibold" style={{ color: 'var(--text-muted)' }}>
                                    Ticker
                                </th>
                                <th className="px-4 py-3 text-left text-xs uppercase tracking-widest font-semibold hidden sm:table-cell" style={{ color: 'var(--text-muted)' }}>
                                    Empresa
                                </th>
                                <th className="px-4 py-3 text-right text-xs uppercase tracking-widest font-semibold" style={{ color: 'var(--text-muted)' }}>
                                    Peso
                                </th>
                                <th className="px-4 py-3 text-right text-xs uppercase tracking-widest font-semibold hidden md:table-cell" style={{ color: 'var(--text-muted)' }}>
                                    Acciones
                                </th>
                                <th className="px-4 py-3 text-right text-xs uppercase tracking-widest font-semibold" style={{ color: 'var(--text-muted)' }}>
                                    Inversion
                                </th>
                                <th className="px-4 py-3 text-right text-xs uppercase tracking-widest font-semibold hidden md:table-cell" style={{ color: 'var(--text-muted)' }}>
                                    Precio Actual
                                </th>
                            </tr>
                        </thead>
                        <tbody>
                            {alloc.asignacion.map((a, i) => (
                                <tr
                                    key={a.ticker}
                                    style={{ borderBottom: '1px solid var(--border-light)' }}
                                    className="hover:brightness-110 transition"
                                >
                                    <td className="px-4 py-3">
                                        <span className="flex items-center gap-2">
                                            <span
                                                className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                                                style={{ background: COLORS[i % COLORS.length] }}
                                            />
                                            <span className="font-semibold">{a.ticker}</span>
                                        </span>
                                    </td>
                                    <td className="px-4 py-3 hidden sm:table-cell" style={{ color: 'var(--text-muted)' }}>
                                        {pricesLoading ? (
                                            <span className="skeleton inline-block h-3 w-24 rounded" />
                                        ) : (
                                            a.name
                                        )}
                                    </td>
                                    <td className="px-4 py-3 text-right font-medium">
                                        <span>{(a.peso_teorico * 100).toFixed(2)}%</span>
                                        {/* Show real weight deviation if prices loaded and it differs */}
                                        {!pricesLoading && a.precio_compra > 0 && Math.abs(a.peso_real - a.peso_teorico) > 0.0001 && (
                                            <span className="block text-[10px]" style={{ color: 'var(--text-muted)' }}>
                                                real: {(a.peso_real * 100).toFixed(2)}%
                                            </span>
                                        )}
                                    </td>
                                    <td className="px-4 py-3 text-right hidden md:table-cell">
                                        {pricesLoading ? (
                                            <span className="skeleton inline-block h-3 w-10 rounded" />
                                        ) : a.precio_compra > 0 ? (
                                            a.acciones
                                        ) : (
                                            <span style={{ color: 'var(--text-muted)' }}>--</span>
                                        )}
                                    </td>
                                    <td className="px-4 py-3 text-right font-medium">
                                        {pricesLoading ? (
                                            <span className="skeleton inline-block h-3 w-16 rounded" />
                                        ) : (
                                            `$${fmtUSD(a.inversion)}`
                                        )}
                                    </td>
                                    <td className="px-4 py-3 text-right hidden md:table-cell">
                                        {pricesLoading ? (
                                            <span className="skeleton inline-block h-3 w-16 rounded" />
                                        ) : a.precio_compra > 0 ? (
                                            `$${fmtUSD(a.precio_compra)}`
                                        ) : (
                                            <span style={{ color: 'var(--text-muted)' }}>N/A</span>
                                        )}
                                    </td>
                                </tr>
                            ))}
                            {/* Totals row */}
                            <tr className="font-semibold" style={{ background: 'rgba(37,99,235,0.05)' }}>
                                <td className="px-4 py-3">Total</td>
                                <td className="px-4 py-3 hidden sm:table-cell" />
                                <td className="px-4 py-3 text-right">
                                    {(tickerList.reduce((s, t) => s + weights[t], 0) * 100).toFixed(1)}%
                                </td>
                                <td className="px-4 py-3 text-right hidden md:table-cell" />
                                <td className="px-4 py-3 text-right">
                                    {pricesLoading ? (
                                        <span className="skeleton inline-block h-3 w-20 rounded" />
                                    ) : (
                                        `$${fmtUSD(alloc.inversion_total)}`
                                    )}
                                </td>
                                <td className="px-4 py-3 hidden md:table-cell" />
                            </tr>
                        </tbody>
                    </table>
                </div>
            </motion.div>

            {/* ════════════════════════════════════════════════════
               4. FRONTERA EFICIENTE (solo markowitz)
               ════════════════════════════════════════════════════ */}
            {model === 'markowitz' && frontierData.length > 0 && (
                <motion.div
                    className="glass-panel p-5"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5, duration: 0.4 }}
                >
                    <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
                        Frontera Eficiente
                        <EducationalTooltip termKey="frontera_eficiente" />
                    </h3>
                    <div style={{ height: 350 }} role="img" aria-label="Gráfico de frontera eficiente: relación riesgo-rendimiento del portafolio">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                                <CartesianGrid
                                    strokeDasharray="3 3"
                                    stroke="rgba(255,255,255,0.06)"
                                />
                                <XAxis
                                    type="number"
                                    dataKey="x"
                                    name="Volatilidad"
                                    unit="%"
                                    domain={[0, 'auto']}
                                    tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                                    label={{
                                        value: 'Volatilidad (%)',
                                        position: 'insideBottom',
                                        offset: -10,
                                        style: { fill: 'var(--text-muted)', fontSize: 11 },
                                    }}
                                />
                                <YAxis
                                    type="number"
                                    dataKey="y"
                                    name="Rendimiento"
                                    unit="%"
                                    domain={[0, 'auto']}
                                    tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                                    label={{
                                        value: 'Rendimiento (%)',
                                        angle: -90,
                                        position: 'insideLeft',
                                        style: { fill: 'var(--text-muted)', fontSize: 11 },
                                    }}
                                />
                                <RechartsTooltip
                                    contentStyle={{
                                        background: 'rgba(15,23,42,0.95)',
                                        border: '1px solid rgba(255,255,255,0.1)',
                                        borderRadius: 8,
                                        color: '#fff',
                                        fontSize: 12,
                                    }}
                                    formatter={(value, name) => [
                                        `${Number(value).toFixed(2)}%`,
                                        name === 'x' ? 'Volatilidad' : 'Rendimiento',
                                    ]}
                                />
                                {/* Frontier curve */}
                                <Scatter
                                    name="Frontera"
                                    data={frontierData}
                                    fill="rgba(99,102,241,0.5)"
                                    line={{ stroke: '#6366f1', strokeWidth: 2 }}
                                    lineType="fitting"
                                    shape="circle"
                                    legendType="line"
                                >
                                    {frontierData.map((_, idx) => (
                                        <Cell
                                            key={idx}
                                            fill="rgba(99,102,241,0.3)"
                                            r={2}
                                        />
                                    ))}
                                </Scatter>
                                {/* Optimal portfolio */}
                                <Scatter
                                    name="Portafolio Optimo"
                                    data={optimalPoint}
                                    fill="#3b82f6"
                                    shape="star"
                                    legendType="star"
                                >
                                    <Cell fill="#3b82f6" r={8} />
                                </Scatter>
                                {/* User profile reference point */}
                                {perfil?.volatilidad_maxima && perfil?.retorno_minimo && (
                                    <Scatter
                                        name="Tu Perfil"
                                        data={[{
                                            x: perfil.volatilidad_maxima * 100,
                                            y: perfil.retorno_minimo * 100,
                                        }]}
                                        fill="#9ca3af"
                                        shape="diamond"
                                        legendType="diamond"
                                    >
                                        <Cell fill="#9ca3af" r={6} />
                                    </Scatter>
                                )}
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                </motion.div>
            )}

            {/* ════════════════════════════════════════════════════
               5. MENSAJES EDUCATIVOS
               ════════════════════════════════════════════════════ */}
            <EducationalAlerts
                volatility={opt.volatility}
                expectedReturn={opt.expected_return}
                sharpe={sr}
                perfil={perfil}
            />

            {/* ════════════════════════════════════════════════════
               5b. ADVERTENCIA DE PERFIL
               ════════════════════════════════════════════════════ */}
            {incompatible && advertencias.length > 0 && (
                <motion.div
                    className="glass-panel p-4 mb-2"
                    style={{
                        border: '1px solid var(--warning)',
                        background: 'rgba(245,158,11,0.08)',
                    }}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.55, duration: 0.3 }}
                >
                    <h4 className="text-sm font-semibold mb-2 flex items-center gap-2" style={{ color: 'var(--warning)' }}>
                        <span>⚠</span> Portafolio incompatible con tu perfil
                    </h4>
                    <ul className="text-xs space-y-1 mb-3" style={{ color: 'var(--text-muted)' }}>
                        {advertencias.map((adv, i) => (
                            <li key={i}>• {adv}</li>
                        ))}
                    </ul>
                    {!riskAccepted && (
                        <button
                            onClick={() => setRiskAccepted(true)}
                            className="text-xs font-medium px-3 py-1.5 rounded-lg transition-colors"
                            style={{
                                border: '1px solid var(--warning)',
                                color: 'var(--warning)',
                                background: 'transparent',
                            }}
                        >
                            Entiendo los riesgos y deseo continuar
                        </button>
                    )}
                    {riskAccepted && (
                        <p className="text-xs font-medium" style={{ color: 'var(--success)' }}>
                            Has aceptado los riesgos. Puedes guardar el portafolio.
                        </p>
                    )}
                </motion.div>
            )}

            {/* ════════════════════════════════════════════════════
               6. BOTONES DE ACCION
               ════════════════════════════════════════════════════ */}
            <motion.div
                className="flex flex-col sm:flex-row gap-3"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6, duration: 0.4 }}
            >
                <button
                    onClick={onRecalculate}
                    className="btn btn-secondary px-5 py-2.5 text-sm flex-1"
                >
                    Recalcular con otros parametros
                </button>
                {onSave && (!incompatible || riskAccepted) && (
                    <button
                        onClick={onSave}
                        className="btn btn-cta glow-effect px-5 py-2.5 text-sm flex-1"
                    >
                        Guardar este portafolio
                    </button>
                )}
                {(userPlan === 'pro' || userPlan === 'ultra') && onCompare && (
                    <button
                        onClick={onCompare}
                        className="btn btn-secondary px-5 py-2.5 text-sm flex-1"
                        style={{ borderColor: 'var(--accent-primary)', color: 'var(--accent-primary)' }}
                    >
                        Comparar con otro modelo
                    </button>
                )}
            </motion.div>
        </div>
    );
}

/* ═══════════════════════════════════════════════════════════
   EDUCATIONAL ALERTS (section 5)
   ═══════════════════════════════════════════════════════════ */
function EducationalAlerts({
    volatility,
    expectedReturn,
    sharpe,
    perfil,
}: {
    volatility: number;
    expectedReturn: number;
    sharpe: number;
    perfil?: ProfileLimits;
}) {
    const alerts: { message: string; severity: 'warning' | 'danger' }[] = [];

    if (perfil?.volatilidad_maxima && volatility > perfil.volatilidad_maxima) {
        alerts.push({
            message: `La volatilidad del portafolio (${(volatility * 100).toFixed(2)}%) supera tu limite configurado de ${(perfil.volatilidad_maxima * 100).toFixed(2)}%. Considera activos mas conservadores.`,
            severity: 'danger',
        });
    }

    if (sharpe < 0.5) {
        alerts.push({
            message: `El Ratio de Sharpe (${sharpe.toFixed(2)}) es bajo. El retorno no compensa adecuadamente el riesgo asumido. Prueba con otros activos o ajusta tus restricciones.`,
            severity: 'warning',
        });
    }

    if (perfil?.retorno_minimo && expectedReturn < perfil.retorno_minimo) {
        alerts.push({
            message: `El rendimiento esperado (${(expectedReturn * 100).toFixed(2)}%) es menor al retorno minimo deseado de ${(perfil.retorno_minimo * 100).toFixed(2)}%. Agrega activos de mayor crecimiento.`,
            severity: 'warning',
        });
    }

    if (alerts.length === 0) return null;

    return (
        <motion.div
            className="flex flex-col gap-3"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.55, duration: 0.4 }}
        >
            {alerts.map((a, i) => (
                <div
                    key={i}
                    className="glass-panel p-4 text-sm flex items-start gap-3"
                    style={{
                        borderLeft: `3px solid ${a.severity === 'danger' ? 'var(--danger)' : 'var(--warning)'}`,
                    }}
                >
                    <span className="text-lg leading-none flex-shrink-0">
                        {a.severity === 'danger' ? '\u26A0\uFE0F' : '\u26A0\uFE0F'}
                    </span>
                    <p style={{ color: a.severity === 'danger' ? 'var(--danger)' : 'var(--warning)' }}>
                        {a.message}
                    </p>
                </div>
            ))}
        </motion.div>
    );
}
