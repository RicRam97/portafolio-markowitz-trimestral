'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
    Tooltip as RechartsTooltip, ResponsiveContainer, Cell,
} from 'recharts';
import Link from 'next/link';
import { ChevronDown, ChevronUp, TrendingUp, Activity, BarChart3, History, FileDown, FileSpreadsheet, Share2, Copy, Check, X } from 'lucide-react';
import EducationalTooltip, { type TooltipKey } from '@/components/EducationalTooltip';
import { createBrowserClient } from '@supabase/ssr';
import { API_BASE } from '@/lib/constants';
import { useNotification } from '@/hooks/useNotification';
import { scaleIn, fadeIn, modalBackdrop, modalContent } from '@/utils/animations';
import * as XLSX from 'xlsx';

interface FrontierPoint {
    expected_return: number;
    volatility: number;
    sharpe_ratio?: number;
}

interface AllocationItem {
    ticker: string;
    weight_pct: number;
    shares: number;
    precio_compra?: number;
}

interface EstrategiaRow {
    id: string;
    nombre: string;
    tipo: string;
    parametros: Record<string, unknown>;
    created_at: string;
    portafolios: {
        presupuesto: number | null;
        rendimiento_pct: number | null;
        volatilidad_pct: number | null;
        sharpe_ratio: number | null;
        allocation: AllocationItem[];
        metricas: Record<string, unknown>;
    }[];
}

interface Props {
    estrategia: EstrategiaRow;
}

const fmt = (n: number, d = 2) => n.toFixed(d);

function sharpeLabel(s: number) {
    if (s >= 1) return { text: 'Excelente', color: '#10b981' };
    if (s >= 0.5) return { text: 'Bueno', color: '#3b82f6' };
    return { text: 'Bajo', color: '#f59e0b' };
}

export default function EstrategiaDetail({ estrategia }: Props) {
    const [open, setOpen] = useState(false);
    const [pdfLoading, setPdfLoading] = useState(false);
    const [shareModal, setShareModal] = useState(false);
    const [shareLink, setShareLink] = useState('');
    const [shareLoading, setShareLoading] = useState(false);
    const [copied, setCopied] = useState(false);
    const notify = useNotification();

    const handleShare = async () => {
        setShareLoading(true);
        try {
            const supabase = createBrowserClient(
                process.env.NEXT_PUBLIC_SUPABASE_URL!,
                process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
            );
            const { data: session } = await supabase.auth.getSession();
            const token = session.session?.access_token;
            if (!token) return;

            const res = await fetch(
                `${API_BASE}/estrategias/${estrategia.id}/generar-link-compartido`,
                {
                    method: 'POST',
                    headers: { Authorization: `Bearer ${token}` },
                },
            );
            if (!res.ok) throw new Error('Error generando link');

            const data = await res.json();
            setShareLink(data.link);
            setShareModal(true);
        } catch {
            notify.error('Error al generar el link de compartir.');
        } finally {
            setShareLoading(false);
        }
    };

    const handleCopy = async () => {
        await navigator.clipboard.writeText(shareLink);
        setCopied(true);
        notify.success('Link copiado al portapapeles.');
        setTimeout(() => setCopied(false), 2000);
    };

    const handleExportPDF = async () => {
        setPdfLoading(true);
        try {
            const supabase = createBrowserClient(
                process.env.NEXT_PUBLIC_SUPABASE_URL!,
                process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
            );
            const { data: session } = await supabase.auth.getSession();
            const token = session.session?.access_token;
            if (!token) return;

            const res = await fetch(
                `${API_BASE}/estrategias/${estrategia.id}/export-pdf`,
                { headers: { Authorization: `Bearer ${token}` } },
            );
            if (!res.ok) throw new Error('Error generando PDF');

            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            window.open(url, '_blank');
        } catch {
            notify.error('Error al generar el PDF.');
        } finally {
            setPdfLoading(false);
        }
    };

    const handleExportExcel = () => {
        const port = estrategia.portafolios?.[0];
        if (!port) return;

        const allocation = port.allocation ?? [];
        const presupuesto = port.presupuesto ?? 0;

        // Hoja 1: Composicion
        const composicion = allocation.map((item) => {
            const monto = typeof item.precio_compra === 'number'
                ? item.shares * item.precio_compra
                : (presupuesto * item.weight_pct) / 100;
            return {
                Ticker: item.ticker,
                Peso_Porcentaje: Number(item.weight_pct.toFixed(2)),
                Acciones: item.shares,
                Precio: item.precio_compra ?? '-',
                Monto: Number(monto.toFixed(2)),
            };
        });

        // Hoja 2: Metricas
        const metricasSheet = [
            { Metrica: 'Retorno Esperado (%)', Valor: port.rendimiento_pct ?? '-' },
            { Metrica: 'Volatilidad (%)', Valor: port.volatilidad_pct ?? '-' },
            { Metrica: 'Sharpe Ratio', Valor: port.sharpe_ratio ?? '-' },
            { Metrica: 'Presupuesto (USD)', Valor: presupuesto },
            { Metrica: 'Modelo', Valor: estrategia.tipo.toUpperCase() },
        ];

        const wb = XLSX.utils.book_new();
        const ws1 = XLSX.utils.json_to_sheet(composicion);
        const ws2 = XLSX.utils.json_to_sheet(metricasSheet);
        XLSX.utils.book_append_sheet(wb, ws1, 'Composicion');
        XLSX.utils.book_append_sheet(wb, ws2, 'Metricas');

        const fechaStr = new Date().toISOString().slice(0, 10);
        const nombreSafe = estrategia.nombre.replace(/[^a-zA-Z0-9_-]/g, '_');
        XLSX.writeFile(wb, `Estrategia_${nombreSafe}_${fechaStr}.xlsx`);
    };

    const portfolio = estrategia.portafolios?.[0];
    const ret = portfolio?.rendimiento_pct;
    const vol = portfolio?.volatilidad_pct;
    const sharpe = portfolio?.sharpe_ratio;

    const metricas = portfolio?.metricas ?? {};
    const frontera = (metricas.frontera_eficiente ?? []) as FrontierPoint[];
    const frontierData = frontera.map((p) => ({
        x: p.volatility * 100,
        y: p.expected_return * 100,
    }));
    const optimalPoint = ret != null && vol != null ? [{ x: vol, y: ret }] : [];

    const tickers = (estrategia.parametros?.tickers as string[] | undefined) ?? [];
    const tipoLabel = estrategia.tipo === 'markowitz' ? 'Markowitz' : 'HRP';
    const fecha = new Date(estrategia.created_at).toLocaleDateString('es-MX', {
        day: 'numeric', month: 'short', year: 'numeric',
    });

    const metrics: {
        label: string;
        sub?: string;
        value: string;
        icon: typeof TrendingUp;
        color: string;
        badge?: { text: string; color: string };
        tooltipKey: TooltipKey;
    }[] = [
        {
            label: 'Retorno Esperado',
            sub: 'Anual',
            value: ret != null ? `${fmt(ret)}%` : '--',
            icon: TrendingUp,
            color: 'var(--success)',
            tooltipKey: 'retorno_esperado',
        },
        {
            label: 'Volatilidad',
            sub: 'Riesgo',
            value: vol != null ? `${fmt(vol)}%` : '--',
            icon: Activity,
            color: 'var(--warning)',
            tooltipKey: 'volatilidad',
        },
        {
            label: 'Ratio de Sharpe',
            value: sharpe != null ? fmt(sharpe) : '--',
            icon: BarChart3,
            color: 'var(--accent-primary)',
            badge: sharpe != null ? sharpeLabel(sharpe) : undefined,
            tooltipKey: 'sharpe_ratio',
        },
    ];

    return (
        <motion.div
            className="glass-panel overflow-hidden"
            variants={scaleIn}
            initial="hidden"
            animate="visible"
        >
            {/* Header row — always visible */}
            <button
                onClick={() => setOpen((o) => !o)}
                className="w-full flex items-center justify-between p-5 text-left hover:brightness-110 transition"
            >
                <div className="flex items-center gap-3 min-w-0">
                    <div
                        className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                        style={{ background: 'rgba(37,99,235,0.12)' }}
                    >
                        <TrendingUp className="w-5 h-5" style={{ color: 'var(--accent-primary)' }} />
                    </div>
                    <div className="min-w-0">
                        <h3 className="text-sm font-semibold truncate" style={{ color: 'var(--text-main)' }}>
                            {estrategia.nombre}
                        </h3>
                        <p className="text-xs flex items-center gap-2" style={{ color: 'var(--text-muted)' }}>
                            <span
                                className="px-1.5 py-0.5 rounded text-[10px] font-medium"
                                style={{
                                    background: 'rgba(37,99,235,0.15)',
                                    color: 'var(--accent-primary)',
                                }}
                            >
                                {tipoLabel}
                            </span>
                            {fecha}
                            {tickers.length > 0 && (
                                <span className="hidden sm:inline truncate max-w-[200px]">
                                    {tickers.join(', ')}
                                </span>
                            )}
                        </p>
                    </div>
                </div>

                <div className="flex items-center gap-4 flex-shrink-0">
                    {/* Quick metrics in header */}
                    {ret != null && (
                        <span className="hidden md:flex items-center gap-1 text-xs font-medium" style={{ color: 'var(--success)' }}>
                            {fmt(ret)}%
                        </span>
                    )}
                    {sharpe != null && (
                        <span className="hidden md:flex items-center gap-1 text-xs font-medium" style={{ color: 'var(--accent-primary)' }}>
                            SR {fmt(sharpe)}
                        </span>
                    )}
                    {open ? (
                        <ChevronUp className="w-5 h-5" style={{ color: 'var(--text-muted)' }} />
                    ) : (
                        <ChevronDown className="w-5 h-5" style={{ color: 'var(--text-muted)' }} />
                    )}
                </div>
            </button>

            {/* Expandable detail */}
            {open && (
                <div className="px-5 pb-5 flex flex-col gap-5" style={{ borderTop: '1px solid var(--border-light)' }}>
                    {/* Metrics cards */}
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-5">
                        {metrics.map((m) => {
                            const Icon = m.icon;
                            return (
                                <div
                                    key={m.label}
                                    className="glass-panel p-4"
                                    style={{ borderLeft: `3px solid ${m.color}` }}
                                >
                                    <div className="flex items-center gap-2 mb-2">
                                        <Icon className="w-4 h-4" style={{ color: m.color }} />
                                        <span className="text-xs uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
                                            {m.label}
                                            {m.sub && (
                                                <span
                                                    className="ml-1 px-1 py-0.5 rounded text-[9px]"
                                                    style={{ background: 'rgba(37,99,235,0.15)', color: 'var(--accent-primary)' }}
                                                >
                                                    {m.sub}
                                                </span>
                                            )}
                                        </span>
                                        <EducationalTooltip termKey={m.tooltipKey} />
                                    </div>
                                    <div className="flex items-baseline gap-2">
                                        <span
                                            className="text-2xl font-bold"
                                            style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-main)' }}
                                        >
                                            {m.value}
                                        </span>
                                        {m.badge && (
                                            <span
                                                className="text-[10px] px-1.5 py-0.5 rounded font-medium"
                                                style={{ color: m.badge.color, background: `${m.badge.color}15` }}
                                            >
                                                {m.badge.text}
                                            </span>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    {/* Efficient Frontier Chart */}
                    {frontierData.length > 0 && (
                        <div className="glass-panel p-5">
                            <h4 className="text-sm font-semibold mb-4 flex items-center gap-2" style={{ color: 'var(--text-main)' }}>
                                Frontera Eficiente
                                <EducationalTooltip termKey="frontera_eficiente" />
                            </h4>
                            <div style={{ height: 320 }}>
                                <ResponsiveContainer width="100%" height="100%">
                                    <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
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
                                            formatter={(value) => [
                                                `${Number(value).toFixed(2)}%`,
                                                'Valor',
                                            ]}
                                        />
                                        {/* Frontier curve (gray dots) */}
                                        <Scatter
                                            name="Frontera"
                                            data={frontierData}
                                            fill="rgba(148,163,184,0.4)"
                                            line={{ stroke: 'rgba(148,163,184,0.6)', strokeWidth: 2 }}
                                            lineType="fitting"
                                            shape="circle"
                                            legendType="line"
                                        >
                                            {frontierData.map((_, idx) => (
                                                <Cell key={idx} fill="rgba(148,163,184,0.3)" r={2} />
                                            ))}
                                        </Scatter>
                                        {/* Optimal portfolio (gold star) */}
                                        {optimalPoint.length > 0 && (
                                            <Scatter
                                                name="Portafolio Optimo"
                                                data={optimalPoint}
                                                fill="#f59e0b"
                                                shape="star"
                                                legendType="star"
                                            >
                                                <Cell fill="#f59e0b" r={8} />
                                            </Scatter>
                                        )}
                                    </ScatterChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    )}

                    {/* No portfolio data message */}
                    {!portfolio && (
                        <div className="glass-panel p-5 text-center">
                            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                                Esta estrategia no tiene portafolios guardados aun.
                            </p>
                        </div>
                    )}

                    {/* Action buttons */}
                    <div className="flex flex-col sm:flex-row gap-3">
                        <Link
                            href={`/dashboard/estrategias/${estrategia.id}`}
                            className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all hover:scale-[1.02]"
                            style={{
                                background: 'rgba(37,99,235,0.08)',
                                border: '1px solid rgba(37,99,235,0.2)',
                                color: 'var(--accent-primary)',
                            }}
                        >
                            <History className="w-4 h-4" />
                            Ver historial de versiones
                        </Link>
                        {portfolio && (
                            <>
                                <button
                                    onClick={handleExportPDF}
                                    disabled={pdfLoading}
                                    className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all hover:scale-[1.02] disabled:opacity-50"
                                    style={{
                                        background: 'rgba(16,185,129,0.08)',
                                        border: '1px solid rgba(16,185,129,0.2)',
                                        color: 'var(--success)',
                                    }}
                                >
                                    <FileDown className="w-4 h-4" />
                                    {pdfLoading ? 'Generando...' : 'Descargar PDF'}
                                </button>
                                <button
                                    onClick={handleExportExcel}
                                    className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all hover:scale-[1.02]"
                                    style={{
                                        background: 'rgba(34,197,94,0.08)',
                                        border: '1px solid rgba(34,197,94,0.2)',
                                        color: '#22c55e',
                                    }}
                                >
                                    <FileSpreadsheet className="w-4 h-4" />
                                    Exportar a Excel
                                </button>
                            </>
                        )}
                        <button
                            onClick={handleShare}
                            disabled={shareLoading}
                            className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all hover:scale-[1.02] disabled:opacity-50"
                            style={{
                                background: 'rgba(139,92,246,0.08)',
                                border: '1px solid rgba(139,92,246,0.2)',
                                color: '#8b5cf6',
                            }}
                        >
                            <Share2 className="w-4 h-4" />
                            {shareLoading ? 'Generando...' : 'Compartir'}
                        </button>
                    </div>
                </div>
            )}

            {/* Share modal */}
            <AnimatePresence>
            {shareModal && (
                <motion.div
                    key="share-backdrop"
                    variants={modalBackdrop}
                    initial="hidden"
                    animate="visible"
                    exit="hidden"
                    className="fixed inset-0 z-50 flex items-center justify-center p-4"
                    style={{ background: 'rgba(0,0,0,0.6)' }}
                    onClick={() => setShareModal(false)}
                >
                    <motion.div
                        key="share-content"
                        variants={modalContent}
                        initial="hidden"
                        animate="visible"
                        exit="hidden"
                        className="glass-panel p-6 w-full max-w-md relative"
                        style={{ border: '1px solid var(--border-light)' }}
                        onClick={(e) => e.stopPropagation()}
                    >
                        <button
                            onClick={() => setShareModal(false)}
                            className="absolute top-4 right-4"
                            style={{ color: 'var(--text-muted)' }}
                        >
                            <X className="w-5 h-5" />
                        </button>

                        <div className="flex items-center gap-3 mb-4">
                            <div
                                className="w-10 h-10 rounded-xl flex items-center justify-center"
                                style={{ background: 'rgba(139,92,246,0.12)' }}
                            >
                                <Share2 className="w-5 h-5" style={{ color: '#8b5cf6' }} />
                            </div>
                            <div>
                                <h3 className="text-sm font-semibold" style={{ color: 'var(--text-main)' }}>
                                    Compartir Estrategia
                                </h3>
                                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                                    Cualquiera con este link podra ver tu estrategia.
                                </p>
                            </div>
                        </div>

                        <div className="flex gap-2">
                            <input
                                type="text"
                                readOnly
                                value={shareLink}
                                className="flex-1 px-3 py-2 rounded-lg text-xs outline-none"
                                style={{
                                    background: 'rgba(15,23,42,0.6)',
                                    border: '1px solid var(--border-light)',
                                    color: 'var(--text-main)',
                                }}
                            />
                            <button
                                onClick={handleCopy}
                                className="px-3 py-2 rounded-lg text-sm font-medium flex items-center gap-1.5 transition-all"
                                style={{
                                    background: copied ? 'rgba(16,185,129,0.15)' : 'rgba(139,92,246,0.15)',
                                    border: `1px solid ${copied ? 'rgba(16,185,129,0.3)' : 'rgba(139,92,246,0.3)'}`,
                                    color: copied ? 'var(--success)' : '#8b5cf6',
                                }}
                            >
                                {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                                {copied ? 'Copiado' : 'Copiar'}
                            </button>
                        </div>
                    </motion.div>
                </motion.div>
            )}
            </AnimatePresence>
        </motion.div>
    );
}
