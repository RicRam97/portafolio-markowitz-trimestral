'use client';

import * as Accordion from '@radix-ui/react-accordion';
import { ChevronDown, TrendingUp, Activity, BarChart3, Clock } from 'lucide-react';

interface AllocationItem {
    ticker: string;
    weight_pct: number;
    shares: number;
}

interface PortfolioVersion {
    id: string;
    nombre: string;
    rendimiento_pct: number | null;
    volatilidad_pct: number | null;
    sharpe_ratio: number | null;
    allocation: AllocationItem[];
    created_at: string;
}

interface Props {
    versions: PortfolioVersion[];
    estrategiaTipo: string;
}

const fmt = (n: number, d = 2) => n.toFixed(d);

const COLORS = [
    '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
    '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
];

function sharpeLabel(s: number) {
    if (s >= 1) return { text: 'Excelente', color: '#10b981' };
    if (s >= 0.5) return { text: 'Bueno', color: '#3b82f6' };
    return { text: 'Bajo', color: '#f59e0b' };
}

export default function VersionHistorial({ versions, estrategiaTipo }: Props) {
    if (versions.length === 0) {
        return (
            <div className="glass-panel p-6 text-center">
                <Clock className="w-8 h-8 mx-auto mb-3" style={{ color: 'var(--text-muted)' }} />
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                    No hay versiones de portafolio para esta estrategia.
                </p>
            </div>
        );
    }

    const tipoLabel = estrategiaTipo === 'markowitz' ? 'Markowitz' : 'HRP';

    return (
        <Accordion.Root type="single" collapsible className="flex flex-col gap-3">
            {versions.map((v, idx) => {
                const isLatest = idx === 0;
                const fecha = new Date(v.created_at).toLocaleDateString('es-MX', {
                    day: 'numeric',
                    month: 'short',
                    year: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                });

                return (
                    <Accordion.Item
                        key={v.id}
                        value={v.id}
                        className="glass-panel overflow-hidden"
                    >
                        <Accordion.Header asChild>
                            <Accordion.Trigger className="group w-full flex items-center justify-between p-4 text-left hover:brightness-110 transition cursor-pointer">
                                <div className="flex items-center gap-3 min-w-0">
                                    <div
                                        className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0"
                                        style={{
                                            background: isLatest
                                                ? 'rgba(16,185,129,0.12)'
                                                : 'rgba(100,116,139,0.1)',
                                        }}
                                    >
                                        <Clock
                                            className="w-4 h-4"
                                            style={{
                                                color: isLatest
                                                    ? 'var(--success)'
                                                    : 'var(--text-muted)',
                                            }}
                                        />
                                    </div>
                                    <div className="min-w-0">
                                        <div className="flex items-center gap-2">
                                            <span
                                                className="text-sm font-semibold truncate"
                                                style={{ color: 'var(--text-main)' }}
                                            >
                                                {v.nombre}
                                            </span>
                                            {isLatest && (
                                                <span
                                                    className="px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider flex-shrink-0"
                                                    style={{
                                                        background: 'rgba(16,185,129,0.15)',
                                                        color: 'var(--success)',
                                                        border: '1px solid rgba(16,185,129,0.3)',
                                                    }}
                                                >
                                                    Vigente
                                                </span>
                                            )}
                                        </div>
                                        <p
                                            className="text-xs flex items-center gap-2"
                                            style={{ color: 'var(--text-muted)' }}
                                        >
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
                                        </p>
                                    </div>
                                </div>

                                <div className="flex items-center gap-4 flex-shrink-0">
                                    {v.sharpe_ratio != null && (
                                        <span
                                            className="hidden sm:block text-xs font-medium"
                                            style={{ color: 'var(--accent-primary)' }}
                                        >
                                            SR {fmt(v.sharpe_ratio)}
                                        </span>
                                    )}
                                    {v.rendimiento_pct != null && (
                                        <span
                                            className="hidden sm:block text-xs font-medium"
                                            style={{ color: 'var(--success)' }}
                                        >
                                            {fmt(v.rendimiento_pct)}%
                                        </span>
                                    )}
                                    <ChevronDown
                                        className="w-4 h-4 transition-transform duration-200 group-data-[state=open]:rotate-180"
                                        style={{ color: 'var(--text-muted)' }}
                                    />
                                </div>
                            </Accordion.Trigger>
                        </Accordion.Header>

                        <Accordion.Content className="overflow-hidden data-[state=open]:animate-accordion-open data-[state=closed]:animate-accordion-closed">
                            <div
                                className="px-4 pb-4 flex flex-col gap-4"
                                style={{ borderTop: '1px solid var(--border-light)' }}
                            >
                                {/* Metrics row */}
                                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-4">
                                    {[
                                        {
                                            label: 'Retorno Esperado',
                                            sub: 'Anual',
                                            value:
                                                v.rendimiento_pct != null
                                                    ? `${fmt(v.rendimiento_pct)}%`
                                                    : '--',
                                            icon: TrendingUp,
                                            color: 'var(--success)',
                                        },
                                        {
                                            label: 'Volatilidad',
                                            sub: 'Riesgo',
                                            value:
                                                v.volatilidad_pct != null
                                                    ? `${fmt(v.volatilidad_pct)}%`
                                                    : '--',
                                            icon: Activity,
                                            color: 'var(--warning)',
                                        },
                                        {
                                            label: 'Ratio de Sharpe',
                                            value:
                                                v.sharpe_ratio != null
                                                    ? fmt(v.sharpe_ratio)
                                                    : '--',
                                            icon: BarChart3,
                                            color: 'var(--accent-primary)',
                                            badge:
                                                v.sharpe_ratio != null
                                                    ? sharpeLabel(v.sharpe_ratio)
                                                    : undefined,
                                        },
                                    ].map((m) => {
                                        const Icon = m.icon;
                                        return (
                                            <div
                                                key={m.label}
                                                className="glass-panel p-3"
                                                style={{ borderLeft: `3px solid ${m.color}` }}
                                            >
                                                <div className="flex items-center gap-2 mb-1">
                                                    <Icon
                                                        className="w-3.5 h-3.5"
                                                        style={{ color: m.color }}
                                                    />
                                                    <span
                                                        className="text-[10px] uppercase tracking-widest"
                                                        style={{ color: 'var(--text-muted)' }}
                                                    >
                                                        {m.label}
                                                        {m.sub && (
                                                            <span
                                                                className="ml-1 px-1 py-0.5 rounded text-[9px]"
                                                                style={{
                                                                    background:
                                                                        'rgba(37,99,235,0.15)',
                                                                    color: 'var(--accent-primary)',
                                                                }}
                                                            >
                                                                {m.sub}
                                                            </span>
                                                        )}
                                                    </span>
                                                </div>
                                                <div className="flex items-baseline gap-2">
                                                    <span
                                                        className="text-xl font-bold"
                                                        style={{
                                                            fontFamily: 'var(--font-mono)',
                                                            color: 'var(--text-main)',
                                                        }}
                                                    >
                                                        {m.value}
                                                    </span>
                                                    {'badge' in m && m.badge && (
                                                        <span
                                                            className="text-[10px] px-1.5 py-0.5 rounded font-medium"
                                                            style={{
                                                                color: m.badge.color,
                                                                background: `${m.badge.color}15`,
                                                            }}
                                                        >
                                                            {m.badge.text}
                                                        </span>
                                                    )}
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>

                                {/* Allocation table */}
                                {v.allocation && v.allocation.length > 0 && (
                                    <div className="glass-panel overflow-hidden">
                                        <h4
                                            className="text-xs font-semibold uppercase tracking-widest p-3 pb-2"
                                            style={{ color: 'var(--text-muted)' }}
                                        >
                                            Composicion
                                        </h4>
                                        <div className="overflow-x-auto">
                                            <table className="w-full text-sm">
                                                <thead>
                                                    <tr
                                                        style={{
                                                            borderBottom:
                                                                '1px solid var(--border-light)',
                                                        }}
                                                    >
                                                        <th
                                                            className="px-3 py-2 text-left text-[10px] uppercase tracking-widest font-semibold"
                                                            style={{
                                                                color: 'var(--text-muted)',
                                                            }}
                                                        >
                                                            Ticker
                                                        </th>
                                                        <th
                                                            className="px-3 py-2 text-right text-[10px] uppercase tracking-widest font-semibold"
                                                            style={{
                                                                color: 'var(--text-muted)',
                                                            }}
                                                        >
                                                            Peso
                                                        </th>
                                                        <th
                                                            className="px-3 py-2 text-right text-[10px] uppercase tracking-widest font-semibold hidden sm:table-cell"
                                                            style={{
                                                                color: 'var(--text-muted)',
                                                            }}
                                                        >
                                                            Acciones
                                                        </th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {v.allocation
                                                        .sort(
                                                            (a, b) =>
                                                                b.weight_pct - a.weight_pct,
                                                        )
                                                        .map((a, i) => (
                                                            <tr
                                                                key={a.ticker}
                                                                style={{
                                                                    borderBottom:
                                                                        '1px solid var(--border-light)',
                                                                }}
                                                                className="hover:brightness-110 transition"
                                                            >
                                                                <td className="px-3 py-2">
                                                                    <span className="flex items-center gap-2">
                                                                        <span
                                                                            className="w-2 h-2 rounded-full flex-shrink-0"
                                                                            style={{
                                                                                background:
                                                                                    COLORS[
                                                                                        i %
                                                                                            COLORS.length
                                                                                    ],
                                                                            }}
                                                                        />
                                                                        <span className="font-medium">
                                                                            {a.ticker}
                                                                        </span>
                                                                    </span>
                                                                </td>
                                                                <td
                                                                    className="px-3 py-2 text-right"
                                                                    style={{
                                                                        fontFamily:
                                                                            'var(--font-mono)',
                                                                    }}
                                                                >
                                                                    {fmt(a.weight_pct, 1)}%
                                                                </td>
                                                                <td
                                                                    className="px-3 py-2 text-right hidden sm:table-cell"
                                                                    style={{
                                                                        fontFamily:
                                                                            'var(--font-mono)',
                                                                        color: 'var(--text-muted)',
                                                                    }}
                                                                >
                                                                    {a.shares}
                                                                </td>
                                                            </tr>
                                                        ))}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </Accordion.Content>
                    </Accordion.Item>
                );
            })}
        </Accordion.Root>
    );
}
