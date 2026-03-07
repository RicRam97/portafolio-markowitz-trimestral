'use client';

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { createBrowserClient } from '@supabase/ssr';
import { TrendingUp, Activity, BarChart3, Copy as CopyIcon } from 'lucide-react';
import Navbar from '@/components/layout/Navbar';
import Footer from '@/components/layout/Footer';
import { API_BASE } from '@/lib/constants';

interface AllocationItem {
    ticker: string;
    weight_pct: number;
    shares: number;
    precio_compra?: number;
}

interface SharedStrategy {
    nombre: string;
    tipo: string;
    parametros: Record<string, unknown>;
    created_at: string;
    portafolio: {
        presupuesto: number;
        rendimiento_pct: number | null;
        volatilidad_pct: number | null;
        sharpe_ratio: number | null;
        allocation: AllocationItem[];
    } | null;
}

const fmt = (n: number, d = 2) => n.toFixed(d);

export default function SharedStrategyPage() {
    const { token } = useParams<{ token: string }>();
    const router = useRouter();
    const [data, setData] = useState<SharedStrategy | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [isAuth, setIsAuth] = useState(false);
    const [cloning, setCloning] = useState(false);

    useEffect(() => {
        const supabase = createBrowserClient(
            process.env.NEXT_PUBLIC_SUPABASE_URL!,
            process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
        );
        supabase.auth.getUser().then(({ data: u }) => {
            if (u.user) setIsAuth(true);
        });
    }, []);

    useEffect(() => {
        if (!token) return;
        fetch(`${API_BASE}/estrategias/compartidas/${token}`)
            .then(async (res) => {
                if (!res.ok) throw new Error('Estrategia no encontrada');
                return res.json();
            })
            .then(setData)
            .catch((e) => setError(e.message))
            .finally(() => setLoading(false));
    }, [token]);

    const handleClone = async () => {
        setCloning(true);
        try {
            const supabase = createBrowserClient(
                process.env.NEXT_PUBLIC_SUPABASE_URL!,
                process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
            );
            const { data: session } = await supabase.auth.getSession();
            const authToken = session.session?.access_token;
            if (!authToken) {
                router.push('/login');
                return;
            }

            const res = await fetch(`${API_BASE}/estrategias/clonar`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${authToken}`,
                },
                body: JSON.stringify({ token }),
            });

            if (!res.ok) throw new Error('Error al clonar');
            router.push('/dashboard/estrategias');
        } catch {
            // silent
        } finally {
            setCloning(false);
        }
    };

    if (loading) {
        return (
            <>
                <Navbar />
                <div className="page-wrapper flex items-center justify-center" style={{ paddingTop: 120, minHeight: '60vh' }}>
                    <div className="text-center">
                        <div className="w-8 h-8 border-2 border-t-transparent rounded-full animate-spin mx-auto mb-4" style={{ borderColor: 'var(--accent-primary)', borderTopColor: 'transparent' }} />
                        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Cargando estrategia...</p>
                    </div>
                </div>
                <Footer />
            </>
        );
    }

    if (error || !data) {
        return (
            <>
                <Navbar />
                <div className="page-wrapper flex items-center justify-center" style={{ paddingTop: 120, minHeight: '60vh' }}>
                    <div className="glass-panel p-8 text-center max-w-md">
                        <h2 className="text-lg font-bold mb-2" style={{ color: 'var(--danger)' }}>Estrategia no encontrada</h2>
                        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                            Este link puede haber expirado o la estrategia ya no es publica.
                        </p>
                    </div>
                </div>
                <Footer />
            </>
        );
    }

    const tipoLabel = data.tipo === 'markowitz' ? 'Markowitz' : 'HRP';
    const tickers = (data.parametros?.tickers as string[] | undefined) ?? [];
    const port = data.portafolio;
    const ret = port?.rendimiento_pct;
    const vol = port?.volatilidad_pct;
    const sharpe = port?.sharpe_ratio;
    const allocation = port?.allocation ?? [];
    const fecha = new Date(data.created_at).toLocaleDateString('es-MX', {
        day: 'numeric', month: 'short', year: 'numeric',
    });

    return (
        <>
            <Navbar />
            <div className="page-wrapper" style={{ paddingTop: 80 }}>
                <div style={{ maxWidth: 900, margin: '0 auto', padding: '0 24px 60px' }}>
                    {/* Header */}
                    <div className="glass-panel p-6 mb-6" style={{ borderTop: '4px solid var(--accent-primary)' }}>
                        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                            <div>
                                <div className="flex items-center gap-2 mb-1">
                                    <span
                                        className="px-2 py-0.5 rounded text-[10px] font-medium"
                                        style={{ background: 'rgba(139,92,246,0.15)', color: '#8b5cf6' }}
                                    >
                                        Estrategia Compartida
                                    </span>
                                    <span
                                        className="px-2 py-0.5 rounded text-[10px] font-medium"
                                        style={{ background: 'rgba(37,99,235,0.15)', color: 'var(--accent-primary)' }}
                                    >
                                        {tipoLabel}
                                    </span>
                                </div>
                                <h1 className="text-xl font-bold" style={{ fontFamily: 'var(--font-display)' }}>
                                    {data.nombre}
                                </h1>
                                <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                                    Creada {fecha}
                                </p>
                            </div>
                            {isAuth && (
                                <button
                                    onClick={handleClone}
                                    disabled={cloning}
                                    className="btn btn-cta glow-effect px-5 py-2.5 text-sm flex items-center gap-2 disabled:opacity-50"
                                >
                                    <CopyIcon className="w-4 h-4" />
                                    {cloning ? 'Clonando...' : 'Clonar a mi cuenta'}
                                </button>
                            )}
                        </div>

                        {tickers.length > 0 && (
                            <div className="flex flex-wrap gap-1.5 mt-4">
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

                    {/* Metrics */}
                    {port && (
                        <>
                            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
                                {[
                                    {
                                        label: 'Retorno Esperado',
                                        sub: 'Anual',
                                        value: ret != null ? `${fmt(ret)}%` : '--',
                                        icon: TrendingUp,
                                        color: 'var(--success)',
                                    },
                                    {
                                        label: 'Volatilidad',
                                        sub: 'Riesgo',
                                        value: vol != null ? `${fmt(vol)}%` : '--',
                                        icon: Activity,
                                        color: 'var(--warning)',
                                    },
                                    {
                                        label: 'Ratio de Sharpe',
                                        value: sharpe != null ? fmt(sharpe) : '--',
                                        icon: BarChart3,
                                        color: 'var(--accent-primary)',
                                    },
                                ].map((m) => {
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
                                            </div>
                                            <span
                                                className="text-2xl font-bold"
                                                style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-main)' }}
                                            >
                                                {m.value}
                                            </span>
                                        </div>
                                    );
                                })}
                            </div>

                            {/* Allocation table */}
                            {allocation.length > 0 && (
                                <div className="glass-panel p-5">
                                    <h3 className="text-sm font-semibold mb-4" style={{ color: 'var(--text-main)' }}>
                                        Composicion del Portafolio
                                    </h3>
                                    <div className="overflow-x-auto">
                                        <table className="w-full text-sm">
                                            <thead>
                                                <tr style={{ borderBottom: '1px solid var(--border-light)' }}>
                                                    <th className="text-left py-2 px-3 text-xs uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>Ticker</th>
                                                    <th className="text-right py-2 px-3 text-xs uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>Peso (%)</th>
                                                    <th className="text-right py-2 px-3 text-xs uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>Acciones</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {allocation.map((item, i) => (
                                                    <tr
                                                        key={item.ticker}
                                                        style={{
                                                            borderBottom: '1px solid var(--border-light)',
                                                            background: i % 2 === 0 ? 'transparent' : 'rgba(37,99,235,0.03)',
                                                        }}
                                                    >
                                                        <td className="py-2.5 px-3 font-medium" style={{ color: 'var(--accent-primary)' }}>
                                                            {item.ticker}
                                                        </td>
                                                        <td className="py-2.5 px-3 text-right" style={{ color: 'var(--text-main)' }}>
                                                            {fmt(item.weight_pct)}%
                                                        </td>
                                                        <td className="py-2.5 px-3 text-right" style={{ color: 'var(--text-main)' }}>
                                                            {item.shares}
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            )}
                        </>
                    )}

                    {!port && (
                        <div className="glass-panel p-8 text-center">
                            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                                Esta estrategia no tiene portafolios calculados.
                            </p>
                        </div>
                    )}
                </div>
            </div>
            <Footer />
        </>
    );
}
