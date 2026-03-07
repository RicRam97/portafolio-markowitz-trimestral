'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { createBrowserClient } from '@supabase/ssr';
import ModelSelector from '@/components/dashboard/ModelSelector';
import OptimizationLoader from '@/components/dashboard/OptimizationLoader';
import DashboardResultados from '@/components/dashboard/DashboardResultados';
import type { OptimizerModel, PlanTier, OptimizationResult } from '@/lib/types';
import { API_BASE } from '@/lib/constants';
import { useNotification } from '@/hooks/useNotification';
import { parseApiError, formatErrorToast, getErrorMessage } from '@/utils/errorMessages';
import { checkAndAwardBadges } from '@/hooks/useBadges';

export default function OptimizarClient() {
    const [model, setModel] = useState<OptimizerModel>('markowitz');
    const [userPlan, setUserPlan] = useState<PlanTier>('basico');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<OptimizationResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [tickerInput, setTickerInput] = useState('');
    const [tickers, setTickers] = useState<string[]>([]);
    const [budget, setBudget] = useState(10000);
    const notify = useNotification();

    // Fetch user plan on mount
    useEffect(() => {
        const supabase = createBrowserClient(
            process.env.NEXT_PUBLIC_SUPABASE_URL!,
            process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
        );
        supabase.auth.getUser().then(({ data }) => {
            if (data.user) {
                supabase
                    .from('profiles')
                    .select('plan')
                    .eq('id', data.user.id)
                    .single()
                    .then(({ data: profile }) => {
                        if (profile?.plan) setUserPlan(profile.plan as PlanTier);
                    });
            }
        });
    }, []);

    const addTicker = () => {
        const cleaned = tickerInput
            .split(',')
            .map((t) => t.trim().toUpperCase())
            .filter((t) => t && !tickers.includes(t));
        if (cleaned.length) {
            setTickers((prev) => [...prev, ...cleaned]);
            setTickerInput('');
        }
    };

    const removeTicker = (ticker: string) => {
        setTickers((prev) => prev.filter((t) => t !== ticker));
    };

    const handleOptimize = async () => {
        if (tickers.length < 2) {
            notify.warning('Selecciona al menos 2 tickers.');
            setError('Selecciona al menos 2 tickers.');
            return;
        }

        setLoading(true);
        setError(null);
        setResult(null);

        try {
            const supabase = createBrowserClient(
                process.env.NEXT_PUBLIC_SUPABASE_URL!,
                process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
            );
            const { data: session } = await supabase.auth.getSession();
            const token = session.session?.access_token;
            if (!token) {
                setError('Sesion expirada. Inicia sesion nuevamente.');
                setLoading(false);
                return;
            }

            const res = await fetch(`${API_BASE}/optimizar/${model}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${token}`,
                },
                body: JSON.stringify({
                    tickers,
                    presupuesto: budget,
                    ...(model === 'montecarlo' && { num_simulaciones: 10000 }),
                }),
            });

            if (!res.ok) {
                const body = await res.json().catch(() => null);
                const detail = body?.detail;
                // Structured error from backend: { code, message }
                if (detail && typeof detail === 'object' && detail.code) {
                    const errorMsg = getErrorMessage(detail.code);
                    setError(formatErrorToast(errorMsg));
                    notify.error(formatErrorToast(errorMsg));
                } else {
                    throw new Error(typeof detail === 'string' ? detail : `Error ${res.status}`);
                }
                setLoading(false);
                return;
            }

            const data: OptimizationResult = await res.json();
            setResult(data);
            notify.success('Portafolio optimizado exitosamente.');
        } catch (err) {
            const errorMsg = parseApiError(err);
            setError(formatErrorToast(errorMsg));
            notify.error(formatErrorToast(errorMsg));
        } finally {
            setLoading(false);
        }
    };

    const handleRecalculate = () => {
        setResult(null);
        setError(null);
    };

    const handleSave = async () => {
        if (!result) return;
        try {
            const supabase = createBrowserClient(
                process.env.NEXT_PUBLIC_SUPABASE_URL!,
                process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
            );
            const { data: session } = await supabase.auth.getSession();
            const token = session.session?.access_token;
            if (!token) return;

            const saveRes = await fetch(`${API_BASE}/guardar-portafolio`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${token}`,
                },
                body: JSON.stringify({
                    result,
                    budget,
                    model,
                }),
            });
            notify.success('Portafolio guardado.');

            if (saveRes.ok) {
                const { data: { user } } = await supabase.auth.getUser();
                if (user) {
                    const { count } = await supabase
                        .from('estrategias')
                        .select('*', { count: 'exact', head: true })
                        .eq('user_id', user.id);

                    checkAndAwardBadges(user.id, {
                        sharpeRatio: result.portafolio_optimo?.sharpe_ratio,
                        tickerCount: tickers.length,
                        strategyCount: count ?? 1,
                    });
                }
            }
        } catch (err) {
            const errorMsg = getErrorMessage('STRATEGY_SAVE_FAILED');
            notify.error(formatErrorToast(errorMsg));
        }
    };

    return (
        <div className="max-w-[1200px] mx-auto">
            {/* Banner */}
            <div className="glass-panel p-6 mb-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4"
                style={{ borderTop: '4px solid var(--accent-primary)' }}>
                <div>
                    <h2 className="text-xl font-bold mb-1" style={{ fontFamily: 'var(--font-display)' }}>
                        Optimizador de Portafolio
                    </h2>
                    <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                        Selecciona tus activos, configura tu estrategia y ejecuta la optimizacion.
                    </p>
                </div>
                <div className="flex items-center gap-2 text-xs px-3 py-1.5 rounded-full"
                    style={{ background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.25)', color: 'var(--success)' }}>
                    <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                    Datos actualizados diariamente
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-[300px_1fr] gap-6">
                {/* ===== CONFIG PANEL ===== */}
                <aside className="glass-panel p-5 flex flex-col gap-5">
                    {/* Budget */}
                    <div>
                        <label htmlFor="budget" className="text-xs font-bold uppercase tracking-widest mb-2 block" style={{ color: 'var(--text-muted)' }}>
                            Presupuesto de Inversion (USD)
                        </label>
                        <div className="flex items-center gap-2">
                            <span className="text-sm font-semibold" style={{ color: 'var(--text-muted)' }}>$</span>
                            <input
                                type="number"
                                id="budget"
                                value={budget}
                                onChange={(e) => setBudget(Number(e.target.value))}
                                min={100}
                                step={100}
                                className="flex-1 px-3 py-2 rounded-lg text-sm outline-none"
                                style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid var(--border-light)', color: 'var(--text-main)' }}
                            />
                        </div>
                    </div>

                    {/* Model Selector */}
                    <ModelSelector
                        value={model}
                        onChange={setModel}
                        userPlan={userPlan}
                        disabled={loading}
                    />

                    {/* Tickers */}
                    <div className="flex-1">
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
                                Universo (Tickers)
                            </span>
                            <span className="text-xs px-2 py-0.5 rounded-full font-semibold"
                                style={{ background: 'rgba(37,99,235,0.1)', color: 'var(--accent-primary)', border: '1px solid rgba(37,99,235,0.2)' }}>
                                {tickers.length} seleccionados
                            </span>
                        </div>
                        <div className="flex gap-2 mb-3">
                            <input
                                type="text"
                                placeholder="ej. NVDA, AAPL"
                                value={tickerInput}
                                onChange={(e) => setTickerInput(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && addTicker()}
                                className="flex-1 px-3 py-2 rounded-lg text-sm outline-none"
                                style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid var(--border-light)', color: 'var(--text-main)' }}
                            />
                            <button onClick={addTicker} className="btn btn-secondary px-3 py-2 text-sm">+</button>
                        </div>

                        {/* Selected tickers */}
                        {tickers.length > 0 ? (
                            <div className="flex flex-wrap gap-1.5 mb-3">
                                {tickers.map((t) => (
                                    <span key={t} className="flex items-center gap-1 text-xs px-2 py-1 rounded-md"
                                        style={{ background: 'rgba(37,99,235,0.15)', color: 'var(--accent-primary)', border: '1px solid rgba(37,99,235,0.25)' }}>
                                        {t}
                                        <button onClick={() => removeTicker(t)} className="hover:text-white ml-0.5">&times;</button>
                                    </span>
                                ))}
                            </div>
                        ) : (
                            <div className="text-center py-6 text-sm rounded-lg mb-3" style={{ background: 'rgba(15,23,42,0.3)', color: 'var(--text-muted)', border: '1px dashed var(--border-light)' }}>
                                Agrega tickers separados por coma
                            </div>
                        )}

                        {tickers.length > 0 && (
                            <button onClick={() => setTickers([])} className="text-xs underline" style={{ color: 'var(--text-muted)' }}>
                                Limpiar todo
                            </button>
                        )}
                    </div>

                    {/* Optimize button */}
                    <motion.button
                        onClick={handleOptimize}
                        disabled={loading || tickers.length < 2}
                        className="btn btn-cta glow-effect w-full text-sm py-3 disabled:opacity-50 disabled:cursor-not-allowed"
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                    >
                        {loading ? 'Optimizando...' : 'Optimizar Portafolio'}
                    </motion.button>
                </aside>

                {/* ===== MAIN CONTENT ===== */}
                <div className="flex flex-col gap-5">
                    {/* Loading state */}
                    {loading && <OptimizationLoader model={model} />}

                    {/* Error */}
                    {error && (
                        <div className="glass-panel p-4 text-sm" style={{ borderLeft: '3px solid var(--danger)', color: 'var(--danger)' }}>
                            {error}
                        </div>
                    )}

                    {/* Results */}
                    {result ? (
                        <DashboardResultados
                            result={result}
                            budget={budget}
                            model={model}
                            userPlan={userPlan}
                            onRecalculate={handleRecalculate}
                            onSave={handleSave}
                        />
                    ) : !loading && (
                        <div className="glass-panel p-10 flex items-center justify-center text-center rounded-xl"
                            style={{ background: 'rgba(15,23,42,0.5)', border: '1px dashed var(--border-light)' }}>
                            <div>
                                <p className="text-sm font-semibold mb-1">Sin datos aun</p>
                                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                                    Selecciona tickers y ejecuta la optimizacion para ver tu distribucion de portafolio.
                                </p>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
