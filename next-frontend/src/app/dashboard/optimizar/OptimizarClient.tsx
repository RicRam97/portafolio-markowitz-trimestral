'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { createBrowserClient } from '@supabase/ssr';
import ModelSelector from '@/components/dashboard/ModelSelector';
import OptimizationLoader from '@/components/dashboard/OptimizationLoader';
import DashboardResultados from '@/components/dashboard/DashboardResultados';
import BudgetEditor from '@/components/dashboard/BudgetEditor';
import TickerSelectorModal from '@/components/dashboard/TickerSelectorModal';
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
        <div className="max-w-[1200px] mx-auto flex flex-col gap-6">
            {/* Banner */}
            <div className="glass-panel p-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4"
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

            {/* ===== CONFIG BAR (horizontal, full-width) ===== */}
            <div className="glass-panel p-5 w-full">
                <div className="flex flex-col lg:flex-row lg:items-end gap-5">
                    {/* Budget */}
                    <BudgetEditor value={budget} onChange={setBudget} />

                    {/* Separator */}
                    <div className="hidden lg:block w-px self-stretch" style={{ background: 'var(--border-light)' }} />

                    {/* Model Selector */}
                    <div className="flex-shrink-0">
                        <ModelSelector
                            value={model}
                            onChange={setModel}
                            userPlan={userPlan}
                            disabled={loading}
                        />
                    </div>

                    {/* Separator */}
                    <div className="hidden lg:block w-px self-stretch" style={{ background: 'var(--border-light)' }} />

                    {/* Tickers */}
                    <TickerSelectorModal
                        selected={tickers}
                        onChange={setTickers}
                        userPlan={userPlan}
                    />

                    {/* Optimize button */}
                    <motion.button
                        onClick={handleOptimize}
                        disabled={loading || tickers.length < 2}
                        className="btn btn-cta glow-effect whitespace-nowrap text-sm px-8 py-3 disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                    >
                        {loading ? 'Optimizando...' : 'Optimizar Portafolio'}
                    </motion.button>
                </div>
            </div>

            {/* ===== RESULTS AREA ===== */}
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
    );
}
