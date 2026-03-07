'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Eye, BarChart3 } from 'lucide-react';
import { toast } from 'sonner';
import { getErrorMessage, formatErrorToast } from '@/utils/errorMessages';

interface Props {
    initialValue: boolean;
}

export default function ConfigModoExperto({ initialValue }: Props) {
    const [enabled, setEnabled] = useState(initialValue);
    const [saving, setSaving] = useState(false);

    const toggle = async () => {
        const next = !enabled;
        setSaving(true);
        try {
            const res = await fetch('/api/profile/modo-experto', {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ modo_experto: next }),
            });
            if (!res.ok) throw new Error('Error al guardar');
            setEnabled(next);
            toast.success(next ? 'Modo Experto activado' : 'Modo Experto desactivado');
        } catch {
            const em = getErrorMessage('PROFILE_UPDATE_FAILED');
            toast.error(formatErrorToast(em));
        } finally {
            setSaving(false);
        }
    };

    return (
        <div
            className="glass-panel p-5 flex items-center justify-between gap-4"
            style={{ borderLeft: '3px solid var(--accent-primary)' }}
        >
            <div className="flex items-start gap-3">
                <div
                    className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 mt-0.5"
                    style={{ background: 'rgba(37,99,235,0.12)' }}
                >
                    {enabled ? (
                        <BarChart3 className="w-5 h-5" style={{ color: 'var(--accent-primary)' }} />
                    ) : (
                        <Eye className="w-5 h-5" style={{ color: 'var(--accent-primary)' }} />
                    )}
                </div>
                <div>
                    <h3
                        className="text-sm font-semibold"
                        style={{ color: 'var(--text-main)' }}
                    >
                        Modo Experto
                    </h3>
                    <p
                        className="text-xs mt-0.5 leading-relaxed"
                        style={{ color: 'var(--text-muted)' }}
                    >
                        Muestra automaticamente todas las metricas avanzadas (Sharpe, volatilidad
                        comparativa, drawdown) sin necesidad de expandir manualmente.
                    </p>
                </div>
            </div>

            {/* Toggle switch */}
            <button
                onClick={toggle}
                disabled={saving}
                className="relative flex-shrink-0 w-12 h-7 rounded-full transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-blue-500 disabled:opacity-50"
                style={{
                    background: enabled ? 'var(--accent-primary)' : 'rgba(100,116,139,0.3)',
                }}
                aria-label="Toggle Modo Experto"
            >
                <motion.span
                    className="absolute top-0.5 left-0.5 w-6 h-6 rounded-full bg-white shadow"
                    animate={{ x: enabled ? 20 : 0 }}
                    transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                />
            </button>
        </div>
    );
}
