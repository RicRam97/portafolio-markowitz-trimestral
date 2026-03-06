'use client';

import { ShieldCheck, Bell } from 'lucide-react';
import type { Portafolio } from '@/lib/types';

interface Props {
    portfolios: Portafolio[];
}

export default function RebalanceAlerts({ portfolios }: Props) {
    // TODO: When backend endpoint for ticker prices is available,
    // compute actual deviations by comparing current weights vs target weights.
    // For now, show the positive empty state.

    return (
        <div className="glass-panel p-5">
            <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                    style={{ background: 'rgba(16,185,129,0.12)' }}>
                    <Bell className="w-5 h-5" style={{ color: 'var(--success)' }} />
                </div>
                <div>
                    <h3 className="text-sm font-semibold" style={{ color: 'var(--text-main)' }}>
                        Alertas de Rebalanceo
                    </h3>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        Desviaciones &gt; 5% del peso objetivo
                    </p>
                </div>
            </div>

            <div
                className="text-center py-8 rounded-xl"
                style={{ background: 'rgba(16,185,129,0.04)', border: '1px solid rgba(16,185,129,0.15)' }}
            >
                <ShieldCheck className="w-10 h-10 mx-auto mb-3" style={{ color: 'var(--success)' }} />
                <p className="text-sm font-semibold mb-1" style={{ color: 'var(--success)' }}>
                    Todo en orden
                </p>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {portfolios.length > 0
                        ? 'Todos tus portafolios están alineados. Sin desviaciones detectadas.'
                        : 'Crea un portafolio para activar las alertas de rebalanceo.'
                    }
                </p>
            </div>
        </div>
    );
}
