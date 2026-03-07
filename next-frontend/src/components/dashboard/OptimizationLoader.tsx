'use client';

import { useEffect, useState } from 'react';
import type { OptimizerModel } from '@/lib/types';

const MESSAGES: Record<OptimizerModel, string[]> = {
    markowitz: [
        'Descargando precios historicos...',
        'Calculando retornos logaritmicos...',
        'Construyendo matriz de covarianza...',
        'Optimizando con frontera eficiente...',
        'Buscando portafolio de maximo Sharpe...',
        'Generando 50 puntos de la frontera...',
        'Casi listo...',
    ],
    hrp: [
        'Descargando precios historicos...',
        'Calculando matriz de correlaciones...',
        'Agrupando activos por similitud (clustering)...',
        'Construyendo dendrograma jerarquico...',
        'Asignando pesos por paridad de riesgo...',
        'Validando diversificacion del portafolio...',
        'Casi listo...',
    ],
    montecarlo: [
        'Descargando precios historicos...',
        'Calculando retornos y covarianza...',
        'Generando portafolios aleatorios...',
        'Simulando 10,000 combinaciones...',
        'Evaluando Sharpe Ratio de cada portafolio...',
        'Identificando portafolio optimo en la nube...',
        'Preparando visualizacion...',
        'Casi listo...',
    ],
};

interface OptimizationLoaderProps {
    model: OptimizerModel;
}

function LoaderInner({ messages }: { messages: string[] }) {
    const [index, setIndex] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setIndex((prev) => (prev < messages.length - 1 ? prev + 1 : prev));
        }, 2200);
        return () => clearInterval(interval);
    }, [messages.length]);

    return (
        <div className="glass-panel p-8 flex flex-col items-center justify-center gap-4 text-center">
            <div className="w-10 h-10 rounded-full border-2 border-t-transparent animate-spin"
                style={{ borderColor: 'var(--accent-primary)', borderTopColor: 'transparent' }} />
            <p className="text-sm font-semibold" style={{ color: 'var(--text-main)' }}>
                {messages[index]}
            </p>
            <div className="flex gap-1.5">
                {messages.map((_, i) => (
                    <div
                        key={i}
                        className="w-1.5 h-1.5 rounded-full transition-all duration-300"
                        style={{
                            background: i <= index ? 'var(--accent-primary)' : 'var(--border-light)',
                        }}
                    />
                ))}
            </div>
        </div>
    );
}

export default function OptimizationLoader({ model }: OptimizationLoaderProps) {
    const messages = MESSAGES[model];

    return (
        <div>
            <LoaderInner key={model} messages={messages} />
            <p className="text-xs text-center mt-3" style={{ color: 'var(--text-muted)' }}>
                {model === 'montecarlo'
                    ? 'La simulacion Monte Carlo puede tardar unos segundos adicionales.'
                    : 'Esto suele tardar menos de 5 segundos.'}
            </p>
        </div>
    );
}
