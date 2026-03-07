'use client';

import { useId } from 'react';
import * as Tooltip from '@radix-ui/react-tooltip';
import tooltips from '@/data/tooltips.json';

const tooltipKeyframes = `
@keyframes tooltipFadeIn {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}
`;

export type TooltipKey = keyof typeof tooltips;

interface EducationalTooltipProps {
    termKey: TooltipKey;
}

export default function EducationalTooltip({ termKey }: EducationalTooltipProps) {
    const data = tooltips[termKey];
    const tooltipId = useId();
    if (!data) return null;

    return (
        <Tooltip.Provider delayDuration={200}>
            <style>{tooltipKeyframes}</style>
            <Tooltip.Root>
                <Tooltip.Trigger asChild>
                    <button
                        type="button"
                        className="inline-flex items-center justify-center w-4 h-4 rounded-full text-[10px] font-bold leading-none cursor-help flex-shrink-0 transition-colors"
                        style={{
                            background: 'rgba(99,102,241,0.15)',
                            color: 'rgba(99,102,241,0.8)',
                            border: '1px solid rgba(99,102,241,0.25)',
                        }}
                        aria-label={`Mas informacion sobre ${data.titulo}`}
                        aria-describedby={tooltipId}
                    >
                        ?
                    </button>
                </Tooltip.Trigger>
                <Tooltip.Portal>
                    <Tooltip.Content
                        id={tooltipId}
                        role="tooltip"
                        side="top"
                        sideOffset={6}
                        className="z-50 max-w-[260px] rounded-lg px-3.5 py-2.5 shadow-xl"
                        style={{
                            background: 'rgba(15,23,42,0.96)',
                            border: '1px solid rgba(255,255,255,0.1)',
                            color: 'var(--text-main)',
                            animationName: 'tooltipFadeIn',
                            animationDuration: '200ms',
                            animationTimingFunction: 'ease-out',
                            animationFillMode: 'forwards',
                        }}
                    >
                        <p className="text-xs font-bold mb-1" style={{ color: 'var(--accent-primary)' }}>
                            {data.titulo}
                        </p>
                        <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                            {data.contenido}
                        </p>
                        <Tooltip.Arrow
                            style={{ fill: 'rgba(15,23,42,0.96)' }}
                            width={10}
                            height={5}
                        />
                    </Tooltip.Content>
                </Tooltip.Portal>
            </Tooltip.Root>
        </Tooltip.Provider>
    );
}
