'use client';

import { useTheme } from 'next-themes';
import { useState } from 'react';
import { motion } from 'framer-motion';
import { Sun, Moon, Monitor } from 'lucide-react';

const options = [
    { value: 'light', label: 'Claro', Icon: Sun },
    { value: 'dark', label: 'Oscuro', Icon: Moon },
    { value: 'system', label: 'Sistema', Icon: Monitor },
] as const;

export default function ConfigTheme() {
    const { theme, setTheme } = useTheme();
    const [mounted] = useState(() => typeof window !== 'undefined');

    if (!mounted) return null;

    return (
        <div
            className="glass-panel p-5 flex items-center justify-between gap-4"
            style={{ borderLeft: '3px solid var(--accent-secondary)' }}
        >
            <div className="flex items-start gap-3">
                <div
                    className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 mt-0.5"
                    style={{ background: 'rgba(20,184,166,0.12)' }}
                >
                    {theme === 'dark' ? (
                        <Moon className="w-5 h-5" style={{ color: 'var(--accent-secondary)' }} />
                    ) : theme === 'light' ? (
                        <Sun className="w-5 h-5" style={{ color: 'var(--accent-secondary)' }} />
                    ) : (
                        <Monitor className="w-5 h-5" style={{ color: 'var(--accent-secondary)' }} />
                    )}
                </div>
                <div>
                    <h3
                        className="text-sm font-semibold"
                        style={{ color: 'var(--text-main)' }}
                    >
                        Apariencia
                    </h3>
                    <p
                        className="text-xs mt-0.5 leading-relaxed"
                        style={{ color: 'var(--text-muted)' }}
                    >
                        Elige entre modo claro, oscuro o automatico segun tu sistema operativo.
                    </p>
                </div>
            </div>

            {/* Theme selector */}
            <div className="flex gap-1 flex-shrink-0 rounded-lg p-1" style={{ background: 'var(--overlay-soft)' }}>
                {options.map(({ value, label, Icon }) => {
                    const active = theme === value;
                    return (
                        <button
                            key={value}
                            onClick={() => setTheme(value)}
                            className="relative px-3 py-1.5 rounded-md text-xs font-medium transition-colors"
                            style={{
                                color: active ? 'var(--text-main)' : 'var(--text-muted)',
                            }}
                            aria-label={label}
                        >
                            {active && (
                                <motion.span
                                    layoutId="theme-indicator"
                                    className="absolute inset-0 rounded-md"
                                    style={{ background: 'var(--overlay-hover)', border: '1px solid var(--border-light)' }}
                                    transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                                />
                            )}
                            <span className="relative flex items-center gap-1.5">
                                <Icon className="w-3.5 h-3.5" />
                                {label}
                            </span>
                        </button>
                    );
                })}
            </div>
        </div>
    );
}
