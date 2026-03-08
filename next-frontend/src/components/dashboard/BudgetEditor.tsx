'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Pencil, X } from 'lucide-react';

const STORAGE_KEY = 'last_budget_usd';

function formatCurrency(value: number): string {
    return value.toLocaleString('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0 });
}

interface BudgetEditorProps {
    value: number;
    onChange: (value: number) => void;
}

export default function BudgetEditor({ value, onChange }: BudgetEditorProps) {
    const [isOpen, setIsOpen] = useState(false);
    const [draft, setDraft] = useState(value);
    const inputRef = useRef<HTMLInputElement>(null);

    // Focus input when modal opens
    useEffect(() => {
        if (isOpen) {
            setTimeout(() => inputRef.current?.focus(), 100);
        }
    }, [isOpen]);

    // Load from localStorage on mount
    useEffect(() => {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored) {
            const parsed = Number(stored);
            if (!isNaN(parsed) && parsed >= 100) {
                onChange(parsed);
            }
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const openModal = useCallback(() => {
        setDraft(value);
        setIsOpen(true);
    }, [value]);

    const handleSave = useCallback(() => {
        const sanitized = Math.max(100, draft);
        onChange(sanitized);
        localStorage.setItem(STORAGE_KEY, String(sanitized));
        setIsOpen(false);
    }, [draft, onChange]);

    return (
        <>
            {/* Static display */}
            <div className="flex items-center gap-3">
                <span className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
                    Presupuesto
                </span>
                <span className="text-sm font-semibold" style={{ color: 'var(--text-main)' }}>
                    {formatCurrency(value)} USD
                </span>
                <button
                    onClick={openModal}
                    className="p-1.5 rounded-md transition-colors hover:bg-white/10"
                    style={{ color: 'var(--accent-primary)' }}
                    aria-label="Editar presupuesto"
                >
                    <Pencil size={14} />
                </button>
            </div>

            {/* Modal overlay */}
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        className="fixed inset-0 z-50 flex items-center justify-center p-4"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.2 }}
                    >
                        {/* Backdrop */}
                        {/* eslint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/no-static-element-interactions */}
                        <div
                            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                            onClick={() => setIsOpen(false)}
                        />

                        {/* Modal content */}
                        <motion.div
                            className="relative w-full max-w-md rounded-xl p-6"
                            style={{
                                background: 'var(--bg-card, rgba(15,23,42,0.95))',
                                border: '1px solid var(--border-light)',
                                boxShadow: '0 25px 50px rgba(0,0,0,0.5)',
                            }}
                            initial={{ scale: 0.9, opacity: 0, y: 20 }}
                            animate={{ scale: 1, opacity: 1, y: 0 }}
                            exit={{ scale: 0.9, opacity: 0, y: 20 }}
                            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
                        >
                            <button
                                onClick={() => setIsOpen(false)}
                                className="absolute top-4 right-4 p-1 rounded-md hover:bg-white/10 transition-colors"
                                style={{ color: 'var(--text-muted)' }}
                                aria-label="Cerrar"
                            >
                                <X size={18} />
                            </button>

                            <h3 className="text-lg font-bold mb-4" style={{ fontFamily: 'var(--font-display)', color: 'var(--text-main)' }}>
                                Con cuanto quieres iniciar esta simulacion?
                            </h3>

                            <div className="flex items-center gap-2 mb-5">
                                <span className="text-lg font-semibold" style={{ color: 'var(--text-muted)' }}>$</span>
                                <input
                                    ref={inputRef}
                                    type="number"
                                    value={draft}
                                    onChange={(e) => setDraft(Number(e.target.value))}
                                    onKeyDown={(e) => e.key === 'Enter' && handleSave()}
                                    min={100}
                                    step={100}
                                    className="flex-1 px-4 py-3 rounded-lg text-base outline-none"
                                    style={{
                                        background: 'rgba(15,23,42,0.6)',
                                        border: '1px solid var(--border-light)',
                                        color: 'var(--text-main)',
                                    }}
                                />
                                <span className="text-sm font-medium" style={{ color: 'var(--text-muted)' }}>USD</span>
                            </div>

                            <motion.button
                                onClick={handleSave}
                                className="btn btn-cta glow-effect w-full py-3 text-sm font-semibold"
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                            >
                                Guardar
                            </motion.button>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </>
    );
}
