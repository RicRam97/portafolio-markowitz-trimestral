'use client';

import { useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import FocusLock from 'react-focus-lock';

interface EducationalModalProps {
    open: boolean;
    titulo: string;
    descripcion: string;
    conexion: string;
    botonTexto: string;
    onContinue: () => void;
}

export default function EducationalModal({
    open,
    titulo,
    descripcion,
    conexion,
    botonTexto,
    onContinue,
}: EducationalModalProps) {
    const handleKeyDown = useCallback((e: KeyboardEvent) => {
        if (e.key === 'Escape') onContinue();
    }, [onContinue]);

    useEffect(() => {
        if (open) {
            document.addEventListener('keydown', handleKeyDown);
            return () => document.removeEventListener('keydown', handleKeyDown);
        }
    }, [open, handleKeyDown]);

    return (
        <AnimatePresence>
            {open && (
                <motion.div
                    key="edu-backdrop"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.25 }}
                    style={{
                        position: 'fixed',
                        inset: 0,
                        zIndex: 100,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        padding: '24px 16px',
                        background: 'rgba(0,0,0,0.65)',
                        backdropFilter: 'blur(4px)',
                    }}
                >
                  <FocusLock returnFocus>
                    <motion.div
                        key="edu-card"
                        role="dialog"
                        aria-modal="true"
                        aria-labelledby="edu-modal-title"
                        initial={{ opacity: 0, y: 60 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 40 }}
                        transition={{ type: 'spring', damping: 28, stiffness: 320 }}
                        style={{
                            width: '100%',
                            maxWidth: '440px',
                            background: 'var(--bg-panel)',
                            border: '1px solid var(--border-light)',
                            borderRadius: '20px',
                            padding: '32px 28px',
                            backdropFilter: 'blur(16px)',
                            textAlign: 'center',
                        }}
                    >
                        {/* Icon */}
                        <div style={{
                            width: '56px',
                            height: '56px',
                            borderRadius: '16px',
                            background: 'rgba(37,99,235,0.12)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '1.6rem',
                            margin: '0 auto 20px',
                        }}>
                            {'\uD83D\uDCA1'}
                        </div>

                        {/* Title */}
                        <h3 id="edu-modal-title" style={{
                            fontFamily: 'var(--font-display)',
                            fontSize: '1.25rem',
                            fontWeight: 700,
                            color: 'var(--text-main)',
                            marginBottom: '12px',
                        }}>
                            {titulo}
                        </h3>

                        {/* Description */}
                        <p style={{
                            color: 'var(--text-muted)',
                            fontSize: '0.88rem',
                            lineHeight: 1.65,
                            marginBottom: '16px',
                        }}>
                            {descripcion}
                        </p>

                        {/* Connection to next step */}
                        <div style={{
                            background: 'rgba(37,99,235,0.08)',
                            border: '1px solid rgba(37,99,235,0.2)',
                            borderRadius: '12px',
                            padding: '12px 16px',
                            marginBottom: '24px',
                        }}>
                            <p style={{
                                color: 'var(--accent-primary)',
                                fontSize: '0.84rem',
                                fontWeight: 500,
                                lineHeight: 1.55,
                            }}>
                                {conexion}
                            </p>
                        </div>

                        {/* Continue button */}
                        <motion.button
                            onClick={onContinue}
                            /* eslint-disable-next-line jsx-a11y/no-autofocus */
                            autoFocus
                            style={{
                                width: '100%',
                                padding: '14px',
                                borderRadius: '12px',
                                border: 'none',
                                cursor: 'pointer',
                                fontWeight: 600,
                                fontSize: '0.95rem',
                                fontFamily: 'var(--font-display)',
                                background: 'linear-gradient(135deg, #2563EB, #1D4ED8)',
                                color: 'white',
                                boxShadow: '0 4px 20px rgba(37,99,235,0.35)',
                            }}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                        >
                            {botonTexto}
                        </motion.button>
                    </motion.div>
                  </FocusLock>
                </motion.div>
            )}
        </AnimatePresence>
    );
}
