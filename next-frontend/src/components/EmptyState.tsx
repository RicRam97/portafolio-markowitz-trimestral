'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { scaleIn } from '@/utils/animations';

interface EmptyStateAction {
    label: string;
    onClick: () => void;
}

interface EmptyStateSecondary {
    label: string;
    href: string;
}

interface EmptyStateProps {
    icon: string;
    title: string;
    description: string;
    primaryAction: EmptyStateAction;
    secondaryAction?: EmptyStateSecondary;
}

export default function EmptyState({
    icon,
    title,
    description,
    primaryAction,
    secondaryAction,
}: EmptyStateProps) {
    return (
        <motion.div
            className="glass-panel p-10 text-center"
            variants={scaleIn}
            initial="hidden"
            animate="visible"
        >
            <div
                className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4"
                style={{ background: 'rgba(37,99,235,0.1)' }}
            >
                <span className="text-3xl" role="img" aria-hidden="true">
                    {icon}
                </span>
            </div>

            <h3
                className="text-lg font-semibold mb-2"
                style={{ color: 'var(--text-main)' }}
            >
                {title}
            </h3>

            <p
                className="text-sm mb-6 max-w-md mx-auto"
                style={{ color: 'var(--text-muted)' }}
            >
                {description}
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
                <motion.button
                    onClick={primaryAction.onClick}
                    className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold"
                    style={{
                        background: 'linear-gradient(135deg, #2563eb, #14b8a6)',
                        color: '#fff',
                    }}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                >
                    {primaryAction.label}
                </motion.button>

                {secondaryAction && (
                    <Link
                        href={secondaryAction.href}
                        className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium transition-all hover:scale-[1.02]"
                        style={{
                            background: 'rgba(37,99,235,0.08)',
                            border: '1px solid rgba(37,99,235,0.2)',
                            color: 'var(--accent-primary)',
                        }}
                    >
                        {secondaryAction.label}
                    </Link>
                )}
            </div>
        </motion.div>
    );
}
