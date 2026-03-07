import type { Metadata } from 'next';
import OptimizarClient from './OptimizarClient';

export const metadata: Metadata = {
    title: 'Optimizar Portafolio — Kaudal',
    description: 'Optimiza tu portafolio de inversión con la Teoría de Markowitz.',
};

export default function OptimizarPage() {
    return <OptimizarClient />;
}
