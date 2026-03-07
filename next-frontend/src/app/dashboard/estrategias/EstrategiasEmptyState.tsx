'use client';

import { useRouter } from 'next/navigation';
import EmptyState from '@/components/EmptyState';

export default function EstrategiasEmptyState() {
    const router = useRouter();

    return (
        <EmptyState
            icon="🚀"
            title="¡Comienza tu viaje!"
            description="Aún no has creado tu primera estrategia. Es rápido y fácil: solo toma 3 minutos."
            primaryAction={{
                label: 'Crear mi Primera Estrategia',
                onClick: () => router.push('/dashboard/optimizar'),
            }}
            secondaryAction={{
                label: 'Ver tutorial en video (2 min)',
                href: '/tutorial',
            }}
        />
    );
}
