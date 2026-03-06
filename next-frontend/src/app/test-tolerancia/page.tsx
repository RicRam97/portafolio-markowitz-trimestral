import type { Metadata } from 'next';
import { redirect } from 'next/navigation';
import { createClient } from '@/utils/supabase/server';
import TestToleranciaRiesgo from '@/components/TestToleranciaRiesgo';

export const metadata: Metadata = {
    title: 'Test de Tolerancia al Riesgo — Kaudal',
    description: 'Descubre tu perfil de inversionista en 3 pasos.',
};

export default async function TestToleranciaPage() {
    const supabase = await createClient();
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) redirect('/login');

    return <TestToleranciaRiesgo />;
}
