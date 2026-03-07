import type { Metadata } from 'next';
import { createClient } from '@/utils/supabase/server';
import PerfilInversor from '@/components/dashboard/PerfilInversor';
import MisLogros from '@/components/dashboard/MisLogros';

export const metadata: Metadata = {
    title: 'Mi Perfil — Kaudal',
    description: 'Tu perfil de inversionista personalizado.',
};

export default async function PerfilPage() {
    const supabase = await createClient();
    const { data: { user } } = await supabase.auth.getUser();

    const [toleranciaRes, suenosRes, historialRes, badgesRes] = await Promise.all([
        supabase
            .from('test_tolerancia')
            .select('perfil_resultado, volatilidad_maxima, puntaje_total, fecha_completado')
            .eq('user_id', user!.id)
            .maybeSingle(),
        supabase
            .from('test_suenos')
            .select('retorno_minimo_requerido, nivel, anos_horizonte, meta_tipo, moneda')
            .eq('user_id', user!.id)
            .maybeSingle(),
        supabase
            .from('perfil_historial')
            .select('fecha, perfil_resultado, retorno_minimo, volatilidad_maxima')
            .eq('user_id', user!.id)
            .order('fecha', { ascending: false })
            .limit(50),
        supabase
            .from('user_badges')
            .select('badge_id, fecha_obtenido')
            .eq('user_id', user!.id)
            .order('fecha_obtenido', { ascending: false }),
    ]);

    return (
        <>
            <PerfilInversor
                tolerancia={toleranciaRes.data ?? null}
                suenos={suenosRes.data ?? null}
                historial={historialRes.data ?? []}
            />
            <div style={{ marginTop: '24px', maxWidth: '860px', marginLeft: 'auto', marginRight: 'auto' }}>
                <MisLogros userBadges={badgesRes.data ?? []} />
            </div>
        </>
    );
}
