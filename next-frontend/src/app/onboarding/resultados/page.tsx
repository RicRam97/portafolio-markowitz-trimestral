import type { Metadata } from 'next';
import { redirect } from 'next/navigation';
import { createClient } from '@/utils/supabase/server';
import ResultadosClient from './ResultadosClient';

export const metadata: Metadata = {
  title: 'Tu Perfil Inversor — Kaudal',
  description: 'Resultados de tu onboarding personalizado.',
};

export default async function ResultadosPage() {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) redirect('/login');

  const [toleranciaRes, suenosRes] = await Promise.all([
    supabase
      .from('test_tolerancia')
      .select('perfil_resultado, volatilidad_maxima, puntaje_total, descripcion_perfil')
      .eq('user_id', user.id)
      .order('fecha_completado', { ascending: false })
      .limit(1)
      .maybeSingle(),
    supabase
      .from('test_suenos')
      .select('retorno_minimo_requerido, nivel, anos_horizonte, meta_tipo')
      .eq('user_id', user.id)
      .maybeSingle(),
  ]);

  return (
    <ResultadosClient
      userId={user.id}
      tolerancia={toleranciaRes.data ?? null}
      suenos={suenosRes.data ?? null}
    />
  );
}
