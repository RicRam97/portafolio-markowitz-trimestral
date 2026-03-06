import type { Metadata } from 'next';
import { redirect } from 'next/navigation';
import { createClient } from '@/utils/supabase/server';
import type { PlanTier, Portafolio, TickerFavorito } from '@/lib/types';
import StrategyCounter from '@/components/dashboard/StrategyCounter';
import RecentPortfolios from '@/components/dashboard/RecentPortfolios';
import FavoriteTickers from '@/components/dashboard/FavoriteTickers';
import RebalanceAlerts from '@/components/dashboard/RebalanceAlerts';

export const metadata: Metadata = {
    title: 'Dashboard — Kaudal',
    description: 'Tu resumen de métricas e inversiones.',
};

export default async function DashboardPage() {
    const supabase = await createClient();
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) redirect('/login');

    const [profileRes, strategiesRes, portfoliosRes, favoritesRes, allPortfoliosRes] = await Promise.all([
        supabase.from('profiles').select('plan').eq('id', user.id).single(),
        supabase.from('estrategias').select('id', { count: 'exact', head: true }).eq('user_id', user.id),
        supabase
            .from('portafolios')
            .select('*')
            .eq('user_id', user.id)
            .order('created_at', { ascending: false })
            .limit(3),
        supabase
            .from('tickers_favoritos')
            .select('*')
            .eq('user_id', user.id)
            .order('added_at', { ascending: false }),
        supabase
            .from('portafolios')
            .select('id, nombre, presupuesto, allocation, created_at')
            .eq('user_id', user.id)
            .neq('allocation', '[]'),
    ]);

    const plan = (profileRes.data?.plan || 'basico') as PlanTier;
    const strategyCount = strategiesRes.count || 0;
    const portfolios = (portfoliosRes.data || []) as Portafolio[];
    const favorites = (favoritesRes.data || []) as TickerFavorito[];
    const allPortfolios = (allPortfoliosRes.data || []) as Portafolio[];

    return (
        <div className="max-w-[1200px] mx-auto">
            <div className="mb-6">
                <h2 className="text-xl font-bold" style={{ fontFamily: 'var(--font-display)' }}>
                    Panel de Métricas
                </h2>
                <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
                    Resumen de tu actividad e inversiones.
                </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <StrategyCounter count={strategyCount} plan={plan} />
                <RecentPortfolios portfolios={portfolios} />
                <FavoriteTickers favorites={favorites} />
                <RebalanceAlerts portfolios={allPortfolios} />
            </div>
        </div>
    );
}
