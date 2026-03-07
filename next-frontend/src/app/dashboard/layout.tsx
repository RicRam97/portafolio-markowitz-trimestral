import { redirect } from 'next/navigation';
import { createClient } from '@/utils/supabase/server';
import DashboardUI from '@/components/dashboard/DashboardUI';

export default async function DashboardLayout({ children }: { children: React.ReactNode }) {
    const supabase = await createClient();
    const { data: { user } } = await supabase.auth.getUser();

    if (!user) redirect('/login');

    // Fetch user profile for greeting and test completion status
    const { data: profile } = await supabase
        .from('profiles')
        .select('nombre, test_completado, plan, onboarding_completado')
        .eq('id', user.id)
        .single();

    // Gate: redirect to onboarding if not yet completed
    if (!profile?.onboarding_completado) redirect('/onboarding');

    const nombre = profile?.nombre || user.email?.split('@')[0] || 'Usuario';
    const testCompletado = profile?.test_completado ?? false;

    return (
        <DashboardUI
            nombre={nombre}
            email={user.email || ''}
            testCompletado={testCompletado}
        >
            {children}
        </DashboardUI>
    );
}
