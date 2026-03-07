import type { Metadata } from 'next';
import { redirect } from 'next/navigation';
import { createClient } from '@/utils/supabase/server';
import ConfigModoExperto from './ConfigModoExperto';
import ConfigTheme from './ConfigTheme';

export const metadata: Metadata = {
    title: 'Configuracion — Kaudal',
    description: 'Ajusta las preferencias de tu cuenta.',
};

export default async function ConfigPage() {
    const supabase = await createClient();
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) redirect('/login');

    const { data: profile } = await supabase
        .from('profiles')
        .select('modo_experto')
        .eq('id', user.id)
        .single();

    return (
        <div className="max-w-[720px] mx-auto">
            <h1
                className="text-2xl font-bold mb-6"
                style={{ fontFamily: 'var(--font-display)', color: 'var(--text-main)' }}
            >
                Configuracion
            </h1>

            <div className="flex flex-col gap-4">
                <ConfigTheme />
                <ConfigModoExperto initialValue={profile?.modo_experto ?? false} />
            </div>
        </div>
    );
}
