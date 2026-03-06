import type { Metadata } from 'next';
import Navbar from '@/components/layout/Navbar';
import Footer from '@/components/layout/Footer';
import { createClient } from '@/utils/supabase/server';
import type { PlanTier } from '@/lib/types';
import SoporteTabs from '@/components/soporte/SoporteTabs';
import ContactSection from '@/components/soporte/ContactSection';

export const metadata: Metadata = {
    title: 'Soporte y Ayuda — Kaudal',
    description: 'Preguntas frecuentes, glosario financiero y contacto de soporte de Kaudal.',
};

export default async function SoportePage() {
    let userPlan: PlanTier | null = null;
    let userId: string | undefined;

    try {
        const supabase = await createClient();
        const { data: { user } } = await supabase.auth.getUser();

        if (user) {
            userId = user.id;
            const { data: profile } = await supabase
                .from('profiles')
                .select('plan')
                .eq('id', user.id)
                .single();
            userPlan = (profile?.plan || 'basico') as PlanTier;
        }
    } catch {
        // Not authenticated — userPlan stays null
    }

    return (
        <>
            <Navbar />
            <main className="page-wrapper" style={{ paddingTop: '100px', minHeight: 'calc(100vh - 60px)' }}>
                <div className="about-page" style={{ maxWidth: '900px' }}>
                    {/* Header */}
                    <div className="mb-8 text-center">
                        <h1 className="text-2xl font-bold mb-2" style={{ fontFamily: 'var(--font-display)', color: 'var(--text-main)' }}>
                            Soporte y Ayuda
                        </h1>
                        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                            Todo lo que necesitas para aprovechar Kaudal al máximo.
                        </p>
                    </div>

                    {/* Tabs: FAQ / Glosario */}
                    <SoporteTabs />

                    {/* Contact Section */}
                    <div style={{ marginBottom: '48px' }}>
                        <ContactSection userPlan={userPlan} userId={userId} />
                    </div>
                </div>
            </main>
            <Footer />
        </>
    );
}
