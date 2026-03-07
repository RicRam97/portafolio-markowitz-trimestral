import { redirect } from 'next/navigation';
import { createClient } from '@/utils/supabase/server';
import OnboardingShell from '@/components/onboarding/OnboardingShell';
import { OnboardingProvider } from '@/components/onboarding/OnboardingContext';

export default async function OnboardingLayout({ children }: { children: React.ReactNode }) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();

  if (!user) redirect('/login');

  return (
    <OnboardingProvider userId={user.id}>
      <OnboardingShell>{children}</OnboardingShell>
    </OnboardingProvider>
  );
}
