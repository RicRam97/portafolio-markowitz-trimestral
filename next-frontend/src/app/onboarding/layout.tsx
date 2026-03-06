import { redirect } from 'next/navigation';
import { createClient } from '@/utils/supabase/server';
import OnboardingShell from '@/components/onboarding/OnboardingShell';

export default async function OnboardingLayout({ children }: { children: React.ReactNode }) {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();

  if (!user) redirect('/login');

  return <OnboardingShell>{children}</OnboardingShell>;
}
