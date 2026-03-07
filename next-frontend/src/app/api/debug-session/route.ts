import { NextResponse } from 'next/server';
import { createClient } from '@/utils/supabase/server';

export async function GET() {
  const supabase = await createClient();
  const { data: { user }, error: authError } = await supabase.auth.getUser();

  if (!user) {
    return NextResponse.json({
      step: 'auth',
      authenticated: false,
      error: authError?.message ?? 'no session',
    });
  }

  const { data: profile, error: profileError } = await supabase
    .from('profiles')
    .select('id, nombre, onboarding_completado, test_completado')
    .eq('id', user.id)
    .single();

  return NextResponse.json({
    step: 'profile',
    authenticated: true,
    userId: user.id,
    email: user.email,
    profile: profile ?? null,
    profileError: profileError?.message ?? null,
    profileErrorCode: profileError?.code ?? null,
  });
}
