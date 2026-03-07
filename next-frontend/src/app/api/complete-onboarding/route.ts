import { NextResponse } from 'next/server';
import { createClient } from '@/utils/supabase/server';

export async function POST() {
  const supabase = await createClient();
  const { data: { user }, error: authError } = await supabase.auth.getUser();

  if (!user) {
    return NextResponse.json(
      { error: 'No autenticado', detail: authError?.message ?? 'sin sesión' },
      { status: 401 },
    );
  }

  const { error, count } = await supabase
    .from('profiles')
    .update({ onboarding_completado: true }, { count: 'exact' })
    .eq('id', user.id);

  if (error) {
    return NextResponse.json(
      { error: error.message, code: error.code },
      { status: 500 },
    );
  }

  return NextResponse.json({ ok: true, userId: user.id, rowsUpdated: count });
}
