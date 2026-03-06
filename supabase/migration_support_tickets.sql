-- ══════════════════════════════════════════════════════════════
-- Kaudal — Migración: Tabla support_tickets
-- ══════════════════════════════════════════════════════════════
-- Ejecuta este archivo en: Supabase Dashboard → SQL Editor → New Query
-- ══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS public.support_tickets (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    asunto     TEXT NOT NULL,
    mensaje    TEXT NOT NULL,
    tier       TEXT NOT NULL CHECK (tier IN ('pro', 'ultra')),
    status     TEXT NOT NULL DEFAULT 'abierto'
                   CHECK (status IN ('abierto', 'en_progreso', 'resuelto')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_support_tickets_user ON public.support_tickets(user_id);

ALTER TABLE public.support_tickets ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users manage own tickets"
    ON public.support_tickets FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);
