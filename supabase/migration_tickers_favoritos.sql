-- ══════════════════════════════════════════════════════════════
-- Kaudal — Migración: Tabla tickers_favoritos
-- ══════════════════════════════════════════════════════════════
-- Ejecuta este archivo en: Supabase Dashboard → SQL Editor → New Query
-- ══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS public.tickers_favoritos (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    ticker      TEXT NOT NULL,
    nombre      TEXT,          -- nombre de la empresa (cacheado al agregar)
    added_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(user_id, ticker)    -- un usuario no puede tener el mismo ticker dos veces
);

CREATE INDEX IF NOT EXISTS idx_favoritos_user ON public.tickers_favoritos(user_id);

ALTER TABLE public.tickers_favoritos ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users manage own favorites"
    ON public.tickers_favoritos FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);
