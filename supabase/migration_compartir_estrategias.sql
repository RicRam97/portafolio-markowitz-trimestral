-- Migration: Add sharing columns to estrategias table
-- compartir_token: unique UUID for public sharing link
-- es_publica: boolean flag to enable/disable sharing

ALTER TABLE public.estrategias
    ADD COLUMN IF NOT EXISTS compartir_token UUID DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS es_publica BOOLEAN NOT NULL DEFAULT FALSE;

CREATE UNIQUE INDEX IF NOT EXISTS idx_estrategias_compartir_token
    ON public.estrategias(compartir_token)
    WHERE compartir_token IS NOT NULL;

-- Allow public read access to shared strategies (no auth required)
CREATE POLICY "Public can view shared strategies"
    ON public.estrategias FOR SELECT
    USING (es_publica = TRUE AND compartir_token IS NOT NULL);

-- Allow public read of portafolios linked to shared strategies
CREATE POLICY "Public can view portfolios of shared strategies"
    ON public.portafolios FOR SELECT
    USING (
        estrategia_id IN (
            SELECT id FROM public.estrategias
            WHERE es_publica = TRUE AND compartir_token IS NOT NULL
        )
    );
