-- ══════════════════════════════════════════════════════════════
-- Migracion: Tablas para el endpoint POST /optimizar/markowitz
-- ══════════════════════════════════════════════════════════════

-- ┌─────────────────────────────────────────────────────────────┐
-- │ MARKET_DATA_CACHE — Cache de precios historicos (FMP)       │
-- └─────────────────────────────────────────────────────────────┘
CREATE TABLE IF NOT EXISTS public.market_data_cache (
    ticker      TEXT NOT NULL,
    fecha       DATE NOT NULL,
    close_price NUMERIC(14,4) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, fecha)
);

CREATE INDEX IF NOT EXISTS idx_mdc_ticker ON public.market_data_cache(ticker);
CREATE INDEX IF NOT EXISTS idx_mdc_fecha ON public.market_data_cache(fecha);

-- Sin RLS — esta tabla es gestionada exclusivamente por el backend con service_role
ALTER TABLE public.market_data_cache ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role manages market_data_cache"
    ON public.market_data_cache FOR ALL
    USING (auth.role() = 'service_role');


-- ┌─────────────────────────────────────────────────────────────┐
-- │ PORTAFOLIOS_CALCULADOS — Resultados de optimizacion         │
-- └─────────────────────────────────────────────────────────────┘
CREATE TABLE IF NOT EXISTS public.portafolios_calculados (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    tickers         TEXT[] NOT NULL,
    parametros      JSONB NOT NULL DEFAULT '{}',
    resultado       JSONB NOT NULL DEFAULT '{}',
    guardado        BOOLEAN NOT NULL DEFAULT false,
    fecha_calculo   TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_pc_user ON public.portafolios_calculados(user_id);
CREATE INDEX IF NOT EXISTS idx_pc_guardado ON public.portafolios_calculados(guardado);
CREATE INDEX IF NOT EXISTS idx_pc_fecha ON public.portafolios_calculados(fecha_calculo DESC);

ALTER TABLE public.portafolios_calculados ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users manage own calculated portfolios"
    ON public.portafolios_calculados FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- El backend con service_role tambien puede insertar
CREATE POLICY "Service role manages portafolios_calculados"
    ON public.portafolios_calculados FOR ALL
    USING (auth.role() = 'service_role');
