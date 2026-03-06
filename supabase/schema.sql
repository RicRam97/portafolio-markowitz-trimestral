-- ══════════════════════════════════════════════════════════════
-- Kaudal — Esquema Completo de Supabase
-- ══════════════════════════════════════════════════════════════
-- Ejecuta este archivo en: Supabase Dashboard → SQL Editor → New Query
-- Prerequisito: Tener habilitado Auth en tu proyecto Supabase.
-- ══════════════════════════════════════════════════════════════


-- ┌─────────────────────────────────────────────────────────────┐
-- │ 1. PROFILES — Perfil de usuario (auto-creado al registrarse) │
-- └─────────────────────────────────────────────────────────────┘
CREATE TABLE IF NOT EXISTS public.profiles (
    id          UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email       TEXT,
    display_name TEXT,
    plan        TEXT NOT NULL DEFAULT 'basico'
                    CHECK (plan IN ('basico', 'pro', 'ultra')),
    onboarding_complete BOOLEAN NOT NULL DEFAULT false,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Índice para búsquedas por email
CREATE INDEX IF NOT EXISTS idx_profiles_email ON public.profiles(email);

-- RLS
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own profile"
    ON public.profiles FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
    ON public.profiles FOR UPDATE
    USING (auth.uid() = id);


-- ┌─────────────────────────────────────────────────────────────┐
-- │ 2. ESTRATEGIAS — Configuraciones de optimización guardadas  │
-- └─────────────────────────────────────────────────────────────┘
CREATE TABLE IF NOT EXISTS public.estrategias (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    nombre      TEXT NOT NULL DEFAULT 'Sin nombre',
    tipo        TEXT NOT NULL DEFAULT 'markowitz'
                    CHECK (tipo IN ('markowitz', 'hrp')),
    parametros  JSONB NOT NULL DEFAULT '{}',
    -- parametros ejemplo: {"budget": 10000, "tickers": ["AAPL","GOOGL"], "start_date": "..."}
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_estrategias_user ON public.estrategias(user_id);
CREATE INDEX IF NOT EXISTS idx_estrategias_created ON public.estrategias(created_at DESC);

ALTER TABLE public.estrategias ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users manage own strategies"
    ON public.estrategias FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);


-- ┌─────────────────────────────────────────────────────────────┐
-- │ 3. PORTAFOLIOS — Snapshots de resultados de optimización    │
-- └─────────────────────────────────────────────────────────────┘
CREATE TABLE IF NOT EXISTS public.portafolios (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    estrategia_id   UUID REFERENCES public.estrategias(id) ON DELETE SET NULL,
    nombre          TEXT NOT NULL DEFAULT 'Mi Portafolio',
    presupuesto     NUMERIC(14,2) NOT NULL DEFAULT 0,
    rendimiento_pct NUMERIC(6,2),
    volatilidad_pct NUMERIC(6,2),
    sharpe_ratio    NUMERIC(6,3),
    allocation      JSONB NOT NULL DEFAULT '[]',
    -- allocation ejemplo: [{"ticker":"AAPL","weight_pct":35,"shares":10}]
    metricas        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_portafolios_user ON public.portafolios(user_id);
CREATE INDEX IF NOT EXISTS idx_portafolios_created ON public.portafolios(created_at DESC);

ALTER TABLE public.portafolios ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users manage own portfolios"
    ON public.portafolios FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);


-- ┌─────────────────────────────────────────────────────────────┐
-- │ 4. DREAM_TEST — Historial del Test de Sueños               │
-- └─────────────────────────────────────────────────────────────┘
CREATE TABLE IF NOT EXISTS public.dream_test (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    meta_costo       NUMERIC(14,2) NOT NULL,
    años             INT NOT NULL,
    capital_inicial  NUMERIC(14,2) NOT NULL DEFAULT 0,
    aporte_mensual   NUMERIC(14,2) NOT NULL DEFAULT 0,
    tasa_requerida   NUMERIC(6,2),
    perfil_sugerido  TEXT,
    resultado        JSONB DEFAULT '{}',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_dream_test_user ON public.dream_test(user_id);

ALTER TABLE public.dream_test ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users manage own dream tests"
    ON public.dream_test FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);


-- ┌─────────────────────────────────────────────────────────────┐
-- │ 5. RISK_TEST — Snapshots del Semáforo de Riesgo            │
-- └─────────────────────────────────────────────────────────────┘
CREATE TABLE IF NOT EXISTS public.risk_test (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    portafolio_id       UUID REFERENCES public.portafolios(id) ON DELETE SET NULL,
    monto_inversion     NUMERIC(14,2) NOT NULL,
    ganancia_esperada   NUMERIC(14,2),
    var_99              NUMERIC(14,2),
    -- var_99: Peor escenario (Value at Risk al 99%)
    resultado           JSONB DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_risk_test_user ON public.risk_test(user_id);

ALTER TABLE public.risk_test ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users manage own risk tests"
    ON public.risk_test FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);


-- ┌─────────────────────────────────────────────────────────────┐
-- │ 6. LEGAL_ACCEPTANCES — Aceptación de términos legales      │
-- └─────────────────────────────────────────────────────────────┘
CREATE TABLE IF NOT EXISTS public.legal_acceptances (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    document     TEXT NOT NULL
                     CHECK (document IN ('terms', 'privacy', 'disclaimer', 'cookies')),
    version      TEXT NOT NULL DEFAULT '1.0',
    ip_address   INET,
    user_agent   TEXT,
    accepted_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_legal_user ON public.legal_acceptances(user_id);

-- Evitar que un usuario acepte el mismo doc+versión más de una vez
CREATE UNIQUE INDEX IF NOT EXISTS idx_legal_unique
    ON public.legal_acceptances(user_id, document, version);

ALTER TABLE public.legal_acceptances ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users view own acceptances"
    ON public.legal_acceptances FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own acceptances"
    ON public.legal_acceptances FOR INSERT
    WITH CHECK (auth.uid() = user_id);


-- ┌─────────────────────────────────────────────────────────────┐
-- │ 7. NEWSLETTER — Suscripciones al boletín                   │
-- └─────────────────────────────────────────────────────────────┘
CREATE TABLE IF NOT EXISTS public.newsletter (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         TEXT NOT NULL,
    is_active     BOOLEAN NOT NULL DEFAULT true,
    source        TEXT DEFAULT 'website',
    subscribed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    unsub_at      TIMESTAMPTZ
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_newsletter_email ON public.newsletter(email);

ALTER TABLE public.newsletter ENABLE ROW LEVEL SECURITY;

-- Newsletter es pública para insertar (formulario anónimo)
CREATE POLICY "Anyone can subscribe"
    ON public.newsletter FOR INSERT
    WITH CHECK (true);

-- Solo el dueño del email puede ver/modificar (via servicio backend con service_role)
CREATE POLICY "Service role manages newsletter"
    ON public.newsletter FOR ALL
    USING (auth.role() = 'service_role');


-- ┌─────────────────────────────────────────────────────────────┐
-- │ TRIGGER: Auto-crear perfil al registrar usuario             │
-- └─────────────────────────────────────────────────────────────┘
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, email, display_name)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'display_name', split_part(NEW.email, '@', 1))
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Ejecutar el trigger cuando se inserta un usuario nuevo en auth.users
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_new_user();


-- ┌─────────────────────────────────────────────────────────────┐
-- │ TRIGGER: Actualizar updated_at automáticamente              │
-- └─────────────────────────────────────────────────────────────┘
CREATE OR REPLACE FUNCTION public.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_updated_at_profiles
    BEFORE UPDATE ON public.profiles
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

CREATE TRIGGER set_updated_at_estrategias
    BEFORE UPDATE ON public.estrategias
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();


-- ┌─────────────────────────────────────────────────────────────┐
-- │ 8b. SUPPORT_TICKETS — Tickets de soporte del usuario       │
-- └─────────────────────────────────────────────────────────────┘
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


-- ══════════════════════════════════════════════════════════════
-- ✅ Esquema listo. Verifica los resultados en Table Editor.
-- ══════════════════════════════════════════════════════════════


-- ┌─────────────────────────────────────────────────────────────┐
-- │ 8. FUNCIÓN RPC: Aceptar documentos legales sin sesión      │
-- └─────────────────────────────────────────────────────────────┘
-- Permite insertar aceptaciones legales durante el registro, 
-- cuando el usuario se acaba de crear pero aún no tiene sesión 
-- (porque requiere confirmación de email).
CREATE OR REPLACE FUNCTION public.accept_legal_documents(
    p_user_id UUID,
    p_documents TEXT[],
    p_version TEXT
)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER   -- IMPORTANTE: Ejecuta con privilegios de creador, saltando RLS
SET search_path = public
AS $$
DECLARE
    doc TEXT;
BEGIN
    FOREACH doc IN ARRAY p_documents
    LOOP
        INSERT INTO public.legal_acceptances (user_id, document, version)
        VALUES (p_user_id, doc, p_version)
        ON CONFLICT (user_id, document, version) DO NOTHING;
    END LOOP;
END;
$$;
