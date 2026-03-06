-- ══════════════════════════════════════════════════════════════
-- Script de Configuración de Políticas de Seguridad (RLS) Estrictas
-- Tablas: profiles (perfiles), estrategias, activos_portafolio, historial_pagos
-- ══════════════════════════════════════════════════════════════

-- 0. CREACIÓN DE TABLAS (SI NO EXISTEN) PARA EVITAR ERRORES
-- (Creamos placeholders básicos para las tablas que falten en tu BD)
CREATE TABLE IF NOT EXISTS public.activos_portafolio (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    ticker TEXT NOT NULL,
    cantidad NUMERIC NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.historial_pagos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    monto NUMERIC NOT NULL,
    estado TEXT NOT NULL,
    fecha_pago TIMESTAMPTZ DEFAULT now()
);

-- 1. Habilitar RLS en todas las tablas requeridas
-- En tu esquema anterior la tabla de perfiles se llama 'profiles'
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.estrategias ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.activos_portafolio ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.historial_pagos ENABLE ROW LEVEL SECURITY;

-- ══════════════════════════════════════════════════════════════
-- 1. PROFILES ( PERFILES )
-- Nota: En la tabla `profiles` el campo que aloja el UID es `id` 
-- según tu schema.sql original.
-- ══════════════════════════════════════════════════════════════
DROP POLICY IF EXISTS "profiles_select" ON public.profiles;
DROP POLICY IF EXISTS "profiles_insert" ON public.profiles;
DROP POLICY IF EXISTS "profiles_update" ON public.profiles;
DROP POLICY IF EXISTS "profiles_delete" ON public.profiles;
DROP POLICY IF EXISTS "Users can view own profile" ON public.profiles;
DROP POLICY IF EXISTS "Users can update own profile" ON public.profiles;

CREATE POLICY "profiles_select" ON public.profiles
    FOR SELECT TO authenticated
    USING (id = auth.uid());

CREATE POLICY "profiles_insert" ON public.profiles
    FOR INSERT TO authenticated
    WITH CHECK (id = auth.uid());

CREATE POLICY "profiles_update" ON public.profiles
    FOR UPDATE TO authenticated
    USING (id = auth.uid())
    WITH CHECK (id = auth.uid());

CREATE POLICY "profiles_delete" ON public.profiles
    FOR DELETE TO authenticated
    USING (id = auth.uid());

-- ══════════════════════════════════════════════════════════════
-- 2. ESTRATEGIAS
-- ══════════════════════════════════════════════════════════════
DROP POLICY IF EXISTS "estrategias_select" ON public.estrategias;
DROP POLICY IF EXISTS "estrategias_insert" ON public.estrategias;
DROP POLICY IF EXISTS "estrategias_update" ON public.estrategias;
DROP POLICY IF EXISTS "estrategias_delete" ON public.estrategias;
DROP POLICY IF EXISTS "Users manage own strategies" ON public.estrategias;

CREATE POLICY "estrategias_select" ON public.estrategias
    FOR SELECT TO authenticated
    USING (user_id = auth.uid() AND deleted_at IS NULL);

CREATE POLICY "estrategias_insert" ON public.estrategias
    FOR INSERT TO authenticated
    WITH CHECK (user_id = auth.uid() AND deleted_at IS NULL);

CREATE POLICY "estrategias_update" ON public.estrategias
    FOR UPDATE TO authenticated
    USING (user_id = auth.uid() AND deleted_at IS NULL)
    WITH CHECK (user_id = auth.uid());

CREATE POLICY "estrategias_delete" ON public.estrategias
    FOR DELETE TO authenticated
    USING (false); -- Prevents hard deletes; frontend will map delete to update deleted_at = now()

-- ══════════════════════════════════════════════════════════════
-- 3. ACTIVOS PORTAFOLIO
-- ══════════════════════════════════════════════════════════════
DROP POLICY IF EXISTS "activos_portafolio_select" ON public.activos_portafolio;
DROP POLICY IF EXISTS "activos_portafolio_insert" ON public.activos_portafolio;
DROP POLICY IF EXISTS "activos_portafolio_update" ON public.activos_portafolio;
DROP POLICY IF EXISTS "activos_portafolio_delete" ON public.activos_portafolio;

CREATE POLICY "activos_portafolio_select" ON public.activos_portafolio
    FOR SELECT TO authenticated
    USING (user_id = auth.uid());

CREATE POLICY "activos_portafolio_insert" ON public.activos_portafolio
    FOR INSERT TO authenticated
    WITH CHECK (user_id = auth.uid());

CREATE POLICY "activos_portafolio_update" ON public.activos_portafolio
    FOR UPDATE TO authenticated
    USING (user_id = auth.uid())
    WITH CHECK (user_id = auth.uid());

CREATE POLICY "activos_portafolio_delete" ON public.activos_portafolio
    FOR DELETE TO authenticated
    USING (user_id = auth.uid());

-- ══════════════════════════════════════════════════════════════
-- 4. HISTORIAL PAGOS
-- Requisito: SOLO lectura (SELECT). 
-- INSERT/UPDATE se deben hacer mediante el Service Role de Supabase.
-- ══════════════════════════════════════════════════════════════
DROP POLICY IF EXISTS "historial_pagos_select" ON public.historial_pagos;

CREATE POLICY "historial_pagos_select" ON public.historial_pagos
    FOR SELECT TO authenticated
    USING (user_id = auth.uid());

-- ══════════════════════════════════════════════════════════════
-- PROTECCIÓN EXTRA DE COLUMNAS A NIVEL SQL (REQUISITO 3)
-- Evitamos la actualización de las llaves foráneas modificando
-- directamente los permisos del ROL.
-- ══════════════════════════════════════════════════════════════
REVOKE UPDATE (id) ON public.profiles FROM authenticated;
REVOKE UPDATE (id, user_id) ON public.estrategias FROM authenticated;
REVOKE UPDATE (id, user_id) ON public.activos_portafolio FROM authenticated;
REVOKE UPDATE (id, user_id) ON public.historial_pagos FROM authenticated;
