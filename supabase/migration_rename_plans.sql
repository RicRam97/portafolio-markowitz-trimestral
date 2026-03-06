-- ══════════════════════════════════════════════════════════════
-- Kaudal — Migración: Renombrar planes free/premium → basico/ultra
-- ══════════════════════════════════════════════════════════════
-- Ejecuta este archivo en: Supabase Dashboard → SQL Editor → New Query
-- IMPORTANTE: Ejecutar ANTES de desplegar código que use los nuevos nombres.
-- ══════════════════════════════════════════════════════════════

-- 1. Eliminar constraint PRIMERO (para permitir nuevos valores)
ALTER TABLE public.profiles DROP CONSTRAINT IF EXISTS profiles_plan_check;

-- 2. Actualizar registros existentes
UPDATE public.profiles SET plan = 'basico' WHERE plan = 'free';
UPDATE public.profiles SET plan = 'ultra'  WHERE plan = 'premium';

-- 3. Recrear constraint con nuevos valores
ALTER TABLE public.profiles ADD CONSTRAINT profiles_plan_check
    CHECK (plan IN ('basico', 'pro', 'ultra'));

-- 4. Actualizar valor por defecto
ALTER TABLE public.profiles ALTER COLUMN plan SET DEFAULT 'basico';
