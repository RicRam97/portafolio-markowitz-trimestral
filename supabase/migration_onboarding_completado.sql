-- Migration: add onboarding_completado to profiles
-- Run this in the Supabase SQL Editor

ALTER TABLE profiles
ADD COLUMN IF NOT EXISTS onboarding_completado BOOLEAN NOT NULL DEFAULT false;

-- Existing users who already have test results are considered to have
-- completed onboarding. Mark them done so they are not re-routed.
UPDATE profiles
SET onboarding_completado = true
WHERE id IN (
    SELECT DISTINCT user_id FROM test_tolerancia
);
