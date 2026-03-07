-- Migration: Add modo_experto to profiles
-- Allows users to toggle expert mode (all metrics visible by default)

ALTER TABLE public.profiles
ADD COLUMN IF NOT EXISTS modo_experto BOOLEAN NOT NULL DEFAULT false;
