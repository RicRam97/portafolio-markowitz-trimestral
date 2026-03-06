import type { PlanTier } from './types';

export const PLAN_LIMITS: Record<PlanTier, number> = {
    basico: 1,
    pro: 3,
    ultra: 10,
};

export const PLAN_LABELS: Record<PlanTier, string> = {
    basico: 'Básico',
    pro: 'Pro',
    ultra: 'Ultra',
};

export const PLAN_UPGRADE_TARGET: Partial<Record<PlanTier, PlanTier>> = {
    basico: 'pro',
    pro: 'ultra',
};

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';
