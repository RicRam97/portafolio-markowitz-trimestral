import type { Metadata } from 'next';
import TestToleranciaRiesgo from '@/components/TestToleranciaRiesgo';

export const metadata: Metadata = {
  title: 'Tu Perfil de Riesgo — Kaudal',
  description: 'Descubre tu perfil de inversionista en 3 pasos.',
};

export default function OnboardingTestToleranciaPage() {
  return <TestToleranciaRiesgo redirectTo="/onboarding/resultados" />;
}
