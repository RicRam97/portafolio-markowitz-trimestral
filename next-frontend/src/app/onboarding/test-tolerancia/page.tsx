'use client';

import TestToleranciaRiesgo from '@/components/TestToleranciaRiesgo';
import { useOnboardingUserId } from '@/components/onboarding/OnboardingContext';

export default function OnboardingTestToleranciaPage() {
  const userId = useOnboardingUserId();
  return <TestToleranciaRiesgo redirectTo="/onboarding/resultados" userId={userId} />;
}
