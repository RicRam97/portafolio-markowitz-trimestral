'use client';

import { createContext, useContext } from 'react';

const OnboardingContext = createContext<string | null>(null);

export function OnboardingProvider({ userId, children }: { userId: string; children: React.ReactNode }) {
  return <OnboardingContext.Provider value={userId}>{children}</OnboardingContext.Provider>;
}

export function useOnboardingUserId() {
  const userId = useContext(OnboardingContext);
  if (!userId) throw new Error('useOnboardingUserId must be used within OnboardingProvider');
  return userId;
}
