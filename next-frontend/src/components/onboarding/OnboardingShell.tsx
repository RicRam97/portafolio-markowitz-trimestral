'use client';

import { usePathname } from 'next/navigation';

const STEPS = [
  { path: '/onboarding', label: 'Bienvenida' },
  { path: '/onboarding/test-suenos', label: 'Tus Metas' },
  { path: '/onboarding/test-tolerancia', label: 'Tu Riesgo' },
  { path: '/onboarding/resultados', label: 'Resultados' },
];

export default function OnboardingShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const stepIndex = STEPS.findIndex((s) => s.path === pathname);
  const currentStep = stepIndex === -1 ? 0 : stepIndex;

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: 'var(--bg-main)' }}>
      {/* Sticky header with progress */}
      <header style={{
        position: 'sticky', top: 0, zIndex: 50,
        background: 'rgba(10, 15, 30, 0.90)',
        backdropFilter: 'blur(16px)',
        borderBottom: '1px solid var(--border-light)',
        padding: '14px 20px 12px',
      }}>
        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', marginBottom: '14px' }}>
          <div style={{
            width: '26px', height: '26px', borderRadius: '7px',
            background: 'linear-gradient(135deg, #2563EB, #14B8A6)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="white">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm.31 13.14v1.67h-1.56v-1.62c-1.5-.31-2.76-1.27-2.86-2.97h1.71c.09.92.72 1.64 2.32 1.64 1.71 0 2.1-.86 2.1-1.39 0-.71-.39-1.4-2.34-1.86-2.17-.52-3.66-1.42-3.66-3.21 0-1.51 1.21-2.49 2.72-2.81V5h1.56v1.72c1.62.4 2.44 1.63 2.49 2.97h-1.71c-.04-.98-.56-1.64-1.94-1.64-1.31 0-2.1.59-2.1 1.43 0 .73.57 1.22 2.34 1.67 1.76.46 3.65 1.22 3.66 3.42-.01 1.61-1.21 2.48-2.73 2.57z"/>
            </svg>
          </div>
          <span style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '0.95rem', color: 'var(--text-main)' }}>
            Kaudal
          </span>
        </div>

        {/* Step indicators */}
        <div style={{ display: 'flex', alignItems: 'center', maxWidth: '420px', margin: '0 auto' }}>
          {STEPS.map((step, idx) => {
            const isCompleted = idx < currentStep;
            const isCurrent = idx === currentStep;

            return (
              <div key={step.path} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '5px' }}>
                {/* Connector + circle row */}
                <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                  {/* Left connector */}
                  {idx > 0 && (
                    <div style={{
                      flex: 1, height: '2px',
                      background: isCompleted || isCurrent
                        ? 'linear-gradient(90deg, #2563EB, #14B8A6)'
                        : 'rgba(255,255,255,0.1)',
                      transition: 'background 0.4s ease',
                    }} />
                  )}

                  {/* Circle */}
                  <div style={{
                    width: '26px', height: '26px', borderRadius: '50%', flexShrink: 0,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '0.65rem', fontWeight: 700,
                    background: isCompleted
                      ? 'linear-gradient(135deg, #14B8A6, #0D9488)'
                      : isCurrent
                        ? 'linear-gradient(135deg, #2563EB, #1D4ED8)'
                        : 'rgba(255,255,255,0.07)',
                    color: isCompleted || isCurrent ? 'white' : 'var(--text-muted)',
                    boxShadow: isCurrent ? '0 0 0 3px rgba(37,99,235,0.25)' : 'none',
                    transition: 'all 0.35s ease',
                  }}>
                    {isCompleted ? '✓' : idx + 1}
                  </div>

                  {/* Right connector */}
                  {idx < STEPS.length - 1 && (
                    <div style={{
                      flex: 1, height: '2px',
                      background: isCompleted
                        ? 'linear-gradient(90deg, #2563EB, #14B8A6)'
                        : 'rgba(255,255,255,0.1)',
                      transition: 'background 0.4s ease',
                    }} />
                  )}
                </div>

                {/* Label */}
                <span style={{
                  fontSize: '0.58rem', textAlign: 'center', lineHeight: 1.2,
                  color: isCurrent ? 'var(--text-main)' : isCompleted ? '#14B8A6' : 'var(--text-muted)',
                  fontWeight: isCurrent ? 600 : 400,
                  transition: 'color 0.3s ease',
                }}>
                  {step.label}
                </span>
              </div>
            );
          })}
        </div>
      </header>

      {/* Page content */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {children}
      </main>
    </div>
  );
}
