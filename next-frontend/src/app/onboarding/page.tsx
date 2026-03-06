'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';

export default function OnboardingWelcomePage() {
  const router = useRouter();
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 60);
    return () => clearTimeout(t);
  }, []);

  const BENEFITS = [
    { icon: '🎯', text: 'Identifica tus metas de inversión y el capital necesario para alcanzarlas' },
    { icon: '🛡️', text: 'Descubre tu perfil de riesgo real con base en tu situación financiera actual' },
    { icon: '📊', text: 'Recibe recomendaciones de portafolio 100% adaptadas a ti' },
  ];

  return (
    <div style={{
      flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: '32px 16px',
    }}>
      <div style={{
        width: '100%', maxWidth: '480px',
        opacity: visible ? 1 : 0,
        transform: visible ? 'translateY(0)' : 'translateY(20px)',
        transition: 'opacity 0.5s ease, transform 0.5s ease',
      }}>
        {/* Illustration */}
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <div style={{
            width: '88px', height: '88px', borderRadius: '22px',
            background: 'linear-gradient(135deg, rgba(37,99,235,0.18), rgba(20,184,166,0.18))',
            border: '1px solid rgba(37,99,235,0.28)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '2.8rem', margin: '0 auto 20px',
          }}>
            🧭
          </div>
          <h1 style={{
            fontFamily: 'var(--font-display)', fontSize: '1.75rem', fontWeight: 800,
            color: 'var(--text-main)', marginBottom: '14px', lineHeight: 1.25,
          }}>
            Bienvenido a Kaudal
          </h1>
          <p style={{
            color: 'var(--text-muted)', fontSize: '0.93rem', lineHeight: 1.75,
            maxWidth: '360px', margin: '0 auto',
          }}>
            Antes de empezar, necesitamos conocer tus metas y nivel de riesgo para darte
            recomendaciones personalizadas.
          </p>
        </div>

        {/* Benefits card */}
        <div style={{
          background: 'var(--bg-panel)', border: '1px solid var(--border-light)',
          borderRadius: '16px', padding: '22px 20px', marginBottom: '20px',
          backdropFilter: 'blur(16px)',
        }}>
          {BENEFITS.map((b, i) => (
            <div
              key={i}
              style={{
                display: 'flex', alignItems: 'flex-start', gap: '14px',
                marginBottom: i < BENEFITS.length - 1 ? '16px' : 0,
                opacity: visible ? 1 : 0,
                transform: visible ? 'translateX(0)' : 'translateX(-10px)',
                transition: `opacity 0.4s ease ${0.15 + i * 0.1}s, transform 0.4s ease ${0.15 + i * 0.1}s`,
              }}
            >
              <span style={{ fontSize: '1.25rem', flexShrink: 0, marginTop: '1px' }}>{b.icon}</span>
              <span style={{ fontSize: '0.875rem', color: 'var(--text-main)', lineHeight: 1.65 }}>
                {b.text}
              </span>
            </div>
          ))}
        </div>

        {/* Time estimate */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          gap: '8px', marginBottom: '24px',
        }}>
          <span style={{ fontSize: '1rem' }}>⏱️</span>
          <span style={{ fontSize: '0.84rem', color: 'var(--text-muted)' }}>
            Solo toma{' '}
            <strong style={{ color: 'var(--text-main)', fontWeight: 600 }}>5 minutos</strong>
          </span>
        </div>

        {/* CTA */}
        <button
          onClick={() => router.push('/onboarding/test-suenos')}
          style={{
            width: '100%', padding: '16px', borderRadius: '12px', border: 'none',
            cursor: 'pointer', fontWeight: 700, fontSize: '1rem',
            fontFamily: 'var(--font-display)',
            background: 'linear-gradient(135deg, #2563EB, #1D4ED8)', color: 'white',
            boxShadow: '0 4px 24px rgba(37,99,235,0.38)',
            transition: 'opacity 0.2s ease, transform 0.2s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.opacity = '0.88';
            e.currentTarget.style.transform = 'translateY(-1px)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.opacity = '1';
            e.currentTarget.style.transform = 'translateY(0)';
          }}
        >
          Empezar ahora →
        </button>
      </div>
    </div>
  );
}
