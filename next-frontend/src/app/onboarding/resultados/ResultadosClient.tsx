'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
interface ToleranciaData {
  perfil_resultado: string;
  volatilidad_maxima: number;
  puntaje_total: number;
  descripcion_perfil?: string | null;
}

interface SuenosData {
  retorno_minimo_requerido: number;
  nivel: string;
  anos_horizonte: number;
  meta_tipo: string;
}

interface Props {
  userId: string;
  tolerancia: ToleranciaData | null;
  suenos: SuenosData | null;
}

// ---------------------------------------------------------------------------
// Profile metadata
// ---------------------------------------------------------------------------

const PERFIL_META: Record<string, { color: string; bg: string; icon: string; mensaje: string }> = {
  conservador: {
    color: '#14B8A6', bg: 'rgba(20,184,166,0.12)', icon: '🌿',
    mensaje: 'Tu cautela es una fortaleza. Un portafolio sólido de renta fija te dará la estabilidad que buscas mientras tu capital crece de forma segura.',
  },
  moderado: {
    color: '#2563EB', bg: 'rgba(37,99,235,0.12)', icon: '🌱',
    mensaje: 'Tienes el equilibrio perfecto entre prudencia y ambición. Con diversificación inteligente, maximizarás tus retornos sin perder el sueño.',
  },
  balanceado: {
    color: '#8B5CF6', bg: 'rgba(139,92,246,0.12)', icon: '⚖️',
    mensaje: 'Tu visión estratégica te distingue. Puedes aprovechar tanto la renta fija como la variable para construir riqueza de forma consistente.',
  },
  crecimiento: {
    color: '#F59E0B', bg: 'rgba(245,158,11,0.12)', icon: '🚀',
    mensaje: 'Tu tolerancia al riesgo abre puertas extraordinarias. Las acciones de alto crecimiento y mercados emergentes son tu terreno natural.',
  },
  agresivo: {
    color: '#EF4444', bg: 'rgba(239,68,68,0.12)', icon: '🔥',
    mensaje: 'Eres un inversionista de alto octanaje. Con horizonte largo y disciplina, el potencial de tu portafolio es prácticamente ilimitado.',
  },
};

function getPerfilMeta(perfil: string) {
  return PERFIL_META[perfil?.toLowerCase()] ?? PERFIL_META['moderado'];
}

const META_LABELS: Record<string, string> = {
  casa: 'Casa propia', retiro: 'Retiro', educacion: 'Educación',
  viaje: 'Viaje', libertad: 'Libertad financiera', otra: 'Meta personal',
};

const NIVEL_LABELS: Record<string, { label: string; color: string }> = {
  bajo:    { label: 'Alcanzable', color: '#14B8A6' },
  medio:   { label: 'Moderado',   color: '#F59E0B' },
  alto:    { label: 'Exigente',   color: '#EF4444' },
  muy_alto:{ label: 'Muy exigente', color: '#DC2626' },
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ResultadosClient({ userId, tolerancia: serverTolerancia, suenos: serverSuenos }: Props) {
  const router = useRouter();
  const [visible, setVisible] = useState(false);
  const [saving, setSaving] = useState(false);

  // Merge server data with localStorage fallback (handles race condition after redirect)
  const [tolerancia] = useState<ToleranciaData | null>(() => {
    if (serverTolerancia) return serverTolerancia;
    if (typeof window === 'undefined') return null;
    try {
      const raw = localStorage.getItem('kaudal_tolerancia_result');
      if (raw) {
        const parsed = JSON.parse(raw);
        return {
          perfil_resultado: parsed.perfil,
          volatilidad_maxima: parsed.volatilidad_maxima,
          puntaje_total: parsed.puntaje_total,
          descripcion_perfil: parsed.descripcion_perfil ?? null,
        };
      }
    } catch { /* ignore */ }
    return null;
  });

  const [suenos] = useState<SuenosData | null>(() => {
    if (serverSuenos) return serverSuenos;
    if (typeof window === 'undefined') return null;
    try {
      const raw = localStorage.getItem('kaudal_suenos_result');
      if (raw) {
        const parsed = JSON.parse(raw);
        return {
          retorno_minimo_requerido: parsed.retorno_minimo_requerido,
          nivel: parsed.nivel,
          anos_horizonte: parsed.horizonte_anos ?? 0,
          meta_tipo: '',
        };
      }
    } catch { /* ignore */ }
    return null;
  });

  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 80);
    return () => clearTimeout(t);
  }, []);

  const [saveError, setSaveError] = useState<string | null>(null);

  async function handleVerDashboard() {
    setSaving(true);
    setSaveError(null);
    try {
      const res = await fetch('/api/complete-onboarding', { method: 'POST' });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setSaveError(body.error || 'No pudimos guardar tu perfil. Intenta de nuevo.');
        setSaving(false);
        return;
      }

      localStorage.removeItem('kaudal_tolerancia_result');
      localStorage.removeItem('kaudal_suenos_result');
      router.push('/dashboard');
    } catch (err) {
      console.error('Error inesperado:', err);
      setSaveError('Error de conexión. Intenta de nuevo.');
      setSaving(false);
    }
  }

  if (!tolerancia) {
    return (
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '48px 16px' }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '2rem', marginBottom: '12px' }}>⏳</div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Cargando tus resultados...</p>
        </div>
      </div>
    );
  }

  const meta = getPerfilMeta(tolerancia.perfil_resultado);
  const volatPct = (tolerancia.volatilidad_maxima * 100).toFixed(0);
  const retornoPct = suenos ? (suenos.retorno_minimo_requerido * 100).toFixed(1) : null;
  const nivelInfo = suenos ? (NIVEL_LABELS[suenos.nivel] ?? { label: suenos.nivel, color: '#94A3B8' }) : null;
  const descripcion = tolerancia.descripcion_perfil ?? meta.mensaje;

  return (
    <div style={{
      flex: 1, display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
      padding: '32px 16px 56px',
    }}>
      <div style={{
        width: '100%', maxWidth: '520px',
        opacity: visible ? 1 : 0,
        transform: visible ? 'translateY(0)' : 'translateY(20px)',
        transition: 'opacity 0.55s ease, transform 0.55s ease',
      }}>
        {/* Success badge */}
        <div style={{ textAlign: 'center', marginBottom: '24px' }}>
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: '6px',
            padding: '5px 14px', borderRadius: '99px',
            background: 'rgba(20,184,166,0.1)', border: '1px solid rgba(20,184,166,0.3)',
            marginBottom: '14px',
          }}>
            <span style={{ color: '#14B8A6', fontSize: '0.75rem', fontWeight: 600 }}>
              ✓ Perfil calculado
            </span>
          </div>
          <h1 style={{
            fontFamily: 'var(--font-display)', fontSize: '1.6rem', fontWeight: 800,
            color: 'var(--text-main)', lineHeight: 1.2, marginBottom: '4px',
          }}>
            Tu Perfil Inversor
          </h1>
        </div>

        {/* Main profile card */}
        <div style={{
          background: 'var(--bg-panel)', border: `1px solid ${meta.color}30`,
          borderRadius: '20px', padding: '28px 24px', marginBottom: '16px',
          backdropFilter: 'blur(16px)',
          boxShadow: `0 8px 40px ${meta.color}18`,
          opacity: visible ? 1 : 0,
          transform: visible ? 'scale(1)' : 'scale(0.97)',
          transition: 'opacity 0.6s ease 0.1s, transform 0.6s ease 0.1s',
        }}>
          {/* Icon + Classification */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '16px' }}>
            <div style={{
              width: '60px', height: '60px', borderRadius: '16px',
              background: meta.bg, border: `1px solid ${meta.color}40`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '1.9rem', flexShrink: 0,
            }}>
              {meta.icon}
            </div>
            <div>
              <div style={{ fontSize: '0.73rem', color: 'var(--text-muted)', marginBottom: '3px', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                Clasificación
              </div>
              <div style={{
                fontFamily: 'var(--font-display)', fontSize: '1.5rem', fontWeight: 800,
                color: meta.color, lineHeight: 1,
                textTransform: 'capitalize',
              }}>
                {tolerancia.perfil_resultado}
              </div>
            </div>
          </div>

          {/* Description */}
          <p style={{
            fontSize: '0.86rem', color: 'var(--text-muted)', lineHeight: 1.7,
            marginBottom: '20px', paddingBottom: '20px',
            borderBottom: '1px solid var(--border-light)',
          }}>
            {descripcion}
          </p>

          {/* Key metrics */}
          <div style={{ display: 'grid', gridTemplateColumns: retornoPct ? '1fr 1fr' : '1fr', gap: '12px' }}>
            {/* Volatility */}
            <div style={{
              padding: '14px', borderRadius: '12px',
              background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-light)',
            }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '5px' }}>
                Volatilidad máxima
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.35rem', fontWeight: 700, color: meta.color }}>
                {volatPct}%
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '2px' }}>anual</div>
            </div>

            {/* Min return */}
            {retornoPct && (
              <div style={{
                padding: '14px', borderRadius: '12px',
                background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-light)',
              }}>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '5px' }}>
                  Retorno mínimo requerido
                </div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.35rem', fontWeight: 700, color: '#F59E0B' }}>
                  {retornoPct}%
                </div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '2px' }}>anual</div>
              </div>
            )}
          </div>
        </div>

        {/* Suenos summary */}
        {suenos && (
          <div style={{
            background: 'var(--bg-panel)', border: '1px solid var(--border-light)',
            borderRadius: '16px', padding: '18px 20px', marginBottom: '16px',
            backdropFilter: 'blur(16px)',
            opacity: visible ? 1 : 0,
            transform: visible ? 'translateY(0)' : 'translateY(10px)',
            transition: 'opacity 0.5s ease 0.2s, transform 0.5s ease 0.2s',
          }}>
            <div style={{ fontSize: '0.73rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '12px' }}>
              Tu meta financiera
            </div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span style={{ fontSize: '1.2rem' }}>
                  {suenos.meta_tipo === 'retiro' ? '🏖️' : suenos.meta_tipo === 'casa' ? '🏡' : suenos.meta_tipo === 'educacion' ? '🎓' : suenos.meta_tipo === 'libertad' ? '🦋' : suenos.meta_tipo === 'viaje' ? '✈️' : '🌟'}
                </span>
                <div>
                  <div style={{ fontSize: '0.87rem', fontWeight: 600, color: 'var(--text-main)' }}>
                    {META_LABELS[suenos.meta_tipo] ?? 'Meta personal'}
                  </div>
                  {suenos.anos_horizonte > 0 && (
                    <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                      Horizonte: {suenos.anos_horizonte} años
                    </div>
                  )}
                </div>
              </div>
              {nivelInfo && (
                <span style={{
                  padding: '4px 10px', borderRadius: '99px', fontSize: '0.72rem', fontWeight: 600,
                  background: `${nivelInfo.color}18`, color: nivelInfo.color,
                  border: `1px solid ${nivelInfo.color}30`,
                }}>
                  {nivelInfo.label}
                </span>
              )}
            </div>
          </div>
        )}

        {/* Motivational message */}
        <div style={{
          padding: '16px 18px', borderRadius: '14px', marginBottom: '24px',
          background: `${meta.color}0D`, border: `1px solid ${meta.color}22`,
          opacity: visible ? 1 : 0,
          transition: 'opacity 0.5s ease 0.3s',
        }}>
          <p style={{ fontSize: '0.86rem', color: 'var(--text-main)', lineHeight: 1.7, margin: 0 }}>
            <strong style={{ color: meta.color }}>Tu camino hacia la libertad financiera empieza hoy.</strong>{' '}
            {meta.mensaje}
          </p>
        </div>

        {/* Error */}
        {saveError && (
          <div style={{
            marginBottom: '16px', padding: '12px 14px',
            background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.4)',
            borderRadius: '10px', color: '#fca5a5', fontSize: '0.85rem',
          }}>
            {saveError}
          </div>
        )}

        {/* CTA */}
        <button
          onClick={handleVerDashboard}
          disabled={saving}
          style={{
            width: '100%', padding: '16px', borderRadius: '12px', border: 'none',
            cursor: saving ? 'wait' : 'pointer',
            fontWeight: 700, fontSize: '1rem', fontFamily: 'var(--font-display)',
            background: 'linear-gradient(135deg, #2563EB, #1D4ED8)', color: 'white',
            boxShadow: '0 4px 24px rgba(37,99,235,0.38)',
            transition: 'opacity 0.2s ease, transform 0.2s ease',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px',
            opacity: visible ? 1 : 0,
            transform: visible ? 'translateY(0)' : 'translateY(8px)',
          }}
          onMouseEnter={(e) => { if (!saving) { e.currentTarget.style.opacity = '0.88'; e.currentTarget.style.transform = 'translateY(-1px)'; } }}
          onMouseLeave={(e) => { e.currentTarget.style.opacity = '1'; e.currentTarget.style.transform = 'translateY(0)'; }}
        >
          {saving ? (
            <>
              <svg style={{ width: '16px', height: '16px', animation: 'spin 1s linear infinite' }} viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="rgba(255,255,255,0.3)" strokeWidth="3" />
                <path d="M12 2a10 10 0 0 1 10 10" stroke="white" strokeWidth="3" strokeLinecap="round" />
              </svg>
              Guardando...
            </>
          ) : (
            'Ver mi Dashboard →'
          )}
          <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
        </button>
      </div>
    </div>
  );
}
