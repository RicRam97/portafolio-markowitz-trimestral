'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { API_BASE } from '@/lib/constants';
import { useOnboardingUserId } from '@/components/onboarding/OnboardingContext';
import EducationalModal from '@/components/onboarding/EducationalModal';

type MetaTipo = 'casa' | 'retiro' | 'educacion' | 'viaje' | 'libertad' | 'otra';
type Moneda = 'MXN' | 'USD';

interface FormValues {
  meta_tipo: MetaTipo | '';
  meta_dinero: string;
  capital_inicial: string;
  ahorro_mensual: string;
  anos_horizonte: string;
  moneda: Moneda;
}

const META_OPTIONS: { value: MetaTipo; label: string; icon: string }[] = [
  { value: 'retiro', label: 'Retiro', icon: '🏖️' },
  { value: 'casa', label: 'Casa propia', icon: '🏡' },
  { value: 'educacion', label: 'Educación', icon: '🎓' },
  { value: 'libertad', label: 'Libertad financiera', icon: '🦋' },
  { value: 'viaje', label: 'Viaje', icon: '✈️' },
  { value: 'otra', label: 'Otra meta', icon: '🌟' },
];

const CURRENCY_SYMBOL: Record<Moneda, string> = { MXN: '$', USD: 'US$' };

function isFormComplete(f: FormValues) {
  return (
    f.meta_tipo !== '' &&
    parseFloat(f.meta_dinero) > 0 &&
    parseFloat(f.capital_inicial) >= 0 &&
    parseFloat(f.ahorro_mensual) >= 0 &&
    parseInt(f.anos_horizonte) >= 1
  );
}

export default function TestSuenosPage() {
  const router = useRouter();
  const userId = useOnboardingUserId();
  const [visible, setVisible] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showModal, setShowModal] = useState(false);
  const [resultData, setResultData] = useState<{ retorno_minimo_pct?: number } | null>(null);
  const [form, setForm] = useState<FormValues>({
    meta_tipo: '',
    meta_dinero: '',
    capital_inicial: '',
    ahorro_mensual: '',
    anos_horizonte: '',
    moneda: 'MXN',
  });

  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 60);
    return () => clearTimeout(t);
  }, []);

  function setField<K extends keyof FormValues>(key: K, value: FormValues[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  async function handleSubmit() {
    if (!isFormComplete(form) || loading) return;
    setError(null);
    setLoading(true);

    try {
      const payload = {
        meta_tipo: form.meta_tipo,
        meta_dinero: parseFloat(form.meta_dinero),
        capital_inicial: parseFloat(form.capital_inicial) || 0,
        ahorro_mensual: parseFloat(form.ahorro_mensual) || 0,
        anos_horizonte: parseInt(form.anos_horizonte),
        moneda: form.moneda,
        user_id: userId,
      };

      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 15000);

      const res = await fetch(`${API_BASE}/api/ml/test-suenos/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      clearTimeout(timeout);

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail?.detail ?? `HTTP ${res.status}`);
      }

      const data = await res.json();
      // Cache result so resultados page can show it immediately
      localStorage.setItem('kaudal_suenos_result', JSON.stringify(data));

      setResultData(data);
      setShowModal(true);
    } catch (err) {
      const msg =
        err instanceof DOMException && err.name === 'AbortError'
          ? 'El servidor no respondió a tiempo. Verifica que el backend esté activo e intenta de nuevo.'
          : err instanceof TypeError
            ? 'No se pudo conectar al servidor. Verifica tu conexión o que el backend esté corriendo.'
            : err instanceof Error
              ? err.message
              : 'Error desconocido. Intenta de nuevo.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  const sym = CURRENCY_SYMBOL[form.moneda];
  const canSubmit = isFormComplete(form) && !loading;

  return (
    <div style={{
      flex: 1, display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
      padding: '32px 16px 48px',
    }}>
      <div style={{
        width: '100%', maxWidth: '520px',
        opacity: visible ? 1 : 0,
        transform: visible ? 'translateY(0)' : 'translateY(16px)',
        transition: 'opacity 0.45s ease, transform 0.45s ease',
      }}>
        {/* Header */}
        <div style={{ marginBottom: '28px' }}>
          <h1 style={{
            fontFamily: 'var(--font-display)', fontSize: '1.5rem', fontWeight: 800,
            color: 'var(--text-main)', marginBottom: '8px',
          }}>
            🎯 Define tus metas
          </h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', lineHeight: 1.65 }}>
            Cuéntanos qué quieres lograr con tu inversión y calcularemos el retorno mínimo que necesitas.
          </p>
        </div>

        {/* Meta tipo */}
        <div style={{ marginBottom: '24px' }}>
          <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '10px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            ¿Para qué quieres invertir?
          </label>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px' }}>
            {META_OPTIONS.map((opt) => {
              const selected = form.meta_tipo === opt.value;
              return (
                <button
                  key={opt.value}
                  onClick={() => setField('meta_tipo', opt.value)}
                  style={{
                    padding: '12px 8px', borderRadius: '12px', border: 'none', cursor: 'pointer',
                    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '5px',
                    background: selected ? 'rgba(37,99,235,0.18)' : 'rgba(255,255,255,0.04)',
                    outline: selected ? '1.5px solid rgba(37,99,235,0.6)' : '1.5px solid rgba(255,255,255,0.08)',
                    transition: 'all 0.18s ease',
                  }}
                >
                  <span style={{ fontSize: '1.4rem' }}>{opt.icon}</span>
                  <span style={{
                    fontSize: '0.7rem', fontWeight: selected ? 600 : 400, lineHeight: 1.2,
                    color: selected ? 'var(--text-main)' : 'var(--text-muted)',
                  }}>
                    {opt.label}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Moneda toggle */}
        <div style={{ marginBottom: '24px' }}>
          <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '10px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Moneda
          </label>
          <div style={{ display: 'flex', gap: '8px' }}>
            {(['MXN', 'USD'] as Moneda[]).map((cur) => (
              <button
                key={cur}
                onClick={() => setField('moneda', cur)}
                style={{
                  flex: 1, padding: '10px', borderRadius: '10px', border: 'none', cursor: 'pointer',
                  fontWeight: 600, fontSize: '0.88rem',
                  background: form.moneda === cur ? 'rgba(37,99,235,0.18)' : 'rgba(255,255,255,0.04)',
                  outline: form.moneda === cur ? '1.5px solid rgba(37,99,235,0.6)' : '1.5px solid rgba(255,255,255,0.08)',
                  color: form.moneda === cur ? 'var(--text-main)' : 'var(--text-muted)',
                  transition: 'all 0.18s ease',
                }}
              >
                {cur === 'MXN' ? '🇲🇽 MXN' : '🇺🇸 USD'}
              </button>
            ))}
          </div>
        </div>

        {/* Numeric fields */}
        {([
          {
            key: 'meta_dinero' as keyof FormValues,
            label: '¿Cuánto dinero necesitas?',
            placeholder: `${sym} 500,000`,
            prefix: sym,
            min: 1,
          },
          {
            key: 'capital_inicial' as keyof FormValues,
            label: '¿Con cuánto empiezas hoy?',
            placeholder: `${sym} 50,000`,
            prefix: sym,
            min: 0,
          },
          {
            key: 'ahorro_mensual' as keyof FormValues,
            label: '¿Cuánto puedes ahorrar al mes?',
            placeholder: `${sym} 5,000`,
            prefix: sym,
            min: 0,
          },
          {
            key: 'anos_horizonte' as keyof FormValues,
            label: '¿En cuántos años quieres lograrlo?',
            placeholder: '10',
            prefix: null,
            suffix: 'años',
            min: 1,
          },
        ] as { key: keyof FormValues; label: string; placeholder: string; prefix: string | null; suffix?: string; min: number }[]).map((field) => (
          <div key={field.key} style={{ marginBottom: '20px' }}>
            <label style={{
              display: 'block', fontSize: '0.82rem', fontWeight: 600,
              color: 'var(--text-muted)', marginBottom: '8px',
              textTransform: 'uppercase', letterSpacing: '0.05em',
            }}>
              {field.label}
            </label>
            <div style={{ position: 'relative' }}>
              {field.prefix && (
                <span style={{
                  position: 'absolute', left: '14px', top: '50%', transform: 'translateY(-50%)',
                  color: 'var(--text-muted)', fontSize: '0.9rem', fontWeight: 600,
                  pointerEvents: 'none',
                }}>
                  {field.prefix}
                </span>
              )}
              <input
                type="number"
                min={field.min}
                placeholder={field.placeholder}
                value={form[field.key] as string}
                onChange={(e) => setField(field.key, e.target.value as FormValues[typeof field.key])}
                style={{
                  width: '100%', padding: `12px ${field.suffix ? '54px' : '14px'} 12px ${field.prefix ? '40px' : '14px'}`,
                  borderRadius: '10px', border: '1.5px solid rgba(255,255,255,0.1)',
                  background: 'rgba(255,255,255,0.04)', color: 'var(--text-main)',
                  fontSize: '0.93rem', fontFamily: 'var(--font-mono)',
                  outline: 'none', boxSizing: 'border-box',
                  transition: 'border-color 0.18s ease',
                }}
                onFocus={(e) => (e.target.style.borderColor = 'rgba(37,99,235,0.6)')}
                onBlur={(e) => (e.target.style.borderColor = 'rgba(255,255,255,0.1)')}
              />
              {field.suffix && (
                <span style={{
                  position: 'absolute', right: '14px', top: '50%', transform: 'translateY(-50%)',
                  color: 'var(--text-muted)', fontSize: '0.82rem', pointerEvents: 'none',
                }}>
                  {field.suffix}
                </span>
              )}
            </div>
          </div>
        ))}

        {/* Error */}
        {error && (
          <div style={{
            marginBottom: '16px', padding: '12px 14px',
            background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.4)',
            borderRadius: '10px', color: '#fca5a5', fontSize: '0.85rem',
          }}>
            {error}
          </div>
        )}

        {/* Submit */}
        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          style={{
            width: '100%', padding: '15px', borderRadius: '12px', border: 'none',
            cursor: canSubmit ? 'pointer' : 'not-allowed',
            fontWeight: 700, fontSize: '0.95rem', fontFamily: 'var(--font-display)',
            background: canSubmit ? 'linear-gradient(135deg, #2563EB, #1D4ED8)' : 'rgba(255,255,255,0.06)',
            color: canSubmit ? 'white' : 'var(--text-muted)',
            boxShadow: canSubmit ? '0 4px 20px rgba(37,99,235,0.35)' : 'none',
            transition: 'all 0.2s ease',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px',
          }}
        >
          {loading ? (
            <>
              <svg style={{ width: '16px', height: '16px', animation: 'spin 1s linear infinite' }} viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="rgba(255,255,255,0.3)" strokeWidth="3" />
                <path d="M12 2a10 10 0 0 1 10 10" stroke="white" strokeWidth="3" strokeLinecap="round" />
              </svg>
              Calculando...
            </>
          ) : (
            'Continuar → Tu Perfil de Riesgo'
          )}
          <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
        </button>
      </div>

      <EducationalModal
        open={showModal}
        titulo="Tu meta financiera esta definida"
        descripcion={
          resultData?.retorno_minimo_pct
            ? `Para alcanzar tu objetivo necesitas un retorno minimo anual de ${resultData.retorno_minimo_pct.toFixed(1)}%. Este dato nos ayuda a calibrar el nivel de riesgo adecuado para ti.`
            : 'Hemos registrado tu meta financiera y calculado el retorno minimo que necesitas para alcanzarla en el plazo que definiste.'
        }
        conexion="Por eso ahora evaluaremos tu tolerancia al riesgo, para asegurarnos de que tu portafolio sea compatible con tu perfil emocional."
        botonTexto="Continuar al Test de Riesgo"
        onContinue={() => {
          setShowModal(false);
          router.push('/onboarding/test-tolerancia');
        }}
      />
    </div>
  );
}
