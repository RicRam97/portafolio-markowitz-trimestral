'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { API_BASE } from '@/lib/constants';

// ---------------------------------------------------------------------------
// Types — aligned with TestToleranciaInput / TestToleranciaOutput (FastAPI)
// ---------------------------------------------------------------------------

type Phase = 1 | 2 | 3 | 'calculating' | 'results';

interface Answers {
  // Paso 1: situación financiera
  fondo_emergencia: boolean | null;
  necesita_dinero_2a: boolean | null;
  tiene_deudas: boolean | null;
  // Paso 2: experiencia inversora
  ha_invertido: boolean | null;
  entiende_acciones: boolean | null;
  conoce_volatilidad: boolean | null;
  // Paso 3: perfil emocional
  caida_15: 'vender' | 'esperar' | 'comprar' | null;
  certeza_vs_riesgo: 'certeza' | 'riesgo' | null;
  preocupacion: 'perder' | 'oportunidad' | null;
}

interface TestResult {
  perfil: string;
  volatilidad_maxima: number;   // 0.08, 0.12, … (fraction)
  puntaje_total: number;
  puntaje_por_dimension: {
    situacion_financiera: number;
    experiencia_inversora: number;
    perfil_emocional: number;
  };
  descripcion_perfil: string;
}

const EMPTY_ANSWERS: Answers = {
  fondo_emergencia: null,
  necesita_dinero_2a: null,
  tiene_deudas: null,
  ha_invertido: null,
  entiende_acciones: null,
  conoce_volatilidad: null,
  caida_15: null,
  certeza_vs_riesgo: null,
  preocupacion: null,
};

// ---------------------------------------------------------------------------
// Question data — values match Pydantic Literal fields exactly
// ---------------------------------------------------------------------------

const STEP1_QUESTIONS: { field: keyof Answers; text: string }[] = [
  { field: 'fondo_emergencia',  text: '¿Tienes un fondo de emergencia de al menos 6 meses de gastos?' },
  { field: 'necesita_dinero_2a', text: '¿Necesitarías disponer de este dinero en los próximos 2 años?' },
  { field: 'tiene_deudas',      text: '¿Tienes deudas de alto interés (tarjetas, créditos personales)?' },
];

const STEP2_QUESTIONS: { field: keyof Answers; text: string }[] = [
  { field: 'ha_invertido',       text: '¿Has invertido en acciones o ETFs alguna vez?' },
  { field: 'entiende_acciones',  text: '¿Entiendes cómo funciona la compra-venta de acciones?' },
  { field: 'conoce_volatilidad', text: '¿Sabes qué es la volatilidad y cómo afecta un portafolio?' },
];

const STEP3_QUESTIONS: {
  field: keyof Answers;
  text: string;
  options: { value: string; label: string }[];
}[] = [
  {
    field: 'caida_15',
    text: 'Si tu portafolio cae un 15% en un mes, ¿qué harías?',
    options: [
      { value: 'vender',  label: 'Vendería todo para evitar más pérdidas' },
      { value: 'esperar', label: 'Esperaría con calma a que se recupere' },
      { value: 'comprar', label: 'Aprovecharía para comprar más activos' },
    ],
  },
  {
    field: 'certeza_vs_riesgo',
    text: '¿Qué prefieres al momento de invertir?',
    options: [
      { value: 'certeza', label: 'Certeza: rendimiento predecible, aunque sea menor' },
      { value: 'riesgo',  label: 'Riesgo: mayor potencial de ganancia con incertidumbre' },
    ],
  },
  {
    field: 'preocupacion',
    text: '¿Qué te preocupa más en tus inversiones?',
    options: [
      { value: 'perder',      label: 'Perder el capital que invertí' },
      { value: 'oportunidad', label: 'Perder una oportunidad de crecimiento' },
    ],
  },
];

// ---------------------------------------------------------------------------
// Local score fallback (mirrors Python scoring exactly, max = 20)
// ---------------------------------------------------------------------------

function computeLocalScore(a: Answers): number {
  const bool2 = (v: boolean | null) => (v === true ? 2 : 0);
  const boolInv = (v: boolean | null) => (v === false ? 2 : 0);

  const dim1 =
    bool2(a.fondo_emergencia) +
    boolInv(a.necesita_dinero_2a) +
    boolInv(a.tiene_deudas);

  const dim2 =
    bool2(a.ha_invertido) +
    bool2(a.entiende_acciones) +
    bool2(a.conoce_volatilidad);

  const caida = a.caida_15 === 'comprar' ? 3 : a.caida_15 === 'esperar' ? 2 : 0;
  const certeza = a.certeza_vs_riesgo === 'riesgo' ? 3 : 0;
  const preoc = a.preocupacion === 'oportunidad' ? 2 : 0;
  const dim3 = caida + certeza + preoc;

  return dim1 + dim2 + dim3;
}

// ---------------------------------------------------------------------------
// Step completion guards
// ---------------------------------------------------------------------------

function step1Complete(a: Answers) {
  return a.fondo_emergencia !== null && a.necesita_dinero_2a !== null && a.tiene_deudas !== null;
}
function step2Complete(a: Answers) {
  return a.ha_invertido !== null && a.entiende_acciones !== null && a.conoce_volatilidad !== null;
}
function step3Complete(a: Answers) {
  return a.caida_15 !== null && a.certeza_vs_riesgo !== null && a.preocupacion !== null;
}

// ---------------------------------------------------------------------------
// Profile display metadata (all 5 profiles from backend)
// ---------------------------------------------------------------------------

const PERFIL_META: Record<string, { color: string; bg: string }> = {
  conservador: { color: '#14B8A6', bg: 'rgba(20,184,166,0.12)' },
  moderado:    { color: '#2563EB', bg: 'rgba(37,99,235,0.12)'  },
  balanceado:  { color: '#8B5CF6', bg: 'rgba(139,92,246,0.12)' },
  crecimiento: { color: '#F59E0B', bg: 'rgba(245,158,11,0.12)' },
  agresivo:    { color: '#EF4444', bg: 'rgba(239,68,68,0.12)'  },
};

function getPerfilMeta(perfil: string) {
  return PERFIL_META[perfil.toLowerCase()] ?? PERFIL_META['moderado'];
}

// ---------------------------------------------------------------------------
// Gauge SVG (max score = 20, 5 segments)
// ---------------------------------------------------------------------------

function GaugeChart({ score, maxScore = 20 }: { score: number; maxScore?: number }) {
  const cx = 110;
  const cy = 110;
  const r = 90;
  const needleLen = 78;

  // score 0 → angle π (left), maxScore → angle 0 (right)
  const angle = Math.PI * (1 - Math.min(score, maxScore) / maxScore);
  const nx = cx + needleLen * Math.cos(angle);
  const ny = cy - needleLen * Math.sin(angle);

  // 5 equal segments over 180°: each 36°
  const segColors = ['#14B8A6', '#2563EB', '#8B5CF6', '#F59E0B', '#EF4444'];
  const segLabels = ['Conservador', 'Moderado', 'Balanceado', 'Crecimiento', 'Agresivo'];
  const segPaths: string[] = [];
  for (let i = 0; i < 5; i++) {
    const a1 = Math.PI - (i / 5) * Math.PI;
    const a2 = Math.PI - ((i + 1) / 5) * Math.PI;
    const x1 = cx + r * Math.cos(a1);
    const y1 = cy - r * Math.sin(a1);
    const x2 = cx + r * Math.cos(a2);
    const y2 = cy - r * Math.sin(a2);
    segPaths.push(`M ${x1.toFixed(1)},${y1.toFixed(1)} A ${r},${r} 0 0,1 ${x2.toFixed(1)},${y2.toFixed(1)}`);
  }

  // Label for center-bottom
  const midAngle = Math.PI - ((Math.floor(score / (maxScore / 5)) + 0.5) / 5) * Math.PI;
  const labelIdx = Math.min(Math.floor(score / (maxScore / 5)), 4);

  return (
    <svg viewBox="0 0 220 130" className="w-full max-w-sm mx-auto">
      {/* Track */}
      <path
        d={`M ${cx - r},${cy} A ${r},${r} 0 0,1 ${cx + r},${cy}`}
        fill="none"
        stroke="rgba(255,255,255,0.05)"
        strokeWidth="16"
        strokeLinecap="round"
      />
      {/* Colored segments */}
      {segPaths.map((d, i) => (
        <path
          key={i}
          d={d}
          fill="none"
          stroke={segColors[i]}
          strokeWidth="12"
          strokeLinecap="butt"
          opacity="0.7"
        />
      ))}
      {/* Needle shadow */}
      <line x1={cx} y1={cy} x2={nx} y2={ny} stroke="rgba(0,0,0,0.3)" strokeWidth="5" strokeLinecap="round" />
      {/* Needle */}
      <line x1={cx} y1={cy} x2={nx} y2={ny} stroke="white" strokeWidth="3" strokeLinecap="round" />
      <circle cx={cx} cy={cy} r="7" fill="white" />
      <circle cx={cx} cy={cy} r="3.5" fill="rgba(0,0,0,0.4)" />
      <circle cx={nx} cy={ny} r="4.5" fill="white" />
      {/* End labels */}
      <text x="14"  y="125" fontSize="8" fill="#94A3B8" textAnchor="middle">{segLabels[0]}</text>
      <text x="206" y="125" fontSize="8" fill="#94A3B8" textAnchor="middle">{segLabels[4]}</text>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Step tab metadata
// ---------------------------------------------------------------------------

const STEPS = [
  { num: 1, label: 'Situación financiera', icon: '💰' },
  { num: 2, label: 'Experiencia inversora', icon: '📊' },
  { num: 3, label: 'Perfil emocional',      icon: '🧠' },
];

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function TestToleranciaRiesgo({ redirectTo = '/dashboard', userId }: { redirectTo?: string; userId?: string }) {
  const router = useRouter();
  const [phase, setPhase] = useState<Phase>(1);
  const [visible, setVisible] = useState(true);
  const [answers, setAnswers] = useState<Answers>(EMPTY_ANSWERS);
  const [result, setResult] = useState<TestResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const canAdvance =
    (phase === 1 && step1Complete(answers)) ||
    (phase === 2 && step2Complete(answers)) ||
    (phase === 3 && step3Complete(answers));

  function setField<K extends keyof Answers>(field: K, value: Answers[K]) {
    setAnswers((prev) => ({ ...prev, [field]: value }));
  }

  function transitionTo(next: Phase) {
    setVisible(false);
    setTimeout(() => {
      setPhase(next);
      setVisible(true);
    }, 280);
  }

  function handleNext() {
    if (phase === 1) transitionTo(2);
    else if (phase === 2) transitionTo(3);
    else if (phase === 3) transitionTo('calculating');
  }

  useEffect(() => {
    if (phase === 'calculating') submitTest();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase]);

  async function submitTest() {
    setError(null);
    const startTime = Date.now();
    try {
      // Payload matches TestToleranciaInput exactly
      const payload = {
        fondo_emergencia:  answers.fondo_emergencia,
        necesita_dinero_2a: answers.necesita_dinero_2a,
        tiene_deudas:      answers.tiene_deudas,
        ha_invertido:      answers.ha_invertido,
        entiende_acciones: answers.entiende_acciones,
        conoce_volatilidad: answers.conoce_volatilidad,
        caida_15:          answers.caida_15,
        certeza_vs_riesgo: answers.certeza_vs_riesgo,
        preocupacion:      answers.preocupacion,
        user_id:           userId ?? null,
      };

      const res = await fetch(`${API_BASE}/api/ml/test-tolerancia/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail?.detail ?? `HTTP ${res.status}`);
      }

      const data: TestResult = await res.json();

      // Ensure "calculating" screen is visible at least 2 s
      const elapsed = Date.now() - startTime;
      if (elapsed < 2000) await new Promise((r) => setTimeout(r, 2000 - elapsed));

      setResult(data);
      if (redirectTo.includes('onboarding')) {
        localStorage.setItem('kaudal_tolerancia_result', JSON.stringify(data));
      }
      transitionTo('results');
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Error desconocido';
      setError(`No pudimos calcular tu perfil: ${msg}. Intenta de nuevo.`);
      setPhase(3);
      setVisible(true);
    }
  }

  // Derived display values
  const perfilMeta = result ? getPerfilMeta(result.perfil) : PERFIL_META['moderado'];
  const gaugeScore = result?.puntaje_total ?? computeLocalScore(answers);
  const volatPct = result ? (result.volatilidad_maxima * 100).toFixed(0) : '—';

  const wrapStyle: React.CSSProperties = {
    opacity: visible ? 1 : 0,
    transform: visible ? 'translateY(0)' : 'translateY(12px)',
    transition: 'opacity 0.28s ease, transform 0.28s ease',
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '24px 16px' }}>
      <div
        style={{
          width: '100%',
          maxWidth: '540px',
          background: 'var(--bg-panel)',
          border: '1px solid var(--border-light)',
          borderRadius: '20px',
          padding: '32px 28px',
          backdropFilter: 'blur(16px)',
        }}
      >

        {/* ── Steps 1 / 2 / 3 ──────────────────────────────── */}
        {(phase === 1 || phase === 2 || phase === 3) && (
          <div style={wrapStyle}>

            {/* Progress bar */}
            <div style={{ marginBottom: '28px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Paso {phase} de 3</span>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>{Math.round(((phase as number) / 3) * 100)}%</span>
              </div>
              <div style={{ height: '5px', background: 'rgba(255,255,255,0.08)', borderRadius: '99px', overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width: `${((phase as number) / 3) * 100}%`,
                  background: 'linear-gradient(90deg, #2563EB, #14B8A6)',
                  borderRadius: '99px',
                  transition: 'width 0.4s ease',
                }} />
              </div>
            </div>

            {/* Step tabs */}
            <div style={{ display: 'flex', gap: '8px', marginBottom: '28px' }}>
              {STEPS.map((s) => (
                <div key={s.num} style={{
                  flex: 1, textAlign: 'center', padding: '8px 4px', borderRadius: '10px',
                  background: phase === s.num ? 'rgba(37,99,235,0.18)' : (phase as number) > s.num ? 'rgba(20,184,166,0.1)' : 'rgba(255,255,255,0.04)',
                  border: phase === s.num ? '1px solid rgba(37,99,235,0.5)' : '1px solid transparent',
                  transition: 'all 0.3s ease',
                }}>
                  <div style={{ fontSize: '1.1rem', marginBottom: '2px' }}>{s.icon}</div>
                  <div style={{
                    fontSize: '0.65rem', lineHeight: 1.3, fontWeight: phase === s.num ? 600 : 400,
                    color: phase === s.num ? 'var(--text-main)' : (phase as number) > s.num ? '#14B8A6' : 'var(--text-muted)',
                  }}>
                    {s.label}
                  </div>
                </div>
              ))}
            </div>

            {/* Step 1 */}
            {phase === 1 && (
              <div>
                <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.3rem', fontWeight: 700, marginBottom: '6px' }}>
                  💰 Tu situación financiera
                </h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '24px' }}>
                  Cuéntanos sobre tu base financiera actual.
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                  {STEP1_QUESTIONS.map((q, i) => (
                    <QuestionSiNo
                      key={q.field}
                      index={i + 1}
                      text={q.text}
                      value={answers[q.field] as boolean | null}
                      onChange={(v) => setField(q.field, v)}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Step 2 */}
            {phase === 2 && (
              <div>
                <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.3rem', fontWeight: 700, marginBottom: '6px' }}>
                  📊 Tu experiencia inversora
                </h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '24px' }}>
                  Hablemos de tu trayectoria con inversiones.
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                  {STEP2_QUESTIONS.map((q, i) => (
                    <QuestionSiNo
                      key={q.field}
                      index={i + 1}
                      text={q.text}
                      value={answers[q.field] as boolean | null}
                      onChange={(v) => setField(q.field, v)}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Step 3 */}
            {phase === 3 && (
              <div>
                <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.3rem', fontWeight: 700, marginBottom: '6px' }}>
                  🧠 Tu perfil emocional
                </h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '24px' }}>
                  Cómo reaccionas define tu verdadero perfil.
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                  {STEP3_QUESTIONS.map((q, i) => (
                    <QuestionMulti
                      key={q.field}
                      index={i + 1}
                      text={q.text}
                      options={q.options}
                      value={answers[q.field] as string | null}
                      onChange={(v) => setField(q.field, v as Answers[typeof q.field])}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Error banner */}
            {error && (
              <div style={{ marginTop: '16px', padding: '12px', background: 'rgba(239,68,68,0.1)', border: '1px solid var(--danger)', borderRadius: '10px', color: '#fca5a5', fontSize: '0.85rem' }}>
                {error}
              </div>
            )}

            {/* Next button */}
            <button
              onClick={handleNext}
              disabled={!canAdvance}
              style={{
                marginTop: '28px', width: '100%', padding: '14px', borderRadius: '12px', border: 'none',
                cursor: canAdvance ? 'pointer' : 'not-allowed',
                fontWeight: 600, fontSize: '0.95rem', fontFamily: 'var(--font-display)',
                background: canAdvance ? 'linear-gradient(135deg, #2563EB, #1D4ED8)' : 'rgba(255,255,255,0.06)',
                color: canAdvance ? 'white' : 'var(--text-muted)',
                transition: 'all 0.2s ease',
                boxShadow: canAdvance ? '0 4px 20px rgba(37,99,235,0.35)' : 'none',
              }}
            >
              {phase === 3 ? 'Ver mi perfil' : 'Siguiente'}
            </button>
          </div>
        )}

        {/* ── Calculating ───────────────────────────────────── */}
        {phase === 'calculating' && (
          <div style={{ ...wrapStyle, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '320px', textAlign: 'center', gap: '20px' }}>
            <svg viewBox="0 0 72 72" style={{ width: '72px', height: '72px', animation: 'spin 1.2s linear infinite' }}>
              <circle cx="36" cy="36" r="30" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="5" />
              <circle cx="36" cy="36" r="30" fill="none" stroke="url(#spinGrad)" strokeWidth="5" strokeLinecap="round" strokeDasharray="120 188" />
              <defs>
                <linearGradient id="spinGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#2563EB" />
                  <stop offset="100%" stopColor="#14B8A6" />
                </linearGradient>
              </defs>
            </svg>
            <div>
              <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.25rem', fontWeight: 700, marginBottom: '8px' }}>
                Calculando tu perfil...
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>
                Analizando tus respuestas con nuestro modelo
              </p>
            </div>
            <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
          </div>
        )}

        {/* ── Results ───────────────────────────────────────── */}
        {phase === 'results' && result && (
          <div style={wrapStyle}>
            {/* Header */}
            <div style={{ textAlign: 'center', marginBottom: '20px' }}>
              <div style={{ display: 'inline-block', padding: '4px 16px', borderRadius: '99px', background: perfilMeta.bg, border: `1px solid ${perfilMeta.color}40`, marginBottom: '10px' }}>
                <span style={{ color: perfilMeta.color, fontSize: '0.76rem', fontWeight: 600 }}>Tu perfil de inversionista</span>
              </div>
              <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.55rem', fontWeight: 800, color: perfilMeta.color, marginBottom: '8px' }}>
                {result.perfil}
              </h2>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.84rem', lineHeight: 1.6 }}>
                {result.descripcion_perfil}
              </p>
            </div>

            {/* Gauge */}
            <div style={{ marginBottom: '16px' }}>
              <GaugeChart score={gaugeScore} maxScore={20} />
            </div>

            {/* Volatility card */}
            <div style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border-light)', borderRadius: '14px', padding: '16px 20px', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '16px' }}>
              <div style={{ width: '44px', height: '44px', borderRadius: '12px', background: perfilMeta.bg, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.3rem', flexShrink: 0 }}>
                📉
              </div>
              <div>
                <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginBottom: '3px' }}>
                  Volatilidad máxima recomendada
                </div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.15rem', fontWeight: 700, color: perfilMeta.color }}>
                  {volatPct}% anual
                </div>
                <div style={{ fontSize: '0.77rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                  Tu portafolio no debería superar este nivel de riesgo.
                </div>
              </div>
            </div>

            {/* Score breakdown */}
            <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-light)', borderRadius: '14px', padding: '16px 20px', marginBottom: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Puntaje total</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '0.95rem', padding: '2px 12px', borderRadius: '99px', background: perfilMeta.bg, color: perfilMeta.color, border: `1px solid ${perfilMeta.color}30` }}>
                  {result.puntaje_total} / 20
                </span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {([
                  ['💰 Situación financiera', result.puntaje_por_dimension.situacion_financiera, 6],
                  ['📊 Experiencia inversora', result.puntaje_por_dimension.experiencia_inversora, 6],
                  ['🧠 Perfil emocional',      result.puntaje_por_dimension.perfil_emocional,    8],
                ] as [string, number, number][]).map(([label, score, max]) => (
                  <div key={label}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.76rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
                      <span>{label}</span>
                      <span style={{ fontFamily: 'var(--font-mono)' }}>{score}/{max}</span>
                    </div>
                    <div style={{ height: '4px', background: 'rgba(255,255,255,0.07)', borderRadius: '99px', overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${(score / max) * 100}%`, background: perfilMeta.color, borderRadius: '99px', opacity: 0.8 }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* CTA */}
            <button
              onClick={() => router.push(redirectTo)}
              style={{
                width: '100%', padding: '14px', borderRadius: '12px', border: 'none', cursor: 'pointer',
                fontWeight: 600, fontSize: '0.95rem', fontFamily: 'var(--font-display)',
                background: 'linear-gradient(135deg, #2563EB, #1D4ED8)', color: 'white',
                boxShadow: '0 4px 20px rgba(37,99,235,0.35)', transition: 'opacity 0.2s ease',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.opacity = '0.88')}
              onMouseLeave={(e) => (e.currentTarget.style.opacity = '1')}
            >
              {redirectTo.includes('onboarding') ? 'Ver mis Resultados →' : 'Continuar al Dashboard'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface QuestionSiNoProps {
  index: number;
  text: string;
  value: boolean | null;
  onChange: (v: boolean) => void;
}

function QuestionSiNo({ index, text, value, onChange }: QuestionSiNoProps) {
  return (
    <div>
      <p style={{ fontSize: '0.88rem', color: 'var(--text-main)', marginBottom: '10px', lineHeight: 1.5, display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
        <span style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', minWidth: '20px', height: '20px', borderRadius: '50%', background: 'rgba(37,99,235,0.2)', color: '#60A5FA', fontSize: '0.7rem', fontWeight: 700, marginTop: '1px' }}>
          {index}
        </span>
        {text}
      </p>
      <div style={{ display: 'flex', gap: '10px' }}>
        {([{ v: true, label: 'Sí' }, { v: false, label: 'No' }] as const).map(({ v, label }) => {
          const selected = value === v;
          const isYes = v === true;
          return (
            <button
              key={label}
              onClick={() => onChange(v)}
              style={{
                flex: 1, padding: '10px 16px', borderRadius: '10px', cursor: 'pointer',
                border: selected ? `1.5px solid ${isYes ? '#14B8A6' : '#EF4444'}` : '1.5px solid rgba(255,255,255,0.1)',
                background: selected ? (isYes ? 'rgba(20,184,166,0.15)' : 'rgba(239,68,68,0.12)') : 'rgba(255,255,255,0.04)',
                color: selected ? (isYes ? '#14B8A6' : '#F87171') : 'var(--text-muted)',
                fontWeight: selected ? 600 : 400, fontSize: '0.9rem',
                transition: 'all 0.18s ease',
              }}
            >
              {isYes ? '✓ ' : '✗ '}{label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

interface QuestionMultiProps {
  index: number;
  text: string;
  options: { value: string; label: string }[];
  value: string | null;
  onChange: (v: string) => void;
}

function QuestionMulti({ index, text, options, value, onChange }: QuestionMultiProps) {
  const colors = ['#14B8A6', '#2563EB', '#F59E0B'];
  return (
    <div>
      <p style={{ fontSize: '0.88rem', color: 'var(--text-main)', marginBottom: '10px', lineHeight: 1.5, display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
        <span style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', minWidth: '20px', height: '20px', borderRadius: '50%', background: 'rgba(37,99,235,0.2)', color: '#60A5FA', fontSize: '0.7rem', fontWeight: 700, marginTop: '1px' }}>
          {index}
        </span>
        {text}
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {options.map((opt, i) => {
          const selected = value === opt.value;
          const c = colors[i % colors.length];
          return (
            <button
              key={opt.value}
              onClick={() => onChange(opt.value)}
              style={{
                width: '100%', padding: '12px 16px', borderRadius: '10px', cursor: 'pointer',
                border: selected ? `1.5px solid ${c}` : '1.5px solid rgba(255,255,255,0.08)',
                background: selected ? `${c}18` : 'rgba(255,255,255,0.03)',
                color: selected ? 'var(--text-main)' : 'var(--text-muted)',
                fontWeight: selected ? 500 : 400, fontSize: '0.86rem', textAlign: 'left',
                display: 'flex', alignItems: 'center', gap: '10px',
                transition: 'all 0.18s ease',
              }}
            >
              <span style={{ width: '16px', height: '16px', borderRadius: '50%', flexShrink: 0, border: selected ? `2px solid ${c}` : '2px solid rgba(255,255,255,0.2)', background: selected ? c : 'transparent', transition: 'all 0.18s ease' }} />
              {opt.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
