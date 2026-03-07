'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';

const DEMO_COMPANIES = [
  { ticker: 'AAPL', name: 'Apple Inc.', sector: 'Tecnologia', weight: 0.18, color: '#3b82f6' },
  { ticker: 'MSFT', name: 'Microsoft Corp.', sector: 'Tecnologia', weight: 0.16, color: '#10b981' },
  { ticker: 'NVDA', name: 'NVIDIA Corp.', sector: 'Semiconductores', weight: 0.14, color: '#f59e0b' },
  { ticker: 'GOOGL', name: 'Alphabet Inc.', sector: 'Tecnologia', weight: 0.12, color: '#ef4444' },
  { ticker: 'AMZN', name: 'Amazon.com Inc.', sector: 'E-commerce', weight: 0.10, color: '#8b5cf6' },
  { ticker: 'META', name: 'Meta Platforms', sector: 'Redes Sociales', weight: 0.08, color: '#ec4899' },
  { ticker: 'TSLA', name: 'Tesla Inc.', sector: 'Automotriz', weight: 0.07, color: '#06b6d4' },
  { ticker: 'JNJ', name: 'Johnson & Johnson', sector: 'Salud', weight: 0.06, color: '#84cc16' },
  { ticker: 'V', name: 'Visa Inc.', sector: 'Finanzas', weight: 0.05, color: '#f97316' },
  { ticker: 'WMT', name: 'Walmart Inc.', sector: 'Retail', weight: 0.04, color: '#6366f1' },
];

const MOCK_RESULT = {
  expectedReturn: 14.5,
  volatility: 11.2,
  sharpe: 1.29,
};

const LOADING_MESSAGES = [
  'Analizando datos historicos...',
  'Calculando matriz de covarianza...',
  'Optimizando pesos con Markowitz...',
  'Generando portafolio optimo...',
];

const pieData = DEMO_COMPANIES.map((c) => ({
  name: c.ticker,
  fullName: c.name,
  value: +(c.weight * 100).toFixed(1),
  color: c.color,
}));

export default function DemoOptimizerLight() {
  const [phase, setPhase] = useState<'idle' | 'loading' | 'done'>('idle');
  const [loadingMsg, setLoadingMsg] = useState(0);

  useEffect(() => {
    if (phase !== 'loading') return;
    let idx = 0;
    const interval = setInterval(() => {
      idx++;
      if (idx >= LOADING_MESSAGES.length) {
        clearInterval(interval);
        setPhase('done');
        return;
      }
      setLoadingMsg(idx);
    }, 500);
    return () => clearInterval(interval);
  }, [phase]);

  const handleSimulate = () => {
    setLoadingMsg(0);
    setPhase('loading');
  };

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 md:py-12">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8">
        {/* LEFT: Input Panel */}
        <motion.div
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          className="glass-panel"
          style={{ padding: '28px', borderTop: '3px solid var(--accent-primary)' }}
        >
          <h2
            style={{
              fontFamily: 'var(--font-display)',
              fontSize: '1.4rem',
              color: 'var(--text-main)',
              marginBottom: '4px',
            }}
          >
            Portafolio de Prueba
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '20px' }}>
            10 empresas seleccionadas para la simulacion
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '24px' }}>
            {DEMO_COMPANIES.map((c, i) => (
              <motion.div
                key={c.ticker}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05, duration: 0.3 }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  padding: '10px 14px',
                  borderRadius: '10px',
                  background: 'var(--overlay-soft)',
                  border: '1px solid var(--border-light)',
                }}
              >
                <div
                  style={{
                    width: '10px',
                    height: '10px',
                    borderRadius: '50%',
                    backgroundColor: c.color,
                    flexShrink: 0,
                  }}
                />
                <span
                  style={{
                    fontWeight: 700,
                    color: 'var(--text-main)',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.85rem',
                    minWidth: '50px',
                  }}
                >
                  {c.ticker}
                </span>
                <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem', flex: 1 }}>
                  {c.name}
                </span>
                <span
                  style={{
                    fontSize: '0.75rem',
                    color: 'var(--text-muted)',
                    background: 'var(--overlay-hover)',
                    padding: '2px 8px',
                    borderRadius: '6px',
                  }}
                >
                  {c.sector}
                </span>
              </motion.div>
            ))}
          </div>

          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleSimulate}
            disabled={phase === 'loading'}
            className="btn btn-cta glow-effect"
            style={{
              width: '100%',
              padding: '16px',
              fontSize: '1.05rem',
              fontWeight: 700,
              cursor: phase === 'loading' ? 'not-allowed' : 'pointer',
              opacity: phase === 'loading' ? 0.7 : 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
            }}
          >
            {phase === 'loading' ? (
              <>
                <span
                  style={{
                    width: '18px',
                    height: '18px',
                    border: '2px solid rgba(255,255,255,0.3)',
                    borderTopColor: '#fff',
                    borderRadius: '50%',
                    animation: 'spin 0.8s linear infinite',
                    flexShrink: 0,
                  }}
                />
                Simulando...
              </>
            ) : (
              'Simular Portafolio Optimo'
            )}
          </motion.button>
        </motion.div>

        {/* RIGHT: Results Panel */}
        <motion.div
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="glass-panel"
          style={{ padding: '28px', borderTop: '3px solid var(--accent-secondary)', minHeight: '500px' }}
        >
          <AnimatePresence mode="wait">
            {/* IDLE STATE */}
            {phase === 'idle' && (
              <motion.div
                key="idle"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '100%',
                  minHeight: '460px',
                  textAlign: 'center',
                  gap: '16px',
                }}
              >
                <div
                  style={{
                    width: '80px',
                    height: '80px',
                    borderRadius: '50%',
                    background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '2rem',
                    opacity: 0.7,
                  }}
                >
                  <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    <path d="M12 8v4l3 3" />
                  </svg>
                </div>
                <h3
                  style={{
                    fontFamily: 'var(--font-display)',
                    fontSize: '1.3rem',
                    color: 'var(--text-main)',
                  }}
                >
                  Resultados de Optimizacion
                </h3>
                <p style={{ color: 'var(--text-muted)', maxWidth: '320px', fontSize: '0.9rem' }}>
                  Haz clic en &quot;Simular Portafolio Optimo&quot; para ver la distribucion optima de activos.
                </p>
              </motion.div>
            )}

            {/* LOADING STATE */}
            {phase === 'loading' && (
              <motion.div
                key="loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '100%',
                  minHeight: '460px',
                  textAlign: 'center',
                  gap: '24px',
                }}
              >
                <div
                  style={{
                    width: '60px',
                    height: '60px',
                    border: '3px solid var(--border-light)',
                    borderTopColor: 'var(--accent-primary)',
                    borderRadius: '50%',
                    animation: 'spin 0.8s linear infinite',
                  }}
                />
                <motion.p
                  key={loadingMsg}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  style={{
                    color: 'var(--accent-primary)',
                    fontWeight: 600,
                    fontSize: '1rem',
                  }}
                >
                  {LOADING_MESSAGES[loadingMsg]}
                </motion.p>
              </motion.div>
            )}

            {/* RESULTS STATE */}
            {phase === 'done' && (
              <motion.div
                key="done"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.4 }}
                style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}
              >
                <h3
                  style={{
                    fontFamily: 'var(--font-display)',
                    fontSize: '1.2rem',
                    color: 'var(--text-main)',
                    marginBottom: '4px',
                  }}
                >
                  Distribucion Optima
                </h3>

                {/* PieChart */}
                <div style={{ width: '100%', height: '260px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={pieData}
                        cx="50%"
                        cy="50%"
                        innerRadius={55}
                        outerRadius={100}
                        dataKey="value"
                        stroke="rgba(0,0,0,0.3)"
                        strokeWidth={1}
                        animationBegin={0}
                        animationDuration={800}
                      >
                        {pieData.map((entry) => (
                          <Cell key={entry.name} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip
                        formatter={(value, name) => {
                          const item = pieData.find((p) => p.name === name);
                          return [`${value}%`, item?.fullName ?? String(name)];
                        }}
                        contentStyle={{
                          background: 'rgba(15, 23, 42, 0.95)',
                          border: '1px solid var(--border-light)',
                          borderRadius: '8px',
                          color: '#F1F5F9',
                          fontSize: '0.85rem',
                        }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>

                {/* Legend grid */}
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))',
                    gap: '6px',
                    fontSize: '0.78rem',
                  }}
                >
                  {pieData.map((entry) => (
                    <div key={entry.name} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <div
                        style={{
                          width: '8px',
                          height: '8px',
                          borderRadius: '50%',
                          backgroundColor: entry.color,
                          flexShrink: 0,
                        }}
                      />
                      <span style={{ color: 'var(--text-muted)' }}>
                        {entry.name}{' '}
                        <span style={{ fontWeight: 700, color: 'var(--text-main)' }}>{entry.value}%</span>
                      </span>
                    </div>
                  ))}
                </div>

                {/* Metric Cards */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  <motion.div
                    initial={{ opacity: 0, y: 15 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="glass-panel"
                    style={{
                      padding: '18px',
                      textAlign: 'center',
                      border: '1px solid rgba(16, 185, 129, 0.2)',
                    }}
                  >
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: '6px' }}>
                      Retorno Esperado
                    </p>
                    <p
                      style={{
                        fontSize: '1.8rem',
                        fontWeight: 800,
                        color: 'var(--success)',
                        fontFamily: 'var(--font-display)',
                      }}
                    >
                      +{MOCK_RESULT.expectedReturn}%
                    </p>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>anualizado</p>
                  </motion.div>

                  <motion.div
                    initial={{ opacity: 0, y: 15 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                    className="glass-panel"
                    style={{
                      padding: '18px',
                      textAlign: 'center',
                      border: '1px solid rgba(245, 158, 11, 0.2)',
                    }}
                  >
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: '6px' }}>
                      Volatilidad (Riesgo)
                    </p>
                    <p
                      style={{
                        fontSize: '1.8rem',
                        fontWeight: 800,
                        color: 'var(--warning)',
                        fontFamily: 'var(--font-display)',
                      }}
                    >
                      {MOCK_RESULT.volatility}%
                    </p>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>anualizada</p>
                  </motion.div>
                </div>

                {/* Sharpe badge */}
                <div
                  style={{
                    textAlign: 'center',
                    padding: '10px',
                    borderRadius: '10px',
                    background: 'var(--overlay-soft)',
                    border: '1px solid var(--border-light)',
                  }}
                >
                  <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                    Sharpe Ratio:{' '}
                    <span style={{ fontWeight: 700, color: 'var(--accent-primary)' }}>{MOCK_RESULT.sharpe}</span>
                  </span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>

      {/* CTA Section */}
      <AnimatePresence>
        {phase === 'done' && (
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.5 }}
            className="glass-panel"
            style={{
              marginTop: '32px',
              padding: '32px',
              textAlign: 'center',
              borderRadius: '16px',
              background: 'linear-gradient(135deg, rgba(37, 99, 235, 0.08), rgba(20, 184, 166, 0.08))',
              border: '1px solid rgba(37, 99, 235, 0.2)',
            }}
          >
            <p
              style={{
                fontSize: '1.2rem',
                fontFamily: 'var(--font-display)',
                color: 'var(--text-main)',
                marginBottom: '8px',
              }}
            >
              Imagina las simulaciones que puedes hacer con un sistema completo y poderoso como Kaudal.
            </p>
            <p style={{ color: 'var(--text-muted)', marginBottom: '20px', fontSize: '0.9rem' }}>
              Accede a +6,000 activos, multiples modelos de optimizacion y analisis en tiempo real.
            </p>
            <Link href="/login">
              <motion.button
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.97 }}
                className="btn btn-cta glow-effect"
                style={{
                  padding: '16px 36px',
                  fontSize: '1.05rem',
                  fontWeight: 700,
                  cursor: 'pointer',
                }}
              >
                Crear mi cuenta gratis
              </motion.button>
            </Link>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Spinner keyframes */}
      <style jsx global>{`
        @keyframes spin {
          to {
            transform: rotate(360deg);
          }
        }
      `}</style>
    </div>
  );
}
