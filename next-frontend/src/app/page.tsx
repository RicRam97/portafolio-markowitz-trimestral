import Navbar from '@/components/layout/Navbar';
import Footer from '@/components/layout/Footer';
import DashboardPreview from '@/components/landing/DashboardPreview';
import Link from 'next/link';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Kaudal — Herramienta Educativa de Portafolios de Inversión',
  description: 'Aprende a construir portafolios de inversión con la Teoría Moderna de Markowitz.',
};

export default function LandingPage() {
  return (
    <>
      <Navbar />
      <div className="landing-page" id="landing-view">
        {/* HERO SECTION */}
        <section className="hero-section">
          <div className="hero-bg-orbs">
            <div className="hero-orb hero-orb--blue" />
            <div className="hero-orb hero-orb--green" />
            <div className="hero-orb hero-orb--purple" />
          </div>
          <div className="hero-content">
            <div className="hero-badge">🎓 Herramienta Educativa</div>
            <h1 className="hero-title">
              Aprende a Construir Portafolios con la <span className="hero-gradient-text">Teoría de Markowitz</span>
            </h1>
            <p className="hero-subtitle" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <span style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--success)' }}>
                🔒 No tocamos tu dinero
              </span>
              <span>Nosotros hacemos las matemáticas, tú inviertes en tu broker de confianza.</span>
            </p>
            <div className="hero-ctas">
              <Link href="/dashboard" className="btn btn-cta glow-effect hero-cta-main" style={{ textDecoration: 'none' }}>
                Explorar la Herramienta
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </Link>
              <Link href="/dashboard?mode=beginner" className="btn btn-secondary hero-cta-alt" style={{ textDecoration: 'none' }}>
                🌱 ¿Nuevo? Modo Principiantes
              </Link>
            </div>
          </div>
          <div className="hero-visual">
            <div className="hero-chart-mock">
              <svg className="hero-svg" viewBox="0 0 400 250" fill="none" xmlns="http://www.w3.org/2000/svg">
                <line x1="50" y1="30" x2="50" y2="220" stroke="rgba(148,163,184,0.15)" strokeWidth="1" />
                <line x1="50" y1="220" x2="380" y2="220" stroke="rgba(148,163,184,0.15)" strokeWidth="1" />
                <line x1="50" y1="160" x2="380" y2="160" stroke="rgba(148,163,184,0.08)" strokeWidth="1" strokeDasharray="4" />
                <line x1="50" y1="100" x2="380" y2="100" stroke="rgba(148,163,184,0.08)" strokeWidth="1" strokeDasharray="4" />
                <path className="hero-frontier-path" d="M70 200 Q120 195 150 175 Q180 155 200 130 Q230 95 270 70 Q310 50 360 45" stroke="url(#frontierGrad)" strokeWidth="3" fill="none" strokeLinecap="round" />
                <circle className="hero-dot hero-dot--1" cx="150" cy="175" r="5" fill="#3b82f6" />
                <circle className="hero-dot hero-dot--2" cx="200" cy="130" r="7" fill="#10b981" />
                <circle className="hero-dot hero-dot--3" cx="270" cy="70" r="5" fill="#f59e0b" />
                <circle className="hero-dot hero-dot--4" cx="360" cy="45" r="4" fill="#ef4444" />
                <circle className="hero-dot hero-dot--optimal" cx="200" cy="130" r="12" fill="none" stroke="#10b981" strokeWidth="2" strokeDasharray="3 3" />
                <text x="40" y="225" fill="#94a3b8" fontSize="10" textAnchor="end">Riesgo →</text>
                <text x="45" y="25" fill="#94a3b8" fontSize="10" textAnchor="end" transform="rotate(-90 45 25)">Retorno →</text>
                <text x="200" y="155" fill="#10b981" fontSize="11" textAnchor="middle" fontWeight="600">Óptimo ★</text>
                <defs>
                  <linearGradient id="frontierGrad" x1="70" y1="200" x2="360" y2="45" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stopColor="#3b82f6" />
                    <stop offset="50%" stopColor="#10b981" />
                    <stop offset="100%" stopColor="#f59e0b" />
                  </linearGradient>
                </defs>
              </svg>
              <div className="hero-chart-label">Frontera Eficiente</div>
            </div>
          </div>
        </section>

        {/* FEATURES SECTION */}
        <section className="landing-section" id="features">
          <h2 className="section-title">¿Qué puedes aprender?</h2>
          <p className="section-subtitle">Explora los conceptos fundamentales de la teoría moderna de portafolios de forma práctica.</p>
          <div className="features-grid">
            <div className="feature-card glass-panel">
              <div className="feature-icon">⚡</div>
              <h3>Optimización Markowitz</h3>
              <p>Calcula los pesos óptimos para maximizar el Sharpe Ratio con tu presupuesto de práctica.</p>
            </div>
            <div className="feature-card glass-panel">
              <div className="feature-icon">🌱</div>
              <h3>Modo Principiantes</h3>
              <p>El "Test de Sueños" traduce tus metas financieras en parámetros de simulación matemática.</p>
            </div>
            <div className="feature-card glass-panel">
              <div className="feature-icon">🌎</div>
              <h3>500+ Activos Globales</h3>
              <p>Explora acciones de la BMV y del S&P 500 con filtros por sector y mercado.</p>
            </div>
            <div className="feature-card glass-panel">
              <div className="feature-icon">🤖</div>
              <h3>Tecnología al alcance</h3>
              <p>Usamos modelos Machine Learning y Big Data para obtener mejores resultados.</p>
            </div>
          </div>
        </section>

        {/* HOW IT WORKS SECTION */}
        <section className="landing-section" id="how-it-works">
          <h2 className="section-title">Tan fácil como 1-2-3</h2>
          <p className="section-subtitle">No necesitas saber de finanzas. La herramienta hace el trabajo pesado por ti.</p>
          <div className="steps-container">
            <div className="step-card">
              <div className="step-number">1</div>
              <div className="step-content">
                <h3>Escoge empresas que te gusten</h3>
                <p>¿Te gusta Apple? ¿Coca-Cola? ¿OXXO? Selecciona las empresas que quieras explorar.</p>
              </div>
            </div>
            <div className="step-connector" />
            <div className="step-card">
              <div className="step-number">2</div>
              <div className="step-content">
                <h3>Presiona un botón</h3>
                <p>Nuestro algoritmo analiza 3 años de datos históricos y encuentra la mejor combinación.</p>
              </div>
            </div>
            <div className="step-connector" />
            <div className="step-card">
              <div className="step-number">3</div>
              <div className="step-content">
                <h3>Recibe tu receta personalizada</h3>
                <p>Calculamos la proporción matemática exacta para cada empresa y la cantidad de títulos.</p>
              </div>
            </div>
          </div>
        </section>

        {/* PREVIEW SECTION */}
        <section className="landing-section" id="preview">
          <h2 className="section-title">Vista previa del Dashboard</h2>
          <p className="section-subtitle">Así luce la herramienta cuando ejecutas una optimización de portafolios.</p>
          <DashboardPreview />
        </section>

        {/* TRANSPARENCIA RADICAL SECTION */}
        <section className="landing-section" id="transparencia">
          <h2 className="section-title">Transparencia Radical</h2>
          <p className="section-subtitle">Construido para combatir las estafas financieras. Aquí no hay cajas negras.</p>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', maxWidth: '800px', margin: '40px auto 0', width: '100%' }}>
            {/* Transparencia de Algoritmos */}
            <div className="glass-panel" style={{ padding: '32px', borderTop: '4px solid var(--accent-primary)', display: 'flex', flexDirection: 'column', width: '100%' }}>
              <div style={{ fontSize: '2.5rem', marginBottom: '16px', textAlign: 'center' }}>🧮</div>
              <h3 style={{ fontSize: '1.5rem', marginBottom: '16px', fontFamily: 'var(--font-display)', textAlign: 'center' }}>Algoritmos Públicos (Open Core)</h3>
              <p style={{ color: 'var(--text-muted)', lineHeight: 1.6, marginBottom: '24px', textAlign: 'center' }}>
                No usamos "inteligencia artificial mágica" que te promete rendimientos irreales. Toda nuestra matemática es pública y auditable, basada en la <strong>Teoría Moderna de Portafolios de Harry Markowitz</strong> (Premio Nobel, 1990).
              </p>
              <ul style={{ listStyle: 'none', padding: 0, color: 'var(--text-muted)', gap: '16px', display: 'flex', flexWrap: 'wrap', justifyContent: 'center' }}>
                <li style={{ display: 'flex', gap: '8px' }}><span style={{ color: 'var(--success)' }}>✓</span> Optimizador de Varianza</li>
                <li style={{ display: 'flex', gap: '8px' }}><span style={{ color: 'var(--success)' }}>✓</span> Riesgo Jerárquico (HRP)</li>
                <li style={{ display: 'flex', gap: '8px' }}><span style={{ color: 'var(--success)' }}>✓</span> Datos públicos de fuentes oficiales</li>
                <li style={{ display: 'flex', gap: '8px' }}><span style={{ color: 'var(--success)' }}>✓</span> Machine Learning y BigData</li>
              </ul>
            </div>

            <p style={{ color: 'var(--text-main)', fontStyle: 'italic', textAlign: 'center', marginTop: '32px', fontSize: '1.2rem', fontWeight: 600, lineHeight: 1.6 }}>
              "Lo más arriesgado que puedes hacer en tu vida es no invertir en tu educación financiera"
              <br />
              <span style={{ fontSize: '1rem', color: 'var(--text-muted)', fontWeight: 400, marginTop: '8px', display: 'block' }}>- R. Kiyosaki</span>
            </p>
          </div>
        </section>
      </div>
      <Footer />
    </>
  );
}
