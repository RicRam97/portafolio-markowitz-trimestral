import Navbar from '@/components/layout/Navbar';
import Footer from '@/components/layout/Footer';

export default function AcercaPage() {
    return (
        <>
            <Navbar />
            <div className="page-wrapper" id="about-view" style={{ paddingTop: '60px' }}>
                <div className="about-page">
                    <article className="about-content glass-panel" style={{ maxWidth: '800px', padding: '48px', margin: '40px auto' }}>
                        <h1 className="about-title">Acerca de Kaudal</h1>

                        <section className="about-section" style={{ marginBottom: '32px' }}>
                            <h2 style={{ fontFamily: 'var(--font-display)', marginBottom: '16px' }}>🎯 Nuestra Misión</h2>
                            <p style={{ color: 'var(--text-muted)', marginBottom: '12px' }}>Democratizar el acceso a herramientas de análisis de portafolios de inversión. Creemos que cualquier persona, sin importar su experiencia financiera, merece entender cómo funciona la diversificación y la optimización de inversiones.</p>
                            <p style={{ color: 'var(--text-muted)' }}>Esta herramienta nació en México con un enfoque especial para el mercado latinoamericano: incluye activos de la <strong>Bolsa Mexicana de Valores (BMV)</strong> y del <strong>S&P 500</strong>, con precios normalizados a USD.</p>
                        </section>

                        <section className="about-section" style={{ marginBottom: '32px' }}>
                            <h2 style={{ fontFamily: 'var(--font-display)', marginBottom: '16px' }}>📊 El Modelo Matemático</h2>
                            <p style={{ color: 'var(--text-muted)', marginBottom: '12px' }}>Implementamos la <strong>Teoría Moderna de Portafolios</strong> de Harry Markowitz (Premio Nobel de Economía, 1990). El modelo de <em>Media-Varianza</em> encuentra la combinación óptima de activos que maximiza el rendimiento esperado para un nivel de riesgo dado.</p>
                            <p style={{ color: 'var(--text-muted)' }}>Además ofrecemos la estrategia <strong>HRP (Hierarchical Risk Parity)</strong>, un método moderno y defensivo que distribuye el riesgo de forma jerárquica sin depender de la inversión de matrices de covarianza.</p>
                        </section>

                        <section className="about-section" style={{ marginBottom: '32px' }}>
                            <h2 style={{ fontFamily: 'var(--font-display)', marginBottom: '16px' }}>📶 Datos y Metodología</h2>
                            <ul className="about-list" style={{ color: 'var(--text-muted)', paddingLeft: '20px' }}>
                                <li style={{ marginBottom: '8px' }}><strong>Fuente de precios:</strong> Yahoo Finance (datos públicos en tiempo real)</li>
                                <li style={{ marginBottom: '8px' }}><strong>Horizonte de análisis:</strong> 3 años de datos históricos</li>
                                <li style={{ marginBottom: '8px' }}><strong>Filtro de liquidez:</strong> Solo incluimos activos con un volumen promedio diario (ADV) mayor a $1M USD. Esto elimina acciones poco negociadas que podrían distorsionar el análisis con precios erráticos.</li>
                                <li style={{ marginBottom: '8px' }}><strong>Smart Beta:</strong> Filtro momentum que selecciona los 15 activos con mejor desempeño reciente</li>
                                <li style={{ marginBottom: '8px' }}><strong>Limpieza:</strong> Detección y tratamiento automático de outliers estadísticos</li>
                            </ul>
                        </section>

                        <section className="about-section" style={{ marginBottom: '32px' }}>
                            <h2 style={{ fontFamily: 'var(--font-display)', marginBottom: '16px' }}>⚠️ Limitaciones y Disclaimer</h2>
                            <div className="about-disclaimer" style={{ background: 'rgba(239, 68, 68, 0.1)', borderLeft: '4px solid var(--danger)', padding: '16px', borderRadius: '4px' }}>
                                <p style={{ marginBottom: '12px', color: 'var(--text-main)' }}><strong>Esta herramienta es estrictamente educativa y de exploración.</strong></p>
                                <ul className="about-list" style={{ color: 'var(--text-muted)', paddingLeft: '20px', margin: 0 }}>
                                    <li style={{ marginBottom: '4px' }}>No constituye, bajo ninguna circunstancia, un servicio de gestión patrimonial o alerta de operaciones.</li>
                                    <li style={{ marginBottom: '4px' }}>Los rendimientos pasados no garantizan resultados futuros.</li>
                                    <li style={{ marginBottom: '4px' }}>No ejecutamos, intermediamos ni facilitamos operaciones de compra o venta de valores.</li>
                                    <li style={{ marginBottom: '4px' }}>Consulta siempre con un asesor financiero certificado antes de tomar decisiones de inversión.</li>
                                </ul>
                            </div>
                        </section>

                        <section className="about-section">
                            <h2 style={{ fontFamily: 'var(--font-display)', marginBottom: '16px' }}>🚀 Contacto y Contribuciones</h2>
                            <p style={{ color: 'var(--text-muted)' }}>¿Tienes comentarios, encontraste un bug, o quieres contribuir? Contáctanos en
                                <a href="mailto:contacto@kaudal.com.mx" className="footer-link" style={{ marginLeft: '4px', textDecoration: 'underline' }}>contacto@kaudal.com.mx</a>.
                            </p>
                            <div className="about-social-row" style={{ display: 'flex', gap: '12px', marginTop: '16px' }}>
                                <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer" className="social-icon">
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.32 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.79M6.88 8.56a1.68 1.68 0 0 0 1.68-1.68c0-.93-.75-1.69-1.68-1.69a1.69 1.69 0 0 0-1.69 1.69c0 .93.76 1.68 1.69 1.68m1.39 9.94v-8.37H5.5v8.37h2.77z" /></svg>
                                </a>
                                <a href="https://x.com" target="_blank" rel="noopener noreferrer" className="social-icon">
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" /></svg>
                                </a>
                            </div>
                        </section>

                        {/* Transparencia del Fundador (Movido desde Landing) */}
                        <section className="about-section" style={{ marginBottom: '32px' }}>
                            <div className="glass-panel" style={{ padding: '32px', borderTop: '4px solid var(--success)', display: 'flex', flexDirection: 'column' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '24px' }}>
                                    <div style={{ width: '64px', height: '64px', borderRadius: '50%', background: 'var(--surface)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '2rem', border: '2px solid var(--success)' }}>
                                        👨‍💻
                                    </div>
                                    <div>
                                        <h3 style={{ fontSize: '1.2rem', marginBottom: '4px', fontFamily: 'var(--font-display)' }}>Creado por Ricardo Ramírez</h3>
                                        <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accent-primary)', textDecoration: 'none', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '4px', fontWeight: 600 }}>
                                            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                                                <path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.32 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.79M6.88 8.56a1.68 1.68 0 0 0 1.68-1.68c0-.93-.75-1.69-1.68-1.69a1.69 1.69 0 0 0-1.69 1.69c0 .93.76 1.68 1.69 1.68m1.39 9.94v-8.37H5.5v8.37h2.77z" />
                                            </svg>
                                            Ver perfil en LinkedIn
                                        </a>
                                    </div>
                                </div>
                                <p style={{ color: 'var(--text-muted)', lineHeight: 1.6, fontStyle: 'italic', borderLeft: '3px solid rgba(255,255,255,0.1)', paddingLeft: '16px' }}>
                                    "Desarrollé Kaudal como una herramienta matemática honesta. Quiero empoderar a los verdaderos inversionistas para que tomen control de su propio proceso. Creo en la democratización de las herramientas de inversión."
                                </p>
                            </div>
                        </section>
                    </article>
                </div>
            </div>
            <Footer />
        </>
    );
}
