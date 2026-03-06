import Navbar from '@/components/layout/Navbar';
import Footer from '@/components/layout/Footer';

export default function ConocePage() {
    return (
        <>
            <Navbar />
            <div className="page-wrapper" id="conoce-view" style={{ paddingTop: '60px' }}>
                <div className="about-page">
                    <article className="about-content glass-panel" style={{ maxWidth: '1000px', padding: '48px', margin: '40px auto' }}>
                        <h1 className="about-title" style={{ textAlign: 'center', marginBottom: '40px' }}>Conoce Kaudal</h1>
                        <p style={{ color: 'var(--text-muted)', textAlign: 'center', fontSize: '1.1rem', marginBottom: '48px', maxWidth: '700px', marginInline: 'auto' }}>
                            Kaudal democratiza el análisis de portafolios, brindando a cada inversionista las herramientas computacionales
                            que usan las grandes firmas, presentadas de forma intuitiva.
                        </p>

                        <div className="conoce-grid" style={{ display: 'flex', flexDirection: 'column', gap: '48px' }}>
                            {/* Feature 1 */}
                            <div className="conoce-feature glass-panel" style={{ display: 'flex', flexWrap: 'wrap', gap: '24px', alignItems: 'center', padding: '24px', border: '1px solid var(--border-light)' }}>
                                <div style={{ flex: 1, minWidth: '300px' }}>
                                    <div style={{ fontSize: '2.5rem', marginBottom: '12px' }}>📊</div>
                                    <h3 style={{ fontSize: '1.4rem', fontWeight: 700, marginBottom: '12px', fontFamily: 'var(--font-display)' }}>Optimización Avanzada de Activos</h3>
                                    <p style={{ color: 'var(--text-muted)', lineHeight: 1.6 }}>
                                        Usando el modelo de Media-Varianza y la Paridad de Riesgo Jerárquico, Kaudal analiza el comportamiento
                                        histórico de tus empresas favoritas para encontrar la distribución matemáticamente más balanceada,
                                        minimizando tu riesgo y maximizando rendimientos.
                                    </p>
                                </div>
                                <div style={{ flex: 1, minWidth: '300px', background: 'rgba(15,23,42,0.8)', borderRadius: '12px', padding: '24px', border: '1px solid rgba(59,130,246,0.3)', textAlign: 'center' }}>
                                    <div style={{ height: '150px', display: 'flex', alignItems: 'flex-end', justifyContent: 'center', gap: '8px' }}>
                                        <div style={{ width: '30px', height: '40%', background: 'var(--accent-primary)', borderRadius: '4px 4px 0 0' }} />
                                        <div style={{ width: '30px', height: '70%', background: 'var(--success)', borderRadius: '4px 4px 0 0' }} />
                                        <div style={{ width: '30px', height: '50%', background: 'var(--warning)', borderRadius: '4px 4px 0 0' }} />
                                        <div style={{ width: '30px', height: '90%', background: '#a78bfa', borderRadius: '4px 4px 0 0' }} />
                                    </div>
                                    <p style={{ marginTop: '12px', fontWeight: 600, color: 'var(--accent-primary)' }}>Distribución Cuantitativa Eficiente</p>
                                </div>
                            </div>

                            {/* Feature 2 */}
                            <div className="conoce-feature glass-panel" style={{ display: 'flex', flexWrap: 'wrap-reverse', gap: '24px', alignItems: 'center', padding: '24px', border: '1px solid var(--border-light)' }}>
                                <div style={{ flex: 1, minWidth: '300px', background: 'rgba(15,23,42,0.8)', borderRadius: '12px', padding: '24px', border: '1px solid rgba(16,185,129,0.3)', textAlign: 'center' }}>
                                    <div style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: '120px', height: '120px', borderRadius: '50%', border: '10px solid var(--success)', borderRightColor: 'var(--accent-primary)', borderBottomColor: 'var(--warning)' }}>
                                        <span style={{ fontSize: '1.5rem', fontWeight: 700 }}>+500</span>
                                    </div>
                                    <p style={{ marginTop: '16px', fontWeight: 600, color: 'var(--success)' }}>Cobertura Global de Mercado</p>
                                </div>
                                <div style={{ flex: 1, minWidth: '300px' }}>
                                    <div style={{ fontSize: '2.5rem', marginBottom: '12px' }}>🌎</div>
                                    <h3 style={{ fontSize: '1.4rem', fontWeight: 700, marginBottom: '12px', fontFamily: 'var(--font-display)' }}>Directorio Global de Acciones</h3>
                                    <p style={{ color: 'var(--text-muted)', lineHeight: 1.6 }}>
                                        Accede a más de 500 activos filtrados por liquidez en el S&P 500 y la Bolsa Mexicana de Valores.
                                        Normalizados automáticamente a dólares para asegurar cálculos coherentes y representaciones gráficas de
                                        alto impacto visual.
                                    </p>
                                </div>
                            </div>

                            {/* Feature 3 */}
                            <div className="conoce-feature glass-panel" style={{ display: 'flex', flexWrap: 'wrap', gap: '24px', alignItems: 'center', padding: '24px', border: '1px solid var(--border-light)' }}>
                                <div style={{ flex: 1, minWidth: '300px' }}>
                                    <div style={{ fontSize: '2.5rem', marginBottom: '12px' }}>🌱</div>
                                    <h3 style={{ fontSize: '1.4rem', fontWeight: 700, marginBottom: '12px', fontFamily: 'var(--font-display)' }}>Test de Sueños y Simulaciones</h3>
                                    <p style={{ color: 'var(--text-muted)', lineHeight: 1.6 }}>
                                        ¿Apenas empiezas? Nuestra plataforma de simulación traduce tus objetivos personales (&ldquo;Ahorrar para una
                                        meta en 10 años&rdquo;) en
                                        parámetros matemáticos. Realizamos simulaciones de estrés para ver cómo habría reaccionado tu
                                        portafolio en momentos de crisis históricas.
                                    </p>
                                </div>
                                <div style={{ flex: 1, minWidth: '300px', background: 'rgba(15,23,42,0.8)', borderRadius: '12px', padding: '24px', border: '1px solid rgba(245,158,11,0.3)', textAlign: 'center' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                                        <span style={{ color: 'var(--text-muted)' }}>Ahorro Proyectado:</span>
                                        <span style={{ color: 'var(--success)', fontWeight: 700, fontSize: '1.2rem' }}>$1.2M</span>
                                    </div>
                                    <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
                                        <div style={{ width: '80%', height: '100%', background: 'var(--warning)', borderRadius: '4px' }} />
                                    </div>
                                    <p style={{ marginTop: '16px', fontWeight: 600, color: 'var(--warning)' }}>Traducción Directa a Tus Metas</p>
                                </div>
                            </div>
                        </div>
                    </article>
                </div>
            </div>
            <Footer />
        </>
    );
}
