import Navbar from '@/components/layout/Navbar';
import Footer from '@/components/layout/Footer';
import type { Metadata } from 'next';

export const metadata: Metadata = {
    title: 'Disclaimer Financiero — Kaudal',
    description: 'Disclaimer financiero de Kaudal. Esta herramienta es matemática con fines educativos, no gestión patrimonial.',
};

export default function DisclaimerPage() {
    return (
        <>
            <Navbar />
            <main className="page-wrapper" id="disclaimer-view" style={{ paddingTop: '100px', minHeight: 'calc(100vh - 60px)' }}>
                <div className="about-page">
                    <article className="glass-panel legal-content text-left" style={{ maxWidth: '800px', margin: '0 auto', padding: '40px' }}>
                        <h1>Disclaimer Financiero</h1>
                        <p className="meta">Última actualización: 3 de marzo de 2026</p>

                        <h2>Versión corta (pie de resultados)</h2>
                        <div style={{ background: 'rgba(15,23,42,0.4)', border: '1px solid var(--border-light)', borderLeft: '4px solid var(--accent-primary)', padding: '1.25rem', margin: '1.5rem 0', borderRadius: '4px' }}>
                            <p style={{ marginBottom: 0 }}><strong>⚠️ Kaudal es una herramienta de simulación matemática. No ofrece gestión patrimonial. Los rendimientos pasados no garantizan resultados futuros. Invierte bajo tu propia responsabilidad.</strong></p>
                        </div>

                        <h2>Versión larga (aviso completo)</h2>
                        <div style={{ background: 'rgba(15,23,42,0.8)', border: '2px solid var(--accent-primary)', borderRadius: '12px', padding: '2rem', margin: '2rem 0' }}>
                            <h3 style={{ color: 'var(--text-main)', marginBottom: '1rem', fontSize: '1.2rem' }}>⚠️ Aviso importante antes de continuar</h3>
                            <p>Al usar <strong>Kaudal</strong>, reconoces y aceptas lo siguiente:</p>
                            <ul>
                                <li>Kaudal es una <strong>herramienta de simulación matemática</strong>. No es, ni pretende ser, un servicio de gestión patrimonial, de inversión ni bursátil.</li>
                                <li>No estamos registrados ni supervisados por la CNBV, CONSAR, Condusef ni ninguna autoridad financiera mexicana.</li>
                                <li>Las optimizaciones se basan en el <strong>modelo de Markowitz</strong>, que utiliza datos históricos. <strong>Los rendimientos pasados no son indicador de rendimientos futuros.</strong></li>
                                <li>Cualquier decisión de inversión que tomes es tu <strong>responsabilidad exclusiva</strong>. Kaudal no se hace responsable de pérdidas, ganancias ni resultados derivados del uso de la plataforma.</li>
                                <li>Es fundamental <strong>consultar a un profesional financiero certificado</strong> antes de tomar decisiones de inversión.</li>
                            </ul>
                        </div>

                        <h2>Naturaleza del servicio</h2>
                        <p>Kaudal aplica matemáticas financieras públicamente reconocidas (Teoría Moderna de Portafolios, Premio Nobel de Economía 1990) para fines educativos y de simulación. Nuestro algoritmo calcula proporciones óptimas basadas en datos históricos, pero no predice el futuro.</p>

                        <h2>Sin asesoría personalizada</h2>
                        <p>Ningún resultado generado por Kaudal constituye asesoría de inversión personalizada. La plataforma no conoce tu situación fiscal, horizonte de inversión específico, tolerance al riesgo real ni objetivos de vida completos. Por esto, ningún output de la plataforma debe considerarse como recomendación de compra o venta de valores.</p>

                        <h2>Fuentes de datos</h2>
                        <p>Kaudal utiliza datos históricos de fuentes de terceros (Yahoo Finance u otras APIs). No garantizamos la exactitud, completitud o disponibilidad continua de estos datos.</p>
                    </article>
                </div>
            </main>
            <Footer />
        </>
    );
}
