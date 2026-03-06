import Navbar from '@/components/layout/Navbar';
import Footer from '@/components/layout/Footer';
import type { Metadata } from 'next';

export const metadata: Metadata = {
    title: 'Términos y Condiciones — Kaudal',
    description: 'Términos y condiciones de uso de Kaudal, herramienta educativa de optimización de portafolios.',
};

export default function TerminosPage() {
    return (
        <>
            <Navbar />
            <main className="page-wrapper" id="terminos-view" style={{ paddingTop: '100px', minHeight: 'calc(100vh - 60px)' }}>
                <div className="about-page">
                    <article className="glass-panel legal-content text-left" style={{ maxWidth: '800px', margin: '0 auto', padding: '40px' }}>
                        <h1>Términos y Condiciones</h1>
                        <p className="meta">Última actualización: 3 de marzo de 2026</p>

                        <h2>1. Naturaleza del servicio</h2>
                        <p>Kaudal es una <strong>herramienta educativa y matemática</strong> de optimización de portafolios de inversión. El servicio <strong>NO constituye gestión patrimonial, de inversión ni bursátil</strong> certificada por ninguna autoridad reguladora mexicana (CNBV, CONSAR, Condusef). Toda decisión de inversión es responsabilidad exclusiva del usuario.</p>

                        <h2>2. Modelo Markowitz y sus limitaciones</h2>
                        <p>La optimización se basa en el modelo de Media-Varianza de Harry Markowitz. Este modelo utiliza <strong>datos históricos</strong> de rendimientos pasados, los cuales <strong>no garantizan rendimientos futuros</strong>. Las proyecciones son aproximaciones matemáticas, no predicciones.</p>

                        <h2>3. Límite de responsabilidad</h2>
                        <p>Kaudal, sus desarrolladores y colaboradores <strong>no se hacen responsables</strong> de:</p>
                        <ul>
                            <li>Pérdidas financieras derivadas de decisiones de inversión del usuario.</li>
                            <li>Errores en datos de mercado proporcionados por fuentes externas.</li>
                            <li>Interrupciones del servicio o fallas técnicas.</li>
                        </ul>
                        <p>El uso de la plataforma es bajo <strong>tu propio riesgo</strong>. Es fundamental consultar a un profesional financiero certificado antes de invertir.</p>

                        <h2>4. Uso permitido</h2>
                        <p>Al usar Kaudal, aceptas:</p>
                        <ul>
                            <li>Usar la plataforma solo con fines educativos y personales.</li>
                            <li>No realizar ingeniería inversa, descompilar o desensamblar el software.</li>
                            <li>No utilizar bots, scrapers o sistemas automatizados para extraer datos de la plataforma.</li>
                            <li>No reproducir, redistribuir ni comercializar los resultados de forma masiva.</li>
                        </ul>

                        <h2>5. Propiedad intelectual</h2>
                        <p>Todo el contenido de Kaudal — incluyendo código, diseño, textos y logotipos — es propiedad de sus creadores y está protegido por la Ley Federal del Derecho de Autor de México.</p>

                        <h2>6. Cuenta y datos</h2>
                        <p>Eres responsable de mantener la confidencialidad de tu cuenta. Nos reservamos el derecho de suspender cuentas que violen estos términos.</p>

                        <h2>7. Modificaciones</h2>
                        <p>Podemos actualizar estos términos en cualquier momento. Si los cambios son significativos, te notificaremos por correo electrónico. El uso continuado de la plataforma implica la aceptación de los términos actualizados.</p>

                        <h2>8. Legislación aplicable</h2>
                        <p>Estos términos se rigen por las leyes de los Estados Unidos Mexicanos. Cualquier controversia será resuelta ante los tribunales competentes de la Ciudad de México.</p>
                    </article>
                </div>
            </main>
            <Footer />
        </>
    );
}
