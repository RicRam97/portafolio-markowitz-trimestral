import Navbar from '@/components/layout/Navbar';
import Footer from '@/components/layout/Footer';

export default function FAQPage() {
    return (
        <>
            <Navbar />
            <div className="page-wrapper" id="faq-view" style={{ paddingTop: '60px' }}>
                <div className="about-page">
                    <article className="about-content glass-panel" style={{ maxWidth: '800px', padding: '48px', margin: '40px auto' }}>
                        <h1 className="about-title">Preguntas Frecuentes</h1>
                        <p style={{ color: 'var(--text-muted)', marginBottom: '28px' }}>Todo lo que necesitas saber antes de empezar.</p>

                        <div className="faq-container">
                            <details className="faq-item glass-panel">
                                <summary className="faq-question">¿Es seguro usar esta herramienta?</summary>
                                <div className="faq-answer">
                                    <p>Sí. Esta herramienta es 100% educativa. No manejamos tu dinero, no tenemos acceso a tus cuentas bancarias ni de inversión. Todo se ejecuta en tu navegador y los datos que ingresas no se almacenan en ningún servidor.</p>
                                </div>
                            </details>
                            <details className="faq-item glass-panel">
                                <summary className="faq-question">¿Necesito crear una cuenta?</summary>
                                <div className="faq-answer">
                                    <p>Por el momento no. Puedes usar la herramienta libremente. En el futuro implementaremos cuentas para guardar tus portafolios y dar seguimiento a tu progreso.</p>
                                </div>
                            </details>
                            <details className="faq-item glass-panel">
                                <summary className="faq-question">¿Van a pedirme dinero real o datos bancarios?</summary>
                                <div className="faq-answer">
                                    <p>Jamás. Esta herramienta no solicita, procesa ni almacena información financiera personal. El "presupuesto" que ingresas es solo un número de referencia para los cálculos educativos.</p>
                                </div>
                            </details>
                            <details className="faq-item glass-panel">
                                <summary className="faq-question">¿En qué se basa la optimización?</summary>
                                <div className="faq-answer">
                                    <p>Usamos la Teoría Moderna de Portafolios creada por Harry Markowitz, ganador del Premio Nobel de Economía en 1990. El modelo de Media-Varianza busca la mejor combinación de activos para maximizar el rendimiento esperado dado un nivel de riesgo.</p>
                                </div>
                            </details>
                            <details className="faq-item glass-panel">
                                <summary className="faq-question">¿Puedo perder dinero por usar esta herramienta?</summary>
                                <div className="faq-answer">
                                    <p>La herramienta no ejecuta compras ni ventas. Es un simulador educativo. Si decides invertir por tu cuenta usando los resultados como referencia, recuerda que toda inversión conlleva riesgo y los rendimientos pasados no garantizan resultados futuros.</p>
                                </div>
                            </details>
                            <details className="faq-item glass-panel">
                                <summary className="faq-question">¿De dónde vienen los datos de precios?</summary>
                                <div className="faq-answer">
                                    <p>Los precios históricos se obtienen en tiempo real de Yahoo Finance, una fuente pública y ampliamente utilizada. Usamos un horizonte de 3 años de datos para los cálculos de optimización.</p>
                                </div>
                            </details>
                            <details className="faq-item glass-panel">
                                <summary className="faq-question">¿Esto es una herramienta de gestión patrimonial automatizada (Robo-advisor)?</summary>
                                <div className="faq-answer">
                                    <p><strong>No.</strong> Esta herramienta tiene fines exclusivamente educativos matemáticos y de exploración de datos. No brindamos recomendaciones personalizadas, alertas de compra/venta, ni servicio de Robo-Advisor. No somos asesores financieros registrados. Antes de tomar cualquier decisión de inversión real, consulta con un profesional certificado.</p>
                                </div>
                            </details>
                        </div>

                    </article>
                </div>
            </div>
            <Footer />
        </>
    );
}
