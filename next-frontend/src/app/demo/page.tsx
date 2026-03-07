import Navbar from '@/components/layout/Navbar';
import Footer from '@/components/layout/Footer';
import DemoOptimizerLight from '@/components/DemoOptimizerLight';

export const metadata = {
    title: 'Demo Interactiva | Kaudal',
    description: 'Prueba el optimizador de portafolios de Kaudal con una simulacion gratuita.',
};

export default function DemoPage() {
    return (
        <>
            <Navbar />
            <div className="page-wrapper" id="demo-view" style={{ paddingTop: '100px', minHeight: 'calc(100vh - 60px)' }}>
                <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 16px' }}>
                    <div style={{ textAlign: 'center', marginBottom: '40px' }}>
                        <h1
                            style={{
                                fontFamily: 'var(--font-display)',
                                fontSize: 'clamp(1.8rem, 4vw, 2.6rem)',
                                color: 'var(--text-main)',
                                marginBottom: '12px',
                                lineHeight: 1.2,
                            }}
                        >
                            Prueba el poder de{' '}
                            <span
                                style={{
                                    background: 'linear-gradient(135deg, #3b82f6, #10b981)',
                                    WebkitBackgroundClip: 'text',
                                    WebkitTextFillColor: 'transparent',
                                }}
                            >
                                Kaudal
                            </span>
                        </h1>
                        <p
                            style={{
                                color: 'var(--text-muted)',
                                fontSize: '1.05rem',
                                maxWidth: '600px',
                                margin: '0 auto',
                            }}
                        >
                            Simulacion estatica con datos pre-calculados. Descubre como Kaudal
                            optimiza la distribucion de un portafolio de 10 activos usando el modelo de Markowitz.
                        </p>
                    </div>

                    <DemoOptimizerLight />
                </div>
            </div>
            <Footer />
        </>
    );
}
