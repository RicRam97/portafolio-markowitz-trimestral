import Navbar from '@/components/layout/Navbar';
import Footer from '@/components/layout/Footer';
import type { Metadata } from 'next';

export const metadata: Metadata = {
    title: 'Aviso de Privacidad — Kaudal',
    description: 'Aviso de privacidad de Kaudal. Conoce cómo protegemos tus datos personales conforme a la LFPDPPP.',
};

export default function PrivacidadPage() {
    return (
        <>
            <Navbar />
            <main className="page-wrapper" id="privacidad-view" style={{ paddingTop: '100px', minHeight: 'calc(100vh - 60px)' }}>
                <div className="about-page">
                    <article className="glass-panel legal-content text-left" style={{ maxWidth: '800px', margin: '0 auto', padding: '40px' }}>
                        <h1>Aviso de Privacidad</h1>
                        <p className="meta">Última actualización: 3 de marzo de 2026</p>

                        <h2>¿Quién es el responsable de tus datos?</h2>
                        <p><strong>Kaudal</strong> (kaudal.com.mx), con domicilio de contacto electrónico en <a href="mailto:contacto@kaudal.com.mx">contacto@kaudal.com.mx</a>, es responsable del tratamiento de tus datos personales conforme a la Ley Federal de Protección de Datos Personales en Posesión de los Particulares (LFPDPPP).</p>

                        <h2>¿Qué datos recopilamos?</h2>
                        <ul>
                            <li><strong>Correo electrónico</strong> — para crear y acceder a tu cuenta.</li>
                            <li><strong>Portafolios guardados</strong> — las combinaciones de activos y pesos que almacenas.</li>
                            <li><strong>Preferencias de uso</strong> — idioma, tickers favoritos y configuraciones de la herramienta.</li>
                        </ul>
                        <p>No recopilamos datos financieros sensibles como números de cuenta bancaria, CLABE ni contraseñas de casas de bolsa.</p>

                        <h2>¿Para qué usamos tus datos?</h2>
                        <ul>
                            <li>Brindarte acceso a tu cuenta y guardar tus portafolios.</li>
                            <li>Mejorar la experiencia de la plataforma.</li>
                            <li>Enviarte comunicaciones relacionadas con el servicio (solo si aceptas).</li>
                        </ul>
                        <p>No vendemos, alquilamos ni compartimos tus datos con terceros para fines publicitarios.</p>

                        <h2>Derechos ARCO</h2>
                        <p>Tienes derecho a <strong>Acceder, Rectificar, Cancelar u Oponerte</strong> al tratamiento de tus datos personales. Para ejercer cualquiera de estos derechos, envía un correo a <a href="mailto:contacto@kaudal.com.mx">contacto@kaudal.com.mx</a> con el asunto "Derechos ARCO" e incluye:</p>
                        <ul>
                            <li>Tu nombre completo y correo registrado.</li>
                            <li>El derecho que deseas ejercer y la razón.</li>
                            <li>Una identificación oficial vigente (copia digital).</li>
                        </ul>
                        <p>Responderemos en un plazo máximo de <strong>20 días hábiles</strong>, conforme a la LFPDPPP.</p>

                        <h2>Transferencia de datos</h2>
                        <p>Tus datos pueden ser almacenados en servidores de <strong>Supabase</strong> (infraestructura en la nube) y <strong>Vercel</strong> (hospedaje web), ambos ubicados fuera de México. Esta transferencia se realiza para la operación del servicio y bajo sus propias políticas de seguridad.</p>

                        <h2>Cambios a este aviso</h2>
                        <p>Nos reservamos el derecho de actualizar este aviso. Cualquier cambio será publicado en esta página con la fecha de actualización.</p>
                    </article>
                </div>
            </main>
            <Footer />
        </>
    );
}
