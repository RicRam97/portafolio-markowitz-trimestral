import Navbar from '@/components/layout/Navbar';
import Footer from '@/components/layout/Footer';
import type { Metadata } from 'next';

export const metadata: Metadata = {
    title: 'Política de Cookies — Kaudal',
    description: 'Política de cookies de Kaudal. Conoce qué cookies usamos y cómo gestionarlas.',
};

export default function CookiesPage() {
    return (
        <>
            <Navbar />
            <main className="page-wrapper" id="cookies-view" style={{ paddingTop: '100px', minHeight: 'calc(100vh - 60px)' }}>
                <div className="about-page">
                    <article className="glass-panel legal-content text-left" style={{ maxWidth: '800px', margin: '0 auto', padding: '40px' }}>
                        <h1>Política de Cookies</h1>
                        <p className="meta">Última actualización: 3 de marzo de 2026</p>

                        <h2>¿Qué son las cookies?</h2>
                        <p>Las cookies son pequeños archivos de texto que se guardan en tu navegador cuando visitas un sitio web. Nos ayudan a que la plataforma funcione correctamente y a entender cómo la usas.</p>

                        <h2>Cookies que utilizamos</h2>
                        <table style={{ width: '100%', borderCollapse: 'collapse', margin: '1.5rem 0' }}>
                            <thead>
                                <tr>
                                    <th style={{ border: '1px solid var(--border-light)', padding: '0.75rem 1rem', background: 'rgba(59,130,246,0.1)', color: 'var(--text-main)', fontWeight: 600, textAlign: 'left' }}>Cookie</th>
                                    <th style={{ border: '1px solid var(--border-light)', padding: '0.75rem 1rem', background: 'rgba(59,130,246,0.1)', color: 'var(--text-main)', fontWeight: 600, textAlign: 'left' }}>Tipo</th>
                                    <th style={{ border: '1px solid var(--border-light)', padding: '0.75rem 1rem', background: 'rgba(59,130,246,0.1)', color: 'var(--text-main)', fontWeight: 600, textAlign: 'left' }}>Propósito</th>
                                    <th style={{ border: '1px solid var(--border-light)', padding: '0.75rem 1rem', background: 'rgba(59,130,246,0.1)', color: 'var(--text-main)', fontWeight: 600, textAlign: 'left' }}>Duración</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td style={{ border: '1px solid var(--border-light)', padding: '0.75rem 1rem', color: 'var(--text-muted)' }}>sb-*-auth-token</td>
                                    <td style={{ border: '1px solid var(--border-light)', padding: '0.75rem 1rem', color: 'var(--text-muted)' }}>Esencial</td>
                                    <td style={{ border: '1px solid var(--border-light)', padding: '0.75rem 1rem', color: 'var(--text-muted)' }}>Mantener tu sesión iniciada (Supabase Auth)</td>
                                    <td style={{ border: '1px solid var(--border-light)', padding: '0.75rem 1rem', color: 'var(--text-muted)' }}>Sesión / 7 días</td>
                                </tr>
                                <tr>
                                    <td style={{ border: '1px solid var(--border-light)', padding: '0.75rem 1rem', color: 'var(--text-muted)' }}>sb-*-auth-token-code-verifier</td>
                                    <td style={{ border: '1px solid var(--border-light)', padding: '0.75rem 1rem', color: 'var(--text-muted)' }}>Esencial</td>
                                    <td style={{ border: '1px solid var(--border-light)', padding: '0.75rem 1rem', color: 'var(--text-muted)' }}>Verificación de seguridad PKCE</td>
                                    <td style={{ border: '1px solid var(--border-light)', padding: '0.75rem 1rem', color: 'var(--text-muted)' }}>Sesión</td>
                                </tr>
                                <tr>
                                    <td style={{ border: '1px solid var(--border-light)', padding: '0.75rem 1rem', color: 'var(--text-muted)' }}>_ga, _gid</td>
                                    <td style={{ border: '1px solid var(--border-light)', padding: '0.75rem 1rem', color: 'var(--text-muted)' }}>Analítica</td>
                                    <td style={{ border: '1px solid var(--border-light)', padding: '0.75rem 1rem', color: 'var(--text-muted)' }}>Google Analytics — métricas de uso anónimas</td>
                                    <td style={{ border: '1px solid var(--border-light)', padding: '0.75rem 1rem', color: 'var(--text-muted)' }}>Hasta 2 años</td>
                                </tr>
                            </tbody>
                        </table>

                        <h2>Cookies esenciales</h2>
                        <p>Las cookies de <strong>Supabase Auth</strong> son necesarias para que puedas iniciar sesión y usar tu cuenta. Sin ellas, la autenticación no funciona. Estas cookies <strong>no se pueden desactivar</strong> mientras uses el servicio.</p>

                        <h2>Cookies analíticas (opcionales)</h2>
                        <p>Si implementamos Google Analytics, utilizaremos cookies analíticas para entender cómo se usa la plataforma (páginas visitadas, tiempo de uso). Estos datos son <strong>anónimos</strong> y nos ayudan a mejorar el servicio.</p>

                        <h2>Cómo gestionar tus cookies</h2>
                        <ul>
                            <li><strong>Banner de cookies:</strong> Al visitar Kaudal por primera vez, verás un banner donde puedes aceptar o rechazar las cookies analíticas.</li>
                            <li><strong>Desde tu navegador:</strong> Puedes eliminar o bloquear cookies en la configuración de tu navegador en cualquier momento.</li>
                            <li><strong>Opt-out de Analytics:</strong> Puedes instalar el <a href="https://tools.google.com/dlpage/gaoptout" target="_blank" rel="noopener">complemento de inhabilitación de Google Analytics</a>.</li>
                        </ul>

                        <h2>Más información</h2>
                        <p>Si tienes dudas sobre nuestra política de cookies, escríbenos a <a href="mailto:contacto@kaudal.com.mx">contacto@kaudal.com.mx</a>.</p>
                    </article>
                </div>
            </main>
            <Footer />
        </>
    );
}
