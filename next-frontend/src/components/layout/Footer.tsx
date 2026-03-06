import Link from 'next/link';

export default function Footer() {
    return (
        <footer className="landing-footer">
            <div className="footer-cta">
                <h2 style={{ fontFamily: 'var(--font-display)', fontWeight: 700 }}>¿Listo para explorar?</h2>
                <p>Aprende cómo funciona la diversificación y la optimización de portafolios de forma interactiva.</p>
                <Link href="/dashboard" className="btn btn-cta glow-effect footer-cta-btn" style={{ textDecoration: 'none', margin: '0 auto' }}>
                    Empezar a Explorar
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M5 12h14M12 5l7 7-7 7" />
                    </svg>
                </Link>
            </div>

            {/* Brand Badges */}
            <div className="footer-badges">
                <span className="brand-badge">🔒 Conexión Segura</span>
                <span className="brand-badge">📊 Herramienta Educativa</span>
            </div>

            {/* Footer Links Grid */}
            <div className="footer-links-row">
                <div className="footer-col">
                    <h4 style={{ fontFamily: 'var(--font-display)', marginBottom: '16px' }}>Navegación</h4>
                    <Link href="/acerca" className="footer-link">Sobre Kaudal</Link>
                    <Link href="/faq" className="footer-link">Soporte</Link>
                    <Link href="/dashboard" className="footer-link">Mi sesión</Link>
                </div>
                <div className="footer-col">
                    <h4 style={{ fontFamily: 'var(--font-display)', marginBottom: '16px' }}>Legal</h4>
                    <Link href="/privacidad" className="footer-link">Privacidad</Link>
                    <Link href="/terminos" className="footer-link">Términos</Link>
                    <Link href="/cookies" className="footer-link">Cookies</Link>
                    <Link href="/disclaimer" className="footer-link">Disclaimer</Link>
                </div>
                <div className="footer-col">
                    <h4 style={{ fontFamily: 'var(--font-display)', marginBottom: '16px' }}>Contacto</h4>
                    <a href="mailto:contacto@kaudal.com.mx" className="footer-link footer-email" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
                            <polyline points="22,6 12,13 2,6" />
                        </svg>
                        contacto@kaudal.com.mx
                    </a>
                    <div className="footer-social" style={{ display: 'flex', gap: '12px', marginTop: '16px' }}>
                        <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer" className="social-icon" aria-label="LinkedIn">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.32 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.79M6.88 8.56a1.68 1.68 0 0 0 1.68-1.68c0-.93-.75-1.69-1.68-1.69a1.69 1.69 0 0 0-1.69 1.69c0 .93.76 1.68 1.69 1.68m1.39 9.94v-8.37H5.5v8.37h2.77z" />
                            </svg>
                        </a>
                        <a href="https://x.com" target="_blank" rel="noopener noreferrer" className="social-icon" aria-label="X / Twitter">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                            </svg>
                        </a>
                        <a href="https://facebook.com" target="_blank" rel="noopener noreferrer" className="social-icon" aria-label="Facebook">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M12 2.04C6.5 2.04 2 6.53 2 12.06C2 17.06 5.66 21.21 10.44 21.96V14.96H7.9V12.06H10.44V9.85C10.44 7.34 11.93 5.96 14.22 5.96C15.31 5.96 16.45 6.15 16.45 6.15V8.62H15.19C13.95 8.62 13.56 9.39 13.56 10.18V12.06H16.34L15.89 14.96H13.56V21.96A10 10 0 0 0 22 12.06C22 6.53 17.5 2.04 12 2.04Z" />
                            </svg>
                        </a>
                        <a href="https://instagram.com" target="_blank" rel="noopener noreferrer" className="social-icon" aria-label="Instagram">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                                <path fillRule="evenodd" clipRule="evenodd" d="M12 7C9.23858 7 7 9.23858 7 12C7 14.7614 9.23858 17 12 17C14.7614 17 17 14.7614 17 12C17 9.23858 14.7614 7 12 7ZM9 12C9 13.6569 10.3431 15 12 15C13.6569 15 15 13.6569 15 12C15 10.3431 13.6569 9 12 9C10.3431 9 9 10.3431 9 12Z" />
                                <path d="M16.5 9C17.3284 9 18 8.32843 18 7.5C18 6.67157 17.3284 6 16.5 6C15.6716 6 15 6.67157 15 7.5C15 8.32843 15.6716 9 16.5 9Z" />
                                <path fillRule="evenodd" clipRule="evenodd" d="M5 5C3.34315 5 2 6.34315 2 8V16C2 17.6569 3.34315 19 5 19H19C20.6569 19 22 17.6569 22 16V8C22 6.34315 20.6569 5 19 5H5ZM4 8C4 7.44772 4.44772 7 5 7H19C19.5523 7 20 7.44772 20 8V16C20 16.5523 19.5523 17 19 17H5C4.44772 17 4 16.5523 4 16V8Z" />
                            </svg>
                        </a>
                    </div>
                </div>
            </div>

            <div className="footer-bottom">
                <span>© 2026 Kaudal · Hecho en México 🇲🇽</span>
                <span className="footer-sep">·</span>
                <span>Herramienta de simulación con fines educativos.</span>
            </div>
        </footer>
    );
}
