'use client';

import { useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navbar() {
    const [mobileOpen, setMobileOpen] = useState(false);
    const pathname = usePathname();

    const links = [
        { label: 'Inicio', href: '/' },
        { label: 'Conoce Kaudal', href: '/conoce' },
        { label: 'Planes', href: '/planes' },
        { label: 'Prueba gratuita', href: '/demo' },
        { label: 'Soporte', href: '/faq' },
        { label: 'Sobre Kaudal', href: '/acerca' },
    ];

    return (
        <>
            <nav className="landing-nav" id="landing-nav">
                <div className="nav-inner">
                    <Link href="/" className="nav-brand" id="nav-brand" style={{ textDecoration: 'none' }}>
                        <Image src="/kaudal-logo2.png" alt="Kaudal" width={120} height={34} className="nav-logo-img" priority />
                    </Link>

                    <div className="nav-links">
                        {links.map((l) => (
                            <Link
                                key={l.href}
                                href={l.href}
                                className={`nav-link ${pathname === l.href ? 'active' : ''}`}
                            >
                                {l.label}
                            </Link>
                        ))}
                        <Link href="/login" className="nav-link nav-auth-btn">
                            Iniciar sesión
                        </Link>
                    </div>

                    <button
                        className="nav-hamburger"
                        onClick={() => setMobileOpen(!mobileOpen)}
                        aria-label="Abrir menú"
                        style={{ display: 'none' /* Will handle mobile via css overriding or specific state */ }}
                    >
                        <span />
                        <span />
                        <span />
                    </button>
                </div>
            </nav>

            {/* For mobile, we would usually toggle a class. Because we ported the CSS directly, we can just inject an inline style if needed, but original CSS used a media query and JS to show/hide `.nav-links`. To respect the prompt, I'll add a mobile menu override exactly as it worked in the original if needed, or just let CSS media queries handle it if that's what `style.css` did. Let's make sure `.nav-links.mobile-open` is applied like the original JS did. */}
            {mobileOpen && (
                <style dangerouslySetInnerHTML={{
                    __html: `
          @media (max-width: 768px) {
            .nav-links {
              display: flex !important;
              flex-direction: column;
              position: absolute;
              top: 60px;
              left: 0;
              right: 0;
              background: rgba(11,17,32,0.98);
              border-top: 1px solid var(--border-light);
              padding: 16px;
            }
          }
        `}} />
            )}
        </>
    );
}
