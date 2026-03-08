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
        { label: 'Demo Interactiva', href: '/demo' },
        { label: 'Soporte', href: '/faq' },
        { label: 'Sobre Kaudal', href: '/acerca' },
    ];

    return (
        <>
            <nav className="landing-nav" id="landing-nav" aria-label="Menú principal">
                <div className="nav-inner">
                    <Link href="/" className="nav-brand" id="nav-brand" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Image src="/kaudal-logo2.png" alt="Kaudal" width={120} height={34} className="nav-logo-img" priority />
                        <span style={{ fontSize: '10px', fontWeight: 700, letterSpacing: '0.05em', padding: '2px 6px', borderRadius: '4px', background: 'linear-gradient(135deg, #3b82f6, #10b981)', color: '#fff', lineHeight: '1.2' }}>BETA</span>
                    </Link>

                    <div className={`nav-links${mobileOpen ? ' open' : ''}`}>
                        {links.map((l) => (
                            <Link
                                key={l.href}
                                href={l.href}
                                className={`nav-link ${pathname === l.href ? 'active' : ''}`}
                                {...(pathname === l.href ? { 'aria-current': 'page' as const } : {})}
                                onClick={() => setMobileOpen(false)}
                            >
                                {l.label}
                            </Link>
                        ))}
                        <Link href="/login" className="nav-link nav-auth-btn" onClick={() => setMobileOpen(false)}>
                            Iniciar sesión
                        </Link>
                    </div>

                    <button
                        className="nav-hamburger"
                        onClick={() => setMobileOpen(!mobileOpen)}
                        aria-label="Menú"
                        aria-expanded={mobileOpen}
                    >
                        <span />
                        <span />
                        <span />
                    </button>
                </div>
            </nav>
        </>
    );
}
